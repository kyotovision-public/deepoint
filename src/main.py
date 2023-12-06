import torch
import dataset
from pathlib import Path
import utils
import random
import hydra
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pl_module import PointingModule
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
import logging


def build_module_and_trainer(cfg, DEVICE):
    network = build_pointing_network(cfg, DEVICE)

    module = PointingModule(network, verbose=cfg.verbose)

    if cfg.hardware.gpus > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    callbacks = pl.callbacks.ModelCheckpoint(
        monitor="validation/direction_loss",
        save_last=True,
        filename="{epoch}-{step}"
        + f"-bb_{cfg.model.featvec_bb}-img_{cfg.model.featvec_img}",
        save_top_k=-1,
    )

    save_dir = f"lightning_logs/split_{cfg.split_method}{'-bb' if cfg.model.featvec_bb else ''}{'-img' if cfg.model.featvec_img else ''}{'-tl='+str(cfg.model.tlength) if cfg.model.tlength!=15 else ''}{'-MLPasTE' if cfg.model.omit_temporal_encoder else ''}{cfg.filter_joint if cfg.filter_joint is not None else ''}"
    print(f"lightning_logs are saved to {save_dir}")
    trainer = pl.Trainer(
        devices=cfg.hardware.gpus,
        strategy=strategy,
        callbacks=[callbacks],
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
    )

    return module, trainer


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu" and cfg.task == "train":
        logging.warning("Training DeePoint with CPU takes a long time.")
    if cfg.task == "test" and cfg.verbose is not True and cfg.shrink_rate != 1:
        logging.warning(
            "Using only part of test dataset. You should set `shrink_rate=1` except for speeding up the performance for visualization"
        )

    assert not (
        cfg.task == "test" and cfg.ckpt is None
    ), "checkpoint should be specified for evaluation"

    module, trainer = build_module_and_trainer(cfg, DEVICE)

    keypoints_path = Path(__file__).parent.parent / cfg.data.keypoints_root
    ds = dataset.DPDataset(keypoints_path, cfg)

    match cfg.split_method:
        case "time":
            train_idx, val_idx, test_idx = utils.subset_time(ds, cfg.train_ratio)
        case "person":
            train_idx, val_idx, test_idx = utils.subset_person(ds)
        case "venue-living":
            train_idx, val_idx, test_idx = utils.subset_venue(ds, "living")
        case "venue-office":
            train_idx, val_idx, test_idx = utils.subset_venue(ds, "office")
        case _:
            raise NotImplementedError

    train_ds = Subset(ds, train_idx)
    train_weights = utils.subset_weight(train_ds)
    train_sampler = WeightedRandomSampler(
        train_weights, int(len(train_ds) * cfg.shrink_rate)
    )

    random.seed(42)
    val_idx = random.sample(val_idx, int(len(val_idx) * cfg.shrink_rate))
    val_ds = Subset(ds, val_idx)

    if cfg.shrink_rate != 1:
        # SPLIT_INTOに分けて、それぞれのpackからshrink_rate割だけ取り出す
        SPLIT_INTO = 30 * 15
        pack_len = len(test_idx) // SPLIT_INTO
        print(f"{pack_len=}")
        test_idx = sum(
            (
                test_idx[i * pack_len : i * pack_len + int(pack_len * cfg.shrink_rate)]
                for i in range(SPLIT_INTO)
            ),
            start=[],
        )
    test_ds = Subset(ds, test_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.hardware.bs,
        sampler=train_sampler,
        num_workers=cfg.hardware.nworkers,
        persistent_workers=True if cfg.hardware.nworkers != 0 else False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
        persistent_workers=True if cfg.hardware.nworkers != 0 else False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
        persistent_workers=True if cfg.hardware.nworkers != 0 else False,
    )

    print(f"Starting {cfg.task}...")
    if cfg.task == "train":
        trainer.fit(module, train_dl, val_dl, ckpt_path=cfg.ckpt)
    elif cfg.task == "test":
        trainer.test(module, test_dl, ckpt_path=cfg.ckpt)
    else:
        raise NotImplementedError

    return


if __name__ == "__main__":
    main()
