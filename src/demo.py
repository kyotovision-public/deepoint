import numpy as np
import cv2
import torch
import dataset
from pathlib import Path
from tqdm import tqdm
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
from draw_arrow import WIDTH, HEIGHT


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        + f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        logging.warning("Running DeePoint with CPU takes a long time.")

    assert (
        cfg.movie is not None
    ), "Please specify movie path as `movie=/path/to/movie.mp4`"

    assert (
        cfg.lr is not None
    ), "Please specify whether the pointing hand is left or right with `lr=l` or `lr=r`."

    assert cfg.ckpt is not None, "checkpoint should be specified for evaluation"

    cfg.hardware.bs = 2
    cfg.hardware.nworkers = 0
    ds = dataset.MovieDataset(cfg.movie, cfg.lr, cfg.model.tlength, DEVICE)
    dl = DataLoader(
        ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
    )

    network = build_pointing_network(cfg, DEVICE)

    # Since the model trained using pytorch lightning contains `model.` as an prefix to the keys of state_dict, we should remove them before loading
    model_dict = torch.load(cfg.ckpt)["state_dict"]
    new_model_dict = dict()
    for k, v in model_dict.items():
        new_model_dict[k[6:]] = model_dict[k]
    model_dict = new_model_dict
    network.load_state_dict(model_dict)
    network.to(DEVICE)

    Path("demo").mkdir(exist_ok=True)
    fps = 15
    out_green = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-green-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )
    out_greenred = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-greenred-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )

    prev_arrow_base = np.array((0, 0))

    for batch in tqdm(dl):
        result = network(batch)
        # bs may be smaller than cfg.hardware.bs for the last iteration
        bs = batch["abs_joint_position"].shape[0]
        for i_bs in range(bs):
            joints = batch["abs_joint_position"][i_bs][-1].to("cpu").numpy()
            image = batch["orig_image"][i_bs].to("cpu").numpy() / 255

            direction = result["direction"][i_bs]
            prob_pointing = float(
                (result["action"][i_bs, 1].exp() / result["action"][i_bs].exp().sum())
            )
            print(f"{prob_pointing=}")

            ORIG_HEIGHT, ORIG_WIDTH = image.shape[:2]
            hand_idx = 9 if batch["lr"][i_bs] == "l" else 10
            if (joints[hand_idx] < 0).any():
                arrow_base = prev_arrow_base
            else:
                arrow_base = (
                    joints[hand_idx] / np.array((ORIG_WIDTH, ORIG_HEIGHT)) * 2 - 1
                )
                prev_arrow_base = arrow_base

            image_green = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        1,
                        0,
                    ),  # Green. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )
            image_greenred = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        prob_pointing,
                        1 - prob_pointing,
                    ),  # Green to red. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )

            cv2.imshow("", image_green)
            cv2.waitKey(10)

            out_green.write((image_green * 255).astype(np.uint8))
            out_greenred.write((image_greenred * 255).astype(np.uint8))

    return


def draw_arrow_on_image(image, arrow_spec, kwargs):
    """
    Params:
    image: np.ndarray(height, width, 3), with dtype=float, value in the range of [0,1]
    arrow_spec, kwargs: options for render_frame
    Returns:
    image: np.ndarray(HEIGHT, WIDTH, 3), with dtype=float, value in the range of [0,1]
    """
    from draw_arrow import render_frame, WIDTH, HEIGHT

    ret_image = cv2.resize(image, (WIDTH, HEIGHT)).astype(float)
    img_arrow = render_frame(*arrow_spec, **kwargs).astype(float) / 255
    arrow_mask = (img_arrow.sum(axis=2) == 0.0).astype(float)[:, :, None]
    ret_image = arrow_mask * ret_image + (1 - arrow_mask) * img_arrow
    return ret_image


if __name__ == "__main__":
    main()
