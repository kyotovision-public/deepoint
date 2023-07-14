import torch
from utils import *
import pytorch_lightning as pl
import torch.nn.functional as TF
import torchmetrics
from einops import rearrange, reduce
import numpy as np
from pathlib import Path
import cv2
import pickle

SHOW_IMAGE = False


class PointingModule(pl.LightningModule):
    def __init__(self, model, verbose=False):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.clasf_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        self.prev_hand_idx = None
        self.prev_arrow_coord = np.array((0.0, 0.0))

    def training_step(self, batch, _):
        bs = batch["idx"].shape[0]
        pred = self.model(batch)
        target: dict[str, torch.Tensor] = batch["answer"]
        classification_loss = TF.cross_entropy(pred["action"], target["action"])

        # At non_nan_idx, All 3 coordinates are not nan
        non_nan_idx = target["direction"].isnan().logical_not().all(dim=1)
        eps = 1e-6
        direction_loss = torch.arccos(
            torch.clamp(
                TF.cosine_similarity(
                    pred["direction"][non_nan_idx], target["direction"][non_nan_idx]
                ),
                -1 + eps,
                1 - eps,
            )
        ).mean()

        if direction_loss.isnan():
            direction_loss = torch.tensor((0))
        del batch
        self.log(
            "train/classification_loss",
            classification_loss,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train/direction_loss",
            direction_loss * 180 / np.pi,
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )
        self.log(
            "train/accuracy",
            self.clasf_acc(pred["action"], target["action"]),
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )
        return direction_loss + classification_loss

    def validation_step(self, batch, batch_idx):
        bs = batch["idx"].shape[0]
        pred = self.model(batch)
        target = batch["answer"]
        classification_loss = TF.cross_entropy(pred["action"], target["action"])
        eps = 1e-6
        # non_nan_idx = indices that *all* directions are not nan
        non_nan_idx = target["direction"].isnan().logical_not().all(dim=1)
        direction_loss = torch.arccos(
            torch.clamp(
                TF.cosine_similarity(
                    pred["direction"][non_nan_idx], target["direction"][non_nan_idx]
                ),
                -1 + eps,
                1 - eps,
            )
        ).mean()

        if direction_loss.isnan():
            direction_loss = torch.tensor((0))
        self.log_dict(
            {
                "validation/classification_loss": classification_loss,
                "validation/direction_loss": direction_loss * 180 / np.pi,
                "validation/accuracy": self.clasf_acc(pred["action"], target["action"]),
            },
            batch_size=bs,
            sync_dist=True,
        )

        return

    def test_step(self, batch, batch_idx):
        """same as validation_step, name of logging are the only difference"""
        bs = batch["idx"].shape[0]
        pred = self.model(batch)
        target = batch["answer"]
        classification_loss = TF.cross_entropy(pred["action"], target["action"])
        eps = 1e-6
        # non_nan_idx = indices that *all* directions are not nan
        non_nan_idx = target["direction"].isnan().logical_not().all(dim=1)
        direction_loss = torch.arccos(
            torch.clamp(
                TF.cosine_similarity(
                    pred["direction"][non_nan_idx], target["direction"][non_nan_idx]
                ),
                -1 + eps,
                1 - eps,
            )
        ).mean()

        if direction_loss.isnan():
            direction_loss = torch.tensor((0))
        self.log_dict(
            {
                "test/classification_loss": classification_loss,
                "test/direction_loss": direction_loss * 180 / np.pi,
                "test/accuracy": self.clasf_acc(pred["action"], target["action"]),
            },
            batch_size=bs,
            sync_dist=True,
        )

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
        }
