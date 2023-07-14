from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as TF
from einops import rearrange, reduce, repeat
import sys
from pathlib import Path
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# feature extractor for preprocessing the image
net = torchvision.models.resnet34(weights="DEFAULT")
net.eval()
fe = create_feature_extractor(
    net,
    {
        "layer3": "feat3",
    },
).to(device)

fe = nn.DataParallel(fe)


def process_single_batch(images: torch.Tensor, pifpaf_det: dict):
    batch_size, _, img_h, img_w = images.shape
    with torch.no_grad():
        feat_map = (
            fe(images.to(device))["feat3"].to("cpu").float()
        )  # (batch_size, channels, H/16, W/16)
        feat_h, feat_w = feat_map.shape[2], feat_map.shape[3]
        strides = (
            img_w / feat_w,
            img_h / feat_h,
        )  # (x,y)方向のストライド

        # キーポイント周り3x3での特徴を取り出す
        if type(pifpaf_det["keypoints"]) is not torch.Tensor:
            keypoints = torch.stack(pifpaf_det["keypoints"]).reshape(
                (17, 3, -1)
            )  # (17, 3, batch_size)
        else:
            keypoints = pifpaf_det["keypoints"]

        abs_joints = rearrange(keypoints[:, :2], "joint xy bs -> bs joint xy")
        abs_joints_3x3 = rearrange(
            torch.stack(
                [
                    abs_joints + torch.tensor(((i * strides[0], j * strides[1])))
                    for j in [-1, 0, 1]
                    for i in [-1, 0, 1]
                ]
            ),
            "nine bs joint xy -> bs nine joint xy",
        )
        norm_joints = abs_joints_3x3 / torch.tensor([img_w, img_h])
        grid = (norm_joints * 2 - 1).float()
        kp_feat_lists = TF.grid_sample(
            feat_map, grid, align_corners=True
        )  # (batch_size, channels, 9, 17)

        # 人のBBのRoIプーリング
        if type(pifpaf_det["bbox"]) is not torch.Tensor:
            bbox = torch.stack(pifpaf_det["bbox"])  # (4, batch_size)
        else:
            bbox = pifpaf_det["bbox"]  # (4, batch_size)
        N = 16  # NxNでRoIする
        grids = []
        for bb in bbox.T:
            x = bb[0] + bb[2] * torch.arange(N) / N
            y = bb[1] + bb[3] * torch.arange(N) / N
            x_n = x / feat_w
            y_n = y / feat_h
            grid_x, grid_y = torch.meshgrid(x_n * 2 - 1, y_n * 2 - 1, indexing="xy")
            grid = rearrange(torch.stack((grid_x, grid_y)), "two n1  n2 -> n1 n2 two")
            grids.append(grid)
        grids = torch.stack(grids)  # (batch_size, N, N, 2)
        bb_feat_lists = TF.grid_sample(
            feat_map, grids.float(), align_corners=True
        )  # (batch_size, channels, N, N)

        # 画像全体のRoIプーリング
        M = 16
        x_n = torch.arange(M) / M
        y_n = torch.arange(M) / M
        grid_x, grid_y = torch.meshgrid(x_n * 2 - 1, y_n * 2 - 1, indexing="xy")
        grid = rearrange(torch.stack((grid_x, grid_y)), "two m1 m2 -> m1 m2 two")
        grid = repeat(grid, f"m1 m2 xy -> {batch_size} m1 m2 xy")
        img_feat_lists = TF.grid_sample(
            feat_map, grid.float(), align_corners=True
        )  # (batch_size, channels, M, M)

    return kp_feat_lists, bb_feat_lists, img_feat_lists
