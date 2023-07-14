from dataclasses import dataclass
import torch
from torch import nn
from torchvision import transforms
from typing import Any
from einops import rearrange, repeat


@dataclass
class GeneralModelConfig:
    num_classes: int
    hidden_size: int = 768
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    attention_heads: int = 12
    device: str = "cpu"


@dataclass
class SpatialModelConfig(GeneralModelConfig):
    use_preprocessed: bool = False
    use_featvec_joint: bool = False
    use_featvec_bb: bool = False
    use_featvec_img: bool = False
    num_joints: int = 17
    num_layers: int = 4
    dim_feature_vector: int = -1


@dataclass
class TemporalModelConfig(GeneralModelConfig):
    token_length: int = 15
    num_layers: int = 4
    use_temporal_special_token: bool = True


class JointEmbeddings(nn.Module):
    def __init__(self, config: SpatialModelConfig):
        super().__init__()
        self.config = config
        self.joint_id_embeddings = nn.Embedding(
            num_embeddings=config.num_joints + 2,  # +2 for pad and cls tokens
            embedding_dim=config.hidden_size,
            padding_idx=0,
        )
        self.joint_relpos_embeddings = nn.Linear(2, config.hidden_size)
        self.joint_cls_embedding = nn.parameter.Parameter(
            torch.randn(config.hidden_size)
        )
        self.feature_vector_joint_embeddings = nn.Linear(
            config.dim_feature_vector * 9, config.hidden_size
        )
        if config.use_featvec_bb:
            self.feature_vector_bb_embeddings = nn.Linear(
                config.dim_feature_vector * 16 ** 2, config.hidden_size
            )
        if config.use_featvec_img:
            self.feature_vector_img_embeddings = nn.Linear(
                config.dim_feature_vector * 16 ** 2, config.hidden_size
            )
        if not self.config.use_preprocessed:
            from prep.resnet_preprocess import process_single_batch

            self.process_single_batch = process_single_batch

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.device = config.device

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        joint_id_embeddings = self.joint_id_embeddings(
            batch["joint_id"].to(self.device)
        )
        joint_relpos_embeddings = self.joint_relpos_embeddings(
            batch["rel_joint_position"]
        )
        embeddings = joint_id_embeddings + joint_relpos_embeddings

        bs = batch["rel_joint_position"].shape[0]

        if self.config.use_preprocessed:
            if self.config.use_featvec_joint:
                featvec_joint = batch["featvec_joint"].flatten(1, 2)

            if self.config.use_featvec_bb:
                featvec_bb = batch["featvec_bb"].flatten(1, 3)

            if self.config.use_featvec_img:
                featvec_img = batch["featvec_img"].flatten(1, 3)

        else:
            detection = {
                "keypoints": rearrange(
                    batch["keypoints"], "bs joint xy -> joint xy bs"
                ),
                "bbox": rearrange(batch["bbox"], "bs xywh -> xywh bs"),
            }
            featvec_joint, featvec_bb, featvec_img = self.process_single_batch(
                batch["images"], detection
            )

            featvec_joint = featvec_joint.flatten(
                1, 2
            )  # (bs, in_size, n_joints). in_size = 9 * config.dim_feature_vector
            featvec_bb = rearrange(
                featvec_bb, "bs c sixt1 sixt2 -> bs (c sixt1 sixt2)"
            )  # (bs, channels * 16**2)
            featvec_img = rearrange(
                featvec_img, "bs c sixt1 sixt2 -> bs (c sixt1 sixt2)"
            )  # (bs, channels * 16**2)

            featvec_joint = featvec_joint.to(self.device)
            if self.config.use_featvec_bb:
                featvec_bb = featvec_bb.to(self.device)
            if self.config.use_featvec_img:
                featvec_img = featvec_img.to(self.device)

        featvec_joint_embd = self.feature_vector_joint_embeddings(
            rearrange(featvec_joint, "bs in_size joints -> bs joints in_size")
        )  # (bs, n_joints, in_size)

        if self.config.use_featvec_bb:
            featvec_bb_embd = self.feature_vector_bb_embeddings(
                featvec_bb
            )  # (bs, config.hidden_size)
        else:
            featvec_bb_embd = torch.zeros((bs, self.config.hidden_size)).to(
                self.device
            )  # (bs, config.hidden_size)

        if self.config.use_featvec_img:
            featvec_img_embd = self.feature_vector_img_embeddings(
                featvec_img
            )  # (bs, config.hidden_size)
        else:
            featvec_img_embd = torch.zeros((bs, self.config.hidden_size)).to(
                self.device
            )  # (bs, config.hidden_size)

        cls_token = (
            (featvec_bb_embd + featvec_img_embd)[:, None]
            if self.config.use_featvec_bb or self.config.use_featvec_img
            else repeat(self.joint_cls_embedding, f"hidden -> {bs} 1 hidden")
        )
        # ゼロテンソルとjoint_cls_embeddingを加える；pad, clsトークンに対してこれらが選ばれるようにする
        # bbまたはimgがあるならjoint_cls_embeddingの代わりにこれを使う
        try:
            featvec_embd = torch.concat(
                (
                    torch.zeros((bs, 1, self.config.hidden_size), device=self.device),
                    cls_token,
                    featvec_joint_embd,
                ),
                dim=1,
            )
        except:
            print(f"{cls_token.device=}")
            print(f"{featvec_joint_embd.device=}")
            exit()
        # featvec_embd: (bs, num_joints+2, dim_hidden)
        # featvec_embdからjoint_id番目のものを取り出す (thank you yenyo)
        embeddings += torch.gather(
            featvec_embd,
            1,
            batch["joint_id"][..., None].expand_as(embeddings),
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(
        self, seq_len, embed_dim, temperature=10000, scale=None, learned=False
    ):
        super().__init__()
        if learned:
            pe = nn.parameter.Parameter(torch.randn(1, seq_len, embed_dim))
        else:
            if scale is None:
                scale = 1
            pos, dim = torch.meshgrid(
                torch.arange(seq_len), torch.arange(embed_dim), indexing="ij"
            )
            pe = (
                scale
                * pos
                / temperature
                ** (2 * torch.div(dim, 2, rounding_mode="floor") / embed_dim)
            )
            pe[..., 0::2] = torch.sin(pe[..., 0::2])
            pe[..., 1::2] = torch.cos(pe[..., 1::2])
            pe = torch.unsqueeze(pe, 0)

        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: [batch_size, seq_len, embed_dim]
        """
        return x + self.positional_encoding[:, : x.shape[1]]


class SpatialTransformer(nn.Module):
    def __init__(self, config: SpatialModelConfig):
        super().__init__()
        self.joint_embeddings = JointEmbeddings(config)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            activation="gelu",
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.num_layers,
            norm=self.layer_norm,
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        joint_embeddings = self.joint_embeddings(batch)  # (bsz, num_joints, embed_dim)
        src_key_padding_mask = batch["src_key_padding_mask"]
        joint_embeddings = joint_embeddings.contiguous()
        src_key_padding_mask = src_key_padding_mask.contiguous()
        spatial_embeddings = self.transformer(
            src=joint_embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )
        return spatial_embeddings


class TemporalTransformer(nn.Module):
    def __init__(self, config: TemporalModelConfig) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            activation="gelu",
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pos_embd = PositionalEncoding(
            config.token_length + (1 if config.use_temporal_special_token else 0),
            config.hidden_size,
            learned=False,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.num_layers,
            norm=self.layer_norm,
        )

    def forward(self, spatial_embeddings: torch.Tensor):
        # (token_length, batch_size, hidden_size) -> (batch_size, token_length, hidden_size)
        spatial_embeddings = spatial_embeddings.transpose(1, 0)
        spatial_embeddings = self.pos_embd(spatial_embeddings)
        token_len = spatial_embeddings.shape[1]
        temporal_embeddings = self.temporal_transformer(
            spatial_embeddings,
            mask=torch.tril(torch.ones((token_len, token_len))).to(
                spatial_embeddings.device
            ),
        )
        return temporal_embeddings


class PointingNetwork(nn.Module):
    def __init__(
        self,
        spatial_config: SpatialModelConfig,
        temporal_config: TemporalModelConfig,
        use_temporal_encoder=True,
    ):
        super().__init__()

        self.spatial_config = spatial_config

        # Extract some of the settings from config
        self.use_temporal_encoder = use_temporal_encoder
        self.use_temporal_special_token = temporal_config.use_temporal_special_token
        self.token_length = temporal_config.token_length

        self.transform = transforms.Compose(
            [
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        self.spatial_transformer = SpatialTransformer(spatial_config)
        if use_temporal_encoder:
            if self.use_temporal_special_token:
                self.temporal_special_token = nn.parameter.Parameter(
                    torch.randn(spatial_config.hidden_size)
                )
            self.temporal_transformer = TemporalTransformer(temporal_config)
        else:
            hidden = temporal_config.hidden_size
            n_token = temporal_config.token_length
            self.temporal_fuser = nn.Sequential(
                nn.Linear(hidden * n_token, hidden * 5),
                nn.Linear(hidden * 5, hidden * 5),
                nn.Linear(hidden * 5, hidden),
            )

        self.classifier = nn.Linear(
            temporal_config.hidden_size, temporal_config.num_classes
        )
        self.direction_estimator = nn.Linear(temporal_config.hidden_size, 3)

    def forward(self, batch: dict[str, Any]):
        # batch['featvec'].shape=[batchsize, token_length, feat_dim, n_joints]
        batch_size = batch["joint_id"].shape[0]

        keys_for_spatial_embd = [
            "joint_id",
            "rel_joint_position",
            "src_key_padding_mask",
        ]

        if self.spatial_config.use_preprocessed:
            if self.spatial_config.use_featvec_joint:
                keys_for_spatial_embd.append("featvec_joint")
            if self.spatial_config.use_featvec_bb:
                keys_for_spatial_embd.append("featvec_bb")
            if self.spatial_config.use_featvec_img:
                keys_for_spatial_embd.append("featvec_img")
        else:
            keys_for_spatial_embd.append("images")
            keys_for_spatial_embd.append("bbox")
            keys_for_spatial_embd.append("keypoints")

        spatial_embeddings = [
            self.spatial_transformer(
                {k: batch[k][:, i] for k in batch.keys() if k in keys_for_spatial_embd}
            )
            for i in range(self.token_length)
        ]
        spatial_embeddings = [
            se[:, -1] for se in spatial_embeddings
        ]  # extract last token from a list of [batch_size, num_joints, hidden_size] whose length is token_length
        if self.use_temporal_encoder:
            if self.use_temporal_special_token:
                spatial_embeddings.append(
                    repeat(
                        self.temporal_special_token, f"hidden -> {batch_size} hidden"
                    )
                )
            spatial_embeddings = torch.stack(spatial_embeddings)
            temporal_embeddings = self.temporal_transformer(spatial_embeddings)
            final_embeddings = temporal_embeddings[
                :, -1
            ]  # extract last token from [batch_size, token_length, hidden_size]
        else:
            final_embeddings = self.temporal_fuser(
                rearrange(
                    torch.stack(spatial_embeddings),
                    "ntoken batch hidden -> batch (ntoken hidden)",
                )
            )

        classification = self.classifier(final_embeddings)
        direction = self.direction_estimator(final_embeddings)
        return {"action": classification, "direction": direction}


def build_pointing_network(cfg, device):
    ACTION_CLS_DICT = {0: [-1], 1: list(range(0, 100))}
    NUM_CLASSES = len(ACTION_CLS_DICT)
    NUM_JOINTS = 17  # predefined by COCO-keypoints
    DIM_FEATURE_VECTOR = 256  # output dimension of ResNet-34

    # model configuration
    spatial_config = SpatialModelConfig(
        num_joints=NUM_JOINTS,
        num_classes=NUM_CLASSES,
        dim_feature_vector=DIM_FEATURE_VECTOR,
        num_layers=cfg.model.num_layers,
        hidden_size=cfg.model.hidden_size,
        device=device,
        use_featvec_joint=cfg.model.featvec_joint,
        use_featvec_bb=cfg.model.featvec_bb,
        use_featvec_img=cfg.model.featvec_img,
        use_preprocessed=cfg.use_preprocessed,
    )
    temporal_config = TemporalModelConfig(
        num_classes=NUM_CLASSES,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
        num_layers=cfg.model.num_layers,
        use_temporal_special_token=True,
        token_length=cfg.model.tlength,
    )
    return PointingNetwork(
        spatial_config,
        temporal_config,
        use_temporal_encoder=not cfg.model.omit_temporal_encoder,
    )
