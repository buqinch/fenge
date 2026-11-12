"""Dual CNN FPN with matched pixel and region-query decoder variants."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import DualCNNEncoder, DINOv3ConvNeXtTiny, MobileNetV3LargeFeatures, GradientSafeGELU


@dataclass
class ModelConfig:
    decoder: str = "region"
    backbone: str = "dual"
    input_size: int = 1024
    num_classes: int = 5
    num_queries: int = 32
    decoder_layers: int = 4
    width: int = 128
    branch_width: int = 64
    attention_heads: int = 8
    dropout: float = 0.0
    weight_dir: str = str(Path(__file__).resolve().parents[2] / "model")
    convnext_checkpoint: str | None = None

    def __post_init__(self):
        if self.decoder not in ("pixel", "region") or self.backbone not in ("dual", "convnext"):
            raise ValueError("decoder must be pixel/region and backbone must be dual/convnext")
        if self.input_size < 32 or self.input_size % 32:
            raise ValueError("input_size must be a positive multiple of 32")
        if self.num_classes != 5:
            raise ValueError("LuSNAR uses exactly five ordered classes")
        if self.width % self.attention_heads or self.width % 4:
            raise ValueError("width must be divisible by attention_heads and four")
        if self.num_queries < self.num_classes or self.decoder_layers < 1:
            raise ValueError("Need at least five queries and one decoder layer")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)


def conv_norm_act(in_channels: int, channels: int, kernel_size: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.GroupNorm(math.gcd(32, channels), channels), GradientSafeGELU(),
    )


class DualFusionFPN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width, branch = config.width, config.branch_width
        self.dual = config.backbone == "dual"
        self.convnext_projections = nn.ModuleList([conv_norm_act(c, branch) for c in DINOv3ConvNeXtTiny.channels])
        if self.dual:
            self.mobile_projections = nn.ModuleList([conv_norm_act(c, branch) for c in MobileNetV3LargeFeatures.channels])
            self.detail_projections = nn.ModuleList([conv_norm_act(branch, width) for _ in range(2)])
        self.fusions = nn.ModuleList([conv_norm_act(branch * (2 if self.dual else 1), width) for _ in range(4)])
        self.smoothing = nn.ModuleList([conv_norm_act(width, width, 3) for _ in range(4)])
        self.mask_features = conv_norm_act(width, width, 3)

    def forward(self, convnext: list[Tensor], mobile: list[Tensor] | None) -> tuple[Tensor, list[Tensor]]:
        lateral, details = [], []
        for index, feature in enumerate(convnext):
            c = self.convnext_projections[index](feature)
            if self.dual:
                m = self.mobile_projections[index](mobile[index])
                if m.shape[-2:] != c.shape[-2:]:
                    raise RuntimeError("CNN stage feature strides disagree")
                lateral.append(self.fusions[index](torch.cat((c, m), dim=1)))
                if index < 2:
                    details.append(self.detail_projections[index](m))
            else:
                lateral.append(self.fusions[index](c))
        pyramid: list[Tensor] = [lateral[-1]]
        for index in range(2, -1, -1):
            pyramid.insert(0, lateral[index] + F.interpolate(pyramid[0], size=lateral[index].shape[-2:], mode="bilinear", align_corners=False))
        pyramid = [layer(x) for layer, x in zip(self.smoothing, pyramid)]
        # Explicit mobile /4 and /8 decoder bypass, beyond same-scale fusion.
        if self.dual:
            pyramid[0] = pyramid[0] + details[0] + F.interpolate(details[1], size=pyramid[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.mask_features(pyramid[0]), pyramid


def sine_position(feature: Tensor) -> Tensor:
    """Deterministic 2D positions, shape (1, H*W, C)."""
    _, channels, height, width = feature.shape
    quarter = channels // 4
    omega = torch.arange(quarter, device=feature.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y = torch.linspace(0, 2 * math.pi, height, device=feature.device)[:, None] * omega[None]
    x = torch.linspace(0, 2 * math.pi, width, device=feature.device)[:, None] * omega[None]
    y = torch.cat((y.sin(), y.cos()), dim=-1)[:, None, :].expand(height, width, -1)
    x = torch.cat((x.sin(), x.cos()), dim=-1)[None, :, :].expand(height, width, -1)
    return torch.cat((y, x), dim=-1).reshape(1, height * width, channels).to(feature.dtype)


class RegionDecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.self_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), GradientSafeGELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, query_positions: Tensor, memory: Tensor, memory_positions: Tensor) -> Tensor:
        q = self.cross_norm(queries)
        value = self.cross_attention(q + query_positions, memory + memory_positions, memory, need_weights=False)[0]
        queries = queries + self.dropout(value)
        q = self.self_norm(queries)
        value = self.self_attention(q + query_positions, q + query_positions, q, need_weights=False)[0]
        queries = queries + self.dropout(value)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class RegionDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.width
        self.query_content = nn.Embedding(config.num_queries, width)
        self.query_positions = nn.Embedding(config.num_queries, width)
        self.level_embeddings = nn.Embedding(3, width)
        self.layers = nn.ModuleList([RegionDecoderLayer(width, config.attention_heads, config.dropout) for _ in range(config.decoder_layers)])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, config.num_classes + 1)
        self.mask_embedding = nn.Sequential(nn.Linear(width, width), GradientSafeGELU(), nn.Linear(width, width))

    def predict(self, queries: Tensor, mask_features: Tensor) -> dict[str, Tensor]:
        queries = self.norm(queries)
        return {"pred_logits": self.classifier(queries), "pred_masks": torch.einsum("bqc,bchw->bqhw", self.mask_embedding(queries), mask_features)}

    def forward(self, mask_features: Tensor, pyramid: list[Tensor]) -> dict:
        batch = mask_features.shape[0]
        queries = self.query_content.weight.unsqueeze(0).expand(batch, -1, -1)
        query_positions = self.query_positions.weight.unsqueeze(0)
        memories, positions = [], []
        # Query-to-pixel cross attention uses /8,/16,/32; /4 is retained for
        # masks and detail, avoiding quadratic attention over 1024^2 pixels.
        for index, feature in enumerate(pyramid[1:]):
            memories.append(feature.flatten(2).transpose(1, 2))
            positions.append(sine_position(feature) + self.level_embeddings.weight[index][None, None])
        predictions = []
        for index, layer in enumerate(self.layers):
            level = index % 3
            queries = layer(queries, query_positions, memories[level], positions[level])
            predictions.append(self.predict(queries, mask_features))
        output = predictions[-1]
        output["aux_outputs"] = predictions[:-1]
        return output


class LunarSegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.encoder = DualCNNEncoder(config.weight_dir, pretrained, config.backbone, config.convnext_checkpoint)
        self.fusion = DualFusionFPN(config)
        self.decoder = RegionDecoder(config) if config.decoder == "region" else nn.Sequential(conv_norm_act(config.width, config.width, 3), nn.Conv2d(config.width, config.num_classes, 1))

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.fusion.parameters()
        yield from self.decoder.parameters()

    def freeze_backbones(self, frozen: bool = True) -> None:
        self.encoder.freeze(frozen)
        self.encoder.train(self.training)

    @property
    def pretrained_report(self) -> dict:
        return {"convnext": self.encoder.convnext.pretrained_report, "mobile": self.encoder.mobile.pretrained_report if self.encoder.mobile else None}

    def forward(self, images: Tensor) -> dict:
        expected = (self.config.input_size, self.config.input_size)
        if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[-2:]) != expected:
            raise ValueError(f"Expected Bx3x{expected[0]}x{expected[1]} normalized RGB; got {tuple(images.shape)}")
        return self._forward_at_size(images, expected)

    def forward_inference(self, images: Tensor) -> dict:
        """Evaluate variable spatial sizes without changing their image scale.

        Both CNNs receive the same tensor, padded only on the right/bottom
        with replicated edge values to a multiple of 32. Continuous semantic
        predictions are upsampled to that padded size, then cropped back to
        the input size. Query masks and auxiliary masks retain their native
        low-resolution coordinates on the padded grid.
        """
        if self.training:
            raise RuntimeError("forward_inference requires model.eval(); training uses fixed-size forward")
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError(f"Expected Bx3xHxW normalized RGB with H,W >= 32; got {tuple(images.shape)}")
        height, width = images.shape[-2:]
        pad_bottom, pad_right = (-height) % 32, (-width) % 32
        if pad_bottom or pad_right:
            images = F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate")
        output = self._forward_at_size(images, tuple(images.shape[-2:]))
        if pad_bottom or pad_right:
            output["semantic_logits"] = output["semantic_logits"][..., :height, :width]
        return output

    def _forward_at_size(self, images: Tensor, expected: tuple[int, int]) -> dict:
        mask_features, pyramid = self.fusion(*self.encoder(images))
        if self.config.decoder == "pixel":
            scores = F.interpolate(self.decoder(mask_features), size=expected, mode="bilinear", align_corners=False)
            return {"semantic_logits": scores}
        output = self.decoder(mask_features, pyramid)
        classes = output["pred_logits"].float().softmax(dim=-1)[..., :self.config.num_classes]
        masks = output["pred_masks"].float().sigmoid()
        scores = torch.einsum("bqc,bqhw->bchw", classes, masks)
        # Resize continuous five-class evidence before normalization / argmax.
        scores = F.interpolate(scores, size=expected, mode="bilinear", align_corners=False)
        scores = scores.clamp_min(1e-8)
        output["semantic_logits"] = (scores / scores.sum(dim=1, keepdim=True)).log()
        return output


def build_model(config: ModelConfig | dict, pretrained: bool = True) -> LunarSegmentationModel:
    return LunarSegmentationModel(ModelConfig.from_dict(config) if isinstance(config, dict) else config, pretrained)
