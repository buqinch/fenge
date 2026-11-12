"""DINOv3 ViT-S/16 + ConvNeXt-Tiny + optional four-stage adapter + Mask2Former."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adapter import BidirectionalAdapter, CNNSpatialProjection, SimpleMultiscaleFusion
from .backbones import DualDINOBackbones
from .decoder import Mask2FormerHead


@dataclass
class ModelConfig:
    variant: str = "adapter"
    input_size: int = 1024
    num_classes: int = 5
    num_queries: int = 32
    decoder_layers: int = 6
    pixel_decoder_layers: int = 6
    width: int = 256
    adapter_dim: int = 384
    adapter_heads: int = 6
    adapter_points: int = 4
    gradient_checkpointing: bool = True
    weight_dir: str = str(Path(__file__).resolve().parents[2] / "model")
    semantic_epsilon: float = 1e-8

    def __post_init__(self):
        if self.variant not in ("fusion", "adapter"):
            raise ValueError("variant must be fusion or adapter")
        if type(self.input_size) is not int or self.input_size < 32 or self.input_size % 32:
            raise ValueError("input_size must be a positive multiple of 32, at least 32")
        if self.num_classes != 5 or self.adapter_dim != 384 or self.width != 256:
            raise ValueError("This architecture uses five classes, 384-channel adapter and 256-channel pixel decoder")
        for name in ("num_queries", "decoder_layers", "pixel_decoder_layers", "adapter_heads", "adapter_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.adapter_dim % self.adapter_heads:
            raise ValueError("adapter_dim must be divisible by adapter_heads")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if isinstance(self.semantic_epsilon, bool) or not math.isfinite(self.semantic_epsilon) or self.semantic_epsilon <= 0:
            raise ValueError("semantic_epsilon must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


class ViTAdapterSegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.encoder = DualDINOBackbones(config.weight_dir, config.input_size, pretrained)
        self.projection = CNNSpatialProjection(config.adapter_dim)
        # Common modules are initialized before variant-specific modules.
        self.decoder = Mask2FormerHead(in_channels=(96, 384, 384, 384), width=config.width,
            num_queries=config.num_queries, num_classes=config.num_classes,
            pixel_decoder_layers=config.pixel_decoder_layers, decoder_layers=config.decoder_layers,
            heads=8, gradient_checkpointing=config.gradient_checkpointing)
        self.adapter, self.fusion = None, None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261113)
            if config.variant == "adapter":
                self.adapter = BidirectionalAdapter(config.adapter_dim, config.adapter_heads, config.adapter_points)
            else:
                self.fusion = SimpleMultiscaleFusion(config.adapter_dim)
        self.set_gradient_checkpointing(config.gradient_checkpointing)

    def set_gradient_checkpointing(self, enabled: bool = True):
        if type(enabled) is not bool:
            raise ValueError("gradient checkpointing must be boolean")
        self._gradient_checkpointing = enabled
        self.decoder.gradient_checkpointing = enabled
        if hasattr(self.decoder, "set_gradient_checkpointing"):
            self.decoder.set_gradient_checkpointing(enabled)

    def freeze_backbones(self, frozen: bool = True):
        self.encoder.freeze(frozen)
        self.encoder.train(self.training)

    def backbone_parameters(self):
        yield from self.encoder.parameters()

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder."):
                yield parameter

    @property
    def pretrained_report(self):
        return self.encoder.pretrained_report

    def forward(self, images: Tensor) -> dict:
        size = (self.config.input_size, self.config.input_size)
        if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[-2:]) != size:
            raise ValueError(f"Expected Bx3x{size[0]}x{size[1]} normalized RGB, got {tuple(images.shape)}")
        return self._forward(images)

    def forward_inference(self, images: Tensor) -> dict:
        if self.training:
            raise RuntimeError("forward_inference requires eval mode")
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32 or any(s % 32 for s in images.shape[-2:]):
            raise ValueError("Inference input dimensions must be multiples of 32 and at least 32")
        return self._forward(images)

    def _forward(self, images: Tensor) -> dict:
        active = self._gradient_checkpointing and self.training and torch.is_grad_enabled()
        native = self.encoder.convnext(images, gradient_checkpointing=active)
        spatial = self.projection(native[1:])
        if self.adapter is not None:
            def inject(index, tokens, grid):
                prefix = self.encoder.vit.num_prefix_tokens
                patches = tokens[:, prefix:]
                function = self.adapter.stages[index].inject
                patches = checkpoint(function, patches, *spatial, use_reentrant=False) if active else function(patches, *spatial)
                return torch.cat((tokens[:, :prefix], patches), 1)

            def extract(index, tokens, grid):
                nonlocal spatial
                patches = tokens[:, self.encoder.vit.num_prefix_tokens:]
                function = self.adapter.stages[index].extract
                spatial = checkpoint(function, patches, *spatial, use_reentrant=False) if active else function(patches, *spatial)

            self.encoder.vit(images, before_stage=inject, after_stage=extract, gradient_checkpointing=active)
        else:
            vit = self.encoder.vit(images, gradient_checkpointing=active)
            spatial = checkpoint(self.fusion, vit["patch_features"], *spatial, use_reentrant=False) if active else self.fusion(vit["patch_features"], *spatial)
        output = self.decoder([native[0], *spatial])
        # Explicitly disable autocast: einsum otherwise downcasts even .float()
        # inputs, and these probabilities are also used for masked-view NLL.
        with torch.autocast(device_type=images.device.type, enabled=False):
            classes = output["pred_logits"].float().softmax(-1)[..., :self.config.num_classes]
            masks = output["pred_masks"].float().sigmoid()
            scores = torch.einsum("bqc,bqhw->bchw", classes, masks)
            scores = F.interpolate(scores, images.shape[-2:], mode="bilinear", align_corners=False)
            epsilon = self.config.semantic_epsilon
            probabilities = (scores + epsilon) / (scores.sum(1, keepdim=True) + self.config.num_classes * epsilon)
        output.update(semantic_scores=scores, semantic_logits=probabilities.log())
        return output


def build_model(config: ModelConfig | dict, pretrained: bool = True):
    return ViTAdapterSegmentationModel(ModelConfig.from_dict(config) if isinstance(config, dict) else config, pretrained)
