"""Pure DINOv3 ViT-S/16, four intermediate feature scales, and Mask2Former."""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import DINOv3ViTSmall
from .decoder import Mask2FormerHead


@dataclass
class ModelConfig:
    variant: str = "vit_only"
    input_size: int = 1024
    num_classes: int = 5
    num_queries: int = 32
    decoder_layers: int = 6
    pixel_decoder_layers: int = 6
    width: int = 256
    gradient_checkpointing: bool = True
    weight_dir: str = str(Path(__file__).resolve().parents[2] / "model")
    semantic_epsilon: float = 1e-8

    def __post_init__(self):
        if self.variant != "vit_only":
            raise ValueError("This project supports only variant=vit_only")
        if type(self.input_size) is not int or self.input_size < 32 or self.input_size % 32:
            raise ValueError("input_size must be a positive multiple of 32, at least 32")
        if self.num_classes != 5 or self.width != 256:
            raise ValueError("This architecture uses five classes and a 256-channel pixel decoder")
        for name in ("num_queries", "decoder_layers", "pixel_decoder_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if isinstance(self.semantic_epsilon, bool) or not math.isfinite(self.semantic_epsilon) or self.semantic_epsilon <= 0:
            raise ValueError("semantic_epsilon must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


class PureViTSegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.encoder = DINOv3ViTSmall(config.weight_dir, config.input_size, pretrained)
        self.projections = nn.ModuleList([nn.Conv2d(self.encoder.channels, config.width, 1) for _ in range(4)])
        self.decoder = Mask2FormerHead(in_channels=(256, 256, 256, 256), width=config.width,
            num_queries=config.num_queries, num_classes=config.num_classes,
            pixel_decoder_layers=config.pixel_decoder_layers, decoder_layers=config.decoder_layers,
            heads=8, gradient_checkpointing=config.gradient_checkpointing)
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
        return {"vit": self.encoder.pretrained_report}

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

    def reassemble(self, features: tuple[Tensor, ...] | list[Tensor]) -> list[Tensor]:
        """Map normalized block 3/6/9/12 patch grids to stride 4/8/16/32."""
        if len(features) != 4:
            raise ValueError("Expected four ViT intermediate feature maps")
        grid = tuple(features[0].shape[-2:])
        if min(grid) < 2 or any(s % 2 for s in grid):
            raise ValueError("Patch grid must have positive even dimensions for stride-32 pooling")
        if any(feature.ndim != 4 or feature.shape[1] != 384 or tuple(feature.shape[-2:]) != grid
               for feature in features):
            raise ValueError("All four intermediate feature maps must share a 384-channel patch grid")
        projected = [layer(feature) for layer, feature in zip(self.projections, features)]
        return [F.interpolate(projected[0], size=(grid[0] * 4, grid[1] * 4), mode="bilinear", align_corners=False),
                F.interpolate(projected[1], size=(grid[0] * 2, grid[1] * 2), mode="bilinear", align_corners=False),
                projected[2], F.avg_pool2d(projected[3], kernel_size=2, stride=2)]

    def _forward(self, images: Tensor) -> dict:
        active = self._gradient_checkpointing and self.training and torch.is_grad_enabled()
        # Unlike interleaved adapters, the pure encoder contains no trainable
        # insertions while frozen, so its entire forward may avoid autograd.
        with torch.no_grad() if self.encoder._frozen else nullcontext():
            native = self.encoder(images, gradient_checkpointing=active)
        output = self.decoder(self.reassemble(native["intermediate_features"]))
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
    return PureViTSegmentationModel(ModelConfig.from_dict(config) if isinstance(config, dict) else config, pretrained)
