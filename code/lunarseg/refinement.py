"""Optional gated semantic-logit residuals with exact zero-head initialization.

Neither module replaces the existing FPN/region decoder. Only five correction
channels and one gate channel are upsampled to the final image grid.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import GradientSafeGELU


REFINEMENT_INIT_SEED = 20261112


def _conv(in_channels: int, out_channels: int, kernel: int = 1, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride,
                  padding=kernel // 2, bias=False),
        nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        GradientSafeGELU(),
    )


def _zero_output(layer: nn.Conv2d) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def directional_context(features: Tensor) -> dict[str, Tensor]:
    """Exact box means on a replicated /4 grid, using float32 integral images.

    Local offsets are [-4,4]^2 (9x9); the surrounding ring is the 33x33
    square [-16,16]^2 excluding that local square. Directional rectangles
    are symmetric 12x9/9x12 strips between the local and outer boundaries:
    up [-16,-5]x[-4,4], down [5,16]x[-4,4], and transposed left/right.
    Every rectangle has its own pixel-count denominator. This avoids the
    quadratic kernel cost of dense 33x33 average-pool windows.
    """
    if features.ndim != 4 or not features.is_floating_point():
        raise ValueError("Context features must be floating BxCxHxW")
    height, width = features.shape[-2:]
    padded = F.pad(features.float(), (16, 16, 16, 16), mode="replicate")
    integral = F.pad(padded.cumsum(-2).cumsum(-1), (1, 0, 1, 0))

    def mean(y0, y1, x0, x1):
        total = (integral[..., y1:y1 + height, x1:x1 + width]
                 - integral[..., y0:y0 + height, x1:x1 + width]
                 - integral[..., y1:y1 + height, x0:x0 + width]
                 + integral[..., y0:y0 + height, x0:x0 + width])
        return total / ((y1 - y0) * (x1 - x0))

    local = mean(12, 21, 12, 21)
    outer = mean(0, 33, 0, 33)
    return {"local": local, "surround": (outer * 1089 - local * 81) / 1008,
            "up": mean(0, 12, 12, 21), "down": mean(21, 33, 12, 21),
            "left": mean(12, 21, 0, 12), "right": mean(12, 21, 21, 33)}


class _GatedResidual(nn.Module):
    def residual_fields(self, mask_features: Tensor, images: Tensor,
                        probabilities: Tensor | None = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, mask_features: Tensor, images: Tensor, semantic_logits: Tensor,
                probabilities: Tensor | None = None) -> Tensor:
        if (semantic_logits.ndim != 4 or semantic_logits.shape[1] != 5
                or semantic_logits.shape[0] != mask_features.shape[0]
                or images.shape[0] != semantic_logits.shape[0]
                or images.shape[-2:] != semantic_logits.shape[-2:]):
            raise ValueError("Refinement needs matching image/features and five full-size semantic logits")
        delta, gate_logits = self.residual_fields(mask_features, images, probabilities)
        size = semantic_logits.shape[-2:]
        # Float32 interpolation/gating avoids BF16 arithmetic altering the
        # identity path. The source logits are never softmaxed or normalized.
        delta = F.interpolate(delta.float(), size=size, mode="bilinear", align_corners=False)
        gate_logits = F.interpolate(gate_logits.float(), size=size, mode="bilinear", align_corners=False)
        correction = (gate_logits.sigmoid() * delta).to(semantic_logits.dtype)
        return semantic_logits + correction


class ContextRefinement(_GatedResidual):
    """A 32-channel context branch; output remains an additive logit residual."""

    def __init__(self, feature_channels: int = 128):
        super().__init__()
        self.projection = _conv(feature_channels, 32)
        self.mixer = nn.Sequential(_conv(32 * 7, 32), _conv(32, 32, 3))
        self.delta = nn.Conv2d(32, 5, 1)
        self.gate = nn.Conv2d(32, 1, 1)
        _zero_output(self.delta)
        _zero_output(self.gate)

    def residual_fields(self, mask_features, images, probabilities=None):
        feature = self.projection(mask_features)
        context = directional_context(feature)
        local = context["local"]
        mixed = torch.cat((feature.float(), local - feature.float(),
                           context["surround"] - local, context["up"] - local,
                           context["down"] - local, context["left"] - local,
                           context["right"] - local), dim=1).to(feature.dtype)
        hidden = self.mixer(mixed)
        return self.delta(hidden), self.gate(hidden)


class DetailRefinement(_GatedResidual):
    """Raw-RGB shallow detail, FPN semantics and learned probability gating.

    The only full-resolution hidden tensor has eight channels. Fusion,
    correction and the gate operate at half resolution; semantic projection
    starts on the original /4 FPN grid. No entropy threshold restricts the gate.
    """

    def __init__(self, feature_channels: int = 128):
        super().__init__()
        self.rgb_full = _conv(3, 8, 3)
        self.rgb_half = _conv(8, 32, 3, stride=2)
        self.semantic_projection = _conv(feature_channels, 32)
        self.fusion = nn.Sequential(_conv(64, 32), _conv(32, 32, 3))
        self.delta = nn.Conv2d(32, 5, 1)
        self.gate = nn.Sequential(_conv(37, 16), nn.Conv2d(16, 1, 1))
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)
        _zero_output(self.delta)
        _zero_output(self.gate[-1])

    def residual_fields(self, mask_features, images, probabilities=None):
        if (probabilities is None or probabilities.ndim != 4
                or probabilities.shape[1] != 5 or probabilities.shape[0] != images.shape[0]
                or probabilities.shape[-2:] != images.shape[-2:]):
            raise ValueError("Detail refinement requires existing normalized five-class probabilities")
        rgb = (images.float() * self.rgb_std + self.rgb_mean).clamp(0, 1)
        detail = self.rgb_half(self.rgb_full(rgb))
        semantics = F.interpolate(self.semantic_projection(mask_features), size=detail.shape[-2:],
                                  mode="bilinear", align_corners=False)
        hidden = self.fusion(torch.cat((detail, semantics), dim=1))
        probability = F.interpolate(probabilities.float(), size=detail.shape[-2:],
                                    mode="bilinear", align_corners=False).to(hidden.dtype)
        return self.delta(hidden), self.gate(torch.cat((hidden, probability), dim=1))
