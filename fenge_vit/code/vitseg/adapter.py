"""Four-stage patch-only bidirectional spatial interaction."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import GradientSafeGELU
from .deformable import MultiScaleDeformableAttention, reference_grid


def flatten_features(features):
    shapes = torch.tensor([x.shape[-2:] for x in features], dtype=torch.long, device=features[0].device)
    return torch.cat([x.flatten(2).transpose(1, 2) for x in features], 1), shapes


def unflatten_features(tokens, features):
    result, start = [], 0
    for feature in features:
        batch, channels, height, width = feature.shape
        count = height * width
        result.append(tokens[:, start:start + count].transpose(1, 2).reshape(batch, channels, height, width))
        start += count
    return tuple(result)


class CNNSpatialProjection(nn.Module):
    def __init__(self, width=384):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, width, 1), nn.GroupNorm(32, width))
                                          for channels in (192, 384, 768)])

    def forward(self, features):
        return tuple(projection(feature) for projection, feature in zip(self.projections, features))


class InteractionStage(nn.Module):
    def __init__(self, width=384, heads=6, points=4, init_scale=1e-4):
        super().__init__()
        self.patch_norm = nn.LayerNorm(width)
        self.spatial_norm = nn.LayerNorm(width)
        self.extract_query_norm = nn.LayerNorm(width)
        self.extract_value_norm = nn.LayerNorm(width)
        self.inject_attention = MultiScaleDeformableAttention(width, heads, 3, points)
        self.extract_attention = MultiScaleDeformableAttention(width, heads, 1, points)
        self.gamma = nn.Parameter(torch.full((width,), init_scale))
        self.eta = nn.Parameter(torch.full((width,), init_scale))
        self.level_embedding = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.level_embedding, std=.02)

    def inject(self, patches: Tensor, *features: Tensor) -> Tensor:
        shapes = torch.tensor([feature.shape[-2:] for feature in features], dtype=torch.long, device=patches.device)
        value_parts = [feature.flatten(2).transpose(1, 2) + self.level_embedding[index]
                       for index, feature in enumerate(features)]
        values = self.spatial_norm(torch.cat(value_parts, 1))
        height, width = features[1].shape[-2:]
        if patches.shape[1] != height * width:
            raise ValueError("ViT patch tokens must retain the native 1/16 spatial grid")
        references = reference_grid(height, width, len(patches), 3, patches.device)
        update = self.inject_attention(self.patch_norm(patches), values, shapes, references)
        return patches + update * self.gamma

    def extract(self, patches: Tensor, *features: Tensor) -> tuple[Tensor, ...]:
        spatial, _ = flatten_features(features)
        height, width = features[1].shape[-2:]
        shape = torch.tensor([[height, width]], dtype=torch.long, device=patches.device)
        references = torch.cat([reference_grid(*feature.shape[-2:], len(patches), 1, patches.device)
                                for feature in features], 1)
        update = self.extract_attention(self.extract_query_norm(spatial), self.extract_value_norm(patches),
                                        shape, references)
        return unflatten_features(spatial + update * self.eta, features)


class BidirectionalAdapter(nn.Module):
    def __init__(self, width=384, heads=6, points=4):
        super().__init__()
        self.stages = nn.ModuleList([InteractionStage(width, heads, points) for _ in range(4)])


class SimpleMultiscaleFusion(nn.Module):
    """Variant A: a final ViT map fused separately into each CNN scale."""
    def __init__(self, width=384):
        super().__init__()
        self.fusions = nn.ModuleList([nn.Sequential(nn.Conv2d(width * 2, width, 1),
                                                   nn.GroupNorm(32, width), GradientSafeGELU()) for _ in range(3)])

    def forward(self, patch_feature: Tensor, *spatial: Tensor):
        return tuple(fusion(torch.cat((feature, F.interpolate(patch_feature, feature.shape[-2:],
                                      mode="bilinear", align_corners=False)), 1))
                     for fusion, feature in zip(self.fusions, spatial))
