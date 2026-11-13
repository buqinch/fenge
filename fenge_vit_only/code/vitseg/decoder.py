"""Mask2Former: deformable pixel encoding and iterative masked query decoding.

This implementation follows the architecture in the official Mask2Former
pixel/transformer decoders without Detectron2 or custom compiled dependencies.
``decoder_layers`` counts actual blocks: six blocks produce seven predictions,
including the initial learned-query prediction.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deformable import MultiScaleDeformableAttention


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _initialize_linear(layer: nn.Linear | nn.Conv2d) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _projection(channels: int, width: int, kernel_size: int = 1,
                activation: bool = False) -> nn.Sequential:
    conv = nn.Conv2d(channels, width, kernel_size, padding=kernel_size // 2, bias=False)
    _initialize_linear(conv)
    # The production width256 uses32groups. Small CPU fixtures also retain at
    # least two channels per group, even on a 1x1 coarsest feature map.
    groups = math.gcd(32, width)
    if width // groups < 2 and groups > 1:
        groups //= 2
    modules = [conv, nn.GroupNorm(groups, width)]
    if activation:
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def sine_position(height: int, width: int, channels: int, device: torch.device) -> Tensor:
    """Return [1,HW,C] sine/cosine encoding of normalized pixel centers."""
    if channels % 4:
        raise ValueError("position channels must be divisible by four")
    half = channels // 2
    frequency = 10000 ** (2 * torch.div(torch.arange(half, device=device), 2,
                                       rounding_mode="floor").float() / half)
    y = ((torch.arange(height, device=device, dtype=torch.float32) + .5) / height * (2 * math.pi))[:, None]
    x = ((torch.arange(width, device=device, dtype=torch.float32) + .5) / width * (2 * math.pi))[:, None]
    yy, xx = y / frequency, x / frequency
    yy = torch.stack((yy[:, 0::2].sin(), yy[:, 1::2].cos()), dim=-1).flatten(1)
    xx = torch.stack((xx[:, 0::2].sin(), xx[:, 1::2].cos()), dim=-1).flatten(1)
    return torch.cat((yy[:, None].expand(height, width, half),
                      xx[None].expand(height, width, half)), dim=-1).reshape(1, height * width, channels)


class DeformablePixelEncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention = MultiScaleDeformableAttention(embed_dim=width, num_heads=heads,
                                                       num_levels=3, num_points=4)
        self.attention_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))
        self.ffn_norm = nn.LayerNorm(width)
        for layer in (self.ffn[0], self.ffn[2]):
            _initialize_linear(layer)

    def forward(self, source: Tensor, positions: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor) -> Tensor:
        update = self.attention(query=source + positions, value=source,
                                spatial_shapes=spatial_shapes, reference_points=reference_points,
                                level_start_index=level_start_index)
        source = self.attention_norm(source + update)
        return self.ffn_norm(source + self.ffn(source))


class MultiScalePixelDecoder(nn.Module):
    """Six three-level deformable encoder layers, followed by the C4 FPN path."""

    def __init__(self, in_channels: Sequence[int], width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.width = width
        # Internally order memories from coarse to fine, as in Mask2Former.
        self.input_projections = nn.ModuleList(_projection(c, width) for c in self.in_channels[:0:-1])
        self.level_embeddings = nn.Embedding(3, width)
        self.layers = nn.ModuleList(DeformablePixelEncoderLayer(width, heads) for _ in range(layers))
        self.detail_lateral = _projection(self.in_channels[0], width)
        self.detail_output = _projection(width, width, 3, activation=True)
        self.mask_projection = nn.Conv2d(width, width, 1)
        _initialize_linear(self.mask_projection)

    def forward(self, features: Sequence[Tensor], gradient_checkpointing: bool = True) -> tuple[Tensor, list[Tensor]]:
        sources, positions, shapes, centers = [], [], [], []
        batch = len(features[0])
        for index, (projection, feature) in enumerate(zip(self.input_projections, features[:0:-1])):
            projected = projection(feature)
            height, width = projected.shape[-2:]
            shapes.append((height, width))
            sources.append(projected.flatten(2).transpose(1, 2))
            position = sine_position(height, width, self.width, feature.device).to(projected.dtype)
            positions.append(position + self.level_embeddings.weight[index][None, None])
            y, x = torch.meshgrid(torch.arange(height, device=feature.device, dtype=torch.float32) + .5,
                                  torch.arange(width, device=feature.device, dtype=torch.float32) + .5,
                                  indexing="ij")
            centers.append(torch.stack((x.reshape(-1) / width, y.reshape(-1) / height), dim=-1))
        source = torch.cat(sources, dim=1)
        position = torch.cat(positions, dim=1)
        spatial_shapes = torch.tensor(shapes, device=source.device, dtype=torch.long)
        counts = spatial_shapes.prod(1)
        starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        references = torch.cat(centers)[None, :, None].expand(batch, -1, 3, -1)
        for layer in self.layers:
            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                source = checkpoint(layer, source, position, references, spatial_shapes, starts,
                                    use_reentrant=False)
            else:
                source = layer(source, position, references, spatial_shapes, starts)
        memories, offset = [], 0
        for height, width in shapes:
            size = height * width
            memories.append(source[:, offset:offset + size].transpose(1, 2).reshape(batch, self.width, height, width))
            offset += size
        detail = self.detail_lateral(features[0])
        fused = self.detail_output(detail + F.interpolate(memories[-1], size=detail.shape[-2:],
                                                         mode="bilinear", align_corners=False))
        return self.mask_projection(fused), memories


class MaskedQueryDecoderLayer(nn.Module):
    """Masked cross-attention, self-attention, FFN; dropout is deliberately0."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.cross_norm, self.self_norm, self.ffn_norm = (nn.LayerNorm(width) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(width, 8 * width), nn.ReLU(), nn.Linear(8 * width, width))
        for attention in (self.cross_attention, self.self_attention):
            nn.init.xavier_uniform_(attention.in_proj_weight)
            nn.init.zeros_(attention.in_proj_bias)
            _initialize_linear(attention.out_proj)
        for layer in (self.ffn[0], self.ffn[2]):
            _initialize_linear(layer)

    def forward(self, queries: Tensor, memory: Tensor, query_position: Tensor,
                memory_position: Tensor, blocked: Tensor) -> Tensor:
        # torch2.1 canonicalizes a bool attention mask to the *input query*
        # dtype before autocast projects Q/K/V. If inputs are FP32 but their
        # projections are BF16, SDPA then rejects the resulting FP32 mask.
        # Cast the attention inputs to the active projection dtype first, so
        # the canonical mask and projected queries agree. Residuals stay in
        # the original dtype; the detached spatial mask remains boolean here.
        attention_dtype = queries.dtype
        if queries.device.type == "cuda" and torch.is_autocast_enabled():
            attention_dtype = torch.get_autocast_gpu_dtype()
        elif queries.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            attention_dtype = torch.get_autocast_cpu_dtype()
        update = self.cross_attention((queries + query_position).to(attention_dtype),
                                      (memory + memory_position).to(attention_dtype),
                                      memory.to(attention_dtype), attn_mask=blocked,
                                      need_weights=False)[0]
        queries = self.cross_norm(queries + update)
        positioned = (queries + query_position).to(attention_dtype)
        update = self.self_attention(positioned, positioned, queries.to(attention_dtype), need_weights=False)[0]
        queries = self.self_norm(queries + update)
        return self.ffn_norm(queries + self.ffn(queries))


class Mask2FormerHead(nn.Module):
    def __init__(self, in_channels: Sequence[int] = (96, 384, 384, 384), width: int = 256,
                 num_queries: int = 32, num_classes: int = 5, pixel_decoder_layers: int = 6,
                 decoder_layers: int = 6, heads: int = 8, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("in_channels must describe four ordered feature levels")
        self.in_channels = tuple(_positive_integer("in_channels", c) for c in in_channels)
        self.width, self.heads = _positive_integer("width", width), _positive_integer("heads", heads)
        self.num_queries, self.num_classes = _positive_integer("num_queries", num_queries), _positive_integer("num_classes", num_classes)
        if self.width % self.heads or self.width % 4 or self.num_queries < self.num_classes:
            raise ValueError("width must divide into heads/four; queries must cover all classes")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.pixel_decoder = MultiScalePixelDecoder(self.in_channels, self.width,
                                                    _positive_integer("pixel_decoder_layers", pixel_decoder_layers), self.heads)
        self.layers = nn.ModuleList(MaskedQueryDecoderLayer(self.width, self.heads)
                                    for _ in range(_positive_integer("decoder_layers", decoder_layers)))
        self.query_content = nn.Embedding(self.num_queries, self.width)
        self.query_positions = nn.Embedding(self.num_queries, self.width)
        self.memory_level_embeddings = nn.Embedding(3, self.width)
        self.prediction_norm = nn.LayerNorm(self.width)
        self.classifier = nn.Linear(self.width, self.num_classes + 1)
        self.mask_embedding = nn.Sequential(nn.Linear(self.width, self.width), nn.ReLU(),
                                           nn.Linear(self.width, self.width), nn.ReLU(),
                                           nn.Linear(self.width, self.width))
        _initialize_linear(self.classifier)
        for layer in self.mask_embedding:
            if isinstance(layer, nn.Linear):
                _initialize_linear(layer)
        self.last_attention_stats: list[dict] = []

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def attention_mask(masks: Tensor, spatial_size: tuple[int, int], heads: int) -> tuple[Tensor, int]:
        """True blocks a key. Empty predicted regions fall back to all keys."""
        with torch.no_grad():
            blocked = F.interpolate(masks.detach().float(), size=spatial_size,
                                     mode="bilinear", align_corners=False).flatten(2) < 0
            all_blocked = blocked.all(-1)
            blocked = blocked & ~all_blocked[..., None]
            blocked = blocked[:, None].expand(-1, heads, -1, -1).reshape(-1, masks.shape[1], blocked.shape[-1])
        return blocked, int(all_blocked.sum())

    def predict(self, queries: Tensor, mask_features: Tensor) -> dict[str, Tensor]:
        normalized = self.prediction_norm(queries)
        return {"pred_logits": self.classifier(normalized),
                "pred_masks": torch.einsum("bqc,bchw->bqhw", self.mask_embedding(normalized), mask_features)}

    def forward(self, features: Sequence[Tensor]) -> dict:
        if len(features) != 4 or any(f.ndim != 4 for f in features):
            raise ValueError("Mask2Former expects four BCHW features")
        batch = len(features[0])
        if batch < 1 or any(len(f) != batch or f.shape[1] != channels for f, channels in zip(features, self.in_channels)):
            raise ValueError("Feature batch/channel dimensions disagree with head configuration")
        for first, second in zip(features, features[1:]):
            if tuple(first.shape[-2:]) != tuple(2 * n for n in second.shape[-2:]):
                raise ValueError("Features must be ordered at dyadic strides1/4,1/8,1/16,1/32")
        mask_features, memories = self.pixel_decoder(features, self.gradient_checkpointing)
        source, position = [], []
        for index, memory in enumerate(memories):
            source.append(memory.flatten(2).transpose(1, 2) + self.memory_level_embeddings.weight[index][None, None])
            position.append(sine_position(*memory.shape[-2:], self.width, memory.device).to(memory.dtype))
        query_position = self.query_positions.weight[None].expand(batch, -1, -1)
        queries = self.query_content.weight[None].expand(batch, -1, -1)
        predictions = [self.predict(queries, mask_features)]
        self.last_attention_stats = []
        for index, layer in enumerate(self.layers):
            level = index % 3
            blocked, fallback = self.attention_mask(predictions[-1]["pred_masks"], memories[level].shape[-2:], self.heads)
            self.last_attention_stats.append({"block": index + 1, "memory_stride": (32, 16, 8)[level],
                                              "memory_hw": list(memories[level].shape[-2:]),
                                              "all_masked_query_rows_fallback": fallback,
                                              "mask_detached": not blocked.requires_grad})
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                queries = checkpoint(layer, queries, source[level], query_position, position[level], blocked,
                                     use_reentrant=False)
            else:
                queries = layer(queries, source[level], query_position, position[level], blocked)
            predictions.append(self.predict(queries, mask_features))
        return {**predictions[-1], "aux_outputs": predictions[:-1]}
