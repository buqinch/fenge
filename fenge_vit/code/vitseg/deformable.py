"""Multi-scale deformable attention with differentiable FP32 bilinear sampling.

Reference geometry and radial initialization follow Deformable DETR's public
operator. This implementation uses PyTorch grid_sample, without custom CUDA
extensions or a dense Q-by-K cross-attention substitute.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4, query_chunk_size=512):
        super().__init__()
        for name, value in (("embed_dim", embed_dim), ("num_heads", num_heads),
                            ("num_levels", num_levels), ("num_points", num_points),
                            ("query_chunk_size", query_chunk_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads, self.num_levels = embed_dim, num_heads, num_levels
        self.num_points, self.query_chunk_size = num_points, query_chunk_size
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        angle = torch.arange(self.num_heads, dtype=torch.float32) * (2 * math.pi / self.num_heads)
        radial = torch.stack((angle.cos(), angle.sin()), -1)
        radial = radial / radial.abs().amax(-1, keepdim=True)
        radial = radial[:, None, None].repeat(1, self.num_levels, self.num_points, 1)
        radial = radial * torch.arange(1, self.num_points + 1, dtype=torch.float32)[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(radial.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query: Tensor, value: Tensor, spatial_shapes: Tensor,
                reference_points: Tensor, level_start_index: Tensor | None = None,
                value_padding_mask: Tensor | None = None) -> Tensor:
        if query.ndim != 3 or value.ndim != 3 or query.shape[0] != value.shape[0]:
            raise ValueError("query/value must be BxNxD with the same batch")
        batch, queries, width = query.shape
        if width != self.embed_dim or value.shape[2] != width:
            raise ValueError("query/value channel count must match embed_dim")
        if tuple(spatial_shapes.shape) != (self.num_levels, 2):
            raise ValueError("spatial_shapes must have shape levels x 2")
        shapes = [(int(height), int(length)) for height, length in spatial_shapes.tolist()]
        if any(min(shape) < 1 for shape in shapes) or sum(h * w for h, w in shapes) != value.shape[1]:
            raise ValueError("spatial_shapes do not cover the value tokens")
        starts, offset = [], 0
        for height, length in shapes:
            starts.append(offset)
            offset += height * length
        if level_start_index is not None and level_start_index.tolist() != starts:
            raise ValueError("level_start_index differs from spatial_shapes")
        if reference_points.shape[:3] != (batch, queries, self.num_levels) or reference_points.shape[-1] not in (2, 4):
            raise ValueError("reference_points must have shape BxQxlevels x 2 (or 4)")
        projected = self.value_proj(value)
        if value_padding_mask is not None:
            if value_padding_mask.shape != value.shape[:2] or value_padding_mask.dtype != torch.bool:
                raise ValueError("value_padding_mask must be boolean BxN")
            projected = projected.masked_fill(value_padding_mask[..., None], 0)
        head_width = width // self.num_heads
        levels = [projected[:, start:start + height * length].reshape(
            batch, height, length, self.num_heads, head_width).permute(0, 3, 4, 1, 2).reshape(
                batch * self.num_heads, head_width, height, length).float()
            for start, (height, length) in zip(starts, shapes)]
        outputs = []
        for start in range(0, queries, self.query_chunk_size):
            current = query[:, start:start + self.query_chunk_size]
            count = current.shape[1]
            offsets = self.sampling_offsets(current).float().reshape(
                batch, count, self.num_heads, self.num_levels, self.num_points, 2)
            weights = self.attention_weights(current).float().reshape(
                batch, count, self.num_heads, self.num_levels * self.num_points).softmax(-1).reshape(
                    batch, count, self.num_heads, self.num_levels, self.num_points)
            reference = reference_points[:, start:start + count].float()
            total = current.new_zeros((batch, count, self.num_heads, head_width), dtype=torch.float32)
            for level, ((height, length), pixels) in enumerate(zip(shapes, levels)):
                if reference.shape[-1] == 2:
                    location = reference[:, :, None, level, None, :] + offsets[:, :, :, level] / offsets.new_tensor((length, height))
                else:
                    location = reference[:, :, None, level, None, :2] + (
                        offsets[:, :, :, level] / self.num_points * reference[:, :, None, level, None, 2:] * .5)
                grid = (location * 2 - 1).permute(0, 2, 1, 3, 4).reshape(
                    batch * self.num_heads, count, self.num_points, 2)
                # grid_sample has no native BF16 kernel; both differentiable
                # casts remain inside autograd, with FP32 coordinates/weights.
                with torch.autocast(device_type=query.device.type, enabled=False):
                    sampled = F.grid_sample(pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                    sampled = sampled.reshape(batch, self.num_heads, head_width, count, self.num_points).permute(0, 3, 1, 4, 2)
                    total = total + (sampled * weights[:, :, :, level, :, None]).sum(3)
            outputs.append(total.flatten(2))
        combined = torch.cat(outputs, 1).to(projected.dtype)
        return self.output_proj(combined).to(query.dtype)


def reference_grid(height: int, width: int, batch: int, levels: int, device) -> Tensor:
    """Normalized pixel centers; (x,y), align_corners=False convention."""
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32) + .5,
                          torch.arange(width, device=device, dtype=torch.float32) + .5, indexing="ij")
    points = torch.stack((x / width, y / height), -1).reshape(1, height * width, 1, 2)
    return points.expand(batch, -1, levels, -1)
