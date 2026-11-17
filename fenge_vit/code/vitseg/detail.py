"""PointRend-inspired dense point classification, not its adaptive subdivision.

Every output pixel is refined, including confidently wrong interiors. Strips
bound intermediate activation memory; they do not crop the source features.
The RGB branch is a lightweight stride-2 detail path, not a third backbone.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .upernet import SpatialGroupNorm


class DenseDetailCorrection(nn.Module):
    def __init__(self, cnn_channels=96, num_classes=5, chunk_rows=64,
                 gradient_checkpointing=True):
        super().__init__()
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError('chunk_rows must be positive')
        self.chunk_rows = chunk_rows
        self.gradient_checkpointing = gradient_checkpointing
        self.rgb_half = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            SpatialGroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            SpatialGroupNorm(8, 32), nn.ReLU())
        self.cnn_project = nn.Sequential(nn.Conv2d(cnn_channels, 32, 1, bias=False),
                                         SpatialGroupNorm(8, 32), nn.ReLU())
        # A shared per-point MLP. No strip-dependent normalization or padding.
        self.mlp = nn.Sequential(nn.Conv2d(64 + num_classes, 64, 1), nn.ReLU(),
                                 nn.Conv2d(64, 32, 1), nn.ReLU())
        self.delta = nn.Conv2d(32, num_classes, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    @staticmethod
    def sampling_grid(height, width, start, end, device):
        x = (torch.arange(width, device=device, dtype=torch.float32) + .5) * (2. / width) - 1
        y = (torch.arange(start, end, device=device, dtype=torch.float32) + .5) * (2. / height) - 1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack((xx, yy), dim=-1).unsqueeze(0)

    def _strip(self, fine, half, coarse, grid):
        # Match the pixel-center / border coordinates of align_corners=False.
        with torch.autocast(device_type=fine.device.type, enabled=False):
            grid = grid.expand(fine.shape[0], -1, -1, -1)
            f = F.grid_sample(fine.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
            h = F.grid_sample(half.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
            p = coarse.float().softmax(1)
        hidden = self.mlp(torch.cat((f, h, p), dim=1))
        return self.delta(hidden).float()

    def forward(self, images, cnn_fine, logits):
        if logits.shape[0] != images.shape[0] or logits.shape[-2:] != images.shape[-2:]:
            raise ValueError('Detail correction requires logits on the original image grid')
        active = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        half = checkpoint(self.rgb_half, images, use_reentrant=False) if active else self.rgb_half(images)
        fine = self.cnn_project(cnn_fine)
        height, width = logits.shape[-2:]
        strips = []
        for start in range(0, height, self.chunk_rows):
            end = min(start + self.chunk_rows, height)
            grid = self.sampling_grid(height, width, start, end, logits.device)
            coarse = logits[:, :, start:end]
            delta = (checkpoint(self._strip, fine, half, coarse, grid, use_reentrant=False)
                     if active else self._strip(fine, half, coarse, grid))
            strips.append(delta)
        return logits.float() + torch.cat(strips, dim=2)
