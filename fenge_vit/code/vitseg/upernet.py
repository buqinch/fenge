"""UPerNet semantic head on the same four encoder/adapter features.

Architecture reference: Xiao et al., ECCV 2018, arXiv:1807.10221, and
https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py
This implementation uses GroupNorm for small per-device batches, with a stride-16
FCN auxiliary classifier. It has no queries, matching, or mask decoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, value):
        # GroupNorm reduces the same spatial elements after flattening. Using
        # N,C,HW avoids PyTorch 2.2's channels-last X/dY mismatch in its backward
        # kernel, including the ambiguous 1x1 tensor from pyramid pooling.
        shape = value.shape
        return super().forward(value.reshape(shape[0], shape[1], -1)).reshape(shape)


def conv_norm_relu(in_channels, out_channels, kernel_size):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2, bias=False),
                         SpatialGroupNorm(32, out_channels), nn.ReLU(inplace=False))


class UPerNetHead(nn.Module):
    def __init__(self, in_channels, width=256, num_classes=5,
                 pool_scales=(1, 2, 3, 6), gradient_checkpointing=True, region_correction=False):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError('UPerNet requires four stride-4/8/16/32 features')
        self.gradient_checkpointing = gradient_checkpointing
        self.ppm = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(scale),
            conv_norm_relu(in_channels[-1], width, 1)) for scale in pool_scales)
        self.ppm_bottleneck = conv_norm_relu(in_channels[-1] + len(pool_scales) * width, width, 3)
        self.laterals = nn.ModuleList(conv_norm_relu(c, width, 1) for c in in_channels[:-1])
        self.fpn = nn.ModuleList(conv_norm_relu(width, width, 3) for _ in in_channels[:-1])
        self.fpn_bottleneck = conv_norm_relu(4 * width, width, 3)
        self.classifier = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(width, num_classes, 1))
        self.auxiliary = nn.Sequential(conv_norm_relu(in_channels[2], width, 3),
            nn.Dropout2d(0.1), nn.Conv2d(width, num_classes, 1))
        # Preserve both the existing weights and the caller's RNG trajectory.
        self.region_correction = None
        if region_correction:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261117)
                self.region_correction = RegionSemanticCorrection(width, num_classes)

    def _forward(self, *features):
        deepest = features[-1]
        pooled = [deepest] + [F.interpolate(branch(deepest), deepest.shape[-2:],
                    mode='bilinear', align_corners=False) for branch in self.ppm]
        laterals = [layer(feature) for layer, feature in zip(self.laterals, features[:-1])]
        laterals.append(self.ppm_bottleneck(torch.cat(pooled, dim=1)))
        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i],
                laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        outputs = [layer(feature) for layer, feature in zip(self.fpn, laterals[:-1])] + [laterals[-1]]
        outputs = [outputs[0]] + [F.interpolate(x, outputs[0].shape[-2:],
                      mode='bilinear', align_corners=False) for x in outputs[1:]]
        fused = self.fpn_bottleneck(torch.cat(outputs, dim=1))
        logits = self.classifier(fused)
        if self.region_correction is not None:
            logits = self.region_correction(fused, logits)
        auxiliary = self.auxiliary(features[2]) if self.training else None
        return logits, auxiliary

    def forward(self, features):
        active = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        logits, auxiliary = (checkpoint(self._forward, *features, use_reentrant=False)
                             if active else self._forward(*features))
        return {'pixel_logits': logits, 'auxiliary_logits': auxiliary}


class RegionSemanticCorrection(nn.Module):
    """OCR-inspired class-region gather/distribute followed by a logit residual.

    This is a task-specific residual head, not a reproduction of OCRNet.
    Five soft class regions may each contain disconnected objects or holes.
    The existing PPM/FPN supplies spatial surroundings. No fixed neighborhood
    smoothing, object filling, confidence mask, or extra loss is introduced.
    """
    def __init__(self, width=256, num_classes=5, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.values = conv_norm_relu(width, hidden, 1)
        self.queries = nn.Conv2d(width + num_classes, hidden, 1, bias=False)
        self.keys = nn.Linear(hidden, hidden, bias=False)
        self.mix = conv_norm_relu(width + hidden + num_classes, hidden, 1)
        self.delta = nn.Conv2d(hidden, num_classes, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    @staticmethod
    def gather_regions(values, probabilities):
        # Sum over the whole spatial domain, independently for each image.
        # Normalize each class by its own soft area, including absent classes.
        with torch.autocast(device_type=values.device.type, enabled=False):
            weights = probabilities.float().flatten(2)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            return torch.bmm(weights, values.float().flatten(2).transpose(1, 2))

    def forward(self, features, logits):
        probabilities = logits.float().softmax(dim=1)
        values = self.values(features)
        regions = self.gather_regions(values, probabilities)
        query = self.queries(torch.cat((features, probabilities.to(features.dtype)), dim=1))
        keys = self.keys(regions.to(query.dtype))
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention = torch.bmm(query.float().flatten(2).transpose(1, 2),
                                  keys.float().transpose(1, 2)) * self.hidden ** -0.5
            context = torch.bmm(attention.softmax(-1), regions.float())
            context = context.transpose(1, 2).reshape(features.shape[0], self.hidden, *features.shape[-2:])
        hidden = self.mix(torch.cat((features, context.to(features.dtype),
                                     probabilities.to(features.dtype)), dim=1))
        # Accumulate in FP32, before interpolation and the model's log_softmax.
        return logits.float() + self.delta(hidden).float()


class PixelCrossEntropyCriterion(nn.Module):
    """Full-resolution semantic CE plus 0.4-times auxiliary FCN CE."""
    def __init__(self, auxiliary_weight=0.4, ignore_index=255):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        valid = targets != self.ignore_index
        denominator = valid.sum().clamp_min(1)
        # semantic_logits are normalized log probabilities computed from the
        # upsampled pixel logits; NLL here is exactly pixel cross-entropy.
        main = F.nll_loss(outputs['semantic_logits'].float(), targets,
                         ignore_index=self.ignore_index, reduction='sum') / denominator
        auxiliary = outputs.get('auxiliary_logits')
        if auxiliary is None:
            raise ValueError('Training UPerNet requires its auxiliary logits')
        auxiliary = F.interpolate(auxiliary.float(), targets.shape[-2:],
                                  mode='bilinear', align_corners=False)
        aux = F.cross_entropy(auxiliary, targets, ignore_index=self.ignore_index,
                             reduction='sum') / denominator
        return {'loss_semantic_ce': main, 'loss_auxiliary_ce': aux,
                'loss_total': main + self.auxiliary_weight * aux}
