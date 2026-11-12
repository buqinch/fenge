"""Sparse semantic affinity and supervised, aligned geometric consistency.

All sampling uses a per-image hash and a local NumPy generator. Neither module
construction nor sampling advances the training path's random generators.
Geometry changes the input context only modestly; every transformed view must
also receive the ordinary GT-supervised segmentation loss in the trainer.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


AFFINITY_RADII = (8, 16, 32, 64)
# Reverse directions describe the same unordered pair: these four offsets
# represent all eight directions without ever including both (i,j) and (j,i).
_CANONICAL_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1))
GEOMETRY_SCALES = (1.0, 0.9375)
GEOMETRY_MAX_TRANSLATION = 16


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _rng(seed: int, sample_key: str, epoch: int, namespace: str) -> np.random.Generator:
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("sample keys must be nonempty strings")
    payload = json.dumps([namespace, _integer("seed", seed), sample_key,
                          _integer("epoch", epoch)], separators=(",", ":"), ensure_ascii=False)
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "little"))


def _keys(sample_keys: Sequence[str], batch: int) -> list[str]:
    if isinstance(sample_keys, (str, bytes)) or len(sample_keys) != batch:
        raise ValueError("one sample key is required for every image")
    values = list(sample_keys)
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError("sample keys must be nonempty strings")
    return values


def _validate_targets(targets: Tensor, batch: int, shape: tuple[int, int] | None = None) -> None:
    if targets.ndim != 3 or targets.dtype != torch.long or len(targets) != batch:
        raise ValueError("targets must be int64 [B,H,W]")
    if shape is not None and tuple(targets.shape[-2:]) != shape:
        raise ValueError("target and image grids must agree")
    if not bool(((targets >= 0) & (targets <= 4) | (targets == 255)).all()):
        raise ValueError("targets allow only class IDs 0..4 and ignored ID255")


class SemanticAffinity(nn.Module):
    """Learn symmetric GT class agreement on <=512 original-grid pairs/image.

    Each class contributes <=2048 original-grid anchors (all pixels when rare).
    Each sign receives at most 256 pairs. Class pair x offset-radius strata are
    served in shuffled round-robin order, exhausting small strata before larger
    ones receive the remaining budget. Exact candidate counts and reservoir
    sampling avoid an all-pixel squared matrix. Sky/255 endpoints are excluded;
    boundary points and tiny rocks remain eligible in the original GT grid.
    """

    def __init__(self, width: int = 128, projected: int = 32,
                 quota: int = 512, seed: int = 4242) -> None:
        super().__init__()
        self.width = _integer("width", width, 1)
        self.projected = _integer("projected", projected, 1)
        self.quota = _integer("quota", quota)
        if self.quota > 512:
            raise ValueError("quota must be <=512 (at most256 pairs per sign)")
        self.seed = _integer("seed", seed)
        # default_generator only seeds CPU: torch.manual_seed would also seed
        # CUDA generators even inside a CPU-only fork_rng context.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.project = nn.Conv2d(self.width, self.projected, 1)
            self.predictor = nn.Sequential(nn.Linear(3 * self.projected + 2, 64),
                                           nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def pair_descriptors(first: Tensor, second: Tensor, relative_offsets: Tensor) -> Tensor:
        """Exactly invariant to exchanging endpoints and negating the offset."""
        return torch.cat(((first + second) * 0.5, (first - second).abs(),
                          first * second, relative_offsets.abs()), dim=-1)

    def sample_pairs(self, target: Tensor, sample_key: str, epoch: int) -> tuple[Tensor, Tensor, dict]:
        """Return CPU xyxy pairs, float sign targets, and JSON-safe diagnostics."""
        if target.ndim != 2 or target.dtype != torch.long:
            raise ValueError("sample_pairs target must be int64 [H,W]")
        _validate_targets(target[None], 1)
        rng = _rng(self.seed, sample_key, epoch, "semantic_affinity_v1")
        labels = target.detach().cpu().numpy()
        height, width = labels.shape
        positive_quota, negative_quota = (self.quota + 1) // 2, self.quota // 2
        capacity = max(positive_quota, negative_quota)
        pools: dict[tuple[int, int, int], np.ndarray] = {}
        counts: dict[tuple[int, int, int], int] = {}
        anchor_limit = 2048
        anchor_rows, anchor_stats = [], {}
        if capacity:
            for label in range(4):
                locations = np.flatnonzero(labels == label)
                count = min(len(locations), anchor_limit)
                chosen = locations if len(locations) <= anchor_limit else locations[rng.choice(len(locations), count, replace=False)]
                anchor_rows.append(chosen)
                anchor_stats[str(label)] = {"available_pixels": int(len(locations)), "selected_anchors": count,
                                            "all_class_pixels_preserved": len(locations) <= anchor_limit}
            anchors = np.concatenate(anchor_rows) if anchor_rows else np.empty(0, dtype=np.int64)
            ay, ax = np.divmod(anchors, width)
            encoded = []
            for radius in AFFINITY_RADII:
                for ux, uy in _CANONICAL_DIRECTIONS:
                    for sign in (-1, 1):
                        xx, yy = ax + sign * ux * radius, ay + sign * uy * radius
                        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                        first, second = anchors[valid], yy[valid] * width + xx[valid]
                        keep = labels.reshape(-1)[second] < 4
                        first, second = first[keep], second[keep]
                        encoded.append(np.minimum(first, second) * (height * width) + np.maximum(first, second))
            # Endpoints anchored at either end can produce the same unordered
            # pair. A single canonical integer removes all reverse duplicates.
            codes = np.unique(np.concatenate(encoded)) if encoded else np.empty(0, dtype=np.int64)
            first, second = np.divmod(codes, height * width)
            ya, xa = np.divmod(first, width)
            yb, xb = np.divmod(second, width)
            distances = np.maximum(np.abs(xa - xb), np.abs(ya - yb))
            a, b = labels.reshape(-1)[first], labels.reshape(-1)[second]
            class_codes = np.minimum(a, b) * 4 + np.maximum(a, b)
            for radius in AFFINITY_RADII:
                for a in range(4):
                    for b in range(a, 4):
                        locations = np.flatnonzero((class_codes == a * 4 + b) & (distances == radius))
                        if not len(locations):
                            continue
                        key = (a, b, radius)
                        counts[key] = len(locations)
                        positions = locations[rng.choice(len(locations), min(capacity, len(locations)), replace=False)]
                        pools[key] = np.column_stack((xa[positions], ya[positions], xb[positions], yb[positions])).astype(np.int64)
        selected, signs, selected_strata = [], [], []
        for positive, budget in ((True, positive_quota), (False, negative_quota)):
            eligible = [key for key in sorted(pools) if (key[0] == key[1]) == positive]
            positions = {key: 0 for key in eligible}
            for key in eligible:
                rng.shuffle(pools[key], axis=0)
            while eligible and budget:
                for index in rng.permutation(len(eligible)):
                    key = eligible[int(index)]
                    pos = positions[key]
                    if pos >= len(pools[key]):
                        continue
                    selected.append(pools[key][pos])
                    signs.append(float(positive))
                    selected_strata.append(key)
                    positions[key] += 1
                    budget -= 1
                    if not budget:
                        break
                eligible = [key for key in eligible if positions[key] < len(pools[key])]
        pairs = np.asarray(selected, dtype=np.int64).reshape(-1, 4)
        sign_values = np.asarray(signs, dtype=np.float32)
        class_counts, distance_counts = {}, {}
        for a, b, radius in selected_strata:
            name = f"{a}-{b}"
            class_counts[name] = class_counts.get(name, 0) + 1
            distance_counts[str(radius)] = distance_counts.get(str(radius), 0) + 1
        if len(pairs):
            endpoints = pairs.reshape(-1, 2)
            oy, ox = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing="ij")
            xx = np.clip(endpoints[:, 0, None] + ox.reshape(1, -1), 0, width - 1)
            yy = np.clip(endpoints[:, 1, None] + oy.reshape(1, -1), 0, height - 1)
            on_boundary = (labels[yy, xx] != labels[endpoints[:, 1], endpoints[:, 0], None]).any(1)
            boundary_count = int(on_boundary.reshape(-1, 2).any(1).sum())
        else:
            boundary_count = 0
        stats = {"sample_key": sample_key, "epoch": int(epoch), "original_hw": [height, width],
                 "positive_count": int(sign_values.sum()), "negative_count": int(len(sign_values) - sign_values.sum()),
                 "candidate_counts": {f"{a}-{b}@{r}": int(n) for (a, b, r), n in sorted(counts.items())},
                 "candidate_count": int(sum(counts.values())), "selected_by_class_pair": class_counts,
                 "selected_by_distance": distance_counts, "boundary_endpoint_pairs": boundary_count,
                 "boundary_radius": 3, "duplicate_unordered_pairs": 0,
                 "sampling_sha256": hashlib.sha256(pairs.astype("<i8").tobytes() + sign_values.astype("<f4").tobytes()).hexdigest(),
                 "selected_pairs_preview": [{"xyxy": p.tolist(), "same_class": bool(s)} for p, s in zip(pairs[:8], sign_values[:8])],
                 "sampling_reliability": {"gt_resolution": "original", "no_downsampled_gt": True,
                    "boundary_points_removed": False, "sky_and_ignore_endpoints_excluded": True,
                    "sampling_without_replacement": True, "quota_shortage_repeated": False,
                    "candidate_domain": "unordered 8-direction pairs from bounded original-GT class-balanced anchors at radii8/16/32/64",
                    "anchor_limit_per_class": anchor_limit, "anchors_by_class": anchor_stats,
                    "rare_class_definition": "class pixel count<=anchor_limit; all its pixels are anchors",
                    "candidate_counts_cover_full_image_pairs": False,
                    "balance_domain": "available class-pair x radius strata within each sign",
                    "physical_instance_or_shadow_supervision": False}}
        return torch.from_numpy(pairs), torch.from_numpy(sign_values), stats

    def forward(self, features: Tensor, targets: Tensor, sample_keys: Sequence[str], epoch: int) -> tuple[Tensor, dict]:
        if features.ndim != 4 or features.shape[1] != self.width or not features.is_floating_point() or len(features) == 0:
            raise ValueError("features must be nonempty floating [B,width,Hf,Wf]")
        _validate_targets(targets, len(features))
        keys = _keys(sample_keys, len(features))
        _integer("epoch", epoch)
        projected = self.project(features)
        loss = projected.sum() * 0
        # Include new parameters in the empty graph too (DDP/empty-image safe).
        loss = loss + sum((p.sum() * 0 for p in self.predictor.parameters()))
        rows = []
        height, width = targets.shape[-2:]
        for index, key in enumerate(keys):
            pairs, signs, stat = self.sample_pairs(targets[index], key, epoch)
            if len(pairs):
                xy = pairs.to(device=features.device, dtype=torch.float32).reshape(-1, 2)
                grid = torch.stack(((xy[:, 0] + 0.5) * (2 / width) - 1,
                                    (xy[:, 1] + 0.5) * (2 / height) - 1), dim=-1)[None, None]
                values = F.grid_sample(projected[index:index + 1].float(), grid, mode="bilinear",
                                       padding_mode="border", align_corners=False)[0, :, 0].T.reshape(-1, 2, self.projected)
                offsets = (xy[1::2] - xy[::2]) / xy.new_tensor([width, height])
                logits = self.predictor(self.pair_descriptors(values[:, 0], values[:, 1], offsets)).flatten()
                signs = signs.to(logits.device)
                bce = F.binary_cross_entropy_with_logits(logits.float(), signs, reduction="none")
                group_losses = []
                for positive, name in ((True, "positive_loss"), (False, "negative_loss")):
                    mask = signs.bool() == positive
                    if bool(mask.any()):
                        group_loss = bce[mask].mean()
                        group_losses.append(group_loss)
                        stat[name] = float(group_loss.detach())
                    else:
                        stat[name] = None
                loss = loss + torch.stack(group_losses).mean()
                stat["valid_sign_groups"] = len(group_losses)
            else:
                stat.update(positive_loss=None, negative_loss=None, valid_sign_groups=0)
            rows.append(stat)
        stats = {"schema": "semantic_affinity_stats_v1", "batch_size": len(rows),
                 "selected_positive_pairs": sum(r["positive_count"] for r in rows),
                 "selected_negative_pairs": sum(r["negative_count"] for r in rows),
                 "valid_image_count": sum(r["valid_sign_groups"] > 0 for r in rows),
                 "gt_resolution_sampling": True, "images": rows}
        return loss / len(features), stats


def _translate(value: Tensor, dx: int, dy: int, fill: float | None) -> Tensor:
    height, width = value.shape[-2:]
    padding = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    if fill is None:
        padded = F.pad(value, padding, mode="replicate")
    else:
        padded = F.pad(value, padding, mode="constant", value=fill)
    left, top = max(-dx, 0), max(-dy, 0)
    return padded[..., top:top + height, left:left + width]


def _warp(value: Tensor, meta: dict, mode: str, fill: float | None = 0,
          antialias: bool = False) -> Tensor:
    if meta["hflip"]:
        value = value.flip(-1)
    size = tuple(meta["resized_hw"])
    if value.shape[-2:] != size:
        kwargs = {"align_corners": False, "antialias": antialias} if mode == "bilinear" else {}
        value = F.interpolate(value, size=size, mode=mode, **kwargs)
    left, top, right, bottom = meta["pad_ltrb"]
    padding = (left, right, top, bottom)
    value = F.pad(value, padding, mode="replicate") if fill is None else F.pad(value, padding, value=fill)
    return _translate(value, meta["dx"], meta["dy"], fill)


def make_geometry_view(images: Tensor, targets: Tensor, sample_keys: Sequence[str], epoch: int,
                       seed: int = 4242) -> tuple[Tensor, Tensor, Tensor, list[dict]]:
    """Hash-selected flip, scale1/.9375, centered pad, then +/-16px shift.

    RGB uses replicate padding; target uses255. ``nearest-exact`` for GT and
    ``align_corners=False`` for RGB share the pixel-center convention. Returned
    validity is in view-B coordinates. Input tensors are never modified.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point() or not len(images):
        raise ValueError("images must be nonempty floating [B,3,H,W]")
    _validate_targets(targets, len(images), tuple(images.shape[-2:]))
    if min(images.shape[-2:]) < 2 or images.device != targets.device:
        raise ValueError("images/targets need the same device and spatial dimensions>=2")
    keys = _keys(sample_keys, len(images))
    height, width = images.shape[-2:]
    transformed, labels, masks, metadata = [], [], [], []
    for index, key in enumerate(keys):
        rng = _rng(seed, key, epoch, "geometry_view_v1")
        scale = GEOMETRY_SCALES[int(rng.integers(len(GEOMETRY_SCALES)))]
        resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
        left, top = (width - resized_w) // 2, (height - resized_h) // 2
        meta = {"schema": "geometry_view_v1", "sample_key": key, "epoch": int(epoch), "seed": int(seed),
                "hflip": bool(rng.integers(2)), "dx": int(rng.integers(-16, 17)), "dy": int(rng.integers(-16, 17)),
                "scale": scale, "source_hw": [height, width], "resized_hw": [resized_h, resized_w],
                "pad_ltrb": [left, top, width - resized_w - left, height - resized_h - top],
                "order": "hflip_resize_centerpad_translate", "align_corners": False,
                "image_resize_antialias": True, "target_resize": "nearest-exact",
                "image_padding": "replicate", "target_padding": 255}
        transformed.append(_warp(images[index:index + 1], meta, "bilinear", None, True))
        target = _warp(targets[index:index + 1, None].float(), meta, "nearest-exact", 255)[:, 0].long()
        labels.append(target)
        masks.append(target != 255)
        metadata.append(meta)
    return torch.cat(transformed), torch.cat(labels), torch.cat(masks), metadata


def align_geometry_logits(logits_b: Tensor, metadata: Sequence[dict]) -> tuple[Tensor, Tensor]:
    """Inverse continuous-logit mapping for diagnostics, not scaled JS training.

    For scaled consistency use geometry_consistency_js, which maps probabilities
    A->B before comparison. Only flip/translation is exactly invertible.
    """
    if logits_b.ndim != 4 or len(metadata) != len(logits_b):
        raise ValueError("one geometry metadata record is required per logit image")
    aligned, masks = [], []
    for index, meta in enumerate(metadata):
        if tuple(logits_b.shape[-2:]) != tuple(meta["source_hw"]):
            raise ValueError("logits must be on the original geometry grid")
        value = _translate(logits_b[index:index + 1], -meta["dx"], -meta["dy"], 0)
        valid_b = _warp(logits_b.new_ones((1, 1, *meta["source_hw"])), meta, "nearest-exact", 0)
        valid = _translate(valid_b, -meta["dx"], -meta["dy"], 0)
        left, top, _, _ = meta["pad_ltrb"]
        height, width = meta["resized_hw"]
        value = value[..., top:top + height, left:left + width]
        valid = valid[..., top:top + height, left:left + width]
        if tuple(value.shape[-2:]) != tuple(meta["source_hw"]):
            value = F.interpolate(value, size=meta["source_hw"], mode="bilinear", align_corners=False)
            valid = F.interpolate(valid, size=meta["source_hw"], mode="nearest-exact")
        if meta["hflip"]:
            value, valid = value.flip(-1), valid.flip(-1)
        aligned.append(value)
        masks.append(valid[:, 0].bool())
    return torch.cat(aligned), torch.cat(masks)


def _js_from_probabilities(first: Tensor, second: Tensor, valid: Tensor) -> tuple[Tensor, dict]:
    if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 5:
        raise ValueError("JS probabilities require matching [B,5,H,W]")
    if valid.dtype != torch.bool or tuple(valid.shape) != (len(first), *first.shape[-2:]):
        raise ValueError("JS validity must be bool [B,H,W]")
    eps = torch.finfo(torch.float32).tiny
    first, second = first.float(), second.float()
    mean = (first + second) * 0.5
    js = (0.5 * (first * (first.clamp_min(eps).log() - mean.clamp_min(eps).log())
                 + second * (second.clamp_min(eps).log() - mean.clamp_min(eps).log()))).sum(1).clamp_min(0)
    image_losses = []
    for index in range(len(first)):
        image_losses.append(js[index][valid[index]].mean() if bool(valid[index].any()) else js[index].sum() * 0)
    return torch.stack(image_losses).mean(), {"valid_pixels_by_image": valid.sum((1, 2)).cpu().tolist(),
                                            "valid_pixels": int(valid.sum()), "reduction": "per_image_valid_pixel_mean_then_batch_mean"}


def masked_js(logits_a: Tensor, aligned_logits_b: Tensor, valid_mask: Tensor,
              targets: Tensor | None = None) -> tuple[Tensor, dict]:
    """JS on aligned logits, with exactly one softmax per view; sky is retained."""
    valid = valid_mask
    if targets is not None:
        _validate_targets(targets, len(logits_a), tuple(logits_a.shape[-2:]))
        valid = valid & (targets != 255)
    return _js_from_probabilities(logits_a.float().softmax(1), aligned_logits_b.float().softmax(1), valid)


def geometry_consistency_js(logits_a: Tensor, logits_b: Tensor, targets_a: Tensor,
                            targets_b: Tensor, valid_b: Tensor,
                            metadata: Sequence[dict]) -> tuple[Tensor, dict]:
    """Compare A probabilities mapped to B with B on stable GT footprints.

    Each original softmax is computed once. Bilinearly mapped probabilities are
    renormalized. A bilinear footprint that mixes GT classes (e.g. a thin rock
    lost by shrinking) is excluded from JS but remains supervised by GT losses.
    This is an auxiliary term, never a replacement for either view's GT loss.
    """
    if logits_a.shape != logits_b.shape or logits_a.ndim != 4 or logits_a.shape[1] != 5 or not len(logits_a):
        raise ValueError("geometry logits must be matching nonempty [B,5,H,W]")
    _validate_targets(targets_a, len(logits_a), tuple(logits_a.shape[-2:]))
    _validate_targets(targets_b, len(logits_a), tuple(logits_a.shape[-2:]))
    if valid_b.dtype != torch.bool or valid_b.shape != targets_b.shape or len(metadata) != len(logits_a):
        raise ValueError("geometry validity/metadata must match the batch")
    probabilities_a, probabilities_b = logits_a.float().softmax(1), logits_b.float().softmax(1)
    mapped, valid_masks, initial_counts, unstable_counts = [], [], [], []
    for index, meta in enumerate(metadata):
        if tuple(meta["source_hw"]) != tuple(logits_a.shape[-2:]):
            raise ValueError("geometry metadata grid does not match logits")
        value = _warp(probabilities_a[index:index + 1], meta, "bilinear", 0)
        value = value / value.sum(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        with torch.no_grad():
            support = _warp((targets_a[index:index + 1, None] != 255).float(), meta, "bilinear", 0)[:, 0] >= 1 - 1e-6
            valid = valid_b[index:index + 1] & (targets_b[index:index + 1] != 255) & support
            initial_counts.append(int(valid.sum()))
            stable = torch.zeros_like(valid)
            for label in range(5):
                contribution = _warp((targets_a[index:index + 1, None] == label).float(), meta, "bilinear", 0)[:, 0]
                stable |= (targets_b[index:index + 1] == label) & (contribution >= 1 - 1e-6)
            unstable_counts.append(int((valid & ~stable).sum()))
            valid &= stable
        mapped.append(value)
        valid_masks.append(valid)
    loss, stats = _js_from_probabilities(torch.cat(mapped), probabilities_b, torch.cat(valid_masks))
    stats.update(schema="geometry_consistency_stats_v1", comparison_grid="transformed_view_B",
                 probability_softmax_count_per_view=1, mapped_probabilities_renormalized=True,
                 sky_retained=True, ignored_gt_excluded=True,
                 pixels_before_stability_by_image=initial_counts,
                 excluded_unstable_gt_pixels_by_image=unstable_counts,
                 gt_stability_rule="all bilinear interpolation mass belongs to transformed GT class",
                 geometry=metadata, requires_gt_loss_for_both_views=True)
    return loss, stats
