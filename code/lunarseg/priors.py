"""Sparse semantic affinity with stateless per-image sampling.

Module construction and sampling preserve the training RNG state.
"""
from __future__ import annotations

import hashlib
import json
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
