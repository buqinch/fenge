"""Hungarian class/mask supervision for Mask2Former semantic region queries.

Each visible class is one potentially disconnected mask. Both matching and
mask-loss samples use original GT pixel centers, guaranteeing samples of small
classes. No low-resolution label map or pixel CE surrogate is used.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


def _draw_indices(locations: Tensor, count: int) -> Tensor:
    if count == 0:
        return locations[:0]
    if not len(locations):
        raise ValueError("Cannot sample an empty original-GT class")
    return locations[torch.randint(len(locations), (count,), device=locations.device)]


def _point_grid(indices: Tensor, height: int, width: int) -> Tensor:
    x = (indices.remainder(width).float() + .5) * (2 / width) - 1
    y = (torch.div(indices, width, rounding_mode="floor").float() + .5) * (2 / height) - 1
    return torch.stack((x, y), dim=-1).reshape(1, 1, -1, 2)


def _sample_prediction(masks: Tensor, indices: Tensor, height: int, width: int) -> Tensor:
    return F.grid_sample(masks[None].float(), _point_grid(indices, height, width),
                         mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0]


class Mask2FormerCriterion(nn.Module):
    """Full class+BCE+Dice set loss, independently matched at every stage.

    Mask matching shares sampled points across queries. Matched-mask losses add
    uncertain points (75% of the remaining budget, from3x candidates), random
    points, and guaranteed foreground/background points at the original grid.
    Every auxiliary prediction receives the same2/5/5 weights as the final one.
    Entirely ignored images contribute graph-connected zero, without fake GT.
    """

    def __init__(self, num_classes: int = 5, num_points: int = 4096,
                 class_weight: float = 2., mask_weight: float = 5., dice_weight: float = 5.,
                 no_object_weight: float = .1, ignore_index: int = 255) -> None:
        super().__init__()
        for name, value in (("num_classes", num_classes), ("num_points", num_points)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes != 5 or num_points < num_classes:
            raise ValueError("LuSNAR requires five classes and at least five points")
        for name, value in (("class_weight", class_weight), ("mask_weight", mask_weight),
                            ("dice_weight", dice_weight), ("no_object_weight", no_object_weight)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if no_object_weight == 0 or class_weight + mask_weight + dice_weight == 0:
            raise ValueError("no_object_weight and at least one set-loss weight must be positive")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral) or 0 <= ignore_index < num_classes:
            raise ValueError("ignore_index must be an integer outside the five classes")
        self.num_classes, self.num_points, self.ignore_index = int(num_classes), int(num_points), int(ignore_index)
        self.class_weight, self.mask_weight, self.dice_weight = float(class_weight), float(mask_weight), float(dice_weight)
        weights = torch.ones(self.num_classes + 1)
        weights[-1] = no_object_weight
        self.register_buffer("no_object_weights", weights)
        self.latest_point_stats: list[dict] = []
        self.latest_matching: list[list[dict]] = []

    def _shared_points(self, target: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Matching coordinates with explicit coverage of every visible class."""
        flat = target.flatten()
        valid = (flat != self.ignore_index).nonzero(as_tuple=False).flatten()
        per_class = max(1, min(64, self.num_points // (2 * len(labels))))
        samples = [_draw_indices((flat == label).nonzero(as_tuple=False).flatten(), per_class) for label in labels]
        samples.append(_draw_indices(valid, self.num_points - per_class * len(labels)))
        indices = torch.cat(samples)
        return indices, (flat[indices][None] == labels[:, None]).float()

    def _mask_points(self, mask: Tensor, target: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        """Uncertainty mining selects indices only; final sampled logits keep gradients."""
        height, width = target.shape
        with torch.no_grad():
            flat = target.flatten()
            valid = (flat != self.ignore_index).nonzero(as_tuple=False).flatten()
            positive = (flat == label).nonzero(as_tuple=False).flatten()
            negative = ((flat != label) & (flat != self.ignore_index)).nonzero(as_tuple=False).flatten()
            guaranteed = min(64, max(1, self.num_points // 8))
            samples = [_draw_indices(positive, guaranteed)]
            if len(negative):
                samples.append(_draw_indices(negative, guaranteed))
            remaining = self.num_points - sum(len(row) for row in samples)
            uncertain_count = int(remaining * .75)
            candidates = _draw_indices(valid, max(remaining * 3, uncertain_count))
            if uncertain_count:
                values = _sample_prediction(mask[None].detach(), candidates, height, width)[0]
                order = torch.topk(-values.abs(), uncertain_count, sorted=False).indices
                samples.append(candidates[order])
            samples.append(_draw_indices(valid, remaining - uncertain_count))
            indices = torch.cat(samples)
            truth = (flat[indices] == label).float()
        return _sample_prediction(mask[None], indices, height, width)[0], truth

    def _stage(self, outputs: dict, targets: Tensor, stage: str) -> dict[str, Tensor]:
        classes, masks = outputs["pred_logits"].float(), outputs["pred_masks"].float()
        if classes.ndim != 3 or masks.ndim != 4 or classes.shape[:2] != masks.shape[:2]:
            raise ValueError("Expected class[B,Q,6] and mask[B,Q,Hm,Wm] logits")
        if classes.shape[0] != len(targets) or classes.shape[2] != self.num_classes + 1:
            raise ValueError("Output class/batch dimensions disagree with GT")
        assigned = torch.full(classes.shape[:2], self.num_classes, device=classes.device, dtype=torch.long)
        valid_images = (targets != self.ignore_index).flatten(1).any(1)
        zero = classes.sum() * 0 + masks.sum() * 0
        mask_bce, mask_dice = [], []
        matching_rows = []
        for image_index, target in enumerate(targets):
            labels = target[target != self.ignore_index].unique(sorted=True)
            if not len(labels):
                matching_rows.append({"batch_index": image_index, "query_indices": [], "labels": []})
                continue
            if len(labels) > classes.shape[1]:
                raise ValueError("There must be at least one query for each GT class")
            with torch.no_grad(), torch.autocast(device_type=masks.device.type, enabled=False):
                indices, truth = self._shared_points(target, labels)
                points = _sample_prediction(masks[image_index].detach(), indices, *target.shape)
                positive, negative = F.softplus(-points), F.softplus(points)
                mask_cost = (positive @ truth.T + negative @ (1 - truth).T) / len(indices)
                probabilities = points.sigmoid()
                dice_cost = 1 - (2 * probabilities @ truth.T + 1) / (
                    probabilities.sum(1)[:, None] + truth.sum(1)[None] + 1)
                class_cost = -classes[image_index].softmax(-1)[:, labels]
                cost = self.class_weight * class_cost + self.mask_weight * mask_cost + self.dice_weight * dice_cost
                query_indices, target_indices = linear_sum_assignment(cost.cpu().numpy())
                qi = torch.as_tensor(query_indices, device=classes.device, dtype=torch.long)
                ti = torch.as_tensor(target_indices, device=classes.device, dtype=torch.long)
                assigned[image_index, qi] = labels[ti]
            matching_rows.append({"batch_index": image_index, "query_indices": query_indices.tolist(),
                                  "labels": labels[ti].cpu().tolist()})
            self.latest_point_stats.append({"stage": stage, "batch_index": image_index,
                "original_gt_hw": list(target.shape), "matching_points": len(indices),
                "points_per_visible_class": {str(int(label)): int((target.flatten()[indices] == label).sum()) for label in labels},
                "ignored_pixels_sampled": 0, "uncertainty_oversample_ratio": 3.,
                "uncertain_fraction_of_remaining_mask_points": .75,
                "mask_loss_points_per_matched_class": self.num_points})
            with torch.autocast(device_type=masks.device.type, enabled=False):
                for query, label in zip(qi, labels[ti]):
                    predicted, truth = self._mask_points(masks[image_index, query], target, label)
                    mask_bce.append(F.binary_cross_entropy_with_logits(predicted, truth))
                    probability = predicted.sigmoid()
                    mask_dice.append(1 - (2 * (probability * truth).sum() + 1) / (
                        probability.sum() + truth.sum() + 1))
        self.latest_matching.append(matching_rows)
        classification = (F.cross_entropy(classes[valid_images].transpose(1, 2), assigned[valid_images],
                                          weight=self.no_object_weights) if bool(valid_images.any()) else zero)
        return {"loss_class": classification, "loss_mask": torch.stack(mask_bce).mean() if mask_bce else zero,
                "loss_dice": torch.stack(mask_dice).mean() if mask_dice else zero}

    def _weighted(self, losses: dict[str, Tensor]) -> Tensor:
        return self.class_weight * losses["loss_class"] + self.mask_weight * losses["loss_mask"] + self.dice_weight * losses["loss_dice"]

    def forward(self, outputs: dict, targets: Tensor) -> dict[str, Tensor]:
        if targets.ndim != 3 or targets.dtype != torch.long or not len(targets) or not targets.numel():
            raise ValueError("GT must be nonempty int64[B,H,W] at original resolution")
        valid = targets != self.ignore_index
        if bool((((targets < 0) | (targets >= self.num_classes)) & valid).any()):
            raise ValueError("GT contains an invalid class ID")
        if targets.device != outputs["pred_logits"].device:
            raise ValueError("GT and predictions must share a device")
        self.latest_point_stats, self.latest_matching = [], []
        losses = self._stage(outputs, targets, "final")
        total = self._weighted(losses)
        auxiliary = total * 0
        for index, stage in enumerate(outputs.get("aux_outputs", [])):
            extra = self._stage(stage, targets, f"aux_{index}")
            auxiliary = auxiliary + self._weighted(extra)
            losses.update({f"{key}_{index}": value for key, value in extra.items()})
        losses["loss_auxiliary"] = auxiliary
        losses["loss_total"] = total + auxiliary
        return losses
