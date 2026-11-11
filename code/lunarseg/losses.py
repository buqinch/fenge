"""Pixel supervision and class/mask set supervision for semantic regions."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


class SegmentationCriterion(nn.Module):
    """Supervise full-size semantics and match region queries to visible classes.

    Each present semantic class supplies one target mask, including disconnected
    pieces and holes. Queries are not physical crater/rock instance proposals.
    Point samples come from original labels, preserving small classes that would
    disappear if target labels were first reduced to the mask feature resolution.
    """

    def __init__(
        self,
        num_classes: int = 5,
        semantic_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        aux_weight: float = 0.5,
        num_points: int = 4096,
        no_object_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2 or num_points < num_classes:
            raise ValueError("num_classes >= 2 and num_points >= num_classes are required")
        self.num_classes = num_classes
        self.semantic_weight = semantic_weight
        self.dice_weight = dice_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.aux_weight = aux_weight
        self.num_points = num_points
        eos_weight = torch.ones(num_classes + 1)
        eos_weight[-1] = no_object_weight
        self.register_buffer("eos_weight", eos_weight)

    def _semantic_losses(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        logits = logits.float()
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
        ce = F.cross_entropy(logits, targets)
        probabilities = logits.softmax(dim=1)
        # Avoid a full BxCxHxW one-hot target allocation at 1024 resolution.
        intersection, mass, present = [], [], []
        for index in range(self.num_classes):
            truth = targets == index
            pred = probabilities[:, index]
            intersection.append((pred * truth).sum())
            mass.append(pred.sum() + truth.sum())
            present.append(truth.any())
        score = (2 * torch.stack(intersection) + 1) / (torch.stack(mass) + 1)
        keep = torch.stack(present)
        dice = (1 - score[keep]).mean()
        return {"loss_semantic_ce": ce, "loss_semantic_dice": dice}

    def _points(self, target: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Shared uniform samples plus guaranteed samples of each visible class."""
        flat = target.flatten()
        per_class = max(1, min(64, self.num_points // (2 * len(labels))))
        selected = []
        for label in labels:
            locations = (flat == label).nonzero(as_tuple=False).flatten()
            choices = torch.randint(len(locations), (per_class,), device=target.device)
            selected.append(locations[choices])
        remaining = self.num_points - per_class * len(labels)
        selected.append(torch.randint(flat.numel(), (remaining,), device=target.device))
        indices = torch.cat(selected)
        height, width = target.shape
        x = (indices.remainder(width).float() + 0.5) / width * 2 - 1
        y = (torch.div(indices, width, rounding_mode="floor").float() + 0.5) / height * 2 - 1
        grid = torch.stack((x, y), dim=-1).reshape(1, 1, -1, 2)
        masks = (flat[indices].unsqueeze(0) == labels.unsqueeze(1)).float()
        return grid, masks

    def _set_losses(self, output: dict[str, Tensor], targets: Tensor) -> dict[str, Tensor]:
        class_logits, mask_logits = output["pred_logits"].float(), output["pred_masks"].float()
        batch, queries, classes = class_logits.shape
        if classes != self.num_classes + 1 or batch != len(targets):
            raise ValueError("Region logits must have shape [batch, queries, num_classes + 1]")
        if mask_logits.shape[:2] != (batch, queries):
            raise ValueError("Region mask and classification query dimensions must agree")
        assigned = torch.full((batch, queries), self.num_classes, device=targets.device, dtype=torch.long)
        matched_masks, matched_truth = [], []
        for item in range(batch):
            labels = targets[item].unique(sorted=True)
            if len(labels) > queries:
                raise ValueError("There must be at least as many queries as present semantic classes")
            grid, truth = self._points(targets[item], labels)
            points = F.grid_sample(
                mask_logits[item].unsqueeze(0), grid,
                mode="bilinear", padding_mode="border", align_corners=False,
            )[0, :, 0, :]
            with torch.no_grad():
                # Pairwise BCE: QxP @ PxK, without materializing QxKxP.
                positive, negative = F.softplus(-points), F.softplus(points)
                mask_cost = (positive @ truth.T + negative @ (1 - truth).T) / truth.shape[1]
                probabilities = points.sigmoid()
                dice_cost = 1 - (2 * probabilities @ truth.T + 1) / (
                    probabilities.sum(-1, keepdim=True) + truth.sum(-1).unsqueeze(0) + 1
                )
                class_cost = -class_logits[item].softmax(-1)[:, labels]
                cost = self.class_weight * class_cost + self.mask_weight * mask_cost + self.dice_weight * dice_cost
                row, column = linear_sum_assignment(cost.detach().cpu().numpy())
                row = torch.as_tensor(row, device=targets.device, dtype=torch.long)
                column = torch.as_tensor(column, device=targets.device, dtype=torch.long)
            assigned[item, row] = labels[column]
            matched_masks.append(points[row])
            matched_truth.append(truth[column])
        classification = F.cross_entropy(class_logits.transpose(1, 2), assigned, weight=self.eos_weight)
        masks, truth = torch.cat(matched_masks), torch.cat(matched_truth)
        mask_bce = F.binary_cross_entropy_with_logits(masks, truth)
        probabilities = masks.sigmoid()
        mask_dice = (1 - (2 * (probabilities * truth).sum(-1) + 1) / (
            probabilities.sum(-1) + truth.sum(-1) + 1
        )).mean()
        return {"loss_region_class": classification, "loss_region_mask": mask_bce, "loss_region_dice": mask_dice}

    def forward(self, outputs: dict, targets: Tensor) -> dict[str, Tensor]:
        if targets.ndim != 3 or targets.dtype != torch.long:
            raise ValueError("Targets must be int64 class IDs with shape [batch, height, width]")
        if targets.numel() == 0 or targets.min() < 0 or targets.max() >= self.num_classes:
            raise ValueError("Targets contain an invalid semantic class ID")
        losses = self._semantic_losses(outputs["semantic_logits"], targets)
        total = self.semantic_weight * losses["loss_semantic_ce"] + self.dice_weight * losses["loss_semantic_dice"]
        if "pred_logits" in outputs:
            region = self._set_losses(outputs, targets)
            losses.update(region)
            total = total + self.class_weight * region["loss_region_class"]
            total = total + self.mask_weight * region["loss_region_mask"] + self.dice_weight * region["loss_region_dice"]
            auxiliary = total.new_zeros(())
            for stage in outputs.get("aux_outputs", []):
                extra = self._set_losses(stage, targets)
                auxiliary = auxiliary + self.class_weight * extra["loss_region_class"]
                auxiliary = auxiliary + self.mask_weight * extra["loss_region_mask"] + self.dice_weight * extra["loss_region_dice"]
            losses["loss_auxiliary"] = auxiliary
            total = total + self.aux_weight * auxiliary
        losses["loss_total"] = total
        return losses
