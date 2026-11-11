"""Scene-level pixel confusion matrices; no averaging of per-image mIoU."""
from __future__ import annotations

from contextlib import nullcontext
import math
import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASS_NAMES


class ConfusionMatrix:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target, prediction):
        target = torch.as_tensor(target).detach().reshape(-1)
        prediction = torch.as_tensor(prediction, device=target.device).detach().reshape(-1)
        if target.numel() != prediction.numel():
            raise ValueError("Prediction/target sizes differ")
        valid = (target >= 0) & (target < self.num_classes)
        if torch.any((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)):
            raise ValueError("Prediction class is out of bounds")
        counts = torch.bincount((target[valid] * self.num_classes + prediction[valid]).long(),
                                minlength=self.num_classes ** 2)
        self.matrix += counts.cpu().numpy().reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.matrix
        gt = cm.sum(axis=1)
        union = gt + cm.sum(axis=0) - np.diag(cm)
        iou = np.divide(np.diag(cm), union, out=np.zeros(self.num_classes, dtype=np.float64), where=union != 0)
        present = gt > 0
        return {"miou": float(iou[present].mean()) if present.any() else None,
                "class_iou": {CLASS_NAMES[i]: float(iou[i]) if present[i] else None for i in range(self.num_classes)},
                "class_precision": {CLASS_NAMES[i]: float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else None
                                    for i in range(self.num_classes)},
                "class_recall": {CLASS_NAMES[i]: float(cm[i, i] / gt[i]) if gt[i] else None
                                 for i in range(self.num_classes)},
                "gt_present_classes": [CLASS_NAMES[i] for i in range(self.num_classes) if present[i]],
                "pixel_accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
                "confusion_matrix_gt_rows_pred_cols": cm.tolist(),
                "absent_gt_policy": "Average only classes present in scene ground truth"}


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    if device.type == "cpu" and amp == "fp16":
        raise ValueError("fp16 AMP requires CUDA; use bf16 or none on CPU")
    return torch.autocast(device_type=device.type, dtype=dtype)


def tta_metadata(tta="none"):
    if tta not in ("none", "hflip"):
        raise ValueError("tta must be none or hflip")
    return {"test_time_augmentation": tta != "none", "tta_mode": tta,
            "tta_views": ["original", "horizontal_flip"] if tta == "hflip" else ["original"]}


@torch.inference_mode()
def predict_semantic_scores(model, images, tta="none"):
    """Return continuous class scores, without assigning any pixel IDs.

    ``none`` preserves the original logits and interpolation behavior exactly.
    ``hflip`` averages two float32 class probability maps in original image
    coordinates. Softmax accepts both pixel logits and the region head's log
    probabilities. Callers resize the resulting scores before final argmax;
    this helper is for validation/inference, never for training losses.
    """
    tta_metadata(tta)  # Validate before the first model forward.
    original = model(images)["semantic_logits"]
    if tta == "none":
        return original
    flipped = model(images.flip(-1))["semantic_logits"]
    if original.shape != flipped.shape:
        raise ValueError("Original and horizontal-flip predictions have different shapes")
    original = original.float().softmax(dim=1)
    flipped = flipped.float().softmax(dim=1).flip(-1)
    return (original + flipped) * 0.5


def validate_mountain_alpha(value):
    if isinstance(value, bool):
        raise ValueError("mountain_alpha must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mountain_alpha must be a finite positive number")
    return value


def adjust_mountain_scores(scores, mountain_alpha=1.0, tta="none"):
    """Scale only final class-3 probability, after resizing and before argmax.

    Single-view scores are logits (region head: log probabilities), so adding
    log(alpha) is exactly the decision obtained by multiplying that class's
    softmax probability by alpha. Multiplying negative logits is incorrect.
    TTA already returns mean probabilities and uses direct multiplication.
    No probability threshold or per-image choice is involved.
    """
    alpha = validate_mountain_alpha(mountain_alpha)
    tta_metadata(tta)
    if scores.ndim != 4 or scores.shape[1] != 5 or not scores.is_floating_point():
        raise ValueError("Expected floating Bx5xHxW final semantic scores")
    if alpha == 1.0:
        return scores
    adjusted = scores.float().clone()
    if tta == "none":
        adjusted[:, 3] += math.log(alpha)
    else:
        adjusted[:, 3] *= alpha
    return adjusted


def _evaluation_result(total, scenes, loader, count, tta):
    result = total.compute()
    result.update(image_count=count, scenes={s: {**v["matrix"].compute(), "image_count": v["image_count"]}
                                          for s, v in sorted(scenes.items())},
                  input_size=loader.dataset.input_size,
                  partial=loader.dataset.partial,
                  data_identity=loader.dataset.identity,
                  **tta_metadata(tta), continuous_scores_resized_before_argmax=True,
                  official_platform_score=False)
    if all(s in scenes for s in ("Moon_2", "Moon_4", "Moon_9")):
        miou = {s: result["scenes"][s]["miou"] for s in scenes}
        result["weighted_score"] = (miou["Moon_2"] + miou["Moon_4"] + 2 * miou["Moon_9"]) / 4
    return result


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp="bf16", tta="none", image_exporter=None,
                    mountain_alpha=1.0, mountain_alphas=None):
    tta_metadata(tta)
    alpha = validate_mountain_alpha(mountain_alpha)
    sweep = mountain_alphas is not None
    alphas = [validate_mountain_alpha(a) for a in mountain_alphas] if sweep else [alpha]
    if sweep and (not alphas or len(set(alphas)) != len(alphas) or 1.0 not in alphas
                  or any(a > 1 for a in alphas) or alpha != 1.0):
        raise ValueError("Sweep needs unique 0 < alphas <= 1 including 1; do not also set a scalar alpha")
    if sweep and image_exporter is not None:
        raise ValueError("Select a single mountain alpha before exporting images")
    was_training = model.training
    model.eval()
    states = {a: {"total": ConfusionMatrix(), "scenes": {}, "diagnostics": {
        "removed_true_mountain_pixels": 0, "removed_false_mountain_pixels": 0,
        "corrected_false_mountain_pixels": 0, "false_mountain_changed_to_another_wrong_class": 0,
        "changed_previously_non_mountain_pixels": 0,
        "small_gt_area_image_count": 0, "small_gt_area_gt_pixels": 0,
        "small_gt_area_true_positive_pixels": 0, "small_gt_area_lost_true_positive_pixels": 0}}
        for a in alphas}
    count, batches = 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = predict_semantic_scores(model, images, tta=tta)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits.float(), targets.shape[-2:], mode="bilinear", align_corners=False)
        if sweep:
            baseline = logits.argmax(1)
            gt_mountain = targets == 3
            areas = gt_mountain.flatten(1).sum(1)
            small_images = (areas > 0) & (areas < 0.01 * targets.shape[-2] * targets.shape[-1])
            small_pixels = gt_mountain & small_images[:, None, None]
        for a in alphas:
            prediction = adjust_mountain_scores(logits, a, tta).argmax(1)
            state = states[a]
            state["total"].update(targets, prediction)
            for i, scene in enumerate(batch["scene"]):
                entry = state["scenes"].setdefault(scene, {"matrix": ConfusionMatrix(), "image_count": 0})
                entry["matrix"].update(targets[i], prediction[i])
                entry["image_count"] += 1
            if sweep:
                removed = (baseline == 3) & (prediction != 3)
                removed_false = removed & ~gt_mountain
                changed_other = (baseline != 3) & (prediction != baseline)
                if torch.any(changed_other):
                    raise RuntimeError("Mountain suppression unexpectedly changed a non-mountain winner")
                counts = {
                    "removed_true_mountain_pixels": (removed & gt_mountain).sum(),
                    "removed_false_mountain_pixels": removed_false.sum(),
                    "corrected_false_mountain_pixels": (removed_false & (prediction == targets)).sum(),
                    "false_mountain_changed_to_another_wrong_class": (removed_false & (prediction != targets)).sum(),
                    "changed_previously_non_mountain_pixels": changed_other.sum(),
                    "small_gt_area_image_count": small_images.sum(),
                    "small_gt_area_gt_pixels": small_pixels.sum(),
                    "small_gt_area_true_positive_pixels": (small_pixels & (prediction == 3)).sum(),
                    "small_gt_area_lost_true_positive_pixels": (small_pixels & removed).sum()}
                values = torch.stack(list(counts.values())).cpu().tolist()
                for name, value in zip(counts, values):
                    state["diagnostics"][name] += int(value)
            if image_exporter is not None:
                image_exporter(batch, targets, prediction)
        count += len(images)
        batches += 1
        if sweep and batches % 100 == 0:
            print(f"Mountain score sweep: evaluated {count} images for {len(alphas)} alphas", flush=True)
    model.train(was_training)
    metrics = {a: _evaluation_result(v["total"], v["scenes"], loader, count, tta) for a, v in states.items()}
    if not sweep:
        return metrics[alpha]
    base = metrics[1.0]
    candidates = []
    for a in alphas:
        diag = states[a]["diagnostics"]
        diag["small_gt_area_recall"] = (diag["small_gt_area_true_positive_pixels"] / diag["small_gt_area_gt_pixels"]
                                        if diag["small_gt_area_gt_pixels"] else None)
        candidates.append({"alpha": a, "metrics": metrics[a], "diagnostics": diag,
                           "delta_miou": metrics[a]["miou"] - base["miou"],
                           "delta_mountain_iou": (metrics[a]["class_iou"]["mountain"] - base["class_iou"]["mountain"])
                           if base["class_iou"]["mountain"] is not None else None})
    valid_selection = not base["partial"] and count == 1599 and set(base["scenes"]) == {"Moon_5"}
    improving = [v for v in candidates if v["delta_miou"] > 1e-12
                 and v["delta_mountain_iou"] is not None and v["delta_mountain_iou"] > 1e-12]
    best_miou = max((v["metrics"]["miou"] for v in improving), default=base["miou"])
    recommended = max((v["alpha"] for v in improving if v["metrics"]["miou"] >= best_miou - 1e-12), default=1.0)
    result = dict(base)
    result["mountain_score_sweep"] = {
        "alphas": alphas, "class_id": 3, "candidates": candidates,
        "formula": "S_mountain *= alpha; four other final class scores unchanged; then argmax",
        "implementation": "After final resize: none adds log(alpha) to class-3 logits; hflip multiplies class-3 mean probability",
        "probability_threshold": False, "global_alpha_for_all_images": True,
        "loader_passes": 1, "model_forward_calls": batches * (2 if tta == "hflip" else 1),
        "selection_valid": valid_selection, "selection_scene": "Moon_5",
        "selection_rule": "Among alphas improving both whole-scene mIoU and mountain IoU, maximize mIoU; ties prefer larger alpha; otherwise 1",
        "recommended_alpha": recommended if valid_selection else None,
        "automatically_applied": False,
        "small_gt_area_definition": "Images with 0 < GT mountain pixels / image pixels < 1%; pixel-weighted recall; not physical mountain instances"}
    return result
