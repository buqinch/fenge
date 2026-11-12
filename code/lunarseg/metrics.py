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


def _validate_tta_mode(tta):
    if tta not in ("none", "hflip", "multiscale_hflip"):
        raise ValueError("tta must be none, hflip, or multiscale_hflip")


def validate_tta_sizes(tta="none", tta_sizes=None):
    """Validate explicit square inference scales without rounding their values."""
    _validate_tta_mode(tta)
    if tta != "multiscale_hflip":
        if tta_sizes is not None:
            raise ValueError("tta_sizes is only valid with multiscale_hflip")
        return None
    if not isinstance(tta_sizes, (list, tuple)) or len(tta_sizes) < 2:
        raise ValueError("multiscale_hflip requires at least two explicit integer tta_sizes")
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 32 for size in tta_sizes):
        raise ValueError("Each TTA size must be an integer >= 32; sizes are never rounded")
    sizes = list(tta_sizes)
    if len(set(sizes)) != len(sizes) or 1024 not in sizes:
        raise ValueError("TTA sizes must be unique and include the original size 1024")
    return sizes


def tta_metadata(tta="none", tta_sizes=None):
    sizes = validate_tta_sizes(tta, tta_sizes)
    if tta == "multiscale_hflip":
        return {"test_time_augmentation": True, "tta_mode": tta,
                "tta_input_sizes": sizes,
                "tta_views": [{"input_size": size, "transform": transform}
                              for size in sizes for transform in ("original", "horizontal_flip")],
                "tta_view_count": 2 * len(sizes),
                "tta_view_weights": [1.0 / (2 * len(sizes))] * (2 * len(sizes)),
                "tta_forward_calls_per_batch": 2 * len(sizes),
                "tta_score_domain": "probability",
                "tta_aggregation": "equal_weight_float32_probability_mean",
                "tta_input_resize": "bilinear_align_corners_false",
                "tta_input_antialias": True,
                "tta_padded_sizes": [((size + 31) // 32) * 32 for size in sizes],
                "tta_score_resize": "softmax_then_inverse_flip_then_bilinear_align_corners_false",
                "tta_output_coordinates": "original_input_image",
                "tta_padding": {"multiple": 32, "mode": "replicate", "sides": "right_bottom",
                                "semantic_logits_cropped_to_unpadded_size_before_softmax": True}}
    # Keep legacy metadata and its original view names unchanged.
    return {"test_time_augmentation": tta != "none", "tta_mode": tta,
            "tta_views": ["original", "horizontal_flip"] if tta == "hflip" else ["original"]}


@torch.inference_mode()
def predict_semantic_scores(model, images, tta="none", tta_sizes=None):
    """Return continuous class scores, without assigning any pixel IDs.

    ``none`` preserves the original logits and interpolation behavior exactly.
    ``hflip`` averages two float32 class probability maps in original image
    coordinates. Multiscale inference performs that pair at every exact
    requested size, reverses each flip, resizes continuous probabilities to
    the original input grid, then averages all views equally in float32.
    Softmax accepts both pixel logits and the region head's log probabilities.
    This helper never assigns pixel IDs and is never used for training losses.
    """
    sizes = validate_tta_sizes(tta, tta_sizes)  # Validate before any model call.
    if tta == "multiscale_hflip":
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError("Expected normalized Bx3xHxW RGB with H,W >= 32")
        original_size = images.shape[-2:]
        # The real network's inference entry point pads each exact scale to a
        # multiple of 32 and crops semantic logits back before returning them.
        forward = getattr(model, "forward_inference", model)
        accumulated = None
        for size in sizes:
            scaled = (images if tuple(original_size) == (size, size) else
                      F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False,
                                    antialias=True))
            for horizontal_flip in (False, True):
                view = scaled.flip(-1) if horizontal_flip else scaled
                probabilities = forward(view)["semantic_logits"].float().softmax(dim=1)
                if probabilities.ndim != 4 or probabilities.shape[:2] != (images.shape[0], 5):
                    raise ValueError("Expected Bx5xHxW semantic logits for every TTA view")
                if horizontal_flip:
                    probabilities = probabilities.flip(-1)
                if probabilities.shape[-2:] != original_size:
                    probabilities = F.interpolate(probabilities, size=original_size, mode="bilinear",
                                                  align_corners=False)
                if accumulated is None:
                    accumulated = probabilities
                else:
                    accumulated.add_(probabilities)
        return accumulated.div_(2 * len(sizes))
    # Preserve these legacy operations and their order exactly.
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
    _validate_tta_mode(tta)
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


def _evaluation_result(total, scenes, loader, count, tta, tta_sizes=None):
    result = total.compute()
    result.update(image_count=count, scenes={s: {**v["matrix"].compute(), "image_count": v["image_count"]}
                                          for s, v in sorted(scenes.items())},
                  input_size=loader.dataset.input_size,
                  partial=loader.dataset.partial,
                  data_identity=loader.dataset.identity,
                  **tta_metadata(tta, tta_sizes), continuous_scores_resized_before_argmax=True,
                  official_platform_score=False)
    if all(s in scenes for s in ("Moon_2", "Moon_4", "Moon_9")):
        miou = {s: result["scenes"][s]["miou"] for s in scenes}
        result["weighted_score"] = (miou["Moon_2"] + miou["Moon_4"] + 2 * miou["Moon_9"]) / 4
    return result


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp="bf16", tta="none", image_exporter=None,
                    mountain_alpha=1.0, mountain_alphas=None, tta_sizes=None, progress_every=0):
    sizes = validate_tta_sizes(tta, tta_sizes)
    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer")
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
            logits = predict_semantic_scores(model, images, tta=tta, tta_sizes=sizes)
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
        if progress_every and batches % progress_every == 0:
            print(f"Evaluation ({tta}): {count}/{len(loader.dataset)} images, {batches} batches", flush=True)
        elif sweep and not progress_every and batches % 100 == 0:
            print(f"Mountain score sweep: evaluated {count} images for {len(alphas)} alphas", flush=True)
    model.train(was_training)
    metrics = {a: _evaluation_result(v["total"], v["scenes"], loader, count, tta, sizes)
               for a, v in states.items()}
    forward_calls_per_batch = 2 * len(sizes) if sizes else 2 if tta == "hflip" else 1
    if tta == "multiscale_hflip":
        for result in metrics.values():
            result["model_forward_calls"] = batches * forward_calls_per_batch
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
        "implementation": "After final resize: none adds log(alpha) to class-3 logits; TTA multiplies class-3 mean probability",
        "probability_threshold": False, "global_alpha_for_all_images": True,
        "loader_passes": 1, "model_forward_calls": batches * forward_calls_per_batch,
        "selection_valid": valid_selection, "selection_scene": "Moon_5",
        "selection_rule": "Among alphas improving both whole-scene mIoU and mountain IoU, maximize mIoU; ties prefer larger alpha; otherwise 1",
        "recommended_alpha": recommended if valid_selection else None,
        "automatically_applied": False,
        "small_gt_area_definition": "Images with 0 < GT mountain pixels / image pixels < 1%; pixel-weighted recall; not physical mountain instances"}
    return result
