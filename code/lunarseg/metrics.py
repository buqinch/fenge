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


def validate_crater_gamma(value):
    if isinstance(value, bool):
        raise ValueError("crater_gamma must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("crater_gamma must be a finite positive number")
    return value


def adjust_crater_scores(scores, gamma=1.0, tta="none"):
    """Scale only final class-2 probability, after resizing and before argmax."""
    gamma = validate_crater_gamma(gamma)
    _validate_tta_mode(tta)
    if scores.ndim != 4 or scores.shape[1] != 5 or not scores.is_floating_point():
        raise ValueError("Expected floating Bx5xHxW final semantic scores")
    if gamma == 1.0:
        return scores
    adjusted = scores.float().clone()
    if tta == "none":
        adjusted[:, 2] += math.log(gamma)
    else:
        adjusted[:, 2] *= gamma
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


class _ScoreDistribution:
    """Bounded streaming distribution; no per-pixel arrays survive a batch."""

    def __init__(self, upper_bounds):
        self.upper_bounds = list(upper_bounds)
        self.bin_counts = [0] * (len(self.upper_bounds) + 1)
        self.count, self.total = 0, 0.0
        self.minimum, self.maximum = None, None

    def update(self, values):
        if not values.numel():
            return
        indices = torch.bucketize(values.contiguous(), values.new_tensor(self.upper_bounds), right=False)
        counts = torch.bincount(indices, minlength=len(self.bin_counts))
        summary = torch.cat((counts.double(), values.sum(dtype=torch.float64).reshape(1),
                             values.min().double().reshape(1), values.max().double().reshape(1))).cpu().tolist()
        for i in range(len(self.bin_counts)):
            self.bin_counts[i] += int(summary[i])
        self.count += values.numel()
        self.total += summary[-3]
        self.minimum = summary[-2] if self.minimum is None else min(self.minimum, summary[-2])
        self.maximum = summary[-1] if self.maximum is None else max(self.maximum, summary[-1])

    def compute(self):
        return {"count": self.count, "mean": self.total / self.count if self.count else None,
                "min": self.minimum, "max": self.maximum,
                "bin_upper_bounds": self.upper_bounds + [None], "bin_counts": self.bin_counts,
                "bin_rule": "(-infinity, first], (previous, next], ..., (last, +infinity); null is +infinity"}


class _CraterMissDiagnostics:
    def __init__(self, gammas, dataset):
        self.gammas = gammas
        self.groups = {}
        for name in ("missed_crater", "missed_crater_pred_regolith"):
            self.groups[name] = {"count": 0, "rank_counts": [0] * 5,
                                 "gap": _ScoreDistribution([0, .001, .003, .01, .02, .03, .05, .1, .2, .5, 1]),
                                 "ratio": _ScoreDistribution([1, 1.03, 1.06, 1.10, 1.2, 1.5, 2, 3, 5, 10, 100]),
                                 "threshold_counts": [0] * len(gammas)}
        self.examples = {"rank2_close": [], "rank2_far": [], "other_rank": []}
        self.example_limits = {"rank2_close": 12, "rank2_far": 12, "other_rank": 8}
        self.examples_per_source = {}
        self.records = {r["source_id"]: r for r in getattr(dataset, "records", []) if "source_id" in r}
        self.image_offset = 0

    @staticmethod
    def _batch_value(batch, key, index):
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and index < len(value) and isinstance(value[index], (str, int)):
            return str(value[index])
        return None

    def update(self, scores, target, baseline, batch, tta):
        probabilities = scores.float().softmax(1) if tta == "none" else scores.float()
        crater = probabilities[:, 2]
        winner = probabilities.gather(1, baseline[:, None]).squeeze(1)
        gap = winner - crater
        ratio = winner / crater.clamp_min(torch.finfo(probabilities.dtype).tiny)
        rank = 1 + (probabilities > crater[:, None]).sum(1)
        missed = (target == 2) & (baseline != 2)
        masks = {"missed_crater": missed, "missed_crater_pred_regolith": missed & (baseline == 0)}
        for name, mask in masks.items():
            entry = self.groups[name]
            selected_rank = rank[mask]
            entry["count"] += selected_rank.numel()
            counts = torch.cat((torch.bincount(selected_rank, minlength=6)[1:6],
                                torch.stack([(mask & (ratio <= g)).sum() for g in self.gammas]))).cpu().tolist()
            for i in range(5):
                entry["rank_counts"][i] += int(counts[i])
            for i in range(len(self.gammas)):
                entry["threshold_counts"][i] += int(counts[5 + i])
            entry["gap"].update(gap[mask])
            entry["ratio"].update(ratio[mask])
        self._collect_examples(missed, rank, ratio, gap, winner, crater, target, baseline, batch)
        self.image_offset += target.shape[0]

    def _collect_examples(self, missed, rank, ratio, gap, winner, crater, target, baseline, batch):
        if all(len(self.examples[k]) >= n for k, n in self.example_limits.items()):
            return
        for i in range(target.shape[0]):
            source_id = self._batch_value(batch, "source_id", i)
            source_key = source_id or f"image_{self.image_offset + i}"
            if self.examples_per_source.get(source_key, 0) >= 2:
                continue
            group_masks = {"rank2_close": missed[i] & (rank[i] == 2) & (ratio[i] <= max(self.gammas)),
                           "rank2_far": missed[i] & (rank[i] == 2) & (ratio[i] > max(self.gammas)),
                           "other_rank": missed[i] & (rank[i] != 2)}
            for name, mask in group_masks.items():
                if (len(self.examples[name]) >= self.example_limits[name]
                        or self.examples_per_source.get(source_key, 0) >= 2):
                    continue
                # The first row-major matching pixel gives stable bounded examples.
                flat_indices = mask.flatten().nonzero().flatten()
                if not flat_indices.numel():
                    continue
                flat = int(flat_indices[0])
                y, x = divmod(flat, target.shape[-1])
                pwin, pcrater, pgap, pratio = torch.stack((winner[i, y, x], crater[i, y, x],
                                                         gap[i, y, x], ratio[i, y, x])).cpu().tolist()
                record = self.records.get(source_id, {})
                example = {"group": name, "source_id": source_id,
                           "scene": self._batch_value(batch, "scene", i),
                           "sample_id": self._batch_value(batch, "sample_id", i),
                           "x": x, "y": y, "gt": int(target[i, y, x]), "pred": int(baseline[i, y, x]),
                           "crater_rank": int(rank[i, y, x]), "pwinner": pwin, "pcrater": pcrater,
                           "gap": pgap, "ratio": pratio,
                           "coordinate_frame": "final_target_grid", "height": target.shape[-2], "width": target.shape[-1]}
                for key in ("image", "mask"):
                    path = record.get(key) or self._batch_value(batch, key + "_path", i)
                    if isinstance(path, str):
                        example[key] = path
                self.examples[name].append(example)
                self.examples_per_source[source_key] = self.examples_per_source.get(source_key, 0) + 1

    def compute(self):
        result = {}
        for name, entry in self.groups.items():
            count = entry["count"]
            result[name] = {"pixel_count": count,
                            "crater_rank_counts": {str(i + 1): n for i, n in enumerate(entry["rank_counts"])},
                            "rank2_pixels": entry["rank_counts"][1],
                            "rank2_fraction": entry["rank_counts"][1] / count if count else None,
                            "winner_minus_crater_probability": entry["gap"].compute(),
                            "winner_to_crater_probability_ratio": entry["ratio"].compute(),
                            "ratio_threshold_counts": [{"gamma": g, "pixels": n,
                                                        "fraction": n / count if count else None}
                                                       for g, n in zip(self.gammas, entry["threshold_counts"])]}
        result.update(crater_rank_definition="1 + number of classes with strictly greater probability; ties may have rank 1 despite argmax choosing a lower class ID",
                      ratio_definition="pwinner / max(pcrater, float32 tiny); threshold is necessary but equality is subject to argmax tie breaking",
                      rank2_close_definition=f"Missed GT crater, crater rank 2, winner/crater probability ratio <= {max(self.gammas)}",
                      pixel_example_groups=self.examples,
                      pixel_example_count=sum(map(len, self.examples.values())),
                      pixel_example_policy="Deterministic first row-major match; at most 32 pixels, at most 2 per source image; illustrative, not a random population sample")
        return result


@torch.inference_mode()
def _evaluate_crater_sweep(model, loader, device, amp, tta, sizes, gammas, progress_every):
    identity = loader.dataset.identity
    if identity.get("split") != "val" or getattr(loader.dataset, "split", "val") != "val":
        raise ValueError("Crater gamma sweeps are allowed only on validation data, never on test")
    states = {g: {"total": ConfusionMatrix(), "scenes": {}, "transition": ConfusionMatrix(),
                  "counts": {k: 0 for k in ("corrected_missed_crater_pixels", "corrected_regolith_to_crater_pixels",
                             "new_false_crater_pixels", "new_regolith_to_crater_false_pixels",
                             "new_false_crater_on_gt_regolith_pixels", "lost_true_crater_pixels",
                             "removed_false_crater_pixels", "changed_between_non_crater_classes_pixels")}}
              for g in gammas}
    missed_diagnostics = _CraterMissDiagnostics(gammas, loader.dataset)
    seen_sources = set()
    count, batches = 0, 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with autocast_context(device, amp):
                scores = predict_semantic_scores(model, images, tta=tta, tta_sizes=sizes)
            if scores.shape[-2:] != target.shape[-2:]:
                scores = F.interpolate(scores.float(), target.shape[-2:], mode="bilinear", align_corners=False)
            baseline = scores.argmax(1)
            missed_diagnostics.update(scores, target, baseline, batch, tta)
            seen_sources.update(batch.get("source_id", []))
            for g, state in states.items():
                prediction = adjust_crater_scores(scores, g, tta).argmax(1)
                state["total"].update(target, prediction)
                state["transition"].update(baseline, prediction)
                for i, scene in enumerate(batch["scene"]):
                    entry = state["scenes"].setdefault(scene, {"matrix": ConfusionMatrix(), "image_count": 0})
                    entry["matrix"].update(target[i], prediction[i])
                    entry["image_count"] += 1
                added = (baseline != 2) & (prediction == 2)
                lost = (baseline == 2) & (prediction != 2)
                changed_other = (baseline != 2) & (prediction != 2) & (prediction != baseline)
                counts = {"corrected_missed_crater_pixels": (added & (target == 2)).sum(),
                          "corrected_regolith_to_crater_pixels": (added & (target == 2) & (baseline == 0)).sum(),
                          "new_false_crater_pixels": (added & (target != 2)).sum(),
                          "new_regolith_to_crater_false_pixels": (added & (target == 0) & (baseline == 0)).sum(),
                          "new_false_crater_on_gt_regolith_pixels": (added & (target == 0)).sum(),
                          "lost_true_crater_pixels": (lost & (target == 2)).sum(),
                          "removed_false_crater_pixels": (lost & (target != 2)).sum(),
                          "changed_between_non_crater_classes_pixels": changed_other.sum()}
                values = torch.stack(list(counts.values())).cpu().tolist()
                for key, value in zip(counts, values):
                    state["counts"][key] += int(value)
                if values[-1] or torch.any(lost):
                    raise RuntimeError("Crater boosting unexpectedly removed an existing crater winner or changed another class")
            count += images.shape[0]
            batches += 1
            if (progress_every and batches % progress_every == 0) or (not progress_every and batches % 100 == 0):
                print(f"Crater score sweep: {count}/{len(loader.dataset)} images, {len(gammas)} gammas", flush=True)
    finally:
        model.train(was_training)
    metrics = {g: _evaluation_result(s["total"], s["scenes"], loader, count, tta, sizes) for g, s in states.items()}
    base = metrics[1.0]
    base_cm = states[1.0]["total"].matrix
    baseline_tp, baseline_fp = int(base_cm[2, 2]), int(base_cm[:, 2].sum() - base_cm[2, 2])
    candidates = []
    for g, state in states.items():
        cm = state["total"].matrix
        tp, fp = int(cm[2, 2]), int(cm[:, 2].sum() - cm[2, 2])
        diag = {**state["counts"], "baseline_crater_true_positive_pixels": baseline_tp,
                "baseline_crater_false_positive_pixels": baseline_fp,
                "crater_true_positive_pixels": tp, "crater_false_positive_pixels": fp,
                "delta_crater_true_positive_pixels": tp - baseline_tp,
                "delta_crater_false_positive_pixels": fp - baseline_fp,
                "transition_matrix_baseline_rows_calibrated_cols": state["transition"].matrix.tolist()}
        candidates.append({"gamma": g, "metrics": metrics[g], "diagnostics": diag,
                           "delta_miou": metrics[g]["miou"] - base["miou"],
                           "delta_crater_iou": (metrics[g]["class_iou"]["crater"] - base["class_iou"]["crater"])
                           if base["class_iou"]["crater"] is not None else None})
    complete = (not base["partial"] and not identity.get("partial", False) and count == len(loader.dataset)
                and count == len(seen_sources) and identity.get("split") == "val")
    custom = bool(identity.get("custom_split"))
    valid_custom = (custom and count == 1601 and identity.get("selection_scope") == "custom_val_originals"
                    and all(identity.get(k) == count for k in ("image_count", "complete_image_count", "validation_image_count"))
                    and not set(base["scenes"]) & {"Moon_2", "Moon_4", "Moon_9"})
    valid_legacy = not custom and count == 1599 and set(base["scenes"]) == {"Moon_5"}
    valid_selection = complete and (valid_custom or valid_legacy)
    improving = [c for c in candidates if c["delta_miou"] > 1e-12 and c["delta_crater_iou"] is not None
                 and c["delta_crater_iou"] > 1e-12]
    best_miou = max((c["metrics"]["miou"] for c in improving), default=base["miou"])
    recommended = min((c["gamma"] for c in improving if c["metrics"]["miou"] >= best_miou - 1e-12), default=1.0)
    result = dict(base)
    forward_calls = batches * (2 * len(sizes) if sizes else 2 if tta == "hflip" else 1)
    result["crater_score_sweep"] = {
        "gammas": gammas, "class_id": 2, "candidates": candidates,
        "baseline_missed_crater_diagnostics": missed_diagnostics.compute(),
        "formula": "Final crater probability *= gamma; four other class scores unchanged; then argmax",
        "implementation": "After final resize: none adds log(gamma) to class-2 logits; TTA multiplies class-2 mean probability",
        "mountain_alpha": 1.0, "probability_threshold": False, "global_gamma_for_all_images": True,
        "loader_passes": 1, "model_forward_calls": forward_calls,
        "selection_valid": valid_selection, "selection_scene": "custom_validation" if custom else "Moon_5",
        "selection_uses_test": False, "validation_unique_source_count": len(seen_sources),
        "selection_rule": "Among gammas improving both aggregate validation mIoU and crater IoU, maximize mIoU; ties prefer smaller gamma; otherwise 1",
        "recommended_gamma": recommended if valid_selection else None,
        "automatically_applied": False,
        "diagnostic_definition": {"corrected_regolith_to_crater_pixels": "GT crater, baseline regolith, calibrated crater",
                                  "new_regolith_to_crater_false_pixels": "GT regolith, baseline regolith, calibrated crater",
                                  "new_false_crater_on_gt_regolith_pixels": "GT regolith, baseline any non-crater class, calibrated crater"}}
    return result


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp="bf16", tta="none", image_exporter=None,
                    mountain_alpha=1.0, mountain_alphas=None, tta_sizes=None, progress_every=0,
                    crater_gamma=1.0, crater_gammas=None):
    sizes = validate_tta_sizes(tta, tta_sizes)
    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer")
    alpha = validate_mountain_alpha(mountain_alpha)
    gamma = validate_crater_gamma(crater_gamma)
    if (crater_gammas is not None or gamma != 1.0) and (alpha != 1.0 or mountain_alphas is not None):
        raise ValueError("Crater calibration and mountain calibration cannot be combined")
    if crater_gammas is not None:
        if not isinstance(crater_gammas, (list, tuple)):
            raise ValueError("crater_gammas must be an explicit list of factors")
        gammas = [validate_crater_gamma(g) for g in crater_gammas]
        if not gammas or len(set(gammas)) != len(gammas) or 1.0 not in gammas or any(g < 1 for g in gammas):
            raise ValueError("Crater sweep needs unique gammas >= 1 including 1")
        if gamma != 1.0 or image_exporter is not None:
            raise ValueError("Select a single crater gamma before exporting images or setting a scalar gamma")
        return _evaluate_crater_sweep(model, loader, device, amp, tta, sizes, gammas, progress_every)
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
            calibrated = adjust_mountain_scores(logits, a, tta)
            if gamma != 1.0:
                calibrated = adjust_crater_scores(calibrated, gamma, tta)
            prediction = calibrated.argmax(1)
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
