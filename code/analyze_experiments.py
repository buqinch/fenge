#!/usr/bin/env python3
"""Validate completed LuSNAR comparisons and write an evidence-based report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("regolith", "rock", "crater", "mountain", "sky")
ZH = {"regolith": "月壤", "rock": "岩石", "crater": "撞击坑", "mountain": "山体", "sky": "天空"}
SCENES = {"Moon_2": 1597, "Moon_4": 1320, "Moon_9": 2087}
SELECTION_COUNTS = {"original": 5000, "crater": 1500, "mountain_rock": 1500}
STOPPED = {"complete", "early_stopped", "stopped"}


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value {value} in {path}")
    with Path(path).open() as stream:
        value = json.load(stream, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
            f"{label} disagrees with independently recomputed value: {actual!r} vs {expected!r}")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def scene_statistics(scene, label, image_count=None, require_all_classes=True):
    matrix = scene.get("confusion_matrix_gt_rows_pred_cols")
    require(isinstance(matrix, list) and len(matrix) == 5
            and all(isinstance(row, list) and len(row) == 5 for row in matrix),
            f"{label}: expected a 5x5 confusion matrix")
    require(all(type(value) is int and value >= 0 for row in matrix for value in row),
            f"{label}: confusion matrix must contain nonnegative integer counts")
    if image_count is not None:
        require(sum(map(sum, matrix)) == image_count * 1024 * 1024,
                f"{label}: confusion matrix is not the complete 1024x1024 labels for {image_count} images")
    gt = [sum(row) for row in matrix]
    require(any(value > 0 for value in gt), f"{label}: empty ground-truth confusion matrix")
    if require_all_classes:
        require(all(value > 0 for value in gt), f"{label}: these complete Moon_5/2/4/9 scenes must contain all five GT classes")
    predicted = [sum(matrix[r][c] for r in range(5)) for c in range(5)]
    derived = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        fp, fn = predicted[i] - tp, gt[i] - tp
        iou = ratio(tp, tp + fp + fn) if gt[i] else None
        derived[name] = {"tp": tp, "fp": fp, "fn": fn, "gt_pixels": gt[i],
                         "predicted_pixels": predicted[i], "iou": iou,
                         "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}
        if gt[i]:
            close(scene.get("class_iou", {}).get(name), iou, f"{label}/{name} IoU")
    miou = sum(value["iou"] for value in derived.values() if value["iou"] is not None) / sum(x > 0 for x in gt)
    close(scene.get("miou"), miou, f"{label} mIoU")
    errors = [{"gt_class": CLASSES[r], "predicted_class": CLASSES[c], "pixels": matrix[r][c],
               "fraction_of_gt_class": ratio(matrix[r][c], gt[r])}
              for r in range(5) for c in range(5) if r != c and matrix[r][c]]
    errors.sort(key=lambda row: (-row["pixels"], row["gt_class"], row["predicted_class"]))
    crater = derived["crater"]
    crater_flows = {
        "false_negative_to_regolith": {"pixels": matrix[2][0], "fraction_of_crater_gt": ratio(matrix[2][0], gt[2]),
                                        "fraction_of_crater_false_negatives": ratio(matrix[2][0], crater["fn"])},
        "false_negative_to_mountain": {"pixels": matrix[2][3], "fraction_of_crater_gt": ratio(matrix[2][3], gt[2]),
                                        "fraction_of_crater_false_negatives": ratio(matrix[2][3], crater["fn"])},
        "false_positive_from_regolith": {"pixels": matrix[0][2],
                                         "fraction_of_crater_false_positives": ratio(matrix[0][2], crater["fp"])},
    }
    return {"miou": miou, "classes": derived, "confusion_matrix_gt_rows_pred_cols": matrix,
            "top_errors": errors[:5], "crater_flows": crater_flows}


def validate_test(path, mode, run, checkpoint_hash, best_epoch, tta_sizes=None, scale_weight=None):
    value = read_json(path)
    label = str(path)
    require(value.get("image_count") == 5004 and value.get("partial") is False,
            f"{label}: final analysis requires all 5004 images with partial=false")
    require(value.get("checkpoint_partial_training") is False, f"{label}: checkpoint has partial or unknown training scope")
    require(value.get("split") == "test" and value.get("evaluation_only") is True,
            f"{label}: expected an explicit evaluation-only test result")
    require(value.get("input_size") == 1024, f"{label}: expected 1024 input")
    require(value.get("tta_mode") == mode and value.get("test_time_augmentation") is (mode != "none"),
            f"{label}: incorrect TTA mode")
    if mode == "multiscale_hflip" and scale_weight is not None:
        require(tta_sizes == [1024, 768], f"{label}: weighted TTA requires exactly 1024 and 768")
        validate_weighted_tta_metadata(value, scale_weight, label)
        batch_size = value.get("batch_size")
        require(type(batch_size) is int and batch_size > 0
                and value.get("model_forward_calls") == math.ceil(5004 / batch_size) * value["tta_view_count"],
                f"{label}: weighted test forward count differs from its active views")
    elif mode == "multiscale_hflip":
        require(tta_sizes in ([1024, 768], [1024, 768, 512]) and all(type(size) is int for size in tta_sizes),
                f"{label}: unexpected multiscale size combination")
        require(value.get("tta_input_sizes") == tta_sizes and value.get("tta_padded_sizes") == tta_sizes,
                f"{label}: multiscale content/padded sizes must match the requested exact input sizes")
        views = [{"input_size": size, "transform": transform}
                 for size in tta_sizes for transform in ("original", "horizontal_flip")]
        require(value.get("tta_views") == views and value.get("tta_view_count") == 2 * len(tta_sizes)
                and value.get("tta_forward_calls_per_batch") == 2 * len(tta_sizes)
                and value.get("tta_view_weights") == [1.0 / len(views)] * len(views),
                f"{label}: expected exactly original and horizontal-flip views at every scale")
        require(value.get("tta_score_domain") == "probability"
                and value.get("tta_aggregation") == "equal_weight_float32_probability_mean"
                and value.get("tta_input_resize") == "bilinear_align_corners_false"
                and value.get("tta_input_antialias") is True
                and value.get("tta_score_resize") == "softmax_then_inverse_flip_then_bilinear_align_corners_false"
                and value.get("tta_output_coordinates") == "original_input_image",
                f"{label}: multiscale scores must use equal view probabilities restored to the original coordinates")
        require(value.get("tta_padding") == {"multiple": 32, "mode": "replicate", "sides": "right_bottom",
                    "semantic_logits_cropped_to_unpadded_size_before_softmax": True},
                f"{label}: incorrect inference padding/crop protocol")
        batch_size = value.get("batch_size")
        require(type(batch_size) is int and batch_size > 0
                and value.get("model_forward_calls") == math.ceil(5004 / batch_size) * len(views),
                f"{label}: model forward count differs from the full dataset and view count")
    else:
        expected_views = ["original", "horizontal_flip"] if mode == "hflip" else ["original"]
        require(value.get("tta_views") == expected_views, f"{label}: incorrect TTA views")
    require(value.get("continuous_scores_resized_before_argmax") is True,
            f"{label}: missing continuous-score-before-argmax protocol")
    require(value.get("checkpoint_epoch") == best_epoch, f"{label}: evaluated checkpoint is not this run's validation-selected best epoch")
    require(value.get("checkpoint_sha256") == checkpoint_hash, f"{label}: evaluated checkpoint SHA256 differs from best.pt")
    checkpoint = Path(value.get("checkpoint", ""))
    require(checkpoint.is_absolute() and checkpoint.resolve() == (run / "best.pt").resolve(),
            f"{label}: checkpoint must identify this run's best.pt")
    require(set(value.get("scenes", {})) == set(SCENES), f"{label}: expected exactly Moon_2/4/9")
    scenes = {}
    summed = [[0] * 5 for _ in range(5)]
    for scene, count in SCENES.items():
        source = value["scenes"][scene]
        require(source.get("image_count") == count, f"{label}/{scene}: incomplete image count")
        scenes[scene] = scene_statistics(source, f"{label}/{scene}", count)
        matrix = scenes[scene]["confusion_matrix_gt_rows_pred_cols"]
        summed = [[summed[r][c] + matrix[r][c] for c in range(5)] for r in range(5)]
    require(value.get("confusion_matrix_gt_rows_pred_cols") == summed, f"{label}: total and scene confusion matrices disagree")
    weighted = (scenes["Moon_2"]["miou"] + scenes["Moon_4"]["miou"] + 2 * scenes["Moon_9"]["miou"]) / 4
    close(value.get("weighted_score"), weighted, f"{label} weighted S")
    return {"source": str(path), "source_sha256": digest(path), "tta_mode": mode,
            "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": best_epoch,
            "image_count": 5004, "weighted_score": weighted, "scenes": scenes}


def load_run(path, label, mode):
    run = Path(path).resolve()
    summary = read_json(run / "summary.json")
    config = read_json(run / "run_config.json")
    require(summary.get("status") in STOPPED, f"{run}: training has not stopped (status={summary.get('status')!r})")
    require(summary.get("partial") is False and config.get("partial") is False, f"{run}: partial training cannot be final analysis")
    require(summary.get("selection_scene") == "Moon_5" and summary.get("official_test_used_for_selection") is False,
            f"{run}: checkpoint selection must use only Moon_5")
    require(config.get("val_tta", "none") == mode and summary.get("val_tta", "none") == mode,
            f"{run}: validation TTA does not match the run's role")
    identity = config.get("data_identity", {})
    training = identity.get("train", {})
    validation = identity.get("val", {})
    require(training.get("image_count") == 8000 and training.get("complete_image_count") == 8000
            and training.get("partial") is False and training.get("selection_counts") == SELECTION_COUNTS,
            f"{run}: expected the fixed 5000 original + 1500 crater + 1500 mountain_rock selection")
    require(validation.get("image_count") == 1599 and validation.get("partial") is False,
            f"{run}: expected the full 1599-image Moon_5 validation")
    selection_path = Path(training.get("selection_manifest", ""))
    require(selection_path.is_file(), f"{run}: fixed selection manifest is unavailable: {selection_path}")
    require(digest(selection_path) == training.get("selection_manifest_sha256"), f"{run}: selection manifest SHA256 mismatch")
    selection = read_json(selection_path)
    require(selection.get("counts") == SELECTION_COUNTS and selection.get("image_count") == 8000
            and selection.get("split") == "train" and selection.get("validation_or_test_used") is False,
            f"{run}: invalid fixed selection provenance")
    require(config.get("input_size") == 1024 and config.get("backbone") == "dual"
            and config.get("decoder") == "region" and config.get("num_queries") == 32
            and config.get("decoder_layers") == 4,
            f"{run}: expected dual-CNN region decoder at 1024 with 32 queries/four layers")
    history_path = run / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    require(history and [row.get("epoch") for row in history] == list(range(1, len(history) + 1)),
            f"{run}: epoch history must be complete, unique, and ordered from one")
    require(summary.get("completed_epochs") == history[-1]["epoch"], f"{run}: summary/history completed epoch mismatch")
    best_row, best_score = None, -math.inf
    for row in history:
        val = row.get("validation", {})
        require(row.get("partial") is False and val.get("partial") is False and val.get("image_count") == 1599
                and set(val.get("scenes", {})) == {"Moon_5"}, f"{run}: epoch {row.get('epoch')} validation is incomplete")
        require(val.get("tta_mode", "none") == mode, f"{run}: validation TTA changed within history")
        score = scene_statistics(val["scenes"]["Moon_5"], f"{run}/epoch_{row['epoch']}/Moon_5", 1599)["miou"]
        require(isinstance(score, (int, float)) and math.isfinite(score), f"{run}: invalid validation score")
        require(math.isfinite(row.get("training_loss", {}).get("loss_total", math.nan)), f"{run}: invalid training loss")
        if score > best_score:
            best_row, best_score = row, score
    require(summary.get("best_epoch") == best_row["epoch"], f"{run}: best epoch is not the first strict maximum of Moon_5 history")
    close(summary.get("best_validation_miou"), best_score, f"{run} best validation mIoU")
    checkpoint_hash = digest(run / "best.pt")
    tests = {}
    for test_mode, name in (("none", "test_metrics.json"), ("hflip", "test_hflip_metrics.json")):
        test_path = run / name
        if test_path.is_file():
            tests[test_mode] = validate_test(test_path, test_mode, run, checkpoint_hash, best_row["epoch"])
        elif test_mode == mode:
            raise ValueError(f"{run}: required complete test result is not ready: {name}")
    stop = summary.get("stop_reason", summary.get("early_stopping", {}).get("reason"))
    return {"label": label, "run_dir": str(run), "val_tta": mode, "summary": summary,
            "source_files_sha256": {name: digest(run / name) for name in ("summary.json", "run_config.json", "history.jsonl")},
            "run_config": config, "history": history, "stop_reason": stop,
            "best_epoch": best_row["epoch"], "best_validation_miou": best_score,
            "completed_epochs": history[-1]["epoch"], "checkpoint_sha256": checkpoint_hash,
            "selection": {"path": str(selection_path), "sha256": training["selection_manifest_sha256"],
                          "counts": selection["counts"], "image_count": 8000,
                          "unique_source_count": selection.get("unique_source_count"), "seed": selection.get("seed")},
            "tests": tests}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_random_split_run(path, label="随机 4:1 新划分组"):
    """Validate the new run against its literal family split and aggregate val."""
    from collections import Counter

    run = Path(path).resolve()
    summary, config = read_json(run / "summary.json"), read_json(run / "run_config.json")
    require(summary.get("status") in STOPPED and summary.get("partial") is False
            and config.get("partial") is False, f"{run}: custom training must be stopped and complete")
    require(summary.get("selection_scene") == "custom_validation"
            and summary.get("checkpoint_selection_uses_test") is False
            and summary.get("official_test_used_for_selection") is True,
            f"{run}: distinguish test-informed recipe choice from validation-only checkpoint choice")
    mode = config.get("val_tta", "none")
    require(mode in ("none", "hflip") and summary.get("val_tta", "none") == mode,
            f"{run}: inconsistent custom validation TTA")
    identity = config.get("data_identity", {})
    training, validation = identity.get("train", {}), identity.get("val", {})
    for split, item, count in (("train", training, 9401), ("val", validation, 1601)):
        require(item.get("custom_split") is True and item.get("partial") is False
                and item.get("image_count") == count and item.get("complete_image_count") == count,
                f"{run}: expected the complete custom {split} split ({count} images)")
    selection_path = Path(training.get("manifest", "")).resolve()
    require(selection_path.is_file() and Path(validation.get("manifest", "")).resolve() == selection_path,
            f"{run}: train and val must identify the same saved custom split")
    selection_sha = digest(selection_path)
    require(training.get("manifest_sha256") == validation.get("manifest_sha256") == selection_sha,
            f"{run}: custom split manifest SHA256 mismatch")
    selection = read_json(selection_path)
    expected_counts = {"original_pool": 8002, "train_original": 6401, "val_original": 1601,
                       "train_augmented": 3000, "train_total": 9401, "val_total": 1601}
    require(selection.get("schema") == "lusnar_random_split_v1" and selection.get("class_names") == list(CLASSES)
            and selection.get("split_protocol") == "random_original_source_family_4_to_1"
            and selection.get("counts") == expected_counts
            and selection.get("augmentation_counts") == {"crater": 1500, "mountain_rock": 1500},
            f"{run}: invalid 6401/1601 + 3000 augmentation split provenance")
    splits, original_split = selection["splits"], selection["original_split"]
    require(set(splits) == {"train", "val"} and len(splits["train"]) == 9401 and len(splits["val"]) == 1601
            and selection.get("splits_sha256") == canonical_digest(splits)
            and selection.get("original_split_sha256") == canonical_digest(original_split)
            and selection.get("ordered_samples_sha256") == {split: canonical_digest(rows) for split, rows in splits.items()},
            f"{run}: custom literal list/count/hash mismatch")
    train_families, val_families = ({row["source_id"] for row in splits[split]} for split in ("train", "val"))
    require(len(train_families) == 6401 and len(val_families) == 1601 and not train_families & val_families
            and train_families == set(original_split["train_source_ids"])
            and val_families == set(original_split["val_source_ids"])
            and all(row["source_kind"] == "original" and row["split"] == "val" for row in splits["val"])
            and Counter(row.get("augmentation_group", "original") for row in splits["train"])
            == {"original": 6401, "crater": 1500, "mountain_rock": 1500},
            f"{run}: custom families overlap, originals are missing, or augmentations enter validation")
    allowed = {"Moon_1", "Moon_3", "Moon_5", "Moon_6", "Moon_7", "Moon_8"}
    require(all(row["scene"] in allowed for rows in splits.values() for row in rows),
            f"{run}: independent Moon_2/4/9 test scenes cannot enter custom train/val")
    for prefix in ("source", "augmented"):
        require(digest(Path(selection[prefix + "_manifest"])) == selection[prefix + "_manifest_sha256"],
                f"{run}: custom {prefix} source manifest SHA256 mismatch")
    scene_counts = dict(sorted(Counter(row["scene"] for row in splits["val"]).items()))
    require(config.get("input_size") == 1024 and config.get("backbone") == "dual"
            and config.get("decoder") == "region" and config.get("num_queries") == 32
            and config.get("decoder_layers") == 4, f"{run}: expected the dual-CNN region decoder at 1024")
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    require(history and [row.get("epoch") for row in history] == list(range(1, len(history) + 1))
            and summary.get("completed_epochs") == history[-1]["epoch"], f"{run}: incomplete custom epoch history")
    best_row, best_score = None, -math.inf
    for row in history:
        val = row.get("validation", {})
        require(row.get("partial") is False and val.get("partial") is False and val.get("image_count") == 1601
                and set(val.get("scenes", {})) == set(scene_counts) and val.get("tta_mode", "none") == mode,
                f"{run}: epoch {row.get('epoch')} does not use the full custom validation split")
        summed = [[0] * 5 for _ in range(5)]
        for scene, count in scene_counts.items():
            require(val["scenes"][scene].get("image_count") == count,
                    f"{run}/epoch_{row['epoch']}/{scene}: wrong custom validation scene count")
            values = scene_statistics(val["scenes"][scene], f"{run}/epoch_{row['epoch']}/{scene}", count,
                                      require_all_classes=False)
            matrix = values["confusion_matrix_gt_rows_pred_cols"]
            summed = [[summed[r][c] + matrix[r][c] for c in range(5)] for r in range(5)]
        require(val.get("confusion_matrix_gt_rows_pred_cols") == summed,
                f"{run}/epoch_{row['epoch']}: aggregate validation confusion matrix differs from its scenes")
        scene_statistics(val, f"{run}/epoch_{row['epoch']}/custom_validation", 1601,
                         require_all_classes=False)
        # Check the stored aggregate against the independent matrix calculation,
        # then reproduce training's exact strict comparison of validation.miou.
        score = val["miou"]
        require(math.isfinite(row.get("training_loss", {}).get("loss_total", math.nan)), f"{run}: non-finite training loss")
        if score > best_score:
            best_row, best_score = row, score
    require(summary.get("best_epoch") == best_row["epoch"],
            f"{run}: best epoch must be the first strict maximum of aggregate validation.miou")
    close(summary.get("best_validation_miou"), best_score, f"{run} best custom validation mIoU")
    checkpoint_hash = digest(run / "best.pt")
    tests = {test_mode: validate_test(run / filename, test_mode, run, checkpoint_hash, best_row["epoch"])
             for test_mode, filename in (("none", "test_metrics.json"), ("hflip", "test_hflip_metrics.json"))}
    reference_path = Path(config.get("design_selection_reference", ""))
    require(reference_path.is_file() and reference_path.is_absolute()
            and digest(reference_path) == config.get("design_selection_reference_sha256"),
            f"{run}: recipe-selection reference file or SHA256 is missing")
    design = read_json(reference_path)
    require(design == config.get("design_selection_reference_data")
            and design.get("schema") == "lusnar_test_selected_recipe_v1" and design.get("selected_using_test") is True,
            f"{run}: recipe-selection reference must explicitly record the prior test-informed choice")
    return {"label": label, "plot_label": "Random 4:1 split (new validation)", "run_dir": str(run),
            "val_tta": mode, "summary": summary, "run_config": config, "history": history,
            "source_files_sha256": {name: digest(run / name) for name in ("summary.json", "run_config.json", "history.jsonl")},
            "stop_reason": summary.get("stop_reason", summary.get("early_stopping", {}).get("reason")),
            "best_epoch": best_row["epoch"], "best_validation_miou": best_score,
            "completed_epochs": history[-1]["epoch"], "checkpoint_sha256": checkpoint_hash,
            "selection_scene": "custom_validation", "selection_metric": "validation.miou",
            "validation_scene_image_counts": scene_counts,
            "selection": {"path": str(selection_path), "sha256": selection_sha, "counts": expected_counts,
                          "image_count": 9401, "unique_source_count": 6401, "seed": selection.get("seed")},
            "design_selection_reference": {"path": str(reference_path), "sha256": digest(reference_path), "data": design},
            "tests": tests}


def validate_design_reference(new_run, reference, mode):
    design = new_run["design_selection_reference"]["data"]
    test = reference["tests"][mode]
    require(Path(design.get("selected_run", "")).resolve() == Path(reference["run_dir"])
            and Path(design.get("selected_checkpoint", "")).resolve() == Path(reference["run_dir"]) / "best.pt"
            and design.get("selected_checkpoint_sha256") == reference["checkpoint_sha256"]
            and design.get("selected_checkpoint_epoch") == reference["best_epoch"]
            and design.get("selected_test_tta") == mode,
            "Recipe selection must identify the requested reference run, checkpoint and test TTA")
    require(Path(design.get("selected_metrics_file", "")).resolve() == Path(test["source"])
            and design.get("selected_metrics_sha256") == test["source_sha256"], "Recipe selection test metrics SHA256 mismatch")
    close(design.get("selected_weighted_score"), test["weighted_score"], "Recipe-selection test weighted S")
    require(Path(design.get("new_split_manifest", "")).resolve() == Path(new_run["selection"]["path"])
            and design.get("new_split_manifest_sha256") == new_run["selection"]["sha256"],
            "Recipe selection references a different new split manifest")
    require(isinstance(design.get("training_recipe"), dict) and design["training_recipe"]
            and isinstance(design.get("candidate_results"), list) and design["candidate_results"],
            "Recipe selection must preserve the selected recipe and evaluated candidates")
    mode = new_run["run_config"].get("initialization_mode")
    require(mode in ("original_pretrained", "checkpoint_finetune", "random")
            and design.get("initialization_mode") == mode, "Recipe reference and actual initialization mode disagree")
    if mode == "checkpoint_finetune":
        config = new_run["run_config"]
        require(Path(config.get("initial_checkpoint", "")).resolve() == Path(reference["run_dir"]) / "best.pt"
                and config.get("initial_checkpoint_sha256") == reference["checkpoint_sha256"],
                "Checkpoint finetuning must record the selected checkpoint path and SHA256")
    elif mode == "original_pretrained":
        require(new_run["run_config"].get("pretrained") is True
                and not new_run["run_config"].get("initial_checkpoint")
                and not new_run["run_config"].get("convnext_checkpoint"),
                "Original pretrained initialization cannot also transfer a lunar checkpoint")


def same_tta_changes(reference, new_run):
    changes = {}
    for mode in ("none", "hflip"):
        old, new = reference["tests"][mode], new_run["tests"][mode]
        scenes = {}
        for scene in SCENES:
            a, b = old["scenes"][scene], new["scenes"][scene]
            require(all(a["classes"][name]["gt_pixels"] == b["classes"][name]["gt_pixels"] for name in CLASSES),
                    f"{scene}/{mode}: reference and new result disagree on ground-truth class counts")
            scenes[scene] = {"miou_change": b["miou"] - a["miou"],
                             "class_iou_changes": {name: b["classes"][name]["iou"] - a["classes"][name]["iou"] for name in CLASSES},
                             "confusion_pixel_changes_gt_rows_pred_cols": [[b["confusion_matrix_gt_rows_pred_cols"][r][c]
                                 - a["confusion_matrix_gt_rows_pred_cols"][r][c] for c in range(5)] for r in range(5)]}
        changes[mode] = {"weighted_score_change": new["weighted_score"] - old["weighted_score"], "scenes": scenes}
    return changes


def percent(value):
    return "—" if value is None else f"{value * 100:.4f}%"


def relative_link(path, output):
    return Path(os.path.relpath(Path(path).resolve(), output.resolve())).as_posix().replace(" ", "%20")


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_curves(runs, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "当前环境未安装 matplotlib；精确曲线数据已导出 history.csv。"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in runs:
        epochs = [row["epoch"] for row in run["history"]]
        label = run.get("plot_label", "val TTA off" if run["val_tta"] == "none" else "val TTA hflip")
        line = axes[0].plot(epochs, [row["training_loss"]["loss_total"] for row in run["history"]], label=label)[0]
        values = [(row["validation"]["miou"] if run.get("selection_scene") == "custom_validation"
                   else row["validation"]["scenes"]["Moon_5"]["miou"]) * 100 for row in run["history"]]
        axes[1].plot(epochs, values, label=label, color=line.get_color())
        axes[1].scatter([run["best_epoch"]], [run["best_validation_miou"] * 100], marker="*", s=100, color=line.get_color())
    for ax in axes:
        ax.set_xlabel("Completed epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Training total loss")
    axes[1].set_ylabel("Custom validation mIoU (%)" if any(run.get("selection_scene") == "custom_validation" for run in runs)
                       else "Full Moon_5 mIoU (%)")
    path = output / "training_validation_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path.name, None


def reference_result(path):
    if path is None:
        return None
    path = Path(path).resolve()
    value = read_json(path)
    require(value.get("image_count") == 5004 and set(value.get("scenes", {})) == set(SCENES),
            "Historical reference must contain all 5004 Moon_2/4/9 images")
    scenes = {name: scene_statistics(value["scenes"][name], f"reference/{name}", count) for name, count in SCENES.items()}
    score = (scenes["Moon_2"]["miou"] + scenes["Moon_4"]["miou"] + 2 * scenes["Moon_9"]["miou"]) / 4
    close(value.get("weighted_score"), score, "Historical reference S")
    return {"source": str(path), "source_sha256": digest(path), "weighted_score": score,
            "scenes": scenes, "name": path.parent.name}


def monitor_training_notes(run, output):
    """Describe actual completion and archived monitoring events without reselection."""
    lines = []
    completed, configured = run["completed_epochs"], run["run_config"].get("epochs")
    if run["summary"]["status"] == "complete":
        detail = f"实际跑满配置的 {configured} 轮" if completed == configured else f"实际完成 {completed} 轮"
        lines += ["", f"训练事实：`summary.status=complete`，{detail}；不能将恢复监控后回放历史得到的早停判定写成实际早停。"]
    run_path = Path(run["run_dir"]).resolve()
    for config_path in sorted((PROJECT / "config").glob("*.json")):
        try:
            config = read_json(config_path)
            matches = [row for row in config.get("runs", []) if isinstance(row, dict)
                       and isinstance(row.get("output"), str)
                       and (PROJECT / row["output"]).resolve() == run_path]
            if not matches or not isinstance(config.get("output"), str):
                continue
            state_path = (PROJECT / config["output"] / "state.json").resolve()
            state = read_json(state_path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        archived = state.get("error_history", [])
        if isinstance(archived, list):
            for event in archived:
                if isinstance(event, dict) and event.get("error"):
                    error = str(event["error"]).replace("`", "'")
                    lines += ["", f"监控曾中断：`{error}`；失败记录时间 `{event.get('failed_at', '未记录')}`，"
                              f"流程恢复时间 `{event.get('recovered_at', '未记录')}`。"]
        if state.get("error"):
            error = str(state["error"]).replace("`", "'")
            lines += ["", f"监控记录仍保留错误 `{error}`；此处不将其解释为训练已执行早停。"]
        lines += ["", f"监控依据：[状态及恢复归档]({relative_link(state_path, output)})；"
                  f"[对应配置]({relative_link(config_path, output)})。"]
        break
    tracker_path = run_path / "early_stopping_monitor.json"
    try:
        tracker = read_json(tracker_path).get("early_stopping", {})
    except (OSError, ValueError, TypeError):
        tracker = {}
    if (run["summary"]["status"] == "complete" and tracker.get("stopped") is True
            and type(tracker.get("stop_epoch")) is int):
        lines += ["", f"恢复后的历史回放显示，当前耐心规则会在已完成第 {tracker['stop_epoch']} 轮时触发；"
                  f"这是回放结论，实际训练仍完成 {completed} 轮，最佳检查点继续按验证历史选出的第 {run['best_epoch']} 轮。"
                  f"[回放状态]({relative_link(tracker_path, output)})。"]
    return lines


def random_split_report(run, reference, reference_tta, changes, output, curves, curve_note):
    design = run["design_selection_reference"]["data"]
    lines = ["# 双 CNN：随机 4:1 划分后的完整测试对照", "",
             "新实验从旧训练原图和 Moon_5 原图组成的 8002 张池中按来源随机划分："
             "6401 张训练原图、1601 张验证原图；另加仅来自新训练来源的 1500 张撞击坑增强和 1500 张山岩增强，训练共 9401 张。"
             "验证集不含增强，同一原图来源及其增强不会跨训练/验证集。Moon_2/4/9 未进入训练或验证。", "",
             "**新旧验证口径不同，验证 mIoU 不能直接比较。** 旧参考用完整 Moon_5 的 1599 张选轮；"
             "新实验仅按 1601 张多场景验证集汇总混淆矩阵的 `validation.miou` 选第一个严格最高轮次，未按测试成绩选轮。", "",
             "**测试成绩已经用于选择当前训练方案。** 下述 Moon_2/4/9 对照属于被查看并用于方案选择后的测试集，"
             "不能当作从未查看的留出集，也不能据一次结果证明划分或损失的因果收益。", "",
             "## 实验与来源", "",
             f"新划分：[保存的完整清单]({relative_link(run['selection']['path'], output)})；SHA256 `{run['selection']['sha256']}`；"
             f"划分种子 {run['selection']['seed']}。", "",
             f"方案选择记录：[JSON]({relative_link(run['design_selection_reference']['path'], output)})；"
             f"SHA256 `{run['design_selection_reference']['sha256']}`。初始化模式为 `{design['initialization_mode']}`。", "",
             "|实验|训练图|验证口径|验证视角|完成轮次|最佳轮次|各自验证最佳 mIoU|", "|---|---:|---|---|---:|---:|---:|",
             f"|旧参考 {Path(reference['run_dir']).name}|8000|完整 Moon_5，1599 张|{reference['val_tta']}|"
             f"{reference['completed_epochs']}|{reference['best_epoch']}|{percent(reference['best_validation_miou'])}|",
             f"|新随机划分 {Path(run['run_dir']).name}|9401|多场景原图，1601 张|{run['val_tta']}|"
             f"{run['completed_epochs']}|{run['best_epoch']}|{percent(run['best_validation_miou'])}|", "",
             "上表两种验证分数仅分别证明各自的检查点选择，不作差值或排名。新验证集各场景张数："
             + "、".join(f"{scene} {count}" for scene, count in run["validation_scene_image_counts"].items()) + "。", "",
             f"新实验停止状态 `{run['summary']['status']}`，停止依据 `{run['stop_reason'] or '见summary'}`；"
             f"[summary]({relative_link(Path(run['run_dir'])/'summary.json', output)})、"
             f"[history]({relative_link(Path(run['run_dir'])/'history.jsonl', output)})、"
             f"[配置]({relative_link(Path(run['run_dir'])/'run_config.json', output)})。", "",
             "|设置|旧参考|新随机划分|", "|---|---|---|"]
    for title, field in (("输入尺寸", "input_size"), ("batch", "batch_size"), ("梯度累积", "accumulation"),
                         ("冻结骨干轮数", "freeze_epochs"), ("骨干学习率", "backbone_lr"),
                         ("解码器学习率", "head_lr"), ("困难像素权重", "hard_pixel_weight"),
                         ("每图每方向配额", "hard_pixel_quota"), ("训练种子", "seed")):
        lines.append(f"|{title}|{reference['run_config'].get(field)}|{run['run_config'].get(field)}|")
    lines += monitor_training_notes(run, output)
    lines += ["", "## 相同 TTA 的完整 Moon_2/4/9 测试", "",
              "两模型均使用各自已选定的同一个 best 检查点做 none 和 hflip 推理。"
              "每种视角均完整评估 Moon_2/4/9 的 1597/1320/2087 张，共 5004 张。"
              "指标由逐场景整数混淆矩阵独立重算：`S=(Moon_2 mIoU + Moon_4 mIoU + 2×Moon_9 mIoU)/4`。", "",
              "|测试视角|模型|Moon_2 mIoU|Moon_4 mIoU|Moon_9 mIoU|加权 S|", "|---|---|---:|---:|---:|---:|"]
    for mode in ("none", "hflip"):
        for label, item in (("旧参考", reference), ("新随机划分", run)):
            test = item["tests"][mode]
            lines.append(f"|{mode}|{label}|" + "|".join(percent(test["scenes"][scene]["miou"]) for scene in SCENES)
                         + f"|{percent(test['weighted_score'])}|")
        lines.append(f"|{mode}|新减旧（百分点）|" + "|".join(f"{changes[mode]['scenes'][scene]['miou_change'] * 100:+.4f}" for scene in SCENES)
                     + f"|{changes[mode]['weighted_score_change'] * 100:+.4f}|")
    lines += ["", f"旧方案选择时使用的测试视角为 `{reference_tta}`；该相同视角下新减旧的加权 S 为 "
              f"**{changes[reference_tta]['weighted_score_change'] * 100:+.4f} 个百分点**。"
              "同时列出另一种相同视角对照，不用新旧不同 TTA 的成绩混算收益。", "",
              "## 逐场景、五类和混淆变化", "",
              "下表均为新减旧 IoU 的百分点变化；完整绝对 IoU、精确率、召回率见 metrics.csv。", "",
              "|TTA|场景|mIoU|月壤|岩石|撞击坑|山体|天空|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode, change in changes.items():
        for scene, item in change["scenes"].items():
            values = [item["miou_change"], *[item["class_iou_changes"][name] for name in CLASSES]]
            lines.append(f"|{mode}|{scene}|" + "|".join(f"{value * 100:+.4f}" for value in values) + "|")
    lines += ["", "主要关注的六个错分方向按真值→预测统计；负变化表示该方向错分像素减少。", "",
              "|TTA|场景|方向|旧错分像素|新错分像素|新减旧像素|", "|---|---|---|---:|---:|---:|"]
    for mode in ("none", "hflip"):
        for scene in SCENES:
            for truth, predicted in ((0, 3), (3, 0), (2, 0), (0, 2), (3, 1), (1, 3)):
                old = reference["tests"][mode]["scenes"][scene]["confusion_matrix_gt_rows_pred_cols"][truth][predicted]
                new = run["tests"][mode]["scenes"][scene]["confusion_matrix_gt_rows_pred_cols"][truth][predicted]
                lines.append(f"|{mode}|{scene}|{ZH[CLASSES[truth]]} → {ZH[CLASSES[predicted]]}|{old:,}|{new:,}|{new-old:+,}|")
    for item in (reference, run):
        gain = item["tests"]["hflip"]["weighted_score"] - item["tests"]["none"]["weighted_score"]
        lines += ["", f"同一 `{Path(item['run_dir']).name}` 检查点中，hflip 相对 none 的 S 变化为 {gain * 100:+.4f} 个百分点；"
                  "这仅是该检查点的推理视角对照。"]
    lines += ["", "## 新实验训练与验证曲线", "",
              "曲线只展示新实验，避免将两种验证口径画成可直接比较的指标。训练损失和验证 mIoU 量纲不同。", ""]
    lines += [f"![新实验训练与验证曲线]({curves})"] if curves else [curve_note]
    lines += ["", "|轮次|训练总损失|完整新验证 mIoU|骨干冻结|", "|---|---:|---:|---|"]
    for row in run["history"]:
        lines.append(f"|{row['epoch']}|{row['training_loss']['loss_total']:.6f}|{percent(row['validation']['miou'])}|{row.get('backbones_frozen')}|")
    lines += ["", "## 复核与解释范围", "",
              "- [逐类别绝对指标](metrics.csv)", "- [完整混淆矩阵](confusion_matrices.csv)",
              "- [新实验逐轮数据](history.csv)", "- [机器可读分析与来源 SHA256](summary.json)", "",
              "旧参考和新实验的训练样本数量、验证组成及可能的训练轨迹均不同；本次没有多种子重复或显著性检验。"
              "Moon_2/4/9 已用于方案选择，后续对比不再具有从未查看的留出评估含义；本地测试成绩也不等同于官方平台成绩。", ""]
    return "\n".join(lines)


def report(runs, output, curves, curve_note, reference, comparison="tta"):
    left, right = runs
    a, b = left["tests"][left["val_tta"]], right["tests"][right["val_tta"]]
    lines = ["# 双 CNN 区域分割：完整测试分析", "",
             "本报告只使用已停止训练后的 Moon_5 最佳检查点及完整 Moon_2/4/9 测试结果。"
             "所有场景 IoU、精确率、召回率和加权分数均由保存的整数混淆矩阵独立重算并核对。", "",
             "## 数据与训练设置", "",
             "两组固定使用同一份 8000 张训练清单：原图 5000、大坑增强 1500、山岩交界增强 1500；"
             f"共 {left['selection']['unique_source_count']} 个来源原图。增强版本与相邻帧存在相关性，8000 张不等于 8000 个独立场景。"
             "完整 Moon_5 共 1599 张用于验证和选轮；Moon_2/4/9 共 5004 张只用于最终评估。", "",
             f"训练清单：[固定选择 JSON]({relative_link(left['selection']['path'], output)})；"
             f"SHA256：`{left['selection']['sha256']}`。", "",
             f"|设置|{left['label']}|{right['label']}|", "|---|---|---|"]
    fields = [("输入尺寸", "input_size"), ("训练 batch", "batch_size"), ("梯度累积", "accumulation"),
              ("冻结骨干轮数", "freeze_epochs"), ("骨干学习率", "backbone_lr"), ("融合/解码器学习率", "head_lr"),
              ("权重衰减", "weight_decay"), ("AMP", "amp"), ("随机种子", "seed"), ("训练随机水平翻转", "horizontal_flip")]
    if comparison == "hard-pixels":
        fields += [("困难像素权重 λ", "hard_pixel_weight"), ("每图每方向最多选点", "hard_pixel_quota")]
    for title, field in fields:
        lines.append(f"|{title}|{left['run_config'].get(field)}|{right['run_config'].get(field)}|")
    lines += ["", "两组架构均为 DINOv3 ConvNeXt-Tiny + MobileNetV3-Large 双 CNN，"
              "四尺度融合及 128 通道 FPN，32 个区域查询、4 层注意力区域解码器。TTA 只用于验证和测试推理，不参与训练损失。", "",
              "## 停止与检查点选择", "",
              "|组|停止状态|完成轮次|最佳轮次|完整 Moon_5 最佳 mIoU|选轮验证视角|", "|---|---|---:|---:|---:|---|"]
    for run in runs:
        lines.append(f"|{run['label']}|{run['summary']['status']}|{run['completed_epochs']}|{run['best_epoch']}|"
                     f"{percent(run['best_validation_miou'])}|{run['val_tta']}|")
    for run in runs:
        lines += ["", f"{run['label']}：停止依据为 `{run['stop_reason'] or '见 summary 状态及监控记录'}`；"
                  f"[summary]({relative_link(Path(run['run_dir'])/'summary.json', output)})、"
                  f"[history]({relative_link(Path(run['run_dir'])/'history.jsonl', output)})、"
                  f"[完整配置]({relative_link(Path(run['run_dir'])/'run_config.json', output)})。"
                  f"最佳检查点 SHA256：`{run['checkpoint_sha256']}`。"]
    lines += ["", "## 完整测试成绩", "",
              "Moon_2/4/9 分别为 1597/1320/2087 张。先汇总每个场景的像素混淆矩阵，"
              "再对该场景真值出现的类别 IoU 求均值；不是逐图 mIoU 平均。"
              "`S=(Moon_2 mIoU + Moon_4 mIoU + 2×Moon_9 mIoU)/4`，与全场景合并混淆矩阵的 mIoU 口径不同。", "",
              "|训练组|测试视角|Moon_2 mIoU|Moon_4 mIoU|Moon_9 mIoU|加权 S|", "|---|---|---:|---:|---:|---:|"]
    for run in runs:
        for mode, result in run["tests"].items():
            lines.append(f"|{run['label']}|{mode}|" + "|".join(percent(result["scenes"][name]["miou"]) for name in SCENES)
                         + f"|{percent(result['weighted_score'])}|")
    delta = b["weighted_score"] - a["weighted_score"]
    checkpoint_note = "不同" if left["checkpoint_sha256"] != right["checkpoint_sha256"] else "相同"
    comparison_note = ("两组验证和测试均使用相同水平翻转 TTA，从同一原始预训练初始化重新训练。"
                       "新增困难像素辅助损失是预设训练变量；这是一次固定种子的对照，不代表跨种子普适提升。"
                       if comparison == "hard-pixels" else
                       "这一比较同时包含训练轨迹、检查点选择和测试视角的差异，不能把全部变化归因于 TTA。")
    lines += ["", f"主要比较（{left['label']}/{left['val_tta']} → {right['label']}/{right['val_tta']}）：加权 S 变化 **{delta * 100:+.4f} 个百分点**。"
              f"两组采用{checkpoint_note}的最佳检查点，并分别按各自验证协议选轮。"
              + comparison_note, "",
              "|场景|mIoU 变化/百分点|月壤 IoU 变化|岩石 IoU 变化|撞击坑 IoU 变化|山体 IoU 变化|天空 IoU 变化|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in SCENES:
        values = [b["scenes"][name]["miou"] - a["scenes"][name]["miou"]]
        values += [b["scenes"][name]["classes"][c]["iou"] - a["scenes"][name]["classes"][c]["iou"] for c in CLASSES]
        lines.append(f"|{name}|" + "|".join(f"{v * 100:+.4f}" for v in values) + "|")
    contribution = {name: sum((b["scenes"][scene]["classes"][name]["iou"] - a["scenes"][scene]["classes"][name]["iou"])
                              * (2 if scene == "Moon_9" else 1) for scene in SCENES) / 20 for name in CLASSES}
    close(sum(contribution.values()), delta, "Weighted score class contribution decomposition")
    lines += ["", "按评分公式将 S 的差值分解到五类："
              + "、".join(f"{ZH[name]} {value * 100:+.4f} 个百分点" for name, value in sorted(contribution.items(), key=lambda item: -item[1]))
              + "。这是分数的算术贡献分解，不是网络模块的因果贡献。"]
    for run in runs:
        primary = run["tests"][run["val_tta"]]
        weakest_scene = min(primary["scenes"], key=lambda name: primary["scenes"][name]["miou"])
        weakest = min(((scene, name, item["iou"]) for scene, value in primary["scenes"].items()
                       for name, item in value["classes"].items()), key=lambda item: item[2])
        lines += ["", f"{run['label']}：最低场景 mIoU 为 {weakest_scene}（{percent(primary['scenes'][weakest_scene]['miou'])}）；"
                  f"最低场景/类别 IoU 为 {weakest[0]} 的{ZH[weakest[1]]}（{percent(weakest[2])}）。"]
    for run in runs:
        if set(run["tests"]) == {"none", "hflip"}:
            gain = run["tests"]["hflip"]["weighted_score"] - run["tests"]["none"]["weighted_score"]
            lines += ["", f"同一 `{run['label']}` 检查点的纯推理对照：hflip 相比 none 的 S 变化为 {gain * 100:+.4f} 个百分点。"
                      "这能隔离该检查点上的推理视角变化，仍不代表跨随机种子的一般结论。"]
    lines += ["", "## 五类 IoU、精确率与召回率", "",
              "精确率 `TP/(TP+FP)`；召回率 `TP/(TP+FN)`。下表与 CSV 保留逐场景、逐类别结果。", "",
              "|训练组/测试视角|场景|类别|IoU|精确率|召回率|", "|---|---|---|---:|---:|---:|"]
    for run in runs:
        for mode, result in run["tests"].items():
            for scene, values in result["scenes"].items():
                for name, item in values["classes"].items():
                    lines.append(f"|{run['label']}/{mode}|{scene}|{ZH[name]}|{percent(item['iou'])}|"
                                 f"{percent(item['precision'])}|{percent(item['recall'])}|")
    lines += ["", "## 撞击坑与主要混淆", ""]
    for run in runs:
        result = run["tests"][run["val_tta"]]
        lines += [f"### {run['label']} / {run['val_tta']}", ""]
        for name, scene in result["scenes"].items():
            crater = scene["classes"]["crater"]
            flows = scene["crater_flows"]
            soil, mountain, fp_soil = (flows[key] for key in ("false_negative_to_regolith", "false_negative_to_mountain", "false_positive_from_regolith"))
            lines += [f"{name}：撞击坑漏检 {crater['fn']:,} 像素、误检 {crater['fp']:,} 像素。"
                      f"漏检中 {percent(soil['fraction_of_crater_false_negatives'])} 流向月壤（占全部坑真值 {percent(soil['fraction_of_crater_gt'])}），"
                      f"{percent(mountain['fraction_of_crater_false_negatives'])} 流向山体；"
                      f"坑误检中 {percent(fp_soil['fraction_of_crater_false_positives'])} 来自月壤。", "",
                      "|主要错分：真值 → 预测|像素数|占该真值类别|", "|---|---:|---:|"]
            for error in scene["top_errors"]:
                lines.append(f"|{ZH[error['gt_class']]} → {ZH[error['predicted_class']]}|{error['pixels']:,}|{percent(error['fraction_of_gt_class'])}|")
            lines.append("")
    lines += ["上述矩阵能说明错分去向和数量，不能单凭这些数值证明标签错误、样本不足或某个模块的因果机制。", "",
              "## 训练与验证曲线", ""]
    if curves:
        lines += [f"![训练损失与完整 Moon_5 验证曲线]({curves})", "", "星号表示各组按 Moon_5 选中的最佳轮次。训练损失与验证 mIoU 是不同量纲，不能直接相减推断过拟合。"]
        if comparison == "hard-pixels":
            lines += ["困难像素组的总损失额外包含辅助项，两组总损失值不能直接据大小比较优劣；主要比较完整 Moon_5 与测试指标。"
                      "每轮训练点选取、边界比例和高损失点证据保存于该组 hard_pixel_diagnostics_epoch_NNN.json，边界集中或高损失不自动等于错标。"]
    else:
        lines.append(curve_note)
    lines += ["", "|组|轮次|训练总损失|Moon_5 mIoU|骨干冻结|", "|---|---:|---:|---:|---|"]
    for run in runs:
        for row in run["history"]:
            lines.append(f"|{run['label']}|{row['epoch']}|{row['training_loss']['loss_total']:.6f}|"
                         f"{percent(row['validation']['scenes']['Moon_5']['miou'])}|{row.get('backbones_frozen')}|")
    if reference:
        lines += ["", "## 历史协议参考", "",
                  f"历史 [{reference['name']}]({relative_link(reference['source'], output)}) 的完整测试加权 S 为 **{percent(reference['weighted_score'])}**。"
                  "该结果属于先前模型和训练/输入协议；例如 gt_1_epoch8 使用 ConvNeXt-Tiny + ViT-S 及关系修正头、768 输入、2000 张大坑组微调。"
                  "它可作为历史成绩参考，不能作为当前双 CNN、1024 输入、8000 训练图的单因素消融。"]
    lines += ["", "## 复核文件与解释范围", "",
              "- [逐类别精确指标 CSV](metrics.csv)", "- [完整混淆矩阵 CSV](confusion_matrices.csv)",
              "- [逐轮曲线数据 CSV](history.csv)", "- [机器可读分析与来源 SHA256](summary.json)", "",
              "本次为各协议的一次训练结果，没有多随机种子重复，也没有统计显著性检验。"
              "本地完整测试标签结果不等同于官方平台成绩。测试诊断用于解释本次表现，后续模块和超参数选择仍须回到训练集与 Moon_5。", ""]
    return "\n".join(lines)


def calibration_validation_statistics(value, identity, scene_counts, label, scale_weight=None):
    """Recompute a complete custom validation result without reading test data."""
    require(value.get("partial") is False and value.get("image_count") == 1601
            and value.get("input_size") == 1024 and value.get("data_identity") == identity
            and set(value.get("scenes", {})) == set(scene_counts),
            f"{label}: expected the unchanged full 1601-image custom validation identity")
    if scale_weight is None:
        require(value.get("tta_mode") == "hflip" and value.get("test_time_augmentation") is True
                and value.get("tta_views") == ["original", "horizontal_flip"]
                and value.get("continuous_scores_resized_before_argmax") is True,
                f"{label}: expected 1024 horizontal-flip continuous-score inference")
    else:
        validate_weighted_tta_metadata(value, scale_weight, label)
    scenes, summed = {}, [[0] * 5 for _ in range(5)]
    for name, count in scene_counts.items():
        source = value["scenes"][name]
        require(source.get("image_count") == count, f"{label}/{name}: incomplete validation scene")
        scenes[name] = scene_statistics(source, f"{label}/{name}", count, require_all_classes=False)
        matrix = scenes[name]["confusion_matrix_gt_rows_pred_cols"]
        summed = [[summed[r][c] + matrix[r][c] for c in range(5)] for r in range(5)]
    require(value.get("confusion_matrix_gt_rows_pred_cols") == summed,
            f"{label}: aggregate validation matrix differs from its scenes")
    pooled = scene_statistics(value, f"{label}/custom_validation", 1601, require_all_classes=False)
    for scene_name, statistics in {"custom_validation": pooled, **scenes}.items():
        source = value if scene_name == "custom_validation" else value["scenes"][scene_name]
        for name, item in statistics["classes"].items():
            for field in ("precision", "recall"):
                actual = source.get("class_" + field, {}).get(name)
                expected = item[field]
                if expected is None:
                    require(actual is None, f"{label}/{scene_name}/{name}: undefined {field} must be null")
                else:
                    close(actual, expected, f"{label}/{scene_name}/{name} {field}")
    return {"pooled": pooled, "scenes": scenes}


def validate_weighted_tta_metadata(value, weight, label):
    require(not isinstance(weight, bool) and weight in (1.0, 0.75, 0.5), f"{label}: unexpected 1024 scale weight")
    sizes, weights = [1024, 768], [weight, 1 - weight]
    requested = [{"input_size": size, "transform": transform}
                 for size in sizes for transform in ("original", "horizontal_flip")]
    requested_weights = [scale / 2 for scale in weights for _ in range(2)]
    views = [view for view, weight in zip(requested, requested_weights) if weight > 0]
    active_weights = [weight for weight in requested_weights if weight > 0]
    require(value.get("tta_mode") == "multiscale_hflip" and value.get("test_time_augmentation") is True
            and value.get("tta_input_sizes") == sizes and value.get("tta_padded_sizes") == sizes
            and value.get("tta_scale_weights") == weights
            and value.get("tta_active_input_sizes") == [size for size, weight in zip(sizes, weights) if weight > 0]
            and value.get("tta_requested_views") == requested and value.get("tta_requested_view_weights") == requested_weights
            and value.get("tta_requested_view_count") == 4 and value.get("tta_views") == views
            and value.get("tta_view_weights") == active_weights and value.get("tta_view_count") == len(views)
            and value.get("tta_forward_calls_per_batch") == len(views),
            f"{label}: scale weights, requested views and active horizontal-flip pairs disagree")
    require(value.get("tta_score_domain") == "probability"
            and value.get("tta_aggregation") == "scale_weighted_float32_probability_mean"
            and value.get("tta_input_resize") == "bilinear_align_corners_false"
            and value.get("tta_input_antialias") is True
            and value.get("tta_score_resize") == "softmax_then_inverse_flip_then_bilinear_align_corners_false"
            and value.get("tta_output_coordinates") == "original_input_image"
            and value.get("continuous_scores_resized_before_argmax") is True
            and value.get("tta_padding") == {"multiple": 32, "mode": "replicate", "sides": "right_bottom",
                "semantic_logits_cropped_to_unpadded_size_before_softmax": True},
            f"{label}: weighted TTA must combine continuous probabilities in the original coordinates")


def weighted_validation_reference(run_path):
    """Load the saved validation reference without opening any test result."""
    from collections import Counter

    directory = Path(run_path).resolve()
    summary, config = read_json(directory / "summary.json"), read_json(directory / "run_config.json")
    epoch = summary.get("best_epoch")
    require(summary.get("status") in STOPPED and summary.get("partial") is False
            and config.get("partial") is False and type(epoch) is int and epoch > 0
            and summary.get("selection_scene") == "custom_validation"
            and summary.get("checkpoint_selection_uses_test") is False
            and summary.get("val_tta") == config.get("val_tta") == "hflip" and config.get("input_size") == 1024,
            "Weighted TTA needs a completed 1024/hflip custom-validation-selected checkpoint")
    saved_path = directory / f"validation_epoch_{epoch:03d}.json"
    saved = read_json(saved_path)
    value = saved.get("validation", {});identity = value.get("data_identity", {})
    require(saved.get("epoch") == epoch and saved.get("partial") is False
            and saved.get("selection_scene") == "custom_validation"
            and saved.get("checkpoint_selection_uses_test") is False
            and identity == config.get("data_identity", {}).get("val")
            and identity.get("custom_split") is True and identity.get("split") == "val"
            and identity.get("protocol") == "random_original_source_family_4_to_1"
            and identity.get("partial") is False
            and identity.get("image_count") == identity.get("complete_image_count") == identity.get("validation_image_count") == 1601,
            "Saved validation must identify the same complete 1601 original sources")
    manifest = Path(identity.get("manifest", ""))
    require(manifest.is_absolute() and manifest.is_file() and digest(manifest) == identity.get("manifest_sha256"),
            "Weighted validation manifest path/SHA256 changed")
    selection = read_json(manifest)
    train, val = selection.get("splits", {}).get("train", []), selection.get("splits", {}).get("val", [])
    require(selection.get("schema") == "lusnar_random_split_v1" and selection.get("class_names") == list(CLASSES)
            and len(train) == 9401 and len(val) == 1601
            and all(row.get("source_kind") == "original" and row.get("split") == "val" for row in val)
            and len({row["source_id"] for row in val}) == 1601
            and not {row["source_id"] for row in train} & {row["source_id"] for row in val}
            and canonical_digest(val) == identity.get("ordered_samples_sha256"),
            "Weighted TTA must use saved original validation families without training overlap")
    counts = dict(sorted(Counter(row["scene"] for row in val).items()))
    require(set(counts).issubset({"Moon_1", "Moon_3", "Moon_5", "Moon_6", "Moon_7", "Moon_8"})
            and len(counts) > 1 and set(counts) == set(identity.get("validation_scenes", [])),
            "Test scenes cannot be used to select scale weights")
    statistics = calibration_validation_statistics(value, identity, counts, str(saved_path))
    close(saved.get("selection_score"), statistics["pooled"]["miou"], "Saved weighted reference selection score")
    close(summary.get("best_validation_miou"), statistics["pooled"]["miou"], "Saved weighted reference best mIoU")
    return {"directory": directory, "summary": summary, "saved_path": saved_path, "identity": identity,
            "scene_counts": counts, "statistics": statistics, "checkpoint_sha256": digest(directory / "best.pt")}


def weighted_tta_analysis(run_path, result_paths, output):
    require(result_paths is not None and len(result_paths) in (1, 2),
            "weighted-tta needs a validation sweep and optionally its validation-selected test result")
    reference = weighted_validation_reference(run_path)
    directory, identity = reference["directory"], reference["identity"]
    counts, saved_stats = reference["scene_counts"], reference["statistics"]
    checkpoint_hash, epoch = reference["checkpoint_sha256"], reference["summary"]["best_epoch"]
    output = output.resolve()
    require(not output.is_relative_to(directory) and not directory.is_relative_to(output),
            "Weighted TTA output and experiment directories must not contain each other")
    if (output / "summary.json").exists():
        require(read_json(output / "summary.json").get("schema") == "lusnar_weighted_tta_analysis_v1",
                "Weighted TTA cannot overwrite another analysis protocol")
    source_path = Path(result_paths[0]).resolve();source = read_json(source_path)
    require(source.get("checkpoint_sha256") == checkpoint_hash and source.get("checkpoint_epoch") == epoch
            and Path(source.get("checkpoint", "")).is_absolute()
            and Path(source["checkpoint"]).resolve() == directory / "best.pt"
            and source.get("checkpoint_partial_training") is False and source.get("evaluation_only") is True
            and source.get("split") == "val" and source.get("mountain_alpha", 1.0) == 1
            and source.get("crater_gamma", 1.0) == 1
            and "mountain_score_sweep" not in source and "crater_score_sweep" not in source,
            "Scale weight selection must use this exact checkpoint on uncalibrated validation data")
    calibration_validation_statistics(source, identity, counts, str(source_path), scale_weight=1.0)
    sweep = source.get("scale_weight_sweep", {})
    weights = [1.0, 0.75, 0.5]
    require(sweep.get("weights_1024") == weights and sweep.get("input_sizes") == [1024, 768]
            and sweep.get("selection_valid") is True and sweep.get("selection_scene") == "custom_validation"
            and sweep.get("selection_uses_test") is False and sweep.get("loader_passes") == 1
            and sweep.get("validation_unique_source_count") == 1601
            and sweep.get("global_scale_weights_for_all_images") is True and sweep.get("automatically_applied") is False
            and sweep.get("shared_views_per_batch") == 4 and sweep.get("mountain_alpha") == sweep.get("crater_gamma") == 1,
            "Expected one full validation-only global scan of 1024 weights 1/.75/.5")
    batch_size = source.get("batch_size")
    require(type(batch_size) is int and batch_size > 0, "Weighted validation batch size is required")
    batches = math.ceil(1601 / batch_size)
    require(source.get("model_forward_calls") == sweep.get("model_forward_calls") == 4 * batches,
            "All three scale weights must reuse four model forwards per validation batch")
    candidates = sweep.get("candidates", [])
    require(isinstance(candidates, list) and [row.get("weight_1024") for row in candidates] == weights,
            "Scale-weight candidates must contain 1/.75/.5 once in the requested order")
    verified, rows, class_rows, matrices = [], [], [], []
    for item in candidates:
        weight, metrics = item["weight_1024"], item.get("metrics", {})
        require(not isinstance(weight, bool) and item.get("scale_weights") == [weight, 1 - weight]
                and item.get("standalone_model_forward_calls") == batches * (2 if weight == 1 else 4)
                and item.get("additional_model_forward_calls") == 0 and item.get("reused_shared_views") is True
                and "model_forward_calls" not in metrics,
                f"w1024={weight}: incorrect scale weights or cached/standalone forward counts")
        stats = calibration_validation_statistics(metrics, identity, counts, f"w1024={weight}", scale_weight=weight)
        for scope, old, new in [("custom_validation", saved_stats["pooled"], stats["pooled"])] + [
                (name, saved_stats["scenes"][name], stats["scenes"][name]) for name in counts]:
            a, b = old["confusion_matrix_gt_rows_pred_cols"], new["confusion_matrix_gt_rows_pred_cols"]
            require(all(sum(a[r]) == sum(b[r]) for r in range(5)), f"w1024={weight}/{scope}: GT class counts differ")
            if weight == 1:
                require(a == b, f"w1024=1/{scope}: confusion matrix must exactly equal saved best validation")
            for name, numbers in new["classes"].items():
                class_rows.append({"weight_1024": weight, "weight_768": 1 - weight, "scope": scope,
                                   "class": name, **numbers, "miou": new["miou"]})
            for r, row in enumerate(b):
                for c, pixels in enumerate(row):
                    matrices.append({"weight_1024": weight, "scope": scope, "gt_class": CLASSES[r],
                                     "predicted_class": CLASSES[c], "pixels": pixels})
        pooled = stats["pooled"]
        delta = pooled["miou"] - saved_stats["pooled"]["miou"]
        close(item.get("delta_miou"), delta, f"w1024={weight} aggregate mIoU change")
        changes = {}
        for name in CLASSES:
            change = pooled["classes"][name]["iou"] - saved_stats["pooled"]["classes"][name]["iou"]
            close(item.get("delta_class_iou", {}).get(name), change, f"w1024={weight}/{name} IoU change")
            changes[name] = change
        row = {"weight_1024": weight, "weight_768": 1 - weight, "miou": pooled["miou"], "delta_miou": delta,
               **{name + "_iou": pooled["classes"][name]["iou"] for name in CLASSES},
               "standalone_model_forward_calls": item["standalone_model_forward_calls"], "additional_model_forward_calls": 0}
        rows.append(row)
        verified.append({**row, "statistics": stats, "delta_class_iou": changes,
                         "tta_metadata": {name: value for name, value in metrics.items() if name.startswith("tta_")}})
    require(source["confusion_matrix_gt_rows_pred_cols"] == saved_stats["pooled"]["confusion_matrix_gt_rows_pred_cols"]
            and all(source["scenes"][name]["confusion_matrix_gt_rows_pred_cols"]
                    == saved_stats["scenes"][name]["confusion_matrix_gt_rows_pred_cols"] for name in counts),
            "The sweep's top-level w=1 result must reproduce saved best validation exactly")
    maximum = max(row["miou"] for row in rows)
    recommended = max(row["weight_1024"] for row in rows if row["miou"] >= maximum - 1e-12)
    close(sweep.get("recommended_weight_1024"), recommended, "Validation-selected 1024 scale weight")
    # The global weight is fixed above, before opening any optional test JSON.
    test_comparison = None
    if len(result_paths) == 2:
        selected_path = Path(result_paths[1]).resolve();selected_raw = read_json(selected_path)
        if selected_raw.get("tta_scale_weights") is not None:
            chosen = validate_test(selected_path, "multiscale_hflip", directory, checkpoint_hash, epoch,
                                   tta_sizes=[1024, 768], scale_weight=recommended)
        elif recommended == 1:
            chosen = validate_test(selected_path, "hflip", directory, checkpoint_hash, epoch)
        elif recommended == 0.5:
            chosen = validate_test(selected_path, "multiscale_hflip", directory, checkpoint_hash, epoch,
                                   tta_sizes=[1024, 768])
        else:
            raise ValueError("The validation-selected .75 weight needs an explicit weighted test result")
        baseline_path = directory / "test_hflip_metrics.json"
        baseline = validate_test(baseline_path, "hflip", directory, checkpoint_hash, epoch)
        baseline_raw = read_json(baseline_path)
        equal_path = directory / "test_multiscale_hflip_1024_768_metrics.json"
        equal = validate_test(equal_path, "multiscale_hflip", directory, checkpoint_hash, epoch, tta_sizes=[1024, 768])
        equal_raw = read_json(equal_path)
        test_identity = baseline_raw.get("data_identity", {})
        require(test_identity.get("split") == "test" and test_identity.get("partial") is False
                and test_identity.get("image_count") == test_identity.get("complete_image_count") == 5004
                and selected_raw.get("data_identity") == equal_raw.get("data_identity") == test_identity,
                "Selected test, baseline and equal-weight reference must identify the same complete test samples")
        manifest = Path(test_identity.get("manifest", ""))
        require(manifest.is_absolute() and manifest.is_file() and digest(manifest) == test_identity.get("manifest_sha256"),
                "Selected test manifest path/SHA256 mismatch")
        for raw, result in ((baseline_raw, baseline), (equal_raw, equal), (selected_raw, chosen)):
            require(raw.get("mountain_alpha", 1.0) == raw.get("crater_gamma", 1.0) == 1.0
                    and not any(key in raw for key in ("mountain_score_sweep", "crater_score_sweep", "scale_weight_sweep")),
                    "Weighted TTA test comparison must use uncalibrated, already selected fixed scale weights")
            result["pooled_statistics"] = scene_statistics(raw, result["source"] + "/all_test", 5004)
            result["tta_metadata"] = {name: value for name, value in raw.items() if name.startswith("tta_")}
            for scene in SCENES:
                require(all(result["scenes"][scene]["classes"][name]["gt_pixels"]
                            == baseline["scenes"][scene]["classes"][name]["gt_pixels"] for name in CLASSES),
                        f"{scene}: selected test changed GT class counts")
        test_comparison = {"baseline": baseline, "equal_weight_reference": equal, "selected": chosen, "weight_1024": recommended,
                           "scale_weights": [recommended, 1 - recommended], "selection_uses_test": False,
                           "weighted_score_change": chosen["weighted_score"] - baseline["weighted_score"],
                           "weighted_score_change_vs_equal": chosen["weighted_score"] - equal["weighted_score"],
                           "scene_miou_changes_vs_baseline": {scene: chosen["scenes"][scene]["miou"] - baseline["scenes"][scene]["miou"]
                                                              for scene in SCENES},
                           "scene_miou_changes_vs_equal": {scene: chosen["scenes"][scene]["miou"] - equal["scenes"][scene]["miou"]
                                                           for scene in SCENES}}
    report_weighted_tta(reference, source_path, source, sweep, output, verified, rows, class_rows, matrices,
                        recommended, test_comparison)


def report_weighted_tta(reference, source_path, source, sweep, output, candidates, rows, class_rows, matrices,
                        recommended, test_comparison):
    directory = reference["directory"]
    test_rows = []
    lines = ["# 双尺度水平翻转：验证集选择全局融合权重", "",
             f"复用 `{directory.name}/best.pt`（第 **{reference['summary']['best_epoch']}** 轮），不重新训练。"
             "在完整 1601 张随机划分验证原图上比较 w1024=1、0.75、0.5，全部图片与场景共用同一权重。", "",
             "先分别将 1024 和 768 尺度的原图、水平翻转连续类别概率还原坐标并求平均，"
             "再计算 `P = w1024 × P1024 + (1−w1024) × P768`，最后执行一次 argmax。"
             "两个尺度均原尺寸进入两支 CNN；下采样输入使用带抗锯齿的双线性插值。山体 α=1、坑 γ=1。", "",
             "w1024=1 的整体及每场景整数混淆矩阵已与保存的最佳轮验证逐位核对一致。"
             "总体 mIoU 由全部验证像素汇总的五类混淆矩阵计算，未对场景 mIoU 简单平均。", "",
             "## 验证候选", "",
             "|w1024|w768|总体 mIoU|较 w1024=1/百分点|月壤 IoU|岩石 IoU|坑 IoU|山体 IoU|天空 IoU|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("|" + "|".join([f"{row['weight_1024']:g}", f"{row['weight_768']:g}", percent(row["miou"]),
                                     f"{row['delta_miou']*100:+.5f}"] + [percent(row[name + "_iou"]) for name in CLASSES]) + "|")
    lines += ["", f"验证推荐为 **w1024={recommended:g}、w768={1-recommended:g}**。规则为最大化总体验证 mIoU；"
              "1e-12 容差内并列时优先较大的 1024 权重。该选择在读取可选测试记录之前完成，不按测试场景分别挑权重。", "",
              f"验证扫描一次遍历，执行 {sweep['model_forward_calls']:,} 次网络前向：每批缓存四视图后复用三个候选，"
              "并非对每个权重重新运行网络。单独部署 w1024=1 只需 1024 的两视图；其他候选需要四视图。", "",
              "## 各验证场景 mIoU", "", "|w1024|" + "|".join(reference["scene_counts"]) + "|",
              "|---:|" + "---:|" * len(reference["scene_counts"])]
    for candidate in candidates:
        lines.append("|" + "|".join([f"{candidate['weight_1024']:g}"] + [percent(candidate["statistics"]["scenes"][scene]["miou"])
                                                                                for scene in reference["scene_counts"]]) + "|")
    if test_comparison is None:
        lines += ["", "## 所选权重的测试评估", "",
                  "尚未提供按验证推荐权重固定执行的完整 Moon_2/4/9 测试记录。本报告此时只完成验证选择，不作测试改善结论。"]
    else:
        old, equal, new = (test_comparison[name] for name in ("baseline", "equal_weight_reference", "selected"))
        lines += ["", "## 已固定权重的完整测试评估", "",
                  "以下三项均使用同一检查点及相同 5004 张测试图（Moon_2/4/9=1597/1320/2087）。"
                  "权重已由验证集固定；等权双尺度只作为既有测试参考，不参与本次选权重。S=(Moon_2 + Moon_4 + 2×Moon_9)/4。", "",
                  "|方式|Moon_2|Moon_4|Moon_9|加权 S|", "|---|---:|---:|---:|---:|"]
        for label, item in (("1024 双视图基线", old), ("等权双尺度 w1024=0.5", equal),
                            (f"验证选择 w1024={recommended:g}", new)):
            lines.append("|" + "|".join([label] + [percent(item["scenes"][scene]["miou"]) for scene in SCENES]
                                        + [percent(item["weighted_score"])]) + "|")
        lines += ["", f"所选方案相对 1024 基线 S 变化为 **{test_comparison['weighted_score_change']*100:+.4f} 个百分点**，"
                  f"相对等权双尺度为 **{test_comparison['weighted_score_change_vs_equal']*100:+.4f} 个百分点**。", "",
                  "|场景|所选较 1024 基线/百分点|所选较等权双尺度/百分点|", "|---|---:|---:|"]
        for scene in SCENES:
            lines.append(f"|{scene}|{test_comparison['scene_miou_changes_vs_baseline'][scene]*100:+.4f}|"
                         f"{test_comparison['scene_miou_changes_vs_equal'][scene]*100:+.4f}|")
        lines += ["", "|场景|类别|基线 IoU|等权 IoU|所选 IoU|所选较等权/百分点|", "|---|---|---:|---:|---:|---:|"]
        for scene in SCENES:
            for name in CLASSES:
                a, b = old["scenes"][scene]["classes"][name]["iou"], new["scenes"][scene]["classes"][name]["iou"]
                e = equal["scenes"][scene]["classes"][name]["iou"]
                lines.append(f"|{scene}|{ZH[name]}|{percent(a)}|{percent(e)}|{percent(b)}|{(b-e)*100:+.4f}|")
                for label, result, weight in (("baseline", old, 1.0), ("equal_weight_reference", equal, 0.5),
                                              ("selected", new, recommended)):
                    test_rows.append({"test_role": label, "weight_1024": weight, "weight_768": 1 - weight,
                                      "scene": scene, "class": name, **result["scenes"][scene]["classes"][name],
                                      "scene_miou": result["scenes"][scene]["miou"], "weighted_score": result["weighted_score"]})
        lines += ["", f"- [1024 双视图测试原始记录]({relative_link(old['source'], output)})",
                  f"- [等权双尺度测试原始记录]({relative_link(equal['source'], output)})",
                  f"- [验证所选权重测试原始记录]({relative_link(new['source'], output)})",
                  "- [基线、等权与所选方案完整测试类别指标](test_metrics.csv)"]
    lines += ["", "## 解释范围与来源", "",
              "权重只在既有验证集上选择，未额外训练。验证集也参与选轮，没有单独独立的校准集或显著性检验；"
              "测试集此前已被查看，后补测试不改变本次验证选出的全局权重。", "",
              f"- [完整验证扫描]({relative_link(source_path, output)})",
              f"- [保存的最佳轮验证]({relative_link(reference['saved_path'], output)})",
              "- [候选总体和类别 IoU](metrics.csv)", "- [全部验证场景逐类别 IoU、精确率、召回率](class_metrics.csv)",
              "- [完整整数混淆矩阵](confusion_matrices.csv)", "- [权重、来源 SHA256 与机器可读结果](summary.json)", ""]
    result = {"schema": "lusnar_weighted_tta_analysis_v1", "state": "complete", "comparison": "weighted-tta",
              "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "run_dir": str(directory),
              "checkpoint_epoch": reference["summary"]["best_epoch"], "checkpoint_sha256": reference["checkpoint_sha256"],
              "source": str(source_path), "source_sha256": digest(source_path),
              "saved_validation_source": str(reference["saved_path"]), "saved_validation_sha256": digest(reference["saved_path"]),
              "validation_image_count": 1601, "data_identity": reference["identity"],
              "validation_scene_image_counts": reference["scene_counts"], "input_sizes": [1024, 768],
              "weights_1024": sweep["weights_1024"], "recommended_weight_1024": recommended,
              "recommended_scale_weights": [recommended, 1 - recommended], "selection_state": "complete",
              "selection_scene": "custom_validation", "selection_uses_test": False, "global_scale_weights_for_all_images": True,
              "baseline_confusion_matrices_exactly_match_saved_best": True, "automatically_applied": False,
              "test_state": "complete" if test_comparison else "not_provided", "test_comparison": test_comparison,
              "retrained": False, "statistical_significance_tested": False, "candidates": candidates,
              "loader_passes": 1, "model_forward_calls": sweep["model_forward_calls"], "selection_rule": sweep.get("selection_rule")}
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("metrics.csv", rows), ("class_metrics.csv", class_rows), ("confusion_matrices.csv", matrices)):
        write_csv(output / filename, data, list(data[0]))
    if test_rows:
        write_csv(output / "test_metrics.csv", test_rows, list(test_rows[0]))
    for filename, text in (("summary.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("report_zh.md", "\n".join(lines))):
        temporary = output / (filename + ".tmp");temporary.write_text(text);temporary.replace(output / filename)
    print(json.dumps({"state": "complete", "recommended_weight_1024": recommended, "test_state": result["test_state"],
                      "report": str(output / "report_zh.md"), "summary": str(output / "summary.json")}, ensure_ascii=False))


def crater_calibration_analysis(run_path, result_paths, output):
    """Select one global crater multiplier using validation evidence only."""
    from collections import Counter

    require(result_paths is not None and len(result_paths) == 1,
            "crater-calibration requires one complete validation --test-results JSON")
    directory, output = Path(run_path).resolve(), output.resolve()
    require(not output.is_relative_to(directory) and not directory.is_relative_to(output),
            "Calibration output and experiment directories must not contain each other")
    if (output / "summary.json").exists():
        require(read_json(output / "summary.json").get("schema") == "lusnar_crater_calibration_analysis_v1",
                "Crater calibration cannot overwrite another analysis protocol")
    summary, config = read_json(directory / "summary.json"), read_json(directory / "run_config.json")
    epoch = summary.get("best_epoch")
    require(summary.get("status") in STOPPED and summary.get("partial") is False
            and config.get("partial") is False and type(epoch) is int and epoch > 0
            and summary.get("selection_scene") == "custom_validation"
            and summary.get("checkpoint_selection_uses_test") is False
            and summary.get("val_tta") == config.get("val_tta") == "hflip"
            and config.get("input_size") == 1024,
            "Crater calibration requires a completed, validation-selected 1024/hflip custom-split checkpoint")
    saved_path = directory / f"validation_epoch_{epoch:03d}.json"
    saved = read_json(saved_path)
    reference = saved.get("validation", {})
    identity = reference.get("data_identity", {})
    require(saved.get("epoch") == epoch and saved.get("partial") is False
            and saved.get("selection_scene") == "custom_validation"
            and saved.get("checkpoint_selection_uses_test") is False
            and identity == config.get("data_identity", {}).get("val")
            and identity.get("custom_split") is True and identity.get("split") == "val"
            and identity.get("protocol") == "random_original_source_family_4_to_1"
            and identity.get("partial") is False
            and identity.get("image_count") == identity.get("complete_image_count") == identity.get("validation_image_count") == 1601,
            "Saved best validation must identify the complete custom validation split")
    manifest = Path(identity.get("manifest", ""))
    require(manifest.is_absolute() and manifest.is_file() and digest(manifest) == identity.get("manifest_sha256"),
            "Custom validation manifest path/SHA256 changed")
    selection = read_json(manifest)
    val_rows = selection.get("splits", {}).get("val", [])
    train_rows = selection.get("splits", {}).get("train", [])
    require(selection.get("schema") == "lusnar_random_split_v1" and selection.get("class_names") == list(CLASSES)
            and len(val_rows) == 1601 and len(train_rows) == 9401
            and all(row.get("split") == "val" and row.get("source_kind") == "original" for row in val_rows)
            and len({row["source_id"] for row in val_rows}) == 1601
            and not {row["source_id"] for row in val_rows} & {row["source_id"] for row in train_rows}
            and canonical_digest(val_rows) == identity.get("ordered_samples_sha256"),
            "Calibration requires all saved original validation sources without training-family overlap")
    counts = dict(sorted(Counter(row["scene"] for row in val_rows).items()))
    allowed = {"Moon_1", "Moon_3", "Moon_5", "Moon_6", "Moon_7", "Moon_8"}
    require(set(counts).issubset(allowed) and len(counts) > 1
            and set(counts) == set(identity.get("validation_scenes", [])),
            "Test scenes cannot enter crater calibration")
    saved_stats = calibration_validation_statistics(reference, identity, counts, str(saved_path))
    close(saved.get("selection_score"), saved_stats["pooled"]["miou"], "Saved best selection score")
    close(summary.get("best_validation_miou"), saved_stats["pooled"]["miou"], "Saved best validation mIoU")
    checkpoint_hash = digest(directory / "best.pt")
    source_path = Path(result_paths[0]).resolve()
    source = read_json(source_path)
    require(source.get("checkpoint_sha256") == checkpoint_hash and source.get("checkpoint_epoch") == epoch
            and Path(source.get("checkpoint", "")).is_absolute()
            and Path(source["checkpoint"]).resolve() == directory / "best.pt"
            and source.get("checkpoint_partial_training") is False
            and source.get("split") == "val" and source.get("evaluation_only") is True
            and source.get("mountain_alpha", 1.0) == 1.0 and "mountain_score_sweep" not in source
            and source.get("crater_gamma", 1.0) == 1.0,
            "Crater sweep must evaluate this exact best checkpoint on validation without other calibration")
    calibration_validation_statistics(source, identity, counts, str(source_path))
    require(source["confusion_matrix_gt_rows_pred_cols"] == reference["confusion_matrix_gt_rows_pred_cols"]
            and all(source["scenes"][name]["confusion_matrix_gt_rows_pred_cols"]
                    == reference["scenes"][name]["confusion_matrix_gt_rows_pred_cols"] for name in counts),
            "Gamma=1 must exactly reproduce the saved best validation confusion matrices")
    sweep = source.get("crater_score_sweep", {})
    gammas = [1.0, 1.03, 1.06, 1.10]
    require(sweep.get("gammas") == gammas and sweep.get("class_id") == 2
            and sweep.get("selection_valid") is True and sweep.get("selection_scene") == "custom_validation"
            and sweep.get("global_gamma_for_all_images") is True and sweep.get("loader_passes") == 1
            and sweep.get("selection_uses_test") is False and sweep.get("mountain_alpha") == 1.0
            and sweep.get("probability_threshold") is False and sweep.get("validation_unique_source_count") == 1601
            and sweep.get("automatically_applied") is False,
            "Expected one-pass global crater-only validation scan of gamma=1/1.03/1.06/1.10")
    batch_size = source.get("batch_size")
    require(type(batch_size) is int and batch_size > 0
            and sweep.get("model_forward_calls") == 2 * math.ceil(1601 / batch_size),
            "All crater candidates must reuse the same 1024/hflip forward pass")
    candidates = sweep.get("candidates", [])
    require(isinstance(candidates, list) and [row.get("gamma") for row in candidates] == gammas,
            "Crater candidates must contain each requested gamma exactly once")
    baseline = saved_stats["pooled"]
    baseline_crater = baseline["classes"]["crater"]
    verified, metric_rows, class_rows, matrices = [], [], [], []
    for item in candidates:
        gamma = item["gamma"]
        value = item.get("metrics", {})
        statistics = calibration_validation_statistics(value, identity, counts, f"gamma={gamma}")
        for scope, old, new in [("custom_validation", saved_stats["pooled"], statistics["pooled"])] + [
                (name, saved_stats["scenes"][name], statistics["scenes"][name]) for name in counts]:
            a, b = old["confusion_matrix_gt_rows_pred_cols"], new["confusion_matrix_gt_rows_pred_cols"]
            require(all(sum(a[r]) == sum(b[r]) for r in range(5)), f"gamma={gamma}/{scope}: GT class counts changed")
            require(all(b[r][2] >= a[r][2] and all(b[r][c] <= a[r][c] for c in range(5) if c != 2)
                        for r in range(5)), f"gamma={gamma}/{scope}: scaling crater changed another winning class")
            if gamma == 1:
                require(a == b, f"gamma=1/{scope}: baseline confusion matrix differs from saved best validation")
            for name, numbers in new["classes"].items():
                class_rows.append({"gamma": gamma, "scope": scope, "class": name, **numbers, "miou": new["miou"]})
            for r, row in enumerate(b):
                for c, pixels in enumerate(row):
                    matrices.append({"gamma": gamma, "scope": scope, "gt_class": CLASSES[r],
                                     "predicted_class": CLASSES[c], "pixels": pixels})
        pooled = statistics["pooled"]
        crater = pooled["classes"]["crater"]
        matrix, old_matrix = pooled["confusion_matrix_gt_rows_pred_cols"], baseline["confusion_matrix_gt_rows_pred_cols"]
        new_tp, new_fp = crater["tp"] - baseline_crater["tp"], crater["fp"] - baseline_crater["fp"]
        expected = {"corrected_missed_crater_pixels": new_tp,
                    "corrected_regolith_to_crater_pixels": old_matrix[2][0] - matrix[2][0],
                    "new_false_crater_pixels": new_fp,
                    "new_regolith_to_crater_false_pixels": old_matrix[0][0] - matrix[0][0],
                    "new_false_crater_on_gt_regolith_pixels": matrix[0][2] - old_matrix[0][2],
                    "baseline_crater_true_positive_pixels": baseline_crater["tp"],
                    "crater_true_positive_pixels": crater["tp"],
                    "baseline_crater_false_positive_pixels": baseline_crater["fp"],
                    "crater_false_positive_pixels": crater["fp"],
                    "delta_crater_true_positive_pixels": new_tp, "delta_crater_false_positive_pixels": new_fp,
                    "lost_true_crater_pixels": 0, "removed_false_crater_pixels": 0,
                    "changed_between_non_crater_classes_pixels": 0}
        diagnostics = item.get("diagnostics", {})
        require(all(type(diagnostics.get(name)) is int and diagnostics[name] == count for name, count in expected.items()),
                f"gamma={gamma}: added crater true/false-positive diagnostics disagree with confusion-matrix differences")
        previous_predictions = [sum(old_matrix[r][c] for r in range(5)) for c in range(5)]
        new_predictions = [sum(matrix[r][c] for r in range(5)) for c in range(5)]
        transition = [[0] * 5 for _ in range(5)]
        for c in range(5):
            transition[c][c] = previous_predictions[c] if c == 2 else new_predictions[c]
            if c != 2:
                transition[c][2] = previous_predictions[c] - new_predictions[c]
        require(diagnostics.get("transition_matrix_baseline_rows_calibrated_cols") == transition,
                f"gamma={gamma}: prediction-transition matrix disagrees with class-2-only calibration")
        delta_miou, delta_iou = pooled["miou"] - baseline["miou"], crater["iou"] - baseline_crater["iou"]
        close(item.get("delta_miou"), delta_miou, f"gamma={gamma} delta mIoU")
        close(item.get("delta_crater_iou"), delta_iou, f"gamma={gamma} delta crater IoU")
        row = {"gamma": gamma, "miou": pooled["miou"], "crater_iou": crater["iou"],
               "crater_precision": crater["precision"], "crater_recall": crater["recall"],
               "delta_miou": delta_miou, "delta_crater_iou": delta_iou, **expected,
               "new_false_pixels_per_corrected_crater_pixel": ratio(new_fp, new_tp),
               "previously_correct_non_crater_pixels_lost": sum(old_matrix[k][k] - matrix[k][k] for k in range(5) if k != 2),
               "net_correct_pixel_change": sum(matrix[k][k] - old_matrix[k][k] for k in range(5))}
        metric_rows.append(row)
        verified.append({**row, "statistics": statistics, "diagnostics": diagnostics})
    improving = [row for row in metric_rows if row["delta_miou"] > 1e-12 and row["delta_crater_iou"] > 1e-12]
    best_score = max((row["miou"] for row in improving), default=baseline["miou"])
    recommended = min((row["gamma"] for row in improving if row["miou"] >= best_score - 1e-12), default=1.0)
    close(sweep.get("recommended_gamma"), recommended, "Validation-selected crater gamma")
    # Rank/gap diagnostics are checked and rendered below; no test result is
    # loaded anywhere in this calibration path.
    rank_diagnostics = sweep.get("baseline_missed_crater_diagnostics", {})
    validate_crater_rank_diagnostics(rank_diagnostics, saved_stats, val_rows, gammas)
    report_crater_calibration(directory, output, source_path, saved_path, summary, checkpoint_hash,
                              identity, counts, verified, metric_rows, class_rows, matrices,
                              rank_diagnostics, recommended, sweep)


def validate_crater_rank_diagnostics(value, saved_stats, val_rows, gammas):
    baseline = saved_stats["pooled"]
    expected = {"missed_crater": baseline["classes"]["crater"]["fn"],
                "missed_crater_pred_regolith": baseline["confusion_matrix_gt_rows_pred_cols"][2][0]}
    for name, count in expected.items():
        group = value.get(name, {})
        ranks = group.get("crater_rank_counts", {})
        require(group.get("pixel_count") == count and set(ranks) == {str(i) for i in range(1, 6)}
                and all(type(n) is int and n >= 0 for n in ranks.values()) and sum(ranks.values()) == count
                and group.get("rank2_pixels") == ranks["2"], f"{name}: crater-rank counts disagree with baseline misses")
        if count:
            close(group.get("rank2_fraction"), ranks["2"] / count, f"{name} rank-2 fraction")
        else:
            require(group.get("rank2_fraction") is None, f"{name}: empty rank fraction must be null")
        for field, minimum in (("winner_minus_crater_probability", 0.0), ("winner_to_crater_probability_ratio", 1.0)):
            distribution = group.get(field, {})
            bounds, bins = distribution.get("bin_upper_bounds", []), distribution.get("bin_counts", [])
            require(distribution.get("count") == count and bounds and bounds[-1] is None
                    and len(bounds) == len(bins) and all(type(n) is int and n >= 0 for n in bins)
                    and sum(bins) == count and isinstance(distribution.get("bin_rule"), str)
                    and all(isinstance(n, (int, float)) and math.isfinite(n) for n in bounds[:-1])
                    and all(a < b for a, b in zip(bounds[:-2], bounds[1:-1])), f"{name}/{field}: malformed histogram")
            if count:
                lo, mean, hi = (distribution.get(key) for key in ("min", "mean", "max"))
                require(all(isinstance(n, (int, float)) and math.isfinite(n) for n in (lo, mean, hi))
                        and minimum - 1e-7 <= lo <= mean <= hi, f"{name}/{field}: invalid score distribution")
            else:
                require(all(distribution.get(key) is None for key in ("min", "mean", "max")),
                        f"{name}/{field}: empty distribution must have null statistics")
        thresholds = group.get("ratio_threshold_counts", [])
        require([row.get("gamma") for row in thresholds] == gammas
                and all(type(row.get("pixels")) is int and 0 <= row["pixels"] <= count for row in thresholds)
                and all(a["pixels"] <= b["pixels"] for a, b in zip(thresholds, thresholds[1:])),
                f"{name}: malformed ratio-threshold counts")
        for row in thresholds:
            if count:
                close(row.get("fraction"), row["pixels"] / count, f"{name}/ratio<={row['gamma']}")
            else:
                require(row.get("fraction") is None, f"{name}: empty ratio fraction must be null")
    require(all(isinstance(value.get(field), str) and value[field] for field in
                ("crater_rank_definition", "ratio_definition", "rank2_close_definition", "pixel_example_policy")),
            "Rank/ratio and bounded-example definitions are required")
    examples = value.get("pixel_example_groups", {})
    limits = {"rank2_close": 12, "rank2_far": 12, "other_rank": 8}
    require(set(examples) == set(limits) and all(isinstance(examples[name], list) and len(examples[name]) <= limit
                                               for name, limit in limits.items())
            and value.get("pixel_example_count") == sum(map(len, examples.values())) <= 32,
            "Crater diagnostic examples must be bounded to 32 points")
    sources, seen = {row["source_id"]: row for row in val_rows}, {}
    for name, points in examples.items():
        for point in points:
            source_id = point.get("source_id")
            require(source_id in sources and point.get("scene") == sources[source_id]["scene"]
                    and point.get("gt") == 2 and point.get("pred") in (0, 1, 3, 4)
                    and point.get("group") == name and point.get("coordinate_frame") == "final_target_grid"
                    and point.get("height") == point.get("width") == 1024
                    and all(type(point.get(key)) is int and 0 <= point[key] < 1024 for key in ("x", "y")),
                    "Crater diagnostic point has invalid source, label, or coordinates")
            seen[source_id] = seen.get(source_id, 0) + 1
            require(seen[source_id] <= 2, "Diagnostic points exceed two per validation source")
            winner, crater = point.get("pwinner"), point.get("pcrater")
            require(all(isinstance(n, (int, float)) and math.isfinite(n) and 0 <= n <= 1 for n in (winner, crater))
                    and winner >= crater and type(point.get("crater_rank")) is int and 1 <= point["crater_rank"] <= 5,
                    "Crater diagnostic probabilities/rank are invalid")
            require(math.isclose(point.get("gap", math.nan), winner - crater, rel_tol=1e-6, abs_tol=1e-7)
                    and math.isclose(point.get("ratio", math.nan), winner / max(crater, 1.1754943508222875e-38),
                                     rel_tol=1e-6, abs_tol=1e-7), "Crater diagnostic gap/ratio is inconsistent")
            require((name == "rank2_close" and point["crater_rank"] == 2 and point["ratio"] <= max(gammas) + 1e-6)
                    or (name == "rank2_far" and point["crater_rank"] == 2 and point["ratio"] > max(gammas))
                    or (name == "other_rank" and point["crater_rank"] != 2), "Crater diagnostic point is in the wrong group")


def report_crater_calibration(directory, output, source_path, saved_path, training_summary, checkpoint_hash,
                              identity, counts, candidates, rows, class_rows, matrices, rank_diagnostics, recommended, sweep):
    names = {"missed_crater": "全部漏检撞击坑", "missed_crater_pred_regolith": "被判为月壤的漏检撞击坑"}
    lines = ["# 撞击坑全局分数缩放：完整验证集扫描", "",
             f"复用 `{directory.name}/best.pt`（第 **{training_summary['best_epoch']}** 轮），固定 **1024 原图＋水平翻转**。"
             "本次没有重新训练；四个 γ 共用一次完整验证数据遍历和同一组网络前向结果。", "",
             "验证集为随机划分保存的 1601 张纯原图，含 Moon_1/3/5/6/7/8；"
             "总体 mIoU 从全部验证像素累积的混淆矩阵计算，未将六个场景的 mIoU 简单平均。"
             "γ=1 的整体和每场景整数混淆矩阵均已与 best 轮保存记录逐位核对一致。", "",
             f"检查点 SHA256：`{checkpoint_hash}`。", "",
             "只将最终撞击坑类别（ID=2）的连续概率乘以同一个 γ，其他四类分数不变，然后 argmax。"
             "γ 对所有验证图片固定；山体 α=1，无阈值或标签后处理。归一化不会改变此 argmax。", "",
             "## 候选结果与推荐", "",
             "|γ|总体 mIoU|变化/百分点|坑 IoU|坑精确率|坑召回率|新增坑 TP|新增坑 FP|新增 FP/修复坑像素|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cost = row["new_false_pixels_per_corrected_crater_pixel"]
        lines.append(f"|{row['gamma']:g}|{percent(row['miou'])}|{row['delta_miou']*100:+.5f}|{percent(row['crater_iou'])}|"
                     f"{percent(row['crater_precision'])}|{percent(row['crater_recall'])}|{row['corrected_missed_crater_pixels']:,}|"
                     f"{row['new_false_crater_pixels']:,}|{'—' if cost is None else f'{cost:.4f}'}|")
    lines += ["", f"按预定验证规则推荐 **γ={recommended:g}**：只在总体 mIoU 和坑 IoU 均比 γ=1 提升的候选中最大化总体 mIoU；"
              "1e-12 容差内并列时选较小 γ；没有双改善则保留 γ=1。该推荐未自动应用，也没有读取测试结果来挑选系数。", "",
              "|γ|修复坑→月壤漏检|新增真月壤→坑误检（全部原预测）|其中原本正确月壤→坑|其他类别原本正确像素损失|净正确像素变化|",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['gamma']:g}|{row['corrected_regolith_to_crater_pixels']:,}|{row['new_false_crater_on_gt_regolith_pixels']:,}|"
                     f"{row['new_regolith_to_crater_false_pixels']:,}|{row['previously_correct_non_crater_pixels_lost']:,}|{row['net_correct_pixel_change']:+,}|")
    lines += ["", "新增 TP/FP 与预测转换统计已由混淆矩阵差独立复算。新增 FP 包含把原本正确和原本已错分的非坑像素改判为坑；"
              "两者对整体正确像素数的影响不同。所有候选均保留原有坑预测，其他非坑类别之间没有互换。", "",
              "## 各场景撞击坑指标", "", "|γ|场景|坑 IoU|坑精确率|坑召回率|", "|---:|---|---:|---:|---:|"]
    for row in candidates:
        for scene, stats in row["statistics"]["scenes"].items():
            crater = stats["classes"]["crater"]
            lines.append(f"|{row['gamma']:g}|{scene}|{percent(crater['iou'])}|{percent(crater['precision'])}|{percent(crater['recall'])}|")
    lines += ["", "Moon_1 没有坑真值，坑 IoU/召回率显示为 —；其坑误检仍进入总体混淆矩阵和坑精确率。", "",
              "## 基线漏坑的类别排名与分数差", "",
              "|漏检范围|像素数|坑排第 2 的像素|第 2 占比|平均赢家−坑概率|平均赢家/坑概率|", "|---|---:|---:|---:|---:|---:|"]
    histogram_rows = []
    for name, title in names.items():
        group = rank_diagnostics[name]
        gap = group["winner_minus_crater_probability"];ratios = group["winner_to_crater_probability_ratio"]
        fmt = lambda n: "—" if n is None else f"{n:.6f}"
        lines.append(f"|{title}|{group['pixel_count']:,}|{group['rank2_pixels']:,}|{percent(group['rank2_fraction'])}|{fmt(gap['mean'])}|{fmt(ratios['mean'])}|")
        for field, distribution in (("winner_minus_crater_probability", gap), ("winner_to_crater_probability_ratio", ratios)):
            previous = None
            for upper, count in zip(distribution["bin_upper_bounds"], distribution["bin_counts"]):
                interval = f"≤{upper:g}" if previous is None else f">{previous:g}" if upper is None else f"({previous:g}, {upper:g}]"
                histogram_rows.append({"scope": name, "score": field, "interval": interval, "pixels": count,
                                       "fraction": ratio(count, group["pixel_count"])})
                previous = upper
    lines += ["", "排名定义为 1＋严格大于坑概率的类别数；概率并列时即使坑排名为 1，argmax 仍可能选择 ID 更小的月壤或岩石。"
              "赢家/坑概率比接近 1 表示较小增益可能改变归属；排名第 2 不是可修复的必要条件。"
              "比值恰在阈值处仍受浮点运算及 argmax 并列规则影响，不能把阈值累计数直接等同于实际修复数。", "",
              "|漏检范围|比值≤1|比值≤1.03|比值≤1.06|比值≤1.10|", "|---|---:|---:|---:|---:|"]
    for name, title in names.items():
        lines.append("|" + "|".join([title] + [f"{row['pixels']:,}（{percent(row['fraction'])}）" for row in rank_diagnostics[name]["ratio_threshold_counts"]]) + "|")
    lines += ["", "分数差和比值的完整直方图见 [rank_score_histograms.csv](rank_score_histograms.csv)。"
              f"原始 JSON 保存 {rank_diagnostics['pixel_example_count']} 个有坐标的案例，按固定遍历顺序收集，最多 32 个、每原图最多 2 个；"
              "这些是示例而非随机抽样，不能用它们估计总体比例。", "",
              "## 使用范围与复核", "",
              "本次只在既有验证集上扫描四个全局系数，没有使用 Moon_2/4/9 指标挑选 γ。验证推荐不代表测试集一定改善，"
              "验证集同时参与了选轮，当前没有额外独立校准集或显著性检验。", "",
              f"- [完整验证扫描原始记录与案例]({relative_link(source_path, output)})",
              f"- [保存的最佳轮验证记录]({relative_link(saved_path, output)})", "- [候选与误检代价](metrics.csv)",
              "- [全部类别与场景指标](class_metrics.csv)", "- [完整混淆矩阵](confusion_matrices.csv)",
              "- [来源 SHA256 与机器可读结果](summary.json)", ""]
    result = {"schema": "lusnar_crater_calibration_analysis_v1", "state": "complete", "comparison": "crater-calibration",
              "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "run_dir": str(directory),
              "checkpoint_epoch": training_summary["best_epoch"], "checkpoint_sha256": checkpoint_hash,
              "source": str(source_path), "source_sha256": digest(source_path),
              "saved_validation_source": str(saved_path), "saved_validation_sha256": digest(saved_path),
              "baseline_confusion_matrices_exactly_match_saved_best": True, "validation_image_count": 1601,
              "validation_scene_image_counts": counts, "data_identity": identity, "tta_mode": "hflip", "input_size": 1024,
              "gammas": sweep["gammas"], "global_gamma_for_all_images": True, "recommended_gamma": recommended,
              "selection_scene": "custom_validation", "selection_uses_test": False, "automatically_applied": False,
              "retrained": False, "candidates": candidates, "baseline_missed_crater_diagnostics": rank_diagnostics,
              "model_forward_calls": sweep["model_forward_calls"], "loader_passes": 1,
              "selection_rule": sweep.get("selection_rule"), "statistical_significance_tested": False}
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("metrics.csv", rows), ("class_metrics.csv", class_rows),
                           ("confusion_matrices.csv", matrices), ("rank_score_histograms.csv", histogram_rows)):
        write_csv(output / filename, data, list(data[0]))
    for filename, text in (("summary.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("report_zh.md", "\n".join(lines))):
        temporary = output / (filename + ".tmp")
        temporary.write_text(text)
        temporary.replace(output / filename)
    print(json.dumps({"state": "complete", "report": str(output / "report_zh.md"),
                      "recommended_gamma": recommended, "summary": str(output / "summary.json")}, ensure_ascii=False))


def multiscale_tta_analysis(run_path, test_paths, output):
    """Compare complete inference-only results for one validation-selected model."""
    require(test_paths is not None and len(test_paths) == 2,
            "multiscale-tta requires the complete four-view and six-view --test-results")
    paths = [Path(path).resolve() for path in test_paths]
    require(len(set(paths)) == 2, "Multiscale test result paths must be distinct")
    run = load_random_split_run(run_path, "同一随机 4:1 训练检查点")
    directory = Path(run["run_dir"])
    output = output.resolve()
    require(not output.is_relative_to(directory) and not directory.is_relative_to(output),
            "Analysis output and experiment directories must not contain each other")
    tests = dict(run["tests"])
    requested = {}
    for path in paths:
        value = read_json(path)
        sizes = value.get("tta_input_sizes")
        require(sizes in ([1024, 768], [1024, 768, 512]),
                f"{path}: expected exactly [1024, 768] or [1024, 768, 512]")
        key = "multiscale_hflip_" + "_".join(map(str, sizes))
        require(key not in requested, "Both required multiscale combinations must be present once")
        requested[key] = (path, sizes)
    require(len(requested) == 2, "Missing one required multiscale size combination")
    for key, (path, sizes) in sorted(requested.items(), key=lambda row: len(row[1][1])):
        tests[key] = validate_test(path, "multiscale_hflip", directory,
                                   run["checkpoint_sha256"], run["best_epoch"], tta_sizes=sizes)

    baseline_identity = None
    for key, test in tests.items():
        source = read_json(test["source"])
        test["pooled_statistics"] = scene_statistics(source, f"{test['source']}/all_test", 5004)
        identity = source.get("data_identity", {})
        require(identity.get("split") == "test" and identity.get("partial") is False
                and identity.get("image_count") == identity.get("complete_image_count") == 5004
                and identity.get("unique_source_count") == 5004
                and isinstance(identity.get("ordered_samples_sha256"), str)
                and len(identity["ordered_samples_sha256"]) == 64,
                f"{test['source']}: incomplete test identity")
        manifest = Path(identity.get("manifest", ""))
        require(manifest.is_absolute() and manifest.is_file()
                and digest(manifest) == identity.get("manifest_sha256"),
                f"{test['source']}: test manifest path/SHA256 mismatch")
        if baseline_identity is None:
            baseline_identity = identity
        require(identity == baseline_identity, f"{test['source']}: test sample/GT identity differs from baseline")
        require(source.get("mountain_alpha", 1.0) == 1.0 and "mountain_score_sweep" not in source
                and source.get("crater_gamma", 1.0) == 1.0 and "crater_score_sweep" not in source,
                f"{test['source']}: compare uncalibrated scores with mountain alpha=1 and crater gamma=1 only")
        for scene in SCENES:
            for name in CLASSES:
                values = test["scenes"][scene]["classes"][name]
                require(values["gt_pixels"] == tests["none"]["scenes"][scene]["classes"][name]["gt_pixels"],
                        f"{test['source']}/{scene}/{name}: GT confusion-matrix row sum changed")
                close(source["scenes"][scene].get("class_precision", {}).get(name), values["precision"],
                      f"{test['source']}/{scene}/{name} precision")
                close(source["scenes"][scene].get("class_recall", {}).get(name), values["recall"],
                      f"{test['source']}/{scene}/{name} recall")
        seconds = source.get("evaluation_seconds")
        require(seconds is None or (not isinstance(seconds, bool) and isinstance(seconds, (int, float))
                                   and math.isfinite(seconds) and seconds > 0),
                f"{test['source']}: invalid evaluation duration")
        sizes = source.get("tta_input_sizes") if key.startswith("multiscale_") else [1024]
        views = 1 if key == "none" else 2 * len(sizes)
        test.update(tta_input_sizes=sizes, tta_view_count=views,
                    relative_input_pixels_vs_none=sum(size * size for size in sizes) * (1 if key == "none" else 2) / 1024 ** 2,
                    evaluation_seconds=seconds, batch_size=source.get("batch_size"), amp=source.get("amp"),
                    tta_metadata={name: value for name, value in source.items() if name.startswith("tta_")})
    none, hflip = tests["none"], tests["hflip"]
    for test in tests.values():
        test["weighted_score_change_vs_none"] = test["weighted_score"] - none["weighted_score"]
        test["weighted_score_change_vs_hflip"] = test["weighted_score"] - hflip["weighted_score"]
    summary = {"schema": "lusnar_multiscale_tta_analysis_v1", "state": "complete", "comparison": "multiscale-tta",
               "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "run_dir": run["run_dir"], "checkpoint_epoch": run["best_epoch"],
               "checkpoint_sha256": run["checkpoint_sha256"], "completed_epochs": run["completed_epochs"],
               "best_validation_miou": run["best_validation_miou"], "checkpoint_selection_scene": "custom_validation",
               "checkpoint_selection_uses_test": False, "retrained_for_this_comparison": False,
               "new_multiscale_modes_selected_on_validation": False, "comparison_requested_on_test": True,
               "test_is_previously_unseen_holdout": False, "statistical_significance_tested": False,
               "protocol": {"test_images": 5004, "scene_image_counts": SCENES, "gt_identity": baseline_identity,
                            "weighted_score": "(Moon_2 + Moon_4 + 2*Moon_9)/4", "mountain_alpha": 1.0,
                            "official_platform_score": False},
               "source_files_sha256": run["source_files_sha256"], "tests": tests,
               "highest_observed_test_view": max(tests, key=lambda key: tests[key]["weighted_score"])}
    names = {"none": "1024 单视图", "hflip": "1024 水平翻转（2 视图）",
             "multiscale_hflip_1024_768": "1024+768 水平翻转（4 视图）",
             "multiscale_hflip_1024_768_512": "1024+768+512 水平翻转（6 视图）"}
    lines = ["# 同一检查点的多尺度水平翻转测试", "",
             f"本次复用 `{directory.name}/best.pt`（第 **{run['best_epoch']}** 轮），没有重新训练。"
             f"该轮由完整 1601 张随机划分验证集的总 mIoU 选择，验证 mIoU 为 {percent(run['best_validation_miou'])}；"
             "两种新增多尺度 TTA 由用户直接指定在测试集比较，未通过验证集选择这两种模式。", "",
             f"检查点 SHA256：`{run['checkpoint_sha256']}`。四种方式均覆盖完整 Moon_2/4/9 共 5004 张"
             "（1597/1320/2087），数据身份和各场景每类真值像素数一致。指标由整数混淆矩阵独立重算。", "",
             "每个尺度使用原图和水平翻转两次前向，翻转预测先还原到原坐标；连续类别概率恢复到 1024×1024 后等权平均，"
             "最后执行一次 argmax。1024、768、512 均原尺寸输入两支 CNN，无补边或替代尺寸；山体缩放 α=1。", "",
             "## 总体结果", "",
             "S=(Moon_2 mIoU + Moon_4 mIoU + 2×Moon_9 mIoU)/4；变化列单位为百分点。", "",
             "|推理方式|Moon_2|Moon_4|Moon_9|加权 S|较单视图|较 1024 翻转|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, test in tests.items():
        lines.append(f"|{names[key]}|{percent(test['scenes']['Moon_2']['miou'])}|{percent(test['scenes']['Moon_4']['miou'])}|"
                     f"{percent(test['scenes']['Moon_9']['miou'])}|{percent(test['weighted_score'])}|"
                     f"{test['weighted_score_change_vs_none']*100:+.4f}|{test['weighted_score_change_vs_hflip']*100:+.4f}|")
    lines += ["", "## 各场景类别 IoU", "",
              "|推理方式|场景|月壤|岩石|撞击坑|山体|天空|", "|---|---|---:|---:|---:|---:|---:|"]
    rows, matrices = [], []
    for key, test in tests.items():
        for scene, values in test["scenes"].items():
            lines.append("|" + "|".join([names[key], scene] + [percent(values["classes"][name]["iou"]) for name in CLASSES]) + "|")
            for name, item in values["classes"].items():
                rows.append({"test_view": key, "tta_sizes": "+".join(map(str, test["tta_input_sizes"])),
                             "view_count": test["tta_view_count"], "scene": scene, "class": name, **item,
                             "scene_miou": values["miou"], "weighted_score": test["weighted_score"],
                             "checkpoint_epoch": run["best_epoch"], "checkpoint_sha256": run["checkpoint_sha256"]})
            for r, row in enumerate(values["confusion_matrix_gt_rows_pred_cols"]):
                for c, count in enumerate(row):
                    matrices.append({"test_view": key, "scene": scene, "gt_class": CLASSES[r],
                                     "predicted_class": CLASSES[c], "pixels": count})
    lines += ["", "## Moon_9 精确率、召回率与撞击坑去向", "",
              "|推理方式|类别|IoU|精确率|召回率|", "|---|---|---:|---:|---:|"]
    for key, test in tests.items():
        scene = test["scenes"]["Moon_9"]
        for name, item in scene["classes"].items():
            lines.append(f"|{names[key]}|{ZH[name]}|{percent(item['iou'])}|{percent(item['precision'])}|{percent(item['recall'])}|")
    for key, test in tests.items():
        scene = test["scenes"]["Moon_9"]
        crater = scene["classes"]["crater"]
        flow = scene["crater_flows"]["false_negative_to_regolith"]
        lines += ["", f"{names[key]}：撞击坑漏检 {crater['fn']:,} 像素，误检 {crater['fp']:,} 像素；"
                  f"漏检流向月壤 {flow['pixels']:,} 像素，占撞击坑漏检 {percent(flow['fraction_of_crater_false_negatives'])}。"]
    lines += ["", "## 推理代价与解释范围", "",
              "|推理方式|每图网络前向次数|输入像素总量/单视图|记录的评估秒数|", "|---|---:|---:|---:|"]
    for key, test in tests.items():
        seconds = test["evaluation_seconds"]
        lines.append(f"|{names[key]}|{test['tta_view_count']}|{test['relative_input_pixels_vs_none']:.4f}×|"
                     f"{'未记录' if seconds is None else f'{seconds:.2f}'}|")
    lines += ["", "输入像素总量仅描述计算输入规模，不等同于实际耗时或显存倍数。两种新测试在不同 GPU 并行运行，"
              "旧测试也未保证相同负载；记录秒数不能用作严格的速度基准。", "",
              "这是同一模型的推理方式比较。测试集已被查看并用于先前方案比较；本次没有多种子重复或显著性检验，"
              "不将测试集上最高分模式表述为在独立验证集上选择的结论，也不等同于官方平台成绩。", "",
              "## 可复核来源", "", "- [逐场景逐类别 IoU、精确率、召回率](metrics.csv)",
              "- [完整整数混淆矩阵](confusion_matrices.csv)", "- [检查点与来源 SHA256、完整指标](summary.json)"]
    for key, test in tests.items():
        lines.append(f"- [{names[key]}原始测试记录]({relative_link(test['source'], output)})")
    lines.append("")
    # No output directory or final report is created before all four results
    # and their shared checkpoint/ground-truth protocol pass the checks above.
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows, list(rows[0]))
    write_csv(output / "confusion_matrices.csv", matrices, list(matrices[0]))
    for filename, text in (("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("report_zh.md", "\n".join(lines))):
        temporary = output / (filename + ".tmp")
        temporary.write_text(text)
        temporary.replace(output / filename)
    print(json.dumps({"state": "complete", "report": str(output / "report_zh.md"),
                      "metrics": str(output / "metrics.csv"), "summary": str(output / "summary.json")}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-off", type=Path)
    parser.add_argument("--run-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "result" / "analysis")
    parser.add_argument("--reference", type=Path, default=None, help="Optional historical full-test summary JSON")
    parser.add_argument("--comparison", choices=("tta", "hard-pixels", "random-split", "multiscale-tta", "crater-calibration", "weighted-tta"), default="tta")
    parser.add_argument("--test-results", type=Path, nargs="+", help="Complete inference-comparison or validation-calibration result JSONs")
    parser.add_argument("--reference-tta", choices=("none", "hflip"), default="none",
                        help="Test TTA used to select the old reference recipe in random-split mode")
    args = parser.parse_args(argv)
    try:
        if args.comparison == "weighted-tta":
            require(args.run_off is None and args.reference is None, "weighted-tta selects one --run-on model's global weights on validation")
            weighted_tta_analysis(args.run_on, args.test_results, args.output)
            return
        if args.comparison == "crater-calibration":
            require(args.run_off is None and args.reference is None, "crater-calibration selects gamma on one --run-on validation split")
            crater_calibration_analysis(args.run_on, args.test_results, args.output)
            return
        if args.comparison == "multiscale-tta":
            require(args.run_off is None and args.reference is None, "multiscale-tta compares one --run-on checkpoint")
            multiscale_tta_analysis(args.run_on, args.test_results, args.output)
            return
        require(args.run_off is not None, f"{args.comparison} requires --run-off")
        require(args.test_results is None, "--test-results is only supported by inference-comparison or calibration modes")
        if args.comparison == "random-split":
            require(args.reference is None, "random-split uses --run-off as its explicit reference run")
            old_config = read_json(args.run_off.resolve() / "run_config.json")
            reference = load_run(args.run_off, "已选旧参考方案", old_config.get("val_tta", "none"))
            reference["plot_label"] = "Selected previous recipe (Moon_5 validation)"
            runs = [load_random_split_run(args.run_on)]
            require(reference["run_dir"] != runs[0]["run_dir"], "Reference and new experiment must be different runs")
            require(set(reference["tests"]) == {"none", "hflip"}, "Reference requires complete none and hflip tests")
            validate_design_reference(runs[0], reference, args.reference_tta)
            changes = same_tta_changes(reference, runs[0])
            for run in (reference, *runs):
                directory = Path(run["run_dir"])
                require(not args.output.resolve().is_relative_to(directory)
                        and not directory.is_relative_to(args.output.resolve()),
                        "Analysis output and experiment directories must not contain each other")
        elif args.comparison == "hard-pixels":
            runs = [load_run(args.run_off, "原损失对照组", "hflip"), load_run(args.run_on, "困难像素辅助组", "hflip")]
            left, right = (r["run_config"] for r in runs)
            require(left.get("hard_pixel_weight", 0) == 0 and right.get("hard_pixel_weight", 0) > 0,
                    "Hard-pixel comparison requires weight=0 versus weight>0")
            for key in ("input_size", "batch_size", "accumulation", "epochs", "freeze_epochs", "head_lr",
                        "backbone_lr", "weight_decay", "gradient_clip", "amp", "seed", "horizontal_flip",
                        "num_points", "hard_pixel_quota", "val_batch_size", "val_tta", "pretrained", "workers",
                        "initial_model_sha256"):
                require(left.get(key) == right.get(key), f"Hard-pixel comparison changed {key}")
            require(bool(left.get("initial_model_sha256")) and left.get("pretrained") is True
                    and not left.get("convnext_checkpoint") and not right.get("convnext_checkpoint"),
                    "Expected identical original pretrained initialization, without lunar checkpoint transfer")
            runs[0]["plot_label"], runs[1]["plot_label"] = "Original loss", "Directional hard pixels"
        else:
            runs = [load_run(args.run_off, "验证 TTA 关闭组", "none"), load_run(args.run_on, "验证 TTA 开启组", "hflip")]
        if args.comparison != "random-split":
            require(runs[0]["run_dir"] != runs[1]["run_dir"], "Expected two different run directories")
            require(runs[0]["selection"]["sha256"] == runs[1]["selection"]["sha256"], "Runs use different fixed training selections")
            reference = reference_result(args.reference)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        parser.exit(2, f"Analysis not ready: {error}\nNo final analysis was written.\n")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, matrices, history_rows = [], [], []
    export_runs = [reference, *runs] if args.comparison == "random-split" else runs
    for run in export_runs:
        for mode, result in run["tests"].items():
            for scene, values in result["scenes"].items():
                for name, item in values["classes"].items():
                    rows.append({"run": Path(run["run_dir"]).name, "test_tta": mode, "scene": scene, "class": name,
                                 **item, "scene_miou": values["miou"], "weighted_score": result["weighted_score"],
                                 "checkpoint_epoch": run["best_epoch"], "checkpoint_sha256": run["checkpoint_sha256"]})
                for r, row in enumerate(values["confusion_matrix_gt_rows_pred_cols"]):
                    for c, count in enumerate(row):
                        matrices.append({"run": Path(run["run_dir"]).name, "test_tta": mode, "scene": scene,
                                         "gt_class": CLASSES[r], "predicted_class": CLASSES[c], "pixels": count})
        for row in run["history"]:
            if args.comparison == "random-split":
                if run is reference:
                    continue
                history_rows.append({"run": Path(run["run_dir"]).name, "val_tta": run["val_tta"], "epoch": row["epoch"],
                                     "training_loss": row["training_loss"]["loss_total"],
                                     "validation_miou": row["validation"]["miou"], "selection_scene": "custom_validation",
                                     "backbones_frozen": row.get("backbones_frozen"), "seconds": row.get("seconds")})
            else:
                history_rows.append({"run": Path(run["run_dir"]).name, "val_tta": run["val_tta"], "epoch": row["epoch"],
                                     "training_loss": row["training_loss"]["loss_total"],
                                     "moon5_miou": row["validation"]["scenes"]["Moon_5"]["miou"],
                                     "backbones_frozen": row.get("backbones_frozen"), "seconds": row.get("seconds")})
    write_csv(output / "metrics.csv", rows, list(rows[0]))
    write_csv(output / "confusion_matrices.csv", matrices, list(matrices[0]))
    write_csv(output / "history.csv", history_rows, list(history_rows[0]))
    curves, curve_note = plot_curves(runs, output)
    summary = {"schema": "lusnar_experiment_analysis_v1", "state": "complete",
               "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "protocol": {"test_images": 5004, "scene_image_counts": SCENES,
                            "weighted_score": "(Moon_2 + Moon_4 + 2*Moon_9)/4", "official_platform_score": False},
               "runs": runs, "reference": reference, "comparison": args.comparison,
               "between_run_weighted_score_change": (changes[args.reference_tta]["weighted_score_change"] if args.comparison == "random-split" else
                   runs[1]["tests"][runs[1]["val_tta"]]["weighted_score"] - runs[0]["tests"][runs[0]["val_tta"]]["weighted_score"]),
               "different_checkpoints": (reference["checkpoint_sha256"] != runs[0]["checkpoint_sha256"] if args.comparison == "random-split" else
                   runs[0]["checkpoint_sha256"] != runs[1]["checkpoint_sha256"]),
               "causal_tta_effect_established": False, "statistical_significance_tested": False,
               "curves": curves, "curve_note": curve_note}
    if args.comparison == "random-split":
        summary.update(reference_tta=args.reference_tta, same_tta_comparisons=changes,
                       validation_scores_directly_comparable=False, official_test_used_for_recipe_selection=True,
                       checkpoint_selection_uses_test=False, test_is_previously_unseen_holdout=False,
                       comparison_is_single_factor_ablation=False)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    temporary = output / "report_zh.md.tmp"
    temporary.write_text(random_split_report(runs[0], reference, args.reference_tta, changes, output, curves, curve_note)
                         if args.comparison == "random-split" else
                         report(runs, output, curves, curve_note, reference, comparison=args.comparison))
    temporary.replace(output / "report_zh.md")
    print(json.dumps({"state": "complete", "report": str(output / "report_zh.md"),
                      "metrics": str(output / "metrics.csv"), "summary": str(output / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
