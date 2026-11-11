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


def validate_test(path, mode, run, checkpoint_hash, best_epoch):
    value = read_json(path)
    label = str(path)
    require(value.get("image_count") == 5004 and value.get("partial") is False,
            f"{label}: final analysis requires all 5004 images with partial=false")
    require(value.get("checkpoint_partial_training") is False, f"{label}: checkpoint has partial or unknown training scope")
    require(value.get("split") == "test" and value.get("evaluation_only") is True,
            f"{label}: expected an explicit evaluation-only test result")
    require(value.get("input_size") == 1024, f"{label}: expected 1024 input")
    require(value.get("tta_mode") == mode and value.get("test_time_augmentation") is (mode == "hflip"),
            f"{label}: incorrect TTA mode")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-off", type=Path, required=True)
    parser.add_argument("--run-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "result" / "analysis")
    parser.add_argument("--reference", type=Path, default=None, help="Optional historical full-test summary JSON")
    parser.add_argument("--comparison", choices=("tta", "hard-pixels", "random-split"), default="tta")
    parser.add_argument("--reference-tta", choices=("none", "hflip"), default="none",
                        help="Test TTA used to select the old reference recipe in random-split mode")
    args = parser.parse_args(argv)
    try:
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
