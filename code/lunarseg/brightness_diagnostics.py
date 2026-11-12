"""Offline brightness-conditioned semantic diagnostics on a fixed GT terrain domain.

Brightness is an RGB-number proxy, not an irradiance or shadow annotation. All
conditioned confusion matrices exclude GT sky/255 but retain predicted sky.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

from .brightness import (brightness_groups, image_tags, local_std, luminance,
                         read_rgb_and_mask, tagging_rules)
from .data import CLASS_NAMES


BRIGHTNESS_NAMES = ("dark", "medium", "bright")
TAG_NAMES = ("large_crater", "edge_crater", "mountain_rock_mixed")
CONDITIONS = ("all_terrain", *BRIGHTNESS_NAMES, "near_black", "flat",
              "near_black_and_flat", "dark_textured", "dark_flat")
PANEL_GROUPS = ("dark_crater_missed", "bright_crater_missed",
                "dark_mountain_rock_confusion", "dark_correct")
NEAR_BLACK = 0.02
FLAT_STD = 0.005
PAIRED_MIN_PIXELS = 64


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _component(value):
    value = str(value)
    if (not value or value in (".", "..") or "/" in value or "\\" in value
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Invalid image identity path component: {value!r}")
    return value


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _confusion(target, prediction, selected):
    encoded = target[selected].astype(np.int64) * 5 + prediction[selected].astype(np.int64)
    return np.bincount(encoded, minlength=25).reshape(5, 5).astype(np.int64)


def _statistics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    gt, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    iou, recall, precision, destinations = {}, {}, {}, {}
    for index, name in enumerate(CLASS_NAMES):
        recall[name] = float(correct[index] / gt[index]) if index < 4 and gt[index] else None
        union = int(gt[index] + predicted[index] - correct[index])
        iou[name] = float(correct[index] / union) if index < 4 and gt[index] and union else None
        precision[name] = float(correct[index] / predicted[index]) if predicted[index] else None
        destinations[name] = ({pred_name: float(matrix[index, j] / gt[index])
                               for j, pred_name in enumerate(CLASS_NAMES)}
                              if index < 4 and gt[index] else {pred_name: None for pred_name in CLASS_NAMES})
    valid_iou = [value for value in iou.values() if value is not None]
    return {"pixel_count": int(matrix.sum()),
            "confusion_matrix_gt_rows_pred_cols": matrix.tolist(),
            "class_gt_pixels": {name: int(gt[index]) for index, name in enumerate(CLASS_NAMES)},
            "class_predicted_pixels": {name: int(predicted[index]) for index, name in enumerate(CLASS_NAMES)},
            "class_recall": recall, "class_precision": precision, "class_iou": iou,
            "miou_terrain_gt_present": float(np.mean(valid_iou)) if valid_iou else None,
            "pixel_accuracy_terrain": float(correct.sum() / matrix.sum()) if matrix.sum() else None,
            "prediction_given_gt": destinations}


def _compact_statistics(matrix):
    result = _statistics(matrix)
    return {key: result[key] for key in ("pixel_count", "confusion_matrix_gt_rows_pred_cols",
                                         "class_gt_pixels", "class_recall", "class_iou",
                                         "miou_terrain_gt_present")}


def _new_bucket():
    return {"image_count": 0, "matrices": {name: np.zeros((5, 5), dtype=np.int64) for name in CONDITIONS},
            "support_images": {name: 0 for name in CONDITIONS}}


def _add_bucket(bucket, matrices):
    bucket["image_count"] += 1
    for name, matrix in matrices.items():
        bucket["matrices"][name] += matrix
        bucket["support_images"][name] += int(matrix.sum() > 0)


def _bucket_statistics(bucket):
    return {"image_count": bucket["image_count"],
            "conditions": {name: {"support_images": bucket["support_images"][name],
                                   **_statistics(matrix)} for name, matrix in bucket["matrices"].items()}}


def _paired_statistics(rows):
    result = {}
    for name in CLASS_NAMES:
        available = [row[name] for row in rows if name in row]
        deltas = [row["dark_minus_bright_recall"] for row in available]
        dark_gt = sum(row["dark_gt_pixels"] for row in available)
        bright_gt = sum(row["bright_gt_pixels"] for row in available)
        dark_correct = sum(row["dark_correct_pixels"] for row in available)
        bright_correct = sum(row["bright_correct_pixels"] for row in available)
        result[name] = {"support_images": len(available),
                        "mean_dark_minus_bright_recall": float(np.mean(deltas)) if deltas else None,
                        "median_dark_minus_bright_recall": float(np.median(deltas)) if deltas else None,
                        "dark_gt_pixels": dark_gt, "bright_gt_pixels": bright_gt,
                        "dark_correct_pixels": dark_correct, "bright_correct_pixels": bright_correct,
                        "pooled_dark_recall": dark_correct / dark_gt if dark_gt else None,
                        "pooled_bright_recall": bright_correct / bright_gt if bright_gt else None}
    return result


def _panel_statistics(matrices):
    dark, bright = matrices["dark"], matrices["bright"]
    dark_crater_gt, bright_crater_gt = int(dark[2].sum()), int(bright[2].sum())
    dark_crater_errors, bright_crater_errors = int(dark[2, 0] + dark[2, 3]), int(bright[2, 0] + bright[2, 3])
    mixed_gt, mixed_errors = int(dark[[1, 3]].sum()), int(dark[1, 3] + dark[3, 1])
    object_gt, object_correct = int(dark[1:4].sum()), int(np.diag(dark)[1:4].sum())
    return {
        "dark_crater_missed": {"eligible": dark_crater_gt >= 64 and dark_crater_errors >= 64,
            "target_gt_pixels": dark_crater_gt, "direction_error_pixels": dark_crater_errors,
            "direction_error_rate": dark_crater_errors / dark_crater_gt if dark_crater_gt else None,
            "crater_to_regolith_pixels": int(dark[2, 0]), "crater_to_mountain_pixels": int(dark[2, 3])},
        "bright_crater_missed": {"eligible": bright_crater_gt >= 64 and bright_crater_errors >= 64,
            "target_gt_pixels": bright_crater_gt, "direction_error_pixels": bright_crater_errors,
            "direction_error_rate": bright_crater_errors / bright_crater_gt if bright_crater_gt else None,
            "crater_to_regolith_pixels": int(bright[2, 0]), "crater_to_mountain_pixels": int(bright[2, 3])},
        "dark_mountain_rock_confusion": {"eligible": mixed_gt >= 64 and mixed_errors >= 64,
            "target_gt_pixels": mixed_gt, "direction_error_pixels": mixed_errors,
            "direction_error_rate": mixed_errors / mixed_gt if mixed_gt else None,
            "mountain_to_rock_pixels": int(dark[3, 1]), "rock_to_mountain_pixels": int(dark[1, 3])},
        "dark_correct": {"eligible": object_gt >= 64 and object_correct >= 64
                                       and object_correct / object_gt >= 0.95,
            "target_gt_pixels": object_gt, "correct_target_pixels": object_correct,
            "target_recall": object_correct / object_gt if object_gt else None}}


def _selection_rules():
    return {"requested_per_group": 10, "minimum_target_gt_pixels": 64,
            "minimum_direction_error_pixels": 64, "group_priority": list(PANEL_GROUPS),
            "unique_source_across_groups": True,
            "scene_balance": "Round-robin sorted scenes; each scene queue ordered by score then source_id",
            "groups": {
                "dark_crater_missed": {"GT": "class2 AND dark", "error": "prediction in {0,3}",
                    "rank": "descending direction_error_rate, direction_error_pixels, target_gt_pixels"},
                "bright_crater_missed": {"GT": "class2 AND bright", "error": "prediction in {0,3}",
                    "rank": "descending direction_error_rate, direction_error_pixels, target_gt_pixels"},
                "dark_mountain_rock_confusion": {"GT": "class1 or class3 AND dark", "error": "1->3 or 3->1",
                    "rank": "descending direction_error_rate, direction_error_pixels, target_gt_pixels"},
                "dark_correct": {"GT": "class1,2,3 AND dark; regolith excluded from eligibility and ranking",
                    "minimum_correct_target_pixels": 64, "minimum_target_recall": 0.95,
                    "rank": "descending correct_target_pixels, target_recall, target_gt_pixels"}}}


class BrightnessDiagnostics:
    """Collect a complete fixed validation pass without retaining probabilities.

    ``records`` must be the exact validation dataset records. RGB/GT are reread
    from those records, resized like the dataset if necessary, and checked
    against the actual scored target. A supplied fixed panel is replayed
    literally; only a B0 pass without ``fixed_panel`` selects new images.
    """

    def __init__(self, records, thresholds_path, output_dir, prediction_root, fixed_panel=None):
        self.thresholds_path = Path(thresholds_path).resolve()
        self.thresholds = json.loads(self.thresholds_path.read_text())
        self.thresholds_sha256 = _digest(self.thresholds_path)
        # Delegate threshold schema/value validation to the shared tool.
        brightness_groups(np.asarray([[0.0, 1.0]], dtype=np.float32), self.thresholds)
        self.output_dir, self.prediction_root = Path(output_dir).resolve(), Path(prediction_root).resolve()
        for name in ("brightness_diagnostics.json", "per_image_records.jsonl",
                     "per_image_metrics.csv", "selected_panel.json"):
            if (self.output_dir / name).exists():
                raise FileExistsError(f"Frozen brightness diagnostic output already exists: {self.output_dir / name}")
        self.records, self.sources = {}, {}
        for raw_record in records:
            record = _json_value(dict(raw_record))
            key = (_component(record["scene"]), _component(record["sample_id"]))
            source_id = str(record.get("source_id", "/".join(key)))
            if key in self.records or source_id in self.sources:
                raise ValueError("Brightness diagnostics require unique validation image/source identities")
            record["source_id"] = source_id
            self.records[key], self.sources[source_id] = record, key
        if not self.records:
            raise ValueError("Brightness diagnostics require nonempty validation records")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_root.mkdir(parents=True, exist_ok=True)
        self.rows, self.seen, self.seen_sources = [], set(), set()
        self.all = _new_bucket()
        self.scenes = defaultdict(_new_bucket)
        self.tags = {name: _new_bucket() for name in TAG_NAMES}
        self.paired, self.paired_scenes = [], defaultdict(list)
        self.finalized = False
        self.fixed_panel, self.fixed_panel_sha256 = None, None
        if fixed_panel is not None:
            if isinstance(fixed_panel, (str, Path)):
                path = Path(fixed_panel)
                self.fixed_panel = json.loads(path.read_text())
                self.fixed_panel_sha256 = _digest(path)
            else:
                self.fixed_panel = _json_value(fixed_panel)
                raw = json.dumps(self.fixed_panel, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
                self.fixed_panel_sha256 = hashlib.sha256(raw).hexdigest()
            self._validate_fixed_panel()

    def _validate_fixed_panel(self):
        panel = self.fixed_panel
        if not isinstance(panel, dict) or panel.get("schema") != "lusnar_brightness_fixed_panel_v1":
            raise ValueError("Unsupported fixed brightness panel schema")
        if panel.get("thresholds_sha256") != self.thresholds_sha256:
            raise ValueError("Fixed panel requires the same frozen training brightness thresholds")
        images = panel.get("images")
        if not isinstance(images, list) or len(images) > 40:
            raise ValueError("Fixed panel requires at most 40 literal images")
        sources, group_counts = set(), defaultdict(int)
        for item in images:
            key = (str(item["scene"]), str(item["sample_id"]))
            source = str(item["source_id"])
            if (item.get("group") not in PANEL_GROUPS or key not in self.records
                    or self.records[key]["source_id"] != source or source in sources):
                raise ValueError("Fixed panel identity/group differs from current validation records")
            if item.get("record") != self.records[key]:
                raise ValueError("Fixed panel source record differs from current validation record")
            sources.add(source)
            group_counts[item["group"]] += 1
        if any(count > 10 for count in group_counts.values()):
            raise ValueError("Fixed panel cannot contain more than ten images in one group")

    def update(self, batch, targets, prediction, probabilities=None):
        if self.finalized:
            raise RuntimeError("Brightness diagnostics have already been finalized")
        targets, prediction = _numpy(targets), _numpy(prediction)
        if (targets.ndim != 3 or prediction.shape != targets.shape
                or not np.issubdtype(targets.dtype, np.integer)
                or not np.issubdtype(prediction.dtype, np.integer)):
            raise ValueError("Brightness diagnostics require integer BxHxW GT/prediction masks")
        if np.any(~np.isin(targets, [0, 1, 2, 3, 4, 255])) or np.any((prediction < 0) | (prediction > 4)):
            raise ValueError("Brightness diagnostics require GT0..4/255 and prediction0..4")
        if probabilities is not None and tuple(probabilities.shape) != (len(targets), 5, *targets.shape[-2:]):
            raise ValueError("Continuous probabilities must have shape Bx5xHxW matching scored masks")
        for field in ("scene", "sample_id", "source_id"):
            if field not in batch or len(batch[field]) != len(targets):
                raise ValueError(f"Brightness diagnostics require per-image batch {field}")
        for index, (target, predicted) in enumerate(zip(targets, prediction)):
            key = (_component(batch["scene"][index]), _component(batch["sample_id"][index]))
            source = str(batch["source_id"][index])
            if (key not in self.records or self.records[key]["source_id"] != source
                    or key in self.seen or source in self.seen_sources):
                raise ValueError("Unknown, duplicate, or mismatched validation source in brightness diagnostics")
            if "horizontal_flipped" in batch and bool(batch["horizontal_flipped"][index]):
                raise ValueError("Brightness diagnostics require unaugmented validation image coordinates")
            record = self.records[key]
            rgb, raw_gt = read_rgb_and_mask(record)
            native_shape = list(raw_gt.shape)
            tags = _json_value(image_tags(raw_gt))
            tags["edge_crater"] = bool(tags.get("border_crater", tags.get("edge_crater", False)))
            if rgb.shape[:2] != target.shape:
                size = (target.shape[1], target.shape[0])
                rgb = np.asarray(Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR))
                raw_gt = np.asarray(Image.fromarray(raw_gt, mode="L").resize(size, Image.Resampling.NEAREST))
            if raw_gt.shape != target.shape or not np.array_equal(raw_gt, target):
                raise ValueError("Reread GT does not match the exact evaluated target after dataset-aligned resize")
            y = luminance(rgb)
            groups = brightness_groups(y, self.thresholds)
            std = local_std(y, window=31)
            if groups.shape != target.shape or std.shape != target.shape:
                raise ValueError("Brightness helper maps must match the evaluated target grid")
            terrain = (target >= 0) & (target < 4)
            flat, black = std <= FLAT_STD, y <= NEAR_BLACK
            masks = {"all_terrain": terrain, "dark": terrain & (groups == 0),
                     "medium": terrain & (groups == 1), "bright": terrain & (groups == 2),
                     "near_black": terrain & black, "flat": terrain & flat,
                     "near_black_and_flat": terrain & black & flat,
                     "dark_textured": terrain & (groups == 0) & ~flat,
                     "dark_flat": terrain & (groups == 0) & flat}
            matrices = {name: _confusion(target, predicted, mask) for name, mask in masks.items()}
            paired = {}
            for class_id, name in enumerate(CLASS_NAMES[:4]):
                dark_gt, bright_gt = int(matrices["dark"][class_id].sum()), int(matrices["bright"][class_id].sum())
                if min(dark_gt, bright_gt) >= PAIRED_MIN_PIXELS:
                    dark_tp, bright_tp = int(matrices["dark"][class_id, class_id]), int(matrices["bright"][class_id, class_id])
                    paired[name] = {"dark_gt_pixels": dark_gt, "bright_gt_pixels": bright_gt,
                                    "dark_correct_pixels": dark_tp, "bright_correct_pixels": bright_tp,
                                    "dark_recall": dark_tp / dark_gt, "bright_recall": bright_tp / bright_gt,
                                    "dark_minus_bright_recall": dark_tp / dark_gt - bright_tp / bright_gt}
            destination = self.prediction_root / key[0] / key[1] / "prediction.png"
            if destination.exists():
                raise FileExistsError(f"Baseline prediction already exists; use a clean prediction root: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".png.tmp")
            Image.fromarray(predicted.astype(np.uint8), mode="L").save(temporary, format="PNG")
            temporary.replace(destination)
            row = {"scene": key[0], "sample_id": key[1], "source_id": source,
                   "record": record, "shape_hw": list(target.shape), "native_shape_hw": native_shape,
                   "prediction_path": str(destination), "prediction_sha256": _digest(destination),
                   "excluded_gt_sky_pixels": int((target == 4).sum()),
                   "excluded_gt_ignore_pixels": int((target == 255).sum()),
                   "gt_image_tags": tags,
                   "conditions": {name: _compact_statistics(matrix) for name, matrix in matrices.items()},
                   "paired_within_image_recall": paired, "panel_statistics": _panel_statistics(matrices)}
            _add_bucket(self.all, matrices)
            _add_bucket(self.scenes[key[0]], matrices)
            for tag in TAG_NAMES:
                if tags.get(tag):
                    _add_bucket(self.tags[tag], matrices)
            self.paired.append(paired)
            self.paired_scenes[key[0]].append(paired)
            self.rows.append(row)
            self.seen.add(key)
            self.seen_sources.add(source)

    __call__ = update

    def _panel_entry(self, row, group):
        return {"group": group, "scene": row["scene"], "sample_id": row["sample_id"],
                "source_id": row["source_id"], "record": row["record"],
                "prediction_path": row["prediction_path"], "shape_hw": row["shape_hw"],
                "gt_image_tags": row["gt_image_tags"],
                "selection_statistics": row["panel_statistics"][group]}

    def _select_panel(self, metadata):
        images, groups, used = [], {}, set()
        by_source = {row["source_id"]: row for row in self.rows}
        if self.fixed_panel is not None:
            for baseline in self.fixed_panel["images"]:
                row = by_source[baseline["source_id"]]
                entry = self._panel_entry(row, baseline["group"])
                entry["selection_statistics"] = baseline["selection_statistics"]
                entry["current_statistics"] = row["panel_statistics"][baseline["group"]]
                images.append(entry)
            for group in PANEL_GROUPS:
                selected = [item for item in images if item["group"] == group]
                baseline_group = self.fixed_panel.get("groups", {}).get(group, {})
                groups[group] = {**baseline_group, "requested_count": 10, "selected_count": len(selected),
                                 "source_ids": [item["source_id"] for item in selected]}
        else:
            for group in PANEL_GROUPS:
                candidates = [row for row in self.rows if row["panel_statistics"][group]["eligible"]]
                def sort_key(row):
                    stat = row["panel_statistics"][group]
                    if group == "dark_correct":
                        return (-stat["correct_target_pixels"], -stat["target_recall"],
                                -stat["target_gt_pixels"], row["source_id"])
                    return (-stat["direction_error_rate"], -stat["direction_error_pixels"],
                            -stat["target_gt_pixels"], row["source_id"])
                queues = defaultdict(deque)
                for row in sorted(candidates, key=sort_key):
                    if row["source_id"] not in used:
                        queues[row["scene"]].append(row)
                selected = []
                while len(selected) < 10 and any(queues.values()):
                    for scene in sorted(queues):
                        if queues[scene] and len(selected) < 10:
                            row = queues[scene].popleft()
                            selected.append(self._panel_entry(row, group))
                            used.add(row["source_id"])
                images.extend(selected)
                groups[group] = {"requested_count": 10, "eligible_images": len(candidates),
                                 "eligible_unused_images": sum(row["source_id"] not in
                                     {item["source_id"] for item in images if item["group"] != group}
                                     for row in candidates),
                                 "selected_count": len(selected),
                                 "selection_rule": _selection_rules()["groups"][group],
                                 "source_ids": [item["source_id"] for item in selected]}
        return {"schema": "lusnar_brightness_fixed_panel_v1", "image_count": len(images), "images": images,
                "groups": groups, "selection_rules": _selection_rules(),
                "thresholds_path": str(self.thresholds_path), "thresholds_sha256": self.thresholds_sha256,
                "selection_mode": "fixed_baseline_replay" if self.fixed_panel is not None else "B0_selection",
                "fixed_panel_sha256": self.fixed_panel_sha256,
                "selection_metadata": self.fixed_panel.get("selection_metadata", {}) if self.fixed_panel is not None else metadata,
                "current_evaluation_metadata": metadata,
                "interpretation": "A fixed explanatory panel, not a replacement for complete validation; no filler images"}

    def _write_csv(self, path):
        columns = ["scene", "sample_id", "source_id", "prediction_path", "terrain_pixels",
                   "excluded_gt_sky_pixels", "excluded_gt_ignore_pixels", *TAG_NAMES]
        for condition in BRIGHTNESS_NAMES:
            columns.append(f"{condition}_pixels")
            columns.extend(f"{condition}_{name}_recall" for name in CLASS_NAMES[:4])
        for name in CLASS_NAMES[:4]:
            columns.extend((f"paired_{name}_eligible", f"paired_{name}_dark_minus_bright_recall"))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in self.rows:
                flat = {key: row[key] for key in ("scene", "sample_id", "source_id", "prediction_path",
                                                 "excluded_gt_sky_pixels", "excluded_gt_ignore_pixels")}
                flat["terrain_pixels"] = row["conditions"]["all_terrain"]["pixel_count"]
                flat.update({name: bool(row["gt_image_tags"].get(name, False)) for name in TAG_NAMES})
                for condition in BRIGHTNESS_NAMES:
                    stat = row["conditions"][condition]
                    flat[f"{condition}_pixels"] = stat["pixel_count"]
                    flat.update({f"{condition}_{name}_recall": stat["class_recall"][name] for name in CLASS_NAMES[:4]})
                for name in CLASS_NAMES[:4]:
                    paired = row["paired_within_image_recall"].get(name)
                    flat[f"paired_{name}_eligible"] = paired is not None
                    flat[f"paired_{name}_dark_minus_bright_recall"] = paired["dark_minus_bright_recall"] if paired else None
                writer.writerow(flat)
        temporary.replace(path)

    def finalize(self, expected_images, metadata=None):
        if self.finalized:
            raise RuntimeError("Brightness diagnostics have already been finalized")
        if (type(expected_images) is not int or expected_images < 1 or len(self.rows) != expected_images
                or self.seen != set(self.records)):
            raise ValueError("Brightness diagnostics require a complete unique pass over the supplied validation records")
        if _digest(self.thresholds_path) != self.thresholds_sha256:
            raise ValueError("Training brightness thresholds changed during validation")
        metadata = _json_value(metadata or {})
        panel = self._select_panel(metadata)
        panel_path = self.output_dir / "selected_panel.json"
        jsonl_path = self.output_dir / "per_image_records.jsonl"
        csv_path = self.output_dir / "per_image_metrics.csv"
        report_path = self.output_dir / "brightness_diagnostics.json"
        temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(jsonl_path)
        self._write_csv(csv_path)
        _write_json(panel_path, panel)
        report = {"schema": "lusnar_brightness_diagnostics_v1", "state": "complete",
                  "image_count": len(self.rows), "expected_images": expected_images,
                  "class_names": list(CLASS_NAMES), "metadata": metadata,
                  "thresholds_path": str(self.thresholds_path), "thresholds_sha256": self.thresholds_sha256,
                  "thresholds": self.thresholds,
                  "domain_definition": "All conditions intersect fixed GT terrain IDs0..3; GT sky4 and ignore255 excluded, predicted IDs0..4 retained",
                  "conditioned_iou_definition": "For each fixed brightness/texture set, accumulate its complete 5x5 CM across all GT terrain classes; TP/(GT+pred-TP), GT-absent class null; average present terrain classes only",
                  "recall_and_destination_definition": "Each true-class row divided by its GT count in the same condition, including predicted sky; zero denominator null",
                  "brightness_definition": "RGB numerical luminance proxy, not physical irradiance and not a shadow label",
                  "luminance_formula": "Y=0.2126*R/255+0.7152*G/255+0.0722*B/255; no sRGB linearization",
                  "brightness_groups": {"0": "dark: Y<low", "1": "medium: low<=Y<high", "2": "bright: Y>=high"},
                  "special_conditions": {"near_black": "Y<=0.02", "flat": "local_std31<=0.005",
                                         "near_black_and_flat": "near_black AND flat",
                                         "dark_textured": "dark AND NOT flat", "dark_flat": "dark AND flat",
                                         "local_std": "31x31 reflect padding, float32 luminance, float64 moments, clipped nonnegative variance, float32 standard deviation",
                                         "local_std_domain": "All raw RGB neighbors enter the window; GT terrain mask applies to scored center pixels"},
                  "gt_image_tag_rules": tagging_rules(), "edge_crater_alias": "image_tags.border_crater",
                  "tag_grid": "Native original GT before any evaluation resize; tags may overlap and are not disjoint totals",
                  "rgb_alignment": "Reread raw RGB; if sizes differ use dataset-matching PIL bilinear RGB and nearest GT; require exact scored GT equality",
                  "global": _bucket_statistics(self.all),
                  "scenes": {scene: _bucket_statistics(bucket) for scene, bucket in sorted(self.scenes.items())},
                  "gt_image_tags": {name: _bucket_statistics(bucket) for name, bucket in self.tags.items()},
                  "paired_within_image_recall": {"definition": "Same image and true class with dark and bright GT counts each>=64; image-weighted dark-minus-bright recall, descriptive association rather than causal effect",
                      "minimum_pixels_each_condition": PAIRED_MIN_PIXELS,
                      "global": _paired_statistics(self.paired),
                      "scenes": {scene: _paired_statistics(rows) for scene, rows in sorted(self.paired_scenes.items())}},
                  "excluded_gt_sky_pixels": sum(row["excluded_gt_sky_pixels"] for row in self.rows),
                  "excluded_gt_ignore_pixels": sum(row["excluded_gt_ignore_pixels"] for row in self.rows),
                  "probabilities_saved_for_all_images": False,
                  "prediction_encoding": "L-mode uint8 class-ID PNG, 0..4, evaluated image coordinates",
                  "prediction_root": str(self.prediction_root),
                  "selected_panel": str(panel_path), "panel_image_count": panel["image_count"],
                  "per_image_records": str(jsonl_path), "per_image_csv": str(csv_path)}
        _write_json(report_path, report)
        self.finalized = True
        return {"image_count": len(self.rows), "brightness_diagnostics": str(report_path),
                "selected_panel": str(panel_path), "panel_image_count": panel["image_count"],
                "per_image_records": str(jsonl_path), "per_image_csv": str(csv_path),
                "prediction_root": str(self.prediction_root)}
