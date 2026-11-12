"""Audit saved segmentation IDs or per-image matrices and select review cases.

The script produces measurements and image paths, never inferred visual causes.
Production CLI requires the complete 5004-image, 1024-square Moon_2/4/9 test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ("regolith", "rock", "crater", "mountain", "sky")
EXPECTED_COUNTS = {"Moon_2": 1597, "Moon_4": 1320, "Moon_9": 2087}
CATEGORIES = ("crater_missed", "crater_false_positive", "mountain_crater_confusion",
              "rock_missed", "typical_median", "good_example")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def statistics(matrix):
    gt, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    ious = {name: ratio(tp[i], gt[i] + predicted[i] - tp[i]) if gt[i] else None
            for i, name in enumerate(CLASSES)}
    return {"miou": float(np.mean([x for x in ious.values() if x is not None])),
            "class_iou": ious,
            "class_precision": {name: ratio(tp[i], predicted[i]) for i, name in enumerate(CLASSES)},
            "class_recall": {name: ratio(tp[i], gt[i]) for i, name in enumerate(CLASSES)},
            "confusion_matrix_gt_rows_pred_cols": matrix.tolist()}


def id_mask(path, expected_size):
    with Image.open(path) as image:
        require(image.mode == "L", f"Expected L-mode class IDs: {path}")
        require(image.size == (expected_size, expected_size), f"Wrong mask size: {path}")
        array = np.asarray(image, dtype=np.uint8).copy()
    require(int(array.max()) <= 4, f"Class-ID mask has values outside 0..4: {path}")
    return array


def image_paths(root, record, output_mode="all"):
    files = record["files"]
    required = ({"comparison", "metrics"} if output_mode == "comparison" else
                {"prediction_ids", "prediction_color", "ground_truth_ids", "comparison"})
    require(required.issubset(files), f"Image index lacks files required by {output_mode} export")
    if output_mode == "comparison":
        require(set(files) == required, "Comparison export must list only comparison and metrics files")
    paths = {}
    for name in required:
        path = (root / files[name]).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"Saved file missing or outside image directory: {path}")
        paths[name] = str(path)
    return paths


def saved_matrix(value, pixels, label):
    require(isinstance(value, list) and len(value) == 5
            and all(isinstance(row, list) and len(row) == 5 for row in value),
            f"{label}: expected a 5x5 confusion matrix")
    require(all(type(cell) is int and 0 <= cell <= pixels for row in value for cell in row),
            f"{label}: confusion counts must be nonnegative integers")
    require(sum(sum(row) for row in value) == pixels, f"{label}: confusion pixel count differs from image size")
    return np.asarray(value, dtype=np.int64)


def check_metric_values(saved, matrix, label):
    """Recompute scalar metrics independently of the exporter and index."""
    expected = statistics(matrix)

    def same_number(actual, wanted, name):
        if wanted is None:
            require(actual is None, f"{label}: {name} must be null for its empty denominator/absent GT")
        else:
            require(type(actual) in (int, float) and math.isfinite(actual)
                    and math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=1e-12),
                    f"{label}: {name} does not reproduce its confusion matrix")

    same_number(saved.get("miou"), expected["miou"], "miou")
    for name in ("class_iou", "class_precision", "class_recall"):
        values = saved.get(name)
        require(isinstance(values, dict) and set(values) == set(CLASSES), f"{label}: invalid {name} classes")
        for cls in CLASSES:
            same_number(values[cls], expected[name][cls], f"{name}.{cls}")
    if "pixel_accuracy" in saved:
        same_number(saved["pixel_accuracy"], float(np.trace(matrix) / matrix.sum()), "pixel_accuracy")
    present = [name for name, count in zip(CLASSES, matrix.sum(1)) if count]
    if "gt_present_classes" in saved:
        require(saved["gt_present_classes"] == present, f"{label}: GT-present classes differ from matrix")
    if "absent_gt_policy" in saved:
        require(saved["absent_gt_policy"] == "Average only classes present in scene ground truth",
                f"{label}: unsupported absent-GT metric policy")
    for key, counts in (("class_gt_pixels", matrix.sum(1)), ("class_predicted_pixels", matrix.sum(0))):
        if key in saved:
            require(saved[key] == {name: int(count) for name, count in zip(CLASSES, counts)},
                    f"{label}: {key} differs from matrix")


def comparison_matrix(record, paths, expected_size):
    saved = read_json(paths["metrics"])
    label = f"{record['scene']}/{record['sample_id']}"
    require(isinstance(saved, dict) and saved == record.get("metrics"),
            f"{label}: saved metrics JSON and image index differ")
    require(saved.get("schema") == "lusnar_per_image_metrics_v1", f"{label}: unsupported per-image metric schema")
    for key in ("scene", "sample_id", "source_id"):
        require(saved.get(key) == record.get(key), f"{label}: {key} identity differs from index")
    require(record.get("shape_hw") == saved.get("shape_hw") == [expected_size, expected_size],
            f"{label}: wrong source image size")
    require(saved.get("class_names") == list(CLASSES), f"{label}: per-image class order differs")
    matrix = saved_matrix(saved.get("confusion_matrix_gt_rows_pred_cols"), expected_size ** 2, label)
    check_metric_values(saved, matrix, label)
    require(record.get("class_pixel_counts") == matrix.sum(0).tolist(),
            f"{label}: index predicted-class pixel counts differ from matrix")
    # Comparison mode uses the original three panels and legend, followed by
    # its 68-pixel IoU footer; small test fixtures use 256-pixel panels.
    preview_size = max(expected_size, 256)
    expected_comparison_hw = [preview_size + 136, preview_size * 3]
    declared_hw = record.get("comparison_shape_hw", saved.get("comparison_shape_hw"))
    require(declared_hw == expected_comparison_hw, f"{label}: wrong declared comparison JPEG dimensions")
    with Image.open(paths["comparison"]) as image:
        require(image.format == "JPEG" and image.mode == "RGB"
                and list(reversed(image.size)) == expected_comparison_hw,
                f"{label}: wrong comparison JPEG format or dimensions")
        image.verify()
    return matrix


def image_row(record, ordinal, matrix, paths):
    stats = statistics(matrix)
    gt, pred, tp, pixels = matrix.sum(1), matrix.sum(0), matrix.diagonal(), int(matrix.sum())
    row = {"scene": record["scene"], "sample_id": str(record["sample_id"]), "scene_ordinal": ordinal,
           "pixels": pixels, "miou": stats["miou"], "pixel_accuracy": float(tp.sum() / pixels),
           "crater_miss_fraction": ratio(gt[2] - tp[2], gt[2]),
           "crater_false_positive_fraction": ratio(pred[2] - tp[2], pred[2]),
           "mountain_crater_confusion_fraction": ratio(matrix[2, 3] + matrix[3, 2], gt[2] + gt[3]),
           "rock_miss_fraction": ratio(gt[1] - tp[1], gt[1]), "comparison_path": paths["comparison"]}
    for i, name in enumerate(CLASSES):
        row.update({f"{name}_gt_pixels": int(gt[i]), f"{name}_pred_pixels": int(pred[i]),
                    f"{name}_gt_fraction": float(gt[i] / pixels), f"{name}_iou": stats["class_iou"][name],
                    f"{name}_recall": stats["class_recall"][name], f"{name}_precision": stats["class_precision"][name]})
        for j, predicted in enumerate(CLASSES):
            row[f"cm_gt_{name}_pred_{predicted}"] = int(matrix[i, j])
    return row


def selection_key(row, category, gt_min_fraction):
    pixels = row["pixels"]
    crater_fn = row["crater_gt_pixels"] - row["cm_gt_crater_pred_crater"]
    crater_fp = row["crater_pred_pixels"] - row["cm_gt_crater_pred_crater"]
    if category == "crater_missed":
        return row["crater_miss_fraction"] if row["crater_gt_fraction"] >= gt_min_fraction and crater_fn >= 64 else None
    if category == "crater_false_positive":
        return row["crater_false_positive_fraction"] if (row["crater_pred_pixels"] >= 0.005 * pixels
                and crater_fp >= 64) else None
    if category == "mountain_crater_confusion":
        forward = row["cm_gt_crater_pred_mountain"]
        reverse = row["cm_gt_mountain_pred_crater"]
        eligible = ((row["crater_gt_fraction"] >= gt_min_fraction and forward >= 64)
                    or (row["mountain_gt_fraction"] >= gt_min_fraction and reverse >= 64))
        return row["mountain_crater_confusion_fraction"] if eligible else None
    if category == "rock_missed":
        rock_fn = row["rock_gt_pixels"] - row["cm_gt_rock_pred_rock"]
        return row["rock_miss_fraction"] if row["rock_gt_fraction"] >= gt_min_fraction and rock_fn >= 64 else None
    # Normal/good cases include at least three visible classes, each >=0.5%.
    return row["miou"] if sum(row[f"{name}_gt_fraction"] >= 0.005 for name in CLASSES) >= 3 else None


def crater_diagnostics(paths, expected_size):
    from scipy.ndimage import binary_erosion
    gt = id_mask(paths["ground_truth_ids"], expected_size) == 2
    pred = id_mask(paths["prediction_ids"], expected_size) == 2
    interior = binary_erosion(gt, structure=np.ones((3, 3), dtype=bool), iterations=8, border_value=0)
    boundary = gt & ~interior
    return {"erosion_pixels": 8, "gt_crater_pixels": int(gt.sum()),
            "interior_pixels": int(interior.sum()), "boundary_pixels": int(boundary.sum()),
            "whole_crater_recall": ratio(np.count_nonzero(pred & gt), np.count_nonzero(gt)),
            "interior_recall": ratio(np.count_nonzero(pred & interior), np.count_nonzero(interior)),
            "boundary_recall": ratio(np.count_nonzero(pred & boundary), np.count_nonzero(boundary)),
            "definition": "Eight 3x3 full-neighborhood erosions, outside-image false; boundary=GT minus interior, including image-edge contact. Semantic pixels, not physical instances."}


def choose_cases(rows, paths_by_key, expected_size, min_index_gap, gt_min_fraction, output_mode="all"):
    selected, unavailable = [], []
    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = [row for row in rows if row["scene"] == scene]
        used = []
        for category in CATEGORIES:
            candidates = [(selection_key(row, category, gt_min_fraction), row) for row in scene_rows]
            candidates = sorted([(value, row) for value, row in candidates if value is not None],
                                key=lambda item: (item[0], item[1]["scene_ordinal"]))
            quantile = 0.5 if category == "typical_median" else 0.85 if category == "good_example" else 0.8
            desired_rank = quantile * max(0, len(candidates) - 1)
            ranked = sorted(enumerate(candidates), key=lambda pair: (abs(pair[0] - desired_rank), pair[0]))
            chosen = next(((rank, value, row) for rank, (value, row) in ranked
                           if all(abs(row["scene_ordinal"] - index) >= min_index_gap for index in used)), None)
            if chosen is None:
                unavailable.append({"scene": scene, "category": category, "eligible_count": len(candidates),
                                    "reason": "No eligible, sufficiently spaced sample; criteria were not relaxed"})
                continue
            rank, value, row = chosen
            used.append(row["scene_ordinal"])
            paths = paths_by_key[(scene, row["sample_id"])]
            selected.append({"case_id": f"{scene}_{category}", "scene": scene, "category": category,
                             "sample_id": row["sample_id"], "scene_ordinal": row["scene_ordinal"],
                             "comparison_path": paths["comparison"], "files": paths, "metrics": row,
                             "selection": {"target_rank_fraction": quantile,
                                           "actual_rank_fraction": rank / max(1, len(candidates) - 1),
                                           "ranking_value": value, "eligible_count": len(candidates),
                                           "eligibility": ({"basis": "predicted_crater_fraction",
                                                            "minimum_predicted_crater_fraction": 0.005,
                                                            "minimum_false_positive_pixels": 64,
                                                            "ground_truth_crater_required": False}
                                                           if category == "crater_false_positive" else
                                                           {"basis": "ground_truth_source_class_fraction",
                                                            "minimum_ground_truth_class_fraction": gt_min_fraction,
                                                            "minimum_error_pixels": 64}
                                                           if category in ("crater_missed", "mountain_crater_confusion", "rock_missed") else
                                                           {"basis": "ground_truth_class_diversity",
                                                            "minimum_class_fraction": 0.005,
                                                            "minimum_class_count": 3}),
                                           "min_scene_ordinal_gap": min_index_gap},
                             "crater_diagnostics": (crater_diagnostics(paths, expected_size) if output_mode == "all" else
                                                    {"status": "unavailable", "available": False,
                                                     "reason": "Comparison export does not save GT/prediction class-ID masks; a confusion matrix cannot locate crater interiors or boundaries."})})
    return selected, unavailable


def render_galleries(cases, output, prefix, thumbnail_width=1000):
    """Read selected comparisons and write two-column review contact sheets."""
    if not cases:
        return {}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    title_font = ImageFont.truetype(str(font_path), 20) if font_path.is_file() else ImageFont.load_default()
    detail_font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    margin, caption_height, galleries = 20, 90, {}
    for scene in sorted({case["scene"] for case in cases}):
        selected = [case for case in cases if case["scene"] == scene]
        cards = []
        for case in selected:
            with Image.open(case["comparison_path"]) as source:
                picture = source.convert("RGB")
                height = max(1, round(picture.height * thumbnail_width / picture.width))
                picture = picture.resize((thumbnail_width, height), getattr(Image, "Resampling", Image).LANCZOS)
            card = Image.new("RGB", (thumbnail_width, caption_height + height), "white")
            draw = ImageDraw.Draw(card)
            draw.text((8, 5), case["case_id"], fill="black", font=title_font)
            values = case["metrics"]
            draw.text((8, 34), f"sample={case['sample_id']}  ordinal={case['scene_ordinal']}  mIoU={values['miou']:.2%}",
                      fill="black", font=detail_font)
            ious = [f"{name}={values[name + '_iou']:.1%}" for name in CLASSES if values[name + "_iou"] is not None]
            draw.text((8, 60), "GT-present class IoU: " + " | ".join(ious), fill="black", font=detail_font)
            card.paste(picture, (0, caption_height))
            cards.append(card)
        row_height = max(card.height for card in cards)
        row_count = (len(cards) + 1) // 2
        canvas = Image.new("RGB", (2 * thumbnail_width + 3 * margin, row_count * row_height + (row_count + 1) * margin), "#e8e8e8")
        for index, card in enumerate(cards):
            canvas.paste(card, (margin + (index % 2) * (thumbnail_width + margin),
                                margin + (index // 2) * (row_height + margin)))
        path = output / f"{prefix}_{scene}_selected_comparisons.jpg"
        canvas.save(path, quality=92, subsampling=0)
        galleries[scene] = str(path)
    return galleries


def analyze(image_index, metrics_path, output, *, expected_counts=None, expected_size=1024,
            min_index_gap=20, gt_min_fraction=0.01, gallery_output=None):
    expected_counts = expected_counts or EXPECTED_COUNTS
    image_index, metrics_path, output = Path(image_index).resolve(), Path(metrics_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"state": "analyzing", "image_index": str(image_index), "metrics": str(metrics_path)})
    index, metrics = read_json(image_index), read_json(metrics_path)
    require(index.get("schema") == "lusnar_evaluation_images_v1" and index.get("state") == "complete", "Image export is not complete")
    require(index.get("class_names") == list(CLASSES), "Image index class order differs")
    output_mode = index.get("output_mode", "all")
    require(output_mode in ("all", "comparison"), "Unsupported image export output mode")
    require(index.get("split") == metrics.get("split") == "test", "Only complete test image exports are supported")
    require(index.get("tta_mode") == metrics.get("tta_mode"), "Image index and scoring TTA differ")
    require(index.get("evaluation", {}).get("checkpoint_sha256") == metrics.get("checkpoint_sha256")
            and metrics.get("checkpoint_sha256"), "Image export and metrics checkpoint identities differ")
    expected_total = sum(expected_counts.values())
    records = index["images"]
    require(index.get("image_count") == metrics.get("image_count") == len(records) == expected_total, "Incomplete image count")
    require(metrics.get("partial") is False, "Partial test metrics cannot be a full image audit")
    keys = [(row["scene"], str(row["sample_id"])) for row in records]
    require(len(set(keys)) == expected_total, "Duplicate scene/sample IDs in image index")
    require({scene: sum(key[0] == scene for key in keys) for scene in set(key[0] for key in keys)} == expected_counts, "Unexpected scene counts")
    numeric_key = lambda row: (row["scene"], (0, int(row["sample_id"])) if str(row["sample_id"]).isdigit() else (1, str(row["sample_id"])))
    records = sorted(records, key=numeric_key)
    totals = {scene: np.zeros((5, 5), dtype=np.int64) for scene in expected_counts}
    ordinals, paths_by_key, rows = {scene: 0 for scene in expected_counts}, {}, []
    for number, record in enumerate(records, 1):
        scene, sample_id = record["scene"], str(record["sample_id"])
        paths = image_paths(image_index.parent, record, output_mode)
        if output_mode == "comparison":
            matrix = comparison_matrix(record, paths, expected_size)
        else:
            gt = id_mask(paths["ground_truth_ids"], expected_size)
            prediction = id_mask(paths["prediction_ids"], expected_size)
            matrix = np.bincount((gt * 5 + prediction).ravel(), minlength=25).reshape(5, 5)
        totals[scene] += matrix
        rows.append(image_row(record, ordinals[scene], matrix, paths))
        ordinals[scene] += 1
        paths_by_key[(scene, sample_id)] = paths
        if number % 250 == 0 or number == expected_total:
            evidence = "per-image metric files" if output_mode == "comparison" else "image pairs"
            print(f"Read and counted {number}/{expected_total} saved {evidence}", flush=True)
    overall = sum(totals.values(), np.zeros((5, 5), dtype=np.int64))
    require(np.array_equal(overall, np.asarray(metrics["confusion_matrix_gt_rows_pred_cols"])), "Saved image evidence does not reproduce the exact scoring confusion matrix")
    for scene, matrix in totals.items():
        require(np.array_equal(matrix, np.asarray(metrics["scenes"][scene]["confusion_matrix_gt_rows_pred_cols"])), f"{scene} saved image evidence differs from its scoring matrix")
    selected, unavailable = choose_cases(rows, paths_by_key, expected_size, min_index_gap, gt_min_fraction, output_mode)
    gallery_output = Path(gallery_output).resolve() if gallery_output else image_index.parent.parent / "visual_review"
    galleries = render_galleries(selected, gallery_output, image_index.parent.name)
    csv_path = output / "per_image_metrics.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    write_json(output / "selected_cases.json", {"schema": "lusnar_saved_image_cases_v1", "cases": selected,
                                                "image_gallery": galleries,
                                                "unavailable_categories": unavailable,
                                                "interpretation": "Quantitative review selection, not random prevalence sampling. Semantic masks are not physical instances."})
    scenes = {}
    for scene, matrix in totals.items():
        values = [row["miou"] for row in rows if row["scene"] == scene]
        scenes[scene] = {**statistics(matrix), "image_count": len(values),
                         "per_image_miou_percentiles": {str(p): float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90)},
                         "images_with_crater_gt_at_least_1_percent": sum(row["scene"] == scene and row["crater_gt_fraction"] >= .01 for row in rows)}
    result = {"schema": "lusnar_saved_image_analysis_v1", "state": "complete", "image_count": expected_total,
              "output_mode": output_mode,
              "audit_basis": ("saved_per_image_confusion_matrices" if output_mode == "comparison" else
                              "saved_ground_truth_and_prediction_class_id_pngs"),
              "production_full_test": expected_counts == EXPECTED_COUNTS and expected_size == 1024,
              "shape_hw": [expected_size, expected_size], "exact_confusion_matrix_match": True,
              "image_index": str(image_index), "image_index_sha256": digest(image_index),
              "metrics": str(metrics_path), "metrics_sha256": digest(metrics_path),
              "checkpoint_sha256": metrics["checkpoint_sha256"], "tta_mode": metrics["tta_mode"],
              "overall": statistics(overall), "scenes": scenes, "selected_case_count": len(selected),
              "gallery_paths": galleries,
              "min_scene_ordinal_gap": min_index_gap, "case_files": str(output / "selected_cases.json"),
              "per_image_metrics_csv": str(csv_path),
              "notes": ["Scene mIoU uses accumulated pixel matrices; per-image mIoU percentiles are a separate statistic.",
                        "Error cases target the 80th eligible-error percentile, median/good cases the 50th/85th image-mIoU percentile, subject to temporal spacing.",
                        "Selection is for visual review; it does not estimate failure prevalence or identify causes."]}
    if output_mode == "comparison":
        result["notes"].append("Per-image metrics were independently recomputed from saved JSON confusion matrices, not from JPEG colors or unavailable ID PNGs. Spatial crater diagnostics are unavailable.")
    if set(totals) == set(EXPECTED_COUNTS):
        result["weighted_score"] = (scenes["Moon_2"]["miou"] + scenes["Moon_4"]["miou"] + 2 * scenes["Moon_9"]["miou"]) / 4
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery-output", type=Path, default=None,
                        help="Gallery directory; defaults to the experiment image directory's visual_review")
    args = parser.parse_args()
    try:
        result = analyze(args.image_index, args.metrics, args.output, gallery_output=args.gallery_output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "summary.json", {"state": "failed", "error": str(error)})
        raise
    print(json.dumps({"state": result["state"], "image_count": result["image_count"],
                      "selected_case_count": result["selected_case_count"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
