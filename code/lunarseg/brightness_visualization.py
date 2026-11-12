"""Render traceable, full-resolution brightness diagnostics from numeric arrays.

Inputs share the original image grid. Probabilities must already be normalized
final scores in the fixed class order; this module never converts logits or
changes exposure, predictions, brightness thresholds, or spatial coordinates.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("regolith", "rock", "crater", "mountain", "sky")
CLASS_PALETTE = np.asarray([[187, 70, 156], [232, 250, 80], [120, 0, 200],
                            [173, 69, 31], [34, 201, 248]], dtype=np.uint8)
BRIGHTNESS_PALETTE = np.asarray([[61, 72, 153], [48, 173, 141], [251, 216, 73]], dtype=np.uint8)
EXCLUDED_COLOR = (105, 105, 105)
DIRECTION_DEFINITIONS = (
    ("correct", None, (25, 25, 25)),
    ("crater(2) -> regolith(0)", (2, 0), (255, 167, 0)),
    ("crater(2) -> mountain(3)", (2, 3), (235, 55, 65)),
    ("mountain(3) -> rock(1)", (3, 1), (0, 205, 235)),
    ("rock(1) -> mountain(3)", (1, 3), (225, 65, 210)),
    ("other error", None, (240, 240, 240)),
    ("ignored GT (255)", None, EXCLUDED_COLOR),
)
# A fixed sequential lookup table, shared by all probability panels and cases.
_ANCHOR_POSITIONS = np.linspace(0, 1, 5)
_ANCHOR_COLORS = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                             [94, 201, 98], [253, 231, 37]], dtype=np.float32)
CONTINUOUS_LUT = np.stack([np.interp(np.linspace(0, 1, 256), _ANCHOR_POSITIONS,
                                    _ANCHOR_COLORS[:, channel]) for channel in range(3)], axis=1).round().astype(np.uint8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _component(value, label):
    text = str(value)
    _require(text not in ("", ".", "..", "None") and "/" not in text and "\\" not in text
             and not any(ord(char) < 32 for char in text), f"Invalid {label}: {value!r}")
    return text


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _text(draw, xy, text, *, width, size=25, fill="black"):
    while size > 10 and draw.textbbox((0, 0), text, font=_font(size))[2] > width:
        size -= 1
    draw.text(xy, text, font=_font(size), fill=fill)


def _fraction(value):
    return "N/A" if value is None else f"{value:.2%}"


def _statistics(gt, prediction):
    valid = gt != 255
    matrix = np.bincount((gt[valid].astype(np.int64) * 5 + prediction[valid]).ravel(), minlength=25).reshape(5, 5)
    counts, predicted, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    iou = {name: divide(diagonal[i], counts[i] + predicted[i] - diagonal[i]) if counts[i] else None
           for i, name in enumerate(CLASS_NAMES)}
    present_iou = [value for value in iou.values() if value is not None]
    return {"miou": float(np.mean(present_iou)) if present_iou else None,
            "class_iou": iou,
            "class_precision": {name: divide(diagonal[i], predicted[i]) for i, name in enumerate(CLASS_NAMES)},
            "class_recall": {name: divide(diagonal[i], counts[i]) for i, name in enumerate(CLASS_NAMES)},
            "class_gt_pixels": dict(zip(CLASS_NAMES, map(int, counts))),
            "class_predicted_pixels": dict(zip(CLASS_NAMES, map(int, predicted))),
            "pixel_accuracy": divide(diagonal.sum(), matrix.sum()),
            "gt_present_classes": [name for name, count in zip(CLASS_NAMES, counts) if count],
            "confusion_matrix_gt_rows_pred_cols": matrix.tolist(),
            "ignored_gt_pixels": int((~valid).sum()),
            "absent_gt_policy": "Average only classes present in scene ground truth"}


def _direction_map(gt, prediction):
    codes = np.zeros(gt.shape, dtype=np.uint8)
    codes[(gt != prediction) & (gt != 255)] = 5
    for code, (_, direction, _) in enumerate(DIRECTION_DEFINITIONS):
        if direction is not None:
            codes[(gt == direction[0]) & (prediction == direction[1])] = code
    codes[gt == 255] = 6
    colors = np.asarray([entry[2] for entry in DIRECTION_DEFINITIONS], dtype=np.uint8)
    return colors[codes], codes


def _continuous_map(values, maximum=1.0):
    indices = np.rint(np.clip(values / maximum, 0, 1) * 255).astype(np.uint8)
    return CONTINUOUS_LUT[indices]


def _colorbar(card, y, maximum, label):
    draw = ImageDraw.Draw(card)
    left, right, bar_height = 38, card.width - 38, 22
    gradient = np.repeat(CONTINUOUS_LUT[np.rint(np.linspace(0, 255, right - left)).astype(int)][None], bar_height, axis=0)
    card.paste(Image.fromarray(gradient, mode="RGB"), (left, y))
    for fraction in np.linspace(0, 1, 5):
        x = left + round(fraction * (right - left - 1))
        draw.line((x, y + bar_height, x, y + bar_height + 5), fill="black", width=2)
        text = f"{maximum * fraction:g}"
        text_width = draw.textbbox((0, 0), text, font=_font(19))[2]
        draw.text((min(max(0, x - text_width // 2), card.width - text_width), y + bar_height + 7), text, fill="black", font=_font(19))
    _text(draw, (12, y + 55), label, width=card.width - 24, size=20)


def _legend(card, entries, y, columns=3):
    draw = ImageDraw.Draw(card)
    cell_width = card.width // columns
    for index, (name, color) in enumerate(entries):
        x, row_y = (index % columns) * cell_width + 10, y + (index // columns) * 31
        draw.rectangle((x, row_y + 4, x + 18, row_y + 22), fill=tuple(color), outline="black")
        _text(draw, (x + 25, row_y), name, width=cell_width - 38, size=20)


def _render_panel(record, rgb, gt, prediction, probabilities, groups, luminance, std, metadata, stats):
    height, width = gt.shape
    header, footer, margin, top = 60, 130, 18, 174
    card_height = header + height + footer
    canvas = Image.new("RGB", (3 * width + 4 * margin, top + 3 * card_height + 4 * margin), "#e4e4e4")
    draw = ImageDraw.Draw(canvas)
    scene, sample = record["scene"], str(record.get("sample_id", record.get("sampleid")))
    _text(draw, (margin, 9), f"Brightness diagnostic | {record['group']} | {scene}/{sample}", width=canvas.width - 2 * margin, size=34)
    _text(draw, (margin, 54), f"Original {width}x{height} grid | mIoU (GT-present): {_fraction(stats['miou'])} | pixel accuracy: {_fraction(stats['pixel_accuracy'])}",
          width=canvas.width - 2 * margin, size=27)
    _legend(canvas, [(f"{i} = {name}", color) for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_PALETTE))], 100, columns=5)
    _text(draw, (margin, 139), "Probability maps: normalized final five-class scores; identical fixed 0..1 color scale. No RGB gamma/exposure adjustment.",
          width=canvas.width - 2 * margin, size=23)
    surface = gt < 4
    gt_rgb = np.full((*gt.shape, 3), EXCLUDED_COLOR, dtype=np.uint8)
    gt_rgb[gt != 255] = CLASS_PALETTE[gt[gt != 255]]
    brightness_rgb = np.full((*gt.shape, 3), EXCLUDED_COLOR, dtype=np.uint8)
    brightness_rgb[surface] = BRIGHTNESS_PALETTE[groups[surface]]
    errors_rgb, codes = _direction_map(gt, prediction)
    std_rgb = _continuous_map(std, maximum=.5)
    std_rgb[~surface] = EXCLUDED_COLOR
    tiles = [
        ("rgb", "Source RGB (unaltered)", rgb),
        ("ground_truth", "Ground truth | IDs 0..4; 255 ignored", gt_rgb),
        ("prediction", "Prediction | original final-score argmax", CLASS_PALETTE[prediction]),
        ("error_directions", "Errors: ground truth -> prediction", errors_rgb),
        ("brightness_groups", "Brightness groups | ground-truth surface only", brightness_rgb),
        ("local_std", "Local luminance std | ground-truth surface only", std_rgb),
        ("probability_crater", "P(crater) | class ID 2", _continuous_map(probabilities[2])),
        ("probability_regolith", "P(regolith) | class ID 0", _continuous_map(probabilities[0])),
        ("probability_mountain", "P(mountain) | class ID 3", _continuous_map(probabilities[3])),
    ]
    layout = {}
    for index, (key, title, array) in enumerate(tiles):
        card = Image.new("RGB", (width, card_height), "white")
        card_draw = ImageDraw.Draw(card)
        _text(card_draw, (10, 12), title, width=width - 20, size=29)
        card.paste(Image.fromarray(array, mode="RGB"), (0, header))
        y = header + height + 10
        if key.startswith("probability_"):
            _colorbar(card, y, 1, "Fixed probability scale; exact float32 arrays saved in probabilities.npz")
        elif key == "error_directions":
            _legend(card, [(name, color) for name, _, color in DIRECTION_DEFINITIONS], y)
            _text(card_draw, (10, y + 96), "Neutral correct pixels are not classified as shadows.", width=width - 20, size=19)
        elif key == "brightness_groups":
            _legend(card, [("0 dark", BRIGHTNESS_PALETTE[0]), ("1 medium", BRIGHTNESS_PALETTE[1]),
                           ("2 bright", BRIGHTNESS_PALETTE[2]), ("GT sky/255: excluded", EXCLUDED_COLOR)], y, columns=2)
            _text(card_draw, (10, y + 65), "Labels use fixed train-fitted thresholds; sky is excluded, not dark terrain.", width=width - 20, size=20)
        elif key == "local_std":
            _colorbar(card, y, .5, "Raw normalized-luminance std; fixed 0..0.5 scale; GT sky/255 neutral")
        elif key == "ground_truth":
            _text(card_draw, (10, y), "Numeric GT is preserved in NPZ; JPEG colors are visual aids only.", width=width - 20, size=23)
        elif key == "prediction":
            _text(card_draw, (10, y), "IoU: " + " | ".join(f"{i}:{_fraction(stats['class_iou'][name])}" for i, name in enumerate(CLASS_NAMES)),
                  width=width - 20, size=23)
        else:
            _text(card_draw, (10, y), "Original RGB values; no augmentation, crop or spatial resampling.", width=width - 20, size=23)
        x = margin + (index % 3) * (width + margin)
        y0 = top + margin + (index // 3) * (card_height + margin)
        canvas.paste(card, (x, y0))
        layout[key] = {"title": title, "image_xyxy": [x, y0 + header, x + width, y0 + header + height]}
    return canvas, layout, codes


def export_brightness_panel(output_dir, record, rgb_uint8, gt_numpy, prediction_numpy,
                            probabilities_numpy_C5HW, brightness_groups_numpy,
                            luminance_numpy, local_std_numpy, metadata=None):
    """Save one nine-tile panel, exact numeric NPZ and traceable metrics JSON.

    Returns a JSON-safe entry containing ``files`` (absolute panel/metrics/
    probabilities paths), their SHA256 values, identity, and image metrics.
    The destination ``output_dir/group/scene/sample_id`` must not already exist.
    Original arrays are never mutated. GT 255 is supported as an ignored label.
    A supplied prediction must select a maximum probability. Exact float32
    probability ties retain the caller's original final-score winner; softmax
    can round distinct input scores to the same floating point probability.
    """
    record = _jsonable(dict(record))
    group = _component(record.get("group"), "group")
    scene = _component(record.get("scene"), "scene")
    sample = _component(record.get("sample_id", record.get("sampleid")), "sample_id")
    record.update(group=group, scene=scene, sample_id=sample)
    record.setdefault("source_id", record.get("sourceid", f"{scene}/{sample}"))
    metadata = _jsonable(dict(metadata or {}))
    rgb, gt, pred = np.asarray(rgb_uint8), np.asarray(gt_numpy), np.asarray(prediction_numpy)
    groups = np.asarray(brightness_groups_numpy)
    probability = np.asarray(probabilities_numpy_C5HW)
    luminance, std = np.asarray(luminance_numpy), np.asarray(local_std_numpy)
    _require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "RGB must be uint8 HxWx3")
    shape = rgb.shape[:2]
    _require(shape == (1024, 1024), "Diagnostics require the original 1024x1024 image grid")
    _require(all(array.shape == shape for array in (gt, pred, groups, luminance, std)), "GT/prediction/brightness arrays must match original RGB coordinates")
    _require(np.issubdtype(gt.dtype, np.integer) and np.isin(gt, [0, 1, 2, 3, 4, 255]).all(), "GT IDs must be 0..4 or ignored 255")
    _require(np.issubdtype(pred.dtype, np.integer) and ((pred >= 0) & (pred < 5)).all(), "Prediction IDs must be 0..4")
    _require(probability.shape == (5, *shape) and np.issubdtype(probability.dtype, np.floating), "Probability array must be floating point 5xHxW in fixed class order")
    _require(np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
             and np.allclose(probability.sum(0), 1, atol=2e-5, rtol=0), "Supply normalized probabilities, not logits or class IDs")
    selected_probability = np.take_along_axis(probability, pred.astype(np.intp)[None], axis=0)[0]
    _require(np.array_equal(selected_probability, probability.max(0)), "Prediction does not select an exact maximum supplied probability")
    surface = gt < 4
    _require(np.issubdtype(groups.dtype, np.integer) and np.isin(groups[surface], [0, 1, 2]).all(), "Surface brightness IDs must be 0=dark, 1=medium, 2=bright")
    _require(np.isfinite(luminance).all() and luminance.min() >= 0 and luminance.max() <= 1, "Luminance must use normalized [0,1] units")
    _require(np.isfinite(std).all() and std.min() >= 0 and std.max() <= .50001, "Local std must retain normalized luminance units, not per-image stretching")
    rgb = np.array(rgb, copy=True)
    gt, pred, groups = (np.array(value, dtype=np.uint8, copy=True) for value in (gt, pred, groups))
    probability = np.array(probability, dtype=np.float32, copy=True)
    selected_probability = np.take_along_axis(probability, pred.astype(np.intp)[None], axis=0)[0]
    _require(np.array_equal(selected_probability, probability.max(0)), "Saving probabilities as float32 would invalidate the source-score winner")
    luminance, std = (np.array(value, dtype=np.float32, copy=True) for value in (luminance, std))
    destination = Path(output_dir).resolve() / group / scene / sample
    _require(not destination.exists(), f"Never overwrite an existing brightness diagnostic: {destination}")
    stats = _statistics(gt, pred)
    panel, layout, error_codes = _render_panel(record, rgb, gt, pred, probability, groups, luminance, std, metadata, stats)
    destination.mkdir(parents=True)
    panel_path, numeric_path, metrics_path = destination / "panel.jpg", destination / "probabilities.npz", destination / "metrics.json"
    panel.save(panel_path, quality=94, subsampling=0)
    np.savez_compressed(numeric_path, probabilities=probability, rgb_uint8=rgb,
                        ground_truth_ids=gt, prediction_ids=pred, brightness_group_ids=groups,
                        luminance=luminance, local_std=std, error_direction_ids=error_codes)
    direction_counts = {name: int((error_codes == code).sum()) for code, (name, _, _) in enumerate(DIRECTION_DEFINITIONS)}
    brightness_counts = {name: int(np.count_nonzero(surface & (groups == code)))
                         for code, name in enumerate(("dark", "medium", "bright"))}
    summary = {"schema": "lusnar_brightness_panel_v1", "group": group, "scene": scene,
               "sample_id": sample, "source_id": record["source_id"], "record": record,
               "shape_hw": list(shape), "class_names": list(CLASS_NAMES), "class_palette_rgb": CLASS_PALETTE.tolist(),
               "panel_shape_hw": [panel.height, panel.width], "panel_layout": layout,
               "metrics": stats, "brightness_surface_pixel_counts": brightness_counts,
               "brightness_excluded_gt_sky_or_255_pixels": int((~surface).sum()),
               "error_direction_pixel_counts": direction_counts,
               "direction_palette": [{"id": i, "label": name, "gt_pred": direction, "rgb": color}
                                      for i, (name, direction, color) in enumerate(DIRECTION_DEFINITIONS)],
               "probability_visualization": {"range": [0, 1], "channels": {"crater": 2, "regolith": 0, "mountain": 3},
                                             "color_lut_rgb": CONTINUOUS_LUT.tolist(), "renormalized_by_renderer": False,
                                             "prediction_tie_policy": "Keep the original final-score winner only when its saved probability equals the exact maximum; never alter probabilities",
                                             "exact_maximum_tie_pixels": int(((probability == probability.max(0)).sum(0) > 1).sum()),
                                             "source_winner_differs_from_probability_first_argmax_pixels": int((pred != probability.argmax(0)).sum())},
               "local_std_visualization_range": [0, .5],
               "metadata": metadata, "numeric_ground_truth_basis": "source GT array saved exactly as ground_truth_ids in NPZ; never JPEG colors",
               "numeric_arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                                           "pixel_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                                  for name, value in (("probabilities", probability), ("rgb_uint8", rgb),
                                                      ("ground_truth_ids", gt), ("prediction_ids", pred),
                                                      ("brightness_group_ids", groups), ("luminance", luminance),
                                                      ("local_std", std), ("error_direction_ids", error_codes))},
               "files": {"panel": str(panel_path), "metrics": str(metrics_path), "probabilities": str(numeric_path)},
               "file_sha256": {"panel": _sha256(panel_path), "probabilities": _sha256(numeric_path)}}
    _write_json(metrics_path, summary)
    return {"group": group, "scene": scene, "sample_id": sample, "source_id": record["source_id"],
            "miou": stats["miou"], "class_iou": stats["class_iou"], "files": summary["files"],
            "panel_path": str(panel_path), "metrics_path": str(metrics_path), "probabilities_path": str(numeric_path),
            "file_sha256": {**summary["file_sha256"], "metrics": _sha256(metrics_path)}}


def build_panel_overviews(entries, output_dir):
    """Create two-column contact sheets, at most ten cases per group/page.

    Returns ``{group: [absolute_overview_jpeg_paths]}``; individual full-size
    panels and NPZ arrays remain the evidence for detailed inspection.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grouped, seen = defaultdict(list), set()
    for entry in entries:
        group = _component(entry.get("group"), "group")
        key = group, entry["scene"], str(entry["sample_id"])
        _require(key not in seen, f"Duplicate overview identity: {key}")
        seen.add(key)
        grouped[group].append(entry)
    result, index = {}, []
    for group in sorted(grouped):
        paths = []
        for start in range(0, len(grouped[group]), 10):
            selected = grouped[group][start:start + 10]
            cards = []
            for entry in selected:
                path = Path(entry.get("panel_path") or entry["files"]["panel"])
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image = image.resize((900, max(1, round(image.height * 900 / image.width))), Image.Resampling.LANCZOS)
                card = Image.new("RGB", (900, image.height + 65), "white")
                draw = ImageDraw.Draw(card)
                _text(draw, (8, 5), f"{group} | {entry['scene']}/{entry['sample_id']}", width=884, size=23)
                _text(draw, (8, 35), f"mIoU (GT-present): {_fraction(entry.get('miou'))} | full panel + numeric arrays linked in overview index",
                      width=884, size=20)
                card.paste(image, (0, 65))
                cards.append(card)
            height, rows = max(card.height for card in cards), (len(cards) + 1) // 2
            canvas = Image.new("RGB", (1860, rows * (height + 20) + 20), "#e4e4e4")
            for i, card in enumerate(cards):
                canvas.paste(card, (20 + (i % 2) * 920, 20 + (i // 2) * (height + 20)))
            target = output / f"{group}_overview_{start // 10 + 1:02d}.jpg"
            _require(not target.exists(), f"Never overwrite an existing overview: {target}")
            canvas.save(target, quality=92, subsampling=0)
            paths.append(str(target))
            index.append({"group": group, "overview": str(target), "sha256": _sha256(target), "entries": selected})
        result[group] = paths
    index_path = output / "overview_index.json"
    _require(not index_path.exists(), f"Never overwrite an existing overview index: {index_path}")
    _write_json(index_path, {"schema": "lusnar_brightness_overviews_v1", "case_count": len(seen),
                             "pages": index, "note": "Thumbnails are for navigation; quantitative data comes from exact saved NPZ arrays."})
    return result
