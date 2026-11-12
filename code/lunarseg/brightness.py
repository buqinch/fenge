"""Fixed training-only luminance thresholds and reusable CPU diagnostics."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

CLASS_NAMES = ("regolith", "rock", "crater", "mountain", "sky")
BRIGHTNESS_GROUP_NAMES = ("dark", "medium", "bright")
HISTOGRAM_BINS = 4096
PAIR_DISTANCES = (8, 16, 32, 64)
PAIR_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rgb_and_mask(record):
    """Read native saved pixels: RGB uint8 HWC and L-mode IDs uint8 HW.

    No resize, training flip, gamma conversion, RGB jitter or normalization is
    applied. Original RGBA files use the same RGB conversion as LunarDataset.
    Palette indices must never be interpreted as semantic class IDs.
    """
    with Image.open(record["image"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(record["mask"]) as image:
        if image.mode != "L":
            raise ValueError(f"Expected L-mode semantic IDs: {record['mask']} ({image.mode})")
        mask = np.asarray(image, dtype=np.uint8).copy()
    if mask.ndim != 2 or rgb.shape[:2] != mask.shape:
        raise ValueError(f"Native image/GT shapes differ: {record.get('source_id')}")
    if np.any((mask > 4) & (mask != 255)):
        raise ValueError(f"GT must contain class IDs 0..4 or ignore ID 255: {record['mask']}")
    return rgb, mask


def luminance(rgb):
    """Return float32 Y in [0,1] from uint8 RGB, without linearizing sRGB."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected native uint8 HxWx3 RGB")
    values = rgb.astype(np.float32) / np.float32(255.0)
    result = (values[..., 0] * np.float32(.2126) + values[..., 1] * np.float32(.7152)
              + values[..., 2] * np.float32(.0722))
    return np.clip(result, np.float32(0), np.float32(1))


def _threshold_values(thresholds):
    if isinstance(thresholds, dict) and "thresholds" in thresholds:
        thresholds = thresholds["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"low", "high"}:
        raise ValueError("Thresholds must be a fitted document or {'low': value, 'high': value}")
    low, high = thresholds["low"], thresholds["high"]
    if (isinstance(low, bool) or isinstance(high, bool) or not np.isfinite(low)
            or not np.isfinite(high) or not 0 <= low <= high <= 1):
        raise ValueError("Expected finite ordered brightness thresholds in [0,1]")
    return float(low), float(high)


def brightness_groups(luma, thresholds):
    """uint8 groups 0=dark, 1=medium, 2=bright; boundaries join upper group."""
    low, high = _threshold_values(thresholds)
    values = np.asarray(luma)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected finite HxW luminance in [0,1]")
    return np.searchsorted(np.asarray([low, high]), values, side="right").astype(np.uint8)


def local_std(luma, window=31):
    """Native luminance standard deviation, reflect border, float64 moments."""
    from scipy.ndimage import uniform_filter
    values = np.asarray(luma)
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise ValueError("local_std window must be a positive odd integer")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite HxW luminance")
    values = values.astype(np.float64)
    mean = uniform_filter(values, size=window, mode="reflect")
    square_mean = uniform_filter(values * values, size=window, mode="reflect")
    return np.sqrt(np.maximum(square_mean - mean * mean, 0)).astype(np.float32)


def tagging_rules():
    return {"large_crater": "Largest 8-connected GT crater component has >=5% of all image pixels",
            "border_crater": "At least one 8-connected GT crater component touches any image edge and has >=0.1% of all image pixels",
            "mountain_rock_mixed": "GT mountain and GT rock each occupy >=0.5% of all image pixels, and a rock pixel is within Chebyshev distance <=8 pixels of a mountain pixel",
            "crater_connectivity": 8, "mountain_rock_neighbor_radius_pixels": 8,
            "denominator": "All native image pixels, including sky and ignored pixels",
            "scene_dependent_rules": False, "thresholds_chosen_using_predictions": False,
            "connected_components_are_physical_instances": False}


def image_tags(mask):
    """Fixed GT-defined image strata; connected regions are not true instances."""
    from scipy.ndimage import label, maximum_filter
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or np.any((mask > 4) & (mask != 255)) or np.any(mask < 0):
        raise ValueError("Expected nonempty native GT class IDs 0..4 or 255")
    pixels = mask.size
    components, _ = label(mask == 2, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(components.ravel()).astype(np.int64)
    areas[0] = 0
    border_ids = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    border_ids = border_ids[border_ids != 0]
    border_qualifying = int(np.count_nonzero(areas[border_ids] >= math.ceil(.001 * pixels)))
    mountain, rock = mask == 3, mask == 1
    mountain_count, rock_count = int(mountain.sum()), int(rock.sum())
    enough = mountain_count >= math.ceil(.005 * pixels) and rock_count >= math.ceil(.005 * pixels)
    touching = int(np.count_nonzero(rock & maximum_filter(mountain, size=17, mode="constant", cval=False))) if enough else 0
    largest = int(areas.max())
    return {"large_crater": largest >= math.ceil(.05 * pixels), "border_crater": border_qualifying > 0,
            "mountain_rock_mixed": enough and touching > 0,
            "crater_largest_component_pixels": largest, "border_crater_qualifying_component_count": border_qualifying,
            "mountain_pixels": mountain_count, "rock_pixels": rock_count,
            "rock_pixels_within_8px_of_mountain": touching, "image_pixels": pixels}


def _file_stats(record):
    result = {}
    for name in ("image", "mask"):
        stat = Path(record[name]).stat()
        result[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "inode": stat.st_ino, "device": stat.st_dev}
    return result


def _pair_statistics(luma, mask, record):
    seed_text = "42|" + "|".join(str(record.get(k, "")) for k in ("scene", "sample_id", "source_id", "augmentation_group"))
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    height, width = mask.shape
    ys = np.arange(64, height, 128) if height > 64 else np.asarray([height // 2])
    xs = np.arange(64, width, 128) if width > 64 else np.asarray([width // 2])
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    cy, cx = cy.ravel(), cx.ravel()
    hist = np.zeros((2, HISTOGRAM_BINS), dtype=np.int64)
    counts = np.zeros((4, 8, 2), dtype=np.int64)
    difference_sums = np.zeros((4, 8, 2), dtype=np.float64)
    for di, distance in enumerate(PAIR_DISTANCES):
        for oi, (dy, dx) in enumerate(PAIR_DIRECTIONS):
            ny, nx = cy + dy * distance, cx + dx * distance
            valid = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            candidates = np.flatnonzero(valid)
            if not candidates.size:
                continue
            candidates = candidates[(mask[cy[candidates], cx[candidates]] < 4)
                                    & (mask[ny[candidates], nx[candidates]] < 4)]
            if candidates.size > 32:
                candidates = np.sort(rng.choice(candidates, 32, replace=False))
            y, x, y2, x2 = cy[candidates], cx[candidates], ny[candidates], nx[candidates]
            difference = np.abs(luma[y, x] - luma[y2, x2])
            relation = (mask[y, x] != mask[y2, x2]).astype(np.int64)
            bins = np.minimum((difference * HISTOGRAM_BINS).astype(np.int64), HISTOGRAM_BINS - 1)
            hist += np.bincount(relation * HISTOGRAM_BINS + bins, minlength=2 * HISTOGRAM_BINS).reshape(2, -1)
            counts[di, oi] = np.bincount(relation, minlength=2)
            difference_sums[di, oi] = np.bincount(relation, weights=difference.astype(np.float64), minlength=2)
    return hist, counts, difference_sums


def _image_statistics(record):
    before = _file_stats(record)
    rgb, mask = read_rgb_and_mask(record)
    luma = luminance(rgb)
    surface = mask < 4
    bins = np.minimum((luma[surface] * HISTOGRAM_BINS).astype(np.int64), HISTOGRAM_BINS - 1)
    hist = np.bincount(mask[surface].astype(np.int64) * HISTOGRAM_BINS + bins,
                       minlength=4 * HISTOGRAM_BINS).reshape(4, -1)
    exact_one = np.bincount(mask[surface & (luma == 1)], minlength=4).astype(np.int64)
    pair_hist, pair_counts, pair_sums = _pair_statistics(luma, mask, record)
    if _file_stats(record) != before:
        raise RuntimeError(f"Source image/GT changed during fitting: {record['source_id']}")
    return {"hist": hist, "exact_one": exact_one, "pair_hist": pair_hist, "pair_counts": pair_counts,
            "pair_sums": pair_sums, "file_stats": before, "shape_hw": list(mask.shape),
            "sky_pixels": int(np.count_nonzero(mask == 4)), "ignored_pixels": int(np.count_nonzero(mask == 255))}


def _group_counts(hist, exact_one, low_edge, high_edge):
    dark = hist[:, :low_edge].sum(axis=1, dtype=np.int64) - (exact_one if low_edge == HISTOGRAM_BINS else 0)
    medium = hist[:, low_edge:high_edge].sum(axis=1, dtype=np.int64)
    if high_edge == HISTOGRAM_BINS:
        medium = medium - exact_one
    if low_edge == HISTOGRAM_BINS:
        medium = medium + exact_one
    counts = np.stack((dark, medium, hist.sum(axis=1, dtype=np.int64) - dark - medium), axis=1)
    return {"surface_pixels": int(counts.sum()),
            "brightness_pixels": {name: int(counts[:, i].sum()) for i, name in enumerate(BRIGHTNESS_GROUP_NAMES)},
            "by_class": {name: {"surface_pixels": int(counts[i].sum()),
                                "brightness_pixels": dict(zip(BRIGHTNESS_GROUP_NAMES, map(int, counts[i])))}
                         for i, name in enumerate(CLASS_NAMES[:4])}}


def _median_from_hist(hist):
    count = int(hist.sum())
    if not count:
        return None
    ranks = ((count + 1) // 2, (count + 2) // 2)
    indices = np.searchsorted(np.cumsum(hist, dtype=np.int64), ranks, side="left")
    return float((indices.mean() + 1) / HISTOGRAM_BINS)


def fit_training_thresholds(dataset, output_path, workers=6):
    """Fit frozen luminance tertiles on all literal native training records.

    All pixels enter int64 histograms. Image sampling, validation data and
    model predictions never enter this fit. The optional pair median is fit
    in the same pass using deterministic, bounded native-grid pixel pairs.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Frozen brightness thresholds already exist: {output_path}")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    identity = dict(dataset.identity)
    records = [dict(row) for row in dataset.records]
    if (getattr(dataset, "split", None) != "train" or identity.get("split") != "train"
            or getattr(dataset, "partial", True) or identity.get("partial", False)
            or not records or len(records) != len(dataset)
            or identity.get("image_count") != len(records) or identity.get("complete_image_count") != len(records)):
        raise ValueError("Threshold fitting requires the entire nonpartial literal training dataset")
    keys = [(r.get("source_kind", "original"), r["scene"], str(r["sample_id"])) for r in records]
    if (len(set(keys)) != len(keys) or len({r["image"] for r in records}) != len(records)
            or len({r["mask"] for r in records}) != len(records)
            or any(r.get("split", "train") != "train" or r["scene"] in ("Moon_2", "Moon_4", "Moon_9") for r in records)):
        raise ValueError("Duplicate or nontraining/test samples in brightness fitting records")
    provenance = {}
    for key in ("manifest", "selection_manifest", "source_manifest", "augmented_manifest"):
        if identity.get(key):
            path = Path(identity[key]).resolve()
            fingerprint = _sha256(path)
            if identity.get(key + "_sha256") != fingerprint:
                raise ValueError(f"Training {key} changed before brightness fitting")
            provenance[key] = {"path": str(path), "sha256": fingerprint}
    total_hist = np.zeros((4, HISTOGRAM_BINS), dtype=np.int64)
    total_one = np.zeros(4, dtype=np.int64)
    strata = {}
    source_records = []
    sky_pixels = ignored_pixels = 0
    pair_hist = np.zeros((2, HISTOGRAM_BINS), dtype=np.int64)
    pair_counts = np.zeros((4, 8, 2), dtype=np.int64)
    pair_sums = np.zeros((4, 8, 2), dtype=np.float64)
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) if workers > 1 else None
    try:
        results = executor.map(_image_statistics, records, chunksize=4) if executor else map(_image_statistics, records)
        for index, (record, result) in enumerate(zip(records, results), 1):
            hist, ones = result["hist"], result["exact_one"]
            total_hist += hist
            total_one += ones
            group = record.get("augmentation_group", "original")
            for key in (("scene", record["scene"]), ("training_group", group), ("scene_training_group", record["scene"] + "/" + group)):
                entry = strata.setdefault(key, [np.zeros_like(total_hist), np.zeros_like(total_one), 0])
                entry[0] += hist
                entry[1] += ones
                entry[2] += 1
            sky_pixels += result["sky_pixels"]
            ignored_pixels += result["ignored_pixels"]
            pair_hist += result["pair_hist"]
            pair_counts += result["pair_counts"]
            pair_sums += result["pair_sums"]
            source_records.append({k: record[k] for k in ("scene", "sample_id", "source_id", "image", "mask")})
            source_records[-1].update(training_group=group, source_kind=record.get("source_kind", "original"),
                                      shape_hw=result["shape_hw"], file_stats=result["file_stats"])
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Brightness training fit: {index}/{len(records)} images", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    surface_hist = total_hist.sum(axis=0, dtype=np.int64)
    count = int(surface_hist.sum())
    if not count:
        raise ValueError("Training set contains no GT0..3 surface pixels")
    ranks = [(count + 2) // 3, (2 * count + 2) // 3]
    edges = np.searchsorted(np.cumsum(surface_hist, dtype=np.int64), ranks, side="left") + 1
    low_edge, high_edge = map(int, edges)
    thresholds = {"low": low_edge / HISTOGRAM_BINS, "high": high_edge / HISTOGRAM_BINS}
    statistics = {"all": _group_counts(total_hist, total_one, low_edge, high_edge),
                  "by_scene": {}, "by_training_group": {}, "by_scene_training_group": {}}
    for (kind, name), (hist, ones, images) in sorted(strata.items()):
        statistics["by_" + kind][name] = {"image_count": images, **_group_counts(hist, ones, low_edge, high_edge)}
    pair_total = pair_hist.sum(axis=0, dtype=np.int64)
    distance_rows = []
    for di, distance in enumerate(PAIR_DISTANCES):
        for oi, (dy, dx) in enumerate(PAIR_DIRECTIONS):
            distance_rows.append({"chebyshev_distance_pixels": distance, "offset_dy_dx": [dy * distance, dx * distance],
                                  "euclidean_distance_pixels": distance * math.hypot(dy, dx),
                                  "same_class_pairs": int(pair_counts[di, oi, 0]), "different_class_pairs": int(pair_counts[di, oi, 1]),
                                  "same_class_mean_abs_difference": float(pair_sums[di, oi, 0] / pair_counts[di, oi, 0]) if pair_counts[di, oi, 0] else None,
                                  "different_class_mean_abs_difference": float(pair_sums[di, oi, 1] / pair_counts[di, oi, 1]) if pair_counts[di, oi, 1] else None})
    for item in provenance.values():
        if _sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("A training source manifest changed while fitting thresholds")
    document = {"schema": "lusnar_training_brightness_thresholds_v1", "status": "complete",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "fit_split": "train",
                "image_count": len(records), "dataset_identity": identity, "sources": provenance,
                "literal_records_sha256": _canonical_sha256(records), "source_records": source_records,
                "source_records_sha256": _canonical_sha256(source_records),
                "brightness_definition": {"formula": "Y=0.2126R+0.7152G+0.0722B after uint8 RGB/255",
                                          "dtype": "float32", "range": [0, 1], "srgb_linearized": False,
                                          "native_resolution": True, "resize": False, "random_augmentation": False},
                "fit_pixel_domain": {"included_gt_class_ids": [0, 1, 2, 3], "excluded_gt_class_ids": [4, 255],
                                     "surface_pixels": count, "excluded_sky_pixels": sky_pixels, "excluded_ignore_pixels": ignored_pixels},
                "thresholds": thresholds, "group_names": list(BRIGHTNESS_GROUP_NAMES),
                "group_rules": ["dark: Y < low", "medium: low <= Y < high", "bright: Y >= high"],
                "tertile_fit": {"bins": HISTOGRAM_BINS, "bin_width": 1 / HISTOGRAM_BINS,
                                "binning": "min(floor(float32 Y * 4096),4095)", "quantile_ranks_1_based": ranks,
                                "quantile_bin_upper_edge_indices": [low_edge, high_edge],
                                "threshold_approximation_max_abs_error": 1 / HISTOGRAM_BINS,
                                "threshold_rule": "Upper edge of bin containing nearest-rank 1/3 and 2/3 quantiles; ties may yield unequal or empty groups",
                                "real_group_count_rule": "Exact histogram-bin sums at frozen bin-edge thresholds with exact Y=1 endpoint correction; identical to direct pixel membership",
                                "histogram_count_dtype": "int64", "surface_histogram": surface_hist.tolist(),
                                "surface_histogram_by_class": {name: total_hist[i].tolist() for i, name in enumerate(CLASS_NAMES[:4])}},
                "statistics": statistics,
                "pair_brightness_difference": {"available": bool(pair_total.sum()), "threshold": _median_from_hist(pair_total),
                    "fit_split": "train", "seed": 42, "independent_of_training_rng": True,
                    "definition": "abs(Y(center)-Y(neighbor)); both GT labels 0..3; same/different semantic class, not instances",
                    "sampling": "Native grid starts at (64,64), stride128; tiny images use their center; for each 4 distances x8 directions choose <=32 valid centers without replacement using per-record NumPy RNG from SHA256 of seed42/scene/sample_id/source_id/group; original dataset order retained",
                    "unordered_pair_duplicates": "Grid stride128 exceeds all offsets, so reverse-pair centers cannot occur; each direction samples without replacement",
                    "distances_are_chebyshev": True, "distance_direction_statistics": distance_rows,
                    "bins": HISTOGRAM_BINS, "threshold_approximation_max_abs_error": 1 / HISTOGRAM_BINS,
                    "median_rule": "Average upper-bin edges of the two central nearest-rank order statistics",
                    "pair_count": int(pair_total.sum()), "same_class_pair_count": int(pair_hist[0].sum()),
                    "different_class_pair_count": int(pair_hist[1].sum()), "histogram": pair_total.tolist(),
                    "histogram_same_class": pair_hist[0].tolist(), "histogram_different_class": pair_hist[1].tolist(),
                    "same_class_median_abs_difference": _median_from_hist(pair_hist[0]),
                    "different_class_median_abs_difference": _median_from_hist(pair_hist[1])},
                "image_tagging_rules": tagging_rules(), "validation_used_for_fit": False, "test_used_for_fit": False,
                "sampling_with_replacement": False, "all_literal_training_images_used": True,
                "individual_image_content_hashes_computed": False,
                "file_provenance_note": "All image/mask native paths, sizes, mtimes, inode/device and shape recorded; file stats checked before/after each read; authoritative manifests and record ordering content-hashed",
                "source_files_modified": False, "implementation_sha256": _sha256(__file__)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output_path.parent, prefix=".brightness-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output_path)
    finally:
        temporary.unlink()
    print(f"Frozen brightness thresholds: {thresholds}; pair difference median={document['pair_brightness_difference']['threshold']}", flush=True)
    return document
