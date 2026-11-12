"""Reproduce train-only GT-context supplements using the original NAS algorithm.

Run with ``PYTHONPATH=code python -m lunarseg.augmentation --config ...``.
The original selection and transform modules are imported only after their
configured SHA256 values have been verified. Existing data is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import PIL
from PIL import Image


CLASSES = ("regolith", "rock", "crater", "mountain", "sky")
GROUPS = ("crater", "mountain_rock")
OLD_TRAIN_SCENES = frozenset(("Moon_1", "Moon_3", "Moon_6", "Moon_7", "Moon_8"))
_CORE = _LEGACY = None


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def report(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def runtime_versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "Pillow": PIL.__version__, "opencv": cv2.__version__}


def load_legacy(directory, expected):
    """Load unmodified original functions, with hashes checked before execution."""
    global _CORE, _LEGACY
    directory = Path(directory).resolve()
    for name in ("selection_core.py", "build_dataset.py"):
        require(name in expected and sha256(directory / name) == expected[name],
                f"Original augmentation source SHA256 differs: {directory / name}")
    core_spec = importlib.util.spec_from_file_location("_lusnar_original_selection_core", directory / "selection_core.py")
    core = importlib.util.module_from_spec(core_spec)
    sys.modules[core_spec.name] = core
    core_spec.loader.exec_module(core)
    # build_dataset imports this exact original module by its historic name.
    prior = sys.modules.get("selection_core")
    sys.modules["selection_core"] = core
    try:
        spec = importlib.util.spec_from_file_location("_lusnar_original_build_dataset", directory / "build_dataset.py")
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)
    finally:
        if prior is None:
            sys.modules.pop("selection_core", None)
        else:
            sys.modules["selection_core"] = prior
    require(tuple(core.CLASS_NAMES) == CLASSES, "Original augmentation class order differs")
    cv2.setNumThreads(1)
    _CORE, _LEGACY = core, legacy
    return core, legacy


def _worker_init(directory, expected):
    load_legacy(directory, expected)


def crop_identity(row):
    source_id = f"{row['source_scene']}/{row['source_sample_id']}"
    crop = row.get("crop_xyxy")
    require(isinstance(crop, list) and len(crop) == 4 and all(type(v) is int for v in crop),
            f"Invalid crop coordinates: {source_id}")
    x0, y0, x1, y1 = crop
    require(0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024 and x1 - x0 == y1 - y0,
            f"Crop is outside source or is not square: {source_id}")
    flip = row.get("horizontal_flip", row.get("flip"))
    require(type(flip) is bool and row.get("flip", flip) == flip, f"Invalid/inconsistent flip: {source_id}")
    return source_id, tuple(crop), flip


def validate_family_split(custom, original):
    require(custom.get("schema") == "lusnar_random_split_v1", "Expected saved random80 family split")
    family = custom["original_split"]
    require(canonical_sha256(family) == custom["original_split_sha256"], "Original family lists were modified")
    require(canonical_sha256(custom["splits"]) == custom["splits_sha256"], "Literal custom split rows were modified")
    for split in ("train", "val"):
        require(canonical_sha256(custom["splits"][split]) == custom["ordered_samples_sha256"][split],
                f"Literal {split} row hash differs")
    train_list, val_list = family["train_source_ids"], family["val_source_ids"]
    train, val = set(train_list), set(val_list)
    require(len(train) == len(train_list) and len(val) == len(val_list) and not train & val,
            "Duplicate or leaking source families")
    source_ids = {f"{r['scene']}/{r['sample_id']}" for r in original["samples"]}
    require(train | val == source_ids and len(source_ids) == len(original["samples"]),
            "Family split does not partition the exact original pool")
    require(len(train) == 4 * len(source_ids) // 5, "Original family split is not fixed 4:1")
    for split, allowed in (("train", train), ("val", val)):
        for row in custom["splits"][split]:
            require(row["source_id"] in allowed and row["scene"] not in ("Moon_2", "Moon_4", "Moon_9"),
                    f"Validation/test source leakage in saved {split} rows")
    return train, val


def _absolute_old_rows(manifest, root):
    rows = []
    for original in manifest["samples"]:
        row = dict(original)
        for key in ("image", "mask", "label", "preview", "source_image", "source_mask", "source_label", "source_manifest"):
            if row.get(key):
                path = Path(row[key])
                row[key] = str((path if path.is_absolute() else root / path).resolve())
        rows.append(row)
    return rows


def prepare_inputs(config):
    for name in ("seed", "workers", "crater_count", "mountain_rock_count"):
        require(type(config.get(name)) is int and config[name] >= (1 if name == "workers" else 0),
                f"Invalid integer configuration: {name}")
    require(config["crater_count"] + config["mountain_rock_count"] > 0, "Supplement must contain samples")
    for key, hash_key in (("custom_split", "custom_split_sha256"), ("existing_manifest", "existing_manifest_sha256")):
        require(sha256(config[key]) == config[hash_key], f"Protected manifest SHA256 differs: {config[key]}")
    require(runtime_versions() == config["runtime_versions"], "Runtime versions differ from the saved reproduction configuration")
    core, legacy = load_legacy(config["legacy_code_dir"], config["legacy_code_sha256"])
    custom = read_json(config["custom_split"])
    original_path = Path(custom["source_manifest"]).resolve()
    require(sha256(original_path) == custom["source_manifest_sha256"], "Original source manifest SHA256 differs")
    original = read_json(original_path)
    existing = read_json(config["existing_manifest"])
    require(existing.get("class_names") == original.get("class_names") == list(CLASSES), "Five-class order differs")
    require(existing.get("validation_or_test_used") is False and existing.get("state") == "verified",
            "Original augmentation manifest is not verified train-only data")
    require(existing.get("source_manifest_sha256") == custom["source_manifest_sha256"]
            and Path(existing["source_manifest"]).resolve() == original_path, "Augmentation source provenance differs")
    require(existing["selection_config"] == asdict(core.SelectionConfig()), "Original selection configuration changed")
    train, val = validate_family_split(custom, original)
    old_rows = _absolute_old_rows(existing, Path(config["existing_manifest"]).resolve().parent)
    # The original builder's transient ROI/GT caches may no longer exist. Its
    # verified manifest retains the exact original source metadata and boxes;
    # use those literal boxes and revalidate each against its real source GT.
    source_keys = ("source_manifest_index", "source_train_index", "source_scene", "source_sample_id",
                   "source_image", "source_mask", "source_label", "source_mask_sha256",
                   "source_gt_pixel_counts", "candidate_boxes")
    by_source = {}
    for row in old_rows:
        sid = f"{row['source_scene']}/{row['source_sample_id']}"
        if sid not in train:
            continue
        source = {key: row[key] for key in source_keys}
        require(source["source_scene"] in OLD_TRAIN_SCENES and sid not in val, "Supplement source leaks validation/test")
        original_row = original["samples"][source["source_manifest_index"]]
        require(original_row["split"] == "train" and original_row["scene"] == source["source_scene"]
                and str(original_row["sample_id"]) == str(source["source_sample_id"]), "Recovered original source identity differs")
        for key in ("image", "mask"):
            path = Path(original_row[key])
            expected = (path if path.is_absolute() else original_path.parent / path).resolve()
            require(Path(source[f"source_{key}"]).resolve() == expected, "Recovered original source path differs")
        require(sid not in by_source or by_source[sid] == source, "Existing rows disagree about original source metadata")
        by_source[sid] = source
    sources = sorted(by_source.values(), key=lambda source: source["source_manifest_index"])
    identities = [crop_identity(row) for row in old_rows]
    require(len(identities) == len(set(identities)), "Existing augmentation crop identities are not unique")
    require(len({r["sample_id"] for r in old_rows}) == len(old_rows), "Existing augmentation sample IDs repeat")
    protected = {str(Path(config[key]).resolve()): config[hash_key] for key, hash_key in
                 (("custom_split", "custom_split_sha256"), ("existing_manifest", "existing_manifest_sha256"))}
    protected[str(original_path)] = custom["source_manifest_sha256"]
    protected.update({str((Path(config["legacy_code_dir"]) / name).resolve()): digest
                      for name, digest in config["legacy_code_sha256"].items()})
    return core, legacy, sources, old_rows, custom, protected


def _propose_source(task):
    source, seed, config_dict, max_candidates = task
    config = _CORE.SelectionConfig(**config_dict)
    groups = [g for g in _LEGACY.geometric_groups(source, config) if g in GROUPS]
    if not groups:
        return {"source": source, "groups": {}}
    require(sha256(source["source_mask"]) == source["source_mask_sha256"], "Source GT SHA256 changed")
    with Image.open(source["source_mask"]) as image:
        require(image.mode == "L" and image.size == (1024, 1024), "Expected original 1024-square L-mode GT")
        mask = np.asarray(image, dtype=np.uint8).copy()
    prepared = _CORE.prepare_mask(mask)
    require(dict(zip(CLASSES, prepared.source_counts)) == source["source_gt_pixel_counts"], "Source GT counts changed")
    candidates = {group: _CORE.propose_from_prepared(
        prepared, source["candidate_boxes"], group,
        _LEGACY.seed_for(seed, source["source_manifest_index"], group), max_candidates=max_candidates, config=config)
        for group in groups}
    return {"source": source, "groups": candidates}


def allocate_supplement(pool, old_rows, counts, seed, legacy):
    """Use original balanced source order and caps; never relax crop eligibility."""
    forbidden = {crop_identity(row) for row in old_rows}
    used, new_used = defaultdict(list), defaultdict(list)
    for row in old_rows:
        used[crop_identity(row)[0]].append(row)
    chosen, capacities = [], {}
    for group in GROUPS:
        order = legacy.source_order(pool, group)
        capacities[group] = {"candidate_sources": len(order),
                             "candidate_crops": sum(len(pool[index]["groups"][group]) for index in order)}
        selected = 0
        for round_index in range(legacy.ALLOCATION["max_per_source_per_group"]):
            for index in order:
                if selected == counts[group]:
                    break
                source = pool[index]["source"]
                sid = f"{source['source_scene']}/{source['source_sample_id']}"
                if len(used[sid]) >= legacy.ALLOCATION["max_per_source_global"]:
                    continue
                if sum(r["primary_group"] == group for r in new_used[sid]) >= legacy.ALLOCATION["max_per_source_per_group"]:
                    continue
                preferred = bool(legacy.seed_for(seed, index, group, round_index) % 2)
                sample = None
                for candidate in pool[index]["groups"][group]:
                    for flip in (preferred, not preferred):
                        identity = sid, tuple(candidate["crop_xyxy"]), flip
                        if identity in forbidden:
                            continue
                        if any(crop_identity(previous)[2] == flip
                               and legacy.crop_iou(previous["crop_xyxy"], candidate["crop_xyxy"])
                               > legacy.ALLOCATION["max_same_flip_crop_iou"] for previous in used[sid]):
                            continue
                        sample = {**source, **candidate, "primary_group": group, "split": "train",
                                  "flip": flip, "horizontal_flip": flip,
                                  "augmentation_seed": legacy.seed_for(seed, index, group, round_index, candidate["crop_xyxy"], flip),
                                  "selection_seed": seed}
                        break
                    if sample is not None:
                        break
                if sample is None:
                    continue
                selected += 1
                sample["sample_id"] = sample["aug_id"] = f"random80_s{seed}_{group}_{selected:04d}"
                chosen.append(sample)
                forbidden.add(crop_identity(sample))
                used[sid].append(sample)
                new_used[sid].append(sample)
            if selected == counts[group]:
                break
        capacities[group]["selected"] = selected
        require(selected == counts[group], f"Insufficient fixed-rule {group} capacity; constraints were not relaxed: {capacities}")
    return chosen, capacities


def make_plan(config):
    started = time.monotonic()
    core, legacy, sources, old_rows, custom, protected = prepare_inputs(config)
    selection = core.SelectionConfig()
    sources = [s for s in sources if any(g in GROUPS for g in legacy.geometric_groups(s, selection))]
    report("candidate_proposals", eligible_oldtrain_newtrain_sources=len(sources), workers=config["workers"])
    tasks = [(s, config["seed"], asdict(selection), legacy.ALLOCATION["candidates_per_source_per_group"]) for s in sources]
    pool = {}
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=_worker_init,
                             initargs=(config["legacy_code_dir"], config["legacy_code_sha256"])) as executor:
        for count, record in enumerate(executor.map(_propose_source, tasks, chunksize=4), 1):
            if any(record["groups"].values()):
                pool[record["source"]["source_manifest_index"]] = record
            if count % 100 == 0 or count == len(tasks):
                report("candidate_progress", completed_sources=count, total_sources=len(tasks),
                       feasible_sources=len(pool), elapsed_seconds=round(time.monotonic() - started, 1))
    counts = {g: config[f"{g}_count"] for g in GROUPS}
    rows, capacities = allocate_supplement(pool, old_rows, counts, config["seed"], legacy)
    for path, expected in protected.items():
        require(sha256(path) == expected, f"Protected input changed while planning: {path}")
    plan = {"schema": "lusnar_random80_supplement_plan_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config, "configuration_sha256": canonical_sha256(config),
            "module_sha256": sha256(__file__), "protected_inputs": protected,
            "selection_config": asdict(selection),
            "allocation_config": {**legacy.ALLOCATION, "per_group": counts, "selection_seed": config["seed"],
                                  "same_flip_iou_exclusion_includes_existing": True},
            "source_family_policy": "original oldtrain intersect saved random80 train_source_ids; no val/test sources",
            "source_metadata_basis": "literal source metadata/candidate boxes from the verified existing augmentation manifest, rechecked against original source GT",
            "source_family_sha256": canonical_sha256(custom["original_split"]),
            "selected_identities_sha256": canonical_sha256([crop_identity(r) for r in rows]),
            "samples_sha256": canonical_sha256(rows),
            "counts": counts, "capacities": capacities, "samples": rows,
            "planning_seconds": time.monotonic() - started}
    return plan


def _check_output_fractions(fractions, group, config):
    tolerance = 1e-4  # Identical nearest-neighbor output acceptance tolerance to original export_source.
    require(fractions["regolith"] >= config.min_regolith_fraction - tolerance
            and fractions["regolith"] + fractions["sky"] >= config.min_background_fraction - tolerance
            and fractions["crater"] <= config.max_crater_fraction + tolerance, "Resized crop violates original background constraints")
    thresholds = {"crater": {"crater": config.min_crater_fraction},
                  "mountain_rock": {"mountain": config.min_mountain_fraction, "rock": config.min_rock_fraction}}[group]
    require(all(fractions[name] >= minimum - tolerance for name, minimum in thresholds.items()),
            f"Resized crop violates original {group} fractions")


def _export_one(sample, output, source_manifest, source_sha):
    require(sha256(sample["source_mask"]) == sample["source_mask_sha256"], "Source GT changed during export")
    image_sha = sha256(sample["source_image"])
    with Image.open(sample["source_image"]) as source:
        rgb = source.convert("RGB")
    with Image.open(sample["source_mask"]) as source:
        require(source.mode == "L", "Source mask is not an ID mask")
        gt = source.copy()
    require(rgb.size == gt.size == (1024, 1024), "Source image/mask dimensions changed")
    image, mask = _LEGACY.transform(rgb, gt, sample["crop_xyxy"], sample["flip"])
    pixels = np.asarray(mask)
    require(mask.mode == "L" and int(pixels.max()) <= 4, "Output mask must retain L-mode class IDs 0..4")
    counts = np.bincount(pixels.ravel(), minlength=5).tolist()
    fractions = dict(zip(CLASSES, [value / pixels.size for value in counts]))
    _check_output_fractions(fractions, sample["primary_group"], _CORE.SelectionConfig())
    image_data, mask_data = _LEGACY.png_bytes(image), _LEGACY.png_bytes(mask)
    relative = Path("groups") / sample["primary_group"]
    image_path, mask_path = output / relative / "images" / (sample["sample_id"] + ".png"), output / relative / "labels" / (sample["sample_id"] + ".png")
    for path, data in ((image_path, image_data), (mask_path, mask_data)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    image_pixels, mask_pixels = image.tobytes(), mask.tobytes()
    return {**sample, "source_dataset": "LuSNAR", "source_manifest": source_manifest,
            "source_manifest_sha256": source_sha, "source_image_sha256": image_sha,
            "source_shape_hw": [1024, 1024], "output_shape_hw": [1024, 1024],
            "image": str(image_path), "mask": str(mask_path), "label": str(mask_path),
            "image_sha256": hashlib.sha256(image_data).hexdigest(), "mask_sha256": hashlib.sha256(mask_data).hexdigest(),
            "image_pixel_sha256": hashlib.sha256(image_pixels).hexdigest(),
            "mask_pixel_sha256": hashlib.sha256(mask_pixels).hexdigest(),
            "image_mask_pair_sha256": hashlib.sha256(image_pixels + mask_pixels).hexdigest(),
            "gt_pixel_counts": dict(zip(CLASSES, counts)), "output_class_fractions": fractions}


def _verify_one(row):
    for key in ("image", "mask", "source_image", "source_mask"):
        require(sha256(row[key]) == row[key + "_sha256"], f"Export/source file SHA256 changed: {row[key]}")
    with Image.open(row["image"]) as source:
        require(source.mode == "RGB" and source.size == (1024, 1024), "Invalid exported RGB")
        image = source.copy()
    with Image.open(row["mask"]) as source:
        require(source.mode == "L" and source.size == (1024, 1024), "Invalid exported class-ID mask")
        mask = source.copy()
    with Image.open(row["source_image"]) as source:
        rgb = source.convert("RGB")
    with Image.open(row["source_mask"]) as source:
        gt = source.copy()
    expected_image, expected_mask = _LEGACY.transform(rgb, gt, row["crop_xyxy"], row["flip"])
    require(image.tobytes() == expected_image.tobytes() and mask.tobytes() == expected_mask.tobytes(),
            f"Literal source/crop/flip replay differs: {row['sample_id']}")
    require(hashlib.sha256(image.tobytes() + mask.tobytes()).hexdigest() == row["image_mask_pair_sha256"],
            "Export pixel-pair SHA256 differs")
    array = np.asarray(mask)
    require(int(array.max()) <= 4, "Exported class IDs outside 0..4")
    counts = dict(zip(CLASSES, np.bincount(array.ravel(), minlength=5).tolist()))
    require(counts == row["gt_pixel_counts"], "Exported class counts differ")
    _check_output_fractions({k: v / array.size for k, v in counts.items()}, row["primary_group"], _CORE.SelectionConfig())
    return row["sample_id"]


def export_plan(plan, config):
    started = time.monotonic()
    require(plan.get("schema") == "lusnar_random80_supplement_plan_v1"
            and plan["configuration_sha256"] == canonical_sha256(config), "Plan configuration differs")
    require(plan["module_sha256"] == sha256(__file__), "Supplement code changed after planning")
    require(plan["selected_identities_sha256"] == canonical_sha256([crop_identity(r) for r in plan["samples"]]),
            "Plan crop identities were modified")
    require(plan["samples_sha256"] == canonical_sha256(plan["samples"]), "Plan sample metadata was modified")
    core, legacy, sources, old_rows, custom, protected = prepare_inputs(config)
    require(plan["protected_inputs"] == protected, "Plan protected inputs differ")
    allowed = {f"{s['source_scene']}/{s['source_sample_id']}": s for s in sources}
    identities = {crop_identity(row) for row in old_rows}
    expected_counts = {g: config[f"{g}_count"] for g in GROUPS}
    require(Counter(r["primary_group"] for r in plan["samples"]) == Counter(expected_counts), "Plan group counts differ")
    for row in plan["samples"]:
        identity = crop_identity(row)
        require(identity[0] in allowed and identity not in identities, "Planned crop duplicates existing data or leaks split")
        identities.add(identity)
        for key, value in allowed[identity[0]].items():
            require(row[key] == value, f"Planned source metadata changed: {identity[0]}/{key}")
        require(row["primary_group"] in row["eligible_groups"] and row["metrics"]["new_target_truncation"] is False,
                "Planned crop has not passed original eligibility")
    output = Path(config["output"]).resolve()
    require(not output.exists(), f"Never overwrite an existing supplement directory: {output}")
    require(not any(Path(p).resolve().is_relative_to(output) for p in protected), "Output contains protected original inputs")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"state": "building", "expected_images": len(plan["samples"])})
    try:
        source_manifest, source_sha = str(Path(custom["source_manifest"]).resolve()), custom["source_manifest_sha256"]
        existing_pairs = {row["image_mask_pair_sha256"] for row in old_rows}
        finished = []
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            for count, row in enumerate(executor.map(lambda r: _export_one(r, output, source_manifest, source_sha), plan["samples"]), 1):
                require(row["image_mask_pair_sha256"] not in existing_pairs, "Duplicate RGB/mask pixel pair found; output retained as failed")
                existing_pairs.add(row["image_mask_pair_sha256"])
                finished.append(row)
                if count % 50 == 0 or count == len(plan["samples"]):
                    write_json(output / "status.json", {"state": "building", "completed_images": count,
                                                       "expected_images": len(plan["samples"])})
                    report("export_progress", completed_images=count, total_images=len(plan["samples"]))
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            for count, _ in enumerate(executor.map(_verify_one, finished), 1):
                if count % 100 == 0 or count == len(finished):
                    report("literal_replay_verification", completed_images=count, total_images=len(finished))
        for path, expected in protected.items():
            require(sha256(path) == expected, f"Protected input changed during export: {path}")
        provenance = output / "provenance"
        provenance.mkdir()
        for name in ("build_dataset.py", "selection_core.py"):
            shutil.copy2(Path(config["legacy_code_dir"]) / name, provenance / name)
        shutil.copy2(__file__, provenance / "augmentation.py")
        write_json(provenance / "configuration.json", config)
        write_json(provenance / "plan.json", plan)
        verification = {"status": "passed", "new_images_replayed_from_source_crop_flip": len(finished),
                        "rgb_shape_mode": "1024x1024 RGB PNG", "gt_shape_mode_ids": "1024x1024 L PNG, IDs 0..4",
                        "new_family_validation_test_leaks": 0, "duplicate_crop_identities": 0,
                        "duplicate_rgb_mask_pixel_pairs": 0, "protected_input_hashes_unchanged": True,
                        "old_samples_inherited_from_verified_manifest": len(old_rows),
                        "old_samples_reexported": False, "existing_manifest_sha256": config["existing_manifest_sha256"]}
        shared = {"state": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "class_names": list(CLASSES), "palette_rgb": legacy.PALETTE,
                  "source_manifest": source_manifest, "source_manifest_sha256": source_sha,
                  "split": "train", "only_original_training_sources": True, "validation_or_test_used": False,
                  "output_root": str(output), "custom_split_manifest": str(Path(config["custom_split"]).resolve()),
                  "custom_split_manifest_sha256": config["custom_split_sha256"], "selection_seed": config["seed"],
                  "selection_config": plan["selection_config"], "allocation_config": plan["allocation_config"],
                  "source_metadata_basis": plan["source_metadata_basis"],
                  "build_plan_sha256": sha256(config["plan"]),
                  "planned_samples_sha256": plan["samples_sha256"],
                  "algorithm_code_sha256": {**config["legacy_code_sha256"], "augmentation.py": sha256(__file__)},
                  "protected_inputs": protected, "runtime_versions": runtime_versions(),
                  "encoding": {"rgb": "RGB PNG", "gt": "L PNG IDs 0..4", "image_interpolation": "Pillow BILINEAR",
                               "mask_interpolation": "Pillow NEAREST", "order": ["crop", "resize_1024", "horizontal_flip"],
                               "exposure_or_color_augmented": False, "output_class_fraction_tolerance": 1e-4},
                  "verification": verification}
        supplement = {**shared, "schema": "lusnar_random80_supplement_v1", "image_count": len(finished), "samples": finished}
        merged_rows = old_rows + finished
        require(len({r["sample_id"] for r in merged_rows}) == len(merged_rows), "Merged sample IDs repeat")
        merged = {**shared, "schema": "lusnar_gt_context_merged_v1", "image_count": len(merged_rows),
                  "existing_manifest": str(Path(config["existing_manifest"]).resolve()),
                  "existing_manifest_sha256": config["existing_manifest_sha256"],
                  "supplement_manifest": str(output / "manifest.json"),
                  "group_counts": dict(Counter(row["primary_group"] for row in merged_rows)), "samples": merged_rows}
        write_json(output / "verification.json", verification)
        write_json(output / "manifest.json", supplement)
        write_json(output / "merged_manifest.json", merged)
        write_json(output / "status.json", {"state": "verified", "completed_images": len(finished),
                                           "merged_images": len(merged_rows), "elapsed_seconds": time.monotonic() - started})
        report("complete", output=str(output), new_images=len(finished), merged_images=len(merged_rows),
               merged_manifest=str(output / "merged_manifest.json"), merged_manifest_sha256=sha256(output / "merged_manifest.json"))
        return merged
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Write an immutable literal crop plan without exporting images")
    mode.add_argument("--export-only", action="store_true", help="Replay the existing immutable plan; never resample")
    args = parser.parse_args()
    config = read_json(args.config)
    plan_path = Path(config["plan"]).resolve()
    if args.export_only:
        plan = read_json(plan_path)
    else:
        require(not plan_path.exists(), f"Never overwrite an existing plan: {plan_path}")
        plan = make_plan(config)
        write_json(plan_path, plan)
        report("plan_saved", plan=str(plan_path), counts=plan["counts"], capacities=plan["capacities"])
    if not args.plan_only:
        export_plan(plan, config)


if __name__ == "__main__":
    main()
