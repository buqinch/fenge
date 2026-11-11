"""Save immutable training selections or source-family-isolated random splits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile

from PIL import Image

from lunarseg.data import CLASS_NAMES, DEFAULT_MANIFEST, LunarDataset, sha256_file

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMENTED = "/media/nas/lzc/lusnar_augmentation/lusnar_gt_context_6000_20261110/manifest.json"
RANDOM_SPLIT_PROTOCOL = "random_original_source_family_4_to_1"
RANDOM_SPLIT_GROUPS = ("crater", "mountain_rock")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def normalized_record(record, split, source_kind):
    row = {"split": split, "source_kind": source_kind,
           "scene": record["scene"], "sample_id": str(record["sample_id"]),
           "source_id": record["source_id"],
           "image": str(Path(record["image"]).resolve()), "mask": str(Path(record["mask"]).resolve())}
    if source_kind == "augmented":
        row["augmentation_group"] = record["augmentation_group"]
    return row


def split_statistics(records):
    groups = Counter(row.get("augmentation_group", "original") for row in records)
    return {"image_count": len(records), "unique_source_count": len({row["source_id"] for row in records}),
            "group_counts": dict(sorted(groups.items())),
            "scene_counts": dict(sorted(Counter(row["scene"] for row in records).items())),
            "group_scene_counts": {group: dict(sorted(Counter(row["scene"] for row in records
                                                              if row.get("augmentation_group", "original") == group).items()))
                                   for group in sorted(groups)}}


def validate_id_masks(splits):
    """Read existing ID masks without copying or changing any source files."""
    checked = 0
    for rows in splits.values():
        for row in rows:
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            with Image.open(row["mask"]) as mask:
                if mask.mode != "L":
                    raise ValueError(f"Expected L-mode class-ID mask, got {mask.mode}: {row['mask']}")
                minimum, maximum = mask.getextrema()
                if minimum < 0 or maximum >= len(CLASS_NAMES):
                    raise ValueError(f"Mask contains values outside class IDs 0..4: {row['mask']}")
            checked += 1
    return {"checked_mask_count": checked, "required_mode": "L", "class_id_range": [0, 4],
            "source_files_modified": False, "raw_palette_labels_used": False}


def generate_random_split(manifest: Path, augmented_manifest: Path,
                          augmentation_counts: dict[str, int], seed: int = 42,
                          *, expected_original_count: int = 8002) -> dict:
    """Partition originals first, then add only augmentations of new train IDs.

    Literal lists and hashes are authoritative on replay. The non-CLI expected
    count argument permits small synthetic protocol tests; real generation is
    fixed to the complete 8002-original source manifest.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Random split seed must be a nonnegative integer")
    if (set(augmentation_counts) != set(RANDOM_SPLIT_GROUPS)
            or any(type(count) is not int or count < 1 for count in augmentation_counts.values())):
        raise ValueError("Supply positive counts for exactly crater and mountain_rock")
    manifest, augmented_manifest = Path(manifest).resolve(), Path(augmented_manifest).resolve()
    source_sha, augmented_sha = sha256_file(manifest), sha256_file(augmented_manifest)
    original_train = LunarDataset(manifest, "train")
    original_val = LunarDataset(manifest, "val")
    originals = [normalized_record(row, "train", "original")
                 for row in [*original_train.records, *original_val.records]]
    originals.sort(key=lambda row: row["source_id"])
    source_ids = [row["source_id"] for row in originals]
    if len(originals) != expected_original_count or len(originals) < 2:
        raise ValueError(f"Expected {expected_original_count} complete originals, found {len(originals)}")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Original source_id values must be unique")
    pairs = {(row["image"], row["mask"]) for row in originals}
    if len(pairs) != len(originals):
        raise ValueError("Original pool contains repeated image/mask files")

    augmented_source = json.loads(augmented_manifest.read_text())
    if (augmented_source.get("source_manifest_sha256") != source_sha
            or not augmented_source.get("source_manifest")
            or Path(augmented_source["source_manifest"]).resolve() != manifest):
        raise ValueError("Augmentation source manifest path/SHA256 differs from the original pool")
    legal_augmented = LunarDataset(manifest, "train", augmented_manifest=augmented_manifest,
                                  augmentation_groups=list(RANDOM_SPLIT_GROUPS), train_source="combined")
    all_augmented = [normalized_record(row, "train", "augmented") for row in legal_augmented.records
                     if "augmentation_group" in row]
    keys = [(row["scene"], row["sample_id"]) for row in all_augmented]
    if len(keys) != len(set(keys)):
        raise ValueError("Augmentation pool contains repeated sample identities")
    augmented_pairs = {(row["image"], row["mask"]) for row in all_augmented}
    if len(augmented_pairs) != len(all_augmented) or augmented_pairs & pairs:
        raise ValueError("Augmentation pool contains repeated original/augmented image-mask files")

    rng = random.Random(seed)
    rng.shuffle(originals)
    train_size = len(originals) * 4 // 5
    train_originals = originals[:train_size]
    val_originals = [{**row, "split": "val"} for row in originals[train_size:]]
    original_split = {"train_source_ids": [row["source_id"] for row in train_originals],
                      "val_source_ids": [row["source_id"] for row in val_originals]}
    train_ids, val_ids = set(original_split["train_source_ids"]), set(original_split["val_source_ids"])
    pools = {group: sorted([row for row in all_augmented
                            if row["augmentation_group"] == group and row["source_id"] in train_ids],
                           key=lambda row: (row["scene"], row["sample_id"]))
             for group in RANDOM_SPLIT_GROUPS}
    pool_counts = Counter(row["augmentation_group"] for row in all_augmented)
    excluded_counts = {group: pool_counts[group] - len(pools[group]) for group in RANDOM_SPLIT_GROUPS}
    train_rows = list(train_originals)
    for group in RANDOM_SPLIT_GROUPS:
        if augmentation_counts[group] > len(pools[group]):
            raise ValueError(f"Requested {augmentation_counts[group]} {group} augmentations, but only {len(pools[group])}"
                             " have new-train sources; validation families must remain excluded")
        train_rows.extend(rng.sample(pools[group], augmentation_counts[group]))
    rng.shuffle(train_rows)
    splits = {"train": train_rows, "val": val_originals}
    if ({row["source_id"] for row in train_rows} != train_ids or train_ids & val_ids
            or any(row["source_kind"] != "original" for row in val_originals)):
        raise RuntimeError("Random split violates source-family isolation")
    mask_validation = validate_id_masks(splits)
    # Source manifests must not change during potentially lengthy mask checks.
    if sha256_file(manifest) != source_sha or sha256_file(augmented_manifest) != augmented_sha:
        raise RuntimeError("A source manifest changed during random split generation")
    return {
        "schema": "lusnar_random_split_v1", "split_protocol": RANDOM_SPLIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "generation_algorithm_version": "random_original_family_split_v1",
        "sampling_algorithm": "One Python random.Random(seed): sort originals by source_id; shuffle; first floor(N*4/5) train;"
                              " remaining suffix val; filter augmented families to train; in crater,mountain_rock order"
                              " sort each pool by (scene,sample_id), sample without replacement; shuffle all train rows",
        "sampling_order": list(RANDOM_SPLIT_GROUPS), "train_fraction": {"numerator": 4, "denominator": 5},
        "python_version": platform.python_version(), "selector_sha256": sha256_file(Path(__file__)),
        "source_manifest": str(manifest), "source_manifest_sha256": source_sha,
        "augmented_manifest": str(augmented_manifest), "augmented_manifest_sha256": augmented_sha,
        "class_names": list(CLASS_NAMES), "test_scenes_used": False,
        "original_pool_definition": "Complete former training originals plus complete Moon_5 originals; independent Moon_2/4/9 test is excluded",
        "counts": {"original_pool": len(originals), "train_original": len(train_originals),
                   "val_original": len(val_originals), "train_augmented": sum(augmentation_counts.values()),
                   "train_total": len(train_rows), "val_total": len(val_originals)},
        "augmentation_counts": {group: augmentation_counts[group] for group in RANDOM_SPLIT_GROUPS},
        "candidate_counts": {"original": len(originals),
                             "augmented_before_family_filter": {group: pool_counts[group] for group in RANDOM_SPLIT_GROUPS},
                             "augmented_after_family_filter": {group: len(pools[group]) for group in RANDOM_SPLIT_GROUPS}},
        "excluded_augmented_counts": excluded_counts,
        "exclusion_reason": "Augmentation source_id belongs to the new validation split",
        "excluded_augmented_source_ids": sorted({row["source_id"] for row in all_augmented if row["source_id"] in val_ids}),
        "unselected_eligible_augmented_counts": {group: len(pools[group]) - augmentation_counts[group]
                                                 for group in RANDOM_SPLIT_GROUPS},
        "split_statistics": {split: split_statistics(rows) for split, rows in splits.items()},
        "mask_validation": mask_validation,
        "original_pool_source_ids_sha256": canonical_sha256(sorted(source_ids)),
        "original_split": original_split, "original_split_sha256": canonical_sha256(original_split),
        "splits": splits, "splits_sha256": canonical_sha256(splits),
        "ordered_samples_sha256": {split: canonical_sha256(rows) for split, rows in splits.items()},
    }


def generate_selection(manifest: Path, augmented_manifest: Path, counts: dict[str, int], seed: int) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("Selection seed must be a nonnegative integer")
    if (not counts or "original" not in counts or
            not set(counts).issubset({"original", "crater", "mountain_rock", "mixed"}) or
            any(type(count) is not int or count < 1 for count in counts.values())):
        raise ValueError("Supply positive counts for original data and the requested augmentation groups")
    groups = sorted(set(counts) - {"original"})
    if not groups:
        raise ValueError("Select at least one augmentation group")
    candidates = LunarDataset(manifest, split="train", augmented_manifest=augmented_manifest,
                              augmentation_groups=groups, train_source="combined")
    pools = {group: [] for group in counts}
    for candidate in candidates.records:
        group = candidate.get("augmentation_group", "original")
        pools[group].append(candidate)
    for pool in pools.values():
        pool.sort(key=lambda row: (row["scene"], str(row["sample_id"])))
    rng = random.Random(seed)
    selected = []
    # Save the literal ordered list: future replay does not depend on random.sample
    # remaining unchanged between Python versions.
    for group in ["original", *groups]:
        if counts[group] > len(pools[group]):
            raise ValueError(f"Requested {counts[group]} {group} samples from only {len(pools[group])} candidates")
        for record in rng.sample(pools[group], counts[group]):
            selected.append({**record, "source_kind": "original" if group == "original" else "augmented"})
    rng.shuffle(selected)
    keys = [(row["source_kind"], row["scene"], str(row["sample_id"])) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Selection contains repeated sample identities")
    canonical_samples = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "lusnar_training_selection_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sampling_algorithm": "Python random.Random(seed); sorted original/group pools; random.sample without replacement; final shuffle",
        "sampling_order": ["original", *groups],
        "python_version": platform.python_version(),
        "selector_sha256": sha256_file(Path(__file__)),
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": sha256_file(manifest),
        "augmented_manifest": str(augmented_manifest.resolve()),
        "augmented_manifest_sha256": sha256_file(augmented_manifest),
        "class_names": list(CLASS_NAMES),
        "split": "train", "validation_or_test_used": False,
        "counts": counts, "image_count": len(selected),
        "candidate_counts": {group: len(pool) for group, pool in pools.items()},
        "scene_counts": dict(sorted(Counter(row["scene"] for row in selected).items())),
        "group_scene_counts": {
            group: dict(sorted(Counter(row["scene"] for row in selected
                                       if row.get("augmentation_group", "original") == group).items()))
            for group in counts
        },
        "unique_source_count": len({row["source_id"] for row in selected}),
        "ordered_samples_sha256": hashlib.sha256(canonical_samples).hexdigest(),
        "samples": selected,
    }


def save_selection(path: Path, selection: dict) -> bool:
    if path.exists():
        previous = json.loads(path.read_text())
        if selection["schema"] == "lusnar_random_split_v1":
            volatile = {"created_at_utc", "python_version", "selector_sha256"}
            stable_fields = (set(previous) | set(selection)) - volatile
            for field in sorted(stable_fields):
                if previous.get(field) != selection.get(field):
                    raise FileExistsError(f"Existing random split differs in {field}; use a new output filename")
            return False
        for field in ("schema", "seed", "source_manifest", "source_manifest_sha256",
                      "augmented_manifest", "augmented_manifest_sha256", "counts", "samples"):
            if previous.get(field) != selection[field]:
                raise FileExistsError(f"Existing selection differs in {field}; use a new output filename to preserve both selections")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically without replacing a previous selection, even if two
    # invocations race. Both files reside on the same filesystem.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".selection-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(selection, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return True


def main(argv=None) -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    defaults = json.loads(known.config.read_text()) if known.config else {}
    if not isinstance(defaults, dict):
        raise ValueError("Selection configuration must be a JSON object")
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--mode", choices=("selection", "random-split"), default="selection",
                        help="selection preserves the scene protocol; random-split partitions all 8002 original source families 4:1")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--augmented-manifest", default=DEFAULT_AUGMENTED)
    parser.add_argument("--original-count", type=int, default=5000,
                        help="Used only in selection mode; random-split includes all new-training originals")
    parser.add_argument("--crater-count", type=int, default=1500)
    parser.add_argument("--mountain-rock-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f"Unknown selection configuration keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.output:
        parser.error("--output is required; each different selection needs its own saved filename")
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()
    manifest, augmented, output = resolve(args.manifest), resolve(args.augmented_manifest), resolve(args.output)
    if args.mode == "random-split":
        selection = generate_random_split(manifest, augmented,
                                          {"crater": args.crater_count, "mountain_rock": args.mountain_rock_count}, args.seed)
        created = save_selection(output, selection)
        # The literal file is the artifact to consume; never resample at load time.
        replay = json.loads(output.read_text())
        if (replay["splits_sha256"] != canonical_sha256(replay["splits"])
                or replay["original_split_sha256"] != canonical_sha256(replay["original_split"])):
            raise RuntimeError("Saved random split failed literal-list hash verification")
        print(json.dumps({"selection": str(output), "created": created, "sha256": sha256_file(output),
                          "schema": selection["schema"], "split_protocol": selection["split_protocol"],
                          "counts": selection["counts"], "augmentation_counts": selection["augmentation_counts"],
                          "candidate_counts": selection["candidate_counts"],
                          "excluded_augmented_counts": selection["excluded_augmented_counts"],
                          "ordered_samples_sha256": selection["ordered_samples_sha256"]},
                         ensure_ascii=False), flush=True)
        return
    counts = {"original": args.original_count, "crater": args.crater_count, "mountain_rock": args.mountain_rock_count}
    selection = generate_selection(manifest, augmented, counts, args.seed)
    created = save_selection(output, selection)
    replay = LunarDataset(manifest, split="train", augmented_manifest=augmented,
                          augmentation_groups=["crater", "mountain_rock"], train_source="combined",
                          selection_manifest=output)
    if len(replay) != sum(counts.values()) or replay.partial:
        raise RuntimeError("Saved selection did not replay as the complete intended training set")
    print(json.dumps({"selection": str(output), "created": created, "sha256": sha256_file(output),
                      "counts": counts, "image_count": len(replay), "partial": replay.partial,
                      "unique_source_count": replay.identity["unique_source_count"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
