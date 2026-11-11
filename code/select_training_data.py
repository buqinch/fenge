"""Save a reproducible, without-replacement selection of original and ROI data."""
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

from lunarseg.data import CLASS_NAMES, DEFAULT_MANIFEST, LunarDataset, sha256_file

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMENTED = "/media/nas/lzc/lusnar_augmentation/lusnar_gt_context_6000_20261110/manifest.json"


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


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    defaults = json.loads(known.config.read_text()) if known.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--augmented-manifest", default=DEFAULT_AUGMENTED)
    parser.add_argument("--original-count", type=int, default=5000)
    parser.add_argument("--crater-count", type=int, default=1500)
    parser.add_argument("--mountain-rock-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f"Unknown selection configuration keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if not args.output:
        parser.error("--output is required; each different selection needs its own saved filename")
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()
    manifest, augmented, output = resolve(args.manifest), resolve(args.augmented_manifest), resolve(args.output)
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
