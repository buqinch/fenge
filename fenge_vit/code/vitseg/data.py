"""Scene/source-family splits and palette-safe loading of the LuSNAR dataset."""
from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ("regolith", "rock", "crater", "mountain", "sky")
PALETTE_RGB = ((187, 70, 156), (232, 250, 80), (120, 0, 200), (173, 69, 31), (34, 201, 248))
TRAIN_SCENES = frozenset(("Moon_1", "Moon_3", "Moon_6", "Moon_7", "Moon_8"))
VAL_SCENES = frozenset(("Moon_5",))
TEST_SCENES = frozenset(("Moon_2", "Moon_4", "Moon_9"))
DEFAULT_MANIFEST = "/media/nas/lzc/fenge/baseline_segformer_20261026/manifest.json"
DEFAULT_TEST_MANIFEST = "/media/nas/lzc/lusnar_test_eval_20261102/data/manifest.json"
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_record_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def decode_palette_label(image: Image.Image) -> np.ndarray:
    """Original P-mode palette indices are NOT our training class IDs."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE_RGB):
        mask[np.all(rgb == color, axis=-1)] = class_id
    if np.any(mask == 255):
        unknown = np.unique(rgb[mask == 255], axis=0)
        raise ValueError(f"Unknown label colors: {unknown[:10].tolist()}")
    return mask


class LunarDataset(Dataset):
    """Load the scene protocol or a validated literal source-family split.

    Offline augmentation may only enter training. Its sources are checked against
    the original training pool and, for a custom split, its new training families.
    Test labels remain in their independent archives, including verified Moon_4.
    """

    def __init__(self, manifest: str | Path = DEFAULT_MANIFEST, split: str = "train",
                 input_size: int = 1024, augmented_manifest: str | Path | None = None,
                 augmentation_groups: list[str] | tuple[str, ...] | None = None,
                 train_source: str = "original", horizontal_flip: bool = False,
                 max_samples: int | None = None,
                 selection_manifest: str | Path | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if train_source not in ("original", "augmented", "combined"):
            raise ValueError("train_source must be original, augmented, or combined")
        if input_size < 32 or input_size % 32:
            raise ValueError("input_size must be a positive multiple of 32")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive")
        if split != "train" and (augmented_manifest is not None or horizontal_flip):
            raise ValueError("Augmentation is permitted only for training")
        if selection_manifest is not None and split != "train":
            raise ValueError("A training selection cannot be used for validation or test")
        self.manifest = Path(manifest).resolve()
        self.split = split
        self.input_size = input_size
        self.horizontal_flip = horizontal_flip
        self._archives: dict[str, zipfile.ZipFile] = {}
        source = json.loads(self.manifest.read_text())
        self.records: list[dict] = []
        self.identity = {"manifest": str(self.manifest), "manifest_sha256": sha256_file(self.manifest),
                         "split": split, "train_source": train_source if split == "train" else None}
        if source.get("schema") == "lusnar_random_split_v1":
            if split == "test":
                raise ValueError("Use the independent Moon_2/4/9 test manifest for test evaluation")
            if (augmented_manifest is not None or augmentation_groups or train_source != "original"):
                raise ValueError("The custom split already embeds its complete training selection; "
                                 "do not add augmentation manifests/groups or train_source overrides")
            self._load_random_split(source)
        elif split == "test":
            records = source.get("records")
            if not isinstance(records, list):
                raise ValueError("Test manifest must contain records")
            for row in records:
                if row["scene"] not in TEST_SCENES:
                    raise ValueError(f"Unexpected test scene: {row['scene']}")
                scene, sid = row["scene"], str(row["sample_id"])
                archive = (Path("/media/nas/lzc/lusnar_test_eval_20261102/labels/Moon_4_labels_verified.zip")
                           if scene == "Moon_4" else Path(f"/media/nas/lzc/{scene}.zip/{scene}.zip"))
                self.records.append({"image": str(resolve_record_path(self.manifest.parent, row["image"])),
                                     "label_archive": str(archive),
                                     "label_member": f"{scene}/image0/label/{sid}.png",
                                     "scene": scene, "sample_id": sid, "source_id": f"{scene}/{sid}"})
        else:
            if source.get("class_names") != list(CLASS_NAMES):
                raise ValueError("Manifest class order differs from the fixed five-class definition")
            rows = source["samples"]
            train_ids = set()
            for row in rows:
                expected = "train" if row["scene"] in TRAIN_SCENES else "val" if row["scene"] in VAL_SCENES else None
                if expected is None or row["split"] != expected:
                    raise ValueError(f"Scene split violation: {row.get('scene')}/{row.get('split')}")
                if expected == "train":
                    train_ids.add((row["scene"], str(row["sample_id"])))
                if row["split"] == split and (split != "train" or train_source != "augmented"):
                    self.records.append(self._normal_record(row, self.manifest.parent))
            if split == "train" and train_source in ("augmented", "combined"):
                if not augmented_manifest:
                    raise ValueError("augmented_manifest is required for augmented/combined training")
                aug_path = Path(augmented_manifest).resolve()
                aug = json.loads(aug_path.read_text())
                if aug.get("class_names") != list(CLASS_NAMES):
                    raise ValueError("Augmented manifest class order differs")
                if aug.get("validation_or_test_used", False):
                    raise ValueError("Augmentation manifest includes validation/test data")
                requested_groups = set(augmentation_groups or ())
                if not requested_groups.issubset({"crater", "mountain_rock", "mixed"}):
                    raise ValueError("Unknown augmentation group")
                for row in aug["samples"]:
                    source_key = (row["source_scene"], str(row["source_sample_id"]))
                    if row.get("split") != "train" or source_key not in train_ids:
                        raise ValueError(f"Augmentation source is outside original train split: {source_key}")
                    group = row.get("primary_group", row.get("focus_group"))
                    if requested_groups and group not in requested_groups:
                        continue
                    self.records.append({"image": str(resolve_record_path(aug_path.parent, row["image"])),
                                         "mask": str(resolve_record_path(aug_path.parent, row["mask"])),
                                         "scene": source_key[0], "sample_id": row["sample_id"],
                                         "source_id": "/".join(source_key), "augmentation_group": group})
                self.identity.update(augmented_manifest=str(aug_path),
                                     augmented_manifest_sha256=sha256_file(aug_path),
                                     augmentation_groups=sorted(requested_groups))
            elif augmented_manifest is not None:
                raise ValueError("Set train_source=augmented or combined to use augmented_manifest")
        if selection_manifest is not None:
            self._apply_selection(Path(selection_manifest).resolve())
        if not self.records:
            raise ValueError("No samples selected")
        self.complete_count = len(self.records)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.partial = len(self.records) != self.complete_count
        source_ids = sorted({row["source_id"] for row in self.records})
        ordered_hash = (_canonical_sha256(self.records) if self.identity.get("custom_split") else
                        hashlib.sha256(json.dumps(self.records, sort_keys=True).encode()).hexdigest())
        self.identity.update(image_count=len(self.records), complete_image_count=self.complete_count,
                             partial=self.partial, unique_source_count=len(source_ids),
                             ordered_samples_sha256=ordered_hash)

    def _load_random_split(self, source: dict):
        """Replay a literal source-family split, validating both halves together.

        No random selection occurs while loading or resuming. The source
        manifests establish the complete original pool and legal augmentation
        provenance; the new literal lists establish the new train/val roles.
        """
        protocol = "random_original_source_family_4_to_1"
        if source.get("split_protocol") != protocol or source.get("class_names") != list(CLASS_NAMES):
            raise ValueError("Custom split protocol or class order differs from the five-class definition")
        if type(source.get("seed")) is not int or source["seed"] < 0:
            raise ValueError("Custom split seed must be a nonnegative integer")
        if source.get("test_scenes_used", False) is not False:
            raise ValueError("Custom original pool cannot contain independent test scenes")
        splits, original_split = source.get("splits"), source.get("original_split")
        if (not isinstance(splits, dict) or set(splits) != {"train", "val"}
                or any(not isinstance(rows, list) or not rows for rows in splits.values())):
            raise ValueError("Custom split must contain nonempty literal train and val lists")
        if (not isinstance(original_split, dict)
                or set(original_split) != {"train_source_ids", "val_source_ids"}
                or any(not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids)
                       for ids in original_split.values())):
            raise ValueError("Custom split must declare literal original source-family lists")
        if source.get("splits_sha256") != _canonical_sha256(splits):
            raise ValueError("Custom split literal lists SHA256 mismatch")
        if source.get("original_split_sha256") != _canonical_sha256(original_split):
            raise ValueError("Custom original source-family lists SHA256 mismatch")
        ordered = source.get("ordered_samples_sha256")
        if (not isinstance(ordered, dict) or set(ordered) != {"train", "val"}
                or any(ordered[split] != _canonical_sha256(rows) for split, rows in splits.items())):
            raise ValueError("Custom split ordered sample SHA256 mismatch")

        provenance = {}
        for prefix in ("source", "augmented"):
            raw_path = source.get(prefix + "_manifest")
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise ValueError(f"Custom {prefix} manifest path must be absolute")
            path = Path(raw_path).resolve()
            fingerprint = sha256_file(path)
            if source.get(prefix + "_manifest_sha256") != fingerprint:
                raise ValueError(f"Custom {prefix} manifest SHA256 mismatch")
            document = json.loads(path.read_text())
            if document.get("class_names") != list(CLASS_NAMES) or not isinstance(document.get("samples"), list):
                raise ValueError(f"Invalid custom {prefix} source manifest")
            provenance[prefix] = (path, fingerprint, document)
        original_path, original_sha, original_document = provenance["source"]
        augmented_path, augmented_sha, augmented_document = provenance["augmented"]
        if (augmented_document.get("validation_or_test_used") is not False
                or not isinstance(augmented_document.get("source_manifest"), str)
                or Path(augmented_document["source_manifest"]).resolve() != original_path
                or augmented_document.get("source_manifest_sha256") != original_sha):
            raise ValueError("Augmentation provenance differs from the original pool or includes validation/test sources")

        resolved_paths = {}
        def canonical_path(root, value):
            if not isinstance(value, str):
                raise ValueError("Sample image/mask paths must be strings")
            path = str(resolve_record_path(root, value))
            if path not in resolved_paths:
                resolved_paths[path] = str(Path(path).resolve())
            return resolved_paths[path]

        originals, old_train_ids = {}, set()
        source_images, source_masks = set(), set()
        for row in original_document["samples"]:
            scene, sid = row.get("scene"), str(row.get("sample_id"))
            expected = "train" if scene in TRAIN_SCENES else "val" if scene in VAL_SCENES else None
            if expected is None or row.get("split") != expected:
                raise ValueError("Original pool contains an invalid scene/split or an independent test scene")
            source_id = f"{scene}/{sid}"
            image = canonical_path(original_path.parent, row.get("image"))
            mask = canonical_path(original_path.parent, row.get("mask"))
            if (source_id in originals or image in source_images or mask in source_masks
                    or image in source_masks or mask in source_images or image == mask):
                raise ValueError("Original pool contains duplicate source identities or image/mask paths")
            originals[source_id] = {"scene": scene, "sample_id": sid, "source_id": source_id,
                                    "image": image, "mask": mask, "source_kind": "original"}
            source_images.add(image)
            source_masks.add(mask)
            if expected == "train":
                old_train_ids.add(source_id)
        train_ids, val_ids = (set(original_split[name]) for name in ("train_source_ids", "val_source_ids"))
        pool_size = len(originals)
        if (pool_size < 2 or len(train_ids) != len(original_split["train_source_ids"])
                or len(val_ids) != len(original_split["val_source_ids"]) or train_ids & val_ids
                or train_ids | val_ids != set(originals)
                or len(train_ids) != pool_size * 4 // 5 or len(val_ids) != pool_size - pool_size * 4 // 5):
            raise ValueError("Custom original families must partition the complete original pool 4:1 without overlap")
        if (source.get("original_pool_source_ids_sha256") is not None
                and source["original_pool_source_ids_sha256"] != _canonical_sha256(sorted(originals))):
            raise ValueError("Custom original pool source-ID SHA256 mismatch")

        augmented = {}
        for row in augmented_document["samples"]:
            source_id = f"{row.get('source_scene')}/{row.get('source_sample_id')}"
            if row.get("split") != "train" or source_id not in old_train_ids:
                raise ValueError("Augmentation source is outside the legal original training pool")
            group = row.get("primary_group", row.get("focus_group"))
            if group not in ("crater", "mountain_rock", "mixed"):
                raise ValueError("Unknown augmentation group in source manifest")
            scene, sid = row["source_scene"], str(row.get("sample_id"))
            key = (scene, sid)
            image = canonical_path(augmented_path.parent, row.get("image"))
            mask = canonical_path(augmented_path.parent, row.get("mask"))
            if (key in augmented or image in source_images or mask in source_masks
                    or image in source_masks or mask in source_images or image == mask):
                raise ValueError("Augmentation pool contains duplicate identities or original/augmented image/mask paths")
            augmented[key] = {"scene": scene, "sample_id": sid, "source_id": source_id,
                              "image": image, "mask": mask, "source_kind": "augmented", "augmentation_group": group}
            source_images.add(image)
            source_masks.add(mask)

        augmentation_counts = source.get("augmentation_counts")
        if (not isinstance(augmentation_counts, dict) or set(augmentation_counts) != {"crater", "mountain_rock"}
                or any(type(count) is not int or count < 1 for count in augmentation_counts.values())):
            raise ValueError("Custom split must declare positive crater and mountain_rock augmentation counts")
        actual_groups = dict.fromkeys(augmentation_counts, 0)
        actual_originals = {"train": set(), "val": set()}
        seen_ids, seen_images, seen_masks = set(), set(), set()
        for split, rows in splits.items():
            for row in rows:
                if not isinstance(row, dict) or row.get("split") != split:
                    raise ValueError("Custom sample has an invalid split assignment")
                scene, sid, kind = row.get("scene"), str(row.get("sample_id")), row.get("source_kind")
                if scene not in TRAIN_SCENES | VAL_SCENES:
                    raise ValueError("Independent test scenes are excluded from custom train/val")
                source_id = row.get("source_id")
                expected_ids = train_ids if split == "train" else val_ids
                if source_id not in expected_ids:
                    raise ValueError("Custom split leaks an original source family across train/val")
                if kind == "original":
                    candidate = originals.get(source_id)
                    if row.get("augmentation_group") is not None:
                        raise ValueError("Original samples cannot declare an augmentation group")
                    if source_id in actual_originals[split]:
                        raise ValueError("Duplicate original family in custom split")
                    actual_originals[split].add(source_id)
                elif kind == "augmented" and split == "train":
                    candidate = augmented.get((scene, sid))
                    group = row.get("augmentation_group")
                    if group not in actual_groups or candidate is None or candidate["augmentation_group"] != group:
                        raise ValueError("Custom augmented sample/group is absent from the source manifest")
                    actual_groups[group] += 1
                else:
                    raise ValueError("Custom validation must contain only originals; training kinds must be original/augmented")
                if candidate is None or any(row.get(field) != candidate[field] for field in
                                            ("scene", "sample_id", "source_id", "source_kind", "image", "mask")):
                    raise ValueError("Custom sample identity, source or paths differ from source provenance")
                for field in ("image", "mask"):
                    if (not Path(row[field]).is_absolute()
                            or canonical_path(self.manifest.parent, row[field]) != row[field]):
                        raise ValueError("Custom sample paths must be absolute canonical paths")
                image, mask, key = row["image"], row["mask"], (scene, sid)
                if (key in seen_ids or image in seen_images or mask in seen_masks
                        or image in seen_masks or mask in seen_images or image == mask):
                    raise ValueError("Custom split contains duplicate sample identities or image/mask paths")
                seen_ids.add(key)
                seen_images.add(image)
                seen_masks.add(mask)
        if actual_originals != {"train": train_ids, "val": val_ids} or actual_groups != augmentation_counts:
            raise ValueError("Custom literal samples do not contain the complete originals/declared augmentations")
        expected_counts = {"original_pool": pool_size, "train_original": len(train_ids), "val_original": len(val_ids),
                           "train_augmented": sum(actual_groups.values()), "train_total": len(splits["train"]),
                           "val_total": len(splits["val"])}
        counts = source.get("counts")
        if (not isinstance(counts, dict) or counts != expected_counts
                or any(type(count) is not int for count in counts.values())):
            raise ValueError("Custom split declared counts differ from literal samples")
        if sha256_file(original_path) != original_sha or sha256_file(augmented_path) != augmented_sha:
            raise ValueError("A source manifest changed while validating the custom split")
        self.records = [dict(row) for row in splits[self.split]]
        self.identity.update(protocol=protocol, custom_split=True,
                             train_source="embedded_custom_manifest" if self.split == "train" else None,
                             selection_metric="validation_all_scenes_miou", selection_scope="custom_val_originals",
                             split_seed=source["seed"], split_counts=counts, augmentation_counts=augmentation_counts,
                             source_manifest=str(original_path), source_manifest_sha256=original_sha,
                             augmented_manifest=str(augmented_path), augmented_manifest_sha256=augmented_sha,
                             original_split_sha256=source["original_split_sha256"], splits_sha256=source["splits_sha256"],
                             literal_ordered_samples_sha256=ordered[self.split],
                             train_original_source_ids_sha256=_canonical_sha256(sorted(train_ids)),
                             val_original_source_ids_sha256=_canonical_sha256(sorted(val_ids)),
                             validation_scenes=sorted({row["scene"] for row in splits["val"]}),
                             validation_image_count=len(splits["val"]),
                             independent_test_scenes=sorted(TEST_SCENES))

    def _apply_selection(self, path: Path):
        """Validate a fixed sample list against the complete legal candidates.

        A selected dataset is the intended complete training set, not a partial
        validation run. No sampling occurs here or on resume: file order and
        identities are retained exactly as recorded by the selection producer.
        """
        selection = json.loads(path.read_text())
        if selection.get("schema") != "lusnar_training_selection_v1":
            raise ValueError("Unsupported training selection schema")
        if type(selection.get("seed")) is not int or selection["seed"] < 0:
            raise ValueError("Training selection seed must be a nonnegative integer")
        for prefix, identity_prefix in (("source", "manifest"), ("augmented", "augmented_manifest")):
            key = prefix + "_manifest"
            expected_path = self.identity.get(identity_prefix)
            expected_sha = self.identity.get(identity_prefix + "_sha256")
            if expected_path is None:
                raise ValueError(f"Training selection requires configured {key} candidates")
            if not isinstance(selection.get(key), str) or Path(selection[key]).resolve() != Path(expected_path):
                raise ValueError(f"Training selection {key} differs from configured candidates")
            if selection.get(key + "_sha256") != expected_sha:
                raise ValueError(f"Training selection {key} SHA256 differs from current source manifest")
        rows = selection.get("samples")
        declared_counts = selection.get("counts")
        if not isinstance(rows, list) or not rows or not isinstance(declared_counts, dict):
            raise ValueError("Training selection must contain nonempty samples and counts")
        if self.identity.get("custom_split"):
            if (selection.get("split") != "train" or selection.get("validation_or_test_used") is not False
                    or selection.get("class_names") != list(CLASS_NAMES)
                    or selection.get("image_count") != len(rows)
                    or selection.get("ordered_samples_sha256") != _canonical_sha256(rows)):
                raise ValueError("Custom training selection metadata or ordered sample SHA256 mismatch")
        allowed_kinds = {"original", "crater", "mountain_rock", "mixed"}
        if (not declared_counts or not set(declared_counts).issubset(allowed_kinds)
                or any(type(value) is not int or value < 0 for value in declared_counts.values())):
            raise ValueError("Invalid training selection counts")
        if sum(declared_counts.values()) != len(rows):
            raise ValueError("Training selection counts do not sum to sample count")
        candidates = {}
        for record in self.records:
            kind = "augmented" if "augmentation_group" in record else "original"
            key = (kind, record["scene"], str(record["sample_id"]))
            if key in candidates:
                raise ValueError(f"Duplicate sample identity in source candidates: {key}")
            candidates[key] = record
        actual_counts = {key: 0 for key in declared_counts}
        selected, seen, seen_paths = [], set(), set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Training selection sample must be an object")
            if self.identity.get("custom_split") and row.get("split") != "train":
                raise ValueError("Custom training selection can contain only train-assigned samples")
            kind = row.get("source_kind")
            if kind not in ("original", "augmented"):
                raise ValueError("Training selection source_kind must be original or augmented")
            key = (kind, row.get("scene"), str(row.get("sample_id")))
            if key in seen:
                raise ValueError(f"Duplicate sample in training selection: {key}")
            candidate = candidates.get(key)
            if candidate is None:
                raise ValueError(f"Training selection sample is outside configured training candidates: {key}")
            for field in ("image", "mask", "scene", "sample_id", "source_id"):
                if row.get(field) != candidate[field]:
                    raise ValueError(f"Training selection {field} mismatch for {key}")
            if not Path(row["image"]).is_absolute() or not Path(row["mask"]).is_absolute():
                raise ValueError("Training selection image/mask paths must be absolute")
            group = candidate.get("augmentation_group")
            if row.get("augmentation_group") != group:
                raise ValueError(f"Training selection augmentation group mismatch for {key}")
            count_key = "original" if kind == "original" else group
            if count_key not in actual_counts:
                raise ValueError(f"Training selection omitted the count of {count_key}")
            paths = (row["image"], row["mask"])
            if paths in seen_paths:
                raise ValueError(f"Duplicate image/mask pair in training selection: {key}")
            seen.add(key)
            seen_paths.add(paths)
            actual_counts[count_key] += 1
            selected.append({**candidate, "source_kind": kind})
        if actual_counts != declared_counts:
            raise ValueError(f"Training selection actual counts differ: {actual_counts} vs {declared_counts}")
        self.identity.update(selection_manifest=str(path), selection_manifest_sha256=sha256_file(path),
                             selection_seed=selection["seed"], selection_counts=declared_counts,
                             candidate_image_count=len(self.records))
        if self.identity.get("custom_split"):
            self.identity.update(candidate_ordered_samples_sha256=_canonical_sha256(self.records),
                                 selection_ordered_samples_sha256=selection["ordered_samples_sha256"])
        self.records = selected

    @staticmethod
    def _normal_record(row, root):
        scene, sid = row["scene"], str(row["sample_id"])
        return {"image": str(resolve_record_path(root, row["image"])),
                "mask": str(resolve_record_path(root, row["mask"])),
                "scene": scene, "sample_id": sid, "source_id": f"{scene}/{sid}"}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["image"]) as source_image:
            image = source_image.convert("RGB")
        if "mask" in row:
            with Image.open(row["mask"]) as source_mask:
                # ID masks must already be 2-D; never reinterpret raw palette indices.
                if source_mask.mode != "L":
                    raise ValueError(f"Expected L-mode class-ID mask, got {source_mask.mode}: {row['mask']}")
                mask_array = np.asarray(source_mask).copy()
            if mask_array.ndim != 2 or np.any(mask_array > 4):
                raise ValueError(f"Invalid ID mask: {row['mask']}")
        else:
            archive_path = row["label_archive"]
            if archive_path not in self._archives:
                self._archives[archive_path] = zipfile.ZipFile(archive_path)
            with self._archives[archive_path].open(row["label_member"]) as stream:
                with Image.open(io.BytesIO(stream.read())) as label:
                    mask_array = decode_palette_label(label)
        mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
        if image.size != mask.size:
            raise ValueError(f"Image/label size mismatch: {row['source_id']}")
        size = (self.input_size, self.input_size)
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
        horizontal_flipped = self.horizontal_flip and random.random() < 0.5
        if horizontal_flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(((image_array - _MEAN) / _STD).copy())
        return {"image": image_tensor, "mask": torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()),
                "scene": row["scene"], "sample_id": row["sample_id"], "source_id": row["source_id"],
                "horizontal_flipped": horizontal_flipped}
