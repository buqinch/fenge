"""Fixed scene splits and palette-safe loading of the LuSNAR dataset."""
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
    """All train/val assignments come from the original scene manifest.

    Offline augmentation may only enter training. Its sources are checked against
    the original training identities before paths are accepted. Test labels are
    read from their original archives, including the verified Moon_4 archive.
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
        if split == "test":
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
        self.identity.update(image_count=len(self.records), complete_image_count=self.complete_count,
                             partial=self.partial, unique_source_count=len(source_ids),
                             ordered_samples_sha256=hashlib.sha256(json.dumps(self.records, sort_keys=True).encode()).hexdigest())

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
        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(((image_array - _MEAN) / _STD).copy())
        return {"image": image_tensor, "mask": torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()),
                "scene": row["scene"], "sample_id": row["sample_id"], "source_id": row["source_id"]}
