"""Training-only prior bookkeeping; inference never constructs the affinity head."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import torch


PRIOR_DEFAULTS = {"refinement": "none", "affinity_weight": 0.0, "consistency_weight": 0.0,
                  "prior_warmup_updates": 1000, "prior_lr": 1e-4, "prior_seed": 4242,
                  "geometry_paired_control": False, "epoch_zero_validation": False,
                  "gradient_checkpointing": False}


def validate_prior_args(args):
    for name in ("affinity_weight", "consistency_weight"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(args.prior_lr, (int, float)) or isinstance(args.prior_lr, bool) or not math.isfinite(args.prior_lr) or args.prior_lr <= 0:
        raise ValueError("prior_lr must be finite and positive")
    for name in ("prior_warmup_updates", "prior_seed"):
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("geometry_paired_control", "epoch_zero_validation", "gradient_checkpointing"):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")


def load_initial_model(model, state):
    """Strict body loading; a newly added complete refiner is the sole exception."""
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    allowed = {name for name in expected if name.startswith("refiner.")}
    if unexpected or (missing and (not allowed or missing != allowed)):
        raise ValueError(f"Initial checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    # Validate shapes before load_state_dict can partially mutate the model.
    for name in set(expected) & set(state):
        if not torch.is_tensor(state[name]) or expected[name].shape != state[name].shape:
            raise ValueError(f"Initial checkpoint shape/type mismatch: {name}")
    result = model.load_state_dict(state, strict=not missing)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("Unexpected mismatch while loading initial model")
    return {"strict_body": True, "new_refiner_keys": sorted(missing),
            "loaded_tensor_count": len(state)}


def warmup_scale(update, warmup_updates):
    """One shared factor per optimizer update, including all its microbatches."""
    return min(1.0, (update + 1) / warmup_updates) if warmup_updates else 1.0


def batch_keys(batch):
    return [f"{scene}/{sample}/{source}" for scene, sample, source in
            zip(batch["scene"], batch["sample_id"], batch["source_id"])]


class PriorDiagnostics:
    """Bounded examples, deterministic epoch hashes and detached count totals."""

    def __init__(self):
        self.order = hashlib.sha256()
        self.flips = hashlib.sha256()
        self.combined = hashlib.sha256()
        self.geometry = hashlib.sha256()
        self.affinity_sampling = hashlib.sha256()
        self.images = self.batches = self.geometry_images = 0
        self.affinity_counts = Counter()
        self.affinity_class_pairs = Counter()
        self.affinity_distances = Counter()
        self.geometry_counts = Counter()
        self.geometry_transforms = Counter()
        self.examples = []

    @staticmethod
    def _add(digest, value):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def observe_batch(self, batch):
        keys = batch_keys(batch)
        flags = batch.get("horizontal_flipped", [False] * len(keys))
        for key, flag in zip(keys, flags):
            self._add(self.order, key)
            self._add(self.flips, bool(flag))
            self._add(self.combined, [key, bool(flag)])
        self.images += len(keys)
        self.batches += 1
        return keys

    def observe_affinity(self, stats):
        for name in ("selected_positive_pairs", "selected_negative_pairs", "valid_image_count"):
            self.affinity_counts[name] += int(stats.get(name, 0))
        for row in stats.get("images", []):
            self._add(self.affinity_sampling, [row.get("sample_key"), row.get("sampling_sha256")])
            self.affinity_counts["boundary_endpoint_pairs"] += int(row.get("boundary_endpoint_pairs", 0))
            self.affinity_class_pairs.update(row.get("selected_by_class_pair", {}))
            self.affinity_distances.update({str(k): int(v) for k, v in row.get("selected_by_distance", {}).items()})
            if len(self.examples) < 16:
                self.examples.append({key: row.get(key) for key in
                                      ("sample_key", "epoch", "sampling_sha256", "positive_count", "negative_count", "selected_pairs_preview")})

    def observe_geometry(self, keys, metadata, stats=None):
        rows = metadata.get("images", metadata.get("transforms", [])) if isinstance(metadata, dict) else metadata
        if len(rows) != len(keys):
            raise ValueError("Geometry metadata must contain one transform per input image")
        for key, row in zip(keys, rows):
            self._add(self.geometry, [key, row])
            self.geometry_transforms.update([json.dumps({k: row.get(k) for k in ("hflip", "scale", "dx", "dy")}, sort_keys=True)])
        self.geometry_images += len(keys)
        for name, value in (stats or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                self.geometry_counts[name] += value

    def result(self):
        return {"schema": "lusnar_training_priors_diagnostics_v1", "images": self.images, "batches": self.batches,
                "sample_order_sha256": self.order.hexdigest(), "horizontal_flip_sha256": self.flips.hexdigest(),
                "sample_and_horizontal_flip_sha256": self.combined.hexdigest(),
                "geometry_view_sha256": self.geometry.hexdigest() if self.geometry_images else None,
                "geometry_images": self.geometry_images, "geometry_transform_counts": dict(self.geometry_transforms),
                "geometry_numeric_batch_sums": dict(self.geometry_counts),
                "affinity_sampling_sha256": self.affinity_sampling.hexdigest() if self.affinity_counts else None,
                "affinity_counts": dict(self.affinity_counts), "affinity_selected_by_class_pair": dict(self.affinity_class_pairs),
                "affinity_selected_by_distance": dict(self.affinity_distances), "affinity_examples": self.examples}
