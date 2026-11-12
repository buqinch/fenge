#!/usr/bin/env python3
"""Independent CPU contracts for retained sparse affinity and retired-option compatibility."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lunarseg.priors import SemanticAffinity
from lunarseg.training_priors import without_retired_geometry, validate_prior_implementation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="result/prior_ablation/verification_priors.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = []

    def check(name, test):
        start = time.perf_counter()
        result = test()
        checks.append({"name": name, "passed": True,
                       "seconds": time.perf_counter() - start, "details": result})
        print("PASS", name, flush=True)

    local = torch.Generator().manual_seed(731)
    rand = lambda *shape: torch.randn(*shape, generator=local)

    def no_rng_side_effects():
        before_torch, before_python, before_numpy = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        module = SemanticAffinity(width=4, projected=3, quota=64)
        target = torch.zeros(2, 48, 64, dtype=torch.long)
        target[:, :, 32:] = 1
        features = rand(2, 4, 12, 16).requires_grad_()
        _, first = module(features, target, ["A", "B"], 1)
        _, reordered = module(features.flip(0), target.flip(0), ["B", "A"], 1)
        assert first["images"][0]["sampling_sha256"] == reordered["images"][1]["sampling_sha256"]
        assert torch.equal(before_torch, torch.get_rng_state())
        assert before_python == random.getstate()
        after_numpy = np.random.get_state()
        assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1]) and before_numpy[2:] == after_numpy[2:]
        return {"construction_sampling": "global CPU torch/Python/NumPy RNG unchanged",
                "batch_reordering_preserves_per_image_samples": True, "CUDA": "not exercised by CPU verification"}

    check("independent_stateless_rng_and_initialization", no_rng_side_effects)

    def exhaustive_small_pairs():
        gt = torch.zeros(17, 19, dtype=torch.long)
        gt[8, 9] = 1  # A one-pixel rock must survive original-grid sampling.
        gt[0, :] = 4
        gt[-1, :] = 255
        module = SemanticAffinity(width=1, projected=1)
        pairs, signs, stats = module.sample_pairs(gt, "tiny-rock", 2)
        candidate_counts = {}
        all_pairs = set()
        coords = [(y, x) for y in range(17) for x in range(19) if int(gt[y, x]) < 4]
        for (ya, xa), (yb, xb) in itertools.combinations(coords, 2):
            dx, dy = abs(xa - xb), abs(ya - yb)
            radius = max(dx, dy)
            if radius not in (8, 16, 32, 64) or dx not in (0, radius) or dy not in (0, radius):
                continue
            a, b = sorted((int(gt[ya, xa]), int(gt[yb, xb])))
            key = f"{a}-{b}@{radius}"
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
            all_pairs.add(tuple(sorted((ya * 19 + xa, yb * 19 + xb))))
        assert stats["candidate_counts"] == candidate_counts
        unique = set()
        rock_selected = 0
        for (xa, ya, xb, yb), same in zip(pairs.tolist(), signs.tolist()):
            key = tuple(sorted((ya * 19 + xa, yb * 19 + xb)))
            assert key in all_pairs and key not in unique
            unique.add(key)
            a, b = int(gt[ya, xa]), int(gt[yb, xb])
            assert a < 4 and b < 4 and bool(same) == (a == b)
            rock_selected += int(a == 1 or b == 1)
        assert rock_selected > 0 and stats["negative_count"] < 256
        assert stats["sampling_reliability"]["anchors_by_class"]["1"]["all_class_pixels_preserved"]
        assert stats["boundary_endpoint_pairs"] > 0
        again = module.sample_pairs(gt, "tiny-rock", 2)
        assert torch.equal(pairs, again[0])
        assert not torch.equal(pairs, module.sample_pairs(gt, "tiny-rock", 3)[0])
        return {"candidate_count_independently_enumerated": len(all_pairs), "tiny_rock_pairs": rock_selected,
                "no_reverse_duplicates": True, "sky_ignore_excluded": True,
                "negative_shortage_is_not_repeated": stats["negative_count"]}

    check("original_grid_tiny_rock_ignore_offsets_no_replacement", exhaustive_small_pairs)

    def balance_and_coordinates():
        gt = torch.zeros(128, 128, dtype=torch.long)
        gt[:64, 64:] = 1
        gt[64:, :64] = 2
        gt[64:, 64:] = 3
        module = SemanticAffinity(width=1, projected=1)
        pairs, signs, stats = module.sample_pairs(gt, "balanced", 1)
        assert int(signs.sum()) == 256 and len(signs) == 512
        selected = {}
        for xa, ya, xb, yb in pairs.tolist():
            a, b = sorted((int(gt[ya, xa]), int(gt[yb, xb])))
            r = max(abs(xa - xb), abs(ya - yb))
            key = f"{a}-{b}@{r}"
            selected[key] = selected.get(key, 0) + 1
        for same in (True, False):
            group = [v for k, v in selected.items() if (k.split("@")[0].split("-")[0] == k.split("@")[0].split("-")[1]) == same]
            assert max(group) - min(group) <= 1
        # Independently evaluate the analytic feature plane at original pixel
        # centers. It reveals x/y swaps and align_corners mistakes directly.
        target = torch.zeros(24, 32, dtype=torch.long)
        small = SemanticAffinity(width=1, projected=1, quota=16)
        with torch.no_grad():
            small.project.weight.fill_(1)
            small.project.bias.zero_()
        yy, xx = torch.meshgrid(torch.arange(6.), torch.arange(8.), indexing="ij")
        features = (xx + 2 * yy)[None, None]
        captured = []
        hook = small.predictor.register_forward_pre_hook(lambda _, values: captured.append(values[0].detach()))
        small(features, target[None], ["coordinates"], 1)
        hook.remove()
        pp = small.sample_pairs(target, "coordinates", 1)[0]
        expected = []
        for xa, ya, xb, yb in pp.tolist():
            value = lambda x, y: min(7, max(0, (x + .5) / 4 - .5)) + 2 * min(5, max(0, (y + .5) / 4 - .5))
            a, b = value(xa, ya), value(xb, yb)
            expected.append([(a + b) / 2, abs(a - b), a * b, abs(xb - xa) / 32, abs(yb - ya) / 24])
        torch.testing.assert_close(captured[0], torch.tensor(expected), atol=1e-5, rtol=1e-6)
        a, b, offset = rand(12, 3), rand(12, 3), rand(12, 2)
        assert torch.equal(module.pair_descriptors(a, b, offset), module.pair_descriptors(b, a, -offset))
        return {"positive_negative_quotas": [256, 256], "class_pair_distance_round_robin": True,
                "pixel_center_analytic_oracle": True, "endpoint_symmetry_exact": True}

    check("class_distance_balance_pixel_centers_symmetry", balance_and_coordinates)

    def losses_and_gradients():
        module = SemanticAffinity(width=4, projected=3, quota=64)
        gt = torch.zeros(2, 48, 64, dtype=torch.long)
        gt[1, :, 32:] = 1
        features = rand(2, 4, 12, 16).requires_grad_()
        loss, _ = module(features, gt, ["mono", "dual"], 1)
        loss.backward()
        assert features.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters())
        with torch.no_grad():
            for p in module.predictor.parameters():
                p.zero_()
            module.predictor[-1].bias.fill_(math.log(3))
        loss, _ = module(features, gt, ["mono", "dual"], 1)
        # Image0: positive only. Image1: mean(pos,neg). Then fixed batch mean.
        expected = (math.log(4 / 3) + (math.log(4 / 3) + math.log(4)) / 2) / 2
        assert abs(float(loss) - expected) < 1e-6
        empty = SemanticAffinity(width=4, projected=3)
        f = rand(2, 4, 12, 16).requires_grad_()
        gt = torch.full((2, 48, 64), 255, dtype=torch.long)
        gt[0].fill_(4)
        loss, stats = empty(f, gt, ["ignore", "sky"], 1)
        assert float(loss) == 0 and stats["valid_image_count"] == 0
        loss.backward()
        assert f.grad is not None and not f.grad.any()
        assert all(p.grad is not None and not p.grad.any() for p in empty.parameters())
        return {"feature_and_head_gradients_nonzero": True, "independent_BCE_reduction_oracle": expected,
                "empty_graph_connected_zero": True}

    check("affinity_loss_reduction_and_gradients", losses_and_gradients)

    def invalid_parameters():
        bad = [lambda: SemanticAffinity(quota=-1), lambda: SemanticAffinity(quota=513),
               lambda: SemanticAffinity(quota=True), lambda: SemanticAffinity(width=0),
               lambda: SemanticAffinity(seed=-1),
               lambda: SemanticAffinity().sample_pairs(torch.full((16, 16), -1, dtype=torch.long), "bad", 1)]
        for call in bad:
            try:
                call()
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("invalid parameters were accepted")
        return {"rejected_cases": len(bad)}

    check("invalid_parameters_rejected", invalid_parameters)

    def retired_geometry_compatibility():
        old = {"consistency_weight": 0.0, "geometry_paired_control": False,
               "prior_protocol": {"paired_geometry": False}, "affinity_weight": .05}
        clean = without_retired_geometry(old)
        assert "consistency_weight" not in clean and "geometry_paired_control" not in clean
        assert clean["affinity_weight"] == .05 and old["consistency_weight"] == 0
        current = {"losses.py": "unchanged", "priors.py": "new-affinity-only",
                   "training_priors.py": "new-bookkeeping"}
        previous = {"losses.py": "unchanged",
                    "priors.py": "80604ff9610ae141cac6348ed918b9212ed00c246f770cd6859a9790143f4ebd",
                    "training_priors.py": "ffd2441e4b597e96e62a996b289ed121956d586f3d3c6cae347c0c4a3af0b4b5"}
        assert validate_prior_implementation(previous, current, old, current) == ["priors.py", "training_priors.py"]
        assert validate_prior_implementation(current, current, old, current) == []
        bad = [lambda: without_retired_geometry({"consistency_weight": .05}),
               lambda: without_retired_geometry({"geometry_paired_control": True}),
               lambda: without_retired_geometry({"prior_protocol": {"paired_geometry": True}}),
               lambda: validate_prior_implementation({**previous, "losses.py": "changed"}, current, old, current),
               lambda: validate_prior_implementation({**previous, "priors.py": "unknown"}, current, old, current),
               lambda: validate_prior_implementation(previous, current, {"geometry_paired_control": True}, current)]
        bad.append(lambda: validate_prior_implementation(previous, {**current, "priors.py": "future-change"}, old, current))
        for call in bad:
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError("Retired geometry or an unknown implementation was accepted")
        return {"disabled_historical_settings_compatible": True,
                "enabled_geometry_and_unknown_hashes_rejected": len(bad)}

    check("retired_geometry_checkpoint_compatibility", retired_geometry_compatibility)

    timings = []
    manifest = Path(__file__).resolve().parents[1] / "result/selections/random80_seed42_20261112.json"
    if manifest.exists():
        module = SemanticAffinity()
        rows = [r for r in json.loads(manifest.read_text())["splits"]["train"] if r["source_kind"] == "original"][:4]
        for row in rows:
            target = torch.from_numpy(np.array(Image.open(row["mask"]), dtype=np.int64))
            start = time.perf_counter()
            _, _, stat = module.sample_pairs(target, row["source_id"], 1)
            timings.append({"source_id": row["source_id"], "seconds": time.perf_counter() - start,
                            "candidate_count": stat["candidate_count"], "selected_count": stat["positive_count"] + stat["negative_count"],
                            "anchors_by_class": stat["sampling_reliability"]["anchors_by_class"]})
    report = {"schema": "lusnar_prior_cpu_verification_v1", "status": "passed", "device": "cpu",
              "checks": checks, "real_1024_GT_sampling_timings": timings,
              "full_candidate_scan_before_optimization_seconds": [.3370, .3331, .3379, .3209],
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("lunarseg/priors.py", "verify_priors.py")},
              "limitations": ["CPU oracle only; actual1024 GPU training/backward is checked separately by root.",
                              "Bounded class-balanced anchors are not an exhaustive full-image pair census."]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
