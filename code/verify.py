"""Verify real weight loading, full-resolution gradients and dataset separation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from lunarseg.losses import SegmentationCriterion
from lunarseg.metrics import ConfusionMatrix, adjust_mountain_scores, evaluate_loader
from lunarseg.model import ModelConfig, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("/media/nas/lzc/fenge/baseline_segformer_20261026/manifest.json")


def check_mountain_calibration() -> dict:
    """Check score-domain calibration on CPU without weights or a dataset."""
    # Pixel zero has a negative mountain logit that must lose after downweighting.
    # Multiplying that negative logit by alpha would incorrectly increase it.
    logits = torch.tensor([
        [-4.0, -3.0, -1.6, -1.5, -5.0],
        [3.0, 1.0, 0.0, -4.0, 2.0],
        [0.0, 3.0, 1.0, 2.0, -1.0],
        [-3.0, -2.0, 3.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0, 2.0, 3.0],
        [-4.0, -3.0, -2.0, -1.0, -5.0],
    ], dtype=torch.float32).T.reshape(1, 5, 2, 3)
    untouched = logits.clone()
    probabilities = logits.softmax(1)
    log_probabilities = logits.log_softmax(1)
    other_classes = [0, 1, 2, 4]
    alphas = [1.0, 0.95, 0.90, 0.85, 0.80]
    for alpha in [*alphas, 1.25]:
        expected_probabilities = probabilities.clone()
        expected_probabilities[:, 3] *= alpha
        expected_logits = logits.clone()
        expected_logits[:, 3] += math.log(alpha)
        adjusted_logits = adjust_mountain_scores(logits, mountain_alpha=alpha, tta="none")
        adjusted_log_probabilities = adjust_mountain_scores(log_probabilities, mountain_alpha=alpha, tta="none")
        adjusted_probabilities = adjust_mountain_scores(probabilities, mountain_alpha=alpha, tta="hflip")
        torch.testing.assert_close(adjusted_logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(adjusted_probabilities, expected_probabilities, rtol=0, atol=0)
        normalized = expected_probabilities / expected_probabilities.sum(1, keepdim=True)
        torch.testing.assert_close(adjusted_logits.softmax(1), normalized)
        torch.testing.assert_close(adjusted_log_probabilities.softmax(1), normalized)
        torch.testing.assert_close(adjusted_logits[:, other_classes], logits[:, other_classes], rtol=0, atol=0)
        torch.testing.assert_close(adjusted_probabilities[:, other_classes], probabilities[:, other_classes], rtol=0, atol=0)
        assert torch.equal(adjusted_logits.argmax(1), expected_probabilities.argmax(1))
        assert torch.equal(adjusted_probabilities.argmax(1), normalized.argmax(1))
        assert torch.equal(adjusted_log_probabilities.argmax(1), adjusted_probabilities.argmax(1))
        before, after = logits.argmax(1), adjusted_logits.argmax(1)
        changed = before != after
        assert torch.all((before[changed] == 3) | (after[changed] == 3))
        if alpha <= 1:
            assert torch.equal(after[before != 3], before[before != 3])
        if alpha == 1:
            assert adjusted_logits is logits
            assert adjusted_probabilities is probabilities
            assert adjusted_log_probabilities is log_probabilities
    assert torch.equal(logits, untouched)
    assert logits.argmax(1)[0, 0, 0].item() == 3
    assert adjust_mountain_scores(logits, mountain_alpha=0.8, tta="none").argmax(1)[0, 0, 0].item() == 2
    # Exact ties preserve torch.argmax's first-class-index rule.
    tie_logits = torch.tensor([-5.0, -4.0, 0.0, 1.0, -3.0]).reshape(1, 5, 1, 1)
    tie_probabilities = torch.tensor([0.0625, 0.0625, 0.25, 0.5, 0.125]).reshape(1, 5, 1, 1)
    assert adjust_mountain_scores(tie_logits, mountain_alpha=math.exp(-1), tta="none").argmax(1).item() == 2
    assert adjust_mountain_scores(tie_probabilities, mountain_alpha=0.5, tta="hflip").argmax(1).item() == 2
    for alpha in (0.0, -0.1, float("nan"), float("inf"), float("-inf")):
        for mode, scores in (("none", logits), ("hflip", probabilities)):
            try:
                adjust_mountain_scores(scores, mountain_alpha=alpha, tta=mode)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid mountain alpha accepted: {alpha} ({mode})")
    return {"alpha_values": alphas, "alpha_one_exact_identity": True,
            "negative_logits_downweighted_in_log_domain": True,
            "hflip_probabilities_scaled_directly": True,
            "normalization_preserves_argmax": True, "other_classes_unchanged": True,
            "non_mountain_classes_cannot_swap": True, "invalid_alpha_rejected": True,
            "argmax_ties_use_first_class_index": True, "input_scores_not_mutated": True}


def check_mountain_sweep() -> dict:
    """One loader pass must reuse predictions for all five global alphas."""
    alphas = [1.0, 0.95, 0.90, 0.85, 0.80]

    class SyntheticDataset(torch.utils.data.Dataset):
        input_size, partial = 16, True
        identity = {"source": "synthetic_calibration_verification"}

        def __init__(self):
            self.reads = 0
            self.items = []
            grid = torch.arange(256).reshape(16, 16)
            for index in range(3):
                image = torch.stack((grid.float() / 256 + index * 0.1,
                                     grid.remainder(16).float() / 16,
                                     torch.full((16, 16), float(index))))
                target = grid.remainder(5).long()
                if index == 0:
                    target[target == 3] = 0
                    target[8, 8] = 3  # One small-GT-area image: 1/256 mountain pixels.
                self.items.append({"image": image, "mask": target, "scene": "Moon_5"})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            self.reads += 1
            return self.items[index]

    def raw_logits(images):
        # Deliberately predict at 4x4; targets are 16x16.
        x = F.avg_pool2d(images[:, :1], 4)
        return torch.cat((0.1 + x * 0.5, 0.45 - x * 0.2, 0.3 + x * 0.1,
                          0.45 + x * 0.05, -0.4 + x * 0.15), dim=1)

    class CountingModel(torch.nn.Module):
        def __init__(self, log_probabilities=False):
            super().__init__()
            self.calls = 0
            self.log_probabilities = log_probabilities

        def forward(self, images):
            self.calls += 1
            logits = raw_logits(images)
            return {"semantic_logits": logits.log_softmax(1) if self.log_probabilities else logits}

    reports = []
    for log_probabilities in (False, True):
        for tta in ("none", "hflip"):
            dataset = SyntheticDataset()
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
            model = CountingModel(log_probabilities)
            sweep = evaluate_loader(model, loader, torch.device("cpu"), amp="none", tta=tta,
                                    mountain_alphas=alphas)
            expected_calls = len(loader) * (2 if tta == "hflip" else 1)
            assert model.calls == expected_calls
            assert dataset.reads == len(dataset)
            assert model.training
            metadata = sweep["mountain_score_sweep"]
            assert metadata["model_forward_calls"] == model.calls
            assert metadata["loader_passes"] == 1 and metadata["alphas"] == alphas
            assert metadata["recommended_alpha"] is None and not metadata["selection_valid"]
            candidates = {item["alpha"]: item for item in metadata["candidates"]}
            images = torch.stack([item["image"] for item in dataset.items])
            targets = torch.stack([item["mask"] for item in dataset.items])
            original = raw_logits(images)
            if log_probabilities:
                original = original.log_softmax(1)
            if tta == "none":
                scores = original
            else:
                flipped = raw_logits(images.flip(-1))
                if log_probabilities:
                    flipped = flipped.log_softmax(1)
                scores = (original.softmax(1) + flipped.softmax(1).flip(-1)) * 0.5
            scores = F.interpolate(scores, targets.shape[-2:], mode="bilinear", align_corners=False)
            baseline = scores.argmax(1)
            gt_mountain = targets == 3
            areas = gt_mountain.flatten(1).sum(1)
            small = (areas > 0) & (areas < 0.01 * 16 * 16)
            small_pixels = gt_mountain & small[:, None, None]
            assert small.sum().item() == 1 and small_pixels.sum().item() == 1
            for alpha in alphas:
                single_model = CountingModel(log_probabilities)
                single = evaluate_loader(single_model, loader, torch.device("cpu"), amp="none", tta=tta,
                                         mountain_alpha=alpha)
                assert single_model.calls == expected_calls
                assert candidates[alpha]["metrics"] == single
                if alpha == 1:
                    assert {key: value for key, value in sweep.items() if key != "mountain_score_sweep"} == single
                adjusted = scores.clone()
                if tta == "none":
                    adjusted[:, 3] += math.log(alpha)
                else:
                    adjusted[:, 3] *= alpha
                prediction = adjusted.argmax(1)
                matrix = ConfusionMatrix()
                matrix.update(targets, prediction)
                assert single["confusion_matrix_gt_rows_pred_cols"] == matrix.matrix.tolist()
                removed = (baseline == 3) & (prediction != 3)
                removed_false = removed & ~gt_mountain
                expected_diagnostics = {
                    "removed_true_mountain_pixels": int((removed & gt_mountain).sum()),
                    "removed_false_mountain_pixels": int(removed_false.sum()),
                    "corrected_false_mountain_pixels": int((removed_false & (prediction == targets)).sum()),
                    "false_mountain_changed_to_another_wrong_class": int((removed_false & (prediction != targets)).sum()),
                    "changed_previously_non_mountain_pixels": int(((baseline != 3) & (prediction != baseline)).sum()),
                    "small_gt_area_image_count": int(small.sum()),
                    "small_gt_area_gt_pixels": int(small_pixels.sum()),
                    "small_gt_area_true_positive_pixels": int((small_pixels & (prediction == 3)).sum()),
                    "small_gt_area_lost_true_positive_pixels": int((small_pixels & removed).sum()),
                    "small_gt_area_recall": float((small_pixels & (prediction == 3)).sum() / small_pixels.sum()),
                }
                assert candidates[alpha]["diagnostics"] == expected_diagnostics
            reports.append({"score_type": "region_log_probabilities" if log_probabilities else "pixel_logits",
                            "tta_mode": tta, "image_count": len(dataset), "batch_count": len(loader),
                            "actual_sweep_forward_calls": model.calls,
                            "all_five_alphas_match_independent_single_alpha_confusion_matrices": True,
                            "alpha_one_matches_original_result": True, "single_dataset_pass": True,
                            "continuous_resize_before_argmax": True, "diagnostics_match_pixel_counts": True})
    return {"alpha_values": alphas, "cases": reports}


def check_hard_pixels() -> dict:
    """Numerical CPU checks with explicit probabilities and independent oracles."""
    from unittest.mock import patch
    from lunarseg.losses import HARD_PIXEL_DIRECTIONS, confusion_hard_pixel_loss

    directions = ((0, 3), (3, 0), (2, 0), (0, 2), (3, 1), (1, 3))
    assert HARD_PIXEL_DIRECTIONS == directions

    def make_case(counts, crowd=False):
        batch = len(counts)
        width = max(sum(row) for row in counts) + 4
        probabilities = torch.full((batch, 5, 1, width), 0.015)
        probabilities[:, 4] = 0.94
        targets = torch.full((batch, 1, width), 4, dtype=torch.long)
        buckets = {}
        for item, item_counts in enumerate(counts):
            position = 0
            for index, ((truth, prediction), count) in enumerate(zip(directions, item_counts)):
                bucket = []
                for offset in range(count):
                    # Increasing true-class probability makes CE strictly fall.
                    # The expected top-k points are therefore the first k, with
                    # no topk/sort/mining implementation used by the oracle.
                    if crowd:
                        true_probability = (0.0001 + offset / count * 0.0009 if index == 0
                                            else 0.1 + offset / count * 0.1)
                    else:
                        true_probability = 0.025 + offset * 0.004 + index * 0.003 + item * 0.002
                    probability = torch.full((5,), (1 - true_probability - 0.72) / 3)
                    probability[truth], probability[prediction] = true_probability, 0.72
                    probabilities[item, :, 0, position] = probability
                    targets[item, 0, position] = truth
                    bucket.append((position, true_probability))
                    position += 1
                buckets[item, index] = bucket
        return probabilities, targets, buckets

    # Different image/group sizes and empty groups distinguish the specified
    # fixed B*6 reduction from a global pixel mean or nonempty-group mean.
    counts = ((13, 3, 1, 0, 4, 2), (1, 0, 4, 2, 3, 1))
    probabilities, targets, buckets = make_case(counts)
    logits = probabilities.log().requires_grad_()
    quota, batch = 2, len(counts)
    expected_loss, selected_points = 0.0, set()
    expected_gradient = torch.zeros_like(logits)
    expected_stats = {}
    for index, (truth, prediction) in enumerate(directions):
        expected_candidates = expected_selected = 0
        selected_ce_sum = 0.0
        for item in range(batch):
            chosen = buckets[item, index][:quota]
            expected_candidates += len(buckets[item, index])
            expected_selected += len(chosen)
            if not chosen:
                continue
            expected_loss += sum(-math.log(p) for _, p in chosen) / len(chosen) / (batch * 6)
            for position, probability in chosen:
                selected_points.add((item, 0, position))
                selected_ce_sum += -math.log(probability)
                derivative = probabilities[item, :, 0, position].clone()
                derivative[truth] -= 1
                expected_gradient[item, :, 0, position] = derivative / (batch * 6 * len(chosen))
        expected_stats[f"{truth}->{prediction}"] = (expected_candidates, expected_selected, selected_ce_sum)
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    loss, stats, examples = confusion_hard_pixel_loss(logits, targets, quota=quota,
                                                     example_limit_per_direction=256)
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state
    torch.testing.assert_close(loss, torch.tensor(expected_loss), rtol=2e-6, atol=2e-6)
    assert {(row["batch_index"], row["y"], row["x"]) for row in examples} == selected_points
    for direction, (candidates, selected, ce_sum) in expected_stats.items():
        assert stats[direction]["candidate_count"] == candidates
        assert stats[direction]["selected_count"] == selected
        assert math.isclose(stats[direction]["selected_ce_sum"], ce_sum, rel_tol=2e-6)
    for row in examples:
        item, y, x = row["batch_index"], row["y"], row["x"]
        assert row["gt"] == targets[item, y, x].item()
        assert row["pred"] == probabilities[item, :, y, x].argmax().item()
        assert row["direction"] == f"{row['gt']}->{row['pred']}"
        assert all(not isinstance(value, torch.Tensor) for value in row.values())
    loss.backward()
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=2e-5, atol=2e-7)
    actual_points = {tuple(index) for index in (logits.grad.abs().sum(1) > 0).nonzero().tolist()}
    assert actual_points == selected_points
    reduction_report = {"batch_size": batch, "quota": quota, "raw_loss": float(loss.detach()),
                        "independent_nll_expectation": expected_loss,
                        "selected_pixels": len(selected_points), "empty_image_direction_groups": 2,
                        "fixed_image_direction_mean_including_empty_groups": True,
                        "analytical_selected_ce_gradients_match": True,
                        "unselected_pixel_gradients_exactly_zero": True,
                        "torch_and_python_rng_unchanged": True}

    # All 1024 majority-class mistakes have higher CE than the other groups.
    # A global hard-pixel pool would crowd them out; six independent quotas must
    # still select exactly 256 candidates in EACH confusion direction.
    crowded_probabilities, crowded_targets, _ = make_case(((1024, 300, 300, 300, 300, 300),), crowd=True)
    crowded_loss, crowded_stats, _ = confusion_hard_pixel_loss(crowded_probabilities.log(), crowded_targets,
                                                              quota=256, example_limit_per_direction=0)
    assert torch.isfinite(crowded_loss)
    assert [row["selected_count"] for row in crowded_stats.values()] == [256] * 6

    # Compute query-class/mask mixture exactly as required by the region head,
    # then compare mining gradients with an independent NLL objective.
    query_classes = torch.full((1, 4, 6), -3.0)
    query_classes[0, torch.arange(4), torch.arange(4)] = 4.0
    query_classes.requires_grad_()
    query_masks = torch.full((1, 4, 2, 3), -4.0)
    region_targets = torch.tensor([truth for truth, _ in directions]).reshape(1, 2, 3)
    for position, (_, prediction) in enumerate(directions):
        query_masks[0, prediction, position // 3, position % 3] = 4.0
    query_masks.requires_grad_()

    def region_log_scores(classes, masks):
        evidence = torch.einsum("bqc,bqhw->bchw", classes.softmax(-1)[..., :5], masks.sigmoid())
        return (evidence / evidence.sum(1, keepdim=True)).log()

    log_scores = region_log_scores(query_classes, query_masks)
    torch.testing.assert_close(log_scores.exp().sum(1), torch.ones_like(region_targets, dtype=torch.float32))
    assert torch.equal(log_scores.argmax(1), torch.tensor([p for _, p in directions]).reshape(1, 2, 3))
    ce = F.cross_entropy(log_scores, region_targets, reduction="none")
    nll = F.nll_loss(log_scores, region_targets, reduction="none")
    torch.testing.assert_close(ce, nll, rtol=1e-6, atol=1e-6)
    region_loss, _, _ = confusion_hard_pixel_loss(log_scores, region_targets, quota=1)
    torch.testing.assert_close(region_loss, nll.mean(), rtol=1e-6, atol=1e-6)
    actual_query_gradients = torch.autograd.grad(region_loss, (query_classes, query_masks))
    independent_classes = query_classes.detach().clone().requires_grad_()
    independent_masks = query_masks.detach().clone().requires_grad_()
    independent_log_scores = region_log_scores(independent_classes, independent_masks)
    reference_nll = -independent_log_scores.gather(1, region_targets.unsqueeze(1)).mean()
    expected_query_gradients = torch.autograd.grad(reference_nll, (independent_classes, independent_masks))
    for actual, expected in zip(actual_query_gradients, expected_query_gradients):
        assert actual.isfinite().all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    # A vertical GT split at x=6 has a radius-r boundary band x=6-r..5+r.
    boundary_targets = torch.zeros(1, 9, 13, dtype=torch.long)
    boundary_targets[:, :, 6:] = 3
    boundary_probabilities = torch.full((1, 5, 9, 13), 0.015)
    boundary_probabilities.scatter_(1, boundary_targets.unsqueeze(1), 0.94)
    mistakes = [(1, 0, 3, .05), (3, 0, 3, .02), (5, 0, 3, .01),
                (6, 3, 0, .01), (8, 3, 0, .02), (11, 3, 0, .05)]
    for x, truth, prediction, true_probability in mistakes:
        boundary_probabilities[0, :, 4, x] = (1 - true_probability - 0.72) / 3
        boundary_probabilities[0, truth, 4, x] = true_probability
        boundary_probabilities[0, prediction, 4, x] = .72
    boundary_report = []
    for radius, expected_count in ((1, 1), (3, 2)):
        _, boundary_stats, boundary_examples = confusion_hard_pixel_loss(
            boundary_probabilities.log(), boundary_targets, quota=2, boundary_radius=radius,
            example_limit_per_direction=256)
        for key in ("0->3", "3->0"):
            assert boundary_stats[key]["candidate_boundary_count"] == expected_count
            assert boundary_stats[key]["selected_boundary_count"] == expected_count
        for example in boundary_examples:
            assert example["boundary"] == (6 - radius <= example["x"] <= 5 + radius)
        boundary_report.append({"radius": radius, "expected_boundary_count_per_active_direction": expected_count})
    homogeneous_targets = torch.zeros(1, 3, 4, dtype=torch.long)
    homogeneous_scores = torch.full((1, 5, 3, 4), -2.0)
    homogeneous_scores[:, 3] = 2.0  # Every pixel is wrong, but GT has no semantic boundary.
    _, homogeneous_stats, homogeneous_examples = confusion_hard_pixel_loss(
        homogeneous_scores, homogeneous_targets, quota=256, boundary_radius=3,
        example_limit_per_direction=256)
    assert homogeneous_stats["0->3"]["candidate_count"] == 12
    assert homogeneous_stats["0->3"]["candidate_boundary_count"] == 0
    assert not any(example["boundary"] for example in homogeneous_examples)

    # Empty directions and quota zero return differentiable exact zero.
    for quota_value, error_free in ((256, True), (0, False)):
        scores = torch.zeros(1, 5, 3, 4)
        scores[:, 0 if error_free else 3] = 3.0
        scores.requires_grad_()
        zero, zero_stats, zero_examples = confusion_hard_pixel_loss(scores, homogeneous_targets, quota=quota_value)
        assert zero.requires_grad and zero.item() == 0 and not zero_examples
        assert all(not any(values.values()) for values in zero_stats.values())
        zero.backward()
        assert scores.grad is not None and torch.equal(scores.grad, torch.zeros_like(scores))

    # Fixture captured with the unchanged, pre-hard-pixel criterion. Its exact
    # losses, all leaf gradient bytes, and final RNG verify lambda=0 compatibility
    # against the OLD computation, not merely two calls to the new helper.
    legacy_losses = {"loss_semantic_ce": 1.981998085975647, "loss_semantic_dice": 0.7075377106666565,
                     "loss_region_class": 1.8356012105941772, "loss_region_mask": 0.7129302024841309,
                     "loss_region_dice": 0.6840599179267883, "loss_auxiliary": 8.3510103225708,
                     "loss_total": 14.784954071044922}
    legacy_gradients = {
        "semantic_logits": "b5ca8af9a0fe238c4f9c3f1510b716fcb1fbdac391a024742af9ea779fcde9fb",
        "pred_logits": "f8baccf26cb7dcad1601ff8bf71d5e6daced828469a6e7c04b3896535f75a590",
        "pred_masks": "e06602e7ecf8cf763bd9494f45ff8d32950673b611f38bf8d55126b9e4d5c45a",
        "aux_pred_logits": "1a2ba7af1e1ffa3a069e990b70b7d70c58a11000e40ca2eb51da0498e27bc074",
        "aux_pred_masks": "d0baa2bf26387a398e8474813a9405b34deb7da6ca1414c2583ffe898cfe0c45",
    }
    values = {"semantic_logits": torch.linspace(-1.1, 1.7, 80).reshape(1, 5, 4, 4).requires_grad_(),
              "pred_logits": torch.linspace(-.8, 1.3, 30).reshape(1, 5, 6).requires_grad_(),
              "pred_masks": torch.linspace(-1.4, 1.2, 20).reshape(1, 5, 2, 2).requires_grad_()}
    auxiliary = {"pred_logits": torch.linspace(-.5, 1.6, 30).reshape(1, 5, 6).requires_grad_(),
                 "pred_masks": torch.linspace(-.9, 1.5, 20).reshape(1, 5, 2, 2).requires_grad_()}
    torch.manual_seed(271828)
    with patch("lunarseg.losses.confusion_hard_pixel_loss", side_effect=AssertionError("lambda=0 must not mine")):
        legacy_actual = SegmentationCriterion(num_points=32, hard_pixel_weight=0)(
            {**values, "aux_outputs": [auxiliary]}, torch.arange(16).remainder(5).reshape(1, 4, 4))
    assert {key: float(value.detach()) for key, value in legacy_actual.items()} == legacy_losses
    assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() == "cbebbd3721d70feb686be69a2dc42f129af6bb3a9685c16de548b1e09c2f5bad"
    legacy_actual["loss_total"].backward()
    for key, value in {**values, **{"aux_" + key: value for key, value in auxiliary.items()}}.items():
        assert hashlib.sha256(value.grad.numpy().tobytes()).hexdigest() == legacy_gradients[key], key

    baseline = SegmentationCriterion(hard_pixel_weight=0)({"semantic_logits": logits.detach()}, targets)
    weighted_criterion = SegmentationCriterion(hard_pixel_weight=.2, hard_pixel_quota=quota)
    weighted = weighted_criterion({"semantic_logits": logits.detach()}, targets)
    for key in baseline:
        if key != "loss_total":
            assert torch.equal(weighted[key], baseline[key])
    assert torch.equal(weighted["loss_hard_pixels"], loss.detach())
    assert torch.equal(weighted["loss_hard_pixels_weighted"], loss.detach() * .2)
    assert torch.equal(weighted["loss_total"], baseline["loss_total"] + .2 * loss.detach())
    assert weighted_criterion.latest_hard_pixel_stats == stats

    rejected = 0
    for field in ("quota", "boundary_radius", "example_limit_per_direction"):
        for value in (-1, .5, True):
            try:
                confusion_hard_pixel_loss(logits.detach(), targets, **{field: value})
            except (ValueError, TypeError):
                rejected += 1
            else:
                raise AssertionError(f"Invalid hard-pixel parameter accepted: {field}={value}")
    for value in (-.1, float("nan"), float("inf"), float("-inf"), True):
        try:
            SegmentationCriterion(hard_pixel_weight=value)
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError(f"Invalid hard_pixel_weight accepted: {value}")
    for value in (-1, .5, True):
        try:
            SegmentationCriterion(hard_pixel_quota=value)
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError(f"Invalid hard_pixel_quota accepted: {value}")
    invalid_arrays = ((torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)),
                      (torch.zeros(1, 5, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, dtype=torch.long)),
                      (torch.zeros(1, 5, 2, 2), torch.zeros(1, 2, 2)),
                      (torch.zeros(2, 5, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)))
    for invalid_scores, invalid_targets in invalid_arrays:
        try:
            confusion_hard_pixel_loss(invalid_scores, invalid_targets)
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError("Invalid hard-pixel tensor shape/dtype accepted")

    return {"directions": [f"{truth}->{prediction}" for truth, prediction in directions],
            "numerical_reduction_and_selected_gradients": reduction_report,
            "majority_candidates": 1024, "independent_quota": 256,
            "selected_per_direction_under_majority_crowding": [256] * 6,
            "region_log_probabilities_match_ce_and_nll": True,
            "region_query_class_and_mask_gradients_match_independent_nll": True,
            "region_gradient_l1": {"query_class": float(actual_query_gradients[0].abs().sum()),
                                   "query_mask": float(actual_query_gradients[1].abs().sum())},
            "boundary_counts_match_independent_geometry": boundary_report,
            "homogeneous_gt_image_edges_are_not_boundaries": True,
            "empty_mistakes_and_zero_quota_have_graph_connected_zero": True,
            "lambda_zero_exact_legacy_loss_gradient_and_rng_fixture": True,
            "legacy_fixture_torch_version": "2.1.1+cu121", "legacy_loss_total": legacy_losses["loss_total"],
            "lambda_point_two_only_adds_weighted_hard_loss": True,
            "invalid_parameters_and_tensor_cases_rejected": rejected}


def check_matching() -> dict:
    torch.manual_seed(42)
    criterion = SegmentationCriterion(num_points=512)
    # Ring-shaped and disconnected target semantics must remain representable.
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[:, 8:56, 8:56] = 2
    target[:, 16:48, 16:48] = 0
    target[:, 22:30, 22:30] = 1
    target[:, :4, :] = 4
    target[:, 48:56, 2:6] = 3
    output = {
        "semantic_logits": torch.randn(1, 5, 64, 64, requires_grad=True),
        "pred_logits": torch.randn(1, 32, 6, requires_grad=True),
        "pred_masks": torch.randn(1, 32, 16, 16, requires_grad=True),
    }
    torch.manual_seed(11)
    loss = criterion(output, target)["loss_total"]
    permutation = torch.randperm(32)
    permuted = {
        "semantic_logits": output["semantic_logits"],
        "pred_logits": output["pred_logits"][:, permutation],
        "pred_masks": output["pred_masks"][:, permutation],
    }
    torch.manual_seed(11)
    other = criterion(permuted, target)["loss_total"]
    torch.testing.assert_close(loss, other, atol=2e-6, rtol=1e-6)
    loss.backward()
    for value in output.values():
        assert value.grad is not None and value.grad.isfinite().all() and value.grad.abs().sum() > 0
    return {"query_permutation_invariant": True, "semantic_class_mask_gradients": True}


def check_data(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    samples = manifest["samples"]
    scenes = {
        split: sorted({sample["scene"] for sample in samples if sample["split"] == split})
        for split in ("train", "val")
    }
    assert scenes == {"train": ["Moon_1", "Moon_3", "Moon_6", "Moon_7", "Moon_8"], "val": ["Moon_5"]}
    counts = Counter(sample["split"] for sample in samples)
    assert counts["train"] == 6403 and counts["val"] == 1599
    for split in ("train", "val"):
        sample = next(sample for sample in samples if sample["split"] == split)
        with Image.open(sample["image"]) as image, Image.open(sample["mask"]) as mask:
            assert image.size == mask.size == (1024, 1024)
            ids = np.asarray(mask)
            assert ids.ndim == 2 and ids.min() >= 0 and ids.max() <= 4
    return {"manifest": str(path), "scenes": scenes, "counts": dict(counts), "id_mask_samples_checked": 2}


def check_random_split(path: Path, *, production: bool = True, read_images: bool = True) -> dict:
    """Audit every saved identity independently, then exercise literal replay."""
    from unittest.mock import patch
    from lunarseg.data import (CLASS_NAMES, TEST_SCENES, TRAIN_SCENES, VAL_SCENES,
                               LunarDataset, resolve_record_path, sha256_file)

    path = path.resolve()
    saved = json.loads(path.read_text())
    assert saved.get("schema") == "lusnar_random_split_v1"
    assert saved["class_names"] == list(CLASS_NAMES)
    assert saved["split_protocol"] == "random_original_source_family_4_to_1"
    assert type(saved["seed"]) is int and saved["seed"] >= 0
    source_path = Path(saved["source_manifest"]).resolve()
    augmented_path = Path(saved["augmented_manifest"]).resolve()
    assert sha256_file(source_path) == saved["source_manifest_sha256"]
    assert sha256_file(augmented_path) == saved["augmented_manifest_sha256"]
    source = json.loads(source_path.read_text())
    augmented = json.loads(augmented_path.read_text())
    assert source["class_names"] == augmented["class_names"] == list(CLASS_NAMES)
    allowed_scenes = TRAIN_SCENES | VAL_SCENES
    originals = {}
    for row in source["samples"]:
        assert row["scene"] in allowed_scenes and row["scene"] not in TEST_SCENES
        assert row["split"] == ("train" if row["scene"] in TRAIN_SCENES else "val")
        source_id = f"{row['scene']}/{row['sample_id']}"
        assert source_id not in originals
        originals[source_id] = {
            "image": str(resolve_record_path(source_path.parent, row["image"]).resolve()),
            "mask": str(resolve_record_path(source_path.parent, row["mask"]).resolve()),
            "scene": row["scene"], "sample_id": str(row["sample_id"]), "source_id": source_id,
            "source_kind": "original"}
    rng = random.Random(saved["seed"])
    original_ids = sorted(originals)
    rng.shuffle(original_ids)
    cutoff = len(original_ids) * 4 // 5
    train_ids, val_ids = original_ids[:cutoff], original_ids[cutoff:]
    train_id_set = set(train_ids)
    assert saved["original_split"] == {"train_source_ids": train_ids, "val_source_ids": val_ids}
    assert not set(train_ids) & set(val_ids)
    pools = {"crater": [], "mountain_rock": []}
    seen_aug = set()
    for row in augmented["samples"]:
        assert row["split"] == "train"
        family = f"{row['source_scene']}/{row['source_sample_id']}"
        assert family in originals
        group = row.get("primary_group", row.get("focus_group"))
        if group not in pools or family not in train_id_set:
            continue
        key = (row["source_scene"], str(row["sample_id"]))
        assert key not in seen_aug
        seen_aug.add(key)
        pools[group].append({"image": str(resolve_record_path(augmented_path.parent, row["image"]).resolve()),
                             "mask": str(resolve_record_path(augmented_path.parent, row["mask"]).resolve()),
                             "scene": row["source_scene"], "sample_id": str(row["sample_id"]),
                             "source_id": family, "source_kind": "augmented", "augmentation_group": group})
    expected_train = [originals[source_id] for source_id in train_ids]
    expected_val = [originals[source_id] for source_id in val_ids]
    for group in ("crater", "mountain_rock"):
        pools[group].sort(key=lambda row: (row["scene"], str(row["sample_id"])))
        expected_train.extend(rng.sample(pools[group], saved["augmentation_counts"][group]))
    rng.shuffle(expected_train)
    fields = ("source_kind", "scene", "sample_id", "source_id", "image", "mask", "augmentation_group")

    def identities(rows):
        return [tuple(row.get(field) for field in fields) for row in rows]

    train, val = saved["splits"]["train"], saved["splits"]["val"]
    assert identities(train) == identities(expected_train)
    assert identities(val) == identities(expected_val)
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in val} == {"val"}
    assert all(row["source_kind"] == "original" for row in val)
    for split_rows in (train, val):
        keys = [(row["source_kind"], row["scene"], str(row["sample_id"])) for row in split_rows]
        assert len(keys) == len(set(keys))
        paths = [(row["image"], row["mask"]) for row in split_rows]
        assert len(paths) == len(set(paths))
        assert all(row["scene"] in allowed_scenes and row["scene"] not in TEST_SCENES for row in split_rows)
        assert all(Path(row[field]).is_absolute() for row in split_rows for field in ("image", "mask"))
    assert not {row["source_id"] for row in train} & {row["source_id"] for row in val}
    assert not {row["image"] for row in train} & {row["image"] for row in val}
    assert not {row["mask"] for row in train} & {row["mask"] for row in val}
    observed_counts = {"original_pool": len(originals), "train_original": len(train_ids),
                       "val_original": len(val_ids), "train_augmented": len(train) - len(train_ids),
                       "train_total": len(train), "val_total": len(val)}
    assert saved["counts"] == observed_counts
    assert dict(Counter(row["augmentation_group"] for row in train if row["source_kind"] == "augmented")) == saved["augmentation_counts"]
    splits_hash = hashlib.sha256(json.dumps(saved["splits"], ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    assert splits_hash == saved["splits_sha256"]
    canonical_hash = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                             separators=(",", ":")).encode()).hexdigest()
    assert saved["ordered_samples_sha256"] == {split: canonical_hash(saved["splits"][split]) for split in ("train", "val")}
    assert saved["original_split_sha256"] == canonical_hash(saved["original_split"])
    if production:
        assert observed_counts == {"original_pool": 8002, "train_original": 6401,
                                   "val_original": 1601, "train_augmented": 3000,
                                   "train_total": 9401, "val_total": 1601}
        assert saved["augmentation_counts"] == {"crater": 1500, "mountain_rock": 1500}
        assert saved["mask_validation"]["checked_mask_count"] == len(train) + len(val)
        assert saved["mask_validation"]["required_mode"] == "L"
        assert saved["mask_validation"]["class_id_range"] == [0, 4]
    # Loading must honor the literal list, without even consulting an RNG.
    replay_torch_rng, replay_python_rng = torch.get_rng_state().clone(), random.getstate()
    with patch("random.Random", side_effect=AssertionError("Dataset replay must not resample")), \
            patch("random.sample", side_effect=AssertionError("Dataset replay must not resample")), \
            patch("random.shuffle", side_effect=AssertionError("Dataset replay must not resample")):
        datasets = {split: LunarDataset(path, split=split, input_size=32) for split in ("train", "val")}
    assert torch.equal(torch.get_rng_state(), replay_torch_rng) and random.getstate() == replay_python_rng
    checked = []
    for split, dataset in datasets.items():
        assert not dataset.partial and dataset.complete_count == len(saved["splits"][split])
        assert identities(dataset.records) == identities(saved["splits"][split])
        if not read_images:
            continue
        groups = {}
        for index, record in enumerate(dataset.records):
            groups.setdefault((record["scene"], record.get("augmentation_group", "original")), index)
        for (scene, group), index in sorted(groups.items()):
            record = dataset.records[index]
            with Image.open(record["image"]) as image, Image.open(record["mask"]) as mask:
                assert image.size == mask.size and image.size[0] > 0
                assert mask.mode == "L"
                label = np.asarray(mask)
                assert label.ndim == 2 and label.size > 0 and label.min() >= 0 and label.max() <= 4
                original_size = list(image.size)
            loaded = dataset[index]
            assert loaded["image"].shape == (3, 32, 32) and loaded["image"].isfinite().all()
            assert loaded["mask"].shape == (32, 32) and loaded["mask"].dtype == torch.long
            assert loaded["mask"].min() >= 0 and loaded["mask"].max() <= 4
            checked.append({"split": split, "scene": scene, "group": group, "sample_id": record["sample_id"],
                            "image": record["image"], "mask": record["mask"], "original_size": original_size,
                            "class_ids": np.unique(label).tolist()})
    return {"manifest": str(path), "manifest_sha256": sha256_file(path), "seed": saved["seed"],
            "counts": observed_counts, "augmentation_counts": saved["augmentation_counts"],
            "source_manifests_sha256_match": True, "all_original_families_partitioned_once": True,
            "augmentation_families_only_in_new_train": True, "test_scenes_absent": True,
            "seed_reproduction_matches_all_saved_sample_identities_and_order": True,
            "literal_replay_uses_no_random_sampling": True, "splits_sha256": splits_hash,
            "class_names": list(CLASS_NAMES), "full_mask_scan": False,
            "producer_full_mask_validation": saved.get("mask_validation"),
            "real_rgb_and_class_id_mask_reads": checked}


def check_random_split_rejections() -> dict:
    """Build a tiny independent legal manifest and deliberately corrupt it."""
    from copy import deepcopy
    from lunarseg.data import CLASS_NAMES, LunarDataset, sha256_file

    with tempfile.TemporaryDirectory(prefix="fenge-split-check-") as temporary:
        root = Path(temporary)
        def images(name):
            image_path, mask_path = root / f"{name}-rgb.png", root / f"{name}-ids.png"
            Image.new("RGB", (32, 32), (80, 90, 100)).save(image_path)
            Image.fromarray((np.indices((32, 32)).sum(0) % 5).astype(np.uint8), mode="L").save(mask_path)
            return str(image_path), str(mask_path)
        source_rows, originals = [], {}
        for index in range(14):
            scene = "Moon_1" if index < 10 else "Moon_5"
            sid = str(index)
            image, mask = images(f"original-{index}")
            source_rows.append({"scene": scene, "sample_id": sid, "split": "train" if index < 10 else "val",
                                "image": image, "mask": mask})
            family = f"{scene}/{sid}"
            originals[family] = {"scene": scene, "sample_id": sid, "source_id": family,
                                 "image": image, "mask": mask, "source_kind": "original"}
        source_path, augmented_path = root / "original.json", root / "augmented.json"
        source_path.write_text(json.dumps({"class_names": list(CLASS_NAMES), "samples": source_rows}))
        aug_rows = []
        for group in ("crater", "mountain_rock"):
            for index in range(10):
                image, mask = images(f"{group}-{index}")
                aug_rows.append({"source_scene": "Moon_1", "source_sample_id": str(index),
                                 "sample_id": f"{group}-{index}", "split": "train", "primary_group": group,
                                 "image": image, "mask": mask})
        augmented_path.write_text(json.dumps({"class_names": list(CLASS_NAMES),
                                              "source_manifest": str(source_path),
                                              "source_manifest_sha256": sha256_file(source_path),
                                              "validation_or_test_used": False, "samples": aug_rows}))
        seed, rng = 29, random.Random(29)
        families = sorted(originals)
        rng.shuffle(families)
        cutoff = len(families) * 4 // 5
        train_ids, val_ids = families[:cutoff], families[cutoff:]
        train = [{**originals[family], "split": "train"} for family in train_ids]
        val = [{**originals[family], "split": "val"} for family in val_ids]
        for group in ("crater", "mountain_rock"):
            candidates = [{"scene": row["source_scene"], "sample_id": row["sample_id"],
                           "source_id": f"{row['source_scene']}/{row['source_sample_id']}",
                           "image": row["image"], "mask": row["mask"], "source_kind": "augmented",
                           "augmentation_group": group, "split": "train"}
                          for row in aug_rows if row["primary_group"] == group
                          and f"{row['source_scene']}/{row['source_sample_id']}" in train_ids]
            candidates.sort(key=lambda row: (row["scene"], str(row["sample_id"])))
            train.extend(rng.sample(candidates, 2))
        rng.shuffle(train)
        valid = {"schema": "lusnar_random_split_v1", "class_names": list(CLASS_NAMES),
                 "split_protocol": "random_original_source_family_4_to_1", "seed": seed,
                 "source_manifest": str(source_path), "source_manifest_sha256": sha256_file(source_path),
                 "augmented_manifest": str(augmented_path), "augmented_manifest_sha256": sha256_file(augmented_path),
                 "original_split": {"train_source_ids": train_ids, "val_source_ids": val_ids},
                 "splits": {"train": train, "val": val}, "augmentation_counts": {"crater": 2, "mountain_rock": 2},
                 "counts": {"original_pool": 14, "train_original": 11, "val_original": 3,
                            "train_augmented": 4, "train_total": 15, "val_total": 3}}
        def save(name, value, refresh=True):
            if refresh:
                canonical_hash = lambda rows: hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                value["splits_sha256"] = canonical_hash(value["splits"])
                value["ordered_samples_sha256"] = {split: canonical_hash(value["splits"][split]) for split in ("train", "val")}
                value["original_split_sha256"] = canonical_hash(value["original_split"])
            target = root / f"{name}.json"
            target.write_text(json.dumps(value))
            return target
        valid_path = save("valid", valid)
        check_random_split(valid_path, production=False)
        variants = {}
        variants["duplicate_sample"] = deepcopy(valid)
        original_indices = [i for i, row in enumerate(train) if row["source_kind"] == "original"]
        variants["duplicate_sample"]["splits"]["train"][original_indices[1]] = deepcopy(train[original_indices[0]])
        variants["family_leak"] = deepcopy(valid)
        aug_index = next(i for i, row in enumerate(train) if row["source_kind"] == "augmented")
        variants["family_leak"]["splits"]["train"][aug_index]["source_id"] = val_ids[0]
        variants["path_tamper"] = deepcopy(valid)
        variants["path_tamper"]["splits"]["train"][0]["image"] = val[0]["image"]
        variants["test_scene"] = deepcopy(valid)
        variants["test_scene"]["splits"]["train"][0]["scene"] = "Moon_2"
        variants["validation_augmentation"] = deepcopy(valid)
        variants["validation_augmentation"]["splits"]["val"][0] = {**train[aug_index], "split": "val"}
        variants["source_sha_tamper"] = deepcopy(valid)
        variants["source_sha_tamper"]["source_manifest_sha256"] = "0" * 64
        variants["splits_sha_tamper"] = deepcopy(valid)
        variants["splits_sha_tamper"]["splits"]["train"] = list(reversed(train))
        variants["class_order_tamper"] = deepcopy(valid)
        variants["class_order_tamper"]["class_names"] = list(reversed(CLASS_NAMES))
        rejected = []
        for name, invalid in variants.items():
            invalid_path = save(name, invalid, refresh=name != "splits_sha_tamper")
            for split in ("train", "val"):
                try:
                    LunarDataset(invalid_path, split=split, input_size=32)
                except (ValueError, TypeError, KeyError):
                    continue
                raise AssertionError(f"Invalid random split accepted by {split} loader: {name}")
            rejected.append(name)
        for split in ("train", "val"):
            dataset = LunarDataset(valid_path, split=split, input_size=32)
            mask_path = Path(dataset.records[0]["mask"])
            original_bytes = mask_path.read_bytes()
            try:
                for mode, value in (("L", 5), ("P", 0)):
                    Image.new(mode, (32, 32), value).save(mask_path)
                    try:
                        dataset[0]
                    except ValueError:
                        continue
                    raise AssertionError(f"Invalid mask accepted by {split}: mode={mode}, value={value}")
            finally:
                mask_path.write_bytes(original_bytes)
        return {"independent_synthetic_valid_manifest_replayed": True,
                "corrupt_manifests_rejected_by_both_splits": rejected,
                "invalid_class_ids_and_palette_mode_rejected_by_both_splits": True}


def check_model(decoder: str, args: argparse.Namespace, device: torch.device) -> dict:
    start = time.monotonic()
    config = ModelConfig(decoder=decoder, input_size=args.size, weight_dir=str(args.model_dir.resolve()))
    model = build_model(config, pretrained=True).to(device)
    actual_inputs = {}
    hooks = []
    for name in ("convnext", "mobile"):
        module = getattr(model.encoder, name)
        def record(_module, inputs, name=name):
            actual_inputs[name] = list(inputs[0].shape)
        hooks.append(module.register_forward_pre_hook(record))
    model.freeze_backbones(True)
    model.train()
    encoder_parameters = list(model.encoder.parameters())
    # Fusion layers live in encoder too; check actual CNN branches explicitly.
    for name in ("convnext", "mobile"):
        branch = getattr(model.encoder, name)
        assert not any(parameter.requires_grad for parameter in branch.parameters())
        assert not any(module.training for module in branch.modules() if isinstance(module, torch.nn.BatchNorm2d))
    model.freeze_backbones(False)
    model.train()
    images = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    target = torch.randint(0, 5, (args.batch_size, args.size, args.size), device=device)
    criterion = SegmentationCriterion(num_points=1024, hard_pixel_weight=args.hard_pixel_weight,
                                      hard_pixel_quota=args.hard_pixel_quota).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
    with context:
        output = model(images)
        losses = criterion(output, target)
    assert output["semantic_logits"].shape == (args.batch_size, 5, args.size, args.size)
    assert output["semantic_logits"].isfinite().all()
    assert actual_inputs == {"convnext": [args.batch_size, 3, args.size, args.size], "mobile": [args.batch_size, 3, args.size, args.size]}
    if decoder == "region":
        assert output["pred_logits"].shape == (args.batch_size, 32, 6)
        assert output["pred_masks"].shape == (args.batch_size, 32, args.size // 4, args.size // 4)
    assert losses["loss_total"].isfinite()
    losses["loss_total"].backward()
    gradients = {}
    for name in ("convnext", "mobile"):
        gradients[name] = sum(float(parameter.grad.detach().float().abs().sum()) for parameter in getattr(model.encoder, name).parameters() if parameter.grad is not None)
        assert gradients[name] > 0 and np.isfinite(gradients[name])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.step()
    with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=args.output.parent) as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.pt"
        torch.save({"model_config": config.to_dict(), "model": model.state_dict()}, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        restored_config = ModelConfig.from_dict(loaded["model_config"])
        assert restored_config.to_dict() == config.to_dict()
        restored = build_model(restored_config, pretrained=False)
        restored.load_state_dict(loaded["model"], strict=True)
        del loaded, restored
    report = {
        "decoder": decoder, "config": config.to_dict(), "input_shapes": actual_inputs,
        "semantic_shape": list(output["semantic_logits"].shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "branch_gradient_l1": gradients,
        "hard_pixel_weight": args.hard_pixel_weight, "hard_pixel_quota": args.hard_pixel_quota,
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "checkpoint_strict_roundtrip": True, "optimizer_step": True,
        "seconds": round(time.monotonic() - start, 3),
    }
    if args.hard_pixel_weight > 0:
        report["hard_pixel_stats"] = criterion.latest_hard_pixel_stats
        report["hard_pixel_examples"] = criterion.latest_hard_examples
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    for hook in hooks:
        hook.remove()
    del model, optimizer, encoder_parameters, output, losses, criterion, images, target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decoder", choices=("both", "pixel", "region"), default="both")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-manifest", type=Path,
                        help="Saved lusnar_random_split_v1 manifest to audit with --split-only")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hard-pixel-weight", type=float, default=0.0)
    parser.add_argument("--hard-pixel-quota", type=int, default=256)
    cpu_checks = parser.add_mutually_exclusive_group()
    cpu_checks.add_argument("--calibration-only", action="store_true",
                            help="Run only lightweight CPU mountain-score checks; no weights or dataset are loaded")
    cpu_checks.add_argument("--hard-pixel-only", action="store_true",
                            help="Run only lightweight CPU hard-pixel loss checks; no weights or dataset are loaded")
    cpu_checks.add_argument("--split-only", action="store_true",
                            help="Audit a saved random split and CPU data-loader rejection fixtures; no model is loaded")
    args = parser.parse_args()
    if args.output is None:
        filename = ("split_verification.json" if args.split_only else
                    "hard_pixel_verification.json" if args.hard_pixel_only else
                    "calibration_verification.json" if args.calibration_only else "verification.json")
        args.output = PROJECT_ROOT / "result" / filename
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not math.isfinite(args.hard_pixel_weight) or args.hard_pixel_weight < 0:
        parser.error("hard-pixel-weight must be finite and nonnegative")
    if args.hard_pixel_quota < 0:
        parser.error("hard-pixel-quota must be nonnegative")
    torch.set_num_threads(4)
    if args.split_manifest is not None and not args.split_only:
        parser.error("--split-manifest requires --split-only")
    if args.split_only:
        split_path = (args.split_manifest or args.manifest).resolve()
        schema = json.loads(split_path.read_text()).get("schema")
        report = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                  "split_only": True,
                  "split_checks": check_random_split(split_path) if schema == "lusnar_random_split_v1" else check_data(split_path),
                  "synthetic_rejections": check_random_split_rejections(), "status": "passed"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"CPU split and data-loader checks passed. Verification report: {args.output}", flush=True)
        return
    if args.hard_pixel_only:
        report = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                  "hard_pixel_only": True, "hard_pixel_checks": check_hard_pixels(), "status": "passed"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"CPU hard-pixel loss checks passed. Verification report: {args.output}", flush=True)
        return
    if args.calibration_only:
        report = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                  "calibration_only": True, "mountain_calibration": check_mountain_calibration(),
                  "mountain_sweep": check_mountain_sweep(), "status": "passed"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"CPU mountain-score checks passed. Verification report: {args.output}", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not accessible in this session")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "input_size": args.size, "batch_size": args.batch_size,
        "matching": check_matching(), "data": check_data(args.manifest),
        "models": [],
    }
    decoders = ("pixel", "region") if args.decoder == "both" else (args.decoder,)
    for decoder in decoders:
        report["models"].append(check_model(decoder, args, device))
        print(f"{decoder}: full forward/backward, backbone gradients, optimizer and checkpoint checks passed", flush=True)
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Verification report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
