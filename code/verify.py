"""Verify real weight loading, full-resolution gradients and dataset separation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
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
    criterion = SegmentationCriterion(num_points=1024).to(device)
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
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "checkpoint_strict_roundtrip": True, "optimizer_step": True,
        "seconds": round(time.monotonic() - start, 3),
    }
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--calibration-only", action="store_true",
                        help="Run only lightweight CPU mountain-score checks; no weights or dataset are loaded")
    args = parser.parse_args()
    if args.output is None:
        args.output = PROJECT_ROOT / "result" / ("calibration_verification.json" if args.calibration_only else "verification.json")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(4)
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
