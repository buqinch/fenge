"""Verify real weight loading, full-resolution gradients and dataset separation."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lunarseg.losses import SegmentationCriterion
from lunarseg.model import ModelConfig, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("/media/nas/lzc/fenge/baseline_segformer_20261026/manifest.json")


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
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result" / "verification.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(4)
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
