"""Real 1024 pretrained-model checks for prior ablations, without training a run."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from lunarseg.data import LunarDataset
from lunarseg.losses import SegmentationCriterion
from lunarseg.metrics import autocast_context, predict_semantic_scores
from lunarseg.model import ModelConfig, build_model
from lunarseg.priors import SemanticAffinity

ROOT = Path(__file__).resolve().parents[1]


def core_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("refiner."):
            continue
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode() + b"\n")
        value.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "result/prior_ablation/verification_runtime.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = LunarDataset(ROOT / "result/selections/random80_4000_2000_2000_pool_20261112.json", "train", 1024,
                        selection_manifest=ROOT / "result/selections/random80_4000_2000_2000_seed42_20261112.json",
                        horizontal_flip=False)
    items = [data[i] for i in range(4)]
    images = torch.stack([x["image"] for x in items]).to(device)
    targets = torch.stack([x["mask"] for x in items]).to(device)
    keys = [x["source_id"] + "/" + str(x["sample_id"]) for x in items]
    rows, reference = [], None
    for mode in ("control", "context", "detail", "affinity"):
        torch.manual_seed(42)
        config = ModelConfig(refinement=mode if mode in ("context", "detail") else "none")
        model = build_model(config, pretrained=True).to(device)
        model.set_gradient_checkpointing(mode in ("context", "detail"))
        initial_hash = core_hash(model)
        assert initial_hash == "48afac5eac0038e6e4a0aa55b6c472a68cde716b0492943b9181c182687a2740"
        model.eval()
        with torch.inference_mode(), autocast_context(device, "bf16"):
            initial_prediction = model(images)["semantic_logits"].cpu()
        if reference is None:
            reference = initial_prediction.clone()
        assert torch.equal(initial_prediction, reference), mode
        model.freeze_backbones(False)
        model.train()
        criterion = SegmentationCriterion().to(device)
        affinity = SemanticAffinity().to(device) if mode == "affinity" else None
        parameters = list(model.parameters()) + (list(affinity.parameters()) if affinity else [])
        optimizer = torch.optim.AdamW(parameters, lr=1e-5)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        extra = {}
        with autocast_context(device, "bf16"):
            predictions = model(images, return_features=affinity is not None)
            loss = criterion(predictions, targets)["loss_total"]
            if affinity is not None:
                affinity_loss, statistics = affinity(predictions["mask_features"], targets, keys, 1)
                gradient = torch.autograd.grad(affinity_loss, predictions["mask_features"], retain_graph=True)[0]
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                extra.update(affinity_feature_gradient_l1=float(gradient.abs().sum()),
                             affinity_loss=float(affinity_loss.detach()), affinity_statistics=statistics)
                loss = loss + .05 * affinity_loss
        assert torch.isfinite(loss)
        loss.backward()
        groups = {"convnext": model.encoder.convnext, "mobile": model.encoder.mobile, "fusion": model.fusion,
                  "decoder": model.decoder}
        if model.refiner is not None:
            groups["refiner"] = model.refiner
        if affinity:
            groups["affinity"] = affinity
        gradient_norms = {}
        for name, module in groups.items():
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), name
            gradient_norms[name] = sum(float(g.float().abs().sum()) for g in gradients)
            assert gradient_norms[name] > 0, name
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode(), autocast_context(device, "bf16"):
            result = predict_semantic_scores(model, images[:1], tta="multiscale_hflip", tta_sizes=[1024, 768])
        assert result.shape == (1, 5, 1024, 1024) and torch.isfinite(result).all()
        torch.testing.assert_close(result.sum(1), torch.ones_like(result[:, 0]), atol=3e-6, rtol=3e-6)
        row = {"mode": mode, "passed": True, "input_shape": list(images.shape), "original_pretrained": True,
               "gradient_checkpointing": mode in ("context", "detail"),
               "common_initial_sha256": initial_hash, "zero_initial_output_equal": True,
               "loss": float(loss.detach()), "gradient_l1": gradient_norms, "four_view_inference_finite_normalized": True,
               "seconds": time.monotonic() - started, "max_allocated_bytes": torch.cuda.max_memory_allocated(device), **extra}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"passed": len(rows) == 4, "partial": True,
                                          "scope": "real execution checks only; no reported trained score",
                                          "device": str(device), "gpu": torch.cuda.get_device_name(device),
                                          "cases": rows}, indent=2) + "\n")
        print(json.dumps({k: row[k] for k in ("mode", "passed", "seconds", "max_allocated_bytes")}), flush=True)
        del model, optimizer, criterion, affinity, predictions, parameters, groups, gradients, loss, result, initial_prediction
        if mode == "affinity":
            del affinity_loss, gradient
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "partial": True, "scope": "real execution checks only; no reported trained score",
                                     "device": str(device), "gpu": torch.cuda.get_device_name(device), "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
