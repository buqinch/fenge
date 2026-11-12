#!/usr/bin/env python3
"""Measure warmed, accumulated full-resolution training memory, never inference.

Each actual batch size runs in a fresh child process. A successful case executes
one warmup optimizer update followed by two measured updates, with two accumulated
microbatches per update. C always adds sequential masked-view NLL at full weight.
This execution benchmark writes metrics/logs only, never training checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_pool_20261112.json"
SELECTION = "/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_seed42_20261112.json"
VARIANTS = {"A": ("fusion", False), "B": ("adapter", False), "C": ("adapter", True),
            "fusion": ("fusion", False), "adapter": ("adapter", False), "masked": ("adapter", True)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--selection-manifest", default=SELECTION)
    parser.add_argument("--weight-dir", default=str(ROOT / "model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_batch-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (not args.batch_sizes or any(b < 1 for b in args.batch_sizes)
            or len(set(args.batch_sizes)) != len(args.batch_sizes) or args.seed < 0):
        parser.error("Batch sizes must be unique positive integers; seed must be nonnegative")
    if not args.device.startswith("cuda"):
        parser.error("This benchmark measures actual CUDA training memory")
    if args._worker and (args._batch_size is None or args._batch_size < 1):
        parser.error("Worker requires a positive batch size")
    return args


def _keys(batch):
    return [f"{scene}/{sample}/{source}" for scene, sample, source in
            zip(batch["scene"], batch["sample_id"], batch["source_id"])]


def accumulated_update(model, criterion, optimizer, batches, device, update, masked=False):
    """Two real backward graphs accumulate before clipping and AdamW.step."""
    import torch
    from vitseg.masking import apply_block_mask, make_block_masks, masked_semantic_nll, prepare_masked_support
    from vitseg.metrics import autocast_context

    optimizer.zero_grad(set_to_none=True)
    images_in_window = sum(len(batch["image"]) for batch in batches)
    prepared = []
    if masked:
        for batch in batches:
            hidden = make_block_masks(batch["mask"], _keys(batch), update, seed=4242, ratio=.1, block_size=32)
            prepared.append((hidden, prepare_masked_support(batch["mask"], hidden)))
    denominator = sum(support[2]["denominator_pixels"] for _, support in prepared)
    base_losses, mask_losses, component_counts = [], [], {"1": 0, "2": 0, "3": 0}
    start = time.monotonic()
    for index, batch in enumerate(batches):
        images = batch["image"].to(device)
        target = batch["mask"].to(device)
        with autocast_context(device, "bf16"):
            outputs = model(images)
            base = criterion(outputs, target)["loss_total"]
        if not bool(torch.isfinite(base)):
            raise FloatingPointError("Non-finite full Mask2Former loss")
        (base * (len(images) / images_in_window)).backward()
        base_losses.append(float(base.detach()))
        del outputs, base
        if masked:
            hidden_cpu, support = prepared[index]
            hidden = hidden_cpu.to(device)
            with autocast_context(device, "bf16"):
                masked_outputs = model(apply_block_mask(images, hidden))
                extra, stats = masked_semantic_nll(masked_outputs["semantic_logits"], target, hidden,
                                                  hidden_weight=.2, prepared=support)
            if not bool(torch.isfinite(extra)):
                raise FloatingPointError("Non-finite masked NLL")
            (extra * (.5 * stats["denominator_pixels"] / max(denominator, 1))).backward()
            mask_losses.append(float(extra.detach()))
            for key in component_counts:
                component_counts[key] += stats["ignored_component_counts"][key]
            del masked_outputs, extra, hidden
        del images, target
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    gradient_norm = float(norm)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"optimizer_update": update + 1, "microbatches": len(batches), "images_processed": images_in_window,
            "full_set_losses": base_losses, "masked_nll_losses": mask_losses,
            "gradient_norm_before_clip": gradient_norm, "finite_losses_and_gradients": True,
            "optimizer_step_completed": True, "seconds": time.monotonic() - start,
            "masking_lambda": .5 if masked else 0.0, "hidden_weight_rho": .2 if masked else None,
            "masked_support_denominator": denominator if masked else None,
            "ignored_fully_hidden_component_counts": component_counts if masked else None}


def optimizer_state_summary(model, optimizer):
    import torch
    groups = {key: {"parameters_with_state": 0, "state_tensor_bytes": 0} for key in ("vit", "convnext", "head")}
    steps = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        key = "vit" if name.startswith("encoder.vit.") else "convnext" if name.startswith("encoder.convnext.") else "head"
        groups[key]["parameters_with_state"] += 1
        groups[key]["state_tensor_bytes"] += sum(value.numel() * value.element_size()
                                                    for value in state.values() if torch.is_tensor(value))
        if "step" in state:
            steps.append(int(state["step"]))
    return {"components": groups, "minimum_parameter_step": min(steps) if steps else None,
            "maximum_parameter_step": max(steps) if steps else None}


def _memory(device):
    import torch
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device)}


def include_full_run_peaks(result):
    """Keep warmup/workspace peaks as well as warmed-state measured peaks."""
    snapshots = [result, result.get("warmup", {}), *result.get("updates", [])]
    for kind in ("allocated", "reserved"):
        values = [row.get(f"peak_{kind}_bytes") for row in snapshots]
        values = [value for value in values if isinstance(value, int)]
        result[f"full_run_peak_{kind}_bytes"] = max(values) if values else None
    result["peak_definition"] = ("peak_allocated/reserved report the warmed measured updates; "
                                  "full_run_peak includes the warmup and all measured updates")
    return result


def worker(args):
    import random
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from vitseg.data import LunarDataset, sha256_file
    from vitseg.losses import Mask2FormerCriterion
    from vitseg.model import ModelConfig, build_model

    variant, masked = VARIANTS[args.variant]
    result = {"status": "running", "variant": args.variant, "model_variant": variant,
              "batch_size": args._batch_size, "accumulation": 2, "effective_batch": args._batch_size * 2,
              "input_size": 1024, "phase": "setup", "training_stage": "joint_all_backbones_trainable",
              "warmup_updates": 1, "measured_updates": 2, "masking_forced_every_update": masked,
              "amp": "bf16", "gradient_checkpointing": True, "device": args.device,
              "pid": os.getpid(), "torch_version": torch.__version__,
              "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                    sorted([Path(__file__), *(ROOT / "code/vitseg").glob("*.py")])}}
    device = torch.device(args.device)
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support BF16")
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        result["device_info"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor], "free_before_model_bytes": free,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_runtime": torch.version.cuda}
        dataset = LunarDataset(args.manifest, "train", 1024, horizontal_flip=False,
                               selection_manifest=args.selection_manifest)
        if len(dataset) < args._batch_size * 2:
            raise ValueError("Not enough distinct real training samples for two microbatches")
        batches = [default_collate([dataset[i] for i in range(start, start + args._batch_size)])
                   for start in (0, args._batch_size)]
        result.update(data_identity=dataset.identity, sample_keys=[key for batch in batches for key in _keys(batch)],
            unique_real_training_samples=args._batch_size * 2,
            gt_class_ids=sorted(set(int(c) for batch in batches for c in batch["mask"].unique().tolist())),
            sample_reuse="Same two distinct microbatches are reused for the warmup and two measurement updates",
            horizontal_flip=False)
        config = ModelConfig(variant=variant, input_size=1024, gradient_checkpointing=True, weight_dir=args.weight_dir)
        model = build_model(config, pretrained=True).to(device)
        model.train(); model.freeze_backbones(False); model.set_gradient_checkpointing(True)
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Joint benchmark unexpectedly contains frozen parameters")
        result.update(model_config=config.to_dict(), pretrained_report=model.pretrained_report)
        criterion = Mask2FormerCriterion(num_points=4096, class_weight=2., mask_weight=5., dice_weight=5.,
                                         no_object_weight=.1).to(device)
        optimizer = torch.optim.AdamW([{"params": list(model.head_parameters()), "lr": 1e-4},
                                      {"params": list(model.backbone_parameters()), "lr": 5e-6}], weight_decay=.01)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        result["phase"] = "warmup"
        result["warmup"] = accumulated_update(model, criterion, optimizer, batches, device, 0, masked)
        result["warmup"].update(_memory(device))
        state = optimizer_state_summary(model, optimizer)
        if any(value["parameters_with_state"] == 0 for value in state["components"].values()):
            raise RuntimeError("AdamW warmup did not create states for both backbones and the head")
        result["warmed_optimizer_state"] = state
        result["updates"] = []
        for update in (1, 2):
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            result["phase"] = f"measured_update_{update}"
            row = accumulated_update(model, criterion, optimizer, batches, device, update, masked)
            row.update(_memory(device)); result["updates"].append(row)
        state = optimizer_state_summary(model, optimizer)
        if state["minimum_parameter_step"] != 3 or state["maximum_parameter_step"] != 3:
            raise RuntimeError("Expected exactly three successful AdamW updates")
        result.update(status="ok", phase="complete", final_optimizer_state=state,
            peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in result["updates"]),
            peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in result["updates"]),
            median_step_seconds=statistics.median(row["seconds"] for row in result["updates"]))
    except Exception as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(error).lower()
        result.update(status="oom" if oom else "error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            result.update(_memory(device))
    include_full_run_peaks(result)
    write_json(args.output, result)
    print(json.dumps({k: result.get(k) for k in ("variant", "batch_size", "status", "phase", "peak_allocated_bytes",
                                               "peak_reserved_bytes", "median_step_seconds")}), flush=True)
    return 0 if result["status"] in ("ok", "oom") else 1


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        return worker(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite benchmark results: {output}; use --overwrite explicitly")
    folder = output.parent / (output.stem + "_cases")
    folder.mkdir(parents=True, exist_ok=True)
    result = {"schema": "fenge_vit_training_memory_v1", "variant": args.variant, "device": args.device,
              "scope": "Actual1024 pretrained full32-query6+6 BF16 checkpointed training with warmed AdamW; no formal training/checkpoints",
              "comparison": "Different actual batches use two accumulated microbatches each; effective batch is therefore2xactual and is not a controlled accuracy comparison",
              "benchmark_completed": False, "cases": []}
    for batch in args.batch_sizes:
        case_output, log = folder / f"batch_{batch}.json", folder / f"batch_{batch}.log"
        if (case_output.exists() or log.exists()) and not args.overwrite:
            raise FileExistsError(f"Existing case files: {case_output}; choose a new output or --overwrite")
        # A stale worker result must never be mistaken for a fresh child's result.
        if case_output.exists():
            case_output.unlink()
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--_worker", "--variant", args.variant,
                   "--device", args.device, "--_batch-size", str(batch), "--manifest", args.manifest,
                   "--selection-manifest", args.selection_manifest, "--weight-dir", args.weight_dir,
                   "--seed", str(args.seed), "--output", str(case_output)]
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        with log.open("w") as stream:
            child = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if case_output.exists():
            row = json.loads(case_output.read_text())
        else:
            row = {"batch_size": batch, "status": "error", "error": "Worker exited without a JSON result"}
        row.update(worker_exit_code=child.returncode, worker_result=str(case_output), worker_log=str(log), worker_command=command)
        if child.returncode != 0 and row.get("status") == "ok":
            row.update(status="error", error="Worker failed after reporting success")
        result["cases"].append(row)
        write_json(output, result)
        print(json.dumps({k: row.get(k) for k in ("batch_size", "status", "peak_allocated_bytes",
                                                  "peak_reserved_bytes", "median_step_seconds")}), flush=True)
    result["benchmark_completed"] = True
    result["all_batches_succeeded"] = all(row["status"] == "ok" for row in result["cases"])
    write_json(output, result)
    return 1 if any(row["status"] == "error" for row in result["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
