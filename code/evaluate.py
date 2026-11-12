#!/usr/bin/env python3
"""Evaluate a checkpoint on its saved validation split or complete Moon_2/4/9."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from lunarseg.data import DEFAULT_MANIFEST, DEFAULT_TEST_MANIFEST, LunarDataset, sha256_file
from lunarseg.metrics import (evaluate_loader, validate_mountain_alpha, validate_tta_sizes,
                              validate_crater_gamma, validate_tta_weights)
from lunarseg.model import ModelConfig, build_model

PROJECT = Path(__file__).resolve().parents[1]


def main(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    defaults = json.loads(known.config.read_text()) if known.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--tta", choices=("none", "hflip", "multiscale_hflip"), default="none",
                        help="none or original plus horizontal-flip continuous-score averaging")
    parser.add_argument("--tta-sizes", type=int, nargs="+", default=None,
                        help="Square image sizes for multiscale_hflip, e.g. 1024 768 512")
    parser.add_argument("--tta-weights", type=float, nargs="+", default=None,
                        help="Nonnegative per-scale weights summing to 1, e.g. .75 .25 for 1024 768")
    parser.add_argument("--tta-1024-weights", type=float, nargs="+", default=None,
                        help="Validation-only sweep with sizes 1024 768, e.g. 1 .75 .5; shares four views")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="Print progress after this many batches; 0 disables progress logging")
    parser.add_argument("--mountain-alpha", type=float, default=1.0,
                        help="Multiply only the final mountain score by this positive factor before argmax")
    parser.add_argument("--mountain-alphas", type=float, nargs="+", default=None,
                        help="Moon_5 only: compare factors including 1 in a single loader pass, e.g. 1 .95 .90 .85 .80")
    parser.add_argument("--crater-gamma", type=float, default=1.0,
                        help="Multiply only the final crater score by this positive global factor")
    parser.add_argument("--crater-gammas", type=float, nargs="+", default=None,
                        help="Validation only: compare factors >=1 including 1 in one pass, e.g. 1 1.03 1.06 1.10")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-images", action="store_true", default=False,
                        help="Optionally save scored class-ID/color predictions and RGB/GT/prediction comparisons")
    parser.add_argument("--image-output-root", default=str(PROJECT.parent / "fenge_result"),
                        help="Image root; each checkpoint's experiment directory name is preserved underneath")
    allowed = {action.dest for action in parser._actions}
    if set(defaults) - allowed:
        parser.error(f"Unknown evaluation config keys: {sorted(set(defaults) - allowed)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.batch_size < 1 or args.workers < 0 or args.progress_every < 0:
        parser.error("batch-size must be positive; workers and progress-every must be nonnegative")
    try:
        args.tta_sizes = validate_tta_sizes(args.tta, args.tta_sizes)
        args.tta_weights = validate_tta_weights(args.tta, args.tta_sizes, args.tta_weights)
        args.mountain_alpha = validate_mountain_alpha(args.mountain_alpha)
        args.crater_gamma = validate_crater_gamma(args.crater_gamma)
        if args.crater_gamma != 1 or args.crater_gammas is not None:
            if args.mountain_alpha != 1 or args.mountain_alphas is not None:
                raise ValueError("Crater calibration requires unchanged mountain scores (alpha=1, no mountain sweep)")
        if args.crater_gammas is not None:
            args.crater_gammas = [validate_crater_gamma(g) for g in args.crater_gammas]
            if (not args.crater_gammas or len(set(args.crater_gammas)) != len(args.crater_gammas)
                    or 1.0 not in args.crater_gammas or any(g < 1 for g in args.crater_gammas)):
                raise ValueError("--crater-gammas requires unique factors >=1 including 1")
            if args.crater_gamma != 1 or args.split != "val" or args.save_images:
                raise ValueError("Crater sweep requires val, gamma=1 and no --save-images")
        if args.mountain_alphas is not None:
            args.mountain_alphas = [validate_mountain_alpha(a) for a in args.mountain_alphas]
            if (not args.mountain_alphas or len(set(args.mountain_alphas)) != len(args.mountain_alphas)
                    or 1.0 not in args.mountain_alphas or any(a > 1 for a in args.mountain_alphas)):
                raise ValueError("--mountain-alphas needs unique 0 < alphas <= 1 including 1")
            if args.mountain_alpha != 1 or args.split != "val" or args.save_images:
                raise ValueError("Use the sweep only on val, without --save-images or a separate --mountain-alpha")
        if args.tta_1024_weights is not None:
            values = args.tta_1024_weights
            if (len(values) < 2 or len(set(values)) != len(values) or 1.0 not in values
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
                raise ValueError("--tta-1024-weights requires at least two unique weights in [0,1] including 1")
            if (args.split != "val" or args.tta != "multiscale_hflip" or args.tta_sizes != [1024, 768]
                    or args.tta_weights is not None or args.save_images or args.mountain_alpha != 1
                    or args.crater_gamma != 1 or args.mountain_alphas is not None or args.crater_gammas is not None):
                raise ValueError("Scale weight sweep requires val, sizes 1024 768, no scalar weights, images or class calibration")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else PROJECT / p
    checkpoint_path = path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    # Match training-time validation so alpha=1 can reproduce its confusion matrix.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate lzc_fenge or use --device cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    saved_validation_manifest = checkpoint.get("run_config", {}).get("manifest", DEFAULT_MANIFEST)
    manifest = path(args.manifest) if args.manifest else path(saved_validation_manifest) if args.split == "val" else DEFAULT_TEST_MANIFEST
    dataset = LunarDataset(manifest, args.split, config.input_size, max_samples=args.max_samples)
    saved_val_identity = checkpoint.get("data_identity", {}).get("val", {})
    if args.split == "val" and args.manifest is None and saved_val_identity:
        if (dataset.identity["manifest_sha256"] != saved_val_identity.get("manifest_sha256")
                or (not dataset.partial and dataset.identity != saved_val_identity)):
            raise ValueError("The checkpoint's saved validation manifest or dataset identity has changed")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                        drop_last=False, pin_memory=device.type == "cuda")
    image_exporter = None
    if args.save_images:
        from lunarseg.visualization import EvaluationImageExporter
        image_exporter = EvaluationImageExporter(path(args.image_output_root), checkpoint_path.parent.name,
                                                 args.split, args.tta, mountain_alpha=args.mountain_alpha,
                                                 tta_sizes=args.tta_sizes, crater_gamma=args.crater_gamma,
                                                 tta_weights=args.tta_weights)
    evaluation_started = time.perf_counter()
    result = evaluate_loader(model, loader, device, args.amp, tta=args.tta, image_exporter=image_exporter,
                             mountain_alpha=args.mountain_alpha, mountain_alphas=args.mountain_alphas,
                             tta_sizes=args.tta_sizes, progress_every=args.progress_every,
                             crater_gamma=args.crater_gamma, crater_gammas=args.crater_gammas,
                             tta_weights=args.tta_weights, tta_1024_weights=args.tta_1024_weights)
    aggregation = (f"validation 1024-weight sweep {args.tta_1024_weights}" if args.tta_1024_weights is not None else
                   f"scale probability weights {args.tta_weights}" if args.tta_weights is not None else
                   "equal view probability averaging")
    view_protocol = (f"sizes {args.tta_sizes}, original + horizontal flip at each size, {aggregation}"
                     if args.tta == "multiscale_hflip" else
                     f"{config.input_size} original + horizontal-flip score averaging" if args.tta == "hflip" else "single view")
    result.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=sha256_file(checkpoint_path),
                  checkpoint_epoch=checkpoint.get("epoch"), split=args.split,
                  evaluation_seconds=time.perf_counter() - evaluation_started,
                  mountain_alpha=args.mountain_alpha,
                  crater_gamma=args.crater_gamma,
                  crater_score_adjustment="Only final class-2 probability multiplied by gamma before argmax; one global factor",
                  mountain_score_adjustment="Only mountain probability multiplied by alpha after final resize, before argmax; not a threshold",
                  batch_size=args.batch_size, amp=args.amp,
                  cudnn_deterministic=True, cudnn_benchmark=False,
                  checkpoint_partial_training=checkpoint.get("partial", False),
                  protocol=("Independent scene confusion matrices, GT-present class mean; "
                            + view_protocol + "; no label postprocessing"),
                  evaluation_only=True)
    if image_exporter is not None:
        result["saved_images"] = image_exporter.finalize(result["image_count"], metadata={
            "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
            "checkpoint_epoch": result["checkpoint_epoch"], "input_size": config.input_size,
            "tta_input_sizes": args.tta_sizes,
            "tta_scale_weights": args.tta_weights,
            "mountain_alpha": args.mountain_alpha,
            "crater_gamma": args.crater_gamma,
            "partial": result["partial"], "continuous_scores_resized_before_argmax": True})
    calibration_suffix = ("_mountain_sweep" if args.mountain_alphas is not None else
                          f"_mountain_{str(args.mountain_alpha).replace('.', 'p')}" if args.mountain_alpha != 1 else "")
    calibration_suffix += ("_crater_sweep" if args.crater_gammas is not None else
                           f"_crater_{str(args.crater_gamma).replace('.', 'p')}" if args.crater_gamma != 1 else "")
    tta_suffix = ("_multiscale_hflip_" + "_".join(map(str, args.tta_sizes)) if args.tta == "multiscale_hflip"
                  else "_hflip" if args.tta == "hflip" else "")
    tta_suffix += ("_weight_sweep" if args.tta_1024_weights is not None else
                   "_weights_" + "_".join(str(w).replace('.', 'p') for w in args.tta_weights)
                   if args.tta_weights is not None else "")
    suffix = tta_suffix + calibration_suffix + ("_partial" if dataset.partial else "")
    output = path(args.output) if args.output else checkpoint_path.parent / f"{args.split}{suffix}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    printed = {"output": str(output), "image_count": result["image_count"], "partial": result["partial"],
                      "tta_mode": args.tta, "tta_input_sizes": args.tta_sizes,
                      "tta_scale_weights": args.tta_weights,
                      "miou": result["miou"], "scene_miou": {s: v["miou"] for s, v in result["scenes"].items()},
                      "weighted_score": result.get("weighted_score")}
    if "saved_images" in result:
        printed["saved_images"] = result["saved_images"]
    if "mountain_score_sweep" in result:
        sweep = result["mountain_score_sweep"]
        printed["mountain_score_sweep"] = {
            "recommended_alpha": sweep["recommended_alpha"], "selection_valid": sweep["selection_valid"],
            "candidates": [{"alpha": c["alpha"], "miou": c["metrics"]["miou"],
                            "mountain_iou": c["metrics"]["class_iou"]["mountain"],
                            "mountain_precision": c["metrics"]["class_precision"]["mountain"],
                            "mountain_recall": c["metrics"]["class_recall"]["mountain"]}
                           for c in sweep["candidates"]]}
    if "crater_score_sweep" in result:
        sweep = result["crater_score_sweep"]
        printed["crater_score_sweep"] = {
            "recommended_gamma": sweep["recommended_gamma"], "selection_valid": sweep["selection_valid"],
            "candidates": [{"gamma": c["gamma"], "miou": c["metrics"]["miou"],
                            "crater_iou": c["metrics"]["class_iou"]["crater"],
                            "crater_precision": c["metrics"]["class_precision"]["crater"],
                            "crater_recall": c["metrics"]["class_recall"]["crater"]}
                           for c in sweep["candidates"]]}
    if "scale_weight_sweep" in result:
        sweep = result["scale_weight_sweep"]
        printed["scale_weight_sweep"] = {
            "recommended_weight_1024": sweep["recommended_weight_1024"], "selection_valid": sweep["selection_valid"],
            "candidates": [{"weight_1024": c["weight_1024"], "miou": c["metrics"]["miou"],
                            "class_iou": c["metrics"]["class_iou"]} for c in sweep["candidates"]]}
    print(json.dumps(printed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
