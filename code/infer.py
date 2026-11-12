"""Run a trained checkpoint on images and export class-ID and color masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from lunarseg.model import ModelConfig, build_model
from lunarseg.metrics import (adjust_mountain_scores, predict_semantic_scores, tta_metadata, validate_tta_sizes,
                              adjust_crater_scores, validate_crater_gamma)


PALETTE = np.asarray([[187, 70, 156], [232, 250, 80], [120, 0, 200], [173, 69, 31], [34, 201, 248]], dtype=np.uint8)
CLASS_NAMES = ["regolith", "rock", "crater", "mountain", "sky"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="One image or a directory searched recursively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--tta", choices=("none", "hflip", "multiscale_hflip"), default="none",
                        help="hflip averages original/undo-flipped class probabilities; none keeps single-view inference")
    parser.add_argument("--tta-sizes", type=int, nargs="+", default=None,
                        help="Square image sizes for multiscale_hflip, e.g. 1024 768 512")
    parser.add_argument("--mountain-alpha", type=float, default=1.0,
                        help="Multiply final mountain class probability by this positive factor before argmax")
    parser.add_argument("--crater-gamma", type=float, default=1.0,
                        help="Multiply only the final crater probability by this positive global factor")
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()
    try:
        args.tta_sizes = validate_tta_sizes(args.tta, args.tta_sizes)
        args.crater_gamma = validate_crater_gamma(args.crater_gamma)
        if args.crater_gamma != 1 and args.mountain_alpha != 1:
            raise ValueError("Crater calibration requires mountain-alpha=1")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if not math.isfinite(args.mountain_alpha) or args.mountain_alpha <= 0:
        parser.error("mountain-alpha must be finite and greater than zero")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this session; use an accessible GPU or --device cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    if args.amp != "none" and device.type != "cuda":
        parser.error("Mixed precision is supported only for CUDA inference")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support BF16")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    source = args.input.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if source.is_dir():
        if args.output.resolve().is_relative_to(source):
            parser.error("Output must be outside the input directory")
        images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        source_root = source
    else:
        images, source_root = [source], source.parent
    if not images:
        parser.error("No input images found")
    args.output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            with Image.open(path) as original:
                rgb = original.convert("RGB")
            original_size = rgb.size
            resized = rgb.resize((config.input_size, config.input_size), Image.Resampling.BILINEAR)
            array = np.array(resized, dtype=np.float32, copy=True) / 255.0
            image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            image = (image - mean) / std
            context = torch.autocast("cuda", dtype=dtype) if args.amp != "none" else nullcontext()
            with context:
                scores = predict_semantic_scores(model, image, tta=args.tta, tta_sizes=args.tta_sizes)
            # Interpolate continuous class scores before assigning class IDs.
            scores = F.interpolate(scores.float(), size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
            scores = adjust_mountain_scores(scores, mountain_alpha=args.mountain_alpha, tta=args.tta)
            scores = adjust_crater_scores(scores, gamma=args.crater_gamma, tta=args.tta)
            mask = scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
            relative = path.relative_to(source_root)
            # Keep the original suffix in the directory name to avoid collisions
            # between e.g. image.jpg and image.png from the same source folder.
            destination = args.output / relative.parent / relative.name
            destination.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(destination / "prediction_ids.png")
            colors = PALETTE[mask]
            Image.fromarray(colors).save(destination / "prediction_color.png")
            if args.overlay:
                overlay = (np.asarray(rgb).astype(np.float32) * 0.55 + colors * 0.45).round().astype(np.uint8)
                Image.fromarray(overlay).save(destination / "overlay.png")
            row = {
                "image": str(path), "output": str(destination),
                "original_size_wh": list(original_size),
                "network_size_wh": [config.input_size, config.input_size],
                "class_pixel_counts": np.bincount(mask.ravel(), minlength=5).tolist(),
            }
            rows.append(row)
            print(f"[{index}/{len(images)}] {relative}", flush=True)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
        "model_config": config.to_dict(), "device": str(device), "amp": args.amp,
        **tta_metadata(args.tta, args.tta_sizes), "continuous_scores_resized_before_argmax": True,
        "mountain_alpha": args.mountain_alpha,
        "crater_gamma": args.crater_gamma,
        "crater_score_adjustment": "Only final class-2 probability multiplied by gamma before argmax; one global factor",
        "mountain_score_adjustment": "Multiply final continuous class-3 probability by alpha after resizing, before argmax; "
                                     "single-view logits/log-probabilities use +log(alpha), hflip probabilities use *alpha; "
                                     "other classes unchanged; alpha=1 preserves original scores",
        "class_names": CLASS_NAMES, "palette_rgb": PALETTE.tolist(),
        "image_count": len(rows), "images": rows,
    }
    (args.output / "predictions.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
