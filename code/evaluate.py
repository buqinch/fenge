#!/usr/bin/env python3
"""Evaluate a fixed checkpoint on Moon_5 or explicitly on complete Moon_2/4/9."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lunarseg.data import DEFAULT_MANIFEST, DEFAULT_TEST_MANIFEST, LunarDataset, sha256_file
from lunarseg.metrics import evaluate_loader
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
    parser.add_argument("--tta", choices=("none", "hflip"), default="none",
                        help="none or original plus horizontal-flip continuous-score averaging")
    parser.add_argument("--max-samples", type=int, default=None)
    allowed = {action.dest for action in parser._actions}
    if set(defaults) - allowed:
        parser.error(f"Unknown evaluation config keys: {sorted(set(defaults) - allowed)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else PROJECT / p
    checkpoint_path = path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate lzc_fenge or use --device cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    manifest = path(args.manifest) if args.manifest else DEFAULT_MANIFEST if args.split == "val" else DEFAULT_TEST_MANIFEST
    dataset = LunarDataset(manifest, args.split, config.input_size, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                        drop_last=False, pin_memory=device.type == "cuda")
    result = evaluate_loader(model, loader, device, args.amp, tta=args.tta)
    result.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=sha256_file(checkpoint_path),
                  checkpoint_epoch=checkpoint.get("epoch"), split=args.split,
                  checkpoint_partial_training=checkpoint.get("partial", False),
                  protocol=("Independent scene confusion matrices, GT-present class mean; "
                            + (f"{config.input_size} original + horizontal-flip score averaging" if args.tta == "hflip" else "single view")
                            + "; no label postprocessing"),
                  evaluation_only=True)
    suffix = ("_hflip" if args.tta == "hflip" else "") + ("_partial" if dataset.partial else "")
    output = path(args.output) if args.output else checkpoint_path.parent / f"{args.split}{suffix}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "image_count": result["image_count"], "partial": result["partial"],
                      "miou": result["miou"], "scene_miou": {s: v["miou"] for s, v in result["scenes"].items()},
                      "weighted_score": result.get("weighted_score")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
