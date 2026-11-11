"""Optional image exports using the exact tensors scored by evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import CLASS_NAMES


PALETTE = np.asarray([[187, 70, 156], [232, 250, 80], [120, 0, 200],
                      [173, 69, 31], [34, 201, 248]], dtype=np.uint8)
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def component(value, label):
    value = str(value)
    if (not value or value in (".", "..") or "/" in value or "\\" in value
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Invalid {label} directory component: {value!r}")
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def comparison_image(rgb, target, prediction):
    height, width = target.shape
    header = 36 if width >= 256 else 20
    legend = 32 if width >= 256 else 0
    canvas = Image.new("RGB", (width * 3, height + header + legend), "white")
    for index, array in enumerate((rgb, PALETTE[target], PALETTE[prediction])):
        canvas.paste(Image.fromarray(array, mode="RGB"), (index * width, header))
    draw = ImageDraw.Draw(canvas)
    label_font = font(22 if width >= 256 else 10)
    for index, text in enumerate(("Input RGB", "Ground truth", "Prediction")):
        draw.text((index * width + 6, 4), text, fill="black", font=label_font)
    if legend:
        legend_font = font(18)
        for index, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
            x, y = index * width * 3 // 5 + 8, header + height + 7
            draw.rectangle((x, y, x + 16, y + 16), fill=tuple(int(v) for v in color))
            draw.text((x + 24, y - 3), name, fill="black", font=legend_font)
    return canvas


class EvaluationImageExporter:
    """Save visual evidence after scoring, without changing prediction tensors.

    ``batch['image']`` is the normalized loader input. Reversing normalization
    and rounding restores its RGB bytes without rereading a different source
    image. Ground truth comes from the already decoded class-ID target used by
    the confusion matrix, including test labels originally stored in a ZIP.
    """

    def __init__(self, output_root, experiment_name, split, tta):
        if split not in ("val", "test") or tta not in ("none", "hflip"):
            raise ValueError("Image exports support val/test and none/hflip")
        self.output_root = Path(output_root).resolve()
        self.experiment_name = component(experiment_name, "experiment")
        self.split, self.tta = split, tta
        self.directory = self.output_root / self.experiment_name / f"{split}_{tta}"
        self.index_path = self.directory / "index.json"
        self.records, self.seen = [], set()
        self.directory.mkdir(parents=True, exist_ok=True)
        # An interrupted export must not retain a previous complete index.
        write_json(self.index_path, self._index("exporting"))

    def _index(self, state):
        return {"schema": "lusnar_evaluation_images_v1", "state": state,
                "experiment": self.experiment_name, "split": self.split, "tta_mode": self.tta,
                "image_count": len(self.records), "class_names": list(CLASS_NAMES),
                "palette_rgb": PALETTE.tolist(),
                "rgb_source": "evaluation input tensor, ImageNet normalization reversed and rounded to uint8",
                "ground_truth_source": "the decoded class-ID mask used for evaluation",
                "mask_encoding": "L-mode uint8 PNG, class IDs 0..4",
                "images": self.records}

    def __call__(self, batch, targets, predictions):
        inputs = batch["image"].detach().to(device="cpu", dtype=torch.float32)
        target = targets.detach().cpu()
        predicted = predictions.detach().cpu()
        if inputs.ndim != 4 or inputs.shape[1] != 3 or target.ndim != 3 or target.shape != predicted.shape:
            raise ValueError("Image export expects Bx3xHxW input and matching BxHxW ID masks")
        if len(inputs) != len(target) or tuple(inputs.shape[-2:]) != tuple(target.shape[-2:]):
            raise ValueError("Export RGB/GT/prediction sizes must match the evaluated input")
        if (torch.any((target < 0) | (target > 4)) or torch.any((predicted < 0) | (predicted > 4))
                or target.is_floating_point() or predicted.is_floating_point()):
            raise ValueError("Export requires integer class-ID masks in 0..4, not raw palette labels")
        rgb = ((inputs * STD + MEAN).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        target, predicted = target.numpy().astype(np.uint8), predicted.numpy().astype(np.uint8)
        for index in range(len(inputs)):
            scene = component(batch["scene"][index], "scene")
            sample = component(batch["sample_id"][index], "sample")
            key = (scene, sample)
            if key in self.seen:
                raise ValueError(f"Duplicate image export identity: {scene}/{sample}")
            self.seen.add(key)
            relative = Path(scene) / sample
            destination = self.directory / relative
            destination.mkdir(parents=True, exist_ok=True)
            files = {"prediction_ids": "prediction_ids.png", "prediction_color": "prediction_color.png",
                     "ground_truth_ids": "ground_truth_ids.png", "comparison": "comparison.jpg"}
            Image.fromarray(predicted[index], mode="L").save(destination / files["prediction_ids"])
            Image.fromarray(PALETTE[predicted[index]], mode="RGB").save(destination / files["prediction_color"])
            Image.fromarray(target[index], mode="L").save(destination / files["ground_truth_ids"])
            comparison_image(rgb[index], target[index], predicted[index]).save(destination / files["comparison"], quality=90, subsampling=0)
            self.records.append({"scene": scene, "sample_id": sample,
                                 "source_id": str(batch["source_id"][index]) if "source_id" in batch else f"{scene}/{sample}",
                                 "shape_hw": list(target[index].shape),
                                 "class_pixel_counts": np.bincount(predicted[index].ravel(), minlength=5).tolist(),
                                 "files": {key: (relative / name).as_posix() for key, name in files.items()}})

    def finalize(self, expected_count, metadata=None):
        if len(self.records) != expected_count:
            raise ValueError(f"Exported {len(self.records)} images, but evaluated {expected_count}")
        index = self._index("complete")
        index["evaluation"] = metadata or {}
        write_json(self.index_path, index)
        return {"enabled": True, "output_root": str(self.output_root), "experiment": self.experiment_name,
                "directory": str(self.directory), "index": str(self.index_path), "image_count": len(self.records)}
