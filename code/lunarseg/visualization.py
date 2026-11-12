"""Optional image exports using the exact tensors scored by evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import CLASS_NAMES
from .metrics import (validate_mountain_alpha, tta_metadata, validate_tta_sizes,
                      validate_crater_gamma, validate_tta_weights)


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


def comparison_image(rgb, target, prediction, metrics=None):
    if metrics is not None:
        height, width = target.shape
        if width < 256:
            preview_size = (256, max(1, round(height * 256 / width)))
            rgb = np.asarray(Image.fromarray(rgb, mode="RGB").resize(preview_size, Image.Resampling.BILINEAR))
            target = np.asarray(Image.fromarray(target, mode="L").resize(preview_size, Image.Resampling.NEAREST))
            prediction = np.asarray(Image.fromarray(prediction, mode="L").resize(preview_size, Image.Resampling.NEAREST))
        original = comparison_image(rgb, target, prediction)
        canvas = Image.new("RGB", (original.width, original.height + 68), "white")
        canvas.paste(original, (0, 0))
        draw = ImageDraw.Draw(canvas)
        formatted = lambda value: "N/A" if value is None else f"{value:.2%}"
        draw.text((10, original.height + 8), f"mIoU (GT-present classes): {formatted(metrics['miou'])}",
                  fill="black", font=font(22))
        text = "IoU: " + " | ".join(f"{name}={formatted(metrics['class_iou'][name])}" for name in CLASS_NAMES)
        size = 20
        while size > 10 and draw.textbbox((0, 0), text, font=font(size))[2] > canvas.width - 20:
            size -= 1
        draw.text((10, original.height + 36), text, fill="black", font=font(size))
        return canvas
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


def per_image_statistics(target, prediction):
    matrix = np.bincount((target.astype(np.int64) * 5 + prediction).ravel(), minlength=25).reshape(5, 5)
    gt, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    present = gt > 0
    iou = {name: float(tp[i] / (gt[i] + predicted[i] - tp[i])) if present[i] else None
           for i, name in enumerate(CLASS_NAMES)}
    return {"miou": float(np.mean([value for value in iou.values() if value is not None])),
            "class_iou": iou,
            "class_precision": {name: float(tp[i] / predicted[i]) if predicted[i] else None for i, name in enumerate(CLASS_NAMES)},
            "class_recall": {name: float(tp[i] / gt[i]) if gt[i] else None for i, name in enumerate(CLASS_NAMES)},
            "class_gt_pixels": {name: int(gt[i]) for i, name in enumerate(CLASS_NAMES)},
            "class_predicted_pixels": {name: int(predicted[i]) for i, name in enumerate(CLASS_NAMES)},
            "gt_present_classes": [name for i, name in enumerate(CLASS_NAMES) if present[i]],
            "pixel_accuracy": float(tp.sum() / matrix.sum()),
            "absent_gt_policy": "Average only classes present in scene ground truth",
            "confusion_matrix_gt_rows_pred_cols": matrix.tolist()}


class EvaluationImageExporter:
    """Save visual evidence after scoring, without changing prediction tensors.

    ``batch['image']`` is the normalized loader input. Reversing normalization
    and rounding restores its RGB bytes without rereading a different source
    image. Ground truth comes from the already decoded class-ID target used by
    the confusion matrix, including test labels originally stored in a ZIP.
    """

    def __init__(self, output_root, experiment_name, split, tta, mountain_alpha=1.0, tta_sizes=None,
                 crater_gamma=1.0, tta_weights=None, image_output_mode="all"):
        if split not in ("val", "test"):
            raise ValueError("Image exports support val/test")
        if image_output_mode not in ("all", "comparison"):
            raise ValueError("image_output_mode must be all or comparison")
        self.image_output_mode = image_output_mode
        self.tta_sizes = validate_tta_sizes(tta, tta_sizes)
        self.tta_weights = validate_tta_weights(tta, self.tta_sizes, tta_weights)
        self.output_root = Path(output_root).resolve()
        self.experiment_name = component(experiment_name, "experiment")
        self.split, self.tta = split, tta
        self.mountain_alpha = validate_mountain_alpha(mountain_alpha)
        self.crater_gamma = validate_crater_gamma(crater_gamma)
        if self.crater_gamma != 1 and self.mountain_alpha != 1:
            raise ValueError("Crater calibration requires mountain-alpha=1")
        suffix = f"_mountain_{str(self.mountain_alpha).replace('.', 'p')}" if self.mountain_alpha != 1 else ""
        if self.crater_gamma != 1:
            suffix += f"_crater_{str(self.crater_gamma).replace('.', 'p')}"
        scale_suffix = "_" + "_".join(map(str, self.tta_sizes)) if self.tta_sizes else ""
        if self.tta_weights is not None:
            scale_suffix += "_weights_" + "_".join(str(w).replace('.', 'p') for w in self.tta_weights)
        self.directory = self.output_root / self.experiment_name / f"{split}_{tta}{scale_suffix}{suffix}"
        self.index_path = self.directory / "index.json"
        self.per_image_metrics_csv = self.directory / "per_image_metrics.csv"
        if image_output_mode == "comparison" and self.directory.exists():
            if self.index_path.exists():
                previous = json.loads(self.index_path.read_text())
                if previous.get("output_mode", "all") != "comparison":
                    raise ValueError("Comparison-only export cannot reuse an all-mode image directory; choose a clean output root")
            if next(self.directory.rglob("*.png"), None) is not None:
                raise ValueError("Comparison-only export found existing PNG files; choose a clean output root")
        self.records, self.seen = [], set()
        self.directory.mkdir(parents=True, exist_ok=True)
        # An interrupted export must not retain a previous complete index.
        write_json(self.index_path, self._index("exporting"))

    def _index(self, state):
        return {"schema": "lusnar_evaluation_images_v1", "state": state,
                "experiment": self.experiment_name, "split": self.split, "tta_mode": self.tta,
                "output_mode": self.image_output_mode,
                **tta_metadata(self.tta, self.tta_sizes, self.tta_weights),
                "mountain_alpha": self.mountain_alpha,
                "crater_gamma": self.crater_gamma,
                "image_count": len(self.records), "class_names": list(CLASS_NAMES),
                "palette_rgb": PALETTE.tolist(),
                "rgb_source": "evaluation input tensor, ImageNet normalization reversed and rounded to uint8",
                "ground_truth_source": "the decoded class-ID mask used for evaluation",
                "mask_encoding": ("L-mode uint8 PNG, class IDs 0..4" if self.image_output_mode == "all" else
                                  "Class-ID masks scored in memory; saved only as color panels in comparison.jpg"),
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
            record = {"scene": scene, "sample_id": sample,
                      "source_id": str(batch["source_id"][index]) if "source_id" in batch else f"{scene}/{sample}",
                      "shape_hw": list(target[index].shape),
                      "class_pixel_counts": np.bincount(predicted[index].ravel(), minlength=5).tolist()}
            if self.image_output_mode == "comparison":
                files = {"comparison": "comparison.jpg", "metrics": "metrics.json"}
                metrics = {"schema": "lusnar_per_image_metrics_v1", "scene": scene, "sample_id": sample,
                           "source_id": record["source_id"], "shape_hw": record["shape_hw"],
                           "class_names": list(CLASS_NAMES), **per_image_statistics(target[index], predicted[index])}
                picture = comparison_image(rgb[index], target[index], predicted[index], metrics=metrics)
                metrics["comparison_shape_hw"] = [picture.height, picture.width]
                picture.save(destination / files["comparison"], quality=90, subsampling=0)
                write_json(destination / files["metrics"], metrics)
                record["metrics"] = metrics
            else:
                files = {"prediction_ids": "prediction_ids.png", "prediction_color": "prediction_color.png",
                         "ground_truth_ids": "ground_truth_ids.png", "comparison": "comparison.jpg"}
                Image.fromarray(predicted[index], mode="L").save(destination / files["prediction_ids"])
                Image.fromarray(PALETTE[predicted[index]], mode="RGB").save(destination / files["prediction_color"])
                Image.fromarray(target[index], mode="L").save(destination / files["ground_truth_ids"])
                comparison_image(rgb[index], target[index], predicted[index]).save(destination / files["comparison"], quality=90, subsampling=0)
            record["files"] = {key: (relative / name).as_posix() for key, name in files.items()}
            self.records.append(record)

    def finalize(self, expected_count, metadata=None):
        if len(self.records) != expected_count:
            raise ValueError(f"Exported {len(self.records)} images, but evaluated {expected_count}")
        index = self._index("complete")
        index["evaluation"] = metadata or {}
        if self.image_output_mode == "comparison":
            fields = ["scene", "sample_id", "source_id", "miou", "pixel_accuracy"]
            fields += [f"{name}_iou" for name in CLASS_NAMES] + [f"{name}_gt_pixels" for name in CLASS_NAMES]
            temporary = self.per_image_metrics_csv.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for record in self.records:
                    metrics = record["metrics"]
                    writer.writerow({"scene": record["scene"], "sample_id": record["sample_id"], "source_id": record["source_id"],
                                     "miou": metrics["miou"], "pixel_accuracy": metrics["pixel_accuracy"],
                                     **{f"{name}_iou": metrics["class_iou"][name] for name in CLASS_NAMES},
                                     **{f"{name}_gt_pixels": metrics["class_gt_pixels"][name] for name in CLASS_NAMES}})
            temporary.replace(self.per_image_metrics_csv)
            index["per_image_metrics_csv"] = str(self.per_image_metrics_csv)
        write_json(self.index_path, index)
        result = {"enabled": True, "output_root": str(self.output_root), "experiment": self.experiment_name,
                  "directory": str(self.directory), "index": str(self.index_path), "image_count": len(self.records),
                  "output_mode": self.image_output_mode}
        if self.image_output_mode == "comparison":
            result["per_image_metrics_csv"] = str(self.per_image_metrics_csv)
        return result
