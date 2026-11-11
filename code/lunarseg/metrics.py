"""Scene-level pixel confusion matrices; no averaging of per-image mIoU."""
from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASS_NAMES


class ConfusionMatrix:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target, prediction):
        target = torch.as_tensor(target).detach().reshape(-1)
        prediction = torch.as_tensor(prediction, device=target.device).detach().reshape(-1)
        if target.numel() != prediction.numel():
            raise ValueError("Prediction/target sizes differ")
        valid = (target >= 0) & (target < self.num_classes)
        if torch.any((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)):
            raise ValueError("Prediction class is out of bounds")
        counts = torch.bincount((target[valid] * self.num_classes + prediction[valid]).long(),
                                minlength=self.num_classes ** 2)
        self.matrix += counts.cpu().numpy().reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.matrix
        gt = cm.sum(axis=1)
        union = gt + cm.sum(axis=0) - np.diag(cm)
        iou = np.divide(np.diag(cm), union, out=np.zeros(self.num_classes, dtype=np.float64), where=union != 0)
        present = gt > 0
        return {"miou": float(iou[present].mean()) if present.any() else None,
                "class_iou": {CLASS_NAMES[i]: float(iou[i]) if present[i] else None for i in range(self.num_classes)},
                "gt_present_classes": [CLASS_NAMES[i] for i in range(self.num_classes) if present[i]],
                "pixel_accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
                "confusion_matrix_gt_rows_pred_cols": cm.tolist(),
                "absent_gt_policy": "Average only classes present in scene ground truth"}


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    if device.type == "cpu" and amp == "fp16":
        raise ValueError("fp16 AMP requires CUDA; use bf16 or none on CPU")
    return torch.autocast(device_type=device.type, dtype=dtype)


def tta_metadata(tta="none"):
    if tta not in ("none", "hflip"):
        raise ValueError("tta must be none or hflip")
    return {"test_time_augmentation": tta != "none", "tta_mode": tta,
            "tta_views": ["original", "horizontal_flip"] if tta == "hflip" else ["original"]}


@torch.inference_mode()
def predict_semantic_scores(model, images, tta="none"):
    """Return continuous class scores, without assigning any pixel IDs.

    ``none`` preserves the original logits and interpolation behavior exactly.
    ``hflip`` averages two float32 class probability maps in original image
    coordinates. Softmax accepts both pixel logits and the region head's log
    probabilities. Callers resize the resulting scores before final argmax;
    this helper is for validation/inference, never for training losses.
    """
    tta_metadata(tta)  # Validate before the first model forward.
    original = model(images)["semantic_logits"]
    if tta == "none":
        return original
    flipped = model(images.flip(-1))["semantic_logits"]
    if original.shape != flipped.shape:
        raise ValueError("Original and horizontal-flip predictions have different shapes")
    original = original.float().softmax(dim=1)
    flipped = flipped.float().softmax(dim=1).flip(-1)
    return (original + flipped) * 0.5


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp="bf16", tta="none"):
    augmentation = tta_metadata(tta)
    was_training = model.training
    model.eval()
    total, scenes = ConfusionMatrix(), {}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = predict_semantic_scores(model, images, tta=tta)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits.float(), targets.shape[-2:], mode="bilinear", align_corners=False)
        prediction = logits.argmax(1)
        total.update(targets, prediction)
        for i, scene in enumerate(batch["scene"]):
            entry = scenes.setdefault(scene, {"matrix": ConfusionMatrix(), "image_count": 0})
            entry["matrix"].update(targets[i], prediction[i])
            entry["image_count"] += 1
        count += len(images)
    model.train(was_training)
    result = total.compute()
    result.update(image_count=count, scenes={s: {**v["matrix"].compute(), "image_count": v["image_count"]}
                                          for s, v in sorted(scenes.items())},
                  input_size=loader.dataset.input_size,
                  partial=loader.dataset.partial,
                  data_identity=loader.dataset.identity,
                  **augmentation, continuous_scores_resized_before_argmax=True,
                  official_platform_score=False)
    if all(s in scenes for s in ("Moon_2", "Moon_4", "Moon_9")):
        miou = {s: result["scenes"][s]["miou"] for s in scenes}
        result["weighted_score"] = (miou["Moon_2"] + miou["Moon_4"] + 2 * miou["Moon_9"]) / 4
    return result
