"""Accumulated pixel metrics and continuous-probability TTA for the ViT model."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from numbers import Real
import numpy as np
import torch
import torch.nn.functional as F
from .data import CLASS_NAMES

DIRECTIONS = ((2, 0), (2, 3), (3, 1), (1, 3), (0, 2), (3, 0))


class ConfusionMatrix:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target, prediction):
        target = torch.as_tensor(target).detach().reshape(-1)
        prediction = torch.as_tensor(prediction, device=target.device).detach().reshape(-1)
        if target.numel() != prediction.numel():
            raise ValueError('Prediction and target sizes differ')
        valid = (target >= 0) & (target < self.num_classes)
        if torch.any((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)):
            raise ValueError('Prediction class is out of bounds')
        counts = torch.bincount((target[valid] * self.num_classes + prediction[valid]).long(),
                                minlength=self.num_classes ** 2)
        self.matrix += counts.cpu().numpy().reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.matrix
        gt, predicted = cm.sum(1), cm.sum(0)
        union = gt + predicted - np.diag(cm)
        present = gt > 0
        iou = np.divide(np.diag(cm), union, out=np.zeros(self.num_classes, np.float64), where=union > 0)
        return {'miou': float(iou[present].mean()) if present.any() else None,
                'pooled_miou': float(iou[present].mean()) if present.any() else None,
                'miou_definition': 'mIoU of pooled pixel confusion matrix across evaluated scenes; not official weighted score',
                'class_iou': {n: float(iou[i]) if present[i] else None for i, n in enumerate(CLASS_NAMES)},
                'class_precision': {n: float(cm[i, i] / predicted[i]) if predicted[i] else None
                                    for i, n in enumerate(CLASS_NAMES)},
                'class_recall': {n: float(cm[i, i] / gt[i]) if gt[i] else None for i, n in enumerate(CLASS_NAMES)},
                'gt_pixels': {n: int(gt[i]) for i, n in enumerate(CLASS_NAMES)},
                'predicted_pixels': {n: int(predicted[i]) for i, n in enumerate(CLASS_NAMES)},
                'pixel_accuracy': float(np.trace(cm) / cm.sum()) if cm.sum() else None,
                'confusion_matrix_gt_rows_pred_cols': cm.tolist(),
                'gt_present_classes': [CLASS_NAMES[i] for i in range(self.num_classes) if present[i]],
                'absent_gt_policy': 'Average only classes present in accumulated ground truth',
                'direction_errors': {f'{a}->{b}': {'pixels': int(cm[a, b]),
                    'fraction_of_gt_class': float(cm[a, b] / gt[a]) if gt[a] else None}
                                     for a, b in DIRECTIONS}}


@contextmanager
def _cpu_bf16_context():
    # torch 2.1 oneDNN can corrupt the heap during multithreaded BF16 deformable
    # attention backward. Disable it only inside this CPU BF16 forward context.
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            yield


def autocast_context(device, amp):
    if amp not in ('none', 'bf16', 'fp16'):
        raise ValueError('amp must be none, bf16, or fp16')
    if amp == 'none':
        return nullcontext()
    if device.type == 'cpu' and amp == 'fp16':
        raise ValueError('fp16 AMP requires CUDA')
    if device.type == 'cpu' and amp == 'bf16':
        return _cpu_bf16_context()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if amp == 'bf16' else torch.float16)


def validate_tta_sizes(tta='none', tta_sizes=None):
    if tta not in ('none', 'hflip', 'multiscale_hflip'):
        raise ValueError('Unknown TTA mode')
    if tta != 'multiscale_hflip':
        if tta_sizes is not None:
            raise ValueError('Explicit scales require multiscale_hflip')
        return None
    if not isinstance(tta_sizes, (list, tuple)) or len(tta_sizes) < 2:
        raise ValueError('Multiscale TTA requires at least two explicit sizes')
    if any(type(s) is not int or s < 32 or s % 32 for s in tta_sizes):
        raise ValueError('ViT inference sizes must be exact positive multiples of 32')
    if len(set(tta_sizes)) != len(tta_sizes) or 1024 not in tta_sizes:
        raise ValueError('TTA sizes must be unique and include 1024')
    return list(tta_sizes)


def validate_tta_weights(tta='none', tta_sizes=None, tta_weights=None):
    sizes = validate_tta_sizes(tta, tta_sizes)
    if tta_weights is None:
        return None
    if tta != 'multiscale_hflip' or not isinstance(tta_weights, (list, tuple)) or len(tta_weights) != len(sizes):
        raise ValueError('Explicit weights require one value per multiscale input')
    if any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 0 for w in tta_weights):
        raise ValueError('TTA weights must be finite and nonnegative')
    if abs(math.fsum(tta_weights) - 1) > 1e-6:
        raise ValueError('TTA weights must sum to one')
    return [float(w) for w in tta_weights]


def tta_metadata(tta='none', tta_sizes=None, tta_weights=None):
    sizes = validate_tta_sizes(tta, tta_sizes)
    weights = validate_tta_weights(tta, sizes, tta_weights)
    if tta == 'multiscale_hflip':
        weights = weights or [1 / len(sizes)] * len(sizes)
        active = [(s, w) for s, w in zip(sizes, weights) if w > 0]
        return {'tta_mode': tta, 'test_time_augmentation': True, 'tta_input_sizes': sizes,
                'tta_scale_weights': weights,
                'tta_views': [{'input_size': s, 'transform': t} for s, _ in active
                              for t in ('original', 'horizontal_flip')],
                'tta_view_weights': [w / 2 for _, w in active for _ in range(2)],
                'tta_forward_calls_per_batch': 2 * len(active), 'tta_input_antialias': True,
                'tta_aggregation': 'scale_weighted_float32_normalized_probability_mean',
                'tta_score_resize': 'exp_log_probability_then_inverse_flip_then_bilinear'}
    return {'tta_mode': tta, 'test_time_augmentation': tta != 'none',
            'tta_views': ['original', 'horizontal_flip'] if tta == 'hflip' else ['original'],
            'tta_forward_calls_per_batch': 2 if tta == 'hflip' else 1}


@torch.inference_mode()
def predict_semantic_scores(model, images, tta='none', tta_sizes=None, tta_weights=None):
    sizes = validate_tta_sizes(tta, tta_sizes)
    weights = validate_tta_weights(tta, sizes, tta_weights)
    predict = getattr(model, 'forward_inference', model)
    if tta == 'none':
        return predict(images)['semantic_logits'].float().exp()
    if tta == 'hflip':
        original = predict(images)['semantic_logits'].float().exp()
        flipped = predict(images.flip(-1))['semantic_logits'].float().exp().flip(-1)
        return (original + flipped) * 0.5
    original_size = images.shape[-2:]
    sizes = sizes or [original_size[0]]
    weights = weights or [1 / len(sizes)] * len(sizes)
    accumulated = None
    for size, weight in zip(sizes, weights):
        if weight == 0:
            continue
        view = images if tuple(original_size) == (size, size) else F.interpolate(
            images, (size, size), mode='bilinear', align_corners=False, antialias=True)
        pair = None
        for flipped in (False, True):
            # The model contract is normalized log probabilities. exp avoids a
            # redundant softmax, and preserves the exact probability definition.
            scores = predict(view.flip(-1) if flipped else view)['semantic_logits'].float().exp()
            if flipped:
                scores = scores.flip(-1)
            if scores.shape[-2:] != original_size:
                scores = F.interpolate(scores, original_size, mode='bilinear', align_corners=False)
            pair = scores if pair is None else pair + scores
        contribution = pair * (weight * 0.5)
        accumulated = contribution if accumulated is None else accumulated + contribution
    return accumulated


def _complete_official_test(identity, count, unique_sources, scene_counts, native_grid):
    return (identity.get('split') == 'test' and not identity.get('partial', False)
            and count == unique_sources == 5004 and native_grid
            and scene_counts == {'Moon_2': 1597, 'Moon_4': 1320, 'Moon_9': 2087})


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp='bf16', tta='none', tta_sizes=None,
                    tta_weights=None, diagnostic_observer=None, image_exporter=None, progress_every=50):
    metadata = tta_metadata(tta, tta_sizes, tta_weights)
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer')
    device = torch.device(device)
    total, scenes, seen, count, batches = ConfusionMatrix(), {}, set(), 0, 0
    native_grid = True
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True)
            native_grid = native_grid and tuple(targets.shape[-2:]) == (1024, 1024)
            with autocast_context(device, amp):
                scores = predict_semantic_scores(model, images, tta, tta_sizes, tta_weights)
            if scores.shape[-2:] != targets.shape[-2:]:
                scores = F.interpolate(scores.float(), targets.shape[-2:], mode='bilinear', align_corners=False)
            prediction = scores.argmax(1)
            total.update(targets, prediction)
            for index, scene in enumerate(batch['scene']):
                entry = scenes.setdefault(str(scene), {'matrix': ConfusionMatrix(), 'image_count': 0})
                entry['matrix'].update(targets[index], prediction[index])
                entry['image_count'] += 1
            probabilities = scores.float()
            if diagnostic_observer is not None:
                diagnostic_observer.update(batch, targets, prediction, probabilities)
            if image_exporter is not None:
                image_exporter.update(batch, targets, prediction, probabilities)
            seen.update(batch.get('source_id', []))
            count += images.shape[0]
            batches += 1
            if progress_every and batches % progress_every == 0:
                print(f'Evaluation: {count}/{len(loader.dataset)} images', flush=True)
    finally:
        model.train(was_training)
    identity = dict(getattr(loader.dataset, 'identity', {}))
    partial = bool(getattr(loader.dataset, 'partial', False) or count != len(loader.dataset))
    result = {**total.compute(), **metadata, 'image_count': count, 'partial': partial,
              'data_identity': identity, 'validation_unique_source_count': len(seen),
              'scenes': {scene: {**entry['matrix'].compute(), 'image_count': entry['image_count']}
                         for scene, entry in sorted(scenes.items())},
              'model_forward_calls': batches * metadata['tta_forward_calls_per_batch'],
              'loader_passes': 1, 'weighted_score': None}
    if identity.get('split') == 'test' or set(scenes) & {'Moon_2', 'Moon_4', 'Moon_9'}:
        complete_test = not partial and _complete_official_test(identity, count, len(seen),
            {scene: entry['image_count'] for scene, entry in scenes.items()}, native_grid)
        result['test_scope'] = 'complete_official_test' if complete_test else 'partial_test'
        result['official_test_scope'] = complete_test
        result['partial'] = partial or not complete_test
        result['weighted_score_definition'] = 'Weighted mean of independently accumulated scene mIoUs; distinct from pooled mIoU'
        result['weighted_score_weights'] = {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}
        result['weighted_score_formula'] = '(Moon_2 mIoU + Moon_4 mIoU + 2*Moon_9 mIoU)/4'
        if complete_test:
            result['weighted_score'] = (result['scenes']['Moon_2']['miou'] + result['scenes']['Moon_4']['miou']
                                        + 2 * result['scenes']['Moon_9']['miou']) / 4
    if diagnostic_observer is not None:
        result['object_diagnostics'] = diagnostic_observer.compute()
    return result
