"""Deterministic full-image block masking and the auxiliary semantic NLL.

All randomness is local and keyed by update/sample, independent of DataLoader,
Hungarian point sampling, dropout, and the global torch/numpy/Python RNGs.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
from scipy import ndimage
import torch


def _seed(*parts):
    text = '\0'.join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def update_is_masked(global_update, seed=4242, fraction=0.25):
    """One decision for an entire optimizer update, including every microbatch."""
    if not 0 <= fraction <= 1 or not math.isfinite(fraction):
        raise ValueError('Mask update fraction must be finite and in [0,1]')
    return _seed('mask-update-v1', seed, global_update) / 2**64 < fraction


def warmup_factor(joint_update, warmup_updates=1000):
    if joint_update < 0 or warmup_updates < 0:
        raise ValueError('Update counts must be nonnegative')
    return 1.0 if warmup_updates == 0 else min(1.0, (joint_update + 1) / warmup_updates)


def make_block_masks(targets, sample_keys, global_update, seed=4242, ratio=0.1, block_size=32):
    """Return BxHxW hidden pixels. Each sample has a separate fixed mask.

    Keys should identify the actual sample (including augmentation), not merely
    its source family, so distinct crops of one source receive distinct masks.
    """
    if targets.ndim != 3 or len(sample_keys) != targets.shape[0]:
        raise ValueError('Expected BxHxW targets and one stable sample key per image')
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Masking ratio must be finite and in [0,1]')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Block size must be a positive integer')
    height, width = targets.shape[-2:]
    rows, columns = math.ceil(height / block_size), math.ceil(width / block_size)
    count = min(rows * columns, int(math.floor(rows * columns * ratio + 0.5)))
    masks = []
    for key in sample_keys:
        rng = np.random.default_rng(_seed('mask-pattern-v1', seed, global_update, key))
        grid = np.zeros(rows * columns, dtype=bool)
        if count:
            grid[rng.choice(rows * columns, size=count, replace=False)] = True
        pixels = np.repeat(np.repeat(grid.reshape(rows, columns), block_size, 0), block_size, 1)
        masks.append(pixels[:height, :width].copy())
    return torch.from_numpy(np.stack(masks)).to(targets.device)


def apply_block_mask(images, hidden):
    if hidden.shape != (images.shape[0], *images.shape[-2:]):
        raise ValueError('Hidden mask geometry differs from normalized images')
    return images.masked_fill(hidden[:, None], 0.0)


def prepare_masked_support(targets, hidden):
    """Compute valid support before forward, allowing exact accumulated loss."""
    if targets.shape != hidden.shape or targets.ndim != 3:
        raise ValueError('Expected matching BxHxW targets and hidden masks')
    truth = targets.detach().cpu().numpy()
    masks = hidden.detach().cpu().numpy().astype(bool)
    ignored = np.zeros(truth.shape, dtype=bool)
    ignored_counts = {str(i): 0 for i in (1, 2, 3)}
    ignored_pixels = {str(i): 0 for i in (1, 2, 3)}
    for index, (target, covered) in enumerate(zip(truth, masks)):
        for label in (1, 2, 3):
            regions, count = ndimage.label(target == label, structure=np.ones((3, 3), dtype=np.uint8))
            if not count:
                continue
            visible = np.bincount(regions[~covered].ravel(), minlength=count + 1)
            remove = np.flatnonzero(visible == 0)
            remove = remove[remove > 0]
            if len(remove):
                pixels = np.isin(regions, remove)
                ignored[index] |= pixels
                ignored_counts[str(label)] += len(remove)
                ignored_pixels[str(label)] += int(pixels.sum())
    ignored = torch.from_numpy(ignored).to(targets.device)
    valid = (targets >= 0) & (targets < 5)
    visible = valid & ~hidden
    eligible = valid & hidden & ~ignored
    denominator = int((visible | eligible).sum())
    stats = {'visible_valid_pixels': int(visible.sum()), 'eligible_hidden_pixels': int(eligible.sum()),
             'denominator_pixels': denominator, 'masked_pixels': int(hidden.sum()),
             'total_pixels': hidden.numel(), 'ignored_component_counts': ignored_counts,
             'ignored_component_pixels': ignored_pixels,
             'denominator_definition': 'visible_valid + eligible_hidden, without rho weighting',
             'component_definition': '8-connected GT regions for classes 1,2,3; not physical instances',
             'mask_sha256': hashlib.sha256(masks.tobytes()).hexdigest()}
    return visible, eligible, stats


def masked_semantic_nll(log_probabilities, targets, hidden, hidden_weight=0.2, prepared=None):
    """Visible NLL + rho*eligible hidden NLL, divided by unweighted support.

    Fully hidden rock/crater/mountain GT components are ignored only here.
    For gradient accumulation the trainer weights each microbatch by its valid
    denominator / complete-window denominator, giving the exact effective-batch
    pixel objective even when fully hidden components have different sizes.
    """
    if (log_probabilities.ndim != 4 or log_probabilities.shape[1] != 5 or
            targets.shape != hidden.shape or
            log_probabilities.shape[0] != targets.shape[0] or
            log_probabilities.shape[-2:] != targets.shape[-2:]):
        raise ValueError('Auxiliary NLL requires full-resolution Bx5xHxW log probabilities')
    if not math.isfinite(hidden_weight) or not 0 <= hidden_weight <= 1:
        raise ValueError('Hidden weight must be finite and in [0,1]')
    visible, eligible, stats = prepare_masked_support(targets, hidden) if prepared is None else prepared
    visible, eligible = visible.to(targets.device), eligible.to(targets.device)
    denominator = stats['denominator_pixels']
    nll = -log_probabilities.float().gather(1, targets.clamp(0, 4)[:, None]).squeeze(1)
    numerator = nll.masked_select(visible).sum() + hidden_weight * nll.masked_select(eligible).sum()
    loss = numerator / max(denominator, 1) if denominator else log_probabilities.sum() * 0.0
    return loss, {**stats, 'hidden_weight': hidden_weight}
