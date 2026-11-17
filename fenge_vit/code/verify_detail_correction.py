#!/usr/bin/env python3
"""Check dense point alignment, initialization, strip invariance and gradients."""
import argparse
import io
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from vitseg.detail import DenseDetailCorrection
from vitseg.model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    height, width = 48, 64
    images = torch.randn(2, 3, height, width)
    features = torch.randn(2, 96, height // 4, width // 4)
    coarse = torch.randn(2, 5, height, width)
    checks = {}
    sampled = []
    for start in range(0, height, 7):
        grid = DenseDetailCorrection.sampling_grid(height, width, start, min(start + 7, height), features.device)
        sampled.append(F.grid_sample(features, grid.expand(2, -1, -1, -1),
                                     padding_mode='border', align_corners=False))
    expected = F.interpolate(features, (height, width), mode='bilinear', align_corners=False)
    torch.testing.assert_close(torch.cat(sampled, 2), expected, rtol=1e-5, atol=1e-5)
    checks['pixel_center_sampling_matches_bilinear_whole_image'] = True
    head = DenseDetailCorrection(chunk_rows=7)
    for training in (True, False):
        head.train(training)
        for mixed in (False, True):
            with torch.no_grad(), torch.autocast('cpu', enabled=mixed, dtype=torch.bfloat16):
                result = head(images, features, coarse)
            assert torch.equal(result, coarse)
    checks['zero_initialized_identity_fp32_bf16_train_eval'] = True
    head.train()
    # The dense objective includes every interior pixel, with no uncertainty gate.
    targets = torch.zeros((2, height, width), dtype=torch.long)
    targets[:, 8:24, 8:24] = 1
    targets[:, 24:40, 24:48] = 2
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = head(images, features, coarse)
        loss = F.cross_entropy(result, targets)
        loss.backward()
        if step == 0:
            assert head.delta.weight.grad.abs().sum() > 0
        else:
            for name, parameter in head.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
    checks['full_image_ce_and_checkpointed_backward_all_branches'] = True
    head.eval()
    with torch.no_grad():
        narrow = head(images, features, coarse)
        head.chunk_rows = height
        whole = head(images, features, coarse)
    torch.testing.assert_close(narrow, whole, rtol=1e-6, atol=1e-6)
    checks['strip_partition_does_not_change_predictions'] = True
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    other = DenseDetailCorrection(chunk_rows=height).eval()
    other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(whole, other(images, features, coarse))
    checks['strict_checkpoint_reload_identity'] = True
    try:
        ModelConfig(detail_correction=True)
    except ValueError:
        checks['upernet_only_configuration_guard'] = True
    else:
        raise AssertionError('Detail correction must require UPerNet')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': True, 'checks': checks, 'torch': torch.__version__}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
