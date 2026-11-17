#!/usr/bin/env python3
"""CPU checks of residual initialization, spatial aggregation and trainability."""
import argparse
import io
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from vitseg.model import ModelConfig
from vitseg.upernet import RegionSemanticCorrection, UPerNetHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = {}
    # An independent weighted-mean oracle, with an entirely absent soft class.
    values = torch.tensor([[[[1., 3.]], [[2., 8.]]]])
    probabilities = torch.tensor([[[[.8, .2]], [[.2, .8]], [[0., 0.]]]])
    regions = RegionSemanticCorrection.gather_regions(values, probabilities)
    expected = torch.tensor([[[1.4, 3.2], [2.6, 6.8], [0., 0.]]])
    torch.testing.assert_close(regions, expected)
    torch.testing.assert_close(RegionSemanticCorrection.gather_regions(
        values.flip(-1), probabilities.flip(-1)), expected)
    checks['weighted_pooling_and_absent_class'] = True

    torch.manual_seed(42)
    base = UPerNetHead((32, 64, 96, 128), width=32, gradient_checkpointing=True)
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    corrected = UPerNetHead((32, 64, 96, 128), width=32,
                            gradient_checkpointing=True, region_correction=True)
    assert torch.equal(base_rng, torch.get_rng_state())
    for name, value in base.state_dict().items():
        assert torch.equal(value, corrected.state_dict()[name]), name
    checks['shared_initial_weights_and_rng_identical'] = True
    features = [torch.randn(2, channels, side, side) for channels, side in
                zip((32, 64, 96, 128), (16, 8, 4, 2))]
    for training in (False, True):
        base.train(training); corrected.train(training)
        for dtype in (torch.float32, torch.bfloat16):
            with torch.no_grad(), torch.autocast('cpu', enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
                torch.manual_seed(100)
                plain = base(features)
                torch.manual_seed(100)
                refined = corrected(features)
            assert torch.equal(plain['pixel_logits'].float(), refined['pixel_logits'].float())
            if training:
                assert torch.equal(plain['auxiliary_logits'], refined['auxiliary_logits'])
    checks['zero_residual_exact_identity_fp32_bf16_train_eval'] = True

    corrected.train()
    target = torch.randint(0, 5, (2, 16, 16))
    optimizer = torch.optim.AdamW(corrected.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = corrected(features)
        loss = F.cross_entropy(outputs['pixel_logits'], target)
        loss.backward()
        if step == 0:
            assert corrected.region_correction.delta.weight.grad.abs().sum() > 0
        else:
            for name, parameter in corrected.region_correction.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
    checks['checkpointed_backward_delta_then_all_region_parameters'] = True

    buffer = io.BytesIO()
    torch.save(corrected.state_dict(), buffer)
    buffer.seek(0)
    restored = UPerNetHead((32, 64, 96, 128), width=32, region_correction=True)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    corrected.eval(); restored.eval()
    with torch.no_grad():
        assert torch.equal(corrected(features)['pixel_logits'], restored(features)['pixel_logits'])
    checks['strict_save_reload_output_identity'] = True
    for value in ('true', 1):
        try:
            ModelConfig(decoder_type='upernet', region_correction=value)
        except ValueError:
            continue
        raise AssertionError('Non-boolean flag accepted')
    try:
        ModelConfig(region_correction=True)
    except ValueError:
        checks['configuration_guard'] = True
    else:
        raise AssertionError('Region correction must require UPerNet')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'passed': True, 'checks': checks, 'torch': torch.__version__}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
