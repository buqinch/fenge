#!/usr/bin/env python3
"""Verify the complete pure ViT model on real full-resolution training data.

Checks execution and checkpoint correctness, not segmentation accuracy. The
saved round-trip checkpoint is temporary; results contain no trained score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F

from vitseg.data import LunarDataset, sha256_file
from vitseg.losses import Mask2FormerCriterion
from vitseg.metrics import autocast_context, predict_semantic_scores
from vitseg.model import ModelConfig, build_model
from vitseg.visualization import write_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = '/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_pool_20261112.json'
SELECTION = '/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_seed42_20261112.json'


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_summary(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)
    total = sum(float(value.float().abs().sum()) for value in gradients)
    assert total > 0
    return {'tensors': len(gradients), 'l1_sum': total}


def optimizer_for(model):
    return torch.optim.AdamW([{'params': list(model.head_parameters()), 'lr': 1e-4},
                             {'params': list(model.backbone_parameters()), 'lr': 5e-6}], weight_decay=.01)


def train_step(model, criterion, optimizer, images, targets, device, amp):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp):
        output = model(images)
        loss = criterion(output, targets)['loss_total']
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = {'decoder': gradient_summary(model.decoder),
                 'projections': [gradient_summary(layer) for layer in model.projections]}
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        gradients['vit'] = gradient_summary(model.encoder)
    else:
        assert all(parameter.grad is None for parameter in model.encoder.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    return {'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
            'gradients': gradients, 'updated_parameters_finite': True}


def verify_native_features(model, images, device, amp):
    model.eval()
    with torch.inference_mode(), autocast_context(device, amp):
        actual = model.encoder(images, gradient_checkpointing=False)
        reference = model.encoder.model.forward_intermediates(images, indices=[2, 5, 8, 11], norm=True,
            output_fmt='NLC', return_prefix_tokens=True, intermediates_only=True)
        height, width = images.shape[-2] // 16, images.shape[-1] // 16
        rows = []
        for block, feature, (tokens, prefix) in zip([3, 6, 9, 12], actual['intermediate_features'], reference):
            assert tuple(tokens.shape) == (1, height * width, 384)
            assert tuple(prefix.shape) == (1, 5, 384)
            expected = tokens.reshape(1, height, width, 384).permute(0, 3, 1, 2).contiguous()
            torch.testing.assert_close(feature, expected, atol=0, rtol=0)
            rows.append({'block': block, 'shape': list(feature.shape), 'native_norm_and_row_major_equal': True})
        torch.testing.assert_close(actual['tokens'][:, :5], reference[-1][1], atol=0, rtol=0)
        torch.testing.assert_close(actual['patch_features'], actual['intermediate_features'][-1], atol=0, rtol=0)
        pyramid = model.reassemble(actual['intermediate_features'])
        expected_shapes = [[1, 256, images.shape[-2] // stride, images.shape[-1] // stride]
                           for stride in (4, 8, 16, 32)]
        assert [list(value.shape) for value in pyramid] == expected_shapes
        # Independently construct the accepted spatial coordinate transforms.
        projected = [layer(feature) for layer, feature in zip(model.projections, actual['intermediate_features'])]
        expected_pyramid = [F.interpolate(projected[0], scale_factor=4, mode='bilinear', align_corners=False),
                            F.interpolate(projected[1], scale_factor=2, mode='bilinear', align_corners=False),
                            projected[2], F.avg_pool2d(projected[3], kernel_size=2, stride=2)]
        for actual_map, expected_map in zip(pyramid, expected_pyramid):
            torch.testing.assert_close(actual_map, expected_map, atol=0, rtol=0)
    return {'native_timm_intermediates': rows, 'prefix_tokens': 5, 'register_tokens': 4,
            'pyramid_shapes': expected_shapes, 'strides': [4, 8, 16, 32],
            'spatial_transforms_and_prefix_exclusion_equal': True}


def verify_outputs(model, images, targets, device, amp):
    model.eval()
    with torch.inference_mode(), autocast_context(device, amp):
        output = model(images)
    size = images.shape[-1]
    assert tuple(output['pred_masks'].shape) == (1, 32, size // 4, size // 4)
    assert tuple(output['pred_logits'].shape) == (1, 32, 6)
    assert len(output['aux_outputs']) == 6
    with torch.autocast(device_type=device.type, enabled=False):
        classes = output['pred_logits'].float().softmax(-1)[..., :5]
        masks = output['pred_masks'].float().sigmoid()
        evidence = torch.einsum('bqc,bqhw->bchw', classes, masks)
        evidence = F.interpolate(evidence, images.shape[-2:], mode='bilinear', align_corners=False)
        eps = model.config.semantic_epsilon
        probabilities = (evidence + eps) / (evidence.sum(1, keepdim=True) + 5 * eps)
    torch.testing.assert_close(output['semantic_scores'], evidence, atol=0, rtol=0)
    torch.testing.assert_close(output['semantic_logits'], probabilities.log(), atol=0, rtol=0)
    torch.testing.assert_close(output['semantic_logits'].exp().sum(1),
                               torch.ones_like(targets, dtype=torch.float32), atol=3e-6, rtol=3e-6)
    return output['semantic_logits'].detach().cpu(), {'query_logits_shape': list(output['pred_logits'].shape),
        'query_masks_shape': list(output['pred_masks'].shape), 'auxiliary_predictions': len(output['aux_outputs']),
        'semantic_shape': list(output['semantic_logits'].shape), 'matches_A_score_order_exactly': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--selection-manifest', default=SELECTION)
    parser.add_argument('--weight-dir', default=str(ROOT / 'model'))
    parser.add_argument('--output', type=Path, default=ROOT / 'result/verification_model.json')
    args = parser.parse_args()
    write_json(args.output, {'schema': 'pure_vit_verification_v1', 'passed': False, 'status': 'running',
                            'scope': 'execution checks only; no segmentation score'})
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    amp = 'bf16' if device.type == 'cuda' else 'none'
    torch.manual_seed(42)
    started = time.monotonic()
    dataset = LunarDataset(args.manifest, 'train', 1024, horizontal_flip=False,
                           selection_manifest=args.selection_manifest)
    print('Validated fixed training data manifest', flush=True)
    item = dataset[0]
    images, targets = item['image'][None].to(device), item['mask'][None].to(device)
    config = ModelConfig(variant='vit_only', input_size=1024, weight_dir=args.weight_dir,
                         gradient_checkpointing=True)
    model = build_model(config, pretrained=True).to(device)
    assert set(model.pretrained_report) == {'vit'}
    assert model.pretrained_report['vit']['loaded']
    forbidden = ('convnext', 'mobilenet', 'adapter', 'rgb_branch', 'detail_branch')
    assert not any(any(word in name.lower() for word in forbidden) for name, _ in model.named_modules())
    assert len(model.encoder.model.blocks) == 12 and model.encoder.num_prefix_tokens == 5
    assert len(model.projections) == 4 and len({id(layer) for layer in model.projections}) == 4
    head_ids = {id(p) for p in model.head_parameters()}
    encoder_ids = {id(p) for p in model.backbone_parameters()}
    assert not head_ids & encoder_ids
    assert head_ids | encoder_ids == {id(p) for p in model.parameters()}
    report = {'schema': 'pure_vit_verification_v1', 'passed': False, 'partial': True,
        'scope': 'real 1024 execution and checkpoint checks, not a trained segmentation result',
        'device': str(device), 'amp': amp, 'model_config': config.to_dict(), 'data_identity': dataset.identity,
        'sample': {key: item[key] for key in ('scene', 'sample_id', 'source_id')},
        'pretrained_report': model.pretrained_report,
        'parameters': {'total': sum(p.numel() for p in model.parameters()),
                       'vit': sum(p.numel() for p in model.encoder.parameters()),
                       'heads': sum(p.numel() for p in model.head_parameters())},
        'single_vit_only': True, 'optimizer_groups_complete_and_disjoint': True}
    report['features'] = verify_native_features(model, images, device, amp)
    _, report['outputs'] = verify_outputs(model, images, targets, device, amp)
    print('Native intermediate features, four scales and A output order passed', flush=True)
    criterion = Mask2FormerCriterion(num_points=4096).to(device)
    optimizer = optimizer_for(model)
    model.train()
    model.freeze_backbones(True)
    assert not model.encoder.training
    encoder_before = tensor_hash(model.encoder.state_dict().items())
    heads_before = tensor_hash((n, p) for n, p in model.state_dict().items() if not n.startswith('encoder.'))
    report['frozen_update'] = train_step(model, criterion, optimizer, images, targets, device, amp)
    assert tensor_hash(model.encoder.state_dict().items()) == encoder_before
    assert tensor_hash((n, p) for n, p in model.state_dict().items() if not n.startswith('encoder.')) != heads_before
    report['frozen_backbone_unchanged_and_heads_updated'] = True
    model.freeze_backbones(False)
    model.train()
    assert model.encoder.training and all(p.requires_grad for p in model.parameters())
    report['unfrozen_update'] = train_step(model, criterion, optimizer, images, targets, device, amp)
    assert tensor_hash(model.encoder.state_dict().items()) != encoder_before
    report['unfrozen_vit_updated'] = True
    print('Frozen and unfrozen optimizer updates passed', flush=True)
    optimizer.zero_grad(set_to_none=True)
    expected, _ = verify_outputs(model, images, targets, device, amp)
    with torch.inference_mode(), autocast_context(device, amp):
        scores = predict_semantic_scores(model, images, tta='multiscale_hflip', tta_sizes=[1024, 768])
        assert tuple(scores.shape) == (1, 5, 1024, 1024) and bool(torch.isfinite(scores).all())
        torch.testing.assert_close(scores.sum(1), torch.ones_like(scores[:, 0]), atol=3e-6, rtol=3e-6)
    report['equal_weight_1024_768_four_view_tta'] = True
    del scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_checkpoint_', dir=args.output.parent) as folder:
        path = Path(folder) / 'checkpoint.pt'
        state = {'model_config': config.to_dict(), 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        saved_model_sha256 = tensor_hash(state['model'].items())
        torch.save(state, path)
        report['checkpoint_sha256'] = sha256_file(path)
        del state
        # A next-step replay checks Adam moments and random loss sampling too.
        model.train()
        replay_reference = train_step(model, criterion, optimizer, images, targets, device, amp)
        assert tensor_hash(model.state_dict().items()) != saved_model_sha256
        reference_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        del optimizer, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        state = torch.load(path, map_location='cpu')
        restored = build_model(ModelConfig.from_dict(state['model_config']), pretrained=False).to(device)
        restored.load_state_dict(state['model'], strict=True)
        assert tensor_hash(restored.state_dict().items()) == saved_model_sha256
        restored.eval()
        actual, _ = verify_outputs(restored, images, targets, device, amp)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        optimizer = optimizer_for(restored)
        optimizer.load_state_dict(state['optimizer'])
        loaded_optimizer = optimizer.state_dict()
        assert loaded_optimizer['param_groups'] == state['optimizer']['param_groups']
        saved_moments = tensor_hash((f'{index}.{key}', value)
            for index, values in state['optimizer']['state'].items() for key, value in values.items()
            if isinstance(value, torch.Tensor))
        loaded_moments = tensor_hash((f'{index}.{key}', value)
            for index, values in loaded_optimizer['state'].items() for key, value in values.items()
            if isinstance(value, torch.Tensor))
        assert saved_moments == loaded_moments
        restored.train()
        restored.freeze_backbones(False)
        torch.set_rng_state(state['rng_cpu'])
        assert torch.equal(torch.get_rng_state(), state['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng_cuda'], device)
            assert torch.equal(torch.cuda.get_rng_state(device), state['rng_cuda'])
        replay_restored = train_step(restored, criterion, optimizer, images, targets, device, amp)
        assert tensor_hash(restored.state_dict().items()) != saved_model_sha256
        assert replay_reference['loss'] == replay_restored['loss']
        # CUDA grid_sample backward is not guaranteed bitwise deterministic.
        # Verify exact restored state above, and disclose the resulting next-step
        # drift instead of asserting an unsupported deterministic-training claim.
        maximum_difference = 0.
        changed_tensors = 0
        for name, value in restored.state_dict().items():
            actual_tensor = value.detach().cpu()
            difference = float((actual_tensor - reference_state[name]).abs().max())
            maximum_difference = max(maximum_difference, difference)
            changed_tensors += int(difference != 0.)
        report['checkpoint_checks'] = {'strict_disk_reload_predictions_bitwise_equal': True,
            'model_parameters_restored_exactly': True,
            'optimizer_groups_and_moments_restored_exactly': True,
            'optimizer_moments_sha256': loaded_moments,
            'torch_cpu_and_cuda_rng_restored_exactly': True,
            'optimizer_and_rng_next_step_loss_equal': True,
            'both_next_updates_finite_and_parameters_changed': True,
            'nondeterministic_update_drift': {'parameter_tensors_with_difference': changed_tensors,
                'max_absolute_parameter_difference': maximum_difference,
                'scope': 'recorded observation, not a numerical equivalence acceptance test'},
            'training_replay_limit': 'CUDA grid_sample backward is nondeterministic; exact parameter replay is not promised',
            'temporary_checkpoint_removed': True}
    report.update(passed=True, seconds=time.monotonic() - started,
        source_sha256={str(path.relative_to(ROOT)): sha256_file(path)
                       for path in sorted([Path(__file__), *(ROOT / 'code/vitseg').glob('*.py')])})
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ('passed', 'parameters', 'seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
