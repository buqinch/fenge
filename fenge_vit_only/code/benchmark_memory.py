#!/usr/bin/env python3
"""Measure complete pure ViT training updates at the requested effective batch.

Includes the full 1024 forward, Hungarian/deep-supervision loss, backward,
gradient clipping, AdamW state allocation and optimizer steps. One warmup and
two measured updates use real train samples. No training checkpoint is saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback

import numpy as np
import torch
from torch.utils.data import default_collate

from vitseg.data import LunarDataset, sha256_file
from vitseg.losses import Mask2FormerCriterion
from vitseg.metrics import autocast_context
from vitseg.model import ModelConfig, build_model
from vitseg.visualization import write_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = '/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_pool_20261112.json'
SELECTION = '/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_seed42_20261112.json'


def update(model, criterion, optimizer, batches, device):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    start = time.monotonic()
    images_total = sum(len(batch['image']) for batch in batches)
    losses = []
    for batch in batches:
        images, targets = batch['image'].to(device), batch['mask'].to(device)
        with autocast_context(device, 'bf16'):
            output = model(images)
            loss = criterion(output, targets)['loss_total']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite Mask2Former loss')
        (loss * (len(images) / images_total)).backward()
        losses.append(float(loss.detach()))
        del output, loss, images, targets
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    return {'microbatches': len(batches), 'images': images_total, 'losses': losses,
            'gradient_norm_before_clip': float(norm), 'optimizer_step_completed': True,
            'seconds': time.monotonic() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def state_summary(model, optimizer):
    components = {key: {'parameters_with_state': 0, 'state_tensor_bytes': 0} for key in ('vit', 'head')}
    steps = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        group = components['vit' if name.startswith('encoder.') else 'head']
        group['parameters_with_state'] += 1
        group['state_tensor_bytes'] += sum(value.numel() * value.element_size() for value in state.values()
                                          if isinstance(value, torch.Tensor))
        steps.append(int(state['step']))
    return {'components': components, 'minimum_parameter_step': min(steps), 'maximum_parameter_step': max(steps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--selection-manifest', default=SELECTION)
    parser.add_argument('--weight-dir', default=str(ROOT / 'model'))
    parser.add_argument('--output', type=Path, default=ROOT / 'result/memory_batch8.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.accumulation < 1:
        parser.error('Batch size and accumulation must be positive')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {args.output}; specify --overwrite')
    result = {'schema': 'pure_vit_training_memory_v1', 'status': 'running', 'partial': True,
        'scope': 'execution benchmark only, not an accuracy experiment', 'variant': 'vit_only',
        'input_size': 1024, 'batch_size': args.batch_size, 'accumulation': args.accumulation,
        'effective_batch': args.batch_size * args.accumulation, 'device': args.device,
        'amp': 'bf16', 'gradient_checkpointing': True, 'all_parameters_trainable': True,
        'warmup_updates': 1, 'measured_updates': 2, 'torch_version': torch.__version__,
        'source_sha256': {str(path.relative_to(ROOT)): sha256_file(path)
                         for path in sorted([Path(__file__), *(ROOT / 'code/vitseg').glob('*.py')])}}
    device = torch.device(args.device)
    try:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('A CUDA device is required for this memory benchmark')
        torch.set_num_threads(2)
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Selected GPU does not support BF16')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42)
        info = torch.cuda.get_device_properties(device)
        result['device_info'] = {'name': info.name, 'total_memory_bytes': info.total_memory,
            'free_before_model_bytes': torch.cuda.mem_get_info(device)[0],
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cuda_runtime': torch.version.cuda}
        dataset = LunarDataset(args.manifest, 'train', 1024, horizontal_flip=False,
                               selection_manifest=args.selection_manifest)
        count = args.batch_size * args.accumulation
        if len(dataset) < count:
            raise ValueError('Insufficient distinct training samples')
        batches = [default_collate([dataset[index] for index in range(start, start + args.batch_size)])
                   for start in range(0, count, args.batch_size)]
        result.update(data_identity=dataset.identity, unique_real_samples=count,
            sample_keys=[str(scene) + '/' + str(sample) for batch in batches
                         for scene, sample in zip(batch['scene'], batch['sample_id'])],
            sample_reuse='Same distinct samples reused for one warmup and two measured updates')
        config = ModelConfig(variant='vit_only', input_size=1024, weight_dir=args.weight_dir,
                             gradient_checkpointing=True)
        model = build_model(config, pretrained=True).to(device)
        model.train(); model.freeze_backbones(False)
        assert all(parameter.requires_grad for parameter in model.parameters())
        result.update(model_config=config.to_dict(), pretrained_report=model.pretrained_report,
            parameters={'total': sum(p.numel() for p in model.parameters()),
                        'vit': sum(p.numel() for p in model.backbone_parameters()),
                        'head': sum(p.numel() for p in model.head_parameters())})
        criterion = Mask2FormerCriterion(num_points=4096).to(device)
        optimizer = torch.optim.AdamW([{'params': list(model.head_parameters()), 'lr': 1e-4},
                                      {'params': list(model.backbone_parameters()), 'lr': 5e-6}], weight_decay=.01)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        result['phase'] = 'warmup'
        result['warmup'] = update(model, criterion, optimizer, batches, device)
        result['warmed_optimizer_state'] = state_summary(model, optimizer)
        assert all(row['parameters_with_state'] > 0 for row in result['warmed_optimizer_state']['components'].values())
        result['updates'] = []
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(device)
            result['phase'] = f'measured_update_{index + 1}'
            result['updates'].append(update(model, criterion, optimizer, batches, device))
        final_state = state_summary(model, optimizer)
        assert final_state['minimum_parameter_step'] == 3 and final_state['maximum_parameter_step'] == 3
        result.update(status='ok', phase='complete', final_optimizer_state=final_state,
            peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in result['updates']),
            peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in result['updates']),
            full_run_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in [result['warmup'], *result['updates']]),
            full_run_peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in [result['warmup'], *result['updates']]),
            median_update_seconds=statistics.median(row['seconds'] for row in result['updates']))
    except Exception as error:
        result.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if device.type == 'cuda' and torch.cuda.is_initialized():
            result.update(failure_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          failure_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.output, result)
    print(json.dumps({key: result.get(key) for key in ('status', 'batch_size', 'accumulation',
        'effective_batch', 'full_run_peak_allocated_bytes', 'full_run_peak_reserved_bytes', 'median_update_seconds')}), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
