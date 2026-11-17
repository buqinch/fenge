#!/usr/bin/env python3
"""Measure real single/multi-GPU fusion or adapter updates at a fixed global batch.

The same eight real 1024 images are reused, with no loader/I/O in timed updates.
Each phase has a warmup and two timed optimizer updates. Masking is deliberately
forced in the masked phase; normal training selects only 25% of joint updates.
No checkpoints or segmentation scores are produced by this benchmark.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from distributed_training import (assert_all_finite, broadcast_model,
    distributed_masked_loss, distributed_set_loss, gather_objects, reduce_sum,
    synchronize_gradients)
from train import seed_all, tensor_state_sha256, _sample_keys, write_json
from vitseg.data import LunarDataset, sha256_file
from vitseg.losses import Mask2FormerCriterion
from vitseg.masking import apply_block_mask, make_block_masks, masked_semantic_nll, prepare_masked_support
from vitseg.metrics import autocast_context
from vitseg.model import ModelConfig, build_model
from benchmark_memory import cuda_device_uuid, gradient_and_update_probes, optimizer_state_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/c_adapter_masking.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--global-batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--vit-size', choices=['small', 'base', 'large'])
    parser.add_argument('--manifest')
    parser.add_argument('--selection-manifest', help='Use none for a manifest with embedded H6 selection')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.global_batch_size < 1 or args.accumulation < 1 or args.global_batch_size % (world * args.accumulation):
        raise ValueError('Global batch must be positive and divisible by world size times accumulation')
    local_microbatch = args.global_batch_size // (world * args.accumulation)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if world > 1 else torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The selected GPU does not support BF16')
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        config = json.loads(args.config.read_text())
        variant = config.get('variant', 'adapter')
        if variant not in ('fusion', 'adapter'):
            raise ValueError('Expected fusion or adapter model variant')
        masking = config.get('masking', False)
        if type(masking) is not bool:
            raise ValueError('masking must be a boolean')
        vit_size = args.vit_size or config.get('vit_size', 'small')
        adapter_dim, adapter_heads = ((384, 6) if variant == 'fusion' else
                                      {'small': (384, 6), 'base': (768, 12), 'large': (1024, 16)}[vit_size])
        seed_all(config['seed'], device)
        manifest = Path(args.manifest or config['manifest'])
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        selection = args.selection_manifest if args.selection_manifest is not None else config.get('selection_manifest')
        if isinstance(selection, str) and selection.lower() in ('none', 'null', ''):
            selection = None
        if selection is not None and not Path(selection).is_absolute():
            selection = str(ROOT / selection)
        dataset_type = LunarDataset
        if json.loads(manifest.read_text()).get('schema') == 'lusnar_all_sources_hardval_v1':
            from hardval_data import HardValidationDataset
            dataset_type = HardValidationDataset
        dataset = dataset_type(manifest, 'train', 1024, horizontal_flip=False, selection_manifest=selection)
        if len(dataset) < args.global_batch_size:
            raise ValueError('Insufficient distinct real training images for the global batch')
        indices = list(range(rank, args.global_batch_size, world))
        batches = [default_collate([dataset[index] for index in indices[start:start + local_microbatch]])
                   for start in range(0, len(indices), local_microbatch)]
        weight_dir = Path(config['weight_dir'])
        if not weight_dir.is_absolute():
            weight_dir = ROOT / weight_dir
        model_config = ModelConfig(variant=variant, vit_size=vit_size, input_size=1024,
            adapter_dim=adapter_dim, adapter_heads=adapter_heads,
            gradient_checkpointing=True, weight_dir=str(weight_dir),
            decoder_type=config.get('decoder_type', 'mask2former'),
            region_correction=config.get('region_correction', False),
            detail_correction=config.get('detail_correction', False))
        model = build_model(model_config, pretrained=True).to(device)
        broadcast_model(model)
        initial_hashes = gather_objects(tensor_state_sha256(model.state_dict()))
        if len(set(initial_hashes)) != 1:
            raise ValueError('Distributed initialization differs')
        criterion = Mask2FormerCriterion(num_points=config.get('num_points', 4096),
            class_weight=config.get('class_weight', 2.), mask_weight=config.get('mask_weight', 5.),
            dice_weight=config.get('dice_weight', 5.), no_object_weight=config.get('no_object_weight', .1)).to(device)
        if model_config.decoder_type == 'upernet':
            from vitseg.upernet import PixelCrossEntropyCriterion
            criterion = PixelCrossEntropyCriterion(config.get('auxiliary_weight', .4)).to(device)
        optimizer = torch.optim.AdamW([
            {'params': list(model.head_parameters()), 'lr': config['head_lr']},
            {'params': list(model.backbone_parameters()), 'lr': config['backbone_lr']}],
            weight_decay=config['weight_decay'])
        device_reports = gather_objects({'rank': rank, 'name': torch.cuda.get_device_name(device),
            'uuid': cuda_device_uuid(torch.cuda.current_device()),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        phases = []
        result = {'passed': False, 'status': 'running', 'world_size': world,
            'variant': variant, 'masking': masking,
            'global_batch_size': args.global_batch_size, 'per_rank_batch_size': local_microbatch,
            'accumulation': args.accumulation, 'input_size': 1024, 'amp': 'bf16',
            'gradient_checkpointing': True, 'vit_size': vit_size, 'model_config': model_config.to_dict(),
            'pretrained_report': model.pretrained_report, 'total_parameters': sum(p.numel() for p in model.parameters()),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__,
            'device_reports': device_reports, 'initial_model_sha256': initial_hashes[0],
            'initial_model_without_region_correction_sha256': tensor_state_sha256(
                {k: v for k, v in model.state_dict().items() if not k.startswith('decoder.region_correction.')}),
            'initial_model_without_corrections_sha256': tensor_state_sha256(
                {k: v for k, v in model.state_dict().items()
                 if not k.startswith(('decoder.region_correction.', 'detail_correction.'))}),
            'phases': phases, 'training_identity': dataset.identity,
            'sample_indices_by_rank': gather_objects(indices),
            'source_ids': [dataset.records[index]['source_id'] for index in range(args.global_batch_size)],
            'limitations': ['Two timed updates per phase; excludes data loading, validation and checkpoint writes.',
                'Masked phase forces auxiliary masking at full configured weight every update; formal training ramps and samples updates.',
                'Changed rank partition and stochastic sampling need not produce identical optimization trajectories.'],
            'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                [Path(__file__), ROOT / 'code/benchmark_memory.py', ROOT / 'code/distributed_training.py',
                 ROOT / 'code/train.py', ROOT / 'code/hardval_data.py',
                 *sorted((ROOT / 'code/vitseg').glob('*.py'))]}}
        if rank == 0:
            write_json(args.output, result)
        phase_plan = [('frozen_full', True, False), ('joint_full', False, False)]
        if masking:
            phase_plan.append(('joint_full_and_masked', False, True))
        for phase, frozen, masked in phase_plan:
            model.train()
            model.freeze_backbones(frozen)
            frozen_before = tensor_state_sha256(model.encoder.state_dict()) if frozen else None
            updates = []
            for step in range(3):
                torch.cuda.reset_peak_memory_stats(device)
                if world > 1:
                    dist.barrier()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prepared = []
                if masked:
                    for batch in batches:
                        hidden_cpu = make_block_masks(batch['mask'], _sample_keys(batch), step,
                            config['masking_seed'], config['masking_ratio'], config['masking_block_size'])
                        prepared.append((hidden_cpu, prepare_masked_support(batch['mask'], hidden_cpu)))
                local_denominator = sum(item[1][2]['denominator_pixels'] for item in prepared)
                global_denominator = int(reduce_sum(torch.tensor(local_denominator, device=device, dtype=torch.int64)))
                full_losses, masked_losses = [], []
                for micro_index, batch in enumerate(batches):
                    images, targets = batch['image'].to(device), batch['mask'].to(device)
                    with autocast_context(device, 'bf16'):
                        outputs = model(images)
                        losses = distributed_set_loss(outputs, targets, criterion)
                        loss = losses['loss_total']
                    assert_all_finite(loss, 'full objective')
                    (loss / args.accumulation).backward()
                    full_losses.append(float(loss.detach()))
                    del outputs, losses, loss
                    if masked:
                        hidden_cpu, support = prepared[micro_index]
                        hidden = hidden_cpu.to(device)
                        with autocast_context(device, 'bf16'):
                            outputs = model(apply_block_mask(images, hidden))
                            extra, stats = masked_semantic_nll(outputs['semantic_logits'], targets, hidden,
                                hidden_weight=config['masking_hidden_weight'], prepared=support)
                            objective = config['masking_weight'] * distributed_masked_loss(
                                extra, stats['denominator_pixels'], global_denominator)
                        assert_all_finite(objective, 'masked objective')
                        objective.backward()
                        masked_losses.append(float(extra.detach()))
                        del outputs, extra, objective, hidden
                    del images, targets
                synchronize_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                assert_all_finite(norm, 'gradient norm')
                gradient_audit, probes = gradient_and_update_probes(model, frozen) if step == 0 else (None, {})
                correction_gradients = None
                if model_config.region_correction and step == 2:
                    correction_gradients = {name: float(parameter.grad.norm())
                        for name, parameter in model.decoder.region_correction.named_parameters()
                        if parameter.grad is not None}
                    if (len(correction_gradients) != len(list(model.decoder.region_correction.parameters()))
                            or any(not 0 < value < float('inf') for value in correction_gradients.values())):
                        raise RuntimeError(f'Region correction has missing/nonfinite/zero gradients: {correction_gradients}')
                detail_gradients = None
                if model_config.detail_correction and step == 2:
                    detail_gradients = {name: float(parameter.grad.norm())
                        for name, parameter in model.detail_correction.named_parameters()
                        if parameter.grad is not None}
                    if (len(detail_gradients) != len(list(model.detail_correction.parameters()))
                            or any(not 0 < value < float('inf') for value in detail_gradients.values())):
                        raise RuntimeError(f'Detail correction has missing/nonfinite/zero gradients: {detail_gradients}')
                optimizer.step()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                changes = {}
                for key, (name, parameter, index, before) in probes.items():
                    delta = float((parameter.detach().reshape(-1)[index] - before).abs())
                    if delta <= 0:
                        raise RuntimeError(f'No actual optimizer change for {phase}: {name}')
                    changes[key] = {'parameter': name, 'flat_index': index, 'absolute_change': delta}
                rows = gather_objects({'rank': rank, 'seconds': seconds,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'gradient_norm_before_clip': float(norm), 'component_gradient_audit': gradient_audit,
                    'component_parameter_updates': changes or None,
                    'region_correction_gradients': correction_gradients,
                    'detail_correction_gradients': detail_gradients,
                    'full_set_losses': full_losses, 'masked_nll_losses': masked_losses,
                    'masked_global_denominator': global_denominator if masked else None,
                    'masking_weight': config['masking_weight'] if masked else 0.0,
                    'optimizer_state': optimizer_state_summary(model, optimizer) if step == 0 else None})
                updates.append({'warmup': step == 0, 'seconds_max_rank': max(r['seconds'] for r in rows),
                                'ranks': rows})
                if rank == 0:
                    result['current_phase'] = {'phase': phase, 'updates': updates}
                    write_json(args.output, result)
            hashes = gather_objects(tensor_state_sha256(model.state_dict()))
            if len(set(hashes)) != 1:
                raise ValueError(f'Parameters diverged after {phase}')
            frozen_unchanged = tensor_state_sha256(model.encoder.state_dict()) == frozen_before if frozen else None
            if frozen and not frozen_unchanged:
                raise ValueError('A frozen pretrained backbone changed during the frozen phase')
            median = statistics.median(row['seconds_max_rank'] for row in updates[1:])
            phases.append({'phase': phase, 'median_seconds_per_update': median,
                           'global_images_per_second': args.global_batch_size / median,
                           'parameters_equal_across_ranks': True, 'model_sha256': hashes[0],
                           'frozen_backbones_unchanged': frozen_unchanged,
                           'full_phase_peak_allocated_bytes': max(r['peak_allocated_bytes'] for row in updates for r in row['ranks']),
                           'full_phase_peak_reserved_bytes': max(r['peak_reserved_bytes'] for row in updates for r in row['ranks']),
                           'updates': updates})
            if rank == 0:
                write_json(args.output, result)
                print(json.dumps({'phase': phase, 'seconds_per_update': median, 'world_size': world}), flush=True)
        result.update(passed=True, status='complete', current_phase=None,
            full_run_peak_allocated_bytes=max([r['peak_allocated_bytes'] for r in device_reports]
                                             +[p['full_phase_peak_allocated_bytes'] for p in phases]),
            full_run_peak_reserved_bytes=max([r['peak_reserved_bytes'] for r in device_reports]
                                            +[p['full_phase_peak_reserved_bytes'] for p in phases]))
        if rank == 0:
            write_json(args.output, result)
    except Exception as error:
        failed = {'passed': False, 'status': 'oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
            'rank': rank, 'world_size': world, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        write_json(args.output.with_name(args.output.stem + f'_rank_{rank:02d}_error.json'), failed)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
