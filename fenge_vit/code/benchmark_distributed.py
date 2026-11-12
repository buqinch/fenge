#!/usr/bin/env python3
"""Compare real single/four-GPU C updates at an identical global batch size.

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

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/c_adapter_masking.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--global-batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.global_batch_size < 1 or args.global_batch_size % world:
        raise ValueError('Global batch must be positive and divisible by world size')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if world > 1 else torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        config = json.loads(args.config.read_text())
        seed_all(config['seed'], device)
        dataset = LunarDataset(config['manifest'], 'train', 1024, horizontal_flip=False,
                               selection_manifest=config['selection_manifest'])
        indices = list(range(rank, args.global_batch_size, world))
        batch = default_collate([dataset[index] for index in indices])
        images, targets = batch['image'].to(device), batch['mask'].to(device)
        weight_dir = Path(config['weight_dir'])
        if not weight_dir.is_absolute():
            weight_dir = ROOT / weight_dir
        model = build_model(ModelConfig(variant='adapter', input_size=1024,
            gradient_checkpointing=True, weight_dir=str(weight_dir)), pretrained=True).to(device)
        broadcast_model(model)
        initial_hashes = gather_objects(tensor_state_sha256(model.state_dict()))
        if len(set(initial_hashes)) != 1:
            raise ValueError('Distributed initialization differs')
        criterion = Mask2FormerCriterion(num_points=4096).to(device)
        optimizer = torch.optim.AdamW([
            {'params': list(model.head_parameters()), 'lr': config['head_lr']},
            {'params': list(model.backbone_parameters()), 'lr': config['backbone_lr']}],
            weight_decay=config['weight_decay'])
        phases = []
        for phase, frozen, masked in [('frozen_full', True, False),
                                       ('joint_full', False, False),
                                       ('joint_full_and_masked', False, True)]:
            model.train()
            model.freeze_backbones(frozen)
            updates = []
            for step in range(3):
                torch.cuda.reset_peak_memory_stats(device)
                if world > 1:
                    dist.barrier()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, 'bf16'):
                    outputs = model(images)
                    losses = distributed_set_loss(outputs, targets, criterion)
                    loss = losses['loss_total']
                assert_all_finite(loss, 'full objective')
                loss.backward()
                del outputs, losses, loss
                if masked:
                    hidden_cpu = make_block_masks(batch['mask'], _sample_keys(batch), step,
                        config['masking_seed'], config['masking_ratio'], config['masking_block_size'])
                    support = prepare_masked_support(batch['mask'], hidden_cpu)
                    global_denominator = int(reduce_sum(torch.tensor(
                        support[2]['denominator_pixels'], device=device, dtype=torch.int64)).item())
                    hidden = hidden_cpu.to(device)
                    with autocast_context(device, 'bf16'):
                        outputs = model(apply_block_mask(images, hidden))
                        extra, stats = masked_semantic_nll(outputs['semantic_logits'], targets, hidden,
                            hidden_weight=config['masking_hidden_weight'], prepared=support)
                        objective = config['masking_weight'] * distributed_masked_loss(
                            extra, stats['denominator_pixels'], global_denominator)
                    assert_all_finite(objective, 'masked objective')
                    objective.backward()
                    del outputs, extra, objective, hidden
                synchronize_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                assert_all_finite(norm, 'gradient norm')
                optimizer.step()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                rows = gather_objects({'rank': rank, 'seconds': seconds,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'gradient_norm_before_clip': float(norm)})
                updates.append({'warmup': step == 0, 'seconds_max_rank': max(r['seconds'] for r in rows),
                                'ranks': rows})
            hashes = gather_objects(tensor_state_sha256(model.state_dict()))
            if len(set(hashes)) != 1:
                raise ValueError(f'Parameters diverged after {phase}')
            median = statistics.median(row['seconds_max_rank'] for row in updates[1:])
            phases.append({'phase': phase, 'median_seconds_per_update': median,
                           'global_images_per_second': args.global_batch_size / median,
                           'parameters_equal_across_ranks': True, 'updates': updates})
            if rank == 0:
                print(json.dumps({'phase': phase, 'seconds_per_update': median, 'world_size': world}), flush=True)
        all_indices = gather_objects(indices)
        result = {'passed': True, 'world_size': world, 'global_batch_size': args.global_batch_size,
            'per_rank_batch_size': args.global_batch_size // world, 'accumulation': 1,
            'input_size': 1024, 'amp': 'bf16', 'gradient_checkpointing': True,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__,
            'gpu': torch.cuda.get_device_name(device), 'initial_model_sha256': initial_hashes[0],
            'phases': phases, 'training_identity': dataset.identity, 'sample_indices_by_rank': all_indices,
            'source_ids': [dataset.records[index]['source_id'] for index in range(args.global_batch_size)],
            'limitations': ['Two timed updates per phase; excludes data loading, validation and checkpoint writes.',
                'Masked phase forces auxiliary masking every update; formal training uses 25%.',
                'Changed rank partition and stochastic sampling need not produce identical optimization trajectories.'],
            'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                [Path(__file__), ROOT / 'code/distributed_training.py', ROOT / 'code/train.py',
                 *sorted((ROOT / 'code/vitseg').glob('*.py'))]}}
        if rank == 0:
            write_json(args.output, result)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
