#!/usr/bin/env python3
"""Train the full ViT model with exact-loss synchronous data parallelism.

Launch with torchrun. batch_size is per rank; the effective global batch is
batch_size * WORLD_SIZE * accumulation. Existing single-GPU modules stay intact.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import train as base
from distributed_training import (NoPaddingShardSampler, assert_all_finite, broadcast_model,
    broadcast_object, distributed_evaluate, distributed_masked_loss, distributed_set_loss,
    gather_objects, reduce_sum, synchronize_gradients, world_info)
from vitseg.data import LunarDataset, sha256_file
from vitseg.losses import Mask2FormerCriterion
from vitseg.masking import (apply_block_mask, make_block_masks, masked_semantic_nll,
    prepare_masked_support, update_is_masked, warmup_factor)
from vitseg.metrics import autocast_context
from vitseg.model import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def _rank_zero_action(function):
    rank, _ = world_info()
    result = None
    if rank == 0:
        try:
            function()
            result = {'ok': True}
        except Exception as error:
            result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    result = broadcast_object(result)
    if not result['ok']:
        raise RuntimeError('Rank-zero I/O failed: ' + result['error'])


def _window_iterator(loader, accumulation):
    iterator = iter(loader)
    for start in range(0, len(loader), accumulation):
        yield start, [next(iterator) for _ in range(min(accumulation, len(loader) - start))]


def _merge_mask_stats(parts):
    total = {'visible_valid_pixels': 0, 'eligible_hidden_pixels': 0, 'denominator_pixels': 0,
             'masked_pixels': 0, 'total_pixels': 0,
             'ignored_component_counts': {str(i): 0 for i in (1, 2, 3)},
             'ignored_component_pixels': {str(i): 0 for i in (1, 2, 3)}}
    for part in parts:
        base._sum_mask_stats(total, part)
    return total


def _prepare_output(output, args):
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ('last.pt', 'history.jsonl', 'summary.json')):
        raise FileExistsError('Output contains a run; choose a new output or resume')
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError('Resume must retain the original output directory')


def _validate_history(output, saved, partial):
    history = output / 'history.jsonl'
    rows = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
    epoch = saved['epoch']
    if len(rows) < epoch or [row['epoch'] for row in rows[:epoch]] != list(range(1, epoch + 1)):
        raise ValueError('Resume requires complete ordered history through the checkpoint epoch')
    if len(rows) > epoch:
        if Path(saved['_resume_path']).name != 'last.pt':
            raise ValueError('Only last.pt can recover an uncommitted history tail')
        archive = output / f'history_uncommitted_after_{epoch:03d}_{time.time_ns()}.jsonl'
        archive.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[epoch:]))
        temporary = history.with_name('history.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[:epoch]))
        temporary.replace(history)
    if not (output / ('best_smoke.pt' if partial else 'best.pt')).exists():
        raise FileNotFoundError('Resume requires its existing validation-best checkpoint')


def main(argv=None):
    args = base.parse_args(argv)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    requested_world = int(os.environ.get('WORLD_SIZE', '1'))
    use_cuda = torch.device(args.device).type == 'cuda'
    device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Selected GPU does not support BF16')
    owns_group = not dist.is_initialized()
    if owns_group and requested_world > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo', timeout=timedelta(minutes=15))
    rank, world = world_info()
    args.device = str(device)
    if world != requested_world:
        raise ValueError('Initialized world size differs from torchrun environment')
    try:
        return _run(args, device, rank, world)
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()


def _run(args, device, rank, world):
    output = Path(args.output)
    _rank_zero_action(lambda: _prepare_output(output, args))
    base.seed_all(args.seed, device)
    train_set = LunarDataset(args.manifest, 'train', args.input_size, horizontal_flip=args.horizontal_flip,
                            max_samples=args.max_train_samples, selection_manifest=args.selection_manifest)
    val_set = LunarDataset(args.manifest, 'val', args.input_size, max_samples=args.max_val_samples)
    if set(row['source_id'] for row in train_set.records) & set(row['source_id'] for row in val_set.records):
        raise ValueError('Training and validation source families overlap')
    identities = {'train': train_set.identity, 'val': val_set.identity}
    partial = train_set.partial or val_set.partial
    identity_digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    if len(set(gather_objects(identity_digest))) != 1:
        raise ValueError('Ranks read different dataset identities')
    sampler = NoPaddingShardSampler(train_set, shuffle=True, seed=args.seed, require_equal=True)
    val_sampler = NoPaddingShardSampler(val_set)
    if len(sampler) < 1:
        raise ValueError('Every training rank must receive at least one image')
    loader_generator = torch.Generator().manual_seed(args.seed + 100003 * rank)
    val_generator = torch.Generator().manual_seed(args.seed + 1 + 100003 * rank)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == 'cuda', worker_init_fn=base.seed_worker,
        generator=loader_generator, drop_last=False,
        multiprocessing_context='spawn' if args.workers else None)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler,
        num_workers=args.workers, pin_memory=device.type == 'cuda', worker_init_fn=base.seed_worker,
        generator=val_generator, drop_last=False,
        multiprocessing_context='spawn' if args.workers else None)
    model_config = ModelConfig(variant=args.variant, input_size=args.input_size, num_queries=args.num_queries,
        decoder_layers=args.decoder_layers, pixel_decoder_layers=args.pixel_decoder_layers, width=args.width,
        adapter_dim=args.adapter_dim, adapter_heads=args.adapter_heads, adapter_points=args.adapter_points,
        gradient_checkpointing=args.gradient_checkpointing, weight_dir=args.weight_dir,
        semantic_epsilon=args.semantic_epsilon)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    model = build_model(model_config, pretrained=args.pretrained and saved is None and not args.initial_checkpoint)
    if saved:
        model.load_state_dict(saved['model'], strict=True)
    elif args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
        if ModelConfig.from_dict(initial['model_config']).to_dict() != model_config.to_dict():
            raise ValueError('Initial checkpoint architecture differs')
        model.load_state_dict(initial['model'], strict=True)
        del initial
    configuration = base._training_config(args)
    source_files = [Path(__file__), Path(__file__).with_name('distributed_training.py'),
                    Path(__file__).with_name('train.py'), *(Path(__file__).parent / 'vitseg').glob('*.py')]
    configuration['source_sha256'] = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(source_files)}
    distributed = {'strategy': 'synchronous_bucket_gradient_allreduce', 'world_size': world,
        'per_rank_batch_size': args.batch_size, 'accumulation': args.accumulation,
        'global_effective_batch': args.batch_size * world * args.accumulation,
        'backend': dist.get_backend() if dist.is_initialized() else 'single_process',
        'train_sampler': 'one seed+epoch permutation, rank-strided disjoint shards, no padding',
        'validation_sampler': 'rank-strided, unequal disjoint shards, no padding',
        'set_loss_reduction': 'global CE target-weight denominator and global matched-mask count at every stage',
        'masked_loss_reduction': 'global eligible-pixel denominator over all accumulation microbatches',
        'gradient_bucket_bytes': 25 * 1024 * 1024, 'loader_worker_start_method': 'spawn' if args.workers else None,
        'training_rng_seeds': [args.seed + 7919 * (i + 1) for i in range(world)]}
    configuration['distributed'] = distributed
    if saved:
        base._check_resume(saved, configuration, model_config, identities)
        if saved.get('distributed', {}).get('world_size') != world:
            raise ValueError('Distributed resume requires the original world size')
        states = saved.get('rank_states')
        if not isinstance(states, list) or len(states) != world or [r.get('rank') for r in states] != list(range(world)):
            raise ValueError('Checkpoint is missing ordered per-rank RNG/loader states')
        for key in ('initial_model_sha256', 'initial_body_sha256', 'initial_shared_model_sha256',
                    'initial_checkpoint_sha256', 'initialization_mode', 'pretrained_report'):
            configuration[key] = saved['train_config'].get(key)
    else:
        configuration['initial_model_sha256'] = base.tensor_state_sha256(model.state_dict())
        configuration['initial_body_sha256'] = base.tensor_state_sha256(
            {key: value for key, value in model.state_dict().items() if key.startswith('encoder.')})
        configuration['initial_shared_model_sha256'] = base.tensor_state_sha256(
            {key: value for key, value in model.state_dict().items() if not key.startswith(('adapter.', 'fusion.'))})
        configuration['initialization_mode'] = ('checkpoint_finetune' if args.initial_checkpoint else
                                               'original_pretrained' if args.pretrained else 'random')
        configuration['initial_checkpoint_sha256'] = sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
        configuration['pretrained_report'] = getattr(model, 'pretrained_report', {})
    if len(set(gather_objects(configuration['initial_model_sha256']))) != 1:
        raise ValueError('Initial model tensors differ across ranks')
    model.to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    broadcast_model(model)
    head_parameters, backbone_parameters = list(model.head_parameters()), list(model.backbone_parameters())
    if {id(p) for p in head_parameters} & {id(p) for p in backbone_parameters}:
        raise ValueError('Optimizer parameter groups overlap')
    if {id(p) for p in head_parameters + backbone_parameters} != {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer does not cover every model parameter')
    optimizer = torch.optim.AdamW([{'params': head_parameters, 'lr': args.head_lr, 'name': 'head'},
                                  {'params': backbone_parameters, 'lr': args.backbone_lr, 'name': 'backbone'}],
                                 weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    schedule = base.PolynomialSchedule(optimizer, args.epochs * updates_per_epoch,
                                       args.head_lr, args.backbone_lr, args.poly_power)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and args.amp == 'fp16')
    criterion = Mask2FormerCriterion(num_classes=5, num_points=args.num_points, class_weight=args.class_weight,
        mask_weight=args.mask_weight, dice_weight=args.dice_weight, no_object_weight=args.no_object_weight,
        ignore_index=255).to(device)
    tracker = base.PatienceTracker(args.early_stopping_patience, args.early_stopping_min_delta,
                                   args.early_stopping_start_epoch)
    start_epoch, global_update, best, best_epoch = 1, 0, -math.inf, None
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        tracker.load_state_dict(saved['early_stopping'])
        global_update, start_epoch = saved['global_update'], saved['epoch'] + 1
        if global_update != schedule.completed_updates:
            raise ValueError('Checkpoint update counters disagree')
        best, best_epoch = saved['best_miou'], saved['best_epoch']
        state = saved['rank_states'][rank]
        if state.get('sampler_epoch') != saved['epoch']:
            raise ValueError('Checkpoint sampler epoch differs')
        loader_generator.set_state(state['loader_generator'])
        val_generator.set_state(state['val_generator'])
        base.restore_rng(state['rng'], device)
        saved['_resume_path'] = args.resume
        _rank_zero_action(lambda: _validate_history(output, saved, partial))
        del saved
    else:
        base.seed_all(distributed['training_rng_seeds'][rank], device)
    _rank_zero_action(lambda: (base.write_json(output / 'run_config.json', configuration),
                              base.write_json(output / 'data_identity.json', identities)))
    if rank == 0:
        print(json.dumps({'output': str(output), 'train_images': len(train_set), 'val_images': len(val_set),
            'partial': partial, 'distributed': distributed, 'updates_per_epoch': updates_per_epoch,
            'initial_model_sha256': configuration['initial_model_sha256']}, ensure_ascii=False), flush=True)
    if tracker.stopped:
        if rank == 0:
            print('Checkpoint already satisfies early stopping; no extra epoch.', flush=True)
        return output
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        frozen = epoch <= args.freeze_epochs
        sampler.set_epoch(epoch)
        model.train()
        model.freeze_backbones(frozen)
        optimizer.zero_grad(set_to_none=True)
        loss_sums = defaultdict(float)
        data_hash, masking_hash = hashlib.sha256(), hashlib.sha256()
        samples_seen, masked_microbatches, masked_updates, successful_updates = 0, 0, 0, 0
        mask_stats = _merge_mask_stats([])
        for window_start, batches in _window_iterator(train_loader, args.accumulation):
            window_images = sum(len(batch['image']) for batch in batches)
            schedule.apply(frozen)
            use_mask = (args.masking and not frozen and args.masking_weight > 0 and
                        update_is_masked(global_update, args.masking_seed, args.masking_update_fraction))
            decisions = gather_objects(use_mask)
            if any(value != use_mask for value in decisions):
                raise ValueError('Ranks disagree about the masked optimizer update')
            joint_update = max(0, global_update - args.freeze_epochs * updates_per_epoch)
            coefficient = args.masking_weight * warmup_factor(joint_update, args.masking_warmup_updates) if use_mask else 0.0
            prepared = []
            if use_mask:
                for batch in batches:
                    hidden = make_block_masks(batch['mask'], base._sample_keys(batch), global_update,
                                              args.masking_seed, args.masking_ratio, args.masking_block_size)
                    prepared.append((hidden, prepare_masked_support(batch['mask'], hidden)))
            local_denominator = sum(item[1][2]['denominator_pixels'] for item in prepared)
            global_denominator = int(reduce_sum(torch.tensor(local_denominator, device=device, dtype=torch.int64)))
            for offset, batch in enumerate(batches):
                batch_index = window_start + offset
                images, targets = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
                batch_size = len(images)
                for key, flipped in zip(base._sample_keys(batch), batch['horizontal_flipped']):
                    data_hash.update((key + '|hflip=' + str(bool(flipped)) + '\n').encode())
                with autocast_context(device, args.amp):
                    outputs = model(images)
                    losses = distributed_set_loss(outputs, targets, criterion)
                    base_loss = losses['loss_total']
                assert_all_finite(base_loss, f'intact loss at epoch {epoch}, batch {batch_index}')
                scaler.scale(base_loss * (batch_size / window_images)).backward()
                base_value = float(base_loss.detach())
                for key, value in losses.items():
                    loss_sums[key] += float(value.detach()) * batch_size
                loss_sums['loss_base'] += base_value * batch_size
                del outputs, losses, base_loss
                masked_value, masked_share = 0.0, 0.0
                if use_mask:
                    hidden, support = prepared[offset]
                    hidden = hidden.to(device, non_blocking=True)
                    with autocast_context(device, args.amp):
                        masked_outputs = model(apply_block_mask(images, hidden))
                        masked_loss, current_stats = masked_semantic_nll(masked_outputs['semantic_logits'], targets,
                            hidden, args.masking_hidden_weight, prepared=support)
                    assert_all_finite(masked_loss, 'masked semantic loss')
                    objective = distributed_masked_loss(masked_loss, current_stats['denominator_pixels'], global_denominator)
                    scaler.scale(coefficient * objective).backward()
                    masked_value = float(masked_loss.detach())
                    masked_share = world * masked_value * current_stats['denominator_pixels'] / max(global_denominator, 1)
                    base._sum_mask_stats(mask_stats, current_stats)
                    masking_hash.update((str(global_update) + '|' + current_stats['mask_sha256'] + '\n').encode())
                    masked_microbatches += 1
                    del masked_outputs, masked_loss, objective, hidden
                loss_sums['loss_masking_raw'] += masked_share * window_images
                loss_sums['loss_masking_weighted'] += coefficient * masked_share * window_images
                loss_sums['loss_combined'] += base_value * batch_size + coefficient * masked_share * window_images
                samples_seen += batch_size
                if args.log_every and (batch_index + 1) % args.log_every == 0 and rank == 0:
                    print(json.dumps({'epoch': epoch, 'batch': batch_index + 1, 'batches': len(train_loader),
                        'global_update': global_update, 'backbones_frozen': frozen,
                        'rank0_scaled_loss_base': base_value, 'rank0_masking_nll': masked_value,
                        'masking_coefficient': coefficient, 'global_effective_batch': distributed['global_effective_batch']},
                        ensure_ascii=False), flush=True)
            synchronize_gradients(model, distributed['gradient_bucket_bytes'])
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip if args.gradient_clip else math.inf)
            assert_all_finite(gradient_norm, 'synchronized gradient norm')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise FloatingPointError('Optimizer skipped a synchronized update')
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            schedule.completed_updates = global_update
            successful_updates += 1
            masked_updates += int(use_mask)
        validation = distributed_evaluate(model, val_loader, device, amp=args.amp, tta=args.val_tta,
            tta_sizes=args.val_tta_sizes, tta_weights=args.val_tta_weights, progress_every=args.val_progress_every)
        if validation['partial'] != val_set.partial or validation['image_count'] != len(val_set):
            raise ValueError('Validation did not cover the configured fixed set')
        score = validation['miou']
        if score is None or not math.isfinite(score):
            raise ValueError('Validation mIoU is undefined or non-finite')
        decision = None
        if rank == 0:
            improved = score > best
            if improved:
                best, best_epoch = score, epoch
            stopped = tracker.update(epoch, score)
            decision = {'improved': improved, 'best': best, 'best_epoch': best_epoch,
                        'stopped': stopped, 'tracker': tracker.state_dict()}
        decision = broadcast_object(decision)
        best, best_epoch, stopped = decision['best'], decision['best_epoch'], decision['stopped']
        tracker.load_state_dict(decision['tracker'])
        status = 'early_stopped' if stopped else 'complete' if epoch == args.epochs else 'training'
        stop_reason = ('No significant validation gain for configured patience' if stopped else
                       'Maximum epochs completed' if status == 'complete' else None)
        # One epoch-boundary tensor digest validates that optimizer updates and
        # model buffers stayed synchronized; it does not consume random state.
        parameter_hashes = gather_objects(base.tensor_state_sha256(model.state_dict()))
        if len(set(parameter_hashes)) != 1:
            raise ValueError('Model tensors diverged between distributed ranks')
        payloads = gather_objects({'rank': rank, 'images': samples_seen, 'loss_sums': dict(loss_sums),
            'mask_stats': mask_stats, 'masked_updates': masked_updates, 'masked_microbatches': masked_microbatches,
            'data_hash': data_hash.hexdigest(), 'masking_hash': masking_hash.hexdigest(),
            'seconds': time.monotonic() - epoch_start, 'optimizer_updates': successful_updates})
        if sum(p['images'] for p in payloads) != len(train_set):
            raise ValueError('Training shards do not cover every selected image exactly once')
        if len({p['optimizer_updates'] for p in payloads}) != 1 or len({p['masked_updates'] for p in payloads}) != 1:
            raise ValueError('Rank optimizer/masking update counts diverged')
        averages = {key: sum(p['loss_sums'].get(key, 0.0) for p in payloads) / len(train_set)
                    for key in sorted(set().union(*(p['loss_sums'] for p in payloads)))}
        averages['masking_to_base_ratio'] = averages['loss_masking_weighted'] / averages['loss_base'] if averages['loss_base'] else None
        mask_record = {'enabled': args.masking, 'selected_optimizer_updates': masked_updates,
            'selected_microbatches': sum(p['masked_microbatches'] for p in payloads),
            'stats': _merge_mask_stats([p['mask_stats'] for p in payloads]),
            'mask_sequence_sha256': hashlib.sha256(json.dumps([p['masking_hash'] for p in payloads]).encode()).hexdigest(),
            'per_rank_mask_sequence_sha256': [p['masking_hash'] for p in payloads],
            'selection_scope': 'one shared decision per global optimizer update, all ranks/microbatches',
            'pattern_scope': 'independent per stable sample key and global optimizer update'}
        record = {'epoch': epoch, 'global_update': global_update, 'optimizer_updates': successful_updates,
            'backbones_frozen': frozen, 'losses': averages, 'validation_miou': score, 'miou': score,
            'best_miou': best, 'best_epoch': best_epoch, 'selection_scene': 'custom_validation',
            'checkpoint_selection_uses_test': False, 'partial': partial, 'status': status, 'stop_reason': stop_reason,
            'loss_log_reduction': 'global-image weighted synchronized microbatch M2F; global-window eligible-pixel masked NLL',
            'data_order_and_hflip_sha256': hashlib.sha256(json.dumps([p['data_hash'] for p in payloads]).encode()).hexdigest(),
            'per_rank_data_order_and_hflip_sha256': [p['data_hash'] for p in payloads],
            'masking': mask_record, 'early_stopping': tracker.state_dict(), 'seconds': max(p['seconds'] for p in payloads),
            'distributed': {**distributed, 'train_images_per_rank': [p['images'] for p in payloads],
                            'epoch_seconds_per_rank': [p['seconds'] for p in payloads],
                            'model_tensor_sha256_per_rank': parameter_hashes, 'model_tensors_identical': True}}
        validation.update(epoch=epoch, selection_scene='custom_validation', checkpoint_selection_uses_test=False)
        rank_states = gather_objects({'rank': rank, 'rng': base.rng_state(device),
            'loader_generator': loader_generator.get_state(), 'val_generator': val_generator.get_state(),
            'sampler_epoch': epoch})
        def commit_epoch():
            base.write_json(output / f'validation_epoch_{epoch:03d}.json', validation)
            base.write_json(output / f'masking_epoch_{epoch:03d}.json', mask_record)
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            checkpoint = {'format': 'lusnar_vit_checkpoint_v1', 'model_config': model_config.to_dict(),
                'train_config': configuration, 'run_config': configuration, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'global_update': global_update, 'best_miou': best, 'best_epoch': best_epoch,
                'metrics': validation, 'data_identity': identities, 'rng': rank_states[0]['rng'],
                'loader_generator': rank_states[0]['loader_generator'], 'val_generator': rank_states[0]['val_generator'],
                'rank_states': rank_states, 'distributed': distributed,
                'model_tensor_sha256_per_rank': parameter_hashes, 'early_stopping': tracker.state_dict(),
                'status': status, 'stop_reason': stop_reason, 'partial': partial}
            base.atomic_checkpoint(output / 'last.pt', checkpoint)
            if decision['improved']:
                base.atomic_checkpoint(output / ('best_smoke.pt' if partial else 'best.pt'), checkpoint)
            summary = {'format': 'lusnar_vit_run_v1', 'status': status, 'stop_reason': stop_reason,
                'completed_epochs': epoch, 'best_epoch': best_epoch, 'best_miou': best, 'partial': partial,
                'train_images': len(train_set), 'validation_images': len(val_set), 'selection_scene': 'custom_validation',
                'checkpoint_selection_uses_test': False, 'official_test_used_for_selection': False,
                'early_stopping': tracker.state_dict(), 'global_update': global_update,
                'model_config': model_config.to_dict(), 'data_identity': identities,
                'initial_model_sha256': configuration['initial_model_sha256'], 'distributed': distributed,
                'model_tensor_sha256_per_rank': parameter_hashes}
            base.write_json(output / 'summary.json', summary)
            print(json.dumps({'epoch': epoch, 'miou': score, 'best_epoch': best_epoch, 'best_miou': best,
                'status': status, 'seconds': record['seconds'], 'global_update': global_update}, ensure_ascii=False), flush=True)
        _rank_zero_action(commit_epoch)
        if stopped:
            break
    return output


if __name__ == '__main__':
    main()
