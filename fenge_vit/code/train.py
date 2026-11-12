#!/usr/bin/env python3
"""Train the independent DINOv3 ViT/ConvNeXt Mask2Former experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vitseg.data import LunarDataset, sha256_file
from vitseg.losses import Mask2FormerCriterion
from vitseg.masking import (apply_block_mask, make_block_masks, masked_semantic_nll,
                            update_is_masked, warmup_factor, prepare_masked_support)
from vitseg.metrics import autocast_context, evaluate_loader, validate_tta_sizes, validate_tta_weights
from vitseg.model import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/home/lzc/workspace/fenge/result/selections')
DEFAULT_MANIFEST = DATA_ROOT / 'random80_4000_2000_2000_pool_20261112.json'
DEFAULT_SELECTION = DATA_ROOT / 'random80_4000_2000_2000_seed42_20261112.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(tensor.dtype).encode() + b'\0')
        digest.update(json.dumps(list(tensor.shape)).encode() + b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)


def rng_state(device):
    numpy = np.random.get_state()
    state = {'python': random.getstate(), 'numpy': [numpy[0], numpy[1].tolist(), *numpy[2:]],
             'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)


class PolynomialSchedule:
    """Update-based two-group schedule, explicitly serialized for exact resume."""
    def __init__(self, optimizer, total_updates, head_lr, backbone_lr, power=0.9):
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.base_lrs = [head_lr, backbone_lr]
        self.power = power
        self.completed_updates = 0

    def apply(self, frozen):
        factor = max(0.0, 1 - self.completed_updates / self.total_updates) ** self.power
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = 0.0 if frozen and group['name'] == 'backbone' else base * factor

    def state_dict(self):
        return {'total_updates': self.total_updates, 'base_lrs': self.base_lrs,
                'power': self.power, 'completed_updates': self.completed_updates}

    def load_state_dict(self, state):
        if any(state[key] != self.state_dict()[key] for key in ('total_updates', 'base_lrs', 'power')):
            raise ValueError('Resume schedule differs from the original run')
        self.completed_updates = state['completed_updates']


class PatienceTracker:
    """Significant gains accumulate against the latest significant anchor."""
    def __init__(self, patience=5, min_delta=0.0005, start_epoch=4):
        self.patience, self.min_delta, self.start_epoch = patience, min_delta, start_epoch
        self.anchor = None
        self.bad_epochs = 0
        self.last_epoch = 0
        self.stopped = False

    def update(self, epoch, score):
        if epoch <= self.last_epoch:
            raise ValueError('Early stopping epochs must increase strictly')
        self.last_epoch = epoch
        if self.patience == 0 or epoch < self.start_epoch:
            return False
        if self.anchor is None or score >= self.anchor + self.min_delta - 1e-12:
            self.anchor, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if any(state[key] != getattr(self, key) for key in ('patience', 'min_delta', 'start_epoch')):
            raise ValueError('Resume early stopping policy differs')
        self.__dict__.update(state)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--selection-manifest', default=str(DEFAULT_SELECTION))
    parser.add_argument('--output', default='result/b_adapter')
    parser.add_argument('--weight-dir', default='model')
    parser.add_argument('--variant', choices=['fusion', 'adapter'], default='adapter')
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--num-queries', type=int, default=32)
    parser.add_argument('--decoder-layers', type=int, default=6)
    parser.add_argument('--pixel-decoder-layers', type=int, default=6)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--adapter-dim', type=int, default=384)
    parser.add_argument('--adapter-heads', type=int, default=6)
    parser.add_argument('--adapter-points', type=int, default=4)
    parser.add_argument('--semantic-epsilon', type=float, default=1e-8)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--initial-checkpoint', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--freeze-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--poly-power', type=float, default=0.9)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'none'], default='bf16')
    parser.add_argument('--horizontal-flip', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-points', type=int, default=4096)
    parser.add_argument('--class-weight', type=float, default=2.0)
    parser.add_argument('--mask-weight', type=float, default=5.0)
    parser.add_argument('--dice-weight', type=float, default=5.0)
    parser.add_argument('--no-object-weight', type=float, default=0.1)
    parser.add_argument('--masking', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--masking-update-fraction', type=float, default=0.25)
    parser.add_argument('--masking-ratio', type=float, default=0.1)
    parser.add_argument('--masking-block-size', type=int, default=32)
    parser.add_argument('--masking-weight', type=float, default=0.5)
    parser.add_argument('--masking-hidden-weight', type=float, default=0.2)
    parser.add_argument('--masking-warmup-updates', type=int, default=1000)
    parser.add_argument('--masking-seed', type=int, default=4242)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0005)
    parser.add_argument('--early-stopping-start-epoch', type=int, default=None)
    parser.add_argument('--val-tta', choices=['none', 'hflip', 'multiscale_hflip'], default='none')
    parser.add_argument('--val-tta-sizes', nargs='+', type=int)
    parser.add_argument('--val-tta-weights', nargs='+', type=float)
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--val-progress-every', type=int, default=50)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        source = json.loads(preliminary.config.read_text())
        if not isinstance(source, dict):
            parser.error('Configuration must be a JSON object')
        known = {action.dest: action for action in parser._actions}
        if set(source) - set(known):
            parser.error(f'Unknown configuration keys: {sorted(set(source) - set(known))}')
        for key, value in source.items():
            action = known[key]
            if isinstance(action, argparse.BooleanOptionalAction) and type(value) is not bool:
                parser.error(f'{key} must be a JSON boolean')
        parser.set_defaults(**source)
    args = parser.parse_args(argv)
    for key in ('manifest', 'selection_manifest', 'output', 'weight_dir', 'resume', 'initial_checkpoint'):
        value = getattr(args, key)
        if value is not None:
            value = Path(value).expanduser()
            setattr(args, key, str((ROOT / value).resolve() if not value.is_absolute() else value.resolve()))
    if args.early_stopping_start_epoch is None:
        args.early_stopping_start_epoch = args.freeze_epochs + 1
    integers_positive = ('input_size', 'epochs', 'batch_size', 'accumulation', 'val_batch_size',
                         'num_queries', 'decoder_layers', 'pixel_decoder_layers', 'width', 'adapter_dim',
                         'adapter_heads', 'adapter_points', 'num_points', 'masking_block_size',
                         'early_stopping_start_epoch')
    for key in integers_positive:
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            parser.error(f'{key} must be a positive integer')
    for key in ('freeze_epochs', 'workers', 'seed', 'masking_seed', 'masking_warmup_updates',
                'early_stopping_patience', 'log_every', 'val_progress_every'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 0:
            parser.error(f'{key} must be a nonnegative integer')
    if args.input_size % 32 or args.freeze_epochs > args.epochs:
        parser.error('input_size must divide by 32 and freeze_epochs cannot exceed epochs')
    for key in ('head_lr', 'backbone_lr', 'semantic_epsilon', 'poly_power', 'class_weight',
                'mask_weight', 'dice_weight', 'no_object_weight'):
        value = getattr(args, key)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            parser.error(f'{key} must be finite and positive')
    for key in ('weight_decay', 'gradient_clip', 'masking_weight', 'early_stopping_min_delta'):
        value = getattr(args, key)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            parser.error(f'{key} must be finite and nonnegative')
    for key in ('masking_update_fraction', 'masking_ratio', 'masking_hidden_weight'):
        value = getattr(args, key)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f'{key} must be finite and in [0,1]')
    for key in ('max_train_samples', 'max_val_samples'):
        if getattr(args, key) is not None and (type(getattr(args, key)) is not int or getattr(args, key) < 1):
            parser.error(f'{key} must be positive or null')
    for key in ('variant', 'amp', 'val_tta'):
        if getattr(args, key) not in next(action.choices for action in parser._actions if action.dest == key):
            parser.error(f'Invalid {key}')
    validate_tta_sizes(args.val_tta, args.val_tta_sizes)
    validate_tta_weights(args.val_tta, args.val_tta_sizes, args.val_tta_weights)
    if args.resume and args.initial_checkpoint:
        parser.error('resume and initial_checkpoint are mutually exclusive')
    return args


def _training_config(args):
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def _check_resume(saved, configuration, model_config, identities):
    if saved.get('format') != 'lusnar_vit_checkpoint_v1':
        raise ValueError('Resume requires an independent ViT training checkpoint')
    if ModelConfig.from_dict(saved['model_config']).to_dict() != model_config.to_dict():
        raise ValueError('Resume model configuration differs')
    if saved.get('data_identity') != identities:
        raise ValueError('Resume data identity differs, including the fixed training selection')
    old = saved['train_config']
    ignored = {'config', 'resume', 'device', 'log_every', 'val_progress_every'}
    generated = {'initial_model_sha256', 'initial_body_sha256', 'initial_shared_model_sha256', 'initial_checkpoint_sha256',
                 'initialization_mode', 'pretrained_report', 'source_sha256'}
    for key, value in configuration.items():
        if key not in ignored | generated and old.get(key) != value:
            raise ValueError(f'Resume training configuration changed: {key}')
    if old.get('source_sha256') != configuration.get('source_sha256'):
        raise ValueError('Resume implementation SHA differs; use the recorded experiment code')


def _sample_keys(batch):
    return [f'{scene}/{sample}|source={source}' for scene, sample, source in
            zip(batch['scene'], batch['sample_id'], batch['source_id'])]


def _sum_mask_stats(total, stats):
    for key in ('visible_valid_pixels', 'eligible_hidden_pixels', 'denominator_pixels', 'masked_pixels', 'total_pixels'):
        total[key] += stats[key]
    for key in ('ignored_component_counts', 'ignored_component_pixels'):
        for label, value in stats[key].items():
            total[key][label] += value


def _training_windows(loader, args, current_update, updates_per_epoch, frozen):
    """Prefetch only one CPU accumulation window to know its exact NLL support.

    No forward graph is retained while determining the denominator. The final
    short window uses its real image count, not the configured accumulation.
    """
    iterator = iter(loader)
    for start in range(0, len(loader), args.accumulation):
        batches = [next(iterator) for _ in range(min(args.accumulation, len(loader) - start))]
        update = current_update()
        use_mask = (args.masking and not frozen and args.masking_weight > 0 and
                    update_is_masked(update, args.masking_seed, args.masking_update_fraction))
        joint_update = max(0, update - args.freeze_epochs * updates_per_epoch)
        coefficient = args.masking_weight * warmup_factor(joint_update, args.masking_warmup_updates) if use_mask else 0.0
        prepared = []
        if use_mask:
            for batch in batches:
                hidden = make_block_masks(batch['mask'], _sample_keys(batch), update, args.masking_seed,
                                          args.masking_ratio, args.masking_block_size)
                support = prepare_masked_support(batch['mask'], hidden)
                prepared.append((hidden, support))
        denominator = sum(item[1][2]['denominator_pixels'] for item in prepared)
        image_count = sum(batch['image'].shape[0] for batch in batches)
        for offset, batch in enumerate(batches):
            yield start + offset, batch, image_count, (use_mask, coefficient, denominator,
                                                       prepared[offset] if use_mask else None)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Selected CUDA device does not support BF16')
    seed_all(args.seed, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ('last.pt', 'history.jsonl', 'summary.json')):
        raise FileExistsError('Output contains a training run; choose a new output or explicitly resume')
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError('Resume must use its original output directory to retain its validation history and best')
    train_set = LunarDataset(args.manifest, 'train', args.input_size, horizontal_flip=args.horizontal_flip,
                             max_samples=args.max_train_samples, selection_manifest=args.selection_manifest)
    val_set = LunarDataset(args.manifest, 'val', args.input_size, max_samples=args.max_val_samples)
    if set(row['source_id'] for row in train_set.records) & set(row['source_id'] for row in val_set.records):
        raise ValueError('Training and validation source families overlap')
    identities = {'train': train_set.identity, 'val': val_set.identity}
    partial = train_set.partial or val_set.partial
    loader_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                              generator=loader_generator, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=val_generator)
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
            raise ValueError('Initial checkpoint model configuration differs')
        model.load_state_dict(initial['model'], strict=True)
    configuration = _training_config(args)
    source_files = [Path(__file__), *(Path(__file__).parent / 'vitseg').glob('*.py')]
    configuration['source_sha256'] = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(source_files)}
    if saved:
        _check_resume(saved, configuration, model_config, identities)
        for key in ('initial_model_sha256', 'initial_body_sha256', 'initial_shared_model_sha256', 'initial_checkpoint_sha256',
                    'initialization_mode', 'pretrained_report'):
            configuration[key] = saved['train_config'].get(key)
    else:
        configuration['initial_model_sha256'] = tensor_state_sha256(model.state_dict())
        configuration['initial_body_sha256'] = tensor_state_sha256(
            {key: value for key, value in model.state_dict().items() if key.startswith('encoder.')})
        configuration['initial_shared_model_sha256'] = tensor_state_sha256(
            {key: value for key, value in model.state_dict().items() if not key.startswith(('adapter.', 'fusion.'))})
        configuration['initialization_mode'] = ('checkpoint_finetune' if args.initial_checkpoint else
                                               'original_pretrained' if args.pretrained else 'random')
        configuration['initial_checkpoint_sha256'] = sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
        configuration['pretrained_report'] = getattr(model, 'pretrained_report', {})
    model.to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    head_parameters, backbone_parameters = list(model.head_parameters()), list(model.backbone_parameters())
    if {id(p) for p in head_parameters} & {id(p) for p in backbone_parameters}:
        raise ValueError('Optimizer parameter groups overlap')
    if {id(p) for p in head_parameters + backbone_parameters} != {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer does not cover every model parameter')
    optimizer = torch.optim.AdamW([{'params': head_parameters, 'lr': args.head_lr, 'name': 'head'},
                                   {'params': backbone_parameters, 'lr': args.backbone_lr, 'name': 'backbone'}],
                                  weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    schedule = PolynomialSchedule(optimizer, args.epochs * updates_per_epoch,
                                  args.head_lr, args.backbone_lr, args.poly_power)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and args.amp == 'fp16')
    criterion = Mask2FormerCriterion(num_classes=5, num_points=args.num_points, class_weight=args.class_weight,
        mask_weight=args.mask_weight, dice_weight=args.dice_weight, no_object_weight=args.no_object_weight,
        ignore_index=255).to(device)
    tracker = PatienceTracker(args.early_stopping_patience, args.early_stopping_min_delta,
                              args.early_stopping_start_epoch)
    start_epoch, global_update, best, best_epoch = 1, 0, -math.inf, None
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        tracker.load_state_dict(saved['early_stopping'])
        global_update, start_epoch = saved['global_update'], saved['epoch'] + 1
        if global_update != schedule.completed_updates:
            raise ValueError('Checkpoint optimizer update counters disagree')
        best, best_epoch = saved['best_miou'], saved['best_epoch']
        loader_generator.set_state(saved['loader_generator'])
        val_generator.set_state(saved['val_generator'])
        restore_rng(saved['rng'], device)
        history = output / 'history.jsonl'
        rows = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        committed_count = saved['epoch']
        if ([row['epoch'] for row in rows[:committed_count]] != list(range(1, committed_count + 1))
                or len(rows) < committed_count):
            raise ValueError('Resume requires matching complete history through the saved checkpoint epoch')
        if len(rows) > committed_count:
            if Path(args.resume).name != 'last.pt':
                raise ValueError('Only last.pt can recover an uncommitted history tail')
            # A validation/history write can finish before last.pt atomically
            # commits. Preserve those uncommitted observations and replay from
            # the optimizer/RNG state of the durable checkpoint boundary.
            archive = output / f'history_uncommitted_after_{committed_count:03d}_{time.time_ns()}.jsonl'
            archive.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[committed_count:]))
            temporary = history.with_name('history.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[:committed_count]))
            temporary.replace(history)
        if not (output / ('best_smoke.pt' if partial else 'best.pt')).exists():
            raise FileNotFoundError('Resume requires the saved best checkpoint in the original output')
    write_json(output / 'run_config.json', configuration)
    write_json(output / 'data_identity.json', identities)
    print(json.dumps({'output': str(output), 'train_images': len(train_set), 'val_images': len(val_set),
                      'partial': partial, 'microbatch': args.batch_size, 'accumulation': args.accumulation,
                      'effective_batch': args.batch_size * args.accumulation, 'initial_model_sha256':
                      configuration['initial_model_sha256']}, ensure_ascii=False), flush=True)
    if tracker.stopped:
        print('Checkpoint already satisfies the configured early stopping rule; no extra epoch.', flush=True)
        return
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        frozen = epoch <= args.freeze_epochs
        model.train()
        model.freeze_backbones(frozen)
        optimizer.zero_grad(set_to_none=True)
        loss_sums = defaultdict(float)
        data_hash, masking_hash = hashlib.sha256(), hashlib.sha256()
        samples_seen, masked_microbatches, masked_updates, successful_updates = 0, 0, 0, 0
        mask_stats = {'visible_valid_pixels': 0, 'eligible_hidden_pixels': 0, 'denominator_pixels': 0,
                      'masked_pixels': 0, 'total_pixels': 0, 'ignored_component_counts': {str(i): 0 for i in (1,2,3)},
                      'ignored_component_pixels': {str(i): 0 for i in (1,2,3)}}
        for batch_index, batch, window_images, mask_setup in _training_windows(
                train_loader, args, lambda: global_update, updates_per_epoch, frozen):
            window_start = (batch_index // args.accumulation) * args.accumulation
            window_end = min(window_start + args.accumulation, len(train_loader))
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True)
            batch_size = images.shape[0]
            sample_keys = _sample_keys(batch)
            for key, flipped in zip(sample_keys, batch['horizontal_flipped']):
                data_hash.update((key + '|hflip=' + str(bool(flipped)) + '\n').encode())
            schedule.apply(frozen)
            use_mask, coefficient, window_denominator, prepared_mask = mask_setup
            # Finish this graph before constructing a masked graph. Nothing from
            # the intact branch is detached before its complete M2F backward.
            with autocast_context(device, args.amp):
                outputs = model(images)
                losses = criterion(outputs, targets)
                base_loss = losses['loss_total']
            if not torch.isfinite(base_loss):
                raise FloatingPointError(f'Non-finite intact loss at epoch {epoch}, batch {batch_index}')
            scaler.scale(base_loss * (batch_size / window_images)).backward()
            base_value = float(base_loss.detach())
            for key, value in losses.items():
                loss_sums[key] += float(value.detach()) * batch_size
            loss_sums['loss_base'] += base_value * batch_size
            del outputs, losses, base_loss
            masked_value, weighted_masked = 0.0, 0.0
            if use_mask:
                hidden, prepared_support = prepared_mask
                hidden = hidden.to(device, non_blocking=True)
                with autocast_context(device, args.amp):
                    masked_outputs = model(apply_block_mask(images, hidden))
                    masked_loss, current_stats = masked_semantic_nll(masked_outputs['semantic_logits'], targets,
                                                                    hidden, args.masking_hidden_weight,
                                                                    prepared=prepared_support)
                if not torch.isfinite(masked_loss):
                    raise FloatingPointError('Non-finite masked semantic loss')
                scaler.scale(masked_loss * (coefficient * current_stats['denominator_pixels'] /
                                          max(window_denominator, 1))).backward()
                masked_value = float(masked_loss.detach())
                weighted_masked = coefficient * masked_value
                _sum_mask_stats(mask_stats, current_stats)
                masking_hash.update((str(global_update) + '|' + current_stats['mask_sha256'] + '\n').encode())
                masked_microbatches += 1
                if batch_index == window_start:
                    masked_updates += 1
                del masked_outputs, masked_loss, hidden
            # Log the objective actually differentiated for each complete
            # accumulation window, weighted by the window's real image count.
            masked_share = (masked_value * current_stats['denominator_pixels'] /
                            max(window_denominator, 1)) if use_mask else 0.0
            loss_sums['loss_masking_raw'] += masked_share * window_images
            loss_sums['loss_masking_weighted'] += coefficient * masked_share * window_images
            loss_sums['loss_combined'] += base_value * batch_size + coefficient * masked_share * window_images
            samples_seen += batch_size
            if batch_index + 1 == window_end:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip
                                                               if args.gradient_clip else math.inf)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError('Non-finite model gradient; update/checkpoint not committed')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Finite gradients above make AMP skipping unexpected. Do not
                # advance a mask/update schedule when an optimizer step failed.
                if scaler.get_scale() < old_scale:
                    raise FloatingPointError('FP16 optimizer update skipped; exact update schedule would diverge')
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                schedule.completed_updates = global_update
                successful_updates += 1
            if args.log_every and (batch_index + 1) % args.log_every == 0:
                print(json.dumps({'epoch': epoch, 'batch': batch_index + 1, 'batches': len(train_loader),
                    'global_update': global_update, 'backbones_frozen': frozen, 'loss_base': base_value,
                    'loss_masking_raw': masked_value, 'loss_masking_weighted': weighted_masked,
                    'masking_coefficient': coefficient}, ensure_ascii=False), flush=True)
        validation = evaluate_loader(model, val_loader, device, amp=args.amp, tta=args.val_tta,
            tta_sizes=args.val_tta_sizes, tta_weights=args.val_tta_weights, progress_every=args.val_progress_every)
        if validation['partial'] != val_set.partial or validation['image_count'] != len(val_set):
            raise ValueError('Validation did not cover its entire configured data set')
        score = validation['miou']
        if score is None or not math.isfinite(score):
            raise ValueError('Validation mIoU is undefined or non-finite')
        improved = score > best
        if improved:
            best, best_epoch = score, epoch
        stopped = tracker.update(epoch, score)
        status = 'early_stopped' if stopped else 'complete' if epoch == args.epochs else 'training'
        stop_reason = ('No significant validation gain for configured patience' if stopped else
                       'Maximum epochs completed' if status == 'complete' else None)
        averages = {key: value / samples_seen for key, value in loss_sums.items()}
        averages['masking_to_base_ratio'] = averages['loss_masking_weighted'] / averages['loss_base'] if averages['loss_base'] else None
        record = {'epoch': epoch, 'global_update': global_update, 'optimizer_updates': successful_updates,
                  'backbones_frozen': frozen, 'losses': averages, 'validation_miou': score,
                  'miou': score, 'best_miou': best, 'best_epoch': best_epoch,
                  'selection_scene': 'custom_validation', 'checkpoint_selection_uses_test': False,
                  'partial': partial, 'status': status, 'stop_reason': stop_reason,
                  'loss_log_reduction': 'Image-count-weighted mean of actual accumulation-window objectives; masked NLL uses complete-window eligible-pixel denominator',
                  'data_order_and_hflip_sha256': data_hash.hexdigest(),
                  'masking': {'enabled': args.masking, 'selected_optimizer_updates': masked_updates,
                              'selected_microbatches': masked_microbatches, 'stats': mask_stats,
                              'mask_sequence_sha256': masking_hash.hexdigest(),
                              'selection_scope': 'one independent hash decision per optimizer update, all microbatches',
                              'pattern_scope': 'independent per stable sample key and optimizer update'},
                  'early_stopping': tracker.state_dict(), 'seconds': time.monotonic() - epoch_start}
        validation.update(epoch=epoch, selection_scene='custom_validation', checkpoint_selection_uses_test=False)
        write_json(output / f'validation_epoch_{epoch:03d}.json', validation)
        write_json(output / f'masking_epoch_{epoch:03d}.json', record['masking'])
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint = {'format': 'lusnar_vit_checkpoint_v1', 'model_config': model_config.to_dict(),
                      'train_config': configuration, 'run_config': configuration, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(),
                      'scaler': scaler.state_dict(), 'epoch': epoch, 'global_update': global_update,
                      'best_miou': best, 'best_epoch': best_epoch, 'metrics': validation,
                      'data_identity': identities, 'rng': rng_state(device),
                      'loader_generator': loader_generator.get_state(), 'val_generator': val_generator.get_state(),
                      'early_stopping': tracker.state_dict(), 'status': status, 'stop_reason': stop_reason,
                      'partial': partial}
        atomic_checkpoint(output / 'last.pt', checkpoint)
        if improved:
            atomic_checkpoint(output / ('best_smoke.pt' if partial else 'best.pt'), checkpoint)
        summary = {'format': 'lusnar_vit_run_v1', 'status': status, 'stop_reason': stop_reason,
                   'completed_epochs': epoch, 'best_epoch': best_epoch, 'best_miou': best,
                   'partial': partial, 'train_images': len(train_set), 'validation_images': len(val_set),
                   'selection_scene': 'custom_validation', 'checkpoint_selection_uses_test': False,
                   'official_test_used_for_selection': False, 'early_stopping': tracker.state_dict(),
                   'global_update': global_update, 'model_config': model_config.to_dict(),
                   'data_identity': identities, 'initial_model_sha256': configuration['initial_model_sha256']}
        # Published only after validation/history and both checkpoints commit.
        write_json(output / 'summary.json', summary)
        print(json.dumps({'epoch': epoch, 'miou': score, 'best_epoch': best_epoch, 'best_miou': best,
                          'status': status, 'seconds': record['seconds']}, ensure_ascii=False), flush=True)
        if stopped:
            break
    return output


if __name__ == '__main__':
    main()
