#!/usr/bin/env python3
"""Independent CPU/Gloo checks for the full-model distributed training path.

The loss oracle is the existing single-batch Mask2Former criterion with fixed
original-grid sample coordinates. It does not reproduce the distributed loss
reduction. No pretrained model or active experiment is modified by this check.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vitseg.losses import Mask2FormerCriterion, _sample_prediction
from vitseg.masking import masked_semantic_nll, prepare_masked_support
from vitseg.metrics import ConfusionMatrix


class FixedPointCriterion(Mask2FormerCriterion):
    """Eliminate point RNG, retaining independent Hungarian and set losses."""

    def _shared_points(self, target, labels):
        flat = target.flatten()
        indices = (flat != self.ignore_index).nonzero(as_tuple=False).flatten()
        return indices, (flat[indices][None] == labels[:, None]).float()

    def _mask_points(self, mask, target, label):
        flat = target.flatten()
        indices = (flat != self.ignore_index).nonzero(as_tuple=False).flatten()
        return _sample_prediction(mask[None], indices, *target.shape)[0], (flat[indices] == label).float()


class ToyQueries(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(731)
        self.classes = nn.Parameter(torch.randn(5, 6, generator=generator) * .2)
        self.masks = nn.Parameter(torch.randn(5, 6, 6, generator=generator) * .3)
        self.unused = nn.Parameter(torch.ones(3))

    def forward(self, identifiers):
        offsets = identifiers.float()
        classes = self.classes[None] + offsets[:, None, None] * torch.linspace(-.1, .1, 6)[None, None]
        masks = self.masks[None] + offsets[:, None, None, None] * torch.linspace(-.05, .05, 6)[None, None, :, None]
        return {'pred_logits': classes, 'pred_masks': masks, 'aux_outputs': [
            {'pred_logits': classes * .8, 'pred_masks': masks * .9},
            {'pred_logits': classes * 1.1, 'pred_masks': masks * .7}]}


def varied_targets(world):
    counts = ([1, 2, 3, 4, 5, 1] + [0, 0])[:world * 2]
    counts[-2:] = [0, 0]  # One rank has no valid semantic targets at all.
    targets = []
    for count in counts:
        targets.append(torch.arange(36).reshape(6, 6).remainder(count) if count else
                       torch.full((6, 6), 255, dtype=torch.long))
    return torch.stack(targets)


class ToyValidation(Dataset):
    def __init__(self, size):
        self.size, self.partial = size, False
        self.identity = {'split': 'val', 'image_count': size, 'partial': False}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        target = (torch.arange(64).reshape(8, 8) + index) % 5
        prediction = target.clone()
        prediction[0, 0] = (prediction[0, 0] + 1) % 5
        image = torch.stack([prediction.float(), torch.zeros(8, 8), torch.zeros(8, 8)])
        return {'image': image, 'mask': target, 'scene': f'Moon_{1 + index % 2 * 2}',
                'sample_id': str(index), 'source_id': f'validation/{index}'}


class ToyPrediction(nn.Module):
    def forward(self, images):
        logits = torch.nn.functional.one_hot(images[:, 0].long(), 5).permute(0, 3, 1, 2).float() * 8
        return {'semantic_logits': logits}


def assert_close(actual, expected, label, atol=2e-6, rtol=2e-6):
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f'{label}: {error}') from error


def worker(rank, world, rendezvous, output):
    from distributed_training import (NoPaddingShardSampler, distributed_set_loss,
                                      synchronize_gradients, reduce_sum, gather_objects,
                                      distributed_evaluate, distributed_masked_loss)
    from train import rng_state, restore_rng
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    checks = []
    try:
        # A full batch with deliberately unequal class counts and an empty rank.
        targets = varied_targets(world)
        ids = torch.arange(len(targets))
        sl = slice(rank * 2, rank * 2 + 2)
        actual_model, oracle_model = ToyQueries(), ToyQueries()
        local = distributed_set_loss(actual_model(ids[sl]), targets[sl], FixedPointCriterion(num_points=36))
        local['loss_total'].backward()
        synchronize_gradients(actual_model, bucket_bytes=64)
        expected = FixedPointCriterion(num_points=36)(oracle_model(ids), targets)
        expected['loss_total'].backward()
        reduced = {key: reduce_sum(value.detach().clone()) / world for key, value in local.items()}
        for key in expected:
            assert_close(reduced[key], expected[key], 'global set loss ' + key)
        for (name, actual), (_, reference) in zip(actual_model.named_parameters(), oracle_model.named_parameters()):
            if name == 'unused':
                assert actual.grad is reference.grad is None
            else:
                assert_close(actual.grad, reference.grad, 'global set gradient ' + name)
        checks.append({'name': 'hungarian_class_weight_and_mask_count_global_reductions',
                       'passed': True, 'loss': float(expected['loss_total']), 'auxiliary_stages': 2,
                       'gt_class_counts': [len(x[x != 255].unique()) for x in targets],
                       'empty_rank_included': True, 'globally_unused_gradient_is_none': True})

        # Parameters used by only one rank must still get the correct global
        # gradient. A globally unused parameter must remain outside optimizer use.
        sparse = nn.ParameterList([nn.Parameter(torch.ones(4)), nn.Parameter(torch.ones(2)),
                                   nn.Parameter(torch.ones(1))])
        sparse[0].grad = torch.full_like(sparse[0], rank + 1.)
        if rank == 0:
            sparse[1].grad = torch.full_like(sparse[1], 7.)
        synchronize_gradients(sparse, bucket_bytes=8)
        assert_close(sparse[0].grad, torch.full_like(sparse[0], (world + 1.) / 2), 'all-rank gradient')
        assert_close(sparse[1].grad, torch.full_like(sparse[1], 7. / world), 'one-rank gradient')
        assert sparse[2].grad is None
        checks.append({'name': 'bucketed_gradient_reduction_with_locally_unused_parameters', 'passed': True})

        # Masked semantic support differs by rank; hidden complete components
        # and ignored targets must not turn this into an average of rank means.
        hidden = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(len(targets)):
            hidden[i, :i % 5 + 1, :i % 4 + 1] = True
        hidden[0] = True
        targets[0, :2, :2] = 1  # Fully hidden rock, auxiliary-only ignore.
        generator = torch.Generator().manual_seed(921)
        parameter = nn.Parameter(torch.randn(5, 6, 6, generator=generator) * .2)
        oracle = nn.Parameter(parameter.detach().clone())
        local_logits = parameter[None].expand(2, -1, -1, -1).log_softmax(1)
        local_loss, stats = masked_semantic_nll(local_logits, targets[sl], hidden[sl], .2)
        denominator = reduce_sum(torch.tensor(stats['denominator_pixels'], dtype=torch.float64))
        weighted = distributed_masked_loss(local_loss, stats['denominator_pixels'], float(denominator))
        weighted.backward()
        synchronize_gradients(nn.ParameterList([parameter]), bucket_bytes=64)
        expected_loss, expected_stats = masked_semantic_nll(
            oracle[None].expand(len(targets), -1, -1, -1).log_softmax(1), targets, hidden, .2)
        expected_loss.backward()
        assert_close(reduce_sum(weighted.detach()) / world, expected_loss, 'global masked NLL')
        assert_close(parameter.grad, oracle.grad, 'global masked gradient')
        assert int(denominator) == expected_stats['denominator_pixels']
        checks.append({'name': 'masked_nll_unequal_visible_and_eligible_hidden_support', 'passed': True,
                       'per_rank_denominator': gather_objects(stats['denominator_pixels']),
                       'global_denominator': int(denominator),
                       'fully_hidden_components_ignored': expected_stats['ignored_component_counts']})

        # Exact 1601-image coverage and an empty validation shard both exercise
        # the actual distributed evaluator, using analytically known predictions.
        for count in (1601, 3):
            dataset = ToyValidation(count)
            sampler = NoPaddingShardSampler(dataset, rank=rank, world_size=world, shuffle=False)
            shards = gather_objects(list(sampler))
            combined = [value for shard in shards for value in shard]
            assert sorted(combined) == list(range(count)) and len(set(combined)) == count
            loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0)
            metrics = distributed_evaluate(ToyPrediction(), loader, torch.device('cpu'), amp='none',
                                           tta='none', progress_every=0)
            oracle_counter = ConfusionMatrix()
            for i in range(count):
                item = dataset[i]
                oracle_counter.update(item['mask'], item['image'][0].long())
            assert metrics['image_count'] == count and not metrics['partial']
            assert metrics['confusion_matrix_gt_rows_pred_cols'] == oracle_counter.matrix.tolist()
            assert_close(torch.tensor(metrics['miou']), torch.tensor(oracle_counter.compute()['miou']), 'global validation mIoU')
            checks.append({'name': f'validation_{count}_images_without_padding_or_duplicates', 'passed': True,
                           'shard_sizes': [len(x) for x in shards], 'matrix_pixels': int(oracle_counter.matrix.sum())})

        # Sampling must cover each training sample once and replay each epoch.
        dataset = list(range(8000))
        sampler = NoPaddingShardSampler(dataset, rank=rank, world_size=world, shuffle=True, seed=42,
                                        require_equal=True)
        sampler.set_epoch(7)
        first = list(sampler)
        sampler.set_epoch(8)
        second = list(sampler)
        sampler.set_epoch(7)
        assert list(sampler) == first and first != second
        assert sorted(x for shard in gather_objects(first) for x in shard) == list(range(8000))
        checks.append({'name': 'training_sampler_exact_coverage_and_epoch_replay', 'passed': True,
                       'samples_per_rank': len(first)})

        # Rank-local state serializes through weights_only loading and replays
        # the independent Python/NumPy/Torch and DataLoader-generator streams.
        random.seed(42 + rank); np.random.seed(42 + rank); torch.manual_seed(42 + rank)
        loader_generator = torch.Generator().manual_seed(1900 + rank)
        state = {'rank': rank, 'rng': rng_state(torch.device('cpu')),
                 'loader_generator': loader_generator.get_state()}
        states = gather_objects(state)
        assert [x['rank'] for x in states] == list(range(world))
        path = Path(output).with_name(f'.distributed_rank_state_{rank}.pt')
        torch.save(states, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)[rank]
        path.unlink()
        expected_random = (random.random(), float(np.random.rand()), torch.rand(4),
                           torch.rand(4, generator=loader_generator))
        restore_rng(restored['rng'], torch.device('cpu'))
        loader_generator.set_state(restored['loader_generator'])
        actual_random = (random.random(), float(np.random.rand()), torch.rand(4),
                         torch.rand(4, generator=loader_generator))
        assert expected_random[:2] == actual_random[:2]
        for expected_value, actual_value in zip(expected_random[2:], actual_random[2:]):
            assert torch.equal(expected_value, actual_value)
        checks.append({'name': 'rank_rng_and_loader_state_weights_only_roundtrip', 'passed': True,
                       'rank_streams_distinct': len({bytes(x['rng']['torch'].tolist()) for x in states}) == world})

        records = gather_objects({'rank': rank, 'checks': checks})
        if rank == 0:
            result = {'schema': 'fenge_vit_distributed_cpu_verification_v1', 'status': 'passed',
                      'world_size': world, 'backend': 'gloo', 'rank_results': records,
                      'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                        for name in ('distributed_training.py', 'verify_distributed.py')},
                      'limitations': ['Small deterministic mathematical fixtures; real1024 NCCL training is verified separately.',
                                      'Fixed point coordinates isolate distributed reductions from rank-dependent sampling RNG.']}
            Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'status': 'passed', 'world_size': world, 'checks_per_rank': len(checks),
                              'output': output}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'result/verification_distributed_cpu.json')
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--runtime-train', action='store_true',
                        help='Run the real trainer with rank agreement checks; use torchrun')
    parser.add_argument('--stop-after-epoch', type=int, default=0,
                        help='Runtime check only: cleanly stop after a fully committed epoch')
    parser.add_argument('--train-args', nargs=argparse.REMAINDER,
                        help='Arguments passed unchanged to the real distributed trainer')
    args = parser.parse_args()
    if args.runtime_train:
        return runtime_train(args.train_args or [], args.stop_after_epoch)
    if args.world_size != 4:
        parser.error('The uneven-target oracle intentionally uses exactly four ranks')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fenge-vit-gloo-') as temporary:
        mp.spawn(worker, args=(args.world_size, str(Path(temporary) / 'rendezvous'), str(args.output)),
                 nprocs=args.world_size, join=True)


def runtime_train(arguments, stop_after_epoch):
    """Verify real training without modifying the trainer or its source digest.

    The only interruption occurs after all ranks have acknowledged a committed
    epoch. Each actual AdamW step additionally checks parameter equality across
    ranks; model weights and gradients are not changed by these observations.
    """
    import train as base
    import train_distributed as trainer
    from distributed_training import gather_objects, world_info, broadcast_object

    configuration = base.parse_args(arguments)
    output = Path(configuration.output)
    observed_steps = []
    original_step = torch.optim.AdamW.step
    original_action = trainer._rank_zero_action

    def checked_step(optimizer, *args, **kwargs):
        gradients = {}
        for i, group in enumerate(optimizer.param_groups):
            gradients[group.get('name', str(i))] = {
                'parameters_with_gradients': sum(p.grad is not None for p in group['params']),
                'nonzero_gradient_parameters': sum(p.grad is not None and bool(p.grad.detach().abs().sum())
                                                    for p in group['params'])}
        result = original_step(optimizer, *args, **kwargs)
        state = {f'group{group_index}/parameter{index}': parameter for group_index, group in
                 enumerate(optimizer.param_groups) for index, parameter in enumerate(group['params'])}
        fingerprint = base.tensor_state_sha256(state)
        fingerprints = gather_objects(fingerprint)
        if len(set(fingerprints)) != 1:
            raise AssertionError('Actual distributed parameters diverged after AdamW update')
        observed_steps.append({'parameter_sha256': fingerprint,
                               'all_rank_parameters_identical': True, 'gradients': gradients})
        return result

    def checked_action(function):
        result = original_action(function)
        if function.__name__ != 'commit_epoch':
            return result
        rank, world = world_info()
        summary = json.loads((output / 'summary.json').read_text()) if rank == 0 else None
        summary = broadcast_object(summary)
        rows = gather_objects({'rank': rank, 'optimizer_steps_observed': list(observed_steps)})
        if rank == 0:
            base.write_json(output / f'verification_rank_state_epoch_{summary["completed_epochs"]:03d}.json',
                            {'world_size': world, 'epoch': summary['completed_epochs'],
                             'partial': summary['partial'], 'ranks': rows,
                             'checkpoint_committed': True, 'verification_only': True})
        if stop_after_epoch and summary['completed_epochs'] >= stop_after_epoch:
            if rank == 0:
                print(json.dumps({'verification_stop_at_completed_epoch': summary['completed_epochs'],
                                  'all_ranks_stop_together': True}), flush=True)
            raise SystemExit(0)
        return result

    torch.optim.AdamW.step = checked_step
    trainer._rank_zero_action = checked_action
    try:
        return trainer.main(arguments)
    finally:
        torch.optim.AdamW.step = original_step
        trainer._rank_zero_action = original_action


if __name__ == '__main__':
    main()
