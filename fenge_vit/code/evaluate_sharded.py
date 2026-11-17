#!/usr/bin/env python3
"""Evaluate disjoint manifest shards with a frozen runtime, then add integer CMs."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import periodic_evaluate as periodic
import gpu_slots


def load_runtime(runtime):
    runtime = Path(runtime)
    sys.path.insert(0, str(runtime / 'code'))
    spec = importlib.util.spec_from_file_location('frozen_sharded_evaluate', runtime / 'code/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_ids(records):
    return [r['source_id'] for r in records]


def validate_partition(parts, expected):
    observed = [s for part in parts for s in part['shard_source_ids']]
    if len(set(expected)) != len(expected) or Counter(observed) != Counter(expected):
        raise ValueError('Shards must cover every source exactly once')
    for rank, part in enumerate(parts):
        if (part['shard_rank'] != rank or part['shard_world_size'] != len(parts)
                or part['shard_source_ids'] != expected[rank::len(parts)]
                or part['image_count'] != len(part['shard_source_ids'])
                or part['validation_unique_source_count'] != part['image_count']):
            raise ValueError('Shard order, count, rank, or unique coverage differs')


def merge_reports(parts, dataset, trigger):
    import numpy as np
    from vitseg.metrics import ConfusionMatrix
    expected = source_ids(dataset.records)
    validate_partition(parts, expected)
    for part in parts:
        if (part['checkpoint_sha256'] != trigger['checkpoint_sha256']
                or part['checkpoint_epoch'] != trigger['checkpoint_epoch']
                or part['tta'] != trigger['evaluation']['tta']
                or part['full_data_identity'] != dataset.identity):
            raise ValueError('Shard checkpoint, TTA, or full dataset differs')
        counts = Counter(s.split('/')[0] for s in part['shard_source_ids'])
        if {s: r['image_count'] for s, r in part['scenes'].items()} != counts:
            raise ValueError('Shard scene counts differ from its source IDs')
        matrices = []
        for scene, row in part['scenes'].items():
            matrix = np.asarray(row['confusion_matrix_gt_rows_pred_cols'])
            if (matrix.shape != (5, 5) or matrix.dtype.kind not in 'iu' or np.any(matrix < 0)
                    or matrix.sum() != counts[scene] * 1024 * 1024):
                raise ValueError('Shard scene confusion matrix does not cover all native pixels')
            matrices.append(matrix)
        if not np.array_equal(np.sum(matrices, axis=0), part['confusion_matrix_gt_rows_pred_cols']):
            raise ValueError('Shard pooled matrix differs from scene matrices')
    def summed(rows):
        cm = ConfusionMatrix()
        for row in rows:
            matrix = np.asarray(row['confusion_matrix_gt_rows_pred_cols'])
            if matrix.shape != (5, 5) or matrix.dtype.kind not in 'iu' or np.any(matrix < 0):
                raise ValueError('Invalid integer confusion matrix')
            cm.matrix += matrix
        return cm.compute()
    result = {k: v for k, v in parts[0].items()
              if not k.startswith('shard_') and k not in ('full_data_identity', 'object_diagnostics')}
    result.update(summed(parts))
    result['scenes'] = {}
    for scene in sorted({r['scene'] for r in dataset.records}):
        rows = [part['scenes'][scene] for part in parts if scene in part['scenes']]
        result['scenes'][scene] = {**summed(rows), 'image_count': sum(r['image_count'] for r in rows)}
    result.update(image_count=len(expected), validation_unique_source_count=len(expected),
                  partial=False, data_identity=dataset.identity, test_scope='complete_official_test',
                  official_test_scope=True, loader_passes=1,
                  model_forward_calls=sum(p['model_forward_calls'] for p in parts),
                  weighted_score=(result['scenes']['Moon_2']['miou'] +
                      result['scenes']['Moon_4']['miou'] + 2 * result['scenes']['Moon_9']['miou']) / 4,
                  distributed_evaluation={'world_size': len(parts), 'partition': 'manifest indices rank::world_size',
                      'unique_coverage_verified': True, 'aggregation': 'sum integer scene confusion matrices before IoU',
                      'per_rank_image_count': [p['image_count'] for p in parts]},
                  object_diagnostics_shards=[p.get('object_diagnostics') for p in parts])
    periodic.validate_metrics(result, trigger)
    return result


def worker(args):
    import torch
    from torch.utils.data import DataLoader
    evaluator = load_runtime(args.runtime)
    trigger = periodic.read_json(args.job / 'trigger.json')
    protocol = trigger['evaluation']
    if protocol['tta'] != 'hflip' or protocol['batch_size'] != 1:
        raise ValueError('This sharded job requires batch-one native horizontal-flip evaluation')
    checkpoint = args.job / 'best.pt'
    if periodic.sha256(checkpoint) != trigger['checkpoint_sha256']:
        raise ValueError('Transferred checkpoint SHA mismatch')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model, saved, model_config = evaluator.load_checkpoint(checkpoint, device)
    if saved['epoch'] != trigger['checkpoint_epoch'] or saved.get('partial') or model_config.input_size != 1024:
        raise ValueError('Checkpoint epoch, completeness or resolution differs')
    dataset = evaluator.LunarDataset(protocol['test_manifest'], 'test', 1024, horizontal_flip=False)
    full_identity = dict(dataset.identity)
    dataset.records = dataset.records[args.rank::args.world_size]
    ids = source_ids(dataset.records)
    dataset.partial = True
    dataset.identity = {**full_identity, 'partial': True, 'image_count': len(ids),
                        'unique_source_count': len(ids), 'ordered_samples_sha256': periodic.fingerprint(dataset.records)}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=protocol['workers'], pin_memory=True)
    observer = evaluator.TerrainDiagnostics(16384, 256, 5)
    start = time.monotonic()
    metrics = evaluator.evaluate_loader(model, loader, device, amp=protocol['amp'], tta='hflip',
                                        diagnostic_observer=observer, progress_every=25)
    metrics.update(checkpoint=str(checkpoint), checkpoint_sha256=trigger['checkpoint_sha256'],
        checkpoint_epoch=saved['epoch'], checkpoint_partial_training=False, model_config=saved['model_config'],
        split='test', tta='hflip', tta_sizes=None, tta_weights=None,
        shard_rank=args.rank, shard_world_size=args.world_size, shard_source_ids=ids,
        full_data_identity=full_identity, seconds=time.monotonic()-start,
        inference_batch_size=1, amp=protocol['amp'])
    periodic.write_json(args.job / f'shard_{args.rank}_metrics.json', metrics)
    print(json.dumps({'status': 'complete', 'rank': args.rank, 'image_count': metrics['image_count']}), flush=True)


def coordinator(args):
    guard = periodic.acquire_lock(args.job / '.sharded.lock')
    if guard is None:
        raise RuntimeError('A coordinator already owns this job')
    children, locks, logs = [], [], []
    start = time.monotonic()
    state_path = args.job / 'sharded_status.json'
    trigger = periodic.read_json(args.job / 'trigger.json')
    def state(status, **kwargs):
        periodic.write_json(state_path, {'status': status, 'updated_at': periodic.now(),
            'worker_identity': periodic.process_identity(os.getpid()), 'gpus': args.gpus,
            'checkpoint_epoch': trigger['checkpoint_epoch'], **kwargs})
    try:
        if len(set(args.gpus)) != len(args.gpus) or len(args.gpus) < 2:
            raise ValueError('At least two distinct physical GPUs are required')
        training = periodic.read_json(args.job / 'training_configuration.json')
        for name, digest in training['source_sha256'].items():
            if periodic.sha256(args.runtime / name) != digest:
                raise ValueError('Frozen training source changed: ' + name)
        if periodic.sha256(args.runtime / 'code/evaluate.py') != trigger['evaluation']['evaluate_sha256']:
            raise ValueError('Frozen evaluator changed')
        evaluator = load_runtime(args.runtime)
        dataset = evaluator.LunarDataset(trigger['evaluation']['test_manifest'], 'test', 1024)
        if dataset.identity['manifest_sha256'] != trigger['evaluation']['test_manifest_sha256']:
            raise ValueError('Manifest SHA mismatch')
        uuids = []
        if args.gpu_lock_fds and len(args.gpu_lock_fds) != len(args.gpus):
            raise ValueError('Each GPU must have one inherited lock descriptor')
        for rank, gpu in enumerate(args.gpus):
            config = {'gpu_index': gpu, 'minimum_free_mib': 12000, 'max_concurrent_evaluations': 1,
                      'gpu_lock': str(args.lock_root / f'.periodic_gpu_{gpu}.lock')}
            if args.gpu_lock_fds:
                fd = args.gpu_lock_fds[rank]
                stat, expected = os.fstat(fd), Path(config['gpu_lock']).stat()
                if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ValueError('Inherited lock does not match the requested physical GPU')
                lock = os.fdopen(os.dup(fd), 'a+')
                uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                                 '--format=csv,noheader'], text=True).strip()
            else:
                lock = gpu_slots.acquire_gpu_slot(config)
                if lock is None: raise RuntimeError(f'GPU {gpu} evaluation slot is busy')
                ready, reason, uuid = gpu_slots.available(config)
                if not ready:
                    lock.close()
                    raise RuntimeError(f'GPU {gpu}: {reason}')
            locks.append(lock)
            uuids.append(uuid)
        for rank, uuid in enumerate(uuids):
            output = (args.job / f'shard_{rank}.log').open('a'); logs.append(output)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                '--runtime', str(args.runtime), '--job', str(args.job), '--rank', str(rank),
                '--world-size', str(len(args.gpus))]
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid,
                    'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'},
                pass_fds=tuple(lock.fileno() for lock in locks) + (guard.fileno(),))
            children.append(child)
        state('running', evaluators=[periodic.process_identity(c.pid) for c in children], gpu_uuids=uuids)
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError('A shard evaluator failed; see shard logs')
            time.sleep(5)
        if any(c.returncode != 0 for c in children):
            raise RuntimeError('A shard evaluator failed; see shard logs')
        parts = [periodic.read_json(args.job / f'shard_{r}_metrics.json') for r in range(len(children))]
        result = merge_reports(parts, dataset, trigger)
        result['seconds'] = time.monotonic() - start
        result['distributed_evaluation']['physical_gpus'] = args.gpus
        periodic.write_json(args.job / 'test_metrics.json', result)
        state('complete', verified=periodic.validate_metrics(result, trigger),
              metrics_sha256=periodic.sha256(args.job / 'test_metrics.json'))
    except Exception as error:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        state('failed', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        for log in logs: log.close()
        for lock in locks: lock.close()
        guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--lock-root', type=Path, default=Path(__file__).resolve().parents[1] / 'result')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--gpu-lock-fds', nargs='+', type=int)
    args = parser.parse_args()
    worker(args) if args.worker else coordinator(args)


if __name__ == '__main__':
    main()
