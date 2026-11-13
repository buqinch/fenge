#!/usr/bin/env python3
"""Native-103 final-only monitor for the explicit hard1500 validation protocol.

Run this independent monitor after launching the new experiment. It waits for
complete/early_stopped, snapshots validation-best, and shares physical GPU3's
existing evaluation lock. It never modifies a training process or old monitor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import periodic_evaluate as base
from evaluate_final_partition import PROTOCOL, validate_bundle, validate_checkpoint_protocol
from remote_evaluate import local_gpu_probe

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_TRAINING = {'complete', 'early_stopped'}


def load_config(path):
    c = base.read_json(path)
    if c.get('schema') != 'lusnar_final_partition_monitor_v1' or c.get('protocol') != PROTOCOL:
        raise ValueError('Expected the dedicated final partition protocol')
    for key in ('run', 'state_directory', 'hard_manifest', 'full_test_manifest', 'gpu_lock'):
        p = Path(c[key]); c[key] = str((ROOT / p if not p.is_absolute() else p).resolve())
    for key in ('run', 'state_directory'):
        if not Path(c[key]).is_relative_to(ROOT / 'result'):
            raise ValueError(key + ' must remain in this project/result')
    for key in ('batch_size', 'poll_seconds', 'minimum_free_mib'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError('Invalid ' + key)
    if type(c.get('gpu_index')) is not int or c['gpu_index'] < 0 or c.get('workers', -1) < 0:
        raise ValueError('Invalid GPU or worker count')
    return c


def verify_training_sources(configuration):
    sources = configuration.get('source_sha256', {})
    if not sources or not any('train' in Path(p).name for p in sources):
        raise ValueError('Missing training entry/source hashes')
    for relative, expected in sources.items():
        p = (ROOT / relative).resolve()
        if not p.is_relative_to(ROOT / 'code') or base.sha256(p) != expected:
            raise ValueError('Training source changed: ' + relative)


def completed_training(run, hard_sha):
    path = run / 'summary.json'
    if not path.exists(): return None
    summary = base.read_json(path)
    if summary.get('status') not in TERMINAL_TRAINING: return None
    configuration = base.read_json(run / 'run_config.json')
    val = summary.get('data_identity', {}).get('val', {})
    if (summary.get('format') != 'lusnar_vit_run_v1' or summary.get('partial') is not False
            or summary.get('train_images') != 6000
            or summary.get('data_identity', {}).get('train', {}).get('image_count') != 6000
            or summary.get('validation_images') != 1500 or val.get('image_count') != 1500
            or val.get('hard_manifest_sha256') != hard_sha
            or configuration.get('protocol') != PROTOCOL
            or summary.get('checkpoint_selection_uses_test') is not True
            or summary.get('official_test_used_for_selection') is not True):
        raise ValueError('Final monitor requires the explicitly declared hard1500 selection protocol')
    rows = {}
    completed = summary['completed_epochs']
    for line in (run / 'history.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'): break
        row = json.loads(line)
        if row['epoch'] <= completed:
            if row['epoch'] in rows: raise ValueError('Duplicate training-history epoch')
            rows[row['epoch']] = row
    if set(rows) != set(range(1, completed + 1)):
        raise ValueError('Final training summary/history coverage differs')
    score, epoch = -math.inf, None
    for current, row in sorted(rows.items()):
        value = row.get('validation_weighted_score')
        if value is None:
            if current % 2 == 0 or current == completed:
                raise ValueError('Every second epoch and the final epoch must have validation metrics')
            if row.get('validation_performed') is not False or row.get('best_epoch') != epoch:
                raise ValueError('Unvalidated history rows must explicitly record validation_performed=false')
            if epoch is not None:
                base.assert_close(row['best_miou'], score, 'Unvalidated epoch changed the previous best score')
            continue
        if row.get('validation_performed') is not True:
            raise ValueError('Validated history rows must explicitly record validation_performed=true')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Invalid validation score')
        if value > score: score, epoch = value, current
        if row.get('best_epoch') != epoch:
            raise ValueError('Best must be the first strict validation maximum')
        base.assert_close(row['best_miou'], score, 'History best score differs')
    if epoch is None or summary['best_epoch'] != epoch:
        raise ValueError('Final best epoch differs from validation history')
    if summary.get('best_metric') != 'validation_weighted_score':
        raise ValueError('The selection metric must be the scene-weighted validation score')
    base.assert_close(summary['best_miou'], score, 'Final best score differs from validation history')
    return summary, configuration


def code_binding():
    files = [Path(__file__), ROOT / 'code/evaluate_final_partition.py', ROOT / 'code/evaluate_hard_subset.py',
             ROOT / 'code/evaluate.py', ROOT / 'code/periodic_evaluate.py', ROOT / 'code/remote_evaluate.py',
             *sorted((ROOT / 'code/vitseg').glob('*.py'))]
    return {str(p.relative_to(ROOT)): base.sha256(p) for p in files}


def check_binding(spec):
    if spec['evaluation_source_sha256'] != code_binding():
        raise ValueError('Final evaluator/monitor dependencies changed after queueing')
    if (base.sha256(spec['config']['hard_manifest']) != spec['hard_manifest_sha256']
            or base.sha256(spec['config']['full_test_manifest']) != spec['full_test_manifest_sha256']):
        raise ValueError('Final partition manifests changed after queueing')
    verify_training_sources(spec['training_configuration'])


def status(job, name, **fields):
    previous = base.read_json(job / 'status.json') if (job / 'status.json').exists() else {}
    base.write_json(job / 'status.json', {**previous, 'status': name, 'updated_at': base.now(), **fields})


def prepare_job(config):
    run = Path(config['run']); job = run / 'final_partition_report'
    if (job / 'spec.json').exists():
        spec = base.read_json(job / 'spec.json')
        if spec['config'] != config:
            raise ValueError('Final job configuration changed after binding')
        check_binding(spec)
        return job
    hard_sha = base.sha256(config['hard_manifest'])
    committed = completed_training(run, hard_sha)
    if committed is None: return None
    summary, training = committed
    verify_training_sources(training)
    job.mkdir(parents=True, exist_ok=True)
    checkpoint = job / 'best.pt'
    base.snapshot(run / 'best.pt', checkpoint)
    saved = base.checkpoint_metadata(checkpoint)
    validate_checkpoint_protocol(saved, hard_sha)
    if (saved['epoch'] != summary['best_epoch'] or saved['train_config'] != training
            or saved['data_identity'] != summary['data_identity']
            or saved['model_config'] != summary['model_config']):
        raise ValueError('Final best checkpoint differs from the completed training summary')
    base.assert_close(saved['best_miou'], summary['best_miou'], 'Checkpoint best score differs')
    metrics = saved['metrics']
    if (metrics.get('image_count') != 1500 or metrics.get('data_identity') != summary['data_identity']['val']
            or metrics.get('checkpoint_selection_uses_test') is not True):
        raise ValueError('Final best does not carry the selected hard-validation metrics')
    base.assert_close(metrics['validation_weighted_score'], summary['best_miou'], 'Checkpoint validation score differs')
    spec = {'schema': 'lusnar_final_partition_job_v1', 'created_at': base.now(), 'config': config,
        'training_configuration': training, 'training_summary_sha256': base.sha256(run / 'summary.json'),
        'training_completed_epochs': summary['completed_epochs'], 'training_status': summary['status'],
        'checkpoint_epoch': saved['epoch'], 'checkpoint_sha256': base.sha256(checkpoint),
        'hard_manifest_sha256': hard_sha, 'full_test_manifest_sha256': base.sha256(config['full_test_manifest']),
        'evaluation_source_sha256': code_binding(), 'checkpoint_selection_uses_test': True,
        'independent_holdout': False, 'score_milestone_92_enabled': False}
    base.write_json(job / 'spec.json', spec)
    status(job, 'queued', checkpoint_epoch=saved['epoch'])
    return job


def worker(job):
    job = job.resolve()
    if not job.is_relative_to(ROOT / 'result') or job.name != 'final_partition_report':
        raise ValueError('Unexpected final job directory')
    lock = base.acquire_lock(job / '.worker.lock')
    if lock is None: return
    gpu_lock = None
    try:
        spec = base.read_json(job / 'spec.json'); config = spec['config']; check_binding(spec)
        checkpoint = job / 'best.pt'; report = job / 'metrics'
        if base.sha256(checkpoint) != spec['checkpoint_sha256']:
            raise ValueError('Final checkpoint snapshot changed')
        if (report / 'summary.json').exists():
            verified = validate_bundle(report, spec['checkpoint_sha256'], spec['hard_manifest_sha256'])
            status(job, 'complete', report=str(report), verified=verified); return
        gpu_lock = base.acquire_lock(config['gpu_lock'])
        if gpu_lock is None:
            status(job, 'queued', waiting_reason='Shared GPU evaluation lock is held'); return
        ready, reason, _ = local_gpu_probe(config)
        if not ready:
            status(job, 'queued', waiting_reason=reason); return
        command = [config['python'], '-B', '-u', str(ROOT / 'code/evaluate_final_partition.py'),
            '--checkpoint', str(checkpoint), '--hard-manifest', config['hard_manifest'],
            '--full-test-manifest', config['full_test_manifest'], '--output', str(report),
            '--device', 'cuda:0', '--amp', config['amp'], '--batch-size', str(config['batch_size']),
            '--workers', str(config['workers'])]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(config['gpu_index'])}
        with (job / 'test.log').open('a') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(), gpu_lock.fileno()))
            status(job, 'running', worker_identity=base.process_identity(os.getpid()),
                   evaluator_identity=base.process_identity(child.pid), command=command)
            code = child.wait()
        if code != 0: raise RuntimeError('Final partition evaluator exited with code ' + str(code))
        verified = validate_bundle(report, spec['checkpoint_sha256'], spec['hard_manifest_sha256'])
        status(job, 'complete', report=str(report), verified=verified)
    except Exception as error:
        status(job, 'failed', error=repr(error))
        raise
    finally:
        if gpu_lock: gpu_lock.close()
        lock.close()


def tick(config):
    job = prepare_job(config)
    if job is None:
        return {'status': 'waiting_for_training_completion', 'training_started': (Path(config['run']) / 'summary.json').exists()}
    state = base.read_json(job / 'status.json')
    if state['status'] in ('complete', 'failed'): return state
    if any(base.alive(state.get(key)) for key in ('worker_identity', 'evaluator_identity')):
        return state
    launch = job / 'worker_launch.json'
    if launch.exists() and base.alive(base.read_json(launch).get('identity')):
        return {**state, 'status': 'launching'}
    with (job / 'worker.log').open('a') as output:
        child = subprocess.Popen([config['python'], '-B', '-u', str(Path(__file__)), '--worker', str(job)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    base.write_json(launch, {'identity': base.process_identity(child.pid), 'created_at': base.now()})
    return {**state, 'status': 'launching'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker); return
    if args.config is None: parser.error('--config is required')
    config = load_config(args.config); directory = Path(config['state_directory']); directory.mkdir(parents=True, exist_ok=True)
    lock = base.acquire_lock(directory / '.monitor.lock')
    if lock is None: raise RuntimeError('This final monitor is already running')
    try:
        while True:
            try:
                result = tick(config)
            except Exception as error:
                result = {'status': 'error', 'error': repr(error)}
            base.write_json(directory / 'state.json', {'updated_at': base.now(), 'config': config, **result})
            if args.once:
                print(json.dumps(result), flush=True); return
            time.sleep(config['poll_seconds'])
    finally:
        lock.close()


if __name__ == '__main__':
    main()
