#!/usr/bin/env python3
"""Observe validation-best checkpoints and evaluate a fixed hard subset remotely.

This monitor never changes training or validation selection. It does not backfill
past boundaries, select new hard samples, or create full-test score milestones.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import periodic_evaluate as base

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {'Moon_2': 188, 'Moon_4': 187, 'Moon_9': 1125}
WEIGHTS = {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}
TERMINAL = {'complete', 'failed', 'missed'}


def manifest_identity(path):
    path = Path(path)
    d = base.read_json(path)
    if d.get('schema') != 'lusnar_fixed_hard_test_v1' or len(d.get('records', [])) != 1500:
        raise ValueError('Expected the frozen 1500-image hard-test manifest')
    if dict(Counter(r['scene'] for r in d['records'])) != QUOTAS:
        raise ValueError('Hard-test quotas differ from 188/187/1125')
    ids = [(r['scene'], str(r['sample_id'])) for r in d['records']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate hard-test identities')
    return {'path': str(path.resolve()), 'sha256': base.sha256(path), 'image_count': 1500,
            'scene_counts': QUOTAS, 'score_weights': WEIGHTS}


def validate_metrics(d, trigger, binding):
    if (d.get('partial') is not False or d.get('image_count') != 1500
            or d.get('official_full_test') is not False or d.get('test_scope') != 'fixed_hard_subset'
            or d.get('checkpoint_epoch') != trigger['checkpoint_epoch']
            or d.get('checkpoint_sha256') != trigger['checkpoint_sha256']
            or d.get('subset_manifest_sha256') != binding['manifest']['sha256']
            or 'weighted_score' in d or d.get('hard_subset_score_weights') != WEIGHTS):
        raise ValueError('Hard-subset metrics scope/checkpoint/manifest differs')
    scene_miou = {}
    for scene, count in QUOTAS.items():
        s = d['scenes'][scene]; matrix = s['confusion_matrix_gt_rows_pred_cols']
        if s.get('image_count') != count or len(matrix) != 5 or any(len(row) != 5 for row in matrix):
            raise ValueError('Hard-subset scene coverage or matrix shape differs')
        if any(type(x) is not int or x < 0 for row in matrix for x in row):
            raise ValueError('Confusion matrix must contain nonnegative integers')
        if sum(map(sum, matrix)) != count * 1024 * 1024:
            raise ValueError('Hard-subset pixel coverage differs')
        present = [i for i in range(5) if sum(matrix[i]) > 0]
        scores = []
        for i in present:
            tp = matrix[i][i]; union = sum(matrix[i]) + sum(row[i] for row in matrix) - tp
            iou = tp / union
            base.assert_close(s['class_iou'][base.CLASSES[i]], iou, 'Hard class IoU mismatch')
            scores.append(iou)
        value = sum(scores) / len(scores)
        base.assert_close(s['miou'], value, 'Hard scene mIoU mismatch')
        scene_miou[scene] = value
    score = sum(WEIGHTS[s] * scene_miou[s] for s in WEIGHTS) / sum(WEIGHTS.values())
    base.assert_close(d['hard_subset_score'], score, 'Hard-subset aggregate mismatch')
    return {'hard_subset_score': score, 'scene_miou': scene_miou, 'image_count': 1500,
            'official_full_test': False, 'checkpoint_selection_uses_test': False}


def set_status(job, status, **fields):
    previous = base.read_json(job / 'status.json') if (job / 'status.json').exists() else {}
    base.write_json(job / 'status.json', {**previous, 'status': status, 'updated_at': base.now(), **fields})


def load_config(path):
    c = base.read_json(path)
    if c.get('schema') != 'vit_hard_subset_monitor_v1' or c.get('every_epochs') != 2:
        raise ValueError('Expected fixed hard-subset every-two-epochs configuration')
    for name in ('state_directory', 'hard_manifest'):
        p = Path(c[name]); c[name] = str((ROOT / p if not p.is_absolute() else p).resolve())
    c['runs'] = [str((ROOT / p).resolve()) for p in c['runs']]
    if any(not Path(p).is_relative_to(ROOT / 'result') for p in c['runs']):
        raise ValueError('Run paths must remain in project/result')
    if not Path(c['state_directory']).is_relative_to(ROOT / 'result'):
        raise ValueError('Monitor state must remain in project/result')
    c.setdefault('poll_seconds', 15)
    if type(c['poll_seconds']) is not int or c['poll_seconds'] < 1:
        raise ValueError('Invalid poll interval')
    return c


def prepare_snapshot(run, epoch, summary, configuration, rows, config):
    job = run / 'hard_periodic_test' / f'epoch_{epoch:03d}'
    if (job / 'status.json').exists(): return
    row = rows[epoch]
    base.validate_validation(base.read_json(run / f'validation_epoch_{epoch:03d}.json'), epoch,
                             row['validation_miou'], summary['data_identity']['val'])
    base.check_sources(configuration)
    trigger = {'schema': 'vit_hard_subset_trigger_v1', 'run': str(run), 'trigger_epoch': epoch,
               'checkpoint_epoch': row['best_epoch'], 'validation_best_miou': row['best_miou'],
               'created_at': base.now(), 'every_epochs': 2, 'checkpoint_selection_uses_test': False,
               'configuration_sha256': base.fingerprint(configuration),
               'validation_identity': summary['data_identity']['val'], 'hard_manifest': config['hard_manifest']}
    job.mkdir(parents=True, exist_ok=True)
    for candidate in [run / 'best.pt', run / 'last.pt']:
        if not candidate.exists(): continue
        base.snapshot(candidate, job / 'best.pt')
        saved = base.checkpoint_metadata(job / 'best.pt')
        if saved.get('epoch') != trigger['checkpoint_epoch']:
            (job / 'best.pt').unlink(); continue
        base.validate_snapshot(saved, trigger, configuration, summary)
        trigger['checkpoint_sha256'] = base.sha256(job / 'best.pt')
        trigger['training_configuration'] = configuration
        base.write_json(job / 'trigger.json', trigger)
        set_status(job, 'waiting_manifest' if not Path(config['hard_manifest']).exists() else 'queued',
                   checkpoint_epoch=trigger['checkpoint_epoch'])
        return
    if summary['completed_epochs'] > epoch:
        base.write_json(job / 'trigger.json', trigger)
        set_status(job, 'missed', reason='The exact historical validation-best snapshot is unavailable')


def binding_for(job, config):
    path = job / 'evaluation_binding.json'
    if path.exists():
        binding = base.read_json(path)
        if manifest_identity(config['hard_manifest']) != binding['manifest']:
            raise ValueError('Frozen hard manifest changed after binding')
        return binding
    identity = manifest_identity(config['hard_manifest'])
    binding = {'manifest': identity, 'evaluator_sha256': base.sha256(ROOT / 'code/evaluate_hard_subset.py'),
               'monitor_sha256': base.sha256(Path(__file__)), 'tta': '1024/768 equal hflip',
               'amp': 'bf16', 'batch_size': 4, 'workers': 4, 'official_full_test': False,
               'full_test_92_milestone_enabled': False}
    base.write_json(path, binding)
    return binding


def remote_path(config, job):
    return Path(config['remote']['project']) / 'result/hard_remote' / job.parent.parent.name / job.name


def ssh_call(config, args, payload=None):
    r = config['remote']; project = Path(r['project'])
    cmd = ['bash', str(project / r.get('wrapper', 'code/run_on_103.sh')), r['python'], '-B',
           str(project / 'code/hard_subset_monitor.py'), *map(str, args)]
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', r['host'], shlex.join(cmd)],
        input=json.dumps(payload) if payload is not None else None, capture_output=True, text=True, timeout=60, check=True)
    d = json.loads(result.stdout)
    if d.get('status') == 'error': raise ValueError(d['error'])
    return d


def remote_dir(value):
    p = Path(value).resolve(); root = ROOT / 'result/hard_remote'
    if not p.is_relative_to(root) or len(p.relative_to(root).parts) != 2:
        raise ValueError('Remote hard jobs must stay inside project/result/hard_remote/run/job')
    return p


def read_spec(job):
    s = base.read_json(job / 'spec.json')
    if (s['project'] != str(ROOT) or s['binding']['monitor_sha256'] != base.sha256(__file__)
            or s['binding']['evaluator_sha256'] != base.sha256(ROOT / 'code/evaluate_hard_subset.py')
            or manifest_identity(s['binding']['manifest']['path']) != s['binding']['manifest']):
        raise ValueError('Remote project/source/manifest changed')
    base.check_sources(s['trigger']['training_configuration'])
    return s


def remote_status(job):
    d = base.read_json(job / 'status.json') if (job / 'status.json').exists() else {'status': 'prepared'}
    return {**d, 'checkpoint_present': (job / 'best.pt').exists(),
            'metrics_present': (job / 'hard_test_metrics.json').exists()}


def prepare_remote(job, spec):
    job = remote_dir(job); job.mkdir(parents=True, exist_ok=True)
    path = job / 'spec.json'
    if path.exists() and base.read_json(path) != spec: raise ValueError('Remote hard-job spec is immutable')
    base.write_json(path, spec); read_spec(job)
    return remote_status(job)


def ensure_remote(job):
    job = remote_dir(job); spec = read_spec(job)
    lock = base.acquire_lock(job / '.launch.lock')
    if lock is None: return remote_status(job)
    try:
        state = remote_status(job)
        if state['status'] in TERMINAL: return state
        if any(base.alive(state.get(k)) for k in ('worker_identity', 'evaluator_identity')): return state
        launch = job / 'worker_launch.json'
        if launch.exists() and base.alive(base.read_json(launch).get('identity')): return state
        if state['metrics_present']:
            verified = validate_metrics(base.read_json(job / 'hard_test_metrics.json'), spec['trigger'], spec['binding'])
            set_status(job, 'complete', verified=verified, metrics_sha256=base.sha256(job / 'hard_test_metrics.json'))
            return remote_status(job)
        if not state['checkpoint_present']: return state
        if base.sha256(job / 'best.pt') != spec['trigger']['checkpoint_sha256']:
            raise ValueError('Remote snapshot SHA differs')
        # Full-test and all hard-test workers use the same physical-GPU lock.
        gpu = base.acquire_lock(spec['gpu_lock'])
        if gpu is None:
            set_status(job, 'queued', waiting_reason='Shared GPU lock is held'); return remote_status(job)
        gpu.close()
        from remote_evaluate import local_gpu_probe
        ready, reason, _ = local_gpu_probe(spec)
        if not ready:
            set_status(job, 'queued', waiting_reason=reason); return remote_status(job)
        with (job / 'worker.log').open('a') as output:
            child = subprocess.Popen([spec['python'], '-B', '-u', str(Path(__file__)), '--remote-worker', str(job)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        base.write_json(launch, {'identity': base.process_identity(child.pid), 'created_at': base.now()})
        return {**remote_status(job), 'status': 'launching'}
    finally:
        lock.close()


def remote_worker(job):
    job = remote_dir(job); spec = read_spec(job)
    own = base.acquire_lock(job / '.worker.lock')
    if own is None: return 0
    gpu = base.acquire_lock(spec['gpu_lock'])
    if gpu is None:
        set_status(job, 'queued', waiting_reason='Shared GPU lock is held'); own.close(); return 0
    try:
        if base.sha256(job / 'best.pt') != spec['trigger']['checkpoint_sha256']:
            raise ValueError('Snapshot changed before hard evaluation')
        set_status(job, 'running', worker_identity=base.process_identity(os.getpid()), started_at=base.now())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu_index']))
        cmd = [spec['python'], '-B', '-u', str(ROOT / 'code/evaluate_hard_subset.py'),
               '--checkpoint', str(job / 'best.pt'), '--hard-manifest', spec['binding']['manifest']['path'],
               '--output', str(job / 'hard_test_metrics.json'), '--device', 'cuda:0', '--amp', 'bf16',
               '--batch-size', '4', '--workers', '4']
        with (job / 'test.log').open('a') as output:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, pass_fds=(gpu.fileno(), own.fileno()))
            set_status(job, 'running', evaluator_identity=base.process_identity(child.pid), command=cmd)
            code = child.wait()
        if code: raise RuntimeError(f'Hard evaluator exited {code}')
        verified = validate_metrics(base.read_json(job / 'hard_test_metrics.json'), spec['trigger'], spec['binding'])
        set_status(job, 'complete', verified=verified, completed_at=base.now(),
                   metrics_sha256=base.sha256(job / 'hard_test_metrics.json'))
        return 0
    except Exception as error:
        set_status(job, 'failed', reason=f'{type(error).__name__}: {error}'); return 1
    finally:
        gpu.close(); own.close()


def transport(config, job):
    trigger = base.read_json(job / 'trigger.json'); binding = binding_for(job, config)
    r = config['remote']; destination = remote_path(config, job)
    spec = {'schema': 'vit_remote_hard_subset_v1', 'project': r['project'], 'python': r['python'],
            'gpu_lock': r['gpu_lock'], 'gpu_index': 3, 'minimum_free_mib': 10000,
            'trigger': trigger, 'binding': binding}
    is_remote = bool(r.get('host'))
    def call(args, payload=None):
        if is_remote: return ssh_call(config, args, payload)
        if args[0] == '--remote-prepare': return prepare_remote(args[1], payload)
        if args[0] == '--remote-ensure': return ensure_remote(args[1])
        raise ValueError('Unknown native operation')
    state = call(['--remote-prepare', destination], spec)
    if not state['checkpoint_present'] and state['status'] not in TERMINAL:
        if is_remote:
            subprocess.run(['rsync', '-a', '--partial', '--protect-args', str(job / 'best.pt'),
                            r['host'] + ':' + str(destination / 'best.pt')], check=True, timeout=1800,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            base.snapshot(job / 'best.pt', destination / 'best.pt')
    state = call(['--remote-ensure', destination])
    if state['status'] == 'complete':
        for name in ('hard_test_metrics.json', 'test.log'):
            if is_remote:
                subprocess.run(['rsync', '-a', '--protect-args', r['host'] + ':' + str(destination / name), str(job / name)],
                               check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            else:
                import shutil
                shutil.copyfile(destination / name, job / name)
        verified = validate_metrics(base.read_json(job / 'hard_test_metrics.json'), trigger, binding)
        if base.sha256(job / 'hard_test_metrics.json') != state['metrics_sha256']:
            raise ValueError('Downloaded hard metrics SHA mismatch')
        set_status(job, 'complete', verified=verified, completed_at=base.now(), execution_host=r.get('host', 'local103'),
                   official_full_test=False, score_milestone=None)
        # Hard-subset scores never cause the full-test >92 retention rule.
        (job / 'best.pt').unlink(missing_ok=True)
        if is_remote:
            ssh_call(config, ['--remote-ack', destination], {'metrics_sha256': state['metrics_sha256']})
        else:
            (destination / 'best.pt').unlink(missing_ok=True)
    elif state['status'] == 'failed':
        set_status(job, 'failed', reason=state.get('reason', 'Remote hard evaluation failed'))
    else:
        set_status(job, 'running' if state['status'] in ('running', 'launching') else 'queued',
                   execution_host=r.get('host', 'local103'), remote_job=str(destination), remote_status=state)


def run_monitor(config, once=False):
    directory = Path(config['state_directory']); directory.mkdir(parents=True, exist_ok=True)
    lock = base.acquire_lock(directory / '.monitor.lock')
    if lock is None: return 0
    try:
        path = directory / 'configuration.json'
        if path.exists() and base.read_json(path) != config: raise ValueError('Monitor configuration changed')
        base.write_json(path, config)
        start_path = directory / 'start_epochs.json'
        if not start_path.exists():
            starts = {}
            for p in config['runs']:
                summary_path = Path(p) / 'summary.json'
                start = config.get('start_after_completed_epoch')
                if start is None:
                    start = base.read_json(summary_path)['completed_epochs'] if summary_path.exists() else 0
                if type(start) is not int or start < 0:
                    raise ValueError('start_after_completed_epoch must be a nonnegative integer')
                starts[p] = start
            base.write_json(start_path, starts)
        starts = base.read_json(start_path)
        while True:
            errors = {}; jobs = {}; candidates = []
            for run_path in config['runs']:
                run = Path(run_path)
                try:
                    committed = base.committed_run(run)
                    if committed is None: continue
                    summary, training, rows = committed
                    if summary['status'] == 'user_stopped': continue
                    first = (starts[run_path] // 2 + 1) * 2
                    for epoch in range(first, summary['completed_epochs'] + 1, 2):
                        prepare_snapshot(run, epoch, summary, training, rows, config)
                    for status_path in sorted((run / 'hard_periodic_test').glob('*/status.json')):
                        job = status_path.parent; status = base.read_json(status_path)
                        if status['status'] not in TERMINAL: candidates.append(job)
                        jobs[str(job)] = status
                except Exception as error:
                    errors[run.name] = f'{type(error).__name__}: {error}'
            manifest_ready = Path(config['hard_manifest']).exists() and (ROOT / 'code/evaluate_hard_subset.py').exists()
            if manifest_ready:
                # Freeze one identity for the entire monitor, including future
                # jobs after older checkpoints and metrics were cleaned up.
                try:
                    identity = manifest_identity(config['hard_manifest'])
                    identity_path = directory / 'subset_identity.json'
                    if identity_path.exists() and base.read_json(identity_path) != identity:
                        raise ValueError('Fixed hard subset changed across evaluation epochs')
                    if not identity_path.exists():
                        base.write_json(identity_path, identity)
                except Exception as error:
                    errors['hard_manifest'] = f'{type(error).__name__}: {error}'
                    manifest_ready = False
            if manifest_ready:
                # Reconnect running jobs first; snapshot capture above never waits for the GPU.
                for job in sorted(candidates, key=lambda p: (base.read_json(p / 'status.json')['status'] != 'running', str(p))):
                    try:
                        transport(config, job); jobs[str(job)] = base.read_json(job / 'status.json')
                        if jobs[str(job)]['status'] == 'running': break
                    except Exception as error:
                        errors[str(job)] = f'{type(error).__name__}: {error}'
            state = {'schema': 'vit_hard_subset_monitor_state_v1', 'updated_at': base.now(),
                     'monitor_identity': base.process_identity(os.getpid()), 'manifest_ready': manifest_ready,
                     'cadence': 2, 'past_boundaries_backfilled': False, 'start_completed_epochs': starts,
                     'jobs': jobs, 'errors': errors, 'training_process_management': False,
                     'official_full_test': False, 'checkpoint_selection_uses_test': False}
            base.write_json(directory / 'state.json', state)
            print(json.dumps({'time': state['updated_at'], 'manifest_ready': manifest_ready, 'jobs': len(jobs), 'errors': errors}), flush=True)
            if once: return 0
            time.sleep(config['poll_seconds'])
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path); p.add_argument('--once', action='store_true')
    for option in ('remote-prepare', 'remote-ensure', 'remote-worker', 'remote-ack'):
        p.add_argument('--' + option, type=Path)
    args = p.parse_args()
    if args.remote_worker: return remote_worker(args.remote_worker)
    if args.remote_prepare or args.remote_ensure or args.remote_ack:
        try:
            if args.remote_prepare: result = prepare_remote(args.remote_prepare, json.load(sys.stdin))
            elif args.remote_ensure: result = ensure_remote(args.remote_ensure)
            else:
                job = remote_dir(args.remote_ack); state = remote_status(job); ack = json.load(sys.stdin)
                if state['status'] != 'complete' or ack['metrics_sha256'] != state['metrics_sha256']:
                    raise ValueError('Cannot acknowledge unverified hard metrics')
                (job / 'best.pt').unlink(missing_ok=True); result = remote_status(job)
            print(json.dumps(result)); return 0
        except Exception as error:
            print(json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'})); return 1
    if not args.config: p.error('--config is required')
    return run_monitor(load_config(args.config), args.once)


if __name__ == '__main__':
    raise SystemExit(main())
