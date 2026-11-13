#!/usr/bin/env python3
"""Run immutable periodic-test snapshots on an SSH host and mirror their results.

The remote worker survives an SSH disconnect. Its job/GPU locks are inherited
by the evaluator; restarting the local observer reconnects to the same job.
Neither endpoint starts, stops, or edits a training process.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

import periodic_evaluate as periodic

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_FILES = ('test.log', 'test_metrics.json', 'remote_execution.json', 'remote_worker.log')


def validate_remote(config):
    remote = config.get('remote')
    if not isinstance(remote, dict):
        raise ValueError('A remote evaluation configuration is required')
    remote = dict(remote)
    host = remote.get('host')
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError('remote.host must be an SSH alias or user@host')
    for name in ('project', 'python', 'gpu_lock'):
        value = remote.get(name)
        if (not isinstance(value, str) or not value.startswith('/') or '..' in Path(value).parts
                or any(ord(c) < 32 for c in value)):
            raise ValueError('remote.' + name + ' must be an absolute path')
    wrapper = remote.get('wrapper', 'code/run_on_103.sh')
    if (not isinstance(wrapper, str) or Path(wrapper).is_absolute() or '..' in Path(wrapper).parts
            or any(ord(c) < 32 for c in wrapper)):
        raise ValueError('remote.wrapper must remain inside the remote project')
    remote['wrapper'] = wrapper
    return remote


def _ssh(config, arguments, payload=None):
    remote = validate_remote(config)
    project = Path(remote['project'])
    command = ['bash', str(project / remote['wrapper']), remote['python'],
               str(project / 'code/remote_evaluate.py'), *map(str, arguments)]
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        remote['host'], shlex.join(command)], input=json.dumps(payload) if payload is not None else None,
        text=True, capture_output=True, check=True, timeout=60)
    document = json.loads(result.stdout)
    if document.get('status') == 'error':
        raise ValueError('Remote evaluator: ' + document['error'])
    return document


def local_gpu_probe(config):
    def query(arguments):
        return subprocess.run(['nvidia-smi', *arguments], check=True, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    row = query(['--id=' + str(config['gpu_index']), '--query-gpu=uuid,memory.free',
                 '--format=csv,noheader,nounits']).split(',')
    uuid, free = row[0].strip(), int(row[1])
    if free < config['minimum_free_mib']:
        return False, f'GPU {config["gpu_index"]}: {free} MiB free', uuid
    processes = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        return False, f'GPU {config["gpu_index"]} has another compute process', uuid
    return True, f'GPU {config["gpu_index"]}: {free} MiB free', uuid


def probe_remote(config):
    result = _ssh(config, ['--probe'], {key: config[key] for key in ('gpu_index', 'minimum_free_mib')})
    return result['ready'], result['reason'], result['gpu_uuid']


def remote_job_path(config, job):
    remote = validate_remote(config)
    job = Path(job).resolve()
    if job.parent.name != 'periodic_test' or not job.is_relative_to(ROOT / 'result'):
        raise ValueError('Expected a local result/run/periodic_test/job directory')
    for name in (job.parent.parent.name, job.name):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError('Unsafe run or job name')
    return Path(remote['project']) / 'result/remote_evaluations' / job.parent.parent.name / job.name


def prepare_spec(config, job, trigger):
    remote = validate_remote(config)
    job = Path(job)
    if periodic.sha256(job / 'best.pt') != trigger['checkpoint_sha256']:
        raise ValueError('Local snapshot SHA changed before remote transfer')
    training = periodic.read_json(Path(trigger['run']) / 'run_config.json')
    if periodic.fingerprint(training) != trigger['configuration_sha256']:
        raise ValueError('Training configuration changed before remote transfer')
    periodic.check_sources(training)
    if periodic.settings(config) != trigger['evaluation']:
        raise ValueError('Evaluation source or manifest changed before remote transfer')
    return {'schema': 'vit_remote_evaluation_v1', 'remote_project': remote['project'],
            'python': remote['python'], 'gpu_index': config['gpu_index'],
            'minimum_free_mib': config['minimum_free_mib'], 'gpu_lock': remote['gpu_lock'],
            'test_manifest': config['test_manifest'], 'training_configuration': training,
            'trigger': trigger, 'evaluation': trigger['evaluation'],
            'remote_executor_sha256': periodic.sha256(Path(__file__)),
            'retain_above_score': config.get('retain_above_score')}


def _rsync(config, source, destination):
    subprocess.run(['rsync', '-a', '--partial', '--delay-updates', '--protect-args',
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', '--', str(source), str(destination)],
        check=True, capture_output=True, text=True, timeout=1800)


def _sync_results(config, remote_job, job, state):
    remote = validate_remote(config)
    for name in state.get('files', []):
        if name not in TRANSPORT_FILES:
            raise ValueError('Unexpected remote result filename')
        _rsync(config, remote['host'] + ':' + str(remote_job / name), job / name)


def run_remote(config, job, trigger):
    """Return 0 after verified download, or 75 to retry a busy/disconnected job."""
    job = Path(job).resolve()
    remote = validate_remote(config)
    destination = remote_job_path(config, job)
    spec = prepare_spec(config, job, trigger)
    spec_path = job / 'remote_spec.json'
    if spec_path.exists() and periodic.read_json(spec_path) != spec:
        raise ValueError('Remote job specification changed after selection')
    periodic.write_json(spec_path, spec)
    try:
        state = _ssh(config, ['--prepare', destination], spec)
        if state.get('status') not in ('running', 'launching', 'complete', 'failed'):
            # rsync publishes by rename; the evaluator never reads a partial transfer.
            _rsync(config, job / 'best.pt', remote['host'] + ':' + str(destination / 'best.pt'))
        state = _ssh(config, ['--ensure', destination])
        while True:
            _sync_results(config, destination, job, state)
            status = state.get('status')
            if status == 'complete':
                metrics = periodic.read_json(job / 'test_metrics.json')
                periodic.validate_metrics(metrics, trigger)
                if periodic.sha256(job / 'test_metrics.json') != state['metrics_sha256']:
                    raise ValueError('Downloaded metrics SHA mismatch')
                # Local checkpoint remains until periodic.complete_job applies retention.
                acknowledged = _ssh(config, ['--ack', destination], {'metrics_sha256': state['metrics_sha256']})
                _sync_results(config, destination, job, acknowledged)
                return 0
            if status == 'failed':
                raise RuntimeError('Remote evaluation failed: ' + state.get('reason', 'see remote_worker.log'))
            if status == 'queued':
                return 75
            time.sleep(10)
            state = _ssh(config, ['--ensure', destination])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        # The remote worker is independent of this connection. Next queue scan
        # reconnects using immutable spec/SHA and its durable PID identities.
        periodic.write_json(job / 'remote_transport.json', {
            'status': 'retry', 'updated_at': periodic.now(), 'host': remote['host'],
            'remote_job': str(destination), 'reason': str(error)})
        return 75


def _job_directory(value):
    job = Path(value).resolve()
    root = ROOT / 'result/remote_evaluations'
    if not job.is_relative_to(root) or len(job.relative_to(root).parts) != 2:
        raise ValueError('Remote jobs must be project/result/remote_evaluations/run/job')
    return job


def _state(job):
    path = job / 'remote_execution.json'
    value = periodic.read_json(path) if path.exists() else {'status': 'prepared'}
    return {**value, 'files': [name for name in TRANSPORT_FILES if (job / name).is_file()]}


def _set_state(job, status, **fields):
    previous = _state(job)
    previous.pop('files', None)
    value = {**previous, 'status': status, 'updated_at': periodic.now(),
             'host': socket.gethostname(), **fields}
    periodic.write_json(job / 'remote_execution.json', value)
    return _state(job)


def _spec(job):
    spec = periodic.read_json(job / 'spec.json')
    if spec.get('schema') != 'vit_remote_evaluation_v1' or Path(spec['remote_project']).resolve() != ROOT:
        raise ValueError('Wrong remote project or job schema')
    if spec['remote_executor_sha256'] != periodic.sha256(Path(__file__)):
        raise ValueError('Remote executor source differs from the submitting host')
    if periodic.fingerprint(spec['training_configuration']) != spec['trigger']['configuration_sha256']:
        raise ValueError('Remote training configuration fingerprint mismatch')
    return spec


def prepare_remote(job, spec):
    job = _job_directory(job)
    job.mkdir(parents=True, exist_ok=True)
    lock = periodic.acquire_lock(job / '.prepare.lock')
    if lock is None:
        return {'status': 'launching', 'files': []}
    try:
        path = job / 'spec.json'
        if path.exists() and periodic.read_json(path) != spec:
            raise ValueError('Refusing to replace an existing remote job specification')
        periodic.write_json(path, spec)
        _spec(job)
        return _state(job)
    finally:
        lock.close()


def _active_processes(job):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [arg.decode() for arg in (entry / 'cmdline').read_bytes().split(b'\0') if arg]
            worker = ('--worker' in args and str(job) in args
                      and any(Path(arg).name == 'remote_evaluate.py' for arg in args))
            evaluator = ('--output' in args and str(job / 'test_metrics.json') in args
                         and any(Path(arg).name == 'evaluate.py' for arg in args))
            identity = periodic.process_identity(int(entry.name))
            if (worker or evaluator) and identity:
                found.append(identity)
        except (OSError, UnicodeError):
            pass
    return found


def ensure(job, spawn=None, probe=None):
    job = _job_directory(job)
    spec = _spec(job)
    launch_lock = periodic.acquire_lock(job / '.launch.lock')
    if launch_lock is None:
        return _state(job)
    try:
        state = _state(job)
        if state['status'] in ('complete', 'failed'):
            return state
        identities = [state.get('worker_identity'), state.get('evaluator_identity')]
        launch_file = job / 'remote_launch.json'
        if launch_file.exists():
            identities.append(periodic.read_json(launch_file).get('worker_identity'))
        if any(periodic.alive(identity) for identity in identities) or _active_processes(job):
            return state if state['status'] in ('running', 'launching') else {**state, 'status': 'running'}
        # A completed evaluator can outlive its worker and publish metrics first.
        if (job / 'test_metrics.json').exists():
            verified = periodic.validate_metrics(periodic.read_json(job / 'test_metrics.json'), spec['trigger'])
            return _set_state(job, 'complete', verified=verified, recovered_after_worker_exit=True,
                              metrics_sha256=periodic.sha256(job / 'test_metrics.json'))
        if state['status'] in ('running', 'launching'):
            return _set_state(job, 'failed', reason='Remote worker and evaluator disappeared before producing complete metrics')
        ready, reason, uuid = (probe or local_gpu_probe)(spec)
        if not ready:
            return _set_state(job, 'queued', waiting_reason=reason, physical_gpu=spec['gpu_index'], gpu_uuid=uuid)
        if periodic.sha256(job / 'best.pt') != spec['trigger']['checkpoint_sha256']:
            raise ValueError('Remote checkpoint transfer SHA mismatch')
        _set_state(job, 'launching', launch_requested_at=periodic.now())
        with (job / 'remote_worker.log').open('a') as output:
            child = (spawn or subprocess.Popen)([spec['python'], '-u', str(Path(__file__).resolve()),
                '--worker', str(job)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
        periodic.write_json(launch_file, {'worker_identity': periodic.process_identity(child.pid),
                            'requested_at': periodic.now(), 'spec_sha256': periodic.sha256(job / 'spec.json')})
        return _state(job)
    finally:
        launch_lock.close()


def remote_worker(job, probe=None):
    job = _job_directory(job)
    worker_lock = periodic.acquire_lock(job / '.worker.lock')
    if worker_lock is None:
        return 0
    gpu_lock = None
    try:
        spec = _spec(job)
        if _state(job)['status'] == 'complete':
            return 0
        _set_state(job, 'running', worker_identity=periodic.process_identity(os.getpid()),
                   stage='checking_snapshot_and_sources', started_at=periodic.now())
        if periodic.sha256(job / 'best.pt') != spec['trigger']['checkpoint_sha256']:
            raise ValueError('Remote checkpoint SHA changed before evaluation')
        periodic.check_sources(spec['training_configuration'])
        if periodic.settings(spec) != spec['evaluation']:
            raise ValueError('Remote evaluation source or test manifest differs from the selected protocol')
        gpu_lock = periodic.acquire_lock(spec['gpu_lock'])
        if gpu_lock is None:
            _set_state(job, 'queued', waiting_reason='Shared remote GPU lock is held')
            return 75
        ready, reason, uuid = (probe or local_gpu_probe)(spec)
        if not ready:
            _set_state(job, 'queued', waiting_reason=reason)
            return 75
        packages = {}
        for name in ('torch', 'torchvision', 'timm', 'numpy', 'scipy'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        _set_state(job, 'running', stage='evaluating', physical_gpu=spec['gpu_index'], gpu_uuid=uuid,
                   python=sys.version, packages=packages, checkpoint_sha256=spec['trigger']['checkpoint_sha256'],
                   source_sha256=spec['training_configuration']['source_sha256'],
                   evaluation=spec['evaluation'])
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'PYTHONUNBUFFERED': '1',
                       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        with (job / 'test.log').open('a') as output:
            child = subprocess.Popen(periodic.evaluator_command(spec, job), cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(gpu_lock.fileno(), worker_lock.fileno()))
            _set_state(job, 'running', evaluator_identity=periodic.process_identity(child.pid))
            returncode = child.wait()
        if returncode:
            raise RuntimeError(f'Evaluator exited with status {returncode}; see test.log')
        verified = periodic.validate_metrics(periodic.read_json(job / 'test_metrics.json'), spec['trigger'])
        _set_state(job, 'complete', completed_at=periodic.now(), verified=verified,
                   metrics_sha256=periodic.sha256(job / 'test_metrics.json'))
        return 0
    except Exception as error:
        _set_state(job, 'failed', failed_at=periodic.now(), reason=f'{type(error).__name__}: {error}')
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        worker_lock.close()


def acknowledge(job, payload):
    job = _job_directory(job)
    state, spec = _state(job), _spec(job)
    if state['status'] != 'complete' or payload.get('metrics_sha256') != state.get('metrics_sha256'):
        raise ValueError('Only verified complete metrics may acknowledge remote checkpoint cleanup')
    threshold = spec.get('retain_above_score')
    keep = threshold is not None and state['verified']['weighted_score'] > threshold
    if not keep:
        (job / 'best.pt').unlink(missing_ok=True)
    return _set_state(job, 'complete', downloaded_at=periodic.now(), checkpoint_retained=keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    for name in ('prepare', 'ensure', 'worker', 'ack'):
        group.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        return remote_worker(args.worker)
    try:
        if args.probe:
            ready, reason, uuid = local_gpu_probe(json.load(sys.stdin))
            result = {'ready': ready, 'reason': reason, 'gpu_uuid': uuid}
        elif args.prepare:
            result = prepare_remote(args.prepare, json.load(sys.stdin))
        elif args.ensure:
            result = ensure(args.ensure)
        else:
            result = acknowledge(args.ack, json.load(sys.stdin))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'}), flush=True)
        return 0  # Structured protocol error; distinct from SSH/transport failure.


if __name__ == '__main__':
    raise SystemExit(main())
