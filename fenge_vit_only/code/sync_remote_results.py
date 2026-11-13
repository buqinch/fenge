#!/usr/bin/env python3
"""Read-only remote SSH/rsync text mirror with atomic epoch publication.

No torch/model dependency, process signaling, or training management. Remote
PIDs remain provenance and are never interpreted as local process identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = frozenset(('.json', '.jsonl', '.log', '.csv', '.md', '.txt'))


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_json(text):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON numeric overflow')
        return number
    return json.loads(text, parse_constant=reject, parse_float=finite_float)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                      prefix='.sync-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path, project=PROJECT):
    project = Path(project).resolve()
    path = Path(path).resolve()
    raw = path.read_bytes()
    config = strict_json(raw.decode('utf-8'))
    if config.get('schema') != 'vit_remote_text_sync_v1':
        raise ValueError('Expected vit_remote_text_sync_v1 configuration')
    host = config.get('host')
    if not isinstance(host, str) or host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@-]+', host):
        raise ValueError('host must be a plain SSH alias or user@host')
    remote_project = config.get('remote_project')
    if (not isinstance(remote_project, str) or not PurePosixPath(remote_project).is_absolute()
            or '..' in PurePosixPath(remote_project).parts or any(ord(c) < 32 for c in remote_project)):
        raise ValueError('remote_project must be an absolute remote project path')
    remote_project = PurePosixPath(remote_project)

    def local_path(value):
        if not isinstance(value, str) or not value:
            raise ValueError('A local result path is required')
        candidate = Path(value)
        candidate = (candidate if candidate.is_absolute() else project / candidate).resolve()
        if not candidate.is_relative_to(project / 'result') or candidate == project / 'result':
            raise ValueError('Local targets must remain strictly under this project/result')
        return candidate

    interval, timeout = config.get('interval_seconds', 60), config.get('timeout_seconds', 60)
    if type(interval) is not int or interval < 1 or type(timeout) is not int or timeout < 1:
        raise ValueError('Intervals and timeouts must be positive integer seconds')
    runs = config.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('At least one explicitly configured run is required')
    targets = []
    for row in runs:
        name, relative = row.get('name'), row.get('remote')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('Invalid run name')
        if (not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
                or '..' in PurePosixPath(relative).parts or any(ord(c) < 32 for c in relative)
                or len(PurePosixPath(relative).parts) < 2 or PurePosixPath(relative).parts[0] != 'result'):
            raise ValueError('Remote run must be a relative directory strictly under remote_project/result')
        targets.append({'name': name, 'remote': str(remote_project / relative), 'local': local_path(row.get('local'))})
    state_directory = local_path(config.get('state_directory', 'result/remote103'))
    paths = [target['local'] for target in targets]
    if len(set(paths)) != len(paths) or len({r['name'] for r in targets}) != len(targets):
        raise ValueError('Duplicate mirror name or destination')
    for index, first in enumerate(paths + [state_directory]):
        for second in (paths + [state_directory])[index + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError('Mirror and state directories cannot overlap or nest')
    identity = {'host': host, 'remote_project': str(remote_project),
                'targets': [{**target, 'local': str(target['local'])} for target in targets]}
    return {'path': str(path), 'config_sha256': hashlib.sha256(raw).hexdigest(), 'host': host,
            'interval_seconds': interval, 'timeout_seconds': timeout, 'targets': targets,
            'state_directory': state_directory, 'identity': identity}


def invoke(command, timeout):
    return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=timeout, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})


def fetch(config, target, stage, run=invoke):
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes']
    probe = run([*ssh, config['host'], 'test -d ' + shlex.quote(target['remote'])], config['timeout_seconds'])
    if probe.returncode == 1:
        return False
    if probe.returncode:
        raise RuntimeError('Remote directory check failed: ' + probe.stderr.strip()[-1500:])
    target['local'].mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-rt', '--checksum', '--safe-links', '--prune-empty-dirs', '--max-size=64m',
               '--timeout=45', '--compare-dest=' + str(target['local']), '-e', shlex.join(ssh), '--include=*/']
    command.extend('--include=*' + suffix for suffix in sorted(TEXT_SUFFIXES))
    command.extend(['--exclude=*', config['host'] + ':' + shlex.quote(target['remote'].rstrip('/') + '/'), str(stage) + '/'])
    result = run(command, config['timeout_seconds'])
    if result.returncode:
        raise RuntimeError(f'rsync failed ({result.returncode}): ' + result.stderr.strip()[-1500:])
    return True


def read_text_files(stage):
    files, documents = {}, {}
    for source in sorted(Path(stage).rglob('*')):
        if source.is_symlink():
            raise ValueError('Symlinks cannot be published as mirrored results')
        if not source.is_file():
            continue
        if source.suffix not in TEXT_SUFFIXES:
            raise ValueError('Non-text extension reached staging')
        relative = source.relative_to(stage)
        text = source.read_text(encoding='utf-8')
        if '\0' in text:
            raise ValueError(f'Binary data reached text staging: {relative}')
        if source.suffix == '.json':
            documents[relative] = strict_json(text)
        elif source.suffix == '.jsonl':
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith(('\n', '\r')):
                lines.pop()  # An unterminated append is not committed yet.
                source.write_text(''.join(lines), encoding='utf-8')
            for line in lines:
                if line.strip():
                    strict_json(line)
        files[relative] = source
    return files, documents


def validate_boundary(target, files, documents):
    """Accept only direct ViT validation metrics and its matching history."""
    local = target['local']
    def document(name):
        key = Path(name)
        if key in documents:
            return documents[key]
        path = local / key
        return strict_json(path.read_text()) if path.is_file() else None
    run_config = document('run_config.json')
    incoming_config = documents.get(Path('run_config.json'))
    old_config = local / 'run_config.json'
    if incoming_config is not None and old_config.exists() and strict_json(old_config.read_text()) != incoming_config:
        raise ValueError('Remote run_config differs from the already published run identity')
    summary = documents.get(Path('summary.json'))
    if summary is None:
        return None, 'waiting_for_run_config' if run_config is None else None
    completed = summary.get('completed_epochs')
    if type(completed) is not int or completed < 0:
        raise ValueError('Summary completed_epochs must be a nonnegative integer')
    previous = document('summary.json') if Path('summary.json') not in documents else (
        strict_json((local / 'summary.json').read_text()) if (local / 'summary.json').is_file() else None)
    if previous and previous.get('completed_epochs', -1) > completed:
        raise ValueError('Remote epoch boundary moved behind the published local summary')
    if run_config is None:
        return None, 'waiting_for_run_config'
    records = []
    for epoch in range(1, completed + 1):
        record = document(f'validation_epoch_{epoch:03d}.json')
        if record is None:
            return None, f'waiting_for_validation_epoch_{epoch:03d}'
        if record.get('epoch') != epoch or not isinstance(record.get('miou'), (float, int)):
            raise ValueError('Validation filename/epoch or direct metrics structure differs')
        records.append(record)
    history_file = files.get(Path('history.jsonl'), local / 'history.jsonl')
    if completed and not history_file.is_file():
        return None, 'waiting_for_history'
    history = [strict_json(line) for line in history_file.read_text().splitlines() if line.strip()] if history_file.is_file() else []
    by_epoch = {}
    for row in history:
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 1 or epoch in by_epoch:
            raise ValueError('History epoch is invalid or duplicated')
        by_epoch[epoch] = row
    for record in records:
        epoch = record['epoch']
        if epoch not in by_epoch:
            return None, f'waiting_for_history_epoch_{epoch:03d}'
        if by_epoch[epoch].get('validation_miou') != record['miou']:
            raise ValueError('History validation mIoU disagrees with the completed epoch metrics')
    if records:
        best = max(records, key=lambda record: record['miou'])  # First strict maximum.
        if summary.get('best_epoch') != best['epoch'] or summary.get('best_miou') != best['miou']:
            raise ValueError('Summary best epoch/score differs from completed validation metrics')
    return summary, None


def publish(target, stage):
    files, documents = read_text_files(stage)
    summary, pending = validate_boundary(target, files, documents)
    paths = sorted((p for p in files if p.name != 'summary.json'), key=lambda p: (p.name != 'run_config.json', str(p)))
    if summary is not None:
        paths.extend(p for p in files if p.name == 'summary.json')
    for relative in paths:
        destination = target['local'] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.remote-text-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copy2(files[relative], temporary)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'status': 'pending' if pending else 'synced', 'pending_reason': pending,
            'published_files': len(paths), 'summary_published': summary is not None,
            'completed_epochs': summary['completed_epochs'] if summary is not None else None}


def sync_once(config, previous=None, run=invoke):
    previous = previous or {}
    if previous.get('identity', config['identity']) != config['identity']:
        raise ValueError('Synchronization source/destination identity changed; refuse reusing prior state')
    state_directory = config['state_directory']
    state_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for target in config['targets']:
        try:
            with tempfile.TemporaryDirectory(prefix='.stage-', dir=state_directory) as temporary:
                stage = Path(temporary)
                if not fetch(config, target, stage, run):
                    result = {'status': 'pending', 'pending_reason': 'remote_directory_not_created'}
                else:
                    result = publish(target, stage)
        except Exception as error:
            result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        results.append({'name': target['name'], 'remote_directory': target['remote'],
                        'local_directory': str(target['local']), **result})
    status = 'failed' if any(r['status'] == 'failed' for r in results) else (
        'pending' if any(r['status'] == 'pending' for r in results) else 'synced')
    state = {'schema': 'vit_remote_text_sync_state_v1', 'status': status, 'updated_at': now(),
             'identity': config['identity'], 'config_sha256': config['config_sha256'],
             'local_host': socket.gethostname(), 'local_sync_pid': os.getpid(), 'runs': results,
             'last_success_at': now() if status == 'synced' else previous.get('last_success_at'),
             'process_management': 'none; remote PIDs are never used to manage local processes',
             'mirrored_suffixes': sorted(TEXT_SUFFIXES), 'weights_and_images_mirrored': False}
    write_json(state_directory / 'sync_state.json', state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config/remote103.json')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    directory = config['state_directory']
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'sync.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'A text sync process already holds this state-directory lock.\n')
        state_path = directory / 'sync_state.json'
        previous = strict_json(state_path.read_text()) if state_path.is_file() else {}
        while True:
            state = sync_once(config, previous)
            print(json.dumps({'at': state['updated_at'], 'status': state['status'], 'runs': state['runs']}, ensure_ascii=False), flush=True)
            if args.once:
                return 1 if state['status'] == 'failed' else 0
            previous = state
            time.sleep(config['interval_seconds'])


if __name__ == '__main__':
    raise SystemExit(main())
