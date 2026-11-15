#!/usr/bin/env python3
"""Change an explicitly listed live training run's selection policy at an epoch boundary.

The old checkpoint bytes are never rewritten. Only the exact Python torchrun
launcher is signalled; wrappers, unrelated jobs and reused PIDs are not targets.
Every irreversible stage is journalled and ambiguous restarts fail closed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SCHEMA = 'lusnar_fulltest_transition_v1'
RECORD_SCHEMA = 'lusnar_training_policy_transition_v1'
CHANGED_SOURCES = {'code/train_hardval.py', 'code/train_hardval_distributed.py'}
NEW_SOURCE = 'code/fulltest_policy.py'
POLICY = {'selection_policy': 'full_test_every5', 'early_stopping_patience': 0}
ENTRY = 'code/train_hardval_distributed.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process_info(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = [a.decode() for a in (root / 'cmdline').read_bytes().split(b'\0') if a]
        return {'pid': int(pid), 'start_ticks': int(fields[19]), 'ppid': int(fields[1]),
                'argv': argv, 'cwd': str((root / 'cwd').resolve()),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def identity(info):
    return {k: info[k] for k in ('pid', 'start_ticks', 'boot_id')} if info else None


def alive(ident):
    return bool(ident and identity(process_info(ident['pid'])) == ident)


def processes():
    return [info for p in Path('/proc').iterdir() if p.name.isdigit()
            if (info := process_info(int(p.name))) is not None]


def matches(info, spec, runtime, launcher=False):
    argv = info['argv']
    if not argv or 'python' not in Path(argv[0]).name:
        return False
    if argv.count('--config') != 1 or argv.index('--config') + 1 >= len(argv):
        return False
    config = Path(argv[argv.index('--config') + 1])
    config = config if config.is_absolute() else Path(info['cwd']) / config
    return (config.resolve() == Path(spec['config'])
            and str(Path(runtime) / ENTRY) in argv
            and (not launcher or 'torch.distributed.run' in argv))


def launcher_access_errors(spec, proc_root=Path('/proc')):
    """Explain cross-user-namespace /proc denial without weakening PID checks.

    A watcher inside a fresh bwrap namespace cannot inspect an older sibling
    namespace's cwd/environment. Its trainer children may still use wrappers;
    this supervisory process must run directly in the host namespace.
    """
    problems = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode() for part in (entry / 'cmdline').read_bytes().split(b'\0') if part]
            if (not argv or 'python' not in Path(argv[0]).name or
                    'torch.distributed.run' not in argv or argv.count('--config') != 1 or
                    str(Path(spec['old_runtime']) / ENTRY) not in argv or
                    argv[argv.index('--config') + 1] != spec['config']):
                continue
            for field in ('cwd', 'environ'):
                try:
                    path = entry / field
                    if field == 'cwd':
                        path.resolve()
                    else:
                        path.read_bytes()
                except PermissionError:
                    problems.append({'pid': int(entry.name), 'field': field,
                                     'reason': 'permission_denied'})
        except (OSError, IndexError, UnicodeError):
            continue
    return problems


def descendants(infos, pid):
    selected = {pid}
    while True:
        added = {p['pid'] for p in infos if p['ppid'] in selected} - selected
        if not added:
            return [identity(p) for p in infos if p['pid'] in selected]
        selected |= added


def validate_launcher_binding(launcher, plan):
    """The exact launcher must descend from the recorded, unreused launch PID."""
    original = plan['original_launch'].get('identity', {})
    current = process_info(original.get('pid', -1))
    if not current or current['start_ticks'] != int(original.get('start_ticks', -1)):
        raise ValueError('Original launch identity is no longer live')
    cursor = launcher
    seen = set()
    while cursor and cursor['pid'] not in seen and cursor['pid'] != current['pid']:
        seen.add(cursor['pid']); cursor = process_info(cursor['ppid'])
    if not cursor or cursor['pid'] != current['pid']:
        raise ValueError('Torchrun is not descended from the recorded launcher')
    raw = Path('/proc', str(launcher['pid']), 'environ').read_bytes().split(b'\0')
    actual = dict(part.decode().split('=', 1) for part in raw if b'=' in part)
    for key, value in plan['environment'].items():
        if actual.get(key) != value:
            raise ValueError('Live launch environment differs for ' + key)


def load_config(path):
    path = Path(path).resolve()
    c = read_json(path)
    if c.get('schema') != SCHEMA or not isinstance(c.get('runs'), list) or not c['runs']:
        raise ValueError('Expected ' + SCHEMA + ' with explicit runs')
    for key in ('state_directory', 'python'):
        if not Path(c[key]).is_absolute():
            raise ValueError(key + ' must be absolute')
        c[key] = str(Path(c[key]).resolve())
    seen = set()
    for run in c['runs']:
        for key in ('run', 'config', 'old_runtime', 'new_runtime', 'launch_record'):
            if not Path(run[key]).is_absolute():
                raise ValueError(key + ' must be absolute')
            run[key] = str(Path(run[key]).resolve())
        if run['run'] in seen or run['old_runtime'] == run['new_runtime']:
            raise ValueError('Duplicate run or unchanged runtime')
        if type(run.get('wait_after_epoch')) is not int or run['wait_after_epoch'] < 0:
            raise ValueError('wait_after_epoch must be nonnegative')
        seen.add(run['run'])
    c['config_path'] = str(path)
    return c


def source_tables(spec, old_training):
    old = old_training.get('source_sha256')
    if not isinstance(old, dict) or ENTRY not in old or not old:
        raise ValueError('Missing old training source binding')
    new = {}
    for relative, expected in old.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'code':
            raise ValueError('Unsafe training source path')
        old_path = Path(spec['old_runtime']) / rel
        new_path = Path(spec['new_runtime']) / rel
        if sha256(old_path) != expected:
            raise ValueError('Old training source changed: ' + relative)
        actual = sha256(new_path)
        if actual != expected and relative not in CHANGED_SOURCES:
            raise ValueError('Non-policy source changed: ' + relative)
        new[relative] = actual
    helper = Path(spec['new_runtime']) / NEW_SOURCE
    new[NEW_SOURCE] = sha256(helper)
    for rel in CHANGED_SOURCES | {NEW_SOURCE}:
        ast.parse((Path(spec['new_runtime']) / rel).read_text(), filename=rel)
    return old, new


def prepare_plan(c, spec):
    run = Path(spec['run'])
    old_training = read_json(run / 'run_config.json')
    old_config = read_json(spec['config'])
    if Path(old_config['output']).resolve() != run:
        raise ValueError('Training config output differs from run')
    if old_training.get('selection_policy', 'hard_validation_best') != 'hard_validation_best':
        raise ValueError('Old run is not the hard-validation selection policy')
    for key, value in old_config.items():
        if old_training.get(key) != value:
            raise ValueError('Current config differs from recorded training: ' + key)
    old_sources, new_sources = source_tables(spec, old_training)
    launch = read_json(spec['launch_record'])
    command = launch.get('command')
    if (not isinstance(command, list) or not command or
            any(not isinstance(x, str) or not x or '\0' in x for x in command) or
            command.count('--config') != 1 or 'torch.distributed.run' not in command or
            str(Path(spec['old_runtime']) / ENTRY) not in command or
            Path(command[command.index('--config') + 1]).resolve() != Path(spec['config'])):
        raise ValueError('Original launcher command does not match this run')
    environment = launch.get('environment', {})
    if (not isinstance(environment, dict) or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()) or
            not environment.get('CUDA_VISIBLE_DEVICES') or
            environment.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'):
        raise ValueError('Original launch must record GPU binding and environment')
    command = list(command)
    if '--resume' in command:
        i = command.index('--resume'); del command[i:i + 2]
    old_root = Path(spec['old_runtime'])
    for i, part in enumerate(command):
        p = Path(part)
        if p.is_absolute() and p.is_relative_to(old_root):
            relative = p.relative_to(old_root)
            replacement = Path(spec['new_runtime']) / relative
            if not replacement.is_file():
                raise ValueError('Missing new launch source: ' + str(relative))
            if str(relative) not in CHANGED_SOURCES and sha256(p) != sha256(replacement):
                raise ValueError('Non-policy launch wrapper changed')
            command[i] = str(replacement)
        elif 'python' in p.name and i < command.index('torch.distributed.run'):
            command[i] = c['python']
    command += ['--resume', str(run / 'last.pt')]
    new_config = {**old_config, **POLICY}
    return {'old_training': old_training, 'old_config': old_config, 'new_config': new_config,
            'old_source_sha256': old_sources, 'new_source_sha256': new_sources,
            'original_launch': launch, 'command': command, 'environment': environment,
            'cwd': launch.get('cwd', str(run.parent.parent)), 'prepared_at': now()}


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def verify_checkpoint(spec, plan, minimum_epoch):
    run = Path(spec['run'])
    saved = load_checkpoint(run / 'last.pt')
    epoch = saved.get('epoch')
    if (saved.get('format') != 'lusnar_vit_checkpoint_v1' or saved.get('partial') is not False or
            type(epoch) is not int or epoch < minimum_epoch):
        raise ValueError('Checkpoint is partial or older than committed epoch')
    if saved.get('train_config') != plan['old_training']:
        raise ValueError('Checkpoint training configuration differs')
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rank_states', 'rng', 'distributed'):
        if key not in saved:
            raise ValueError('Full-state resume checkpoint is missing ' + key)
    world = saved['distributed'].get('world_size')
    states = saved['rank_states']
    if (type(world) is not int or world < 1 or not isinstance(states, list) or
            len(states) != world or [x.get('rank') for x in states] != list(range(world))):
        raise ValueError('Missing ordered per-rank checkpoint state')
    for state in states:
        if any(key not in state for key in ('rng', 'loader_generator', 'val_generator')):
            raise ValueError('Rank RNG/loader state is incomplete')
    summary = read_json(run / 'summary.json')
    if (saved.get('data_identity') != summary.get('data_identity') or
            saved.get('model_config') != summary.get('model_config') or
            summary['completed_epochs'] > epoch):
        raise ValueError('Checkpoint summary/identity/model mismatch')
    rows = []
    for line in (run / 'history.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        rows.append(json.loads(line))
    if len(rows) < epoch or [r['epoch'] for r in rows[:epoch]] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint history prefix is incomplete')
    if rows[epoch - 1].get('global_update') != saved.get('global_update'):
        raise ValueError('Checkpoint/history update count differs')
    return epoch


def stop_exact(spec, launcher, tracked, timeout=60):
    current = process_info(launcher['pid'])
    if identity(current) != identity(launcher) or not matches(current, spec, spec['old_runtime'], True):
        raise ValueError('Torchrun identity changed before SIGTERM')
    os.kill(launcher['pid'], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while True:
        remaining = [p for p in tracked if alive(p)]
        remaining += [identity(p) for p in processes() if matches(p, spec, spec['old_runtime'])]
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Old ranks/DataLoader processes still alive after SIGTERM: ' + str(remaining))
        time.sleep(1)


def archive_small(run, spec, plan):
    directory = Path(run) / 'policy_transition_archive'
    directory.mkdir(parents=True, exist_ok=True)
    for src, name in [(Path(run) / 'run_config.json', 'run_config.json'),
                      (Path(run) / 'summary.json', 'summary.json'),
                      (Path(spec['launch_record']), 'launch.json'), (Path(spec['config']), 'config.json')]:
        if src.stat().st_size > 20 * 1024 * 1024:
            raise ValueError('Refusing a large transition archive')
        destination = directory / name
        if destination.exists():
            if destination.read_bytes() != src.read_bytes():
                raise ValueError('Transition archive already exists with different contents')
        else:
            destination.write_bytes(src.read_bytes())
    write_json(directory / 'transition_plan.json', plan)


def checkpoint_transition(spec, plan, epoch):
    return {'schema': RECORD_SCHEMA, 'created_at': now(),
            'from_policy': 'hard_validation_best', 'to_policy': 'full_test_every5',
            'resume_checkpoint_sha256': sha256(Path(spec['run']) / 'last.pt'),
            'old_source_sha256': plan['old_source_sha256'],
            'new_source_sha256': plan['new_source_sha256'], 'resume_epoch': epoch,
            'old_runtime': spec['old_runtime'], 'new_runtime': spec['new_runtime'],
            'preserved_state': ['model', 'optimizer', 'scheduler', 'scaler', 'rank_states', 'rng', 'global_update'],
            'changed_training_parameters': POLICY, 'checkpoint_bytes_modified': False}


def run_key(spec):
    return Path(spec['run']).name + '-' + hashlib.sha256(spec['run'].encode()).hexdigest()[:10]


def tick_run(c, spec):
    state_file = Path(c['state_directory']) / (run_key(spec) + '.json')
    state = read_json(state_file) if state_file.exists() else {}
    run = Path(spec['run'])
    launch_file = run / 'policy_transition_launch.json'

    def publish(status, **fields):
        state.update(status=status, updated_at=now(), run=spec['run'], **fields)
        write_json(state_file, state)
        return state

    try:
        if state.get('status') == 'failed':
            return state
        if launch_file.exists() or state.get('status') in ('launch_intent', 'launched', 'running'):
            launch = read_json(launch_file) if launch_file.exists() else None
            current = [p for p in processes() if matches(p, spec, spec['new_runtime'], True)]
            if len(current) == 1:
                return publish('running', trainer_identity=identity(current[0]), transition_complete=True)
            if len(current) > 1:
                raise RuntimeError('Multiple new-runtime launchers exist')
            if launch and alive(launch.get('identity')):
                return publish('launched', reason='New wrapper is starting torchrun')
            raise RuntimeError('Launch intent exists but no verified live process; refusing duplicate launch')
        if state.get('status') in ('stopping', 'stopped', 'prepared'):
            raise RuntimeError('Interrupted transition requires review; refusing to repeat signals or spawn')
        summary_path = run / 'summary.json'
        if not summary_path.exists():
            return publish('waiting', reason='Waiting for first completed epoch')
        summary = read_json(summary_path)
        completed = summary.get('completed_epochs')
        if type(completed) is not int or completed <= spec['wait_after_epoch']:
            return publish('waiting', completed_epochs=completed,
                           reason='Waiting for a saved epoch after requested boundary')
        if summary.get('partial') is not False:
            raise ValueError('Cannot transition a partial run')
        plan = prepare_plan(c, spec)
        if completed >= plan['old_training']['epochs']:
            return publish('already_complete', completed_epochs=completed,
                           reason='Training budget exhausted; no extra training started')
        verify_checkpoint(spec, plan, completed)
        infos = processes()
        launchers = [p for p in infos if matches(p, spec, spec['old_runtime'], True)]
        if len(launchers) != 1:
            denied = launcher_access_errors(spec)
            if denied:
                raise PermissionError('Cannot inspect the existing trainer through /proc: ' +
                    str(denied) + '. Run the transition watcher directly on the host, outside '
                    'bwrap/run_on_103.sh; retain the wrapper in the trainer child command.')
            raise ValueError('Expected exactly one live old Python torchrun launcher')
        launcher = launchers[0]
        validate_launcher_binding(launcher, plan)
        tracked = descendants(infos, launcher['pid'])
        archive_small(run, spec, plan)
        publish('stopping', old_trainer_identity=identity(launcher), tracked_processes=tracked)
        stop_exact(spec, launcher, tracked)
        publish('stopped')
        epoch = verify_checkpoint(spec, plan, completed)
        source_tables(spec, plan['old_training'])
        if read_json(spec['config']) != plan['old_config']:
            raise ValueError('Training config changed while stopping')
        record = checkpoint_transition(spec, plan, epoch)
        write_json(run / 'policy_transition.json', record)
        write_json(spec['config'], plan['new_config'])
        publish('launch_intent', resume_epoch=epoch, command=plan['command'])
        with (run / 'train.log').open('ab') as log:
            child = subprocess.Popen(plan['command'], cwd=plan['cwd'],
                env={**os.environ, **plan['environment']}, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ident = identity(process_info(child.pid))
        write_json(launch_file, {'created_at': now(), 'identity': ident, 'command': plan['command'],
            'cwd': plan['cwd'], 'environment': plan['environment'], 'resume_epoch': epoch,
            'transition_record_sha256': sha256(run / 'policy_transition.json'),
            'config_sha256': sha256(spec['config'])})
        return publish('launched', launch_identity=ident, resume_epoch=epoch)
    except Exception as error:
        return publish('failed', error=repr(error), manual_review_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    c = load_config(args.config)
    directory = Path(c['state_directory']); directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.transition.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This transition daemon is already running')
        while True:
            results = [tick_run(c, spec) for spec in c['runs']]
            write_json(directory / 'state.json', {'schema': SCHEMA, 'updated_at': now(), 'runs': results})
            print(json.dumps({'at': now(), 'runs': [{k: r.get(k) for k in
                  ('run', 'status', 'resume_epoch', 'error')} for r in results]}), flush=True)
            if args.once or all(r['status'] in ('running', 'failed', 'already_complete') for r in results):
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
