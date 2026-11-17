"""Shared one/two-slot GPU admission for evaluators, with inherited flock handles.

Use ``acquire_gpu_slot`` then ``available`` immediately before launch. Single-slot
acquisition retains the legacy pure-lock behavior; GPU probing is in available.
Keep the
returned handle open for the entire evaluation, including child processes via
``pass_fds``. All projects using one GPU must use the same ``gpu_lock`` path.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import weakref

_OWNED = weakref.WeakValueDictionary()
_EVALUATORS = {'evaluate.py', 'diagnose_bc_scores.py', 'export_worst_predictions.py',
               'evaluate_with_memory_limit.py'}


def _configuration(config):
    limit = config.get('max_concurrent_evaluations', 1)
    if type(limit) is not int or limit not in (1, 2):
        raise ValueError('max_concurrent_evaluations must be 1 or 2')
    if type(config.get('gpu_index')) is not int or config['gpu_index'] < 0:
        raise ValueError('gpu_index must be a nonnegative physical index')
    if type(config.get('minimum_free_mib')) is not int or config['minimum_free_mib'] < 1:
        raise ValueError('minimum_free_mib must be a positive integer')
    base = Path(config['gpu_lock']).expanduser().absolute()
    grace = config.get('gpu_startup_grace_seconds', 30)
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not 0 <= grace <= 300:
        raise ValueError('gpu_startup_grace_seconds must be in [0,300]')
    trainers = config.get('allowed_training_processes', [])
    if (not isinstance(trainers, list) or len(trainers) > 1
            or (trainers and limit != 1)):
        raise ValueError('Training sharing requires one evaluation slot and at most one pinned trainer')
    for identity in trainers:
        if (not isinstance(identity, dict) or set(identity) != {'pid', 'start_ticks', 'boot_id'}
                or type(identity['pid']) is not int or identity['pid'] < 1
                or not isinstance(identity['start_ticks'], str) or not identity['start_ticks'].isdigit()
                or not isinstance(identity['boot_id'], str) or not identity['boot_id']):
            raise ValueError('Pin each trainer by PID, start ticks, and boot ID')
    return limit, [base, Path(str(base) + '.slot1')], float(grace)


def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _owned(path):
    handle = _OWNED.get(str(path))
    return handle is not None and not handle.closed


def _occupied(paths):
    result = []
    for path in paths:
        if _owned(path):
            result.append(path)
            continue
        handle = _lock(path)
        if handle is None:
            result.append(path)
        else:
            handle.close()
    return result


def _query(arguments):
    return subprocess.run(['nvidia-smi', *arguments], check=True, capture_output=True,
                          text=True, timeout=15).stdout.strip()


def _gpu_state(config):
    fields = _query(['--id=' + str(config['gpu_index']), '--query-gpu=uuid,memory.free',
                     '--format=csv,noheader,nounits']).split(',')
    if len(fields) != 2:
        raise ValueError('Unexpected GPU query result')
    uuid, free = fields[0].strip(), int(fields[1])
    if not uuid:
        raise ValueError('Missing GPU UUID')
    pids = set()
    for line in _query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).splitlines():
        row = line.split(',')
        if len(row) != 2:
            raise ValueError('Unexpected compute-process query result')
        if row[0].strip() == uuid:
            pids.add(int(row[1]))
    return uuid, free, pids


def _self_pids():
    result = {os.getpid()}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('NSpid:'):
                result.update(map(int, line.split()[1:]))
    except (OSError, ValueError):
        pass
    return result


def _pinned_trainers(config):
    result = set()
    for identity in config.get('allowed_training_processes', []):
        try:
            stat = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if stat[0] != 'Z' and stat[19] == identity['start_ticks'] and boot == identity['boot_id']:
                result.add(identity['pid'])
        except (OSError, ValueError, IndexError):
            pass
    return result


def evaluation_command(command, config):
    """Limit only this evaluator's CUDA allocator, preserving its frozen source."""
    fraction = config.get('evaluation_memory_fraction')
    if fraction is None:
        return command
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not 0 < fraction <= 1):
        raise ValueError('evaluation_memory_fraction must be in (0,1]')
    script = next(i for i, part in enumerate(command[1:], 1) if not part.startswith('-'))
    wrapper = str(Path(__file__).with_name('evaluate_with_memory_limit.py').resolve())
    return command[:script] + [wrapper, '--fraction', str(fraction), '--'] + command[script:]


def _evaluation_process(pid):
    try:
        args = [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
    except (OSError, UnicodeError):
        return False
    if not args:
        return False
    if Path(args[0]).name in _EVALUATORS:
        return True
    # Only a direct Python script is accepted. An arbitrary training command
    # mentioning an evaluator path later in its arguments is not sufficient.
    if not Path(args[0]).name.startswith('python'):
        return False
    for arg in args[1:]:
        if arg in ('-c', '-m'):
            return False
        if not arg.startswith('-'):
            return Path(arg).name in _EVALUATORS
    return False


def _initializing(path, grace):
    try:
        metadata = json.loads(path.read_text())
        age = time.monotonic() - metadata['acquired_monotonic']
        return (metadata.get('schema') == 'lusnar_gpu_slot_v1'
                and metadata.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and 0 <= age < grace)
    except (OSError, ValueError, TypeError, KeyError):
        # The original single-slot worker does not write slot metadata.
        return False


def _available(config, state=None):
    limit, paths, grace = _configuration(config)
    occupied = _occupied(paths)
    owned = [path for path in occupied if _owned(path)]
    others = [path for path in occupied if path not in owned]
    try:
        uuid, free, pids = state or _gpu_state(config)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, f'GPU state unavailable: {type(error).__name__}: {error}', None
    prefix = f'GPU {config["gpu_index"]}: {free} MiB free'
    if free < config['minimum_free_mib']:
        return False, prefix + '; below the required free memory', uuid
    external = pids - _self_pids()
    if limit == 1:
        unexpected = external - _pinned_trainers(config)
        if others or unexpected:
            return False, prefix + '; another slot or compute process is active', uuid
        return True, prefix + ('; one pinned trainer may share' if external else ''), uuid
    if not owned and len(occupied) >= limit:
        return False, prefix + '; both evaluation slots are occupied', uuid
    if external:
        if not others:
            return False, prefix + '; GPU processes have no existing shared evaluation slot', uuid
        if any(not _evaluation_process(pid) for pid in external):
            return False, prefix + '; an unknown or training GPU process is active', uuid
    elif others:
        return False, prefix + '; waiting for the existing slot to initialize its GPU process', uuid
    if any(_initializing(path, grace) for path in others):
        return False, prefix + '; existing evaluation is within its GPU startup grace period', uuid
    return True, prefix + f'; shared evaluation limit {limit}', uuid


def available(config):
    """Return (ready, reason, GPU UUID), without occupying a slot.

    A caller's own CUDA context is ignored, including host/namespace PID aliases.
    Free-memory accounting still includes that context. Unknown processes are
    never admitted merely because their count is below the slot limit.
    """
    return _available(config)


def group_configs(config):
    """Expand a dedicated GPU set while retaining the shared slot paths."""
    indices = config.get('sharded_gpus')
    if (not isinstance(indices, list) or len(indices) < 2 or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 for i in indices)
            or indices[0] != config['gpu_index'] or config.get('max_concurrent_evaluations', 1) != 1
            or config.get('allowed_training_processes')):
        raise ValueError('Sharded evaluation requires distinct dedicated GPUs and one slot per GPU')
    base = Path(config['gpu_lock'])
    if base.name != f'.periodic_gpu_{indices[0]}.lock':
        raise ValueError('Sharded GPUs must use shared .periodic_gpu_INDEX.lock paths')
    return [{**config, 'gpu_index': i, 'gpu_lock': str(base.with_name(f'.periodic_gpu_{i}.lock'))}
            for i in indices]


def group_available(config):
    replies = [available(item) for item in group_configs(config)]
    return all(r[0] for r in replies), '; '.join(r[1] for r in replies), replies[0][2]


def acquire_gpu_group(config):
    """Reserve every GPU or release the entire partial reservation."""
    handles = []
    try:
        for item in group_configs(config):
            handle = acquire_gpu_slot(item)
            if handle is None:
                for previous in handles: previous.close()
                return None
            handles.append(handle)
        return handles
    except Exception:
        for handle in handles: handle.close()
        raise


def acquire_gpu_slot(config):
    """Atomically admit and lock one slot; return an open file or None.

    Slot 0 is the legacy gpu_lock; slot 1 appends '.slot1'. A short guard lock
    serializes admission, and a held slot without a visible GPU process blocks
    the second admission until initialization. A bounded startup grace period
    also protects against a just-created, still almost-empty CUDA context.
    """
    limit, paths, _ = _configuration(config)
    if any(_owned(path) for path in paths):
        return None  # One evaluation process must not reserve both slots.
    guard = _lock(Path(str(paths[0]) + '.startup'))
    if guard is None:
        return None
    try:
        if limit == 2:
            ready, _, _ = _available(config)
            if not ready:
                return None
        occupied = _occupied(paths)
        if limit == 1 and paths[1] in occupied:
            return None
        choices = [path for path in paths[:limit] if path not in occupied]
        if not choices:
            return None
        handle = _lock(choices[0])
        if handle is None:
            return None
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({'schema': 'lusnar_gpu_slot_v1', 'pid': os.getpid(),
                       'acquired_monotonic': time.monotonic(),
                       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       'max_concurrent_evaluations': limit}, handle)
            handle.flush()
        except Exception:
            handle.close()
            raise
        _OWNED[str(choices[0])] = handle
        return handle
    finally:
        guard.close()
