#!/usr/bin/env python3
"""Stop explicitly bound live trainers after a fully committed epoch.

No trainer restart, source edits, scheduler changes, or evaluator signals. The
controller first freezes only the bound process tree, verifies the committed
checkpoint, preserves final.pt, then terminates that tree. Original checkpoint
bytes/config/history stay intact; a sidecar documents this external stop.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import platform
import time

from periodic_evaluate import (acquire_lock, alive, checkpoint_metadata, committed_run,
    fingerprint, now, process_identity, read_json, sha256, validate_validation, write_json)

ROOT = Path(__file__).resolve().parents[1]


def argv_for(pid):
    return [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]


def send(identity, sig):
    """Use a pidfd and start/boot identity, never a bare reused PID."""
    try:
        fd = (os.pidfd_open(identity['pid']) if hasattr(os, 'pidfd_open')
              else pidfd_syscall(434, identity['pid'], 0))
    except ProcessLookupError:
        return False
    try:
        if not alive(identity):
            return False
        if hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(fd, sig)
        else:
            pidfd_syscall(424, fd, sig, 0, 0)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def pidfd_syscall(number, *args):
    # Some Conda Python builds omit these wrappers although Linux supports them.
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('This platform needs Python pidfd support')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(x) for x in args))
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def child_identities(pid):
    children = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[1]) == pid:
                identity = process_identity(int(p.name))
                if identity:
                    children.append(identity)
        except (OSError, ValueError, IndexError):
            continue
    return children


def freeze_tree(root, receipt_path, receipt):
    queue, seen, frozen = [root], set(), []
    while queue:
        identity = queue.pop(0)
        key = (identity['pid'], identity['start_ticks'])
        if key in seen or not alive(identity):
            continue
        seen.add(key)
        # Persist before signaling, so a restarted controller can recover a pause.
        frozen.append(identity)
        receipt.update(phase='freezing', process_identities=frozen)
        write_json(receipt_path, receipt)
        send(identity, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while alive(identity):
            state = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0]
            if state in ('T', 't'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Bound process did not acknowledge pause')
            time.sleep(0.01)
        queue.extend(child_identities(identity['pid']))
    return frozen


def terminate_tree(identities, grace_seconds=15):
    # TERM is pending while stopped; CONT lets the normal handler terminate.
    for identity in reversed(identities):
        send(identity, signal.SIGTERM)
    for identity in identities:
        send(identity, signal.SIGCONT)
    deadline = time.monotonic() + grace_seconds
    while any(alive(p) for p in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    for identity in identities:
        if alive(identity):
            send(identity, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while any(alive(p) for p in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    if any(alive(p) for p in identities):
        raise RuntimeError('Bound training processes have not all exited')


def bind_run(run, target):
    run = Path(run).resolve()
    configuration = read_json(run / 'run_config.json')
    launch = read_json(run / 'launch.json')
    identity = process_identity(launch['pid'])
    if not identity:
        raise ValueError('Trainer is not alive at enrollment: ' + str(run))
    argv = argv_for(identity['pid'])
    trainer = str(run / 'runtime/code/train_distributed.py')
    config_path = str(Path(launch['config']).resolve())
    if (trainer not in argv or '--config' not in argv
            or argv[argv.index('--config') + 1] != config_path
            or Path(configuration['output']).resolve() != run):
        raise ValueError('Launcher does not belong to the requested run')
    if (type(target) is not int or target < 1 or target > configuration['epochs']
            or configuration.get('early_stopping_patience') != 0):
        raise ValueError('Invalid fixed epoch cap or early stopping is enabled')
    summary = read_json(run / 'summary.json') if (run / 'summary.json').exists() else {}
    if summary.get('completed_epochs', 0) >= target:
        raise ValueError('Enroll before the requested epoch finishes')
    return {'run': str(run), 'target_epoch': target, 'trainer_identity': identity,
            'trainer_argv': argv, 'configuration_sha256': fingerprint(configuration),
            'original_schedule_epochs': configuration['epochs']}


def stop_committed(binding, metadata_reader=checkpoint_metadata):
    run, target = Path(binding['run']), binding['target_epoch']
    receipt_path = run / 'epoch_limit_stop.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else {}
    if receipt.get('phase') == 'complete':
        return receipt
    configuration = read_json(run / 'run_config.json')
    if fingerprint(configuration) != binding['configuration_sha256']:
        raise ValueError('Bound training configuration changed')
    # Recover a monitor interrupted before its checkpoint validation completed.
    if receipt.get('phase') == 'freezing':
        for identity in receipt.get('process_identities', []):
            send(identity, signal.SIGCONT)
        receipt = {}
    if receipt.get('phase') != 'validated':
        identity = binding['trainer_identity']
        if not alive(identity) or argv_for(identity['pid']) != binding['trainer_argv']:
            raise ValueError('Bound trainer identity no longer matches; no signal sent')
        receipt = {'schema': 'vit_external_epoch_stop_v1', 'requested_epoch': target,
                   'started_at': now(), 'binding': binding, 'phase': 'freezing'}
        frozen = []
        try:
            frozen = freeze_tree(identity, receipt_path, receipt)
            committed = committed_run(run)
            if committed is None:
                raise ValueError('No committed epoch')
            summary, current_config, rows = committed
            if summary['completed_epochs'] != target:
                raise ValueError('Requested committed epoch is not current')
            if fingerprint(current_config) != binding['configuration_sha256']:
                raise ValueError('Configuration changed while freezing')
            validate_validation(read_json(run / f'validation_epoch_{target:03d}.json'), target,
                                rows[target]['validation_miou'], summary['data_identity']['val'])
            saved = metadata_reader(run / 'last.pt')
            if (saved.get('format') != 'lusnar_vit_checkpoint_v1' or saved.get('epoch') != target
                    or saved.get('partial') is not False or saved.get('train_config') != current_config
                    or saved.get('data_identity') != summary['data_identity']
                    or saved.get('model_config') != summary['model_config']
                    or saved.get('best_epoch') != summary['best_epoch']
                    or saved.get('best_miou') != summary['best_miou']):
                raise ValueError('last.pt does not match the fully committed epoch')
            if not (run / 'best.pt').is_file():
                raise ValueError('Validation BEST is missing')
            final, last = run / 'final.pt', run / 'last.pt'
            digest = sha256(last)
            if final.exists() and sha256(final) != digest:
                raise ValueError('Refusing to replace a different final checkpoint')
            if not final.exists():
                temporary = run / '.epoch_limit_final.pt'
                temporary.unlink(missing_ok=True)
                os.link(last, temporary)
                temporary.replace(final)
            archive = run / f'summary_before_epoch_limit_{target:03d}.json'
            if not archive.exists():
                write_json(archive, summary)
            receipt.update(phase='validated', validated_at=now(), process_identities=frozen,
                           checkpoint_epoch=target, checkpoint_sha256=digest,
                           final_checkpoint=str(final), summary_before_stop=summary)
            write_json(receipt_path, receipt)
        except BaseException:
            # No failed validation may leave the live training tree paused.
            for member in frozen or receipt.get('process_identities', []):
                send(member, signal.SIGCONT)
            receipt.update(phase='validation_failed', failed_at=now())
            write_json(receipt_path, receipt)
            raise
    terminate_tree(receipt['process_identities'])
    summary = receipt['summary_before_stop']
    reason = f'User-requested stop after committed epoch {target}; original learning-rate schedule retained'
    stopped_at = now()
    final_record = {'schema': 'vit_external_final_checkpoint_v1', 'epoch': target,
                    'checkpoint': receipt['final_checkpoint'], 'sha256': receipt['checkpoint_sha256'],
                    'created_at': stopped_at, 'stop_reason': reason,
                    'checkpoint_bytes_unchanged': True, 'embedded_checkpoint_status': 'training',
                    'original_schedule_epochs': binding['original_schedule_epochs']}
    write_json(run / 'final.json', final_record)
    write_json(run / 'summary.json', {**summary, 'status': 'complete', 'stop_reason': reason,
               'termination_kind': 'user_requested_epoch_limit', 'requested_stop_epoch': target,
               'configured_schedule_epochs': binding['original_schedule_epochs'],
               'stopped_at': stopped_at, 'final_checkpoint': receipt['final_checkpoint']})
    receipt.update(phase='complete', completed_at=stopped_at, stop_reason=reason,
                   evaluation_processes_untouched=True)
    write_json(receipt_path, receipt)
    return receipt


def tick(binding):
    run = Path(binding['run'])
    receipt_path = run / 'epoch_limit_stop.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else {}
    if receipt.get('phase') in ('validated', 'complete'):
        stopped = stop_committed(binding)
        return {'status': 'complete', 'completed_epochs': stopped['checkpoint_epoch']}
    summary_path = run / 'summary.json'
    summary = read_json(summary_path) if summary_path.exists() else {}
    completed = summary.get('completed_epochs', 0)
    if completed >= binding['target_epoch'] or receipt.get('phase') == 'freezing':
        stopped = stop_committed(binding)
        return {'status': 'complete', 'completed_epochs': stopped['checkpoint_epoch']}
    if not alive(binding['trainer_identity']):
        raise ValueError('Trainer exited before epoch cap; controller will not restart it')
    return {'status': 'watching', 'completed_epochs': completed,
            'target_epoch': binding['target_epoch'], 'trainer_alive': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    if config.get('schema') != 'vit_stop_at_epoch_v1' or config.get('poll_seconds') != 1:
        raise ValueError('Expected vit_stop_at_epoch_v1 with a one-second poll')
    root = Path(config['project']).resolve()
    directory = Path(config['state_directory']).resolve()
    if root != ROOT or not directory.is_relative_to(root / 'result'):
        raise ValueError('Controller must operate on its own project results')
    runs = [Path(p).resolve() for p in config['runs']]
    if not runs or len(set(runs)) != len(runs) or any(not p.is_relative_to(root / 'result') for p in runs):
        raise ValueError('Invalid run paths')
    lock = acquire_lock(directory / '.lock')
    if lock is None:
        raise RuntimeError('Another epoch-stop controller is already running')
    try:
        path = directory / 'bindings.json'
        if path.exists():
            data = read_json(path)
            if data['config_sha256'] != fingerprint(config) or data['host'] != socket.gethostname():
                raise ValueError('Controller enrollment identity changed')
        else:
            data = {'config_sha256': fingerprint(config), 'host': socket.gethostname(),
                    'created_at': now(), 'runs': [bind_run(r, config['stop_after_epoch']) for r in runs]}
            write_json(path, data)
        for binding in data['runs']:
            write_json(Path(binding['run']) / 'epoch_limit_request.json', {
                'target_epoch': binding['target_epoch'], 'requested_at': data['created_at'],
                'controller_state': str(directory / 'state.json'),
                'original_schedule_epochs': binding['original_schedule_epochs'],
                'training_restart': False, 'evaluation_every_epochs': 5,
                'best_selection': 'validation_miou', 'process_binding': binding['trainer_identity']})
        last_print, previous = 0.0, None
        while True:
            states = {}
            for binding in data['runs']:
                try:
                    states[binding['run']] = tick(binding)
                except Exception as error:
                    states[binding['run']] = {'status': 'error', 'error': f'{type(error).__name__}: {error}'}
            status = 'complete' if all(v['status'] == 'complete' for v in states.values()) else (
                     'error' if any(v['status'] == 'error' for v in states.values()) else 'watching')
            document = {'schema': 'vit_epoch_stop_state_v1', 'updated_at': now(), 'status': status,
                        'controller_identity': process_identity(os.getpid()), 'host': socket.gethostname(),
                        'target_epoch': config['stop_after_epoch'], 'runs': states}
            write_json(directory / 'state.json', document)
            if states != previous or time.monotonic() - last_print >= 60:
                print(json.dumps(document, ensure_ascii=False), flush=True)
                previous, last_print = states, time.monotonic()
            if args.once or status == 'complete':
                return int(status == 'error')
            time.sleep(config['poll_seconds'])
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
