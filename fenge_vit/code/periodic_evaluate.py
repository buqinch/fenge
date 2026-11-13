#!/usr/bin/env python3
"""Queue immutable validation-best tests at the configured epoch interval.

This process observes training files only. It never changes or signals a trainer.
Each worker holds a GPU lock inherited by its evaluator, including if the worker
dies. JSON status and Linux PID/start-time identities allow restart recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = {'Moon_2': 1597, 'Moon_4': 1320, 'Moon_9': 2087}
CLASSES = ('regolith', 'rock', 'crater', 'mountain', 'sky')
DONE = {'complete', 'reused', 'failed', 'missed'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.periodic-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
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


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': stat[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def alive(identity):
    return bool(identity and process_identity(identity['pid']) == identity)


def job_processes(job):
    """Cover the spawn-to-status-write crash window without matching by PID alone."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            args = [arg.decode() for arg in args if arg]
            is_worker = ('--worker' in args and str(job) in args
                         and any(Path(arg).name == 'periodic_evaluate.py' for arg in args))
            is_eval = ('--output' in args and str(job / 'test_metrics.json') in args
                       and any(Path(arg).name == 'evaluate.py' for arg in args))
            identity = process_identity(int(entry.name))
            if (is_worker or is_eval) and identity:
                found.append(identity)
        except (OSError, UnicodeError):
            continue
    return found


def acquire_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, 'a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def settings(config):
    return {'split': 'test', 'tta': 'multiscale_hflip', 'tta_sizes': [1024, 768],
            'tta_weights': [0.5, 0.5], 'amp': 'bf16', 'batch_size': 4, 'workers': 4,
            'test_manifest': config['test_manifest'],
            'test_manifest_sha256': sha256(config['test_manifest']),
            'evaluate_sha256': sha256(ROOT / 'code/evaluate.py'), 'save_images': False}


def load_config(path):
    config = read_json(path)
    if config.get('schema') != 'vit_periodic_test_v1':
        raise ValueError('Expected vit_periodic_test_v1 configuration')
    config['project'] = str(ROOT)
    for key in ('state_directory', 'test_manifest'):
        value = Path(config[key])
        config[key] = str((ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    state = Path(config['state_directory'])
    if not state.is_relative_to(ROOT / 'result') or state == ROOT / 'result':
        raise ValueError('State must be strictly inside project/result')
    if type(config['gpu_index']) is not int or config['gpu_index'] < 0:
        raise ValueError('gpu_index must be a nonnegative physical GPU index')
    for key, default in [('every_epochs', 5), ('poll_seconds', 10), ('minimum_free_mib', 10000)]:
        config.setdefault(key, default)
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for key in ('periodic_evaluation', 'final_evaluation'):
        config.setdefault(key, True)
        if type(config[key]) is not bool:
            raise ValueError(key + ' must be a boolean')
    validate_retention_threshold(config.get('retain_above_score'))
    paths = []
    for value in config['runs']:
        run = (ROOT / value).resolve()
        if not run.is_relative_to(ROOT / 'result') or run == ROOT / 'result':
            raise ValueError('Runs must be strictly inside project/result')
        paths.append(str(run))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Expected distinct run directories')
    config['runs'] = paths
    config['python'] = str(Path(config.get('python', sys.executable)).resolve())
    # Independent projects evaluating on one physical GPU must share this lock.
    config['gpu_lock'] = str(Path(config.get('gpu_lock', ROOT / 'result' /
                              f'.periodic_gpu_{config["gpu_index"]}.lock')).expanduser().resolve())
    config['evaluation'] = settings(config)
    return config


def gpu_available(config):
    if config.get('remote'):
        from remote_evaluate import probe_remote
        return probe_remote(config)
    def query(arguments):
        result = subprocess.run(['nvidia-smi', *arguments], check=True, capture_output=True,
                                text=True, timeout=15)
        return result.stdout.strip()
    row = query(['--id=' + str(config['gpu_index']), '--query-gpu=uuid,memory.free',
                 '--format=csv,noheader,nounits']).split(',')
    uuid, free = row[0].strip(), int(row[1])
    if free < config['minimum_free_mib']:
        return False, f'GPU {config["gpu_index"]}: {free} MiB free', uuid
    processes = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        return False, f'GPU {config["gpu_index"]} has another compute process', uuid
    return True, f'GPU {config["gpu_index"]}: {free} MiB free', uuid


def assert_close(a, b, description):
    if (isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a)
            or not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(description)


def committed_run(run):
    """Ignore history appended ahead of the trainer's atomic summary publication."""
    summary_path = run / 'summary.json'
    if not summary_path.exists():
        return None
    summary, configuration = read_json(summary_path), read_json(run / 'run_config.json')
    expected_count = summary.get('data_identity', {}).get('val', {}).get('image_count')
    if (summary.get('format') != 'lusnar_vit_run_v1' or summary.get('partial') is not False
            or type(expected_count) is not int or expected_count < 1
            or summary.get('validation_images') != expected_count
            or summary.get('checkpoint_selection_uses_test') is not False
            or summary.get('official_test_used_for_selection') is not False):
        raise ValueError('Expected full validation-only ViT training')
    completed = summary['completed_epochs']
    if type(completed) is not int or completed < 1:
        raise ValueError('Invalid completed_epochs')
    rows = {}
    for line in (run / 'history.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        row = json.loads(line)
        if row['epoch'] <= completed:
            if row['epoch'] in rows:
                raise ValueError('Duplicate committed history epoch')
            rows[row['epoch']] = row
    if set(rows) != set(range(1, completed + 1)):
        raise ValueError('Committed summary/history are incomplete')
    best_score, best_epoch = -math.inf, None
    for epoch, row in sorted(rows.items()):
        score = row['validation_miou']
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError('Invalid validation mIoU')
        if score > best_score:
            best_score, best_epoch = score, epoch
        if (row.get('partial') is not False or row.get('checkpoint_selection_uses_test') is not False
                or row['best_epoch'] != best_epoch):
            raise ValueError('History does not use first strict validation maximum')
        assert_close(row['best_miou'], best_score, 'History best score mismatch')
    if best_epoch != summary['best_epoch']:
        raise ValueError('Summary best epoch mismatch')
    assert_close(summary['best_miou'], best_score, 'Summary best score mismatch')
    return summary, configuration, rows


def validate_validation(document, epoch, score, identity):
    expected_count = identity.get('image_count')
    expected_sources = identity.get('unique_source_count', expected_count)
    if type(expected_count) is not int or expected_count < 1 or type(expected_sources) is not int or expected_sources < 1:
        raise ValueError('Validation identity is missing positive coverage counts')
    if (document.get('epoch') != epoch or document.get('partial') is not False
            or document.get('image_count') != expected_count
            or document.get('validation_unique_source_count') != expected_sources
            or document.get('checkpoint_selection_uses_test') is not False
            or document.get('data_identity') != identity):
        raise ValueError('Full fixed-validation identity/coverage mismatch')
    assert_close(document.get('miou'), score, 'Validation score differs from history')


def check_sources(configuration):
    sources = configuration.get('source_sha256')
    if not sources or 'code/train.py' not in sources:
        raise ValueError('Training source hashes missing')
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT / 'code') or sha256(path) != expected:
            raise ValueError('Training/evaluation source changed: ' + relative)


def snapshot(source, destination):
    temporary = destination.with_name('.snapshot-' + str(os.getpid()) + '.pt')
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_metadata(path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    return {key: saved.get(key) for key in ('format', 'epoch', 'best_epoch', 'best_miou',
            'partial', 'metrics', 'data_identity', 'train_config', 'model_config')}


def validate_snapshot(saved, trigger, configuration, summary):
    if (saved.get('format') != 'lusnar_vit_checkpoint_v1' or saved.get('partial') is not False
            or saved.get('epoch') != trigger['checkpoint_epoch']
            or saved.get('best_epoch') != trigger['checkpoint_epoch']
            or saved.get('train_config') != configuration
            or saved.get('data_identity') != summary['data_identity']
            or saved.get('model_config') != summary['model_config']):
        raise ValueError('Checkpoint metadata/configuration/data identity mismatch')
    assert_close(saved['best_miou'], trigger['validation_best_miou'], 'Checkpoint best score mismatch')
    validate_validation(saved['metrics'], trigger['checkpoint_epoch'],
                        trigger['validation_best_miou'], summary['data_identity']['val'])


def validate_metrics(document, trigger):
    if (document.get('partial') is not False or document.get('checkpoint_partial_training') is not False
            or document.get('split') != 'test' or document.get('image_count') != 5004
            or document.get('validation_unique_source_count') != 5004
            or document.get('official_test_scope') is not True
            or document.get('test_scope') != 'complete_official_test'
            or document.get('checkpoint_sha256') != trigger['checkpoint_sha256']
            or document.get('checkpoint_epoch') != trigger['checkpoint_epoch']):
        raise ValueError('Incomplete test or wrong checkpoint provenance')
    if (document.get('tta') != 'multiscale_hflip' or document.get('tta_sizes') != [1024, 768]
            or document.get('tta_input_sizes') != [1024, 768]
            or document.get('tta_scale_weights') != [0.5, 0.5]
            or document.get('tta_view_weights') != [0.25] * 4
            or document.get('tta_forward_calls_per_batch') != 4):
        raise ValueError('Test did not use equal 1024/768 horizontal-flip TTA')
    identity = document.get('data_identity', {})
    if (identity.get('split') != 'test' or identity.get('partial') is not False
            or identity.get('manifest_sha256') != trigger['evaluation']['test_manifest_sha256']):
        raise ValueError('Test manifest identity mismatch')
    scenes = document.get('scenes', {})
    if set(scenes) != set(SCENES):
        raise ValueError('Unexpected test scenes')
    scores = {}
    pooled = [[0] * 5 for _ in range(5)]
    for scene, count in SCENES.items():
        row = scenes[scene]
        matrix = row.get('confusion_matrix_gt_rows_pred_cols', [])
        if (row.get('image_count') != count or len(matrix) != 5
                or any(len(r) != 5 or any(type(x) is not int or x < 0 for x in r) for r in matrix)
                or set(row.get('class_iou', {})) != set(CLASSES)):
            raise ValueError('Invalid scene count, classes, or integer confusion matrix')
        # The test loader rejects all mask IDs outside 0..4, so all native
        # 1024-square pixels must be represented; there is no ignore-label area.
        if sum(map(sum, matrix)) != count * 1024 * 1024:
            raise ValueError('Scene confusion matrix does not cover every native test pixel')
        for i in range(5):
            for j in range(5):
                pooled[i][j] += matrix[i][j]
        present, ious = [], []
        for index, name in enumerate(CLASSES):
            gt, predicted = sum(matrix[index]), sum(r[index] for r in matrix)
            if gt:
                iou = matrix[index][index] / (gt + predicted - matrix[index][index])
                assert_close(row['class_iou'][name], iou, 'Class IoU mismatch')
                present.append(name)
                ious.append(iou)
            elif row['class_iou'][name] is not None:
                raise ValueError('Absent GT class must not participate in scene mIoU')
        if row.get('gt_present_classes') != present or not ious:
            raise ValueError('Ground-truth-present class set mismatch')
        scores[scene] = sum(ious) / len(ious)
        assert_close(row.get('miou'), scores[scene], 'Scene mIoU mismatch')
    if document.get('confusion_matrix_gt_rows_pred_cols') != pooled:
        raise ValueError('Pooled confusion matrix differs from the sum of scene matrices')
    pooled_ious, pooled_present = [], []
    if set(document.get('class_iou', {})) != set(CLASSES):
        raise ValueError('Pooled class names mismatch')
    for index, name in enumerate(CLASSES):
        gt, predicted = sum(pooled[index]), sum(r[index] for r in pooled)
        if gt:
            iou = pooled[index][index] / (gt + predicted - pooled[index][index])
            assert_close(document['class_iou'][name], iou, 'Pooled class IoU mismatch')
            pooled_ious.append(iou)
            pooled_present.append(name)
        elif document['class_iou'][name] is not None:
            raise ValueError('Pooled absent GT class must not contribute to mIoU')
    if document.get('gt_present_classes') != pooled_present:
        raise ValueError('Pooled GT-present classes mismatch')
    pooled_miou = sum(pooled_ious) / len(pooled_ious)
    assert_close(document.get('miou'), pooled_miou, 'Pooled mIoU mismatch')
    assert_close(document.get('pooled_miou'), pooled_miou, 'Explicit pooled mIoU mismatch')
    if document.get('weighted_score_weights') != {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}:
        raise ValueError('Official scene weights differ from 1:1:2')
    score = (scores['Moon_2'] + scores['Moon_4'] + 2 * scores['Moon_9']) / 4
    assert_close(document.get('weighted_score'), score, 'Weighted score mismatch')
    return {'weighted_score': score, 'scene_miou': scores, 'image_count': 5004}


def set_status(job, status, **fields):
    path = job / 'status.json'
    previous = read_json(path) if path.exists() else {}
    value = {**previous, 'status': status, 'updated_at': now(), **fields}
    write_json(path, value)
    return value


def validate_retention_threshold(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('retain_above_score must be null or a finite number in [0, 1]')
    return value


def finalize_checkpoint(job, trigger, verified, retain_above_score=None, reused_from=None):
    """Protect an exact tested checkpoint before normal temporary-file cleanup."""
    threshold = validate_retention_threshold(retain_above_score if retain_above_score is not None
                                            else trigger.get('retain_above_score'))
    checkpoint, marker = job / 'best.pt', job / 'score_milestone.json'
    if marker.exists():
        recorded = read_json(marker)
        if (recorded.get('checkpoint_sha256') != trigger['checkpoint_sha256']
                or recorded.get('metrics_sha256') != sha256(job / 'test_metrics.json')
                or not checkpoint.exists() or sha256(checkpoint) != trigger['checkpoint_sha256']):
            raise ValueError('Retained milestone checkpoint or metrics no longer match')
        return recorded
    if threshold is None or verified['weighted_score'] <= threshold:
        checkpoint.unlink(missing_ok=True)
        return None
    if not checkpoint.exists():
        run = Path(trigger['run'])
        candidates = ([Path(reused_from) / 'best.pt'] if reused_from else [])
        candidates += [run / 'best.pt', run / 'last.pt']
        candidates += sorted((run / 'periodic_test').glob('*/best.pt'))
        for source in candidates:
            if source != checkpoint and source.exists() and sha256(source) == trigger['checkpoint_sha256']:
                snapshot(source, checkpoint)
                break
    if not checkpoint.exists() or sha256(checkpoint) != trigger['checkpoint_sha256']:
        raise ValueError('Threshold exceeded but the exact tested checkpoint could not be retained')
    recorded = {'schema': 'vit_test_score_milestone_v1', 'recorded_at': now(),
                'host': socket.gethostname(), 'metric': 'weighted_score', 'comparison': '>',
                'threshold': threshold, **verified, 'trigger_epoch': trigger['trigger_epoch'],
                'checkpoint_epoch': trigger['checkpoint_epoch'],
                'checkpoint_sha256': trigger['checkpoint_sha256'],
                'checkpoint': str(checkpoint), 'metrics': str(job / 'test_metrics.json'),
                'metrics_sha256': sha256(job / 'test_metrics.json'),
                'checkpoint_selection_uses_test': False, 'training_process_management': False}
    write_json(marker, recorded)
    return recorded


def complete_job(job, trigger, reused_from=None, retain_above_score=None):
    verified = validate_metrics(read_json(job / 'test_metrics.json'), trigger)
    milestone = finalize_checkpoint(job, trigger, verified, retain_above_score, reused_from)
    state = set_status(job, 'reused' if reused_from else 'complete', verified=verified,
                       completed_at=now(), reused_from=reused_from,
                       checkpoint_sha256=trigger['checkpoint_sha256'],
                       checkpoint_epoch=trigger['checkpoint_epoch'], score_milestone=milestone)
    return state


def prepare_job(run, boundary, summary, configuration, rows, config, final=False):
    job = run / 'periodic_test' / (f'final_epoch_{boundary:03d}' if final else f'epoch_{boundary:03d}')
    if (job / 'status.json').exists():
        return job
    job.mkdir(parents=True, exist_ok=True)
    row = rows[boundary]
    expected_epoch, expected_score = row['best_epoch'], row['best_miou']
    validate_validation(read_json(run / f'validation_epoch_{boundary:03d}.json'), boundary,
                        row['validation_miou'], summary['data_identity']['val'])
    validate_validation(read_json(run / f'validation_epoch_{expected_epoch:03d}.json'), expected_epoch,
                        expected_score, summary['data_identity']['val'])
    check_sources(configuration)
    trigger = {'schema': 'vit_periodic_trigger_v1', 'run': str(run), 'trigger_epoch': boundary,
               'trigger_kind': ('final' if final else 'every_three_epochs' if config['every_epochs'] == 3
                                else f'every_{config["every_epochs"]}_epochs'),
               'every_epochs': config['every_epochs'], 'created_at': now(),
               'checkpoint_epoch': expected_epoch, 'validation_best_miou': expected_score,
               'configuration_sha256': fingerprint(configuration),
               'validation_identity': summary['data_identity']['val'],
               'evaluation': config['evaluation'], 'checkpoint_selection_uses_test': False}
    if config.get('retain_above_score') is not None:
        trigger['retain_above_score'] = validate_retention_threshold(config['retain_above_score'])
    # A retained older snapshot can recover a boundary after best.pt advances.
    candidates = [run / 'best.pt', run / 'last.pt']
    for earlier in sorted((run / 'periodic_test').glob('*/trigger.json')):
        if earlier.parent == job:
            continue
        other = read_json(earlier)
        comparable = ('checkpoint_epoch', 'validation_best_miou', 'configuration_sha256',
                      'validation_identity', 'evaluation')
        if not all(other.get(key) == trigger[key] for key in comparable):
            continue
        other_state = read_json(earlier.parent / 'status.json')
        if other_state['status'] in ('complete', 'reused'):
            trigger['checkpoint_sha256'] = other['checkpoint_sha256']
            validate_metrics(read_json(earlier.parent / 'test_metrics.json'), trigger)
            write_json(job / 'trigger.json', trigger)
            shutil.copyfile(earlier.parent / 'test_metrics.json', job / 'test_metrics.json')
            complete_job(job, trigger, reused_from=str(earlier.parent))
            return job
        if (earlier.parent / 'best.pt').exists():
            candidates.insert(0, earlier.parent / 'best.pt')
    mismatches = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        snapshot(candidate, job / 'best.pt')
        saved = checkpoint_metadata(job / 'best.pt')
        if saved.get('epoch') != expected_epoch:
            mismatches.append(saved.get('epoch'))
            (job / 'best.pt').unlink(missing_ok=True)
            continue
        validate_snapshot(saved, trigger, configuration, summary)
        trigger['checkpoint_sha256'] = sha256(job / 'best.pt')
        trigger['snapshot_source'] = str(candidate)
        write_json(job / 'trigger.json', trigger)
        set_status(job, 'queued', queued_at=now(), checkpoint_epoch=expected_epoch)
        return job
    # A newer atomic best may precede its new summary. Retry that short window.
    if summary['completed_epochs'] == boundary:
        (job / 'best.pt').unlink(missing_ok=True)
        return job
    write_json(job / 'trigger.json', trigger)
    set_status(job, 'missed', reason='Historical validation-best checkpoint is no longer available; '
               'a later checkpoint was not substituted', expected_checkpoint_epoch=expected_epoch,
               observed_checkpoint_epochs=mismatches)
    return job


def evaluator_command(config, job):
    return [config['python'], '-u', str(ROOT / 'code/evaluate.py'), '--checkpoint', str(job / 'best.pt'),
            '--split', 'test', '--test-manifest', config['test_manifest'], '--device', 'cuda:0',
            '--amp', 'bf16', '--batch-size', '4', '--workers', '4', '--tta', 'multiscale_hflip',
            '--tta-sizes', '1024', '768', '--tta-weights', '0.5', '0.5',
            '--output', str(job / 'test_metrics.json')]


def worker(job, probe=gpu_available):
    """Runs in the daemon's mount namespace; never manipulates a trainer process."""
    job = Path(job).resolve()
    job_lock = acquire_lock(job / '.worker.lock')
    if job_lock is None:
        return 0
    gpu_lock = None
    try:
        if read_json(job / 'status.json')['status'] in DONE:
            return 0
        launch = read_json(job / 'launch.json')
        config, trigger = launch['configuration'], read_json(job / 'trigger.json')
        if not config.get('remote'):
            gpu_lock = acquire_lock(config['gpu_lock'])
            if gpu_lock is None:
                set_status(job, 'queued', waiting_reason='GPU evaluation lock is held')
                return 0
        # Reconnecting to an existing remote task must work while it owns the GPU.
        ready, reason, uuid = (True, 'Remote worker checks its shared GPU lock', None) if config.get('remote') else probe(config)
        if not ready:
            set_status(job, 'queued', waiting_reason=reason)
            return 0
        if sha256(job / 'best.pt') != trigger['checkpoint_sha256']:
            raise ValueError('Snapshot SHA changed before evaluation')
        training_configuration = read_json(Path(trigger['run']) / 'run_config.json')
        if fingerprint(training_configuration) != trigger['configuration_sha256']:
            raise ValueError('Training configuration changed after snapshot selection')
        check_sources(training_configuration)
        if settings(config) != trigger['evaluation']:
            raise ValueError('Evaluator source or test manifest changed after queueing')
        set_status(job, 'running', worker_identity=process_identity(os.getpid()),
                   started_at=now(), physical_gpu=config['gpu_index'], gpu_uuid=uuid)
        if config.get('remote'):
            from remote_evaluate import run_remote
            set_status(job, 'running', execution_host=config['remote']['host'])
            returncode = run_remote(config, job, trigger)
            if returncode == 75:
                set_status(job, 'queued', waiting_reason='Remote GPU is occupied; waiting for its shared lock')
                return 0
            if returncode:
                raise RuntimeError(f'Remote evaluator exited with status {returncode}; see remote_execution.json/test.log')
            complete_job(job, trigger, retain_above_score=config.get('retain_above_score'))
            return 0
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'PYTHONUNBUFFERED': '1'}
        with (job / 'test.log').open('a') as output:
            child = subprocess.Popen(evaluator_command(config, job), cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(gpu_lock.fileno(), job_lock.fileno()))
            set_status(job, 'running', evaluator_identity=process_identity(child.pid))
            returncode = child.wait()
        if returncode:
            raise RuntimeError(f'Evaluator exited with status {returncode}; see test.log')
        complete_job(job, trigger, retain_above_score=config.get('retain_above_score'))
        return 0
    except Exception as error:
        set_status(job, 'failed', reason=f'{type(error).__name__}: {error}', failed_at=now())
        print(f'{now()} {job}: {error}', flush=True)
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        job_lock.close()


class Monitor:
    def __init__(self, config):
        self.config = config
        self.directory = Path(config['state_directory'])
        self.directory.mkdir(parents=True, exist_ok=True)
        saved = self.directory / 'configuration.json'
        if saved.exists() and read_json(saved) != config:
            previous = read_json(saved)
            before = {k: v for k, v in previous.items() if k != 'retain_above_score'}
            after = {k: v for k, v in config.items() if k != 'retain_above_score'}
            if before != after:
                raise ValueError('Monitor configuration changed; use a different state directory')
            validate_retention_threshold(config.get('retain_above_score'))
            write_json(self.directory / f'configuration_before_retention_{time.time_ns()}.json', previous)
        write_json(saved, config)
        self.children = []

    def tick(self):
        # Capture every newly committed boundary, even while the GPU queue is busy.
        errors = {}
        for run_path in self.config['runs']:
            run = Path(run_path)
            try:
                committed = committed_run(run)
                if committed is None:
                    continue
                summary, configuration, rows = committed
                completed = summary['completed_epochs']
                interval = self.config['every_epochs']
                if self.config.get('periodic_evaluation', True):
                    for epoch in range(interval, completed + 1, interval):
                        prepare_job(run, epoch, summary, configuration, rows, self.config)
                if (self.config.get('final_evaluation', True)
                        and summary['status'] in ('complete', 'early_stopped')):
                    prepare_job(run, completed, summary, configuration, rows, self.config, final=True)
            except Exception as error:
                errors[run.name] = f'{type(error).__name__}: {error}'
        pending, active, all_status = [], [], {}
        for run_path in self.config['runs']:
            milestones = []
            for path in sorted((Path(run_path) / 'periodic_test').glob('*/status.json')):
                job, state = path.parent, read_json(path)
                if state['status'] in ('launching', 'running'):
                    identities = [state.get('worker_identity'), state.get('evaluator_identity')]
                    if any(alive(identity) for identity in identities) or job_processes(job):
                        active.append(job)
                    else:
                        try:
                            complete_job(job, read_json(job / 'trigger.json'),
                                         retain_above_score=self.config.get('retain_above_score'))
                        except Exception as error:
                            if self.config.get('remote') and (job / 'remote_spec.json').exists():
                                set_status(job, 'queued', waiting_reason='Reconnect to persistent remote evaluator: ' + str(error))
                            else:
                                set_status(job, 'failed', reason='Worker/evaluator disappeared: ' + str(error))
                        state = read_json(path)
                if state['status'] == 'queued':
                    pending.append(job)
                if state['status'] in ('complete', 'reused'):
                    try:
                        trigger = read_json(job / 'trigger.json')
                        verified = validate_metrics(read_json(job / 'test_metrics.json'), trigger)
                        milestone = finalize_checkpoint(job, trigger, verified,
                            self.config.get('retain_above_score'), state.get('reused_from'))
                        if milestone:
                            milestones.append(milestone)
                            if state.get('score_milestone') != milestone:
                                state = set_status(job, state['status'], score_milestone=milestone)
                    except Exception as error:
                        errors[str(job)] = f'Checkpoint retention: {type(error).__name__}: {error}'
                all_status[str(job)] = state
            if self.config.get('retain_above_score') is not None or milestones:
                write_json(Path(run_path) / 'score_milestones.json', {
                    'schema': 'vit_test_score_milestones_v1', 'updated_at': now(),
                    'metric': 'weighted_score', 'comparison': '>',
                    'threshold': self.config.get('retain_above_score'),
                    'count': len(milestones), 'records': milestones})
        waiting = None
        if pending and not active:
            try:
                reconnect = [job for job in pending if self.config.get('remote') and (job / 'remote_spec.json').exists()]
                ready, waiting, _ = (True, 'Reconnect to existing remote task', None) if reconnect else gpu_available(self.config)
                if ready:
                    job = min(reconnect or pending, key=lambda p: (read_json(p / 'trigger.json')['trigger_epoch'], str(p)))
                    launch = {'created_at': now(), 'configuration': self.config,
                              'command': evaluator_command(self.config, job), 'host': socket.gethostname()}
                    if self.config.get('remote'):
                        launch.update(executor='remote', execution_host=self.config['remote']['host'])
                    write_json(job / 'launch.json', launch)
                    set_status(job, 'launching', launch_requested_at=now())
                    with (job / 'worker.log').open('a') as output:
                        child = subprocess.Popen([self.config['python'], '-u', str(Path(__file__).resolve()),
                            '--worker', str(job)], cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    self.children.append(child)
                    active.append(job)
                    # Worker owns status from here; no parent write can clobber its completion.
                    all_status[str(job)] = read_json(job / 'status.json')
            except Exception as error:
                errors['scheduler'] = f'{type(error).__name__}: {error}'
        self.children = [child for child in self.children if child.poll() is None]
        report = {'schema': 'vit_periodic_monitor_v1', 'updated_at': now(),
                  'monitor_identity': process_identity(os.getpid()), 'host': socket.gethostname(),
                  'active_jobs': list(map(str, active)), 'jobs': all_status, 'errors': errors,
                  'gpu_wait_reason': waiting, 'training_process_management': False}
        write_json(self.directory / 'state.json', report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--once', action='store_true', help='Scan and schedule once; a launched worker continues')
    parser.add_argument('--poll-seconds', type=int)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.config:
        parser.error('--config is required')
    config = load_config(args.config)
    if args.poll_seconds is not None:
        if args.poll_seconds < 1:
            parser.error('--poll-seconds must be positive')
        config['poll_seconds'] = args.poll_seconds
    lock = acquire_lock(Path(config['state_directory']) / '.monitor.lock')
    if lock is None:
        print('Periodic evaluator monitor already running', flush=True)
        return 0
    try:
        monitor = Monitor(config)
        while True:
            report = monitor.tick()
            print(json.dumps({'time': report['updated_at'], 'active': report['active_jobs'],
                              'errors': report['errors']}, ensure_ascii=False), flush=True)
            if args.once:
                return 0
            time.sleep(config['poll_seconds'])
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
