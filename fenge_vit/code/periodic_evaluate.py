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
HARD_PROTOCOL = 'pool8002_train6000_hard1500_validation_v1'


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
            'tta_weights': [0.5, 0.5], 'amp': 'bf16',
            'batch_size': config.get('evaluation_batch_size', 4),
            'workers': config.get('evaluation_workers', 4),
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
    if (type(config.get('evaluation_batch_size', 4)) is not int
            or config.get('evaluation_batch_size', 4) < 1
            or type(config.get('evaluation_workers', 4)) is not int
            or config.get('evaluation_workers', 4) < 0):
        raise ValueError('Invalid evaluation batch size or worker count')
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
    if configuration.get('selection_policy') == 'full_test_every5':
        return committed_fulltest_run(run, summary, configuration)
    if configuration.get('protocol') == HARD_PROTOCOL:
        return committed_hard_run(run, summary, configuration)
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


def committed_fulltest_run(run, summary, configuration):
    if (configuration.get('protocol') != HARD_PROTOCOL
            or configuration.get('early_stopping_patience') != 0
            or summary.get('format') != 'lusnar_vit_run_v1' or summary.get('partial') is not False
            or summary.get('train_images') != 6000
            or summary.get('data_identity', {}).get('train', {}).get('image_count') != 6000
            or summary.get('data_identity', {}).get('val', {}).get('image_count') != 1500):
        raise ValueError('Expected full-budget training with explicit full-test selection')
    completed = summary.get('completed_epochs')
    if type(completed) is not int or not 1 <= completed <= configuration['epochs']:
        raise ValueError('Invalid completed epoch for full-test policy')
    rows = {}
    for line in (run / 'history.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'): break
        row = json.loads(line)
        if row['epoch'] <= completed:
            if row['epoch'] in rows: raise ValueError('Duplicate committed epoch')
            rows[row['epoch']] = row
    if set(rows) != set(range(1, completed + 1)):
        raise ValueError('Incomplete committed full-test history')
    if summary.get('status') == 'early_stopped':
        raise ValueError('Early stopping is disabled under full-test policy')
    return summary, configuration, rows


def committed_hard_run(run, summary, configuration):
    """Validate the published prefix; odd training epochs need no new validation."""
    identity = summary.get('data_identity', {}).get('val', {})
    if (summary.get('format') != 'lusnar_vit_run_v1' or summary.get('partial') is not False
            or configuration.get('validation_every') != 2
            or summary.get('train_images') != 6000 or summary.get('validation_images') != 1500
            or summary.get('data_identity', {}).get('train', {}).get('image_count') != 6000
            or identity.get('image_count') != 1500 or identity.get('unique_source_count') != 1500
            or identity.get('selection_metric') != 'validation_weighted_score'
            or identity.get('validation_scene_counts') != {'Moon_2': 188, 'Moon_4': 187, 'Moon_9': 1125}
            or not identity.get('hard_manifest_sha256')
            or summary.get('best_metric') != 'validation_weighted_score'
            or summary.get('checkpoint_selection_uses_test') is not True
            or summary.get('official_test_used_for_selection') is not True):
        raise ValueError('Expected explicit complete hard1500 validation selection protocol')
    completed = summary.get('completed_epochs')
    if type(completed) is not int or not 1 <= completed <= configuration['epochs']:
        raise ValueError('Invalid completed_epochs')
    rows = {}
    for line in (run / 'history.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'): break
        row = json.loads(line)
        if row['epoch'] <= completed:
            if row['epoch'] in rows: raise ValueError('Duplicate committed history epoch')
            rows[row['epoch']] = row
    if set(rows) != set(range(1, completed + 1)):
        raise ValueError('Committed hard-validation summary/history are incomplete')
    best, best_epoch = None, None
    for epoch, row in sorted(rows.items()):
        validates = epoch % 2 == 0 or epoch == configuration['epochs']
        if (row.get('validation_performed') is not validates or row.get('partial') is not False
                or row.get('checkpoint_selection_uses_test') is not True):
            raise ValueError('Hard-validation cadence or selection provenance changed')
        score = row.get('validation_weighted_score')
        if validates:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('Missing finite hard-validation weighted score')
            document = read_json(run / f'validation_epoch_{epoch:03d}.json')
            validate_validation(document, epoch, score, identity)
            assert_close(document['miou'], row.get('validation_miou'), 'Pooled validation mIoU differs')
            if best is None or score > best: best, best_epoch = score, epoch
        elif score is not None or row.get('validation_miou') is not None:
            raise ValueError('Skipped validation must not invent a measured score')
        if row.get('best_epoch') != best_epoch or row.get('best_miou') != best:
            raise ValueError('Hard best must be the first strict weighted validation maximum')
    if summary.get('best_epoch') != best_epoch or summary.get('best_miou') != best:
        raise ValueError('Hard-validation summary best differs from committed history')
    if summary.get('status') in ('complete', 'early_stopped') and not rows[completed]['validation_performed']:
        raise ValueError('Terminal training must have a measured final validation')
    return summary, configuration, rows


def validate_validation(document, epoch, score, identity):
    expected_count = identity.get('image_count')
    expected_sources = identity.get('unique_source_count', expected_count)
    if type(expected_count) is not int or expected_count < 1 or type(expected_sources) is not int or expected_sources < 1:
        raise ValueError('Validation identity is missing positive coverage counts')
    hard = identity.get('selection_metric') == 'validation_weighted_score'
    if (document.get('epoch') != epoch or document.get('partial') is not False
            or document.get('image_count') != expected_count
            or document.get('validation_unique_source_count') != expected_sources
            or document.get('checkpoint_selection_uses_test') is not hard
            or document.get('data_identity') != identity):
        raise ValueError('Full fixed-validation identity/coverage mismatch')
    if hard and document.get('official_test_used_for_selection') is not True:
        raise ValueError('Hard validation must disclose test use for selection')
    assert_close(document.get('validation_weighted_score' if hard else 'miou'), score,
                 'Validation selection score differs from history')


def check_sources(configuration, source_root=None):
    sources = configuration.get('source_sha256')
    if not sources or 'code/train.py' not in sources:
        raise ValueError('Training source hashes missing')
    source_root = ROOT if source_root is None else Path(source_root).resolve()
    for relative, expected in sources.items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root / 'code') or sha256(path) != expected:
            raise ValueError('Training/evaluation source changed: ' + relative)
    # An archived trainer is allowed, but actual inference must still use the
    # exact model/data/loss implementation recorded by that trainer.
    if source_root != ROOT:
        actual = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'code/vitseg').glob('*.py')}
        expected = {p: digest for p, digest in sources.items()
                    if Path(p).parent == Path('code/vitseg') and p.endswith('.py')}
        if not actual or actual != expected:
            raise ValueError('Actual evaluator vitseg sources differ from the checkpoint training sources')


def job_training_configuration(job, trigger):
    """Read an immutable per-job configuration, with legacy live-file fallback."""
    job = Path(job)
    pinned = job / 'training_configuration.json'
    configuration = read_json(pinned if pinned.exists() else Path(trigger['run']) / 'run_config.json')
    if fingerprint(configuration) != trigger['configuration_sha256']:
        raise ValueError('Training configuration changed after snapshot selection')
    source_root = trigger.get('training_source_root')
    if source_root is not None:
        run, requested = Path(trigger['run']).resolve(), Path(source_root)
        if (not requested.is_absolute() or requested not in {run / 'runtime', run / 'runtime_fulltest'}
                or not requested.is_dir() or not requested.resolve().is_relative_to(run)):
            raise ValueError('training_source_root must be this run runtime or runtime_fulltest')
    check_sources(configuration, source_root=source_root)
    saved = checkpoint_metadata(job / 'best.pt')
    if fingerprint(saved.get('train_config')) != trigger['configuration_sha256']:
        raise ValueError('Snapshot training configuration differs from its pinned configuration')
    return configuration


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


def recovered_pretransition_snapshot(trigger, configuration):
    return (trigger.get('recovered_pretransition_current_epoch') is True
            and configuration.get('selection_policy', 'hard_validation_best') == 'hard_validation_best'
            and trigger.get('checkpoint_choice') == 'current_epoch'
            and trigger.get('training_source_root') == str(Path(trigger['run']).resolve() / 'runtime'))


def validate_snapshot(saved, trigger, configuration, summary):
    current_epoch = trigger.get('selection_policy') == 'full_test_every5'
    if (saved.get('format') != 'lusnar_vit_checkpoint_v1' or saved.get('partial') is not False
            or saved.get('epoch') != trigger['checkpoint_epoch']
            or (not current_epoch and saved.get('best_epoch') != trigger['checkpoint_epoch'])
            or saved.get('train_config') != configuration
            or saved.get('data_identity') != summary['data_identity']
            or saved.get('model_config') != summary['model_config']):
        raise ValueError('Checkpoint metadata/configuration/data identity mismatch')
    if current_epoch:
        if (configuration.get('selection_policy') != 'full_test_every5'
                and not recovered_pretransition_snapshot(trigger, configuration)):
            raise ValueError('Current-epoch selection requires the explicit full-test training policy')
        return
    assert_close(saved['best_miou'], trigger['validation_best_miou'], 'Checkpoint best score mismatch')
    validate_validation(saved['metrics'], trigger['checkpoint_epoch'],
                        trigger['validation_best_miou'], summary['data_identity']['val'])


def validate_metrics(document, trigger):
    hard = trigger.get('checkpoint_selection_uses_test') is True
    if hard:
        if (trigger.get('protocol') != HARD_PROTOCOL
                or document.get('protocol') != HARD_PROTOCOL
                or document.get('checkpoint_selection_uses_test') is not True
                or document.get('official_test_used_for_selection') is not True
                or document.get('independent_holdout') is not False
                or document.get('validation_images_included') != 1500
                or document.get('full_test_coverage') is not True):
            raise ValueError('Full hard-validation test must disclose checkpoint selection overlap')
    if (document.get('partial') is not False or document.get('checkpoint_partial_training') is not False
            or document.get('split') != 'test' or document.get('image_count') != 5004
            or document.get('validation_unique_source_count') != 5004
            or document.get('official_test_scope') is not (not hard)
            or document.get('test_scope') != ('full5004_including_selection_validation' if hard else 'complete_official_test')
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
    if trigger.get('selection_policy') == 'full_test_every5':
        return retain_fulltest_best(job, trigger, verified, reused_from)
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
                'checkpoint_selection_uses_test': trigger.get('checkpoint_selection_uses_test', False),
                'training_process_management': False}
    write_json(marker, recorded)
    return recorded


def retain_fulltest_best(job, trigger, verified, reused_from=None):
    """Keep one exact highest full-test checkpoint, independently of final.pt."""
    run = Path(trigger['run'])
    lock = acquire_lock(run / '.best_full_test.lock')
    if lock is None: raise RuntimeError('Full-test best update is already in progress')
    try:
        marker, best_path = run / 'best_full_test.json', run / 'best_full_test.pt'
        previous = read_json(marker) if marker.exists() else None
        if previous and (not best_path.exists() or sha256(best_path) != previous['checkpoint_sha256']):
            raise ValueError('Previously retained full-test best is missing or changed')
        is_new = previous is None or verified['weighted_score'] > previous['weighted_score']
        if is_new:
            candidates = [job / 'best.pt', run / 'full_test_checkpoints' / f"epoch_{trigger['checkpoint_epoch']:03d}.pt",
                          run / 'final.pt', run / 'last.pt']
            if reused_from: candidates.append(Path(reused_from) / 'best.pt')
            source = next((p for p in candidates if p.exists() and sha256(p) == trigger['checkpoint_sha256']), None)
            if source is None: raise ValueError('Winning full-test checkpoint bytes are unavailable')
            snapshot(source, best_path)
            previous = {'schema': 'lusnar_best_full_test_v1', 'updated_at': now(),
                        'selection_policy': 'full_test_every5', 'selection_metric': 'weighted_score',
                        'checkpoint_epoch': trigger['checkpoint_epoch'],
                        'checkpoint_sha256': trigger['checkpoint_sha256'], 'checkpoint': str(best_path),
                        'metrics': str(job / 'test_metrics.json'), 'metrics_sha256': sha256(job / 'test_metrics.json'),
                        'checkpoint_selection_uses_test': True, 'independent_holdout': False, **verified}
            write_json(marker, previous)
        # Publish completion proof before deleting the immutable training candidate.
        receipt_path = run / 'full_test_checkpoints' / f"epoch_{trigger['checkpoint_epoch']:03d}.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get('checkpoint_sha256') != trigger['checkpoint_sha256']:
                raise ValueError('Training candidate receipt differs from evaluated checkpoint')
            receipt['evaluation'] = {'status': 'complete', 'completed_at': now(),
                                     'checkpoint_sha256': trigger['checkpoint_sha256'],
                                     'metrics_path': str(job / 'test_metrics.json'),
                                     'metrics_sha256': sha256(job / 'test_metrics.json'),
                                     'weighted_score': verified['weighted_score']}
            write_json(receipt_path, receipt)
        # These are evaluation-owned temporary files, never trainer last/final.
        temporary = [job / 'best.pt', run / 'full_test_checkpoints' / f"epoch_{trigger['checkpoint_epoch']:03d}.pt"]
        for p in temporary:
            if p.exists():
                if sha256(p) != trigger['checkpoint_sha256']:
                    raise ValueError('Temporary checkpoint changed before cleanup')
                p.unlink()
        return previous
    finally:
        lock.close()


def complete_job(job, trigger, reused_from=None, retain_above_score=None):
    document = read_json(job / 'test_metrics.json')
    verified = validate_metrics(document, trigger)
    if (trigger.get('recovered_pretransition_current_epoch') is True
            and document.get('checkpoint_choice') != 'current_epoch'):
        configuration = job_training_configuration(job, trigger)
        if not recovered_pretransition_snapshot(trigger, configuration):
            raise ValueError('Invalid pre-transition current-epoch recovery binding')
        if sha256(job / 'best.pt') != trigger['checkpoint_sha256']:
            raise ValueError('Recovered snapshot changed before metadata correction')
        saved = checkpoint_metadata(job / 'best.pt')
        if saved['epoch'] != trigger['checkpoint_epoch']:
            raise ValueError('Recovered snapshot epoch differs before metadata correction')
        original_sha = sha256(job / 'test_metrics.json')
        archive = job / 'test_metrics_before_metadata_correction.json'
        if archive.exists() and sha256(archive) != original_sha:
            raise ValueError('Existing uncorrected metrics archive differs')
        if not archive.exists():
            shutil.copyfile(job / 'test_metrics.json', archive)
        document.update(reported_checkpoint_choice=document.get('checkpoint_choice'),
            checkpoint_choice='current_epoch', metadata_correction={
                'kind': 'recovered_pretransition_current_epoch',
                'reason': 'Legacy evaluator inferred validation-best selection from the old training policy; '
                          'this explicitly pinned checkpoint is the current epoch.',
                'original_metrics_sha256': original_sha,
                'checkpoint_epoch': saved['epoch'], 'legacy_validation_best_epoch': saved.get('best_epoch'),
                'scores_and_confusion_matrices_unchanged': True})
        write_json(job / 'test_metrics.json', document)
        verified = validate_metrics(document, trigger)
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
    current_epoch = configuration.get('selection_policy') == 'full_test_every5'
    expected_epoch, expected_score = (boundary, None) if current_epoch else (row['best_epoch'], row['best_miou'])
    hard = configuration.get('protocol') == HARD_PROTOCOL
    if not current_epoch and (not hard or row['validation_performed']):
        validate_validation(read_json(run / f'validation_epoch_{boundary:03d}.json'), boundary,
                            row['validation_weighted_score' if hard else 'validation_miou'],
                            summary['data_identity']['val'])
    if not current_epoch:
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
               'evaluation': config['evaluation'], 'checkpoint_selection_uses_test': hard}
    if hard:
        trigger.update(protocol=HARD_PROTOCOL, best_metric='validation_weighted_score',
                       validation_best_score=expected_score, official_test_used_for_selection=True,
                       independent_holdout=False)
    if current_epoch:
        trigger.update(selection_policy='full_test_every5', checkpoint_choice='current_epoch',
                       best_metric='weighted_score', validation_best_score=None,
                       retention_policy='best_full_test_and_final_per_run')
    if config.get('retain_above_score') is not None:
        trigger['retain_above_score'] = validate_retention_threshold(config['retain_above_score'])
    # A retained older snapshot can recover a boundary after best.pt advances.
    candidates = ([run / 'full_test_checkpoints' / f'epoch_{boundary:03d}.pt', run / 'final.pt', run / 'last.pt']
                  if current_epoch else [run / 'best.pt', run / 'last.pt'])
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
            write_json(job / 'training_configuration.json', configuration)
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
        write_json(job / 'training_configuration.json', configuration)
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
            '--amp', 'bf16', '--batch-size', str(config.get('evaluation_batch_size', 4)),
            '--workers', str(config.get('evaluation_workers', 4)), '--tta', 'multiscale_hflip',
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
        job_training_configuration(job, trigger)
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
                if (self.config.get('required_selection_policy') is not None
                        and configuration.get('selection_policy') != self.config['required_selection_policy']):
                    continue  # A scheduled epoch-boundary policy transition has not completed yet.
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
                        # Full-test winners may be replaced by a later epoch. Their
                        # per-job completion record is historical, not another request
                        # to restore the old winner or hash gigabytes on every poll.
                        milestone = (None if trigger.get('selection_policy') == 'full_test_every5' else
                            finalize_checkpoint(job, trigger, verified,
                                self.config.get('retain_above_score'), state.get('reused_from')))
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
