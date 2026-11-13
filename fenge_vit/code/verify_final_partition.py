#!/usr/bin/env python3
"""CPU integrity checks for the future hard-validation final report protocol."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

import evaluate_final_partition as evaluation
import final_partition_monitor as monitor
from evaluate_hard_subset import canonical, write_json
from vitseg.metrics import ConfusionMatrix


def rejects(function, *args):
    try: function(*args)
    except (ValueError, KeyError): return
    raise AssertionError('Invalid state was accepted')


def fixtures():
    full, hard, rows = [], [], []
    for scene, count in evaluation.FULL_COUNTS.items():
        for index in range(count):
            record = {'scene': scene, 'sample_id': str(index), 'image': f'{scene}/{index}.png'}
            full.append(record)
            if index < evaluation.QUOTAS[scene]: hard.append(record)
            cm = ConfusionMatrix()
            # Moon4 deliberately has no mountain GT; count only its four present classes.
            present = [0, 1, 2, 4] if scene == 'Moon_4' else list(range(5))
            base = 1024 * 1024 // len(present)
            for label in present: cm.matrix[label, label] = base
            cm.matrix[0, 0] += 1024 * 1024 - int(cm.matrix.sum())
            moved = index % 1000
            cm.matrix[2, 2] -= moved; cm.matrix[2, 0] += moved
            result = cm.compute()
            rows.append({'source_id': f'{scene}/{index}', 'scene': scene, 'sample_id': str(index),
                         'miou': result['miou'], 'class_iou': result['class_iou'],
                         'confusion_matrix_gt_rows_pred_cols': cm.matrix.tolist()})
    return full, hard, rows


def main():
    checks = []
    full, hard, rows = fixtures()
    report = evaluation.partition_rows(rows, full, hard)
    assert {k: v['image_count'] for k, v in report.items()} == {'full5004': 5004, 'validation1500': 1500, 'complement3504': 3504}
    assert {s: d['image_count'] for s, d in report['complement3504']['scenes'].items()} == evaluation.COMPLEMENT_COUNTS
    for s in (None, *evaluation.FULL_COUNTS):
        cm = [np.asarray((r if s is None else r['scenes'][s])['confusion_matrix_gt_rows_pred_cols']) for r in report.values()]
        assert np.array_equal(cm[0], cm[1] + cm[2])
    checks.append('Exact 5004 = 1500 + 3504 disjoint source partition and scene/global confusion additivity')
    for group in report.values():
        assert group['scenes']['Moon_4']['class_iou']['mountain'] is None
        assert group['independent_holdout'] is False and group['official_full_test'] is False
        expected = sum(evaluation.SCORE_WEIGHTS[s] * group['scenes'][s]['miou'] for s in evaluation.FULL_COUNTS) / 4
        assert expected == group['weighted_scene_score']
    checks.append('GT-present class means, scene weighting 1:1:2, and no independent-test claims')
    changed = list(rows); changed[0] = rows[1]
    rejects(evaluation.partition_rows, changed, full, hard)
    rejects(evaluation.partition_rows, rows[:-1], full, hard)
    rejects(evaluation.partition_rows, rows, full, hard[:-1] + [hard[0]])
    changed = deepcopy(rows); changed[0]['confusion_matrix_gt_rows_pred_cols'][0][0] -= 1
    rejects(evaluation.partition_rows, changed, full, hard)
    checks.append('Duplicate/missing samples and incomplete pixel coverage rejected')
    identity = {'image_count': 1500, 'unique_source_count': 1500, 'hard_manifest_sha256': 'hard-sha'}
    saved = {'format': 'lusnar_vit_checkpoint_v1', 'partial': False,
             'data_identity': {'val': identity, 'train': {'image_count': 6000}},
             'train_config': {'protocol': evaluation.PROTOCOL}, 'epoch': 4, 'best_epoch': 4}
    evaluation.validate_checkpoint_protocol(saved, 'hard-sha')
    changed = deepcopy(saved); changed['data_identity']['val']['image_count'] = 1601
    rejects(evaluation.validate_checkpoint_protocol, changed, 'hard-sha')
    changed = deepcopy(saved); changed['epoch'] = 5
    rejects(evaluation.validate_checkpoint_protocol, changed, 'hard-sha')
    rejects(evaluation.validate_checkpoint_protocol, saved, 'another-sha')
    checks.append('Wrong validation/checkpoint/subset identity rejected')
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        assert monitor.completed_training(run, 'hard-sha') is None
        summary = {'format': 'lusnar_vit_run_v1', 'status': 'complete', 'partial': False,
            'validation_images': 1500, 'train_images': 6000,
            'data_identity': {'val': identity, 'train': {'image_count': 6000}}, 'completed_epochs': 4,
            'best_epoch': 4, 'best_miou': .85, 'checkpoint_selection_uses_test': True,
            'official_test_used_for_selection': True, 'best_metric': 'validation_weighted_score'}
        write_json(run / 'summary.json', summary)
        write_json(run / 'run_config.json', {'protocol': evaluation.PROTOCOL})
        history = [{'epoch': 1, 'validation_weighted_score': None, 'best_epoch': None, 'best_miou': None},
                   {'epoch': 2, 'validation_weighted_score': .8, 'best_epoch': 2, 'best_miou': .8},
                   {'epoch': 3, 'validation_weighted_score': None, 'best_epoch': 2, 'best_miou': .8},
                   {'epoch': 4, 'validation_weighted_score': .85, 'best_epoch': 4, 'best_miou': .85}]
        for row in history: row['validation_performed'] = row['validation_weighted_score'] is not None
        (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        assert monitor.completed_training(run, 'hard-sha')[0]['best_epoch'] == 4
        for status in ('running', 'user_stopped', 'failed'):
            write_json(run / 'summary.json', {**summary, 'status': status})
            assert monitor.completed_training(run, 'hard-sha') is None
        write_json(run / 'summary.json', {**summary, 'status': 'early_stopped'})
        assert monitor.completed_training(run, 'hard-sha') is not None
        changed = deepcopy(history); changed[3].update(validation_weighted_score=.8, best_miou=.8)
        (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in changed))
        rejects(monitor.completed_training, run, 'hard-sha')
    checks.append('Final-only after complete/early_stopped; no user-stopped trigger; sparse two-epoch strict best')
    # Audit actual frozen construction scores without launching inference or changing that protocol.
    manifest = evaluation.ROOT / 'result/hard_test_1500_20261113/manifest.json'
    source = json.loads(manifest.read_text())
    actual_full = evaluation.validate_full_source(source['full_test_manifest'])
    actual_rows = [json.loads(line) for line in Path(source['source_per_image_jsonl']).read_text().splitlines()]
    actual = evaluation.partition_rows(actual_rows, actual_full, source['records'])
    original = json.loads(Path(source['source_full_metrics']).read_text())
    assert actual['full5004']['confusion_matrix_gt_rows_pred_cols'] == original['confusion_matrix_gt_rows_pred_cols']
    assert abs(actual['validation1500']['weighted_scene_score'] - .8073788305473157) < 1e-12
    checks.append('Existing 5004-image source journal independently reconstructs exact fixed1500 and complement3504')
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        provenance = {'checkpoint_sha256': source['checkpoint_sha256'],
                      'hard_manifest_sha256': evaluation.sha256_file(manifest)}
        for name, group in actual.items():
            write_json(directory / (name + '_metrics.json'), {**group, **provenance})
        (directory / 'per_image.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in actual_rows))
        filenames = [name + '_metrics.json' for name in actual] + ['per_image.jsonl']
        published = {**provenance, 'schema': 'lusnar_final_partition_report_v1', 'status': 'complete',
            'independent_holdout': False, 'image_count': 5004, 'hard_manifest': str(manifest),
            'full_test_manifest': source['full_test_manifest'],
            'full_test_manifest_sha256': source['full_test_manifest_sha256'],
            'artifact_sha256': {name: evaluation.sha256_file(directory / name) for name in filenames}}
        write_json(directory / 'summary.json', published)
        evaluation.validate_bundle(directory, source['checkpoint_sha256'], provenance['hard_manifest_sha256'])
        changed = json.loads((directory / 'complement3504_metrics.json').read_text())
        changed['weighted_scene_score'] += .01
        write_json(directory / 'complement3504_metrics.json', changed)
        rejects(evaluation.validate_bundle, directory)
        # Even if a changed file's checksum is updated, the image CMs remain authoritative.
        published['artifact_sha256']['complement3504_metrics.json'] = evaluation.sha256_file(directory / 'complement3504_metrics.json')
        write_json(directory / 'summary.json', published)
        rejects(evaluation.validate_bundle, directory)
    checks.append('Completed report replay validates immutable files and independently reconstructs edited numeric fields')
    output = evaluation.ROOT / 'result/verification_final_partition.json'
    write_json(output, {'status': 'passed', 'checks': checks, 'check_count': len(checks),
        'synthetic_cpu_checks': True, 'actual_source_journal_audit': True, 'new_model_inference': False,
        'training_started': False, 'actual_source_partition_scores': {
            k: r['weighted_scene_score'] for k, r in actual.items()},
        'note': 'Source journal scores test reconstruction only; they are not results of the future training run.'})
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'output': str(output)}))


if __name__ == '__main__': main()
