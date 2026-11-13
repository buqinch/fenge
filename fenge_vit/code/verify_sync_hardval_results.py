#!/usr/bin/env python3
"""Verify sparse hard-validation mirroring without SSH or training processes."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile

import sync_hardval_results as sync
import sync_remote_results as mirror


def main():
    mirror.validate_boundary = sync.validate_boundary
    val_identity = {'image_count': 1500, 'hard_manifest_sha256': 'immutable-sha'}
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); stage = root / 'stage'; local = root / 'local'
        stage.mkdir(); local.mkdir(); target = {'local': local}
        config = {'protocol': sync.PROTOCOL, 'validation_every': 2, 'epochs': 3}
        mirror.write_json(stage / 'run_config.json', config)
        base_summary = {'best_metric': 'validation_weighted_score', 'checkpoint_selection_uses_test': True,
            'official_test_used_for_selection': True, 'data_identity': {'val': val_identity}}
        row1 = {'epoch': 1, 'validation_performed': False, 'validation_miou': None,
                'validation_weighted_score': None, 'best_epoch': None, 'best_miou': None}
        (stage / 'history.jsonl').write_text(json.dumps(row1) + '\n')
        mirror.write_json(stage / 'summary.json', {**base_summary, 'completed_epochs': 1,
            'best_epoch': None, 'best_miou': None})
        status = mirror.publish(target, stage)
        assert status['summary_published'] and json.loads((local / 'summary.json').read_text())['completed_epochs'] == 1
        checks.append('First unvalidated epoch publishes without fake validation/best')
        row2 = {'epoch': 2, 'validation_performed': True, 'validation_miou': .9,
                'validation_weighted_score': .8, 'best_epoch': 2, 'best_miou': .8}
        (stage / 'history.jsonl').write_text(json.dumps(row1) + '\n' + json.dumps(row2) + '\n')
        mirror.write_json(stage / 'summary.json', {**base_summary, 'completed_epochs': 2,
            'best_epoch': 2, 'best_miou': .8})
        pending = mirror.publish(target, stage)
        assert pending['status'] == 'pending' and not pending['summary_published']
        assert json.loads((local / 'summary.json').read_text())['completed_epochs'] == 1
        checks.append('An incomplete second-epoch transfer cannot advance local summary')
        val2 = {'epoch': 2, 'miou': .9, 'validation_weighted_score': .8,
                'data_identity': val_identity, 'checkpoint_selection_uses_test': True}
        mirror.write_json(stage / 'validation_epoch_002.json', val2)
        assert mirror.publish(target, stage)['summary_published']
        row3 = {'epoch': 3, 'validation_performed': True, 'validation_miou': .95,
                'validation_weighted_score': .79, 'best_epoch': 2, 'best_miou': .8}
        (stage / 'history.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in (row1, row2, row3)))
        mirror.write_json(stage / 'validation_epoch_003.json', {**val2, 'epoch': 3, 'miou': .95, 'validation_weighted_score': .79})
        mirror.write_json(stage / 'summary.json', {**base_summary, 'completed_epochs': 3,
            'best_epoch': 2, 'best_miou': .8})
        assert mirror.publish(target, stage)['summary_published']
        assert json.loads((local / 'summary.json').read_text())['best_epoch'] == 2
        checks.append('Odd final epoch validates; best follows weighted score, not increasing pooled mIoU')
        # An unchanged top-level summary is omitted by rsync --compare-dest.
        for file in list(stage.iterdir()): file.unlink()
        mirror.write_json(stage / 'final_partition_report/metrics/summary.json', {'status': 'complete', 'independent_holdout': False})
        mirror.write_json(stage / 'final_partition_report/metrics/full5004_metrics.json', {'weighted_score': .93})
        assert mirror.publish(target, stage)['status'] == 'synced'
        assert (local / 'final_partition_report/metrics/summary.json').exists()
        checks.append('Final nested summary publishes with unchanged existing training boundary')
        mirror.write_json(stage / 'summary.json', {**base_summary, 'completed_epochs': 3,
            'best_epoch': 3, 'best_miou': .95})
        try: mirror.publish(target, stage)
        except ValueError: pass
        else: raise AssertionError('Incorrect pooled-score best was accepted')
        checks.append('Incorrect checkpoint-selection score rejected')
    output = Path(__file__).resolve().parents[1] / 'result/verification_sync_hardval_results.json'
    mirror.write_json(output, {'status': 'passed', 'checks': checks, 'check_count': len(checks),
                             'ssh_called': False, 'training_started': False})
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'output': str(output)}))


if __name__ == '__main__': main()
