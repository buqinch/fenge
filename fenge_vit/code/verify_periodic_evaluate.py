#!/usr/bin/env python3
"""CPU fixtures for periodic scheduling/recovery, never real model/test scores."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location('periodic_evaluate', Path(__file__).with_name('periodic_evaluate.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class PeriodicFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vit-periodic-fixture-')
        self.root = Path(self.temporary.name)
        self.old_root, m.ROOT = m.ROOT, self.root
        (self.root / 'code/vitseg').mkdir(parents=True)
        (self.root / 'code/train.py').write_text('# fixture source only\n')
        (self.root / 'code/vitseg/data.py').write_text('# fixture data source\n')
        (self.root / 'test_manifest.json').write_text('{}')
        # This subprocess checks worker hand-off/exit/status only, not inference.
        (self.root / 'code/evaluate.py').write_text(
            'import json,sys,pathlib\n'
            'out=pathlib.Path(sys.argv[sys.argv.index("--output")+1])\n'
            'out.write_text((out.parent/"fixture_metrics.json").read_text())\n')
        self.configuration = {'source_sha256': {p: m.sha256(self.root / p) for p in
            ('code/train.py', 'code/vitseg/data.py')}, 'variant': 'fixture', 'epochs': 30}
        self.identity = {'val': {'split': 'val', 'image_count': 1601, 'partial': False,
                                 'ordered_samples_sha256': 'fixed-fixture-validation'}}
        self.config = {'schema': 'vit_periodic_test_v1', 'project': str(self.root),
            'runs': [str(self.root / 'result/b'), str(self.root / 'result/c')],
            'state_directory': str(self.root / 'result/monitor'), 'python': sys.executable,
            'gpu_index': 3, 'minimum_free_mib': 10000, 'every_epochs': 3, 'poll_seconds': 10,
            'final_evaluation': True, 'test_manifest': str(self.root / 'test_manifest.json'),
            'gpu_lock': str(self.root / 'result/.gpu.lock')}
        self.config['evaluation'] = m.settings(self.config)

    def tearDown(self):
        m.ROOT = self.old_root
        self.temporary.cleanup()

    def publish(self, name='b', scores=(0.5, 0.6, 0.7), committed=None):
        run = self.root / 'result' / name
        run.mkdir(parents=True, exist_ok=True)
        m.write_json(run / 'run_config.json', self.configuration)
        rows, best, best_epoch = [], -1, 0
        documents = {}
        for epoch, score in enumerate(scores, 1):
            if score > best:
                best, best_epoch = score, epoch
            validation = {'epoch': epoch, 'image_count': 1601, 'validation_unique_source_count': 1601,
                'partial': False, 'checkpoint_selection_uses_test': False,
                'data_identity': self.identity['val'], 'miou': score}
            documents[epoch] = validation
            m.write_json(run / f'validation_epoch_{epoch:03d}.json', validation)
            rows.append({'epoch': epoch, 'validation_miou': score, 'best_epoch': best_epoch,
                         'best_miou': best, 'partial': False, 'checkpoint_selection_uses_test': False})
        (run / 'history.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        completed = committed or len(scores)
        row = rows[completed - 1]
        summary = {'format': 'lusnar_vit_run_v1', 'completed_epochs': completed,
            'best_epoch': row['best_epoch'], 'best_miou': row['best_miou'], 'partial': False,
            'validation_images': 1601, 'checkpoint_selection_uses_test': False,
            'official_test_used_for_selection': False, 'data_identity': self.identity,
            'model_config': {'variant': 'fixture'}, 'status': 'training'}
        m.write_json(run / 'summary.json', summary)
        checkpoint = {'format': 'lusnar_vit_checkpoint_v1', 'epoch': best_epoch,
            'best_epoch': best_epoch, 'best_miou': best, 'partial': False,
            'train_config': self.configuration, 'data_identity': self.identity,
            'model_config': summary['model_config'], 'metrics': documents[best_epoch],
            'model': {'fixture': torch.tensor([best_epoch])}}
        temporary = run / 'best.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(run / 'best.pt')
        return run

    def prepare(self, run, epoch=3):
        summary, configuration, rows = m.committed_run(run)
        return m.prepare_job(run, epoch, summary, configuration, rows, self.config)

    def metrics(self, trigger):
        scenes = {}
        for scene, count in m.SCENES.items():
            each, remainder = divmod(count * 1024 * 1024, 5)
            matrix = [[each + (i < remainder) if i == j else 0 for j in range(5)] for i in range(5)]
            scenes[scene] = {'image_count': count, 'miou': 1.0,
                'class_iou': {name: 1.0 for name in m.CLASSES}, 'gt_present_classes': list(m.CLASSES),
                'confusion_matrix_gt_rows_pred_cols': matrix}
        return {'partial': False, 'checkpoint_partial_training': False, 'split': 'test',
            'image_count': 5004, 'validation_unique_source_count': 5004,
            'official_test_scope': True, 'test_scope': 'complete_official_test',
            'checkpoint_sha256': trigger['checkpoint_sha256'], 'checkpoint_epoch': trigger['checkpoint_epoch'],
            'tta': 'multiscale_hflip', 'tta_sizes': [1024, 768], 'tta_input_sizes': [1024, 768],
            'tta_scale_weights': [0.5, 0.5], 'tta_view_weights': [0.25] * 4,
            'tta_forward_calls_per_batch': 4,
            'data_identity': {'split': 'test', 'partial': False,
                'manifest_sha256': self.config['evaluation']['test_manifest_sha256']},
            'scenes': scenes, 'weighted_score': 1.0,
            'confusion_matrix_gt_rows_pred_cols': [[sum(row['confusion_matrix_gt_rows_pred_cols'][i][j]
                for row in scenes.values()) for j in range(5)] for i in range(5)],
            'class_iou': {name: 1.0 for name in m.CLASSES}, 'gt_present_classes': list(m.CLASSES),
            'miou': 1.0, 'pooled_miou': 1.0,
            'weighted_score_weights': {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}}

    def test_history_ahead_of_summary_does_not_trigger(self):
        run = self.publish(committed=2)
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', 'fixture')):
            m.Monitor(self.config).tick()
        self.assertFalse((run / 'periodic_test/epoch_003/status.json').exists())

    def test_atomic_snapshot_survives_replacement(self):
        run = self.publish()
        job = self.prepare(run)
        original = m.sha256(job / 'best.pt')
        self.publish(scores=(0.5, 0.6, 0.7, 0.8))
        self.assertEqual(m.sha256(job / 'best.pt'), original)
        self.assertNotEqual(m.sha256(run / 'best.pt'), original)
        self.assertEqual(m.checkpoint_metadata(job / 'best.pt')['epoch'], 3)

    def test_best_ahead_of_summary_retries_then_marks_missed(self):
        run = self.publish(scores=(0.5, 0.6, 0.7, 0.8), committed=3)
        job = self.prepare(run)
        self.assertFalse((job / 'status.json').exists())
        self.publish(scores=(0.5, 0.6, 0.7, 0.8))
        self.prepare(run)
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'missed')
        self.assertFalse((job / 'best.pt').exists())

    def test_same_checkpoint_reuse_and_snapshot_cleanup(self):
        run = self.publish()
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(job, trigger)
        self.assertFalse((job / 'best.pt').exists())
        self.publish(scores=(0.5, 0.6, 0.7, 0.69, 0.68, 0.67))
        later = self.prepare(run, 6)
        self.assertEqual(m.read_json(later / 'status.json')['status'], 'reused')
        self.assertEqual(m.read_json(later / 'status.json')['reused_from'], str(job))
        self.assertFalse((later / 'best.pt').exists())

    def test_snapshots_other_run_while_first_active_and_restart(self):
        first = self.prepare(self.publish())
        second = self.publish('c')
        m.set_status(first, 'running', worker_identity=m.process_identity(os.getpid()))
        with patch.object(m, 'gpu_available', side_effect=AssertionError('Must not schedule when active')):
            report = m.Monitor(self.config).tick()
            restarted = m.Monitor(self.config).tick()
        self.assertEqual(report['active_jobs'], [str(first)])
        self.assertEqual(restarted['active_jobs'], [str(first)])
        self.assertTrue((second / 'periodic_test/epoch_003/best.pt').exists())

    def test_incomplete_or_wrong_result_rejected(self):
        trigger = m.read_json(self.prepare(self.publish()) / 'trigger.json')
        valid = self.metrics(trigger)
        self.assertEqual(m.validate_metrics(valid, trigger)['weighted_score'], 1.0)
        for field, value in [('partial', True), ('image_count', 5003),
            ('checkpoint_sha256', 'bad'), ('checkpoint_epoch', 4),
            ('tta_scale_weights', [0.75, 0.25]), ('weighted_score', 0.9)]:
            with self.subTest(field=field):
                wrong = copy.deepcopy(valid)
                wrong[field] = value
                with self.assertRaises(ValueError):
                    m.validate_metrics(wrong, trigger)
        wrong = copy.deepcopy(valid)
        wrong['scenes']['Moon_9']['image_count'] -= 1
        with self.assertRaises(ValueError):
            m.validate_metrics(wrong, trigger)
        for field in ('scenes', 'confusion_matrix_gt_rows_pred_cols', 'pooled_miou'):
            with self.subTest(pixel_guard=field):
                wrong = copy.deepcopy(valid)
                if field == 'scenes':
                    wrong['scenes']['Moon_2']['confusion_matrix_gt_rows_pred_cols'][0][0] -= 1
                elif field == 'confusion_matrix_gt_rows_pred_cols':
                    wrong[field][0][0] -= 1
                else:
                    wrong[field] = 0.9
                with self.assertRaises(ValueError):
                    m.validate_metrics(wrong, trigger)

    def test_validation_identity_and_source_change_rejected(self):
        run = self.publish()
        document = m.read_json(run / 'validation_epoch_003.json')
        document['data_identity'] = {'different': True}
        m.write_json(run / 'validation_epoch_003.json', document)
        with self.assertRaises(ValueError):
            self.prepare(run)
        self.publish()
        (self.root / 'code/train.py').write_text('# changed\n')
        with self.assertRaises(ValueError):
            self.prepare(run)

    def test_worker_success_and_failure_never_change_trainer(self):
        run = self.publish()
        job = self.prepare(run)
        before = m.sha256(run / 'best.pt')
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'fixture_metrics.json', self.metrics(trigger))
        m.write_json(job / 'launch.json', {'configuration': self.config})
        self.assertEqual(m.worker(job, probe=lambda _: (True, 'fixture only', 'GPU-fixture')), 0)
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')
        self.assertEqual(m.sha256(run / 'best.pt'), before)
        other = self.prepare(self.publish('c'))
        m.write_json(other / 'launch.json', {'configuration': self.config})
        self.assertEqual(m.worker(other, probe=lambda _: (True, 'fixture only', 'GPU-fixture')), 1)
        self.assertEqual(m.read_json(other / 'status.json')['status'], 'failed')
        self.assertTrue((other / 'best.pt').exists())

    def test_lock_and_pid_identity(self):
        path = self.root / 'lock'
        lock = m.acquire_lock(path)
        self.assertIsNotNone(lock)
        self.assertIsNone(m.acquire_lock(path))
        lock.close()
        identity = m.process_identity(os.getpid())
        self.assertTrue(m.alive(identity))
        self.assertFalse(m.alive({**identity, 'start_ticks': '-1'}))
        job = self.root / 'result/job'
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(1)',
                                    'periodic_evaluate.py', '--worker', str(job)])
        try:
            self.assertTrue(any(p['pid'] == process.pid for p in m.job_processes(job)))
        finally:
            process.wait()

    def test_recover_evaluation_complete_after_worker_disappears(self):
        job = self.prepare(self.publish())
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        m.set_status(job, 'running', worker_identity={'pid': 99999999, 'start_ticks': '0', 'boot_id': 'none'})
        m.Monitor(self.config).tick()
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')
        self.assertFalse((job / 'best.pt').exists())

    def test_inherited_gpu_lock_survives_worker_exit(self):
        lock_path = self.root / 'inherited.lock'
        child_pid = self.root / 'child_pid.txt'
        helper = ('import fcntl,subprocess,sys;'
                  'stream=open(sys.argv[1],"a+");'
                  'fcntl.flock(stream,fcntl.LOCK_EX);'
                  'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(1)"],'
                  'pass_fds=(stream.fileno(),));'
                  'open(sys.argv[2],"w").write(str(child.pid))')
        subprocess.run([sys.executable, '-c', helper, str(lock_path), str(child_pid)], check=True)
        self.assertIsNone(m.acquire_lock(lock_path))
        identity = m.process_identity(int(child_pid.read_text()))
        deadline = time.monotonic() + 5
        while m.alive(identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        lock = m.acquire_lock(lock_path)
        self.assertIsNotNone(lock)
        lock.close()

    def test_terminal_off_cycle_gets_separate_final_label(self):
        run = self.publish(scores=(0.5, 0.6, 0.7, 0.8))
        summary = m.read_json(run / 'summary.json')
        summary['status'] = 'early_stopped'
        m.write_json(run / 'summary.json', summary)
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', 'fixture')):
            m.Monitor(self.config).tick()
        final = run / 'periodic_test/final_epoch_004'
        self.assertEqual(m.read_json(final / 'trigger.json')['checkpoint_epoch'], 4)
        self.assertEqual(m.read_json(final / 'trigger.json')['trigger_kind'], 'final')

    def test_configurable_five_epoch_cadence_and_final(self):
        self.config['every_epochs'] = 5
        config_path = self.root / 'config.json'
        m.write_json(config_path, self.config)
        self.assertEqual(m.load_config(config_path)['every_epochs'], 5)
        m.write_json(config_path, {key: value for key, value in self.config.items() if key != 'every_epochs'})
        self.assertEqual(m.load_config(config_path)['every_epochs'], 5)
        for invalid in (0, -1, True, 2.5):
            bad = {**self.config, 'every_epochs': invalid}
            m.write_json(config_path, bad)
            with self.assertRaises(ValueError):
                m.load_config(config_path)
        monitor = m.Monitor(self.config)
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', 'fixture')):
            run = self.publish(scores=(0.5, 0.6, 0.7, 0.8))
            monitor.tick()
            self.assertFalse((run / 'periodic_test').exists())
            self.publish(scores=(0.5, 0.6, 0.7, 0.8, 0.9))
            monitor.tick()
            first = run / 'periodic_test/epoch_005'
            self.assertEqual(m.read_json(first / 'trigger.json')['every_epochs'], 5)
            first_sha = m.sha256(first / 'best.pt')
            self.publish(scores=(0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92))
            monitor.tick()
            self.assertEqual(m.sha256(first / 'best.pt'), first_sha)
            self.assertFalse((run / 'periodic_test/epoch_006').exists())
            summary = m.read_json(run / 'summary.json')
            summary['status'] = 'early_stopped'
            m.write_json(run / 'summary.json', summary)
            monitor.tick()
            final = run / 'periodic_test/final_epoch_007'
            self.assertEqual(m.read_json(final / 'trigger.json')['checkpoint_epoch'], 7)
            self.assertEqual(m.read_json(final / 'trigger.json')['trigger_kind'], 'final')
            self.publish(scores=(0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95))
            monitor.tick()
            self.assertTrue((run / 'periodic_test/epoch_010/status.json').exists())
            self.assertFalse((run / 'periodic_test/epoch_009').exists())

    def test_new_five_epoch_monitor_preserves_active_old_job(self):
        run = self.publish()
        old_job = self.prepare(run)
        old_sha = m.sha256(old_job / 'best.pt')
        m.set_status(old_job, 'running', worker_identity=m.process_identity(os.getpid()))
        config = {**self.config, 'every_epochs': 5,
                  'state_directory': str(self.root / 'result/monitor_every5')}
        with patch.object(m, 'gpu_available', side_effect=AssertionError('Old worker must continue')):
            report = m.Monitor(config).tick()
        self.assertEqual(report['active_jobs'], [str(old_job)])
        self.assertEqual(m.sha256(old_job / 'best.pt'), old_sha)
        self.assertEqual(m.read_json(old_job / 'status.json')['status'], 'running')
        self.assertEqual(list((run / 'periodic_test').iterdir()), [old_job])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PeriodicFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    destination = Path(__file__).resolve().parents[1] / 'result/verification_periodic_evaluate.json'
    m.write_json(destination, {'schema': 'vit_periodic_cpu_fixture_checks_v1', 'created_at': m.now(),
        'success': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'scope': 'Synthetic CPU control-flow fixtures only; no real model or test evaluation',
        'source_sha256': m.sha256(Path(__file__).with_name('periodic_evaluate.py'))})
    raise SystemExit(0 if result.wasSuccessful() else 1)
