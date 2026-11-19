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
from unittest.mock import Mock, patch

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
        # GPU admission is covered independently with mocked nvidia-smi in
        # verify_gpu_slots.py. These fixtures exercise real inherited file locks
        # and CPU evaluator processes without probing any physical GPU.
        self.slot_patch = patch.object(m, 'acquire_gpu_slot',
                                       side_effect=lambda config: m.acquire_lock(config['gpu_lock']))
        self.slot_patch.start()

    def tearDown(self):
        self.slot_patch.stop()
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

    def test_hflip_protocol_validates_only_native_two_view_results(self):
        self.config['evaluation_tta'] = 'hflip'
        self.config['evaluation'] = m.settings(self.config)
        run = self.publish()
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        document = self.metrics(trigger)
        with self.assertRaises(ValueError):
            m.validate_metrics(document, trigger)
        document.update(tta='hflip', tta_mode='hflip', tta_sizes=None, tta_weights=None,
                        tta_views=['original', 'horizontal_flip'], tta_forward_calls_per_batch=2,
                        model_config={'input_size': 1024})
        for key in ('tta_input_sizes', 'tta_scale_weights', 'tta_view_weights'):
            document.pop(key)
        self.assertEqual(m.validate_metrics(document, trigger)['image_count'], 5004)
        command = m.evaluator_command(self.config, job)
        self.assertEqual(command[command.index('--tta') + 1], 'hflip')
        self.assertNotIn('--tta-sizes', command)
        document['model_config']['input_size'] = 768
        with self.assertRaises(ValueError):
            m.validate_metrics(document, trigger)

    def test_changing_future_tta_preserves_previously_queued_protocol(self):
        run = self.publish()
        job = self.prepare(run)
        digest = m.sha256(job / 'trigger.json')
        self.config['evaluation_tta'] = 'hflip'
        self.config['evaluation'] = m.settings(self.config)
        self.prepare(run)
        self.assertEqual(m.sha256(job / 'trigger.json'), digest)
        command = m.evaluator_command(self.config, job)
        self.assertEqual(command[command.index('--tta') + 1], 'multiscale_hflip')
        self.assertIn('--tta-sizes', command)
        m.validate_metrics(self.metrics(m.read_json(job / 'trigger.json')), m.read_json(job / 'trigger.json'))
        self.publish(scores=(.5, .6, .7, .8, .9, .95))
        newjob = self.prepare(run, 6)
        self.assertEqual(m.read_json(newjob / 'trigger.json')['evaluation']['tta'], 'hflip')

    def test_unsupported_inference_policy_is_rejected(self):
        for mode in ('none', 'invalid'):
            with self.assertRaises(ValueError):
                m.settings({**self.config, 'evaluation_tta': mode})
        with self.assertRaises(ValueError):
            m.tta_arguments({'tta': 'hflip', 'tta_sizes': [1024, 768], 'tta_weights': [0.5, 0.5]})

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

    def test_final_queued_before_periodic_finishes_reuses_at_dispatch(self):
        run = self.publish()
        self.config['runs'] = [str(run)]
        summary = m.read_json(run / 'summary.json')
        summary['status'] = 'complete'
        m.write_json(run / 'summary.json', summary)
        monitor = m.Monitor(self.config)
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', None)):
            monitor.tick()
        first, final = run / 'periodic_test/epoch_003', run / 'periodic_test/final_epoch_003'
        self.assertEqual(m.read_json(final / 'status.json')['status'], 'queued')
        trigger = m.read_json(first / 'trigger.json')
        m.write_json(first / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(first, trigger)
        with patch.object(m.subprocess, 'Popen') as launch:
            report = monitor.tick()
        launch.assert_not_called()
        self.assertFalse(report['errors'])
        state = m.read_json(final / 'status.json')
        self.assertEqual(state['status'], 'reused')
        self.assertEqual(state['reused_from'], str(first))
        self.assertEqual(m.sha256(first / 'test_metrics.json'), m.sha256(final / 'test_metrics.json'))
        self.assertFalse((final / 'best.pt').exists())

    def test_queued_reuse_requires_same_weights_and_protocol(self):
        run = self.publish()
        first = self.prepare(run)
        summary, configuration, rows = m.committed_run(run)
        final = m.prepare_job(run, 3, summary, configuration, rows, self.config, final=True)
        trigger = m.read_json(first / 'trigger.json')
        m.write_json(first / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(first, trigger)
        original = m.read_json(final / 'trigger.json')
        for field in ('checkpoint_sha256', 'evaluation', 'configuration_sha256'):
            changed = copy.deepcopy(original)
            if field == 'evaluation':
                changed[field]['tta'] = ('hflip' if original[field]['tta'] != 'hflip'
                                         else 'multiscale_hflip')
            else:
                changed[field] = 'different'
            m.write_json(final / 'trigger.json', changed)
            self.assertIsNone(m.reuse_completed_result(final))
            self.assertFalse((final / 'test_metrics.json').exists())
        m.write_json(final / 'trigger.json', original)
        broken = self.metrics(trigger)
        broken['image_count'] = 1
        m.write_json(first / 'test_metrics.json', broken)
        with self.assertRaises(ValueError):
            m.reuse_completed_result(final)
        self.assertFalse((final / 'test_metrics.json').exists())

    def test_score_retention_requires_strictly_greater_than_threshold(self):
        # The fixture score is exactly 1.0, so equality cannot be obscured by
        # floating-point rounding in an invented 0.92 confusion matrix.
        for threshold, retain in ((0.92, True), (1.0, False)):
            with self.subTest(threshold=threshold):
                self.config['retain_above_score'] = threshold
                job = self.prepare(self.publish(name=f'threshold_{threshold}'))
                trigger = m.read_json(job / 'trigger.json')
                original = m.sha256(job / 'best.pt')
                m.write_json(job / 'test_metrics.json', self.metrics(trigger))
                state = m.complete_job(job, trigger)
                self.assertEqual(state['status'], 'complete')
                self.assertEqual((job / 'best.pt').exists(), retain)
                self.assertEqual((job / 'score_milestone.json').exists(), retain)
                if retain:
                    self.assertEqual(m.sha256(job / 'best.pt'), original)

    def test_score_retention_rejects_unverified_metrics(self):
        self.config['retain_above_score'] = 0.92
        for field, value in (('image_count', 5003), ('checkpoint_sha256', 'wrong'),
                             ('weighted_score', 0.99), ('partial', True)):
            with self.subTest(field=field):
                job = self.prepare(self.publish(name=f'invalid_{field}'))
                trigger = m.read_json(job / 'trigger.json')
                invalid = self.metrics(trigger)
                invalid[field] = value
                m.write_json(job / 'test_metrics.json', invalid)
                with self.assertRaises(ValueError):
                    m.complete_job(job, trigger)
                self.assertFalse((job / 'score_milestone.json').exists())
                self.assertTrue((job / 'best.pt').exists())
                self.assertEqual(m.read_json(job / 'status.json')['status'], 'queued')

    def test_score_retention_rejects_wrong_checkpoint_bytes(self):
        self.config['retain_above_score'] = 0.92
        job = self.prepare(self.publish())
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        # Atomic replacement leaves the trainer's original hardlink untouched.
        replacement = job / 'wrong.tmp'
        replacement.write_bytes(b'wrong-checkpoint-fixture')
        replacement.replace(job / 'best.pt')
        with self.assertRaises(ValueError):
            m.complete_job(job, trigger)
        self.assertFalse((job / 'score_milestone.json').exists())

    def test_reused_score_retains_matching_checkpoint_in_each_job(self):
        self.config['retain_above_score'] = 0.92
        run = self.publish()
        first = self.prepare(run)
        trigger = m.read_json(first / 'trigger.json')
        original = m.sha256(first / 'best.pt')
        m.write_json(first / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(first, trigger)
        self.publish(scores=(0.5, 0.6, 0.7, 0.69, 0.68, 0.67))
        later = self.prepare(run, 6)
        self.assertEqual(m.read_json(later / 'status.json')['status'], 'reused')
        self.assertEqual(m.read_json(later / 'status.json')['reused_from'], str(first))
        for job in (first, later):
            self.assertTrue((job / 'score_milestone.json').exists())
            self.assertEqual(m.sha256(job / 'best.pt'), original)

    def test_reused_score_restores_previously_cleaned_snapshot(self):
        run = self.publish()
        first = self.prepare(run)
        trigger = m.read_json(first / 'trigger.json')
        original = m.sha256(first / 'best.pt')
        m.write_json(first / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(first, trigger)
        self.assertFalse((first / 'best.pt').exists())
        # Previously completed jobs may predate the retention option. Reuse
        # must recover exact bytes from the still-identical run best, not a
        # checkpoint selected at a later epoch.
        self.publish(scores=(0.5, 0.6, 0.7, 0.69, 0.68, 0.67))
        self.assertEqual(m.sha256(run / 'best.pt'), original)
        self.config['retain_above_score'] = 0.92
        later = self.prepare(run, 6)
        self.assertEqual(m.read_json(later / 'status.json')['status'], 'reused')
        self.assertEqual(m.sha256(later / 'best.pt'), original)
        self.assertTrue((later / 'score_milestone.json').exists())

    def test_restart_preserves_retained_snapshot_even_when_option_disabled(self):
        self.config['retain_above_score'] = 0.92
        run = self.publish()
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        original = m.sha256(job / 'best.pt')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(job, trigger)
        m.Monitor(self.config).tick()
        self.assertTrue((run / 'score_milestones.json').exists())
        restarted_config = {**self.config, 'retain_above_score': None}
        m.Monitor(restarted_config).tick()
        m.Monitor(restarted_config).tick()
        self.assertEqual(m.sha256(job / 'best.pt'), original)
        self.assertTrue((job / 'score_milestone.json').exists())
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')

    def test_disappeared_worker_recovery_retains_verified_milestone(self):
        self.config['retain_above_score'] = 0.92
        job = self.prepare(self.publish())
        trigger = m.read_json(job / 'trigger.json')
        original = m.sha256(job / 'best.pt')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        m.set_status(job, 'running', worker_identity={'pid': 99999999, 'start_ticks': '0', 'boot_id': 'none'})
        m.Monitor(self.config).tick()
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')
        self.assertEqual(m.sha256(job / 'best.pt'), original)
        self.assertTrue((job / 'score_milestone.json').exists())

    def test_retention_config_threshold_validation(self):
        path = self.root / 'retention_config.json'
        for value in (None, 0.0, 0.92, 1.0):
            m.write_json(path, {**self.config, 'retain_above_score': value})
            self.assertEqual(m.load_config(path)['retain_above_score'], value)
        for invalid in (-0.01, 1.01, True, '0.92', float('nan'), float('inf')):
            with self.subTest(invalid=invalid):
                # Bypass the production writer so malformed external JSON
                # reaches the configuration reader's own validation.
                path.write_text(json.dumps({**self.config, 'retain_above_score': invalid}))
                with self.assertRaises(ValueError):
                    m.load_config(path)

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

    def test_final_only_skips_all_boundaries_and_includes_exact_multiple(self):
        self.config.update(every_epochs=5, periodic_evaluation=False)
        run = self.publish(scores=tuple(.5 + .01 * i for i in range(10)))
        monitor = m.Monitor(self.config)
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', 'fixture')):
            monitor.tick()
            self.assertFalse((run / 'periodic_test').exists())
            summary = m.read_json(run / 'summary.json')
            summary['status'] = 'early_stopped'
            m.write_json(run / 'summary.json', summary)
            monitor.tick()
            final = run / 'periodic_test/final_epoch_010'
            self.assertEqual(m.read_json(final / 'trigger.json')['checkpoint_epoch'], 10)
            self.assertEqual(m.read_json(final / 'trigger.json')['trigger_kind'], 'final')
            self.assertEqual(list((run / 'periodic_test').iterdir()), [final])
            monitor.tick()
            self.assertEqual(list((run / 'periodic_test').iterdir()), [final])

    def test_final_only_preserves_active_worker_and_user_stopped_exclusion(self):
        run = self.publish()
        old_job = self.prepare(run)
        m.set_status(old_job, 'running', worker_identity=m.process_identity(os.getpid()))
        other = self.publish(name='c', scores=(.5, .6, .7, .8, .9))
        summary = m.read_json(other / 'summary.json')
        summary['status'] = 'user_stopped'
        m.write_json(other / 'summary.json', summary)
        self.config.update(periodic_evaluation=False)
        with patch.object(m, 'gpu_available', side_effect=AssertionError('Active worker continues')):
            report = m.Monitor(self.config).tick()
        self.assertEqual(report['active_jobs'], [str(old_job)])
        self.assertFalse((other / 'periodic_test').exists())

    def test_schedule_config_boolean_and_shared_lock(self):
        config_path = self.root / 'config.json'
        custom_lock = str(self.root / 'shared_gpu.lock')
        m.write_json(config_path, {**self.config, 'periodic_evaluation': False, 'gpu_lock': custom_lock})
        loaded = m.load_config(config_path)
        self.assertFalse(loaded['periodic_evaluation'])
        self.assertEqual(loaded['gpu_lock'], custom_lock)
        for key in ('periodic_evaluation', 'final_evaluation'):
            m.write_json(config_path, {**self.config, key: 'false'})
            with self.assertRaises(ValueError):
                m.load_config(config_path)

    def test_two_slot_configuration_is_explicit_and_bounded(self):
        path = self.root / 'config.json'
        m.write_json(path, self.config)
        self.assertEqual(m.load_config(path)['max_concurrent_evaluations'], 1)
        m.write_json(path, {**self.config, 'max_concurrent_evaluations': 2})
        self.assertEqual(m.load_config(path)['max_concurrent_evaluations'], 2)
        for value in (0, 3, True, '2'):
            with self.subTest(value=value):
                m.write_json(path, {**self.config, 'max_concurrent_evaluations': value})
                with self.assertRaises(ValueError):
                    m.load_config(path)

    def test_resource_migration_preserves_state_but_rejects_inference_changes(self):
        m.Monitor(self.config)
        directory = Path(self.config['state_directory'])
        marker = directory / 'state.json'
        marker.write_text('{"history":"preserved"}')
        updated = {**self.config, 'max_concurrent_evaluations': 2,
                   'minimum_free_mib': 12000, 'poll_seconds': 15}
        m.Monitor(updated)
        self.assertEqual(m.read_json(marker), {'history': 'preserved'})
        archives = list(directory.glob('configuration_before_resources_*.json'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(m.read_json(archives[0]), self.config)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            m.Monitor({**updated, 'evaluation_batch_size': 1})

    def test_two_slots_schedule_other_run_while_first_is_active(self):
        first = self.prepare(self.publish())
        second_run = self.publish('c')
        self.config['max_concurrent_evaluations'] = 2
        m.set_status(first, 'running', worker_identity=m.process_identity(os.getpid()))
        child = Mock(pid=123456)
        child.poll.return_value = None
        with patch.object(m, 'gpu_available', return_value=(True, 'slot1 free', 'GPU-fixture')), \
             patch.object(m.subprocess, 'Popen', return_value=child) as spawn:
            report = m.Monitor(self.config).tick()
        self.assertEqual(len(report['active_jobs']), 2)
        self.assertIn(str(first), report['active_jobs'])
        self.assertIn(str(second_run / 'periodic_test/epoch_003'), report['active_jobs'])
        self.assertEqual(spawn.call_count, 1)

    def test_two_slots_do_not_overlap_epochs_of_same_run(self):
        run = self.publish()
        first = self.prepare(run)
        self.publish(scores=(.5, .6, .7, .71, .72, .73))
        self.config['runs'] = [str(run)]
        self.config['max_concurrent_evaluations'] = 2
        m.set_status(first, 'running', worker_identity=m.process_identity(os.getpid()))
        with patch.object(m, 'gpu_available', side_effect=AssertionError('Same run remains serial')):
            report = m.Monitor(self.config).tick()
        self.assertEqual(report['active_jobs'], [str(first)])
        self.assertEqual(m.read_json(run / 'periodic_test/epoch_006/status.json')['status'], 'queued')

    def test_worker_inherits_the_selected_second_slot(self):
        run = self.publish()
        job = self.prepare(run)
        self.config['max_concurrent_evaluations'] = 2
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'fixture_metrics.json', self.metrics(trigger))
        m.write_json(job / 'launch.json', {'configuration': self.config})
        handle = m.acquire_lock(self.config['gpu_lock'] + '.slot1')
        descriptor = handle.fileno()
        original_popen = m.subprocess.Popen
        with patch.object(m, 'acquire_gpu_slot', return_value=handle) as acquire, \
             patch.object(m.subprocess, 'Popen', wraps=original_popen) as spawn:
            self.assertEqual(m.worker(job, probe=lambda _: (True, 'fixture slot1', 'GPU-fixture')), 0)
        self.assertEqual(acquire.call_args.args[0]['max_concurrent_evaluations'], 2)
        self.assertIn(descriptor, spawn.call_args.kwargs['pass_fds'])
        self.assertTrue(handle.closed)


class HardValidationFixtures(unittest.TestCase):
    """Sparse hard-validation history with deliberately different pooled mIoU."""

    setUp = PeriodicFixtures.setUp
    tearDown = PeriodicFixtures.tearDown

    def publish_hard(self, name='b', scores=None, completed=5, committed=None, status='training'):
        if scores is None:
            scores = {2: 0.70, 4: 0.80}
        self.config['every_epochs'] = 5
        self.configuration.update(protocol='pool8002_train6000_hard1500_validation_v1',
                                  validation_every=2, pretrained=True)
        counts = {'Moon_2': 188, 'Moon_4': 187, 'Moon_9': 1125}
        weights = {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}
        self.identity = {
            'train': {'split': 'train', 'image_count': 6000, 'partial': False},
            'val': {'split': 'val', 'image_count': 1500, 'unique_source_count': 1500,
                    'partial': False, 'selection_metric': 'validation_weighted_score',
                    'hard_manifest_sha256': 'fixed-fixture-hard1500',
                    'ordered_samples_sha256': 'fixed-fixture-validation',
                    'validation_image_count': 1500, 'validation_scene_counts': counts,
                    'sample_scene_counts': counts, 'validation_weights': weights,
                    'checkpoint_selection_uses_test': True, 'official_test_used_for_selection': True,
                    'full_original_test_overlaps_validation': True}}
        run = self.root / 'result' / name
        run.mkdir(parents=True, exist_ok=True)
        m.write_json(run / 'run_config.json', self.configuration)
        rows, documents, best, best_epoch = [], {}, None, None
        for epoch in range(1, completed + 1):
            score = scores.get(epoch)
            pooled = 0.95 - epoch * 0.001 if score is not None else None
            if score is not None:
                if best is None or score > best:
                    best, best_epoch = score, epoch
                validation = {
                    'epoch': epoch, 'image_count': 1500, 'validation_unique_source_count': 1500,
                    'partial': False, 'checkpoint_selection_uses_test': True,
                    'official_test_used_for_selection': True, 'official_test_scope': False,
                    'test_scope': 'fixed_hard_validation', 'best_metric': 'validation_weighted_score',
                    'data_identity': self.identity['val'], 'miou': pooled, 'weighted_score': None,
                    'validation_weighted_score': score, 'validation_weighted_score_weights': weights,
                    'scenes': {scene: {'image_count': count, 'miou': score}
                               for scene, count in counts.items()}}
                documents[epoch] = validation
                m.write_json(run / f'validation_epoch_{epoch:03d}.json', validation)
            rows.append({'epoch': epoch, 'validation_miou': pooled, 'miou': pooled,
                         'validation_weighted_score': score, 'validation_performed': score is not None,
                         'best_epoch': best_epoch, 'best_miou': best, 'best_validation_score': best,
                         'best_metric': 'validation_weighted_score', 'partial': False,
                         'checkpoint_selection_uses_test': True})
        (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        durable = completed if committed is None else committed
        row = rows[durable - 1]
        summary = {'format': 'lusnar_vit_run_v1', 'status': status, 'completed_epochs': durable,
                   'best_epoch': row['best_epoch'], 'best_miou': row['best_miou'],
                   'best_validation_score': row['best_miou'], 'best_metric': 'validation_weighted_score',
                   'partial': False, 'train_images': 6000, 'validation_images': 1500,
                   'validation_interval': 2, 'checkpoint_selection_uses_test': True,
                   'official_test_used_for_selection': True, 'data_identity': self.identity,
                   'model_config': {'variant': 'fixture'}}
        m.write_json(run / 'summary.json', summary)
        if best_epoch is not None:
            checkpoint = {'format': 'lusnar_vit_checkpoint_v1', 'epoch': best_epoch,
                          'best_epoch': best_epoch, 'best_miou': best, 'partial': False,
                          'train_config': self.configuration, 'data_identity': self.identity,
                          'model_config': summary['model_config'], 'metrics': documents[best_epoch],
                          'model': {'fixture': torch.tensor([best_epoch])}}
            temporary = run / 'best.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(run / 'best.pt')
        return run

    def prepare(self, run, boundary=5):
        summary, configuration, rows = m.committed_run(run)
        return m.prepare_job(run, boundary, summary, configuration, rows, self.config)

    def metrics(self, trigger):
        document = PeriodicFixtures.metrics(self, trigger)
        document.update(protocol='pool8002_train6000_hard1500_validation_v1',
                        test_scope='full5004_including_selection_validation',
                        official_test_scope=False, official_full_test=False,
                        checkpoint_selection_uses_test=True, official_test_used_for_selection=True,
                        independent_holdout=False, validation_images_included=1500,
                        full_test_coverage=True)
        return document

    def test_epoch_five_uses_best_four_without_validation_five(self):
        run = self.publish_hard()
        self.assertFalse((run / 'validation_epoch_005.json').exists())
        summary, _, rows = m.committed_run(run)
        self.assertEqual(summary['best_epoch'], 4)
        self.assertGreater(rows[2]['validation_miou'], rows[4]['validation_miou'])
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        self.assertEqual(trigger['trigger_epoch'], 5)
        self.assertEqual(trigger['checkpoint_epoch'], 4)
        self.assertEqual(trigger['validation_best_miou'], 0.80)
        self.assertEqual(trigger['best_metric'], 'validation_weighted_score')
        self.assertTrue(trigger['checkpoint_selection_uses_test'])
        self.assertEqual(trigger['protocol'], self.configuration['protocol'])
        self.assertEqual(m.checkpoint_metadata(job / 'best.pt')['epoch'], 4)

    def test_five_epoch_monitor_accepts_sparse_history(self):
        run = self.publish_hard()
        with patch.object(m, 'gpu_available', return_value=(False, 'fixture busy', 'fixture')):
            report = m.Monitor(self.config).tick()
        self.assertEqual(report['errors'], {})
        self.assertTrue((run / 'periodic_test/epoch_005/best.pt').exists())
        self.assertFalse((run / 'periodic_test/epoch_004').exists())

    def test_sparse_history_rejects_wrong_schedule_flags_and_metric(self):
        cases = ('odd_marked_validated', 'even_not_validated', 'odd_has_score',
                 'selection_false', 'official_false', 'pooled_best_metric', 'history_selection_false')
        for case in cases:
            with self.subTest(case=case):
                run = self.publish_hard(name=case)
                summary = m.read_json(run / 'summary.json')
                rows = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
                if case == 'odd_marked_validated': rows[2]['validation_performed'] = True
                elif case == 'even_not_validated':
                    rows[3].update(validation_weighted_score=None, validation_miou=None,
                                   validation_performed=False)
                elif case == 'odd_has_score': rows[2]['validation_weighted_score'] = 0.75
                elif case == 'selection_false': summary['checkpoint_selection_uses_test'] = False
                elif case == 'official_false': summary['official_test_used_for_selection'] = False
                elif case == 'pooled_best_metric': summary['best_metric'] = 'miou'
                elif case == 'history_selection_false': rows[3]['checkpoint_selection_uses_test'] = False
                m.write_json(run / 'summary.json', summary)
                (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError): m.committed_run(run)

    def test_missing_even_validation_document_and_bad_identity_rejected(self):
        run = self.publish_hard()
        (run / 'validation_epoch_004.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.prepare(run)
        for field, value in (('checkpoint_selection_uses_test', False), ('image_count', 1499),
                             ('validation_weighted_score', 0.946), ('validation_unique_source_count', 1499)):
            with self.subTest(field=field):
                run = self.publish_hard(name='invalid_' + field)
                document = m.read_json(run / 'validation_epoch_004.json')
                document[field] = value
                m.write_json(run / 'validation_epoch_004.json', document)
                with self.assertRaises(ValueError): self.prepare(run)

    def test_odd_terminal_requires_its_final_validation(self):
        run = self.publish_hard(status='early_stopped')
        with self.assertRaises(ValueError): m.committed_run(run)
        self.configuration['epochs'] = 5
        run = self.publish_hard(name='final_validated', scores={2: .70, 4: .80, 5: .81},
                                status='complete')
        summary, configuration, rows = m.committed_run(run)
        job = m.prepare_job(run, 5, summary, configuration, rows, self.config, final=True)
        self.assertEqual(m.read_json(job / 'trigger.json')['checkpoint_epoch'], 5)

    def test_same_best_is_reused_at_epoch_ten(self):
        run = self.publish_hard()
        first = self.prepare(run)
        trigger = m.read_json(first / 'trigger.json')
        m.write_json(first / 'test_metrics.json', self.metrics(trigger))
        m.complete_job(first, trigger)
        self.assertFalse((first / 'best.pt').exists())
        self.publish_hard(completed=10, scores={2: .70, 4: .80, 6: .78, 8: .80, 10: .79})
        later = self.prepare(run, 10)
        state = m.read_json(later / 'status.json')
        self.assertEqual(state['status'], 'reused')
        self.assertEqual(state['reused_from'], str(first))
        self.assertEqual(state['checkpoint_sha256'], trigger['checkpoint_sha256'])
        self.assertEqual(state['checkpoint_epoch'], 4)

    def test_snapshot_stays_fixed_when_best_advances(self):
        run = self.publish_hard()
        job = self.prepare(run)
        original = m.sha256(job / 'best.pt')
        self.publish_hard(completed=6, scores={2: .70, 4: .80, 6: .85})
        self.assertEqual(m.sha256(job / 'best.pt'), original)
        self.assertNotEqual(m.sha256(run / 'best.pt'), original)
        self.assertEqual(m.checkpoint_metadata(job / 'best.pt')['epoch'], 4)

    def test_newer_checkpoint_cannot_replace_missed_boundary(self):
        run = self.publish_hard(completed=6, committed=5, scores={2: .70, 4: .80, 6: .85})
        job = self.prepare(run)
        self.assertFalse((job / 'status.json').exists())
        self.publish_hard(completed=6, scores={2: .70, 4: .80, 6: .85})
        self.prepare(run)
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'missed')
        self.assertFalse((job / 'best.pt').exists())

    def test_full_metrics_reject_false_independence_and_changed_identity(self):
        trigger = m.read_json(self.prepare(self.publish_hard()) / 'trigger.json')
        valid = self.metrics(trigger)
        self.assertEqual(m.validate_metrics(valid, trigger)['weighted_score'], 1.0)
        for field, value in (('protocol', 'different-protocol'), ('official_test_scope', True),
                             ('test_scope', 'complete_official_test'),
                             ('checkpoint_selection_uses_test', False),
                             ('official_test_used_for_selection', False), ('independent_holdout', True),
                             ('validation_images_included', 0), ('full_test_coverage', False),
                             ('checkpoint_sha256', 'different-checkpoint'), ('checkpoint_epoch', 6),
                             ('image_count', 5003), ('partial', True), ('weighted_score', .99)):
            with self.subTest(field=field):
                document = copy.deepcopy(valid)
                document[field] = value
                with self.assertRaises(ValueError): m.validate_metrics(document, trigger)

    def test_retained_hard_protocol_snapshot_requires_exact_bytes(self):
        self.config['retain_above_score'] = .92
        job = self.prepare(self.publish_hard())
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'test_metrics.json', self.metrics(trigger))
        replacement = job / 'wrong.tmp'
        replacement.write_bytes(b'not-the-tested-checkpoint')
        replacement.replace(job / 'best.pt')
        with self.assertRaises(ValueError): m.complete_job(job, trigger)
        self.assertFalse((job / 'score_milestone.json').exists())


class FullTestPolicyFixtures(unittest.TestCase):
    """Every-five-epoch selection uses exact current weights and full-test scores."""

    setUp = PeriodicFixtures.setUp
    tearDown = PeriodicFixtures.tearDown
    prepare = HardValidationFixtures.prepare

    def publish_full(self, name='b', completed=5, committed=None, status='training',
                     retain_candidate=True):
        self.configuration.update(selection_policy='full_test_every5',
                                  early_stopping_patience=0)
        run = HardValidationFixtures.publish_hard(
            self, name=name, completed=completed, committed=committed, status=status)
        saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        # Keep best_epoch=4 deliberately: this is the old diagnostic-validation
        # best, while the model tensor and epoch identify the current weights.
        saved.update(epoch=completed, metrics=None,
                     model={'fixture': torch.tensor([completed])})
        temporary = run / 'last.tmp'
        torch.save(saved, temporary)
        temporary.replace(run / 'last.pt')
        if retain_candidate:
            directory = run / 'full_test_checkpoints'
            directory.mkdir(exist_ok=True)
            candidate = directory / f'epoch_{completed:03d}.pt'
            m.snapshot(run / 'last.pt', candidate)
            m.write_json(candidate.with_suffix('.json'), {
                'schema': 'lusnar_full_test_candidate_receipt_v1',
                'checkpoint_epoch': completed, 'checkpoint_path': str(candidate),
                'checkpoint_sha256': m.sha256(candidate)})
        return run

    def metrics(self, trigger, correct_tenths=8):
        document = HardValidationFixtures.metrics(self, trigger)

        def measures(matrix):
            ious = {name: matrix[i][i] /
                    (sum(matrix[i]) + sum(row[i] for row in matrix) - matrix[i][i])
                    for i, name in enumerate(m.CLASSES)}
            return ious, sum(ious.values()) / len(ious)

        for scene in document['scenes'].values():
            matrix = scene['confusion_matrix_gt_rows_pred_cols']
            for i, row in enumerate(matrix):
                gt = sum(row)
                row[:] = [0] * 5
                row[i] = gt * correct_tenths // 10
                row[(i + 1) % 5] = gt - row[i]
            scene['class_iou'], scene['miou'] = measures(matrix)
        pooled = [[sum(scene['confusion_matrix_gt_rows_pred_cols'][i][j]
                       for scene in document['scenes'].values())
                   for j in range(5)] for i in range(5)]
        document['confusion_matrix_gt_rows_pred_cols'] = pooled
        document['class_iou'], document['miou'] = measures(pooled)
        document['pooled_miou'] = document['miou']
        scene = document['scenes']
        document['weighted_score'] = (scene['Moon_2']['miou'] + scene['Moon_4']['miou']
                                      + 2 * scene['Moon_9']['miou']) / 4
        return document

    def finish(self, job, correct_tenths=8):
        trigger = m.read_json(job / 'trigger.json')
        metrics = self.metrics(trigger, correct_tenths)
        m.write_json(job / 'test_metrics.json', metrics)
        state = m.complete_job(job, trigger)
        self.assertEqual(state['status'], 'complete')
        return trigger, metrics

    def test_epoch_five_uses_current_five_and_needs_no_validation_five(self):
        run = self.publish_full()
        self.assertEqual(m.checkpoint_metadata(run / 'best.pt')['epoch'], 4)
        self.assertFalse((run / 'validation_epoch_005.json').exists())
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        self.assertEqual(trigger['checkpoint_epoch'], 5)
        self.assertEqual(trigger['checkpoint_choice'], 'current_epoch')
        self.assertEqual(trigger['best_metric'], 'weighted_score')
        self.assertIsNone(trigger['validation_best_miou'])
        saved = m.checkpoint_metadata(job / 'best.pt')
        self.assertEqual(saved['epoch'], 5)
        self.assertEqual(saved['best_epoch'], 4)
        self.assertIsNone(saved['metrics'])

    def test_current_snapshot_sha_survives_later_training(self):
        run = self.publish_full()
        job = self.prepare(run)
        digest = m.sha256(job / 'best.pt')
        self.publish_full(completed=6)
        self.assertNotEqual(m.sha256(run / 'last.pt'), digest)
        self.assertEqual(m.sha256(job / 'best.pt'), digest)
        self.assertEqual(m.read_json(job / 'trigger.json')['checkpoint_sha256'], digest)
        self.assertEqual(m.checkpoint_metadata(job / 'best.pt')['epoch'], 5)

    def test_epoch_six_never_substitutes_for_missing_five(self):
        run = self.publish_full(completed=6, committed=5, retain_candidate=False)
        job = self.prepare(run)
        self.assertFalse((job / 'status.json').exists())
        self.assertFalse((job / 'best.pt').exists())
        self.publish_full(completed=6, retain_candidate=False)
        self.prepare(run)
        state = m.read_json(job / 'status.json')
        self.assertEqual(state['status'], 'missed')
        self.assertEqual(state['expected_checkpoint_epoch'], 5)
        self.assertIn(6, state['observed_checkpoint_epochs'])
        self.assertFalse((job / 'best.pt').exists())

    def test_exact_last_is_accepted_without_pending_candidate(self):
        run = self.publish_full(retain_candidate=False)
        digest = m.sha256(run / 'last.pt')
        job = self.prepare(run)
        self.assertEqual(m.sha256(job / 'best.pt'), digest)
        self.assertEqual(m.read_json(job / 'trigger.json')['snapshot_source'], str(run / 'last.pt'))

    def test_best_full_updates_only_on_strict_improvement(self):
        winning_epoch, winning_digest, winning_score = None, None, -1.0
        for epoch, accuracy in ((5, 7), (10, 8), (15, 6), (20, 8)):
            with self.subTest(epoch=epoch):
                run = self.publish_full(completed=epoch)
                job = self.prepare(run, epoch)
                digest = m.sha256(job / 'best.pt')
                legacy_digest = m.sha256(run / 'best.pt')
                last_digest = m.sha256(run / 'last.pt')
                _, metrics = self.finish(job, accuracy)
                if metrics['weighted_score'] > winning_score:
                    winning_epoch, winning_digest = epoch, digest
                    winning_score = metrics['weighted_score']
                marker = m.read_json(run / 'best_full_test.json')
                self.assertEqual(marker['checkpoint_epoch'], winning_epoch)
                self.assertEqual(marker['checkpoint_sha256'], winning_digest)
                self.assertEqual(marker['weighted_score'], winning_score)
                self.assertEqual(m.sha256(run / 'best_full_test.pt'), winning_digest)
                self.assertEqual(m.sha256(run / 'best.pt'), legacy_digest)
                self.assertEqual(m.sha256(run / 'last.pt'), last_digest)
                self.assertFalse((job / 'best.pt').exists())
                self.assertFalse((run / 'full_test_checkpoints' / f'epoch_{epoch:03d}.pt').exists())
        self.assertEqual(winning_epoch, 10)

    def test_only_completed_candidate_is_cleaned_other_pending_kept(self):
        run = self.publish_full()
        first = self.prepare(run)
        self.publish_full(completed=10)
        later = self.prepare(run, 10)
        later_digest = m.sha256(later / 'best.pt')
        self.finish(first)
        self.assertFalse((run / 'full_test_checkpoints/epoch_005.pt').exists())
        self.assertFalse((first / 'best.pt').exists())
        self.assertEqual(m.sha256(run / 'full_test_checkpoints/epoch_010.pt'), later_digest)
        self.assertEqual(m.sha256(later / 'best.pt'), later_digest)
        self.assertEqual(m.read_json(later / 'status.json')['status'], 'queued')

    def test_completion_receipt_binds_metrics_before_candidate_cleanup(self):
        run = self.publish_full()
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        metrics_path = job / 'test_metrics.json'
        receipt_path = run / 'full_test_checkpoints/epoch_005.json'
        candidate = receipt_path.with_suffix('.pt')
        metrics = self.metrics(trigger)
        m.write_json(metrics_path, metrics)
        digest = m.sha256(metrics_path)
        temporary = {candidate, job / 'best.pt'}
        cleanup_order = []
        original_unlink = Path.unlink

        def checked_unlink(path, *args, **kwargs):
            if path in temporary:
                proof = m.read_json(receipt_path)['evaluation']
                self.assertEqual(proof['status'], 'complete')
                self.assertEqual(proof['checkpoint_sha256'], trigger['checkpoint_sha256'])
                self.assertEqual(proof['metrics_path'], str(metrics_path))
                self.assertEqual(proof['metrics_sha256'], digest)
                self.assertEqual(m.sha256(Path(proof['metrics_path'])), proof['metrics_sha256'])
                self.assertEqual(proof['weighted_score'], metrics['weighted_score'])
                self.assertTrue(proof['completed_at'])
                cleanup_order.append(path)
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', new=checked_unlink):
            state = m.complete_job(job, trigger)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(set(cleanup_order), temporary)
        self.assertEqual(len(cleanup_order), 2)
        self.assertTrue(receipt_path.exists())
        self.assertTrue(metrics_path.exists())
        self.assertFalse(candidate.exists())
        self.assertFalse((job / 'best.pt').exists())

    def test_wrong_candidate_receipt_sha_prevents_checkpoint_cleanup(self):
        run = self.publish_full()
        job = self.prepare(run)
        trigger = m.read_json(job / 'trigger.json')
        metrics_path = job / 'test_metrics.json'
        m.write_json(metrics_path, self.metrics(trigger))
        receipt_path = run / 'full_test_checkpoints/epoch_005.json'
        candidate = receipt_path.with_suffix('.pt')
        receipt = m.read_json(receipt_path)
        receipt['checkpoint_sha256'] = '0' * 64
        m.write_json(receipt_path, receipt)
        candidate_digest = m.sha256(candidate)
        job_digest = m.sha256(job / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'candidate receipt differs'):
            m.complete_job(job, trigger)
        self.assertEqual(m.sha256(candidate), candidate_digest)
        self.assertEqual(m.sha256(job / 'best.pt'), job_digest)
        self.assertEqual(m.read_json(receipt_path), receipt)
        self.assertNotEqual(m.read_json(job / 'status.json')['status'], 'complete')

    def test_final_and_legacy_best_survive_lower_final_score(self):
        run = self.publish_full()
        self.finish(self.prepare(run), 9)
        winning_digest = m.sha256(run / 'best_full_test.pt')
        self.publish_full(completed=30, status='complete')
        m.snapshot(run / 'last.pt', run / 'final.pt')
        final_digest, legacy_digest = m.sha256(run / 'final.pt'), m.sha256(run / 'best.pt')
        summary, configuration, rows = m.committed_run(run)
        job = m.prepare_job(run, 30, summary, configuration, rows, self.config, final=True)
        self.assertEqual(m.read_json(job / 'trigger.json')['checkpoint_epoch'], 30)
        self.finish(job, 7)
        self.assertEqual(m.sha256(run / 'final.pt'), final_digest)
        self.assertEqual(m.checkpoint_metadata(run / 'final.pt')['epoch'], 30)
        self.assertEqual(m.sha256(run / 'best.pt'), legacy_digest)
        self.assertEqual(m.sha256(run / 'best_full_test.pt'), winning_digest)
        self.assertFalse((run / 'full_test_checkpoints/epoch_030.pt').exists())

    def test_invalid_metrics_never_replace_best_or_clean_candidates(self):
        run = self.publish_full()
        first = self.prepare(run)
        self.finish(first)
        previous_digest = m.sha256(run / 'best_full_test.pt')
        self.publish_full(completed=10)
        later = self.prepare(run, 10)
        trigger = m.read_json(later / 'trigger.json')
        invalid = self.metrics(trigger, 9)
        invalid['weighted_score'] = 0.999
        m.write_json(later / 'test_metrics.json', invalid)
        with self.assertRaises(ValueError):
            m.complete_job(later, trigger)
        self.assertEqual(m.sha256(run / 'best_full_test.pt'), previous_digest)
        self.assertTrue((later / 'best.pt').exists())
        self.assertTrue((run / 'full_test_checkpoints/epoch_010.pt').exists())

    def test_noearlystop_config_rejects_positive_patience(self):
        run = self.publish_full()
        configuration = m.read_json(run / 'run_config.json')
        configuration['early_stopping_patience'] = 5
        m.write_json(run / 'run_config.json', configuration)
        with self.assertRaises(ValueError):
            m.committed_run(run)

    def test_disabled_early_stopping_rejects_early_stopped_summary(self):
        run = self.publish_full(status='early_stopped')
        with self.assertRaises(ValueError):
            m.committed_run(run)


class PinnedConfigurationFixtures(unittest.TestCase):
    """Queued snapshots remain bound when a trainer changes policy/source root."""

    setUp = PeriodicFixtures.setUp
    tearDown = PeriodicFixtures.tearDown
    publish = PeriodicFixtures.publish
    prepare = PeriodicFixtures.prepare
    metrics = PeriodicFixtures.metrics

    def archive_trainer(self, run, job, name='runtime'):
        archived = run / name
        for relative in self.configuration['source_sha256']:
            target = archived / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.root / relative).read_bytes())
        (archived / 'code/evaluate.py').write_bytes((self.root / 'code/evaluate.py').read_bytes())
        trigger = m.read_json(job / 'trigger.json')
        trigger['training_source_root'] = str(archived)
        m.write_json(job / 'trigger.json', trigger)
        return archived, trigger

    def run_worker(self, job):
        trigger = m.read_json(job / 'trigger.json')
        m.write_json(job / 'fixture_metrics.json', self.metrics(trigger))
        m.write_json(job / 'launch.json', {'configuration': self.config})
        return m.worker(job, probe=lambda _: (True, 'fixture only', 'GPU-fixture'))

    def test_prepare_pins_exact_configuration(self):
        run = self.publish(); job = self.prepare(run)
        pinned, trigger = m.read_json(job / 'training_configuration.json'), m.read_json(job / 'trigger.json')
        self.assertEqual(pinned, self.configuration)
        self.assertEqual(m.fingerprint(pinned), trigger['configuration_sha256'])

    def test_pinned_worker_survives_live_configuration_policy_change(self):
        run = self.publish(); job = self.prepare(run)
        m.write_json(run / 'run_config.json', {**self.configuration, 'selection_policy': 'full_test_every5'})
        self.assertEqual(self.run_worker(job), 0)
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')

    def test_legacy_job_without_pinned_configuration_keeps_live_guard(self):
        run = self.publish(); job = self.prepare(run)
        (job / 'training_configuration.json').unlink()
        m.write_json(run / 'run_config.json', {**self.configuration, 'epochs': 31})
        with self.assertRaisesRegex(ValueError, 'Training configuration changed'):
            m.job_training_configuration(job, m.read_json(job / 'trigger.json'))

    def test_pinned_configuration_tampering_is_rejected(self):
        run = self.publish(); job = self.prepare(run)
        m.write_json(job / 'training_configuration.json', {**self.configuration, 'variant': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'Training configuration changed'):
            m.job_training_configuration(job, m.read_json(job / 'trigger.json'))

    def test_snapshot_train_configuration_must_match_binding(self):
        run = self.publish(); job = self.prepare(run)
        saved = torch.load(job / 'best.pt', map_location='cpu', weights_only=True)
        saved['train_config'] = {**self.configuration, 'variant': 'changed'}
        temporary = job / 'changed.tmp'; torch.save(saved, temporary); temporary.replace(job / 'best.pt')
        trigger = m.read_json(job / 'trigger.json')
        trigger['checkpoint_sha256'] = m.sha256(job / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'Snapshot training configuration differs'):
            m.job_training_configuration(job, trigger)

    def test_archived_trainer_allows_main_trainer_change_not_model_change(self):
        for runtime in ('runtime', 'runtime_fulltest'):
            with self.subTest(runtime=runtime):
                run = self.publish(name=runtime); job = self.prepare(run)
                self.archive_trainer(run, job, runtime)
                (self.root / 'code/train.py').write_text('# next training policy\n')
                m.write_json(run / 'run_config.json', {**self.configuration, 'selection_policy': 'full_test_every5'})
                self.assertEqual(self.run_worker(job), 0)
                (self.root / 'code/train.py').write_text('# fixture source only\n')

    def test_archived_model_is_the_actual_inference_runtime(self):
        run = self.publish(); job = self.prepare(run)
        archived, trigger = self.archive_trainer(run, job)
        (self.root / 'code/vitseg/data.py').write_text('# main project belongs to a newer model\n')
        m.job_training_configuration(job, trigger)
        command = m.evaluator_command(self.config, job)
        self.assertIn(str(archived / 'code/evaluate.py'), command)
        self.assertEqual(self.run_worker(job), 0)

    def test_archived_model_inventory_cannot_gain_untracked_sources(self):
        run = self.publish(); job = self.prepare(run)
        archived, trigger = self.archive_trainer(run, job)
        (archived / 'code/vitseg/untracked.py').write_text('# not part of training\n')
        with self.assertRaisesRegex(ValueError, 'Archived evaluator vitseg sources differ'):
            m.job_training_configuration(job, trigger)

    def test_archived_training_source_hash_still_checked(self):
        run = self.publish(); job = self.prepare(run)
        archived, trigger = self.archive_trainer(run, job)
        (archived / 'code/train.py').write_text('# changed archived trainer\n')
        with self.assertRaisesRegex(ValueError, 'Training/evaluation source changed'):
            m.job_training_configuration(job, trigger)

    def test_source_root_is_limited_to_same_run_runtime_directories(self):
        run = self.publish(); job = self.prepare(run)
        for invalid in (self.root, self.root / 'result/other/runtime', run / 'runtime/code'):
            with self.subTest(invalid=invalid):
                invalid.mkdir(parents=True, exist_ok=True)
                trigger = m.read_json(job / 'trigger.json'); trigger['training_source_root'] = str(invalid)
                with self.assertRaisesRegex(ValueError, 'training_source_root must'):
                    m.job_training_configuration(job, trigger)

    def test_actual_evaluator_sha_still_checked_with_archived_trainer(self):
        run = self.publish(); job = self.prepare(run)
        self.archive_trainer(run, job)
        with (self.root / 'code/evaluate.py').open('a') as stream:
            stream.write('# changed actual evaluator\n')
        self.assertEqual(self.run_worker(job), 1)
        self.assertIn('Evaluator source or test manifest changed', m.read_json(job / 'status.json')['reason'])

    def test_explicit_pretransition_current_epoch_and_metadata_correction(self):
        run = HardValidationFixtures.publish_hard(self)
        job = HardValidationFixtures.prepare(self, run)
        saved = torch.load(job / 'best.pt', map_location='cpu', weights_only=True)
        saved.update(epoch=5, model={'fixture': torch.tensor([5])})
        replacement = job / 'epoch5.tmp'; torch.save(saved, replacement); replacement.replace(job / 'best.pt')
        _, trigger = self.archive_trainer(run, job)
        trigger.update(checkpoint_epoch=5, checkpoint_sha256=m.sha256(job / 'best.pt'),
                       selection_policy='full_test_every5', checkpoint_choice='current_epoch',
                       recovered_pretransition_current_epoch=True, best_metric='weighted_score')
        summary = m.read_json(run / 'summary.json')
        m.validate_snapshot(saved, trigger, self.configuration, summary)
        for overrides in ({'recovered_pretransition_current_epoch': False},
                          {'training_source_root': str(run / 'runtime_fulltest')},
                          {'checkpoint_choice': 'validation_best'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                m.validate_snapshot(saved, {**trigger, **overrides}, self.configuration, summary)
        m.write_json(job / 'trigger.json', trigger)
        metrics = HardValidationFixtures.metrics(self, trigger)
        metrics['checkpoint_choice'] = 'validation_best'  # Old evaluator's protocol inference.
        m.write_json(job / 'test_metrics.json', metrics)
        original_sha = m.sha256(job / 'test_metrics.json')
        self.assertEqual(m.complete_job(job, trigger)['status'], 'complete')
        corrected = m.read_json(job / 'test_metrics.json')
        for key, value in metrics.items():
            if key != 'checkpoint_choice': self.assertEqual(corrected[key], value)
        self.assertEqual(corrected['checkpoint_choice'], 'current_epoch')
        self.assertEqual(corrected['reported_checkpoint_choice'], 'validation_best')
        self.assertEqual(corrected['metadata_correction']['original_metrics_sha256'], original_sha)
        self.assertEqual(m.sha256(job / 'test_metrics_before_metadata_correction.json'), original_sha)
        self.assertEqual(m.checkpoint_metadata(run / 'best_full_test.pt')['epoch'], 5)
        self.assertEqual(m.checkpoint_metadata(run / 'best_full_test.pt')['best_epoch'], 4)
        self.assertEqual(m.checkpoint_metadata(run / 'best_full_test.pt')['train_config'], self.configuration)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (PeriodicFixtures, HardValidationFixtures, FullTestPolicyFixtures,
                                           PinnedConfigurationFixtures))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    destination = Path(__file__).resolve().parents[1] / 'result/verification_periodic_evaluate.json'
    m.write_json(destination, {'schema': 'vit_periodic_cpu_fixture_checks_v1', 'created_at': m.now(),
        'success': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'scope': 'Synthetic CPU control-flow fixtures only; no real model or test evaluation',
        'source_sha256': m.sha256(Path(__file__).with_name('periodic_evaluate.py'))})
    raise SystemExit(0 if result.wasSuccessful() else 1)
