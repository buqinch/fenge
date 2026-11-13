"""CPU checks for fixed-subset provenance, cadence and snapshot isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

import hard_subset_monitor as h
import verify_periodic_evaluate as existing


class HardFixtures(unittest.TestCase):
    def setUp(self):
        existing.m = h.base
        self.fixture = existing.PeriodicFixtures()
        self.fixture.setUp()
        self.old_root, h.ROOT = h.ROOT, self.fixture.root

    def tearDown(self):
        h.ROOT = self.old_root
        self.fixture.tearDown()

    def config(self):
        return {'schema': 'vit_hard_subset_monitor_v1', 'runs': [str(h.ROOT / 'result/b')],
                'state_directory': str(h.ROOT / 'result/hard_monitor'), 'hard_manifest': str(h.ROOT / 'hard.json'),
                'every_epochs': 2, 'poll_seconds': 15,
                'remote': {'project': str(h.ROOT), 'python': 'fixture', 'host': None, 'gpu_lock': str(h.ROOT / 'gpu.lock')}}

    def test_pending_manifest_snapshots_new_boundaries_without_backfill(self):
        run = self.fixture.publish(scores=(.5, .6, .7))
        cfg = self.config()
        h.run_monitor(cfg, once=True)
        self.assertFalse((run / 'hard_periodic_test').exists())
        self.fixture.publish(scores=(.5, .6, .7, .8))
        h.run_monitor(cfg, once=True)
        job = run / 'hard_periodic_test/epoch_004'
        self.assertEqual(h.base.read_json(job / 'status.json')['status'], 'waiting_manifest')
        trigger = h.base.read_json(job / 'trigger.json')
        self.assertEqual(trigger['checkpoint_epoch'], 4)
        self.assertFalse(trigger['checkpoint_selection_uses_test'])
        saved_sha = h.base.sha256(job / 'best.pt')
        self.fixture.publish(scores=(.5, .6, .7, .8, .9))
        h.run_monitor(cfg, once=True)
        self.assertEqual(h.base.sha256(job / 'best.pt'), saved_sha)
        self.assertEqual([p.name for p in (run / 'hard_periodic_test').iterdir()], ['epoch_004'])

    def test_new_validation_count_is_identity_driven(self):
        identity = {'image_count': 1500, 'unique_source_count': 1500}
        d = {'epoch': 1, 'partial': False, 'image_count': 1500, 'validation_unique_source_count': 1500,
             'checkpoint_selection_uses_test': False, 'data_identity': identity, 'miou': .8}
        h.base.validate_validation(d, 1, .8, identity)
        with self.assertRaises(ValueError):
            h.base.validate_validation({**d, 'image_count': 1601}, 1, .8, identity)

    def test_future_run_waits_without_fake_summary_then_starts_at_two(self):
        cfg = self.config(); cfg['start_after_completed_epoch'] = 0
        run = h.ROOT / 'result/b'
        h.run_monitor(cfg, once=True)
        self.assertFalse((run / 'summary.json').exists())
        self.fixture.publish(scores=(.5, .6))
        h.run_monitor(cfg, once=True)
        self.assertTrue((run / 'hard_periodic_test/epoch_002/best.pt').exists())

    def test_subset_identity_is_frozen_across_future_epochs(self):
        self.fixture.publish()
        cfg = self.config()
        (h.ROOT / 'code/evaluate_hard_subset.py').write_text('# fixture only\n')
        records = [{'scene': s, 'sample_id': str(i)} for s, n in h.QUOTAS.items() for i in range(n)]
        h.base.write_json(h.ROOT / 'hard.json', {'schema': 'lusnar_fixed_hard_test_v1', 'records': records})
        h.run_monitor(cfg, once=True)
        frozen = h.base.read_json(Path(cfg['state_directory']) / 'subset_identity.json')
        records[0]['sample_id'] = 'replaced_sample'
        h.base.write_json(h.ROOT / 'hard.json', {'schema': 'lusnar_fixed_hard_test_v1', 'records': records})
        h.run_monitor(cfg, once=True)
        state = h.base.read_json(Path(cfg['state_directory']) / 'state.json')
        self.assertIn('hard_manifest', state['errors'])
        self.assertFalse(state['manifest_ready'])
        self.assertEqual(h.base.read_json(Path(cfg['state_directory']) / 'subset_identity.json'), frozen)

    def test_committed_run_accepts_declared_1500_validation(self):
        run = self.fixture.publish(scores=(.5, .6))
        summary = h.base.read_json(run / 'summary.json')
        summary['validation_images'] = 1500
        summary['data_identity']['val']['image_count'] = 1500
        h.base.write_json(run / 'summary.json', summary)
        self.assertEqual(h.base.committed_run(run)[0]['validation_images'], 1500)
        summary['validation_images'] = 1601
        h.base.write_json(run / 'summary.json', summary)
        with self.assertRaises(ValueError): h.base.committed_run(run)

    def test_hard_score_is_not_a_full_score(self):
        trigger = {'checkpoint_epoch': 4, 'checkpoint_sha256': 'frozen'}
        binding = {'manifest': {'sha256': 'subset'}}
        d = {'partial': False, 'image_count': 1500, 'official_full_test': False,
             'test_scope': 'fixed_hard_subset', **trigger, 'subset_manifest_sha256': 'subset',
             'hard_subset_score_weights': h.WEIGHTS, 'hard_subset_score': 1., 'scenes': {}}
        for s, n in h.QUOTAS.items():
            counts = [n * 1024 * 1024 // 5] * 5
            counts[-1] += n * 1024 * 1024 - sum(counts)
            d['scenes'][s] = {'image_count': n, 'miou': 1., 'class_iou': dict.fromkeys(h.base.CLASSES, 1.),
                'confusion_matrix_gt_rows_pred_cols': [[counts[i] if i == j else 0 for j in range(5)] for i in range(5)]}
        self.assertEqual(h.validate_metrics(d, trigger, binding)['hard_subset_score'], 1.)
        for bad in ({**d, 'weighted_score': 1.}, {**d, 'official_full_test': True},
                    {**d, 'subset_manifest_sha256': 'changed'}, {**d, 'hard_subset_score': .99}):
            with self.assertRaises(ValueError): h.validate_metrics(bad, trigger, binding)

    def test_user_stopped_is_not_restarted_or_scheduled(self):
        run = self.fixture.publish()
        cfg = self.config(); h.run_monitor(cfg, once=True)
        self.fixture.publish(scores=(.5, .6, .7, .8))
        d = h.base.read_json(run / 'summary.json'); d['status'] = 'user_stopped'
        h.base.write_json(run / 'summary.json', d)
        h.run_monitor(cfg, once=True)
        self.assertFalse((run / 'hard_periodic_test').exists())

    def test_durable_remote_worker_and_completed_result_recovery(self):
        # Exercise the real subprocess and GPU-file-lock handoff with a CPU
        # fixture evaluator, never a real checkpoint or GPU computation.
        manifest = h.ROOT / 'hard.json'
        records = [{'scene': scene, 'sample_id': str(i)} for scene, n in h.QUOTAS.items() for i in range(n)]
        h.base.write_json(manifest, {'schema': 'lusnar_fixed_hard_test_v1', 'records': records})
        evaluator = h.ROOT / 'code/evaluate_hard_subset.py'
        evaluator.write_text('import json,sys,pathlib\np=pathlib.Path(sys.argv[sys.argv.index("--output")+1])\np.write_text((p.parent/"fixture.json").read_text())\n')
        job = h.ROOT / 'result/hard_remote/b/epoch_004'; job.mkdir(parents=True)
        (job / 'best.pt').write_bytes(b'fixture-not-a-model')
        trigger = {'checkpoint_epoch': 4, 'checkpoint_sha256': h.base.sha256(job / 'best.pt'),
                   'training_configuration': self.fixture.configuration}
        binding = {'manifest': h.manifest_identity(manifest), 'monitor_sha256': h.base.sha256(h.__file__),
                   'evaluator_sha256': h.base.sha256(evaluator)}
        spec = {'project': str(h.ROOT), 'python': sys.executable, 'gpu_lock': str(h.ROOT / 'gpu.lock'),
                'gpu_index': 3, 'minimum_free_mib': 10000, 'trigger': trigger, 'binding': binding}
        d = {'partial': False, 'image_count': 1500, 'official_full_test': False, 'test_scope': 'fixed_hard_subset',
             'checkpoint_epoch': 4, 'checkpoint_sha256': trigger['checkpoint_sha256'],
             'subset_manifest_sha256': binding['manifest']['sha256'], 'hard_subset_score_weights': h.WEIGHTS,
             'hard_subset_score': 1., 'scenes': {}}
        for scene, count in h.QUOTAS.items():
            cm = [[0] * 5 for _ in range(5)]; cm[0][0] = count * 1024 * 1024
            d['scenes'][scene] = {'image_count': count, 'miou': 1., 'class_iou': {'regolith': 1.},
                                 'confusion_matrix_gt_rows_pred_cols': cm}
        h.base.write_json(job / 'fixture.json', d)
        h.prepare_remote(job, spec)
        self.assertEqual(h.remote_worker(job), 0)
        self.assertEqual(h.remote_status(job)['status'], 'complete')
        self.assertIsNone(h.remote_status(job).get('score_milestone'))
        h.set_status(job, 'running', worker_identity=None, evaluator_identity=None)
        recovered = h.ensure_remote(job)
        self.assertEqual(recovered['status'], 'complete')
        self.assertEqual(recovered['verified']['hard_subset_score'], 1.)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HardFixtures))
    h.base.write_json(Path(__file__).resolve().parents[1] / 'result/verification_hard_subset_monitor.json',
        {'success': result.wasSuccessful(), 'tests_run': result.testsRun, 'created_at': h.base.now(),
         'monitor_sha256': h.base.sha256(Path(__file__).with_name('hard_subset_monitor.py')),
         'scope': 'CPU synthetic fixtures; no inference or training'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
