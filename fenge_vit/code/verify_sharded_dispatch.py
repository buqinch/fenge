"""CPU fixtures for two-GPU reservation and periodic/remote worker dispatch."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_slots
import verify_remote_evaluate as fixtures

m, r = fixtures.m, fixtures.r


class ShardedDispatch(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RemoteEvaluationFixtures(methodName='runTest')
        self.fixture.setUp()
        self.config = self.fixture.config
        self.config.update(gpu_index=6, sharded_gpus=[6, 7], evaluation_tta='hflip',
                           evaluation_batch_size=1, max_concurrent_evaluations=1)
        self.config['gpu_lock'] = str(self.fixture.root / 'result/.periodic_gpu_6.lock')
        self.config['remote']['gpu_lock'] = self.config['gpu_lock']
        self.config['evaluation'] = m.settings(self.config)
        self.trigger = self.fixture.trigger
        self.trigger['evaluation'] = self.config['evaluation']
        m.write_json(self.fixture.job / 'trigger.json', self.trigger)
        script = self.fixture.root / 'code/evaluate_sharded.py'
        script.write_text('import os,json,sys,pathlib\n'
            'job=pathlib.Path(sys.argv[sys.argv.index("--job")+1])\n'
            'fds=list(map(int,sys.argv[sys.argv.index("--gpu-lock-fds")+1:]))\n'
            'assert len(fds)==2\n'
            'for fd in fds: os.fstat(fd)\n'
            '(job/"inherited_fds.json").write_text(json.dumps(fds))\n'
            '(job/"test_metrics.json").write_text((job/"fixture_metrics.json").read_text())\n')

    def tearDown(self):
        self.fixture.tearDown()

    def metrics(self):
        doc = self.fixture.fixture.metrics(self.trigger)
        doc.update(tta='hflip', tta_mode='hflip', tta_sizes=None, tta_weights=None,
                   tta_views=['original', 'horizontal_flip'], tta_forward_calls_per_batch=2,
                   model_config={'input_size': 1024})
        return doc

    def test_all_gpu_locks_or_none(self):
        configs = gpu_slots.group_configs(self.config)
        blocker = gpu_slots.acquire_gpu_slot(configs[1])
        self.assertIsNotNone(blocker)
        self.assertIsNone(gpu_slots.acquire_gpu_group(self.config))
        first = gpu_slots.acquire_gpu_slot(configs[0])
        self.assertIsNotNone(first)  # Failed pair reservation did not strand GPU 6.
        first.close(); blocker.close()
        pair = gpu_slots.acquire_gpu_group(self.config)
        self.assertEqual(len(pair), 2)
        for h in pair: h.close()

    def test_remote_probe_reserves_the_requested_pair(self):
        with patch.object(r, '_ssh', return_value={'ready': True, 'reason': 'free', 'gpu_uuid': 'u'}) as ssh:
            r.probe_remote(self.config)
        self.assertEqual(ssh.call_args.args[2]['sharded_gpus'], [6, 7])

    def test_local_worker_passes_both_locks_to_sharded_child(self):
        config = {k: v for k, v in self.config.items() if k != 'remote'}
        job = self.fixture.job
        m.write_json(job / 'launch.json', {'configuration': config})
        m.write_json(job / 'fixture_metrics.json', self.metrics())
        self.assertEqual(m.worker(job, probe=lambda _: (True, 'free', 'uuid')), 0)
        self.assertTrue((job / 'inherited_fds.json').exists())
        self.assertEqual(m.read_json(job / 'status.json')['status'], 'complete')

    def test_remote_worker_passes_both_locks_and_hflip(self):
        job = self.fixture.remote_job()
        m.write_json(job / 'fixture_metrics.json', self.metrics())
        self.assertEqual(r.remote_worker(job, probe=lambda _: (True, 'free', 'uuid')), 0)
        self.assertTrue((job / 'inherited_fds.json').exists())
        self.assertEqual(m.read_json(job / 'remote_execution.json')['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
