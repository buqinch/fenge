#!/usr/bin/env python3
"""CPU-only remote evaluation orchestration checks; no real SSH or GPU inference."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import verify_periodic_evaluate as fixtures

# Share the fixture's isolated periodic module with the remote executor.
m = fixtures.m
sys.modules['periodic_evaluate'] = m
import remote_evaluate as r


class RemoteEvaluationFixtures(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PeriodicFixtures(methodName='runTest')
        self.fixture.setUp()
        self.root = self.fixture.root
        self.old_root = r.ROOT
        r.ROOT = self.root
        self.config = copy.deepcopy(self.fixture.config)
        self.config['retain_above_score'] = .92
        self.config['remote'] = {
            'host': 'fixture103', 'project': str(self.root), 'python': sys.executable,
            'wrapper': 'code/run_on_103.sh',
            'gpu_lock': str(self.root / 'result/shared_remote_gpu.lock'),
        }
        self.run = self.fixture.publish()
        self.job = self.fixture.prepare(self.run)
        self.trigger = m.read_json(self.job / 'trigger.json')

    def tearDown(self):
        r.ROOT = self.old_root
        self.fixture.tearDown()

    def remote_job(self):
        job = Path(r.remote_job_path(self.config, self.job))
        job.mkdir(parents=True, exist_ok=True)
        r.prepare_remote(job, r.prepare_spec(self.config, self.job, self.trigger))
        shutil.copyfile(self.job / 'best.pt', job / 'best.pt')
        return job


    def test_remote_reconnect_is_scheduled_even_when_gpu_is_busy(self):
        m.write_json(self.job / 'remote_spec.json', {'fixture': 'already submitted'})
        child = Mock(pid=123456)
        child.poll.return_value = None
        with patch.object(m, 'gpu_available', side_effect=AssertionError('Reconnection must not wait for its own GPU')), \
             patch.object(m.subprocess, 'Popen', return_value=child) as spawn:
            report = m.Monitor(self.config).tick()
        self.assertEqual(report['active_jobs'], [str(self.job)])
        self.assertEqual(m.read_json(self.job / 'status.json')['status'], 'launching')
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(m.read_json(self.job / 'launch.json')['execution_host'], 'fixture103')

    def test_disappeared_local_worker_reconnects_persistent_remote_job(self):
        m.write_json(self.job / 'remote_spec.json', {'fixture': 'persistent remote task'})
        m.set_status(self.job, 'running', worker_identity={'pid': 123456, 'start_ticks': '0', 'boot_id': 'fixture'})
        child = Mock(pid=234567)
        child.poll.return_value = None
        with patch.object(m, 'alive', return_value=False), patch.object(m, 'job_processes', return_value=[]), \
             patch.object(m, 'gpu_available', side_effect=AssertionError('Remote task may still own GPU')), \
             patch.object(m.subprocess, 'Popen', return_value=child) as spawn:
            report = m.Monitor(self.config).tick()
        self.assertEqual(report['active_jobs'], [str(self.job)])
        self.assertEqual(spawn.call_count, 1)
        self.assertNotEqual(m.read_json(self.job / 'status.json')['status'], 'failed')
        self.assertTrue((self.job / 'best.pt').exists())

    def test_remote_busy_worker_keeps_snapshot_and_requeues(self):
        m.write_json(self.job / 'launch.json', {'configuration': self.config})
        before = m.sha256(self.job / 'best.pt')
        with patch.object(r, 'run_remote', return_value=75) as execute:
            code = m.worker(self.job, probe=lambda config: self.fail('Must not probe the local GPU'))
        self.assertEqual(code, 0)
        self.assertEqual(m.read_json(self.job / 'status.json')['status'], 'queued')
        self.assertEqual(m.sha256(self.job / 'best.pt'), before)
        execute.assert_called_once()

    def test_remote_completed_metrics_use_full_validation_and_threshold_retention(self):
        m.write_json(self.job / 'launch.json', {'configuration': self.config})
        before = m.sha256(self.job / 'best.pt')
        def complete(config, job, trigger):
            m.write_json(job / 'test_metrics.json', self.fixture.metrics(trigger))
            return 0
        with patch.object(r, 'run_remote', side_effect=complete):
            self.assertEqual(m.worker(self.job), 0)
        state = m.read_json(self.job / 'status.json')
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(m.sha256(self.job / 'best.pt'), before)
        self.assertEqual(m.read_json(self.job / 'score_milestone.json')['checkpoint_sha256'], before)

    def test_remote_invalid_metrics_are_not_accepted_or_checkpoint_deleted(self):
        m.write_json(self.job / 'launch.json', {'configuration': self.config})
        before = m.sha256(self.job / 'best.pt')
        def incomplete(config, job, trigger):
            document = self.fixture.metrics(trigger)
            document['image_count'] = 5003
            m.write_json(job / 'test_metrics.json', document)
            return 0
        with patch.object(r, 'run_remote', side_effect=incomplete):
            self.assertEqual(m.worker(self.job), 1)
        self.assertEqual(m.read_json(self.job / 'status.json')['status'], 'failed')
        self.assertFalse((self.job / 'score_milestone.json').exists())
        self.assertEqual(m.sha256(self.job / 'best.pt'), before)


    def test_probe_dispatches_to_remote_without_local_nvidia_query(self):
        reply = {'ready': False, 'reason': 'remote busy', 'gpu_uuid': 'remote-uuid'}
        with patch.object(r, '_ssh', return_value=reply) as ssh, \
             patch.object(r, 'local_gpu_probe', side_effect=AssertionError('Local GPU must not be queried')):
            self.assertEqual(m.gpu_available(self.config), (False, 'remote busy', 'remote-uuid'))
        self.assertEqual(ssh.call_args.args[1], ['--probe'])
        self.assertEqual(ssh.call_args.args[2], {'gpu_index': 3, 'minimum_free_mib': 10000})

    def test_ssh_command_quotes_remote_paths_as_single_arguments(self):
        config = copy.deepcopy(self.config)
        config['remote']['project'] = "/tmp/space ' quote $(never_execute)"
        config['remote']['python'] = '/tmp/python with spaces'
        reply = subprocess.CompletedProcess([], 0, stdout='{"status":"queued"}', stderr='')
        with patch.object(r.subprocess, 'run', return_value=reply) as run:
            r._ssh(config, ['--ensure', '/tmp/a quoted job'])
        argv = run.call_args.args[0]
        self.assertEqual(argv[-2], 'fixture103')
        self.assertEqual(shlex.split(argv[-1]), ['bash', config['remote']['project'] + '/code/run_on_103.sh',
            '/tmp/python with spaces', config['remote']['project'] + '/code/remote_evaluate.py',
            '--ensure', '/tmp/a quoted job'])
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_unsafe_host_and_path_escapes_are_rejected(self):
        for key, value in [('host', '-oProxyCommand=bad'), ('host', 'host;bad'),
                           ('project', '/tmp/../escape'), ('wrapper', '../outside.sh')]:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config['remote'][key] = value
                with self.assertRaises(ValueError):
                    r.validate_remote(config)
        with self.assertRaises(ValueError):
            r.remote_job_path(self.config, self.root / 'unowned/periodic_test/epoch_003')
        with self.assertRaises(ValueError):
            r._job_directory(self.root / 'result/remote_evaluations/../../outside')

    def test_spec_is_stable_and_checkpoint_hash_is_checked_before_transfer(self):
        first = r.prepare_spec(self.config, self.job, self.trigger)
        self.assertEqual(first, r.prepare_spec(self.config, self.job, self.trigger))
        with (self.job / 'best.pt').open('ab') as stream:
            stream.write(b'corrupt')
        with patch.object(r, '_ssh', side_effect=AssertionError('Must reject before network')):
            with self.assertRaisesRegex(ValueError, 'snapshot SHA'):
                r.run_remote(self.config, self.job, self.trigger)

    def test_existing_remote_spec_cannot_be_replaced(self):
        job = self.remote_job()
        spec = r.prepare_spec(self.config, self.job, self.trigger)
        changed = copy.deepcopy(spec)
        changed['gpu_index'] = 2
        with self.assertRaisesRegex(ValueError, 'replace'):
            r.prepare_remote(job, changed)
        self.assertEqual(m.read_json(job / 'spec.json'), spec)

    def test_source_change_is_rejected_on_remote_without_gpu_or_evaluator(self):
        job = self.remote_job()
        (self.root / 'code/train.py').write_text('# changed source\n')
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('No evaluator for changed sources')):
            self.assertEqual(r.remote_worker(job, probe=lambda _: self.fail('No GPU probe for bad source')), 1)
        self.assertEqual(r._state(job)['status'], 'failed')
        self.assertIn('source changed', r._state(job)['reason'])
        self.assertTrue((job / 'best.pt').exists())

    def test_evaluator_change_is_rejected_on_remote(self):
        job = self.remote_job()
        with (self.root / 'code/evaluate.py').open('a') as stream:
            stream.write('# modified evaluation\n')
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('Changed evaluator must not execute')):
            self.assertEqual(r.remote_worker(job, probe=lambda _: self.fail('No GPU probe for bad evaluator')), 1)
        self.assertEqual(r._state(job)['status'], 'failed')
        self.assertIn('evaluation source or test manifest', r._state(job)['reason'])

    def test_shared_remote_gpu_lock_prevents_evaluator_launch(self):
        job = self.remote_job()
        held = m.acquire_lock(self.config['remote']['gpu_lock'])
        self.assertIsNotNone(held)
        try:
            with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('Remote GPU lock already held')):
                self.assertEqual(r.remote_worker(job, probe=lambda _: self.fail('Lock checked before GPU probe')), 75)
        finally:
            held.close()
        self.assertEqual(r._state(job)['status'], 'queued')
        self.assertTrue((job / 'best.pt').exists())

    def test_ensure_reconnects_live_worker_without_duplicate_spawn(self):
        job = self.remote_job()
        child = Mock(pid=os.getpid())
        spawn = Mock(return_value=child)
        first = r.ensure(job, spawn=spawn, probe=lambda _: (True, 'fixture free', 'fixture-uuid'))
        self.assertEqual(first['status'], 'launching')
        second = r.ensure(job, spawn=spawn, probe=lambda _: self.fail('Live worker already owns task'))
        self.assertEqual(second['status'], 'launching')
        self.assertEqual(spawn.call_count, 1)

    def test_remote_worker_runs_cpu_fixture_evaluator_and_publishes_verified_completion(self):
        job = self.remote_job()
        m.write_json(job / 'fixture_metrics.json', self.fixture.metrics(self.trigger))
        # The fixture evaluator merely copies synthetic JSON; it imports no model
        # or CUDA code, even though the real evaluator's CLI arguments are used.
        code = r.remote_worker(job, probe=lambda _: (True, 'fixture free', 'fixture-uuid'))
        self.assertEqual(code, 0)
        state = r._state(job)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['gpu_uuid'], 'fixture-uuid')
        self.assertEqual(state['metrics_sha256'], m.sha256(job / 'test_metrics.json'))
        self.assertEqual(state['verified']['weighted_score'], 1.0)
        self.assertTrue((job / 'best.pt').exists())

    def test_ack_cleans_only_remote_checkpoint_after_verified_nonqualifying_result(self):
        # Equality is deliberately nonqualifying: retention is strictly greater.
        self.config['retain_above_score'] = 1.0
        job = self.remote_job()
        metrics = self.fixture.metrics(self.trigger)
        m.write_json(job / 'test_metrics.json', metrics)
        verified = m.validate_metrics(metrics, self.trigger)
        checksum = m.sha256(job / 'test_metrics.json')
        r._set_state(job, 'complete', verified=verified, metrics_sha256=checksum)
        result = r.acknowledge(job, {'metrics_sha256': checksum})
        self.assertFalse(result['checkpoint_retained'])
        self.assertFalse((job / 'best.pt').exists())
        self.assertTrue((self.job / 'best.pt').exists())
        self.assertTrue((job / 'test_metrics.json').exists())

    def test_completed_evaluator_recovers_after_worker_exit_and_ack_retains_threshold(self):
        job = self.remote_job()
        m.write_json(job / 'test_metrics.json', self.fixture.metrics(self.trigger))
        r._set_state(job, 'running', worker_identity={'pid': 123456, 'start_ticks': '0', 'boot_id': 'fixture'})
        with patch.object(m, 'alive', return_value=False), patch.object(r, '_active_processes', return_value=[]):
            state = r.ensure(job, spawn=lambda *a, **k: self.fail('Completed evaluator must not repeat'),
                             probe=lambda _: self.fail('Recovery does not require GPU'))
        self.assertEqual(state['status'], 'complete')
        self.assertTrue(state['recovered_after_worker_exit'])
        with self.assertRaises(ValueError):
            r.acknowledge(job, {'metrics_sha256': 'wrong'})
        result = r.acknowledge(job, {'metrics_sha256': state['metrics_sha256']})
        self.assertTrue(result['checkpoint_retained'])
        self.assertTrue((job / 'best.pt').exists())

    def test_transfer_failure_keeps_local_snapshot_and_records_retry(self):
        before = m.sha256(self.job / 'best.pt')
        with patch.object(r, '_ssh', return_value={'status': 'prepared', 'files': []}), \
             patch.object(r, '_rsync', side_effect=OSError('fixture disconnect')) as transfer:
            self.assertEqual(r.run_remote(self.config, self.job, self.trigger), 75)
        transfer.assert_called_once()
        self.assertEqual(m.sha256(self.job / 'best.pt'), before)
        self.assertEqual(m.read_json(self.job / 'remote_transport.json')['status'], 'retry')
        self.assertTrue((self.job / 'remote_spec.json').exists())

    def test_downloaded_complete_metrics_verified_and_retained_locally(self):
        expected = self.root / 'expected_metrics.json'
        m.write_json(expected, self.fixture.metrics(self.trigger))
        reply = {'status': 'complete', 'files': ['test_metrics.json'], 'metrics_sha256': m.sha256(expected)}
        def transfer(config, source, destination):
            self.assertTrue(str(source).endswith('/test_metrics.json'))
            shutil.copyfile(expected, destination)
        with patch.object(r, '_ssh', return_value=reply) as ssh, patch.object(r, '_rsync', side_effect=transfer):
            self.assertEqual(r.run_remote(self.config, self.job, self.trigger), 0)
        self.assertEqual([call.args[1][0] for call in ssh.call_args_list], ['--prepare', '--ensure', '--ack'])
        self.assertTrue((self.job / 'best.pt').exists())
        m.complete_job(self.job, self.trigger, retain_above_score=.92)
        self.assertEqual(m.read_json(self.job / 'status.json')['status'], 'complete')
        self.assertTrue((self.job / 'score_milestone.json').exists())
        self.assertTrue((self.job / 'best.pt').exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RemoteEvaluationFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    destination = Path(__file__).resolve().parents[1] / 'result/verification_remote_evaluate.json'
    m.write_json(destination, {'schema': 'vit_remote_eval_cpu_fixtures_v1', 'created_at': m.now(),
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'scope': 'Synthetic CPU orchestration fixtures only; no real model, SSH or GPU evaluation',
        'source_sha256': {p.name: m.sha256(p) for p in [Path(r.__file__), Path(m.__file__)]}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
