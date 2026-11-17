#!/usr/bin/env python3
"""CPU-only GPU admission tests, with real cross-process flock inheritance."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import gpu_slots as slots


@contextlib.contextmanager
def foreign_lock(path):
    command = ('import fcntl,sys; f=open(sys.argv[1],"a+"); '
               'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); '
               'print("ready",flush=True); sys.stdin.read(1)')
    child = subprocess.Popen([sys.executable, '-c', command, str(path)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if child.stdout.readline().strip() != 'ready':
            raise AssertionError(child.stderr.read())
        yield child
    finally:
        child.communicate('x', timeout=10)
        if child.returncode != 0:
            raise AssertionError('Lock fixture exited unsuccessfully')


class GpuSlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gpu-slot-fixture-')
        self.path = Path(self.temporary.name) / 'gpu3.lock'
        self.config = {'gpu_index': 3, 'minimum_free_mib': 10000,
                       'gpu_lock': str(self.path), 'max_concurrent_evaluations': 2}
        self.gpu = patch.object(slots, '_gpu_state', return_value=('GPU-fixture', 24000, set()))
        self.query = self.gpu.start()

    def tearDown(self):
        for handle in list(slots._OWNED.values()):
            if not handle.closed:
                handle.close()
        self.gpu.stop()
        self.temporary.cleanup()

    def test_default_single_slot_and_close_release(self):
        configuration = dict(self.config); configuration.pop('max_concurrent_evaluations')
        handle = slots.acquire_gpu_slot(configuration)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.name, str(self.path))
        self.assertTrue(slots.available(configuration)[0])
        self.assertIsNone(slots.acquire_gpu_slot(configuration))
        handle.close()
        replacement = slots.acquire_gpu_slot(configuration)
        self.assertIsNotNone(replacement)
        replacement.close()

    def test_default_single_slot_acquire_does_not_query_gpu(self):
        configuration = dict(self.config); configuration.pop('max_concurrent_evaluations')
        self.query.side_effect = AssertionError('Legacy acquisition must remain a pure lock')
        handle = slots.acquire_gpu_slot(configuration)
        self.assertIsNotNone(handle)
        handle.close()

    def test_real_legacy_slot_zero_can_share_slot_one_and_cap_at_two(self):
        with foreign_lock(self.path) as first:
            self.query.return_value = ('GPU-fixture', 17000, {first.pid})
            with patch.object(slots, '_evaluation_process', return_value=True):
                self.assertTrue(slots.available(self.config)[0])
                handle = slots.acquire_gpu_slot(self.config)
                self.assertEqual(handle.name, str(self.path) + '.slot1')
                self.assertTrue(slots.available(self.config)[0])
                with patch.object(slots, '_owned', return_value=False):
                    self.assertFalse(slots.available(self.config)[0])
                    self.assertIsNone(slots.acquire_gpu_slot(self.config))
        handle.close()

    def test_two_foreign_slots_reject_third_admission(self):
        with foreign_lock(self.path), foreign_lock(Path(str(self.path) + '.slot1')):
            self.assertIsNone(slots.acquire_gpu_slot(self.config))

    def test_other_slot_waits_for_initialization_even_when_slot_zero_is_free(self):
        for path in (self.path, Path(str(self.path) + '.slot1')):
            with self.subTest(path=path), foreign_lock(path):
                ready, reason, _ = slots.available(self.config)
                self.assertFalse(ready)
                self.assertIn('initialize', reason)
                self.assertIsNone(slots.acquire_gpu_slot(self.config))

    def test_recent_slot_cuda_context_gets_startup_grace(self):
        metadata = {'schema': 'lusnar_gpu_slot_v1', 'acquired_monotonic': time.monotonic(),
                    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        self.path.write_text(json.dumps(metadata))
        with foreign_lock(self.path) as first, patch.object(slots, '_evaluation_process', return_value=True):
            self.query.return_value = ('GPU-fixture', 23000, {first.pid})
            self.assertFalse(slots.available(self.config)[0])
            self.assertIsNone(slots.acquire_gpu_slot(self.config))
            metadata['acquired_monotonic'] -= 31
            self.path.write_text(json.dumps(metadata))
            handle = slots.acquire_gpu_slot(self.config)
            self.assertIsNotNone(handle)
            handle.close()

    def test_no_slot_unknown_training_and_insufficient_memory_are_refused(self):
        self.query.return_value = ('GPU-fixture', 24000, {100000000})
        with patch.object(slots, '_evaluation_process', return_value=True):
            self.assertFalse(slots.available(self.config)[0])
        with foreign_lock(self.path), patch.object(slots, '_evaluation_process', return_value=False):
            self.assertFalse(slots.available(self.config)[0])
            self.assertIsNone(slots.acquire_gpu_slot(self.config))
        self.query.return_value = ('GPU-fixture', 9999, set())
        self.assertFalse(slots.available(self.config)[0])
        self.assertIsNone(slots.acquire_gpu_slot(self.config))

    def test_single_slot_does_not_share_even_with_known_evaluation(self):
        configuration = {**self.config, 'max_concurrent_evaluations': 1}
        with foreign_lock(Path(str(self.path) + '.slot1')):
            self.assertIsNone(slots.acquire_gpu_slot(configuration))
        with foreign_lock(self.path), patch.object(slots, '_evaluation_process', return_value=True):
            self.query.return_value = ('GPU-fixture', 17000, {100000000})
            self.assertFalse(slots.available(configuration)[0])

    def test_one_pinned_trainer_can_share_but_second_process_or_reused_pid_cannot(self):
        with foreign_lock(self.path.with_name('trainer_fixture.lock')) as trainer:
            stat = Path(f'/proc/{trainer.pid}/stat').read_text().rsplit(')', 1)[1].split()
            identity = {'pid': trainer.pid, 'start_ticks': stat[19],
                        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            configuration = {**self.config, 'max_concurrent_evaluations': 1,
                             'allowed_training_processes': [identity]}
            self.query.return_value = ('GPU-fixture', 12000, {trainer.pid})
            handle = slots.acquire_gpu_slot(configuration)
            self.assertTrue(slots.available(configuration)[0])
            self.query.return_value = ('GPU-fixture', 12000, {trainer.pid, 999000})
            self.assertFalse(slots.available(configuration)[0])
            self.query.return_value = ('GPU-fixture', 12000, {trainer.pid})
            self.assertFalse(slots.available({**configuration, 'allowed_training_processes': [
                {**identity, 'start_ticks': '0'}]})[0])
            with foreign_lock(Path(str(self.path) + '.slot1')):
                self.assertFalse(slots.available(configuration)[0])
            handle.close()
            with foreign_lock(self.path):
                self.assertIsNone(slots.acquire_gpu_slot(configuration))

    def test_training_sharing_never_allows_two_trainers_or_two_evaluations(self):
        identity = {'pid': 1, 'start_ticks': '1', 'boot_id': 'fixture'}
        for fields in ({'allowed_training_processes': [identity]},
                       {'max_concurrent_evaluations': 1, 'allowed_training_processes': [identity, identity]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                slots.available({**self.config, **fields})

    def test_allocator_wrapper_preserves_frozen_script_and_rejects_bad_limit(self):
        command = ['/python', '-B', '-u', '/frozen/evaluate.py', '--checkpoint', '/a.pt']
        self.assertEqual(slots.evaluation_command(command, {}), command)
        wrapped = slots.evaluation_command(command, {'evaluation_memory_fraction': 0.22})
        self.assertEqual(wrapped[-3:], ['/frozen/evaluate.py', '--checkpoint', '/a.pt'])
        self.assertEqual(wrapped[3:7], [str(Path(slots.__file__).with_name('evaluate_with_memory_limit.py')),
                                      '--fraction', '0.22', '--'])
        for fraction in (0, 1.01, float('nan'), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                slots.evaluation_command(command, {'evaluation_memory_fraction': fraction})

    def test_own_context_and_host_namespace_alias_are_ignored(self):
        self.query.return_value = ('GPU-fixture', 17000, {os.getpid(), 999000})
        with patch.object(slots, '_self_pids', return_value={os.getpid(), 999000}):
            handle = slots.acquire_gpu_slot(self.config)
            self.assertIsNotNone(handle)
            self.assertTrue(slots.available(self.config)[0])
            handle.close()
        with patch.object(Path, 'read_text', return_value='Name:\tpython\nNSpid:\t999000\t123\n'):
            self.assertEqual(slots._self_pids(), {os.getpid(), 999000, 123})

    def test_slot_fd_inherited_by_child_outlives_parent_handle(self):
        handle = slots.acquire_gpu_slot(self.config)
        child = subprocess.Popen([sys.executable, '-c',
            'import os,sys; os.fstat(int(sys.argv[1])); print("ready",flush=True); sys.stdin.read(1)',
            str(handle.fileno())], pass_fds=(handle.fileno(),), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            handle.close()
            self.assertIsNone(slots._lock(self.path))
        finally:
            child.communicate('x', timeout=10)
        self.assertEqual(child.returncode, 0)
        released = slots._lock(self.path)
        self.assertIsNotNone(released)
        released.close()

    def test_cmdline_checks_script_not_arbitrary_argument(self):
        for args, expected in [
                (['/env/bin/python', '-B', '-u', '/project/code/evaluate.py', '--checkpoint', 'a'], True),
                (['/env/bin/python', '/project/code/diagnose_bc_scores.py'], True),
                (['/env/bin/python', '-B', '/project/code/export_worst_predictions.py', '--config', 'a.json'], True),
                (['/env/bin/python', '/project/code/train.py', '/project/code/export_worst_predictions.py'], False),
                (['/env/bin/python', '/project/code/train.py', '/project/code/evaluate.py'], False),
                (['/env/bin/python', '-c', 'something', '/project/code/evaluate.py'], False),
                (['/env/bin/python', '-m', 'torch.distributed.run', '/project/code/evaluate.py'], False),
                (['unrelated'], False)]:
            with self.subTest(args=args), patch.object(Path, 'read_bytes', return_value='\0'.join(args).encode()):
                self.assertEqual(slots._evaluation_process(123), expected)

    def test_query_parsing_gpu_filter_and_fail_closed(self):
        with patch.object(slots, '_query', side_effect=['GPU-X, 12345', 'GPU-X, 123\nGPU-other, 456']):
            self.assertEqual(slots._gpu_state.__wrapped__(self.config) if hasattr(slots._gpu_state, '__wrapped__')
                             else self.gpu.temp_original(self.config), ('GPU-X', 12345, {123}))
        self.query.side_effect = subprocess.TimeoutExpired('nvidia-smi', 15)
        self.assertFalse(slots.available(self.config)[0])
        self.assertIsNone(slots.acquire_gpu_slot(self.config))

    def test_invalid_limit_or_startup_settings_rejected(self):
        for limit in (0, 3, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                slots.available({**self.config, 'max_concurrent_evaluations': limit})
        for grace in (-1, 301, float('nan'), True):
            with self.subTest(grace=grace), self.assertRaises(ValueError):
                slots.acquire_gpu_slot({**self.config, 'gpu_startup_grace_seconds': grace})


if __name__ == '__main__':
    unittest.main(verbosity=2)
