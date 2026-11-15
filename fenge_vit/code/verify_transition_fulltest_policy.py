#!/usr/bin/env python3
"""CPU/mock checks; never signal or launch real training processes."""
import copy
import json
import signal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import transition_fulltest_policy as transition


class TransitionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.run = root / 'run'; self.run.mkdir()
        self.old = self.run / 'runtime'; self.new = self.run / 'runtime_fulltest'
        files = {'code/train_hardval.py': 'POLICY = "old"\n',
                 'code/train_hardval_distributed.py': 'TRAIN = True\n',
                 'code/vitseg/model.py': 'MODEL = "unchanged"\n'}
        for folder in (self.old, self.new):
            for name, content in files.items():
                p = folder / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
        (self.new / 'code/train_hardval.py').write_text('POLICY = "full_test_every5"\n')
        (self.new / 'code/fulltest_policy.py').write_text('FULL_TEST = True\n')
        self.config_path = root / 'training.json'
        self.training = {'output': str(self.run), 'epochs': 30, 'batch_size': 4,
                         'accumulation': 1, 'early_stopping_patience': 5, 'pretrained': True,
                         'resume': None, 'initial_checkpoint': None}
        transition.write_json(self.config_path, self.training)
        self.recorded = {**self.training, 'source_sha256':
                         {name: transition.sha256(self.old / name) for name in files}}
        transition.write_json(self.run / 'run_config.json', self.recorded)
        self.summary = {'completed_epochs': 2, 'status': 'training', 'partial': False,
                        'data_identity': {'train': 'fixture', 'val': 'fixture'},
                        'model_config': {'variant': 'adapter'}}
        transition.write_json(self.run / 'summary.json', self.summary)
        (self.run / 'history.jsonl').write_text(''.join(json.dumps({'epoch': e, 'global_update': e * 10}) + '\n'
                                                  for e in (1, 2)))
        (self.run / 'last.pt').write_bytes(b'unmodified full checkpoint fixture')
        self.launcher = {'pid': 100, 'start_ticks': 300, 'boot_id': 'fixture', 'ppid': 1,
            'argv': ['/usr/bin/python3', '-m', 'torch.distributed.run', '--nproc_per_node=2',
                     str(self.old / transition.ENTRY), '--config', str(self.config_path)], 'cwd': str(root)}
        self.rank = {**self.launcher, 'pid': 101, 'ppid': 100,
                     'argv': ['/usr/bin/python3', str(self.old / transition.ENTRY), '--config', str(self.config_path)]}
        launch = {'identity': transition.identity(self.launcher), 'command': self.launcher['argv'],
                  'cwd': str(root), 'environment': {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second',
                                                   'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'OMP_NUM_THREADS': '2'}}
        transition.write_json(self.run / 'launch.json', launch)
        self.spec = {'run': str(self.run), 'config': str(self.config_path), 'old_runtime': str(self.old),
                     'new_runtime': str(self.new), 'launch_record': str(self.run / 'launch.json'), 'wait_after_epoch': 1}
        self.c = {'schema': transition.SCHEMA, 'state_directory': str(root / 'state'),
                  'python': '/usr/bin/python3', 'runs': [self.spec]}
        self.saved = {'format': 'lusnar_vit_checkpoint_v1', 'partial': False, 'epoch': 2,
                      'train_config': copy.deepcopy(self.recorded), 'model': {}, 'optimizer': {},
                      'scheduler': {}, 'scaler': {}, 'rng': {}, 'global_update': 20,
                      'distributed': {'world_size': 2}, 'rank_states': [
                          {'rank': r, 'rng': {}, 'loader_generator': [], 'val_generator': []} for r in range(2)],
                      'data_identity': self.summary['data_identity'], 'model_config': self.summary['model_config']}
        self.load = patch.object(transition, 'load_checkpoint', side_effect=lambda _: copy.deepcopy(self.saved)).start()
        self.ps = patch.object(transition, 'processes', return_value=[self.launcher, self.rank]).start()
        self.binding = patch.object(transition, 'validate_launcher_binding').start()
        self.real_stop = transition.stop_exact
        self.stop = patch.object(transition, 'stop_exact').start()
        self.spawn = patch.object(transition.subprocess, 'Popen', return_value=SimpleNamespace(pid=200)).start()
        self.info = patch.object(transition, 'process_info', return_value={**self.launcher, 'pid': 200}).start()
        self.addCleanup(patch.stopall)

    def state_path(self):
        return Path(self.c['state_directory']) / (transition.run_key(self.spec) + '.json')

    def test_waits_for_saved_epoch_strictly_after_requested_boundary(self):
        self.spec['wait_after_epoch'] = 2
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'waiting')
        self.load.assert_not_called(); self.stop.assert_not_called(); self.spawn.assert_not_called()

    def test_nonpolicy_model_change_fails_before_signal(self):
        (self.new / 'code/vitseg/model.py').write_text('MODEL = "changed"\n')
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Non-policy source', result['error'])
        self.stop.assert_not_called(); self.spawn.assert_not_called()

    def test_old_source_mutation_fails_before_signal(self):
        (self.old / 'code/train_hardval.py').write_text('POLICY = "modified"\n')
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called()

    def test_changed_batch_cannot_transition(self):
        transition.write_json(self.config_path, {**self.training, 'batch_size': 8})
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called()

    def test_checkpoint_missing_optimizer_cannot_stop(self):
        del self.saved['optimizer']
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called()

    def test_checkpoint_identity_mismatch_cannot_stop(self):
        self.saved['data_identity'] = {'train': 'different'}
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called()

    def test_history_gap_cannot_stop(self):
        (self.run / 'history.jsonl').write_text(json.dumps({'epoch': 2, 'global_update': 20}) + '\n')
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called()

    def test_cross_namespace_denial_explains_host_watcher_and_never_signals(self):
        self.ps.return_value = []
        denied = [{'pid': self.launcher['pid'], 'field': 'environ', 'reason': 'permission_denied'}]
        with patch.object(transition, 'launcher_access_errors', return_value=denied):
            result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('outside bwrap/run_on_103.sh', result['error'])
        self.stop.assert_not_called(); self.spawn.assert_not_called()

    def test_proc_permission_diagnostics_are_limited_to_the_exact_launcher(self):
        proc = Path(self.temporary.name) / 'fixture_proc'
        for pid, args in ((100, self.launcher['argv']),
                          (101, [a.replace(str(self.config_path), '/other/config.json')
                                 for a in self.launcher['argv']])):
            entry = proc / str(pid); entry.mkdir(parents=True)
            (entry / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in args) + b'\0')
            (entry / 'cwd').mkdir()
        read_bytes = Path.read_bytes

        def deny_environment(path):
            if path.name == 'environ':
                raise PermissionError('fixture inaccessible sibling namespace')
            return read_bytes(path)

        with patch.object(Path, 'read_bytes', new=deny_environment):
            denied = transition.launcher_access_errors(self.spec, proc)
        self.assertEqual(denied, [{'pid': 100, 'field': 'environ', 'reason': 'permission_denied'}])

    def test_success_preserves_checkpoint_and_only_changes_policy_parameters(self):
        before = (self.run / 'last.pt').read_bytes()
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'launched')
        self.stop.assert_called_once(); self.spawn.assert_called_once()
        self.assertEqual((self.run / 'last.pt').read_bytes(), before)
        config = transition.read_json(self.config_path)
        self.assertEqual(config, {**self.training, **transition.POLICY})
        record = transition.read_json(self.run / 'policy_transition.json')
        self.assertEqual(record['resume_epoch'], 2)
        self.assertEqual(record['resume_checkpoint_sha256'], transition.sha256(self.run / 'last.pt'))
        self.assertEqual(set(record['new_source_sha256']), set(self.recorded['source_sha256']) | {transition.NEW_SOURCE})
        call = self.spawn.call_args
        self.assertEqual(call.args[0][-2:], ['--resume', str(self.run / 'last.pt')])
        self.assertIn(str(self.new / transition.ENTRY), call.args[0])
        self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-first,GPU-second')
        self.assertTrue(call.kwargs['start_new_session'])
        self.assertEqual(transition.read_json(self.run / 'policy_transition_archive/config.json'), self.training)

    def test_successful_restart_detects_live_new_trainer_without_respawn(self):
        transition.tick_run(self.c, self.spec)
        new = copy.deepcopy(self.launcher)
        new['argv'] = [a.replace(str(self.old), str(self.new)) for a in new['argv']]
        self.ps.return_value = [new]
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'running')
        self.spawn.assert_called_once(); self.stop.assert_called_once()

    def test_ambiguous_spawn_intent_does_not_retry(self):
        transition.write_json(self.state_path(), {'status': 'launch_intent'})
        self.ps.return_value = []
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.spawn.assert_not_called(); self.stop.assert_not_called()

    def test_interrupted_stop_fails_closed_and_never_repeats_signal(self):
        transition.write_json(self.state_path(), {'status': 'stopping'})
        transition.tick_run(self.c, self.spec)
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.stop.assert_not_called(); self.spawn.assert_not_called()

    def test_poststop_config_mutation_prevents_spawn(self):
        self.stop.side_effect = lambda *args, **kwargs: transition.write_json(self.config_path, {**self.training, 'seed': 999})
        result = transition.tick_run(self.c, self.spec)
        self.assertEqual(result['status'], 'failed'); self.spawn.assert_not_called()

    def test_stop_rejects_reused_pid_and_wrapper(self):
        with patch.object(transition.os, 'kill') as kill:
            self.info.return_value = {**self.launcher, 'start_ticks': 999}
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.real_stop(self.spec, self.launcher, [])
            self.assertFalse(transition.matches({**self.launcher, 'argv': ['bwrap', *self.launcher['argv']]},
                                               self.spec, self.old, launcher=True))
            kill.assert_not_called()

    def test_timeout_signals_exact_launcher_once_and_does_not_escalate(self):
        self.info.return_value = self.launcher
        with patch.object(transition.os, 'kill') as kill, patch.object(transition, 'alive', return_value=True):
            with self.assertRaises(TimeoutError):
                self.real_stop(self.spec, self.launcher, [transition.identity(self.rank)], timeout=0)
            kill.assert_called_once_with(self.launcher['pid'], signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
