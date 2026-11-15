#!/usr/bin/env python3
"""CPU-only fixtures for atomic text mirroring across a training-policy change."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sync_hardval_results as sync
import sync_remote_results as mirror


class PolicyMirrorFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='vit-policy-mirror-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local, self.stage = self.root / 'local', self.root / 'stage'
        self.local.mkdir(); self.stage.mkdir()
        self.target = {'local': self.local}
        replacement = patch.object(mirror, 'validate_boundary', sync.validate_boundary)
        replacement.start(); self.addCleanup(replacement.stop)
        self.identity = {'train': {'image_count': 6000, 'ordered_samples_sha256': 'train-fixed'},
                         'val': {'image_count': 1500, 'hard_manifest_sha256': 'hard-fixed'}}
        self.old_config = {'protocol': sync.PROTOCOL, 'validation_every': 2, 'epochs': 30,
                           'early_stopping_patience': 5, 'batch_size': 4,
                           'source_sha256': {'code/train_hardval.py': 'old-training',
                                             'code/vitseg/data.py': 'unchanged-data'},
                           'resume': None, 'config': 'old-config.json'}
        self.new_config = {**self.old_config, 'selection_policy': sync.FULL_TEST_POLICY,
                           'early_stopping_patience': 0, 'resume': '/run/last.pt',
                           'config': 'new-config.json',
                           'source_sha256': {'code/train_hardval.py': 'new-training',
                                             'code/vitseg/data.py': 'unchanged-data',
                                             'code/fulltest_policy.py': 'new-helper'}}
        self.transition = {'schema': 'lusnar_training_policy_transition_v1',
                           'from_policy': sync.LEGACY_POLICY, 'to_policy': sync.FULL_TEST_POLICY,
                           'resume_epoch': 2, 'resume_checkpoint_sha256': 'a' * 64,
                           'old_source_sha256': self.old_config['source_sha256'],
                           'new_source_sha256': self.new_config['source_sha256']}
        self.old_rows = [
            {'epoch': 1, 'validation_performed': False, 'validation_miou': None,
             'validation_weighted_score': None, 'best_epoch': None, 'best_miou': None},
            {'epoch': 2, 'validation_performed': True, 'validation_miou': .9,
             'validation_weighted_score': .8, 'best_epoch': 2, 'best_miou': .8}]
        self.old_summary = {'completed_epochs': 2, 'status': 'training',
                            'best_metric': 'validation_weighted_score', 'best_epoch': 2, 'best_miou': .8,
                            'checkpoint_selection_uses_test': True,
                            'official_test_used_for_selection': True, 'data_identity': self.identity}
        self.put('run_config.json', self.old_config)
        self.put('summary.json', self.old_summary)
        self.history(self.old_rows)
        self.put('validation_epoch_002.json', {'epoch': 2, 'miou': .9,
                 'validation_weighted_score': .8, 'data_identity': self.identity['val'],
                 'checkpoint_selection_uses_test': True})
        self.assertTrue(self.publish()['summary_published'])
        self.clear_stage()

    def put(self, name, value):
        mirror.write_json(self.stage / name, value)

    def history(self, rows):
        (self.stage / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def clear_stage(self):
        for path in self.stage.iterdir():
            path.unlink()

    def publish(self):
        return mirror.publish(self.target, self.stage)

    def local_json(self, name):
        return mirror.strict_json((self.local / name).read_text())

    def metadata(self):
        return {'selection_policy': sync.FULL_TEST_POLICY, 'best_metric': 'weighted_score',
                'best_epoch': None, 'best_miou': None, 'best_validation_score': None,
                'validation_performed': False, 'validation_miou': None,
                'validation_weighted_score': None, 'miou': None, 'validation_interval': None,
                'latest_validation_epoch': None, 'early_stopping_enabled': False,
                'hard_validation_enabled': False, 'full_test_every_epochs': 5,
                'checkpoint_selection_uses_test': True,
                'early_stopping': {'patience': 0, 'stopped': False}, 'status': 'training'}

    def migration_payload(self, completed=3):
        self.put('run_config.json', self.new_config)
        self.put('policy_transition.json', self.transition)
        summary = {**self.old_summary, **self.metadata(), 'completed_epochs': completed}
        rows = copy.deepcopy(self.old_rows) + [{**self.metadata(), 'epoch': epoch}
                                              for epoch in range(3, completed + 1)]
        self.put('summary.json', summary); self.history(rows)
        return summary, rows

    def test_full_summary_uses_weighted_score_and_null_validation_best(self):
        self.migration_payload()
        result = self.publish()
        self.assertTrue(result['summary_published'])
        self.assertEqual(self.local_json('summary.json')['best_metric'], 'weighted_score')
        self.assertIsNone(self.local_json('summary.json')['best_epoch'])
        self.assertEqual(self.local_json('summary.json')['data_identity'], self.identity)
        self.assertFalse((self.local / 'validation_epoch_003.json').exists())

    def test_new_config_with_old_summary_preserves_legacy_boundary(self):
        self.put('run_config.json', self.new_config)
        self.put('policy_transition.json', self.transition)
        self.assertTrue(self.publish()['summary_published'])
        self.assertEqual(self.local_json('summary.json'), self.old_summary)
        self.assertEqual(self.local_json('run_config.json'), self.new_config)
        self.clear_stage()
        summary = {**self.old_summary, **self.metadata(), 'completed_epochs': 3}
        self.put('summary.json', summary)
        self.history(self.old_rows + [{**self.metadata(), 'epoch': 3}])
        self.assertTrue(self.publish()['summary_published'])
        self.assertEqual(self.local_json('summary.json'), summary)

    def test_history_ahead_does_not_advance_old_summary(self):
        self.migration_payload()
        self.put('summary.json', self.old_summary)
        result = self.publish()
        self.assertEqual(result['completed_epochs'], 2)
        self.assertEqual(self.local_json('summary.json'), self.old_summary)

    def test_summary_waits_for_missing_new_history_epoch(self):
        summary, rows = self.migration_payload()
        self.history(rows[:2])
        result = self.publish()
        self.assertEqual(result['status'], 'pending')
        self.assertFalse(result['summary_published'])
        self.assertEqual(self.local_json('summary.json'), self.old_summary)
        self.clear_stage(); self.put('summary.json', summary); self.history(rows)
        self.assertTrue(self.publish()['summary_published'])

    def test_new_summary_before_new_config_waits_without_relabeling(self):
        summary, rows = self.migration_payload()
        (self.stage / 'run_config.json').unlink()
        result = self.publish()
        self.assertEqual(result['pending_reason'], 'waiting_for_full_test_run_config')
        self.assertEqual(self.local_json('summary.json'), self.old_summary)
        self.clear_stage(); self.put('run_config.json', self.new_config)
        self.put('summary.json', summary); self.history(rows)
        self.assertTrue(self.publish()['summary_published'])

    def test_fresh_full_test_run_needs_no_fake_legacy_prefix(self):
        for path in self.local.iterdir(): path.unlink()
        self.put('run_config.json', self.new_config)
        self.put('summary.json', {**self.old_summary, **self.metadata(), 'completed_epochs': 1})
        self.history([{**self.metadata(), 'epoch': 1}])
        self.assertTrue(self.publish()['summary_published'])

    def test_config_change_requires_matching_transition_record(self):
        cases = ('missing', 'schema', 'from_policy', 'to_policy', 'resume_epoch', 'old_source', 'new_source')
        for case in cases:
            with self.subTest(case=case):
                self.clear_stage(); self.migration_payload()
                transition = copy.deepcopy(self.transition)
                if case == 'missing': (self.stage / 'policy_transition.json').unlink()
                else:
                    if case in ('schema', 'from_policy', 'to_policy'): transition[case] = 'wrong'
                    elif case == 'resume_epoch': transition[case] = 0
                    else: transition[case + '_sha256'] = {'wrong': 'source'}
                    self.put('policy_transition.json', transition)
                with self.assertRaises(ValueError): self.publish()
                self.assertEqual(self.local_json('run_config.json'), self.old_config)

    def test_transition_cannot_change_model_data_or_batch(self):
        for changed in ('batch_size', 'data_source'):
            with self.subTest(changed=changed):
                self.clear_stage(); self.migration_payload()
                config, transition = copy.deepcopy(self.new_config), copy.deepcopy(self.transition)
                if changed == 'batch_size': config['batch_size'] = 8
                else:
                    config['source_sha256']['code/vitseg/data.py'] = 'changed-data'
                    transition['new_source_sha256'] = config['source_sha256']
                    self.put('policy_transition.json', transition)
                self.put('run_config.json', config)
                with self.assertRaises(ValueError): self.publish()

    def test_full_summary_rejects_validation_best_scores_and_enabled_stopping(self):
        cases = [('best_metric', 'validation_weighted_score'), ('best_epoch', 2),
                 ('best_miou', .8), ('best_validation_score', .8), ('validation_performed', True),
                 ('validation_weighted_score', .8), ('early_stopping_enabled', True),
                 ('hard_validation_enabled', True), ('status', 'early_stopped'),
                 ('early_stopping', {'patience': 5, 'stopped': False})]
        for field, value in cases:
            with self.subTest(field=field):
                self.clear_stage(); summary, _ = self.migration_payload()
                summary[field] = value; self.put('summary.json', summary)
                with self.assertRaises(ValueError): self.publish()

    def test_new_history_rows_must_follow_full_policy(self):
        for field, value in [('selection_policy', sync.LEGACY_POLICY), ('best_epoch', 2),
                             ('validation_performed', True), ('validation_miou', .99),
                             ('early_stopping_enabled', True)]:
            with self.subTest(field=field):
                self.clear_stage(); _, rows = self.migration_payload()
                rows[-1][field] = value; self.history(rows)
                with self.assertRaises(ValueError): self.publish()

    def test_inherited_training_and_validation_identities_cannot_change(self):
        for split in ('train', 'val'):
            with self.subTest(split=split):
                self.clear_stage(); summary, _ = self.migration_payload()
                summary = copy.deepcopy(summary)
                summary['data_identity'][split]['image_count'] -= 1
                self.put('summary.json', summary)
                with self.assertRaises(ValueError): self.publish()

    def test_legacy_validation_prefix_remains_checked_after_migration(self):
        self.migration_payload()
        val = self.local_json('validation_epoch_002.json')
        self.put('validation_epoch_002.json', {**val, 'validation_weighted_score': .999})
        with self.assertRaises(ValueError): self.publish()

    def test_new_policy_patience_must_be_zero(self):
        self.migration_payload()
        self.put('run_config.json', {**self.new_config, 'early_stopping_patience': 5})
        with self.assertRaises(ValueError): self.publish()

    def test_final_epoch_has_no_new_validation_or_best(self):
        summary, rows = self.migration_payload(completed=30)
        summary['status'] = rows[-1]['status'] = 'complete'
        self.put('summary.json', summary); self.history(rows)
        self.assertTrue(self.publish()['summary_published'])
        self.assertIsNone(self.local_json('summary.json')['best_epoch'])
        self.assertFalse((self.local / 'validation_epoch_030.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
