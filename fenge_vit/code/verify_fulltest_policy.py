#!/usr/bin/env python3
"""CPU checks for explicit policy migration and exact training continuation.

The runtime fixture uses the real distributed trainer in one-process CPU mode
with a tiny stochastic model/dataset. It exercises AdamW, the real schedule,
sampler, RNG restoration, checkpoint publication and both resume paths.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fulltest_policy as policy
import train_hardval as base
import train_hardval_distributed as distributed


def binding(output, saved, new_sources):
    document = {'schema': policy.TRANSITION_SCHEMA, 'from_policy': policy.LEGACY_POLICY,
                'to_policy': policy.FULL_TEST_POLICY, 'resume_epoch': saved['epoch'],
                'resume_checkpoint_sha256': policy.file_sha256(output / 'last.pt'),
                'old_source_sha256': saved['train_config']['source_sha256'],
                'new_source_sha256': new_sources}
    base.write_json(output / 'policy_transition.json', document)
    return document


class PolicyFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.model = base.ModelConfig()
        self.identities = {'train': {'manifest_sha256': 'train'}, 'val': {'manifest_sha256': 'val'}}
        self.old = {'output': str(self.output), 'epochs': 30, 'batch_size': 4, 'accumulation': 1,
                    'head_lr': 0.0001, 'early_stopping_patience': 5,
                    'source_sha256': {'code/train_hardval.py': 'old', 'code/vitseg/model.py': 'same'},
                    'distributed': {'world_size': 2}}
        self.saved = {'format': 'lusnar_vit_checkpoint_v1', 'model_config': self.model.to_dict(),
                      'data_identity': self.identities, 'train_config': self.old,
                      'epoch': 4, 'global_update': 3000, 'best_epoch': 4, 'best_miou': 0.8,
                      'model': {'weight': torch.tensor([1.25])},
                      'early_stopping': base.PatienceTracker().state_dict()}
        base.atomic_checkpoint(self.output / 'last.pt', self.saved)
        self.original_sha = policy.file_sha256(self.output / 'last.pt')
        self.new = {**copy.deepcopy(self.old), 'selection_policy': policy.FULL_TEST_POLICY,
                    'early_stopping_patience': 0, 'resume': str(self.output / 'last.pt'),
                    'source_sha256': {'code/train_hardval.py': 'new', 'code/fulltest_policy.py': 'helper',
                                      'code/vitseg/model.py': 'same'}}
        binding(self.output, self.saved, self.new['source_sha256'])

    def validate(self, saved=None, configuration=None, identities=None):
        return base._check_resume(saved or self.saved, configuration or self.new,
                                  self.model, identities or self.identities)

    def test_migration_preserves_checkpoint_bytes_and_tensor(self):
        audit = self.validate()
        self.assertEqual(audit['legacy_validation_best']['score'], 0.8)
        self.assertEqual(audit['checkpoint_epoch'], 4)
        self.assertEqual(policy.file_sha256(self.output / 'last.pt'), self.original_sha)
        self.assertTrue(torch.equal(self.saved['model']['weight'], torch.tensor([1.25])))
        self.assertNotIn('selection_policy', self.old)

    def test_rejects_unrelated_training_changes(self):
        for key, value in [('head_lr', 0.01), ('batch_size', 2), ('accumulation', 2),
                           ('epochs', 31), ('distributed', {'world_size': 4})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(configuration={**self.new, key: value})
        with self.assertRaisesRegex(ValueError, 'data identity'):
            self.validate(identities={'train': 'different', 'val': 'different'})
        saved = {**self.saved, 'model_config': {**self.model.to_dict(), 'num_queries': 33}}
        with self.assertRaisesRegex(ValueError, 'model configuration'):
            self.validate(saved=saved)

    def test_requires_exact_binding(self):
        path = self.output / 'policy_transition.json'
        original = json.loads(path.read_text())
        for key, value in [('resume_checkpoint_sha256', 'wrong'), ('old_source_sha256', {}),
                           ('new_source_sha256', {}), ('resume_epoch', 3)]:
            with self.subTest(key=key):
                base.write_json(path, {**original, key: value})
                with self.assertRaises(ValueError):
                    self.validate()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_even_bound_model_code_changes_are_rejected(self):
        configuration = copy.deepcopy(self.new)
        configuration['source_sha256']['code/vitseg/model.py'] = 'changed'
        binding(self.output, self.saved, configuration['source_sha256'])
        with self.assertRaisesRegex(ValueError, 'implementation sources'):
            self.validate(configuration=configuration)

    def test_same_policy_resume_stays_strict_and_cannot_reverse(self):
        saved = {**self.saved, 'train_config': self.new, 'selection_policy': policy.FULL_TEST_POLICY}
        self.assertIsNone(self.validate(saved=saved))
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.validate(saved=saved, configuration={**self.new, 'source_sha256': {}})
        with self.assertRaisesRegex(ValueError, 'Only hard-validation'):
            self.validate(saved=saved, configuration=self.old)
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            self.validate(saved={**saved, 'inference_only': True})

    def test_full_policy_has_no_validation_even_final_and_no_early_stop(self):
        args = base.parse_args(['--selection-policy', policy.FULL_TEST_POLICY, '--early-stopping-patience', '0'])
        self.assertFalse(any(base.should_validate(epoch, args) for epoch in range(1, 31)))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            base.parse_args(['--selection-policy', policy.FULL_TEST_POLICY])
        metadata = policy.policy_metadata(args, 5, self.validate())
        self.assertIsNone(metadata['best_epoch'])
        self.assertFalse(metadata['selected_by_test'])
        self.assertTrue(metadata['full_test_candidate'])
        self.assertFalse(metadata['early_stopping_enabled'])

    def test_history_preserves_legacy_prefix_without_repairing_legacy_best(self):
        raw = b''.join((json.dumps({'epoch': i, 'best_epoch': 2, 'best_miou': 0.8}) + '\n').encode()
                       for i in range(1, 5))
        (self.output / 'history.jsonl').write_bytes(raw)
        saved = {**self.saved, 'best_epoch': 2, '_resume_path': str(self.output / 'last.pt')}
        for trainer in (base, distributed):
            trainer._validate_history(self.output, saved, False, policy.FULL_TEST_POLICY)
            self.assertEqual((self.output / 'history.jsonl').read_bytes(), raw)
            self.assertFalse((self.output / 'best.pt').exists())
        with self.assertRaises(FileNotFoundError):
            base._validate_history(self.output, saved, False)

    def test_candidate_is_lightweight_immutable_and_final_shares_last(self):
        args = SimpleNamespace(selection_policy=policy.FULL_TEST_POLICY, epochs=5, early_stopping_patience=0)
        checkpoint = {**self.saved, 'epoch': 5, 'train_config': self.new,
                      'optimizer': {'large': torch.ones(100)}, 'scheduler': {}, 'scaler': {},
                      'rng': {}, 'rank_states': [], 'loader_generator': torch.ones(3), 'val_generator': torch.ones(3)}
        checkpoint.update(policy.policy_metadata(args, 5, self.validate()))
        lightweight = policy.inference_checkpoint(checkpoint)
        self.assertIs(lightweight['model']['weight'], checkpoint['model']['weight'])
        self.assertFalse(set(lightweight) & policy.REMOVED_INFERENCE_KEYS)
        base.atomic_checkpoint(self.output / 'last.pt', checkpoint)
        policy.publish_checkpoints(self.output, checkpoint, 5, base.atomic_checkpoint)
        candidate = self.output / 'full_test_checkpoints/epoch_005.pt'
        self.assertTrue(torch.load(candidate, weights_only=True)['inference_only'])
        self.assertEqual((self.output / 'last.pt').stat().st_ino, (self.output / 'final.pt').stat().st_ino)
        candidate_sha = policy.file_sha256(candidate)
        with self.assertRaises(FileExistsError):
            policy.publish_checkpoints(self.output, checkpoint, 5, base.atomic_checkpoint)
        policy.publish_checkpoints(self.output, checkpoint, 5, base.atomic_checkpoint, recovering=True)
        self.assertEqual(policy.file_sha256(candidate), candidate_sha)
        changed = {**checkpoint, 'model': {'weight': torch.tensor([9.])}}
        with self.assertRaisesRegex(ValueError, 'differs'):
            policy.publish_checkpoints(self.output, changed, 5, base.atomic_checkpoint, recovering=True)

    def test_receipt_skips_deleted_weights_only_with_verified_completed_metrics(self):
        args = SimpleNamespace(selection_policy=policy.FULL_TEST_POLICY, epochs=30, early_stopping_patience=0)
        checkpoint = {**self.saved, 'epoch': 5, 'train_config': self.new}
        checkpoint.update(policy.policy_metadata(args, 5, self.validate()))
        policy.publish_checkpoints(self.output, checkpoint, 30, base.atomic_checkpoint)
        path = self.output / 'full_test_checkpoints/epoch_005.pt'
        receipt_path = path.with_suffix('.json')
        path.unlink()
        policy.publish_checkpoints(self.output, checkpoint, 30, base.atomic_checkpoint, recovering=True)
        receipt = json.loads(receipt_path.read_text())
        self.assertTrue(path.exists())
        self.assertEqual(len(receipt['reconstruction_history']), 1)
        metrics = self.output / 'test_metrics.json'
        base.write_json(metrics, {'image_count': 5004, 'weighted_score': 0.91})
        receipt['evaluation'] = {'status': 'complete', 'checkpoint_sha256': receipt['checkpoint_sha256'],
                                 'metrics_path': str(metrics), 'metrics_sha256': policy.file_sha256(metrics)}
        base.write_json(receipt_path, receipt)
        path.unlink()
        policy.publish_checkpoints(self.output, checkpoint, 30, base.atomic_checkpoint, recovering=True)
        self.assertFalse(path.exists())
        # A corrupt/missing metrics proof must not silently lose pending work.
        metrics.write_text('{}\n')
        policy.publish_checkpoints(self.output, checkpoint, 30, base.atomic_checkpoint, recovering=True)
        self.assertTrue(path.exists())
        self.assertEqual(len(json.loads(receipt_path.read_text())['reconstruction_history']), 2)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, *args, **kwargs):
        self.split = split
        self.records = [{'source_id': f'{split}/{i}', 'scene': 'Moon_2', 'sample_id': str(i)}
                        for i in range(4 if split == 'train' else 1)]
        self.partial = True
        self.identity = {'schema': base.SCHEMA, 'split': split, 'image_count': len(self), 'partial': True,
                         'hard_manifest_sha256': 'fixture', 'checkpoint_selection_uses_test': True,
                         'sample_scene_counts': {'Moon_2': len(self)}}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {**self.records[index], 'image': torch.rand(3, 4, 4) + random.random() + np.random.random(),
                'mask': torch.full((4, 4), index % 5, dtype=torch.long), 'horizontal_flipped': bool(random.getrandbits(1))}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 5, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, images):
        return {'semantic_logits': self.decoder(self.dropout(self.encoder(images))).log_softmax(1)}

    def head_parameters(self):
        return self.decoder.parameters()

    def backbone_parameters(self):
        return self.encoder.parameters()

    def freeze_backbones(self, frozen):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)

    def set_gradient_checkpointing(self, value):
        pass


class TinyCriterion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, outputs, targets):
        return {'loss_total': F.nll_loss(outputs['semantic_logits'], targets)}


class RuntimeFixtures(unittest.TestCase):
    def test_real_trainer_migrates_then_resumes_exactly_without_validation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(distributed, 'LunarDataset', TinyDataset))
            stack.enter_context(patch.object(distributed, 'build_model', lambda *a, **k: TinyModel()))
            stack.enter_context(patch.object(distributed, 'Mask2FormerCriterion', TinyCriterion))
            stack.enter_context(patch.object(distributed, 'distributed_set_loss', lambda x, y, criterion: criterion(x, y)))
            evaluated = []

            def evaluate(model, loader, *args, **kwargs):
                evaluated.append(str(loader.dataset.identity))
                return {'miou': 0.8, 'partial': True, 'image_count': 1, 'validation_unique_source_count': 1,
                        'data_identity': loader.dataset.identity, 'scenes': {'Moon_2': {'miou': 0.8, 'image_count': 1}}}

            stack.enter_context(patch.object(distributed, 'distributed_evaluate', evaluate))
            completed = []
            for name, interrupt in [('uninterrupted', False), ('resumed', True)]:
                output = Path(directory) / name
                cli = ['--output', str(output), '--device', 'cpu', '--amp', 'none', '--workers', '0',
                       '--batch-size', '2', '--accumulation', '1', '--epochs', '6', '--freeze-epochs', '1',
                       '--input-size', '32', '--no-pretrained', '--max-train-samples', '4', '--max-val-samples', '1',
                       '--masking', '--masking-update-fraction', '1', '--masking-block-size', '2',
                       '--masking-ratio', '0.25', '--masking-warmup-updates', '2',
                       '--log-every', '0']
                original_write = base.write_json

                def run(arguments, stop=None):
                    def write(path, value):
                        original_write(path, value)
                        if Path(path).name == 'summary.json' and value.get('completed_epochs') == stop:
                            raise RuntimeError('fixture_epoch_boundary_stop')
                    with patch.object(base, 'write_json', write), contextlib.redirect_stdout(io.StringIO()):
                        if stop:
                            with self.assertRaisesRegex(RuntimeError, 'fixture_epoch_boundary_stop'):
                                distributed.main(arguments)
                        else:
                            distributed.main(arguments)

                run(cli, stop=2)
                old = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
                old_history = (output / 'history.jsonl').read_bytes()
                old_best_sha = policy.file_sha256(output / 'best_smoke.pt')
                binding(output, old, old['train_config']['source_sha256'])
                full_cli = cli + ['--resume', str(output / 'last.pt'), '--selection-policy', policy.FULL_TEST_POLICY,
                                  '--early-stopping-patience', '0']
                evaluation_count = len(evaluated)
                run(full_cli, stop=5 if interrupt else None)
                if interrupt:
                    five_sha = policy.file_sha256(output / 'full_test_checkpoints/epoch_005.pt')
                    run(full_cli)
                    self.assertEqual(policy.file_sha256(output / 'full_test_checkpoints/epoch_005.pt'), five_sha)
                self.assertEqual(len(evaluated), evaluation_count)
                self.assertTrue((output / 'history.jsonl').read_bytes().startswith(old_history))
                self.assertEqual(policy.file_sha256(output / 'best_smoke.pt'), old_best_sha)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'complete')
                self.assertIsNone(summary['best_epoch'])
                self.assertFalse(summary['early_stopping_enabled'])
                self.assertFalse((output / 'validation_epoch_004.json').exists())
                self.assertEqual((output / 'last.pt').stat().st_ino, (output / 'final.pt').stat().st_ino)
                completed.append(torch.load(output / 'last.pt', map_location='cpu', weights_only=True))
            reference, resumed = completed
            self.assertEqual(reference['global_update'], 12)
            self.assertEqual(reference['scheduler'], resumed['scheduler'])
            self.assertEqual(reference['scaler'], resumed['scaler'])
            self.assertEqual(reference['early_stopping'], resumed['early_stopping'])
            for key in reference['model']:
                self.assertTrue(torch.equal(reference['model'][key], resumed['model'][key]), key)
            for index, state in reference['optimizer']['state'].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, resumed['optimizer']['state'][index][key]), key)
            for key in ('loader_generator', 'val_generator'):
                self.assertTrue(torch.equal(reference['rank_states'][0][key], resumed['rank_states'][0][key]))
            self.assertTrue(torch.equal(reference['rank_states'][0]['rng']['torch'], resumed['rank_states'][0]['rng']['torch']))
            self.assertEqual(reference['rank_states'][0]['rng']['python'], resumed['rank_states'][0]['rng']['python'])
            self.assertEqual(reference['rank_states'][0]['rng']['numpy'], resumed['rank_states'][0]['rng']['numpy'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
