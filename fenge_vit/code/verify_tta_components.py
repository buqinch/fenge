#!/usr/bin/env python3
"""Verify shared TTA statistics against independent predictions, on CPU."""
import copy
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_tta_components import evaluate_with_component, publish_component, validate_component_document
from vitseg.metrics import evaluate_loader


class Images(Dataset):
    identity = {'split': 'test', 'partial': True}
    partial = True

    def __len__(self):
        return 2

    def __getitem__(self, index):
        x = torch.linspace(-1, 1, 1024)[None, :].expand(1024, -1)
        y = x.T
        image = torch.stack((x, y, (torch.arange(1024)[None, :] % 7).expand(1024, -1).float() / 3 - 1))
        return {'image': image, 'mask': ((x + 1) * 2).long().clamp(max=4),
                'scene': ['Moon_2', 'Moon_9'][index], 'source_id': str(index)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_inference(self, images):
        self.calls.append(tuple(images.shape[-2:]))
        position = torch.linspace(-0.7, 0.7, images.shape[-1])[None, None, :]
        logits = torch.stack((images[:, 0] + position, images[:, 1], -images[:, 0],
                              images[:, 2], -images[:, 2] - position), dim=1)
        return {'semantic_logits': logits.log_softmax(1)}


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.options = {'amp': 'none', 'tta': 'multiscale_hflip', 'tta_sizes': [1024, 768],
                       'tta_weights': [0.5, 0.5], 'progress_every': 0}
        cls.model = Model().train()
        cls.shared = evaluate_with_component(evaluate_loader, cls.model, DataLoader(Images()),
                                             'cpu', **cls.options)
        cls.standalone = evaluate_loader(Model(), DataLoader(Images()), 'cpu', amp='none',
                                        tta='hflip', progress_every=0)
        cls.fused = evaluate_loader(Model(), DataLoader(Images()), 'cpu', **cls.options)

    def test_shared_component_matches_standalone_hflip_without_extra_forwards(self):
        component = self.shared['tta_component_metrics']['1024']
        self.assertEqual(self.model.calls, [(1024, 1024), (1024, 1024), (768, 768), (768, 768)] * 2)
        self.assertTrue(self.model.training)
        self.assertEqual(self.shared['loader_passes'], 1)
        self.assertEqual(component['additional_model_forward_calls'], 0)
        self.assertEqual(component['reused_model_forward_calls'], 4)
        self.assertEqual(component['confusion_matrix_gt_rows_pred_cols'],
                         self.standalone['confusion_matrix_gt_rows_pred_cols'])
        self.assertEqual(component['scenes'], self.standalone['scenes'])
        validate_component_document(self.shared, component)

    def test_primary_fusion_is_unchanged_and_not_confused_with_component(self):
        for key in ('confusion_matrix_gt_rows_pred_cols', 'scenes', 'miou', 'class_iou',
                    'model_forward_calls', 'tta_views', 'tta_view_weights'):
            self.assertEqual(self.shared[key], self.fused[key])
        self.assertNotEqual(self.shared['confusion_matrix_gt_rows_pred_cols'],
                            self.standalone['confusion_matrix_gt_rows_pred_cols'])

    def test_checkpoint_provenance_is_published_and_tampering_is_rejected(self):
        document = copy.deepcopy(self.shared)
        document.update(checkpoint_sha256='fixture_sha', checkpoint_epoch=14,
                        model_config={'input_size': 1024}, split='test', seconds=1.5)
        written = {}
        publish_component(document, '/fixture/component.json', lambda path, value: written.update({str(path): value}))
        component = written['/fixture/component.json']
        self.assertEqual(component['checkpoint_epoch'], 14)
        self.assertEqual(component['checkpoint_sha256'], 'fixture_sha')
        self.assertEqual(component['tta'], 'hflip')
        for key, value in [('checkpoint_sha256', 'wrong'), ('model_forward_calls', 8), ('miou', 0)]:
            invalid = copy.deepcopy(component)
            invalid[key] = value
            with self.assertRaises(ValueError):
                validate_component_document(document, invalid)

    def test_missing_component_is_rejected_before_any_forward(self):
        model = Model()
        with self.assertRaises(ValueError):
            evaluate_with_component(evaluate_loader, model, DataLoader(Images()), 'cpu',
                                    **{**self.options, 'tta_weights': [0, 1]})
        self.assertEqual(model.calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
