#!/usr/bin/env python3
"""Report 1024 hflip metrics from the forwards of a frozen multiscale evaluator.

The frozen evaluator still computes the primary prediction, unchanged. This
adapter observes its actual forward outputs and accumulates a second confusion
matrix, without another model call, dataset pass, or change to training sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys


def evaluate_with_component(evaluate_loader, model, loader, device, *, component_size=1024, **kwargs):
    from vitseg.metrics import ConfusionMatrix, tta_metadata

    metadata = tta_metadata(kwargs.get('tta', 'none'), kwargs.get('tta_sizes'), kwargs.get('tta_weights'))
    if (metadata['tta_mode'] != 'multiscale_hflip' or component_size != 1024
            or component_size not in [v['input_size'] for v in metadata['tta_views']]):
        raise ValueError('Component reporting requires an active 1024 scale in multiscale hflip')
    views = metadata['tta_views']
    original_observer = kwargs.get('diagnostic_observer')
    total, scenes = ConfusionMatrix(), {}

    class CaptureModel:
        def __init__(self):
            self.position = 0
            self.pair = None
            self.forward_calls = 0
            self.component_calls = 0

        @property
        def training(self):
            return model.training

        def eval(self):
            model.eval()
            return self

        def train(self, mode=True):
            model.train(mode)
            return self

        def forward_inference(self, images):
            expected = views[self.position]
            if tuple(images.shape[-2:]) != (expected['input_size'], expected['input_size']):
                raise ValueError('Frozen evaluator forward order differs from declared TTA views')
            output = getattr(model, 'forward_inference', model)(images)
            self.forward_calls += 1
            self.position += 1
            if expected['input_size'] == component_size:
                scores = output['semantic_logits'].float().exp()
                if tuple(scores.shape[-2:]) != (component_size, component_size):
                    raise ValueError('1024 component must use the native model output grid')
                if expected['transform'] == 'horizontal_flip':
                    scores = scores.flip(-1)
                self.pair = scores if self.pair is None else self.pair + scores
                self.component_calls += 1
            return output

    capture = CaptureModel()

    class Observer:
        def update(self, batch, targets, prediction, probabilities):
            if capture.position != len(views) or capture.pair is None:
                raise ValueError('Incomplete multiscale forward batch')
            if tuple(targets.shape[-2:]) != (component_size, component_size):
                raise ValueError('Component metrics require native 1024 targets')
            component_prediction = (capture.pair * 0.5).argmax(1)
            total.update(targets, component_prediction)
            for index, scene in enumerate(batch['scene']):
                entry = scenes.setdefault(str(scene), {'matrix': ConfusionMatrix(), 'image_count': 0})
                entry['matrix'].update(targets[index], component_prediction[index])
                entry['image_count'] += 1
            capture.position = 0
            capture.pair = None
            if original_observer is not None:
                original_observer.update(batch, targets, prediction, probabilities)

        def compute(self):
            return original_observer.compute() if original_observer is not None else {}

    primary = evaluate_loader(capture, loader, device, **{**kwargs, 'diagnostic_observer': Observer()})
    if capture.position or capture.forward_calls != primary['model_forward_calls']:
        raise ValueError('Actual forwards differ from the primary evaluation call count')
    if original_observer is None:
        primary.pop('object_diagnostics', None)
    component = {**primary, **total.compute(), **tta_metadata('hflip'),
        'tta': 'hflip', 'tta_sizes': None, 'tta_weights': None,
        'tta_input_sizes': [component_size],
        'scenes': {scene: {**entry['matrix'].compute(), 'image_count': entry['image_count']}
                   for scene, entry in sorted(scenes.items())},
        'weighted_score': None, 'model_forward_calls': 0, 'loader_passes': 0,
        'reused_model_forward_calls': capture.component_calls, 'shared_loader_passes': 1,
        'additional_model_forward_calls': 0, 'component_input_size': component_size,
        'source_tta': metadata['tta_mode'], 'source_tta_input_sizes': metadata['tta_input_sizes'],
        'source_model_forward_calls': capture.forward_calls,
        'tta_aggregation': 'float32_mean_of_original_and_inverse_horizontal_flip_probabilities',
        'model_forward_calls_definition': 'No additional calls; reused_model_forward_calls counts the shared 1024 views'}
    component.pop('object_diagnostics', None)
    for key in ('tta_scale_weights', 'tta_view_weights', 'tta_score_resize', 'tta_input_antialias'):
        component.pop(key, None)
    if primary.get('weighted_score') is not None:
        values = component['scenes']
        component['weighted_score'] = (values['Moon_2']['miou'] + values['Moon_4']['miou']
                                       + 2 * values['Moon_9']['miou']) / 4
    primary['tta_component_metrics'] = {str(component_size): component}
    primary['additional_model_forward_calls'] = 0
    return primary


def publish_component(document, component_output, write_json):
    component = document['tta_component_metrics']['1024']
    # The frozen CLI adds checkpoint/selection provenance after evaluate_loader.
    for key, value in document.items():
        if (key.startswith(('checkpoint', 'official_', 'validation_', 'hard_'))
                or key in ('model_config', 'split', 'protocol', 'independent_holdout',
                           'caveat', 'score_used_for_ranking', 'test_scope', 'full_test_coverage')):
            component[key] = value
    component['seconds_shared_evaluation'] = document.get('seconds')
    extension = {'name': Path(__file__).name,
                 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'component_size': 1024, 'additional_model_forward_calls': 0,
                 'frozen_primary_evaluation_unchanged': True}
    component['evaluation_extension'] = extension
    document['evaluation_extension'] = extension
    document['tta_component_outputs'] = {'1024': str(component_output)}
    validate_component_document(document, component)
    write_json(component_output, component)


def validate_component_document(primary, component):
    import math
    import numpy as np
    from vitseg.metrics import ConfusionMatrix

    for key in ('checkpoint_sha256', 'checkpoint_epoch', 'model_config', 'data_identity',
                'image_count', 'partial', 'validation_unique_source_count', 'official_test_scope'):
        if component.get(key) != primary.get(key):
            raise ValueError('1024 component provenance differs: ' + key)
    if (component.get('tta') != 'hflip' or component.get('tta_mode') != 'hflip'
            or component.get('tta_views') != ['original', 'horizontal_flip']
            or component.get('tta_forward_calls_per_batch') != 2
            or component.get('additional_model_forward_calls') != 0
            or component.get('model_forward_calls') != 0 or component.get('loader_passes') != 0
            or primary.get('loader_passes') != 1
            or component.get('source_model_forward_calls') != primary.get('model_forward_calls')
            or component.get('reused_model_forward_calls', 0) * len(primary['tta_input_sizes'])
                != primary.get('model_forward_calls')):
        raise ValueError('1024 component did not reuse the original/flip multiscale forwards')
    if set(component['scenes']) != set(primary['scenes']):
        raise ValueError('1024 component scene coverage differs')

    def check_matrix(document, reference):
        matrix = np.asarray(document['confusion_matrix_gt_rows_pred_cols'])
        other = np.asarray(reference['confusion_matrix_gt_rows_pred_cols'])
        if (matrix.shape != (5, 5) or not np.issubdtype(matrix.dtype, np.integer)
                or (matrix < 0).any() or not np.array_equal(matrix.sum(1), other.sum(1))):
            raise ValueError('1024 component confusion matrix/ground-truth coverage differs')
        accumulator = ConfusionMatrix()
        accumulator.matrix = matrix
        expected = accumulator.compute()
        for key in ('miou', 'pixel_accuracy', 'class_iou', 'class_precision', 'class_recall',
                    'gt_pixels', 'predicted_pixels', 'gt_present_classes'):
            actual, wanted = document[key], expected[key]
            if isinstance(wanted, dict):
                if actual.keys() != wanted.keys():
                    raise ValueError('1024 component classes differ')
                values = [(actual[k], v) for k, v in wanted.items()]
            else:
                values = [(actual, wanted)]
            for actual_value, expected_value in values:
                if isinstance(expected_value, float):
                    matches = (isinstance(actual_value, (int, float))
                               and math.isclose(actual_value, expected_value, rel_tol=1e-10, abs_tol=1e-10))
                else:
                    matches = actual_value == expected_value
                if not matches:
                    raise ValueError('1024 component independently recomputed metric differs: ' + key)
        return matrix

    accumulated = np.zeros((5, 5), dtype=np.int64)
    for scene, entry in component['scenes'].items():
        reference = primary['scenes'][scene]
        if entry['image_count'] != reference['image_count']:
            raise ValueError('1024 component per-scene image count differs')
        accumulated += check_matrix(entry, reference)
    pooled = check_matrix(component, primary)
    if not np.array_equal(pooled, accumulated):
        raise ValueError('1024 component scene confusion matrices do not sum to the pooled matrix')
    if primary.get('weighted_score') is None:
        if component.get('weighted_score') is not None:
            raise ValueError('Partial component cannot report a full test score')
    else:
        scenes = component['scenes']
        expected_score = (scenes['Moon_2']['miou'] + scenes['Moon_4']['miou'] + 2 * scenes['Moon_9']['miou']) / 4
        if not math.isclose(component['weighted_score'], expected_score, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('1024 component weighted score differs from 1:1:2 scene aggregation')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        raise ValueError('Pass frozen evaluator arguments after --')
    boundary = argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-code', required=True, type=Path)
    parser.add_argument('--component-output', required=True, type=Path)
    args = parser.parse_args(argv[:boundary])
    evaluate_args = argv[boundary + 1:]
    runtime = args.runtime_code.resolve()
    if not (runtime / 'evaluate.py').is_file():
        raise ValueError('Missing frozen evaluate.py')
    if '--output' not in evaluate_args:
        raise ValueError('An explicit primary metrics output is required')
    primary_output = Path(evaluate_args[evaluate_args.index('--output') + 1]).resolve()
    if primary_output == args.component_output.resolve():
        raise ValueError('Component and fused metrics must have distinct output paths')
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('_frozen_evaluate_with_components', runtime / 'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_evaluate_loader, original_write_json = evaluator.evaluate_loader, evaluator.write_json

    def evaluate(model, loader, device, **kwargs):
        return evaluate_with_component(original_evaluate_loader, model, loader, device, **kwargs)

    def write(path, document):
        if 'tta_component_metrics' in document:
            publish_component(document, args.component_output, original_write_json)
        original_write_json(path, document)

    evaluator.evaluate_loader, evaluator.write_json = evaluate, write
    sys.argv = [str(runtime / 'evaluate.py'), *evaluate_args]
    evaluator.main()


if __name__ == '__main__':
    main()
