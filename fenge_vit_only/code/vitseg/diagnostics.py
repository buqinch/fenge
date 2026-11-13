"""GT-defined component recall; these conditional recalls are not IoU."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy import ndimage


class TerrainDiagnostics:
    def __init__(self, large_crater_area=16384, small_rock_area=256, erosion_pixels=5):
        if min(large_crater_area, small_rock_area, erosion_pixels) < 1:
            raise ValueError('Component area thresholds and erosion must be positive')
        self.large_crater_area = large_crater_area
        self.small_rock_area = small_rock_area
        self.erosion_pixels = erosion_pixels
        self.counts = defaultdict(lambda: defaultdict(lambda: {
            'components': 0, 'edge_touching_components': 0, 'gt_pixels': 0,
            'true_positive_pixels': 0, 'predicted_class_counts': [0] * 5,
            'sum_component_recall': 0.0, 'empty_after_erosion_components': 0}))

    def _update_image(self, scene, target, prediction):
        for label, name, threshold in [(2, 'large_crater_interior', self.large_crater_area),
                                        (1, 'small_rock', self.small_rock_area)]:
            components, count = ndimage.label(target == label, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(components.ravel(), minlength=count + 1)
            objects = ndimage.find_objects(components)
            for index, box in enumerate(objects, 1):
                if box is None or (label == 2 and sizes[index] < threshold) or (label == 1 and sizes[index] > threshold):
                    continue
                component = components[box] == index
                touching = (box[0].start == 0 or box[1].start == 0 or
                            box[0].stop == target.shape[0] or box[1].stop == target.shape[1])
                region = (ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool),
                           iterations=self.erosion_pixels, border_value=0) if label == 2 else component)
                for key in ('all', scene):
                    row = self.counts[key][name]
                    row['components'] += 1
                    row['edge_touching_components'] += int(touching)
                    if not region.any():
                        row['empty_after_erosion_components'] += 1
                        continue
                    histogram = np.bincount(prediction[box][region], minlength=5)
                    pixels, correct = int(region.sum()), int(histogram[label])
                    row['gt_pixels'] += pixels
                    row['true_positive_pixels'] += correct
                    row['sum_component_recall'] += correct / pixels
                    row['predicted_class_counts'] = [a + int(b) for a, b in zip(row['predicted_class_counts'], histogram)]

    def update(self, batch, targets, prediction, probabilities=None):
        truth, predicted = targets.detach().cpu().numpy(), prediction.detach().cpu().numpy()
        for scene, target, result in zip(batch['scene'], truth, predicted):
            self._update_image(str(scene), target, result)

    def compute(self):
        result = {}
        for scene, groups in self.counts.items():
            result[scene] = {}
            for name, row in groups.items():
                count = row['components'] - row['empty_after_erosion_components']
                result[scene][name] = {**row,
                    'pixel_recall': row['true_positive_pixels'] / row['gt_pixels'] if row['gt_pixels'] else None,
                    'mean_component_recall': row['sum_component_recall'] / count if count else None}
        return {'definition': 'Conditional GT-pixel recall, not IoU; 8-connected label components are not physical object instances.',
                'large_crater_min_gt_pixels': self.large_crater_area,
                'large_crater_interior_erosion_pixels': self.erosion_pixels,
                'small_rock_max_gt_pixels': self.small_rock_area,
                'coordinate_frame': 'native 1024 evaluation label grid; do not compare different target resolutions',
                'threshold_selection': 'fixed diagnostic definitions, never selected on test scores',
                'scenes': result}
