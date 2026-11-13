#!/usr/bin/env python3
"""One final 5004-image pass, reporting validation1500 and its complement3504.

This protocol deliberately selects checkpoints using 1500 former test images.
The complete score therefore includes validation; the complement shares scenes
and was involved in difficulty screening. Neither is an independent holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_checkpoint
from evaluate_hard_subset import (FULL_COUNTS, QUOTAS, SCORE_WEIGHTS, TTA,
    ScoreJournal, aggregate, canonical, identity, load_hard_manifest,
    validate_full_source, weighted_scene_score, write_json)
from vitseg.data import DEFAULT_TEST_MANIFEST, LunarDataset, sha256_file
from vitseg.metrics import evaluate_loader, tta_metadata

ROOT = Path(__file__).resolve().parents[1]
COMPLEMENT_COUNTS = {s: FULL_COUNTS[s] - QUOTAS[s] for s in FULL_COUNTS}
PROTOCOL = 'pool8002_train6000_hard1500_validation_v1'


def validate_checkpoint_protocol(saved, hard_sha):
    val = saved.get('data_identity', {}).get('val', {})
    training = saved.get('train_config', {})
    if (saved.get('format') != 'lusnar_vit_checkpoint_v1' or saved.get('partial') is not False
            or saved.get('data_identity', {}).get('train', {}).get('image_count') != 6000
            or val.get('image_count') != 1500 or val.get('unique_source_count') != 1500
            or val.get('hard_manifest_sha256') != hard_sha
            or training.get('protocol') != PROTOCOL
            or saved.get('epoch') != saved.get('best_epoch')):
        raise ValueError('Expected the validation-selected best from the explicit hard1500 training protocol')


def partition_rows(rows, full_records, hard_records):
    expected = [identity(r) for r in full_records]
    observed = [r['source_id'] for r in rows]
    val_ids = {identity(r) for r in hard_records}
    if (len(expected) != 5004 or len(set(expected)) != 5004
            or Counter(r['scene'] for r in full_records) != FULL_COUNTS
            or len(hard_records) != 1500 or len(val_ids) != 1500
            or Counter(r['scene'] for r in hard_records) != QUOTAS
            or not val_ids.issubset(expected) or observed != expected):
        raise ValueError('Final pass must cover the exact unique 5004 sequence and fixed 1500 subset')
    for row in rows:
        if row['source_id'] != f"{row['scene']}/{row['sample_id']}":
            raise ValueError('Per-image scene/sample identity changed')
        cm = np.asarray(row['confusion_matrix_gt_rows_pred_cols'])
        if cm.shape != (5, 5) or cm.dtype.kind not in 'iu' or np.any(cm < 0) or cm.sum() != 1024 * 1024:
            raise ValueError('Each image must contribute an integer native-1024 confusion matrix')
    validation = [r for r in rows if r['source_id'] in val_ids]
    complement = [r for r in rows if r['source_id'] not in val_ids]
    if Counter(r['scene'] for r in complement) != COMPLEMENT_COUNTS:
        raise ValueError('The complementary scene counts must be 1409/1133/962')
    reports = {'full5004': aggregate(rows), 'validation1500': aggregate(validation),
               'complement3504': aggregate(complement)}
    for scene in (None, *FULL_COUNTS):
        matrices = [np.asarray((m if scene is None else m['scenes'][scene])[
            'confusion_matrix_gt_rows_pred_cols']) for m in reports.values()]
        if not np.array_equal(matrices[0], matrices[1] + matrices[2]):
            raise ValueError('Full confusion matrix must equal validation plus complement, per scene and globally')
    common = {'partial': False, 'independent_holdout': False, 'official_full_test': False,
              'official_test_scope': False, 'checkpoint_selection_uses_test': True,
              'official_test_used_for_selection': True,
              'weighted_scene_score_weights': SCORE_WEIGHTS,
              'weighted_scene_score_formula': '(Moon_2 mIoU + Moon_4 mIoU + 2*Moon_9 mIoU)/4',
              'per_image_confusion_reconstruction_verified': True,
              'score_is_mean_of_image_ious': False}
    descriptions = {
        'full5004': ('full5004_including_selection_validation', 1500,
            'All 5004 former test images, including 1500 used to select the checkpoint; not an independent test score.'),
        'validation1500': ('hard1500_selection_validation', 1500,
            'Fixed model-dependent difficult subset used by this experiment for validation and checkpoint selection.'),
        'complement3504': ('complement3504_after_hard_validation_selection', 0,
            'Excludes the 1500 validation images, but shares their scenes and participated in prior model-based difficulty screening; not a clean independent holdout.')}
    for name, m in reports.items():
        scope, included, caveat = descriptions[name]
        m.update(common, test_scope=scope, validation_images_included=included, caveat=caveat,
                 scene_miou={s: d['miou'] for s, d in m['scenes'].items()},
                 weighted_scene_score=weighted_scene_score(m), full_test_coverage=name == 'full5004')
        # Only the complete scene score carries the historical formula's name.
        if name == 'full5004':
            m['weighted_score'] = m['weighted_scene_score']
        elif name == 'complement3504':
            m['complement_score'] = m['weighted_scene_score']
    return reports


def validate_bundle(directory, expected_checkpoint_sha=None, expected_hard_sha=None):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    if (summary.get('schema') != 'lusnar_final_partition_report_v1' or summary.get('status') != 'complete'
            or summary.get('independent_holdout') is not False or summary.get('image_count') != 5004
            or (expected_checkpoint_sha and summary['checkpoint_sha256'] != expected_checkpoint_sha)
            or (expected_hard_sha and summary['hard_manifest_sha256'] != expected_hard_sha)):
        raise ValueError('Final report scope/checkpoint/subset provenance differs')
    for name, digest in summary['artifact_sha256'].items():
        path = directory / name
        if path.parent != directory or sha256_file(path) != digest:
            raise ValueError('Final report artifact is missing or changed: ' + name)
    full_path = Path(summary['full_test_manifest'])
    hard_path = Path(summary['hard_manifest'])
    if (sha256_file(full_path) != summary['full_test_manifest_sha256']
            or sha256_file(hard_path) != summary['hard_manifest_sha256']):
        raise ValueError('Final evaluation source manifest changed')
    full = validate_full_source(full_path)
    _, hard = load_hard_manifest(hard_path, full_path.resolve(), full)
    rows = [json.loads(line) for line in (directory / 'per_image.jsonl').read_text().splitlines()]
    reports = partition_rows(rows, full, hard)
    for name, recomputed in reports.items():
        stored = json.loads((directory / (name + '_metrics.json')).read_text())
        for key, value in recomputed.items():
            if stored.get(key) != value:
                raise ValueError('Reconstructed report differs: ' + name + '/' + key)
        if (stored['checkpoint_sha256'] != summary['checkpoint_sha256']
                or stored['hard_manifest_sha256'] != summary['hard_manifest_sha256']):
            raise ValueError('Report provenance differs from the completed bundle')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hard-manifest', type=Path, required=True)
    parser.add_argument('--full-test-manifest', type=Path, default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument('--output', type=Path, required=True, help='Dedicated final-report directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'none'], default='bf16')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid batch size or worker count')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha, hard_sha = sha256_file(args.checkpoint), sha256_file(args.hard_manifest)
    if (output / 'summary.json').exists():
        previous = validate_bundle(output, checkpoint_sha, hard_sha)
        if (previous['full_test_manifest_sha256'] != sha256_file(args.full_test_manifest)
                or previous['amp'] != args.amp or previous['inference_batch_size'] != args.batch_size):
            raise ValueError('Completed report exists with different inference/source settings')
        print(json.dumps({**previous, 'status': 'complete_reused'}), flush=True)
        return
    full_path, hard_path = args.full_test_manifest.resolve(), args.hard_manifest.resolve()
    records = validate_full_source(full_path)
    _, hard = load_hard_manifest(hard_path, full_path, records)
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    model, saved, model_config = load_checkpoint(args.checkpoint, device)
    validate_checkpoint_protocol(saved, hard_sha)
    if model_config.input_size != 1024:
        raise ValueError('This final protocol requires native 1024 training input')
    provenance = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_epoch': saved['epoch'], 'checkpoint_partial_training': False,
        'hard_manifest': str(hard_path), 'hard_manifest_sha256': hard_sha,
        'full_test_manifest': str(full_path), 'full_test_manifest_sha256': sha256_file(full_path),
        'protocol': PROTOCOL, 'model_config': saved['model_config'],
        'training_data_identity': saved['data_identity'], 'amp': args.amp,
        'checkpoint_selection_metric': 'validation_weighted_score',
        'checkpoint_selected_validation_score': saved['best_miou'],
        'inference_batch_size': args.batch_size, **TTA, **tta_metadata(**TTA)}
    files = [Path(__file__), ROOT / 'code/evaluate.py', ROOT / 'code/evaluate_hard_subset.py',
             *sorted((ROOT / 'code/vitseg').glob('*.py'))]
    run_identity = {**provenance, 'expected_source_ids_sha256': canonical([identity(r) for r in records]),
        'expected_image_count': 5004, 'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in files}}
    journal = ScoreJournal(output / 'per_image.jsonl', [identity(r) for r in records], run_identity, args.batch_size)
    start, begin = len(journal.rows), time.monotonic()
    try:
        if start < 5004:
            dataset = LunarDataset(full_path, 'test', 1024)
            if [r['source_id'] for r in dataset.records] != [identity(r) for r in records]:
                raise ValueError('Dataset order differs from the frozen full manifest')
            dataset.records = dataset.records[start:]
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == 'cuda')
            engine = evaluate_loader(model, loader, device, amp=args.amp, **TTA,
                                     image_exporter=journal, progress_every=25)
            tail = aggregate(journal.rows[start:])
            for scene in (None, *tail['scenes']):
                a = engine if scene is None else engine['scenes'][scene]
                b = tail if scene is None else tail['scenes'][scene]
                if a['confusion_matrix_gt_rows_pred_cols'] != b['confusion_matrix_gt_rows_pred_cols']:
                    raise ValueError('Per-image journal disagrees with the inference accumulator')
        if (sha256_file(args.checkpoint) != checkpoint_sha or sha256_file(hard_path) != hard_sha
                or sha256_file(full_path) != provenance['full_test_manifest_sha256']):
            raise ValueError('Checkpoint or evaluation manifest changed during inference')
        reports = partition_rows(journal.rows, records, hard)
        for name, report in reports.items():
            write_json(output / (name + '_metrics.json'), {**report, **provenance})
        summary = {**provenance, 'schema': 'lusnar_final_partition_report_v1', 'status': 'complete',
            'image_count': 5004, 'validation_images': 1500, 'complement_images': 3504,
            'independent_holdout': False, 'checkpoint_selection_uses_test': True,
            'single_full_inference_pass': True, 'resumed_images': start, 'seconds': time.monotonic() - begin,
            'partition_confusion_additivity_verified': True,
            'weighted_full5004_score': reports['full5004']['weighted_scene_score'],
            'weighted_complement3504_score': reports['complement3504']['weighted_scene_score'],
            'score_milestone_92_enabled': False,
            'artifact_sha256': {name: sha256_file(output / name) for name in
                ('full5004_metrics.json', 'validation1500_metrics.json', 'complement3504_metrics.json',
                 'per_image.jsonl', 'per_image.identity.json')}}
        write_json(output / 'summary.json', summary)
        validate_bundle(output, checkpoint_sha, hard_sha)
        print(json.dumps({'status': 'complete', 'output': str(output),
            'full5004_score': summary['weighted_full5004_score'],
            'complement3504_score': summary['weighted_complement3504_score']}), flush=True)
    finally:
        journal.close()


if __name__ == '__main__':
    main()
