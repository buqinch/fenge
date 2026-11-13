#!/usr/bin/env python3
"""Freeze a worst-image test subset once, then evaluate that same subset.

Construction uses one frozen checkpoint on all 5004 official test images. This
is deliberately a difficulty-selected diagnostic test set, never a validation
set or an official complete-test score. RGB/data splits are unaffected.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from evaluate import load_checkpoint
from vitseg.data import LunarDataset, DEFAULT_TEST_MANIFEST, sha256_file
from vitseg.metrics import ConfusionMatrix, evaluate_loader, tta_metadata

SCHEMA = 'lusnar_fixed_hard_test_v1'
SCENES = ('Moon_2', 'Moon_4', 'Moon_9')
FULL_COUNTS = {'Moon_2': 1597, 'Moon_4': 1320, 'Moon_9': 2087}
QUOTAS = {'Moon_2': 188, 'Moon_4': 187, 'Moon_9': 1125}
SCORE_WEIGHTS = {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}
TTA = {'tta': 'multiscale_hflip', 'tta_sizes': [1024, 768], 'tta_weights': [.5, .5]}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def identity(row):
    return f"{row['scene']}/{row['sample_id']}"


def validate_full_source(path):
    source = json.loads(Path(path).read_text())
    records = source['records']
    if (len(records) != 5004 or Counter(r['scene'] for r in records) != FULL_COUNTS
            or len({identity(r) for r in records}) != 5004):
        raise ValueError('Full test manifest must contain the fixed unique 1597/1320/2087 images')
    return records


def load_hard_manifest(path, full_path, full_records):
    document = json.loads(Path(path).read_text())
    if document.get('schema') != SCHEMA or document.get('selection_quotas') != QUOTAS:
        raise ValueError('Unknown hard-test manifest schema or changed sample quotas')
    if document.get('full_test_manifest_sha256') != sha256_file(full_path):
        raise ValueError('Full test source changed since hard-subset construction')
    if Path(document['full_test_manifest']).resolve() != Path(full_path).resolve():
        raise ValueError('Full test manifest path changed')
    records = document.get('records', [])
    if (len(records) != 1500 or Counter(r['scene'] for r in records) != QUOTAS
            or len({identity(r) for r in records}) != 1500):
        raise ValueError('Hard-test identities/counts differ from the fixed 1500-image protocol')
    source = {identity(r): r for r in full_records}
    if any(r != source.get(identity(r)) for r in records):
        raise ValueError('Hard-test row is not a literal record from the full test source')
    if canonical(records) != document.get('ordered_records_sha256'):
        raise ValueError('Hard-test ordered records SHA256 mismatch')
    if document.get('hard_subset_score_weights') != SCORE_WEIGHTS:
        raise ValueError('Hard-test scoring weights changed')
    if document.get('tta') != TTA:
        raise ValueError('Hard-test TTA differs from the fixed 1024/768 equal horizontal flip')
    return document, records


def matrix_for(row):
    cm = np.asarray(row['confusion_matrix_gt_rows_pred_cols'])
    if cm.shape != (5, 5) or cm.dtype.kind not in 'iu' or np.any(cm < 0):
        raise ValueError('Invalid per-image integer confusion matrix')
    result = ConfusionMatrix(); result.matrix = cm.astype(np.int64)
    metrics = result.compute()
    if metrics['miou'] is None or abs(metrics['miou'] - row['miou']) > 1e-12:
        raise ValueError('Per-image mIoU disagrees with its integer confusion matrix')
    if any(metrics['class_iou'][k] != row['class_iou'][k] for k in metrics['class_iou']):
        raise ValueError('Per-image class IoU disagrees with its confusion matrix')
    return result


def aggregate(rows):
    total = ConfusionMatrix(); scenes = {}
    for row in rows:
        cm = matrix_for(row)
        total.matrix += cm.matrix
        entry = scenes.setdefault(row['scene'], {'matrix': ConfusionMatrix(), 'image_count': 0})
        entry['matrix'].matrix += cm.matrix; entry['image_count'] += 1
    return {**total.compute(), 'image_count': len(rows),
            'scenes': {s: {**entry['matrix'].compute(), 'image_count': entry['image_count']}
                       for s, entry in sorted(scenes.items())}}


def weighted_scene_score(metrics):
    if set(metrics['scenes']) != set(SCENES):
        return None
    return sum(SCORE_WEIGHTS[s] * metrics['scenes'][s]['miou'] for s in SCENES) / 4


def select_worst(scores, quotas=QUOTAS):
    if len({r['source_id'] for r in scores}) != len(scores):
        raise ValueError('Duplicate source IDs in ranking')
    selected = []
    for scene in SCENES:
        group = [r for r in scores if r['scene'] == scene]
        if len(group) < quotas[scene]:
            raise ValueError(f'Not enough ranked images for {scene}')
        for rank, row in enumerate(sorted(group, key=lambda r: (r['miou'], r['source_id']))[:quotas[scene]], 1):
            selected.append({**row, 'within_scene_difficulty_rank': rank})
    return selected


class ScoreJournal:
    """Append durable full batches; recover only a complete, identity-checked prefix."""
    def __init__(self, path, expected_ids, run_identity, batch_size):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.expected_ids = expected_ids; self.key = canonical(run_identity); self.batch_size = batch_size
        metadata = self.path.with_suffix('.identity.json')
        if metadata.exists():
            if json.loads(metadata.read_text()) != run_identity:
                raise ValueError('Cannot mix checkpoints, source manifests or inference recipes in the score journal')
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError('Score journal exists without its immutable identity')
        else:
            write_json(metadata, run_identity)
        self.rows = []
        if self.path.exists():
            raw = self.path.read_bytes(); end = raw.rfind(b'\n') + 1
            for line in raw[:end].splitlines():
                row = json.loads(line)
                if row.get('run_identity_sha256') != self.key:
                    raise ValueError('Score journal contains another inference identity')
                if len(self.rows) >= len(expected_ids) or row['source_id'] != expected_ids[len(self.rows)]:
                    raise ValueError('Score journal is not the unique expected sample prefix')
                matrix_for(row); self.rows.append(row)
            # If interrupted during a batch write, replay that batch with identical batch boundaries.
            complete = len(self.rows) if len(self.rows) == len(expected_ids) else len(self.rows) // batch_size * batch_size
            self.rows = self.rows[:complete]
            rewritten = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in self.rows)
            if raw != rewritten.encode():
                temp = self.path.with_suffix('.recover.tmp'); temp.write_text(rewritten); temp.replace(self.path)
        self.handle = self.path.open('a')

    def update(self, batch, targets, prediction, probabilities):
        pending = []
        for i, source_id in enumerate(batch['source_id']):
            position = len(self.rows) + len(pending)
            if source_id != self.expected_ids[position]:
                raise ValueError('Inference order differs from the immutable journal')
            if tuple(targets[i].shape) != (1024, 1024):
                raise ValueError('Hard-test ranking requires native 1024x1024 masks')
            cm = ConfusionMatrix(); cm.update(targets[i], prediction[i]); m = cm.compute()
            pending.append({'source_id': source_id, 'scene': str(batch['scene'][i]),
                            'sample_id': str(batch['sample_id'][i]), 'miou': m['miou'],
                            'class_iou': m['class_iou'], 'gt_present_classes': m['gt_present_classes'],
                            'gt_pixels': m['gt_pixels'], 'confusion_matrix_gt_rows_pred_cols': cm.matrix.tolist(),
                            'run_identity_sha256': self.key})
        self.handle.write(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in pending))
        self.handle.flush(); os.fsync(self.handle.fileno()); self.rows.extend(pending)

    def close(self):
        self.handle.close()


def subset_metrics(rows, provenance, expected_count=1500):
    metrics = aggregate(rows)
    complete = len(rows) == expected_count and Counter(r['scene'] for r in rows) == QUOTAS
    metrics.update(provenance, **tta_metadata(**TTA), test_scope='fixed_hard_subset', official_full_test=False,
                   official_test_scope=False, partial=not complete, hard_subset_complete=complete,
                   expected_image_count=1500, scene_miou={s: m['miou'] for s, m in metrics['scenes'].items()},
                   hard_subset_score=weighted_scene_score(metrics) if complete else None,
                   hard_subset_score_weights=SCORE_WEIGHTS,
                   hard_subset_score_formula='(Moon_2 subset mIoU + Moon_4 subset mIoU + 2*Moon_9 subset mIoU)/4',
                   selection_quotas=QUOTAS,
                   score_used_for_ranking='hard_subset_score is a diagnostic trend only; never official full-test ranking',
                   caveat='The fixed 1500 images were selected as worst per-image mIoU for one source model. This is a deliberately biased diagnostic subset, not validation or complete test.')
    return metrics


def self_test():
    def row(s, sid, correct, errors):
        cm = ConfusionMatrix(); cm.matrix[0,0] = correct; cm.matrix[0,2] = errors
        m=cm.compute()
        return {'source_id': f'{s}/{sid}', 'scene':s, 'sample_id':sid, 'miou':m['miou'],
                'class_iou':m['class_iou'],'confusion_matrix_gt_rows_pred_cols':cm.matrix.tolist()}
    rows=[row(s, sid, correct,10-correct) for s in SCENES for sid,correct in [('b',2),('a',2),('c',9)]]
    chosen=select_worst(rows,{s:1 for s in SCENES})
    assert [r['sample_id'] for r in chosen]==['a','a','a']
    assert all(r['class_iou']['crater'] is None for r in rows)  # GT-absent classes do not enter image mIoU.
    a=aggregate(rows);assert abs(a['miou']-13/30)<1e-12
    assert abs(weighted_scene_score(a)-13/30)<1e-12
    try:select_worst(rows+rows[:1],{s:1 for s in SCENES})
    except ValueError:pass
    else:raise AssertionError('Duplicate source identities were accepted')
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'scores.jsonl'; run={'checkpoint_sha256':'abc','test':'journal'}
        j=ScoreJournal(p,[r['source_id'] for r in rows],run,2);j.close()
        saved=[{**r,'run_identity_sha256':canonical(run)} for r in rows[:3]]
        p.write_text(''.join(json.dumps(r)+'\n' for r in saved)+'{"partial":')
        j=ScoreJournal(p,[r['source_id'] for r in rows],run,2)
        assert len(j.rows)==2;j.close()
        try:ScoreJournal(p,[r['source_id'] for r in rows],{'checkpoint_sha256':'other'},2)
        except ValueError:pass
        else:raise AssertionError('Mixed checkpoints were accepted')
    with tempfile.TemporaryDirectory() as t:
        full_path=Path(t)/'full.json'; hard_path=Path(t)/'hard.json'
        full_rows=[{'scene':scene,'sample_id':str(i),'image':f'{scene}/{i}.png'}
                   for scene,count in FULL_COUNTS.items() for i in range(count)]
        write_json(full_path,{'records':full_rows})
        selected=[r for scene in SCENES for r in [x for x in full_rows if x['scene']==scene][:QUOTAS[scene]]]
        doc={'schema':SCHEMA,'selection_quotas':QUOTAS,'records':selected,
             'full_test_manifest':str(full_path),'full_test_manifest_sha256':sha256_file(full_path),
             'ordered_records_sha256':canonical(selected),'hard_subset_score_weights':SCORE_WEIGHTS,'tta':TTA}
        write_json(hard_path,doc)
        assert len(load_hard_manifest(hard_path,full_path,validate_full_source(full_path))[1])==1500
        damaged=json.loads(json.dumps(doc));damaged['records'][0]['image']='another.png'
        damaged['ordered_records_sha256']=canonical(damaged['records']);write_json(hard_path,damaged)
        try:load_hard_manifest(hard_path,full_path,full_rows)
        except ValueError:pass
        else:raise AssertionError('Modified source record was accepted')
        damaged=json.loads(json.dumps(doc));damaged['records'][0]=damaged['records'][1]
        damaged['ordered_records_sha256']=canonical(damaged['records']);write_json(hard_path,damaged)
        try:load_hard_manifest(hard_path,full_path,full_rows)
        except ValueError:pass
        else:raise AssertionError('Duplicate subset record was accepted')
    print(json.dumps({'self_test':'passed','checks':['GT-present image IoU','stable hardest ranking','duplicate rejection','scene score reconstruction','crash-prefix recovery','mixed-checkpoint rejection','fixed 1500 subset integrity','modified/duplicate manifest record rejection']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--build-hard-manifest',type=Path)
    modes.add_argument('--hard-manifest',type=Path)
    parser.add_argument('--full-test-manifest',type=Path,default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--per-image-jsonl',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--amp',choices=['none','bf16','fp16'],default='bf16')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.checkpoint or not(args.build_hard_manifest or args.hard_manifest) or not args.output:
        parser.error('--checkpoint, --output and one manifest mode are required')
    if args.batch_size<1 or args.workers<0 or (args.max_samples is not None and args.max_samples<1):
        parser.error('Invalid batch/worker/sample count')
    full_path=args.full_test_manifest.resolve(); full_records=validate_full_source(full_path)
    if args.hard_manifest:
        hard_doc, records=load_hard_manifest(args.hard_manifest,full_path,full_records)
    else:
        hard_doc=None;records=full_records
    if args.max_samples is not None:records=records[:args.max_samples]
    expected_ids=[identity(r) for r in records]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha256_file(args.checkpoint)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    if args.build_hard_manifest and args.build_hard_manifest.exists():
        existing=json.loads(args.build_hard_manifest.read_text())
        if existing.get('checkpoint_sha256')!=checkpoint_sha:
            raise FileExistsError('Immutable hard-test subset already belongs to another source checkpoint')
    model, saved, config=load_checkpoint(args.checkpoint,device)
    if bool(saved.get('partial',False)):raise ValueError('A partial-training checkpoint cannot define/score the hard subset')
    if config.input_size!=1024:raise ValueError('Hard subset requires the fixed 1024 training input protocol')
    provenance={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha,
                'checkpoint_epoch':saved['epoch'],'checkpoint_partial_training':False,'split':'test',
                'model_config':saved['model_config'],'full_test_manifest':str(full_path),
                'full_test_manifest_sha256':sha256_file(full_path),
                'subset_manifest_sha256':sha256_file(args.hard_manifest) if args.hard_manifest else None,
                'tta':TTA['tta'],'tta_sizes':TTA['tta_sizes'],'tta_weights':TTA['tta_weights']}
    dependency_files=[Path(__file__),Path(__file__).with_name('evaluate.py'),*sorted((Path(__file__).parent/'vitseg').glob('*.py'))]
    run_identity={**provenance,'amp':args.amp,'batch_size':args.batch_size,'expected_source_ids_sha256':canonical(expected_ids),
                  'expected_image_count':len(records),'code_sha256':{str(p.relative_to(Path(__file__).parent)):sha256_file(p) for p in dependency_files}}
    journal=ScoreJournal(args.per_image_jsonl or args.output.with_name(args.output.stem+'_per_image.jsonl'),expected_ids,run_identity,args.batch_size)
    begin=time.monotonic();start=len(journal.rows)
    print(json.dumps({'status':'starting','checkpoint_epoch':saved['epoch'],'checkpoint_sha256':checkpoint_sha,
                      'image_count':len(records),'resumed_images':start,'device':args.device,'mode':'build' if args.build_hard_manifest else 'fixed_subset'}),flush=True)
    try:
        if start<len(records):
            dataset=LunarDataset(full_path,'test',1024,horizontal_flip=False)
            mapping={r['source_id']:r for r in dataset.records}
            dataset.records=[mapping[s] for s in expected_ids[start:]]
            dataset.partial=False
            dataset.identity={**dataset.identity,'image_count':len(dataset.records),'partial':False,
                              'hard_subset_manifest_sha256':provenance['subset_manifest_sha256'],
                              'scoring_remaining_prefix_offset':start}
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                              pin_memory=str(args.device).startswith('cuda'))
            engine=evaluate_loader(model,loader,args.device,amp=args.amp,**TTA,image_exporter=journal,progress_every=25)
            reconstructed=aggregate(journal.rows[start:])
            if reconstructed['confusion_matrix_gt_rows_pred_cols']!=engine['confusion_matrix_gt_rows_pred_cols']:
                raise RuntimeError('Per-image accumulated CM differs from evaluator CM')
            for s,value in reconstructed['scenes'].items():
                if value['confusion_matrix_gt_rows_pred_cols']!=engine['scenes'][s]['confusion_matrix_gt_rows_pred_cols']:
                    raise RuntimeError(f'Per-image scene CM differs from evaluator: {s}')
        if len(journal.rows)!=len(records):raise RuntimeError('Incomplete image score stream')
        if sha256_file(args.checkpoint)!=checkpoint_sha:raise RuntimeError('Checkpoint changed during scoring')
        if sha256_file(full_path)!=provenance['full_test_manifest_sha256']:
            raise RuntimeError('Full test manifest changed during scoring')
        if args.hard_manifest and sha256_file(args.hard_manifest)!=provenance['subset_manifest_sha256']:
            raise RuntimeError('Fixed hard subset changed during scoring')
        if args.hard_manifest:
            metrics=subset_metrics(journal.rows,provenance)
        else:
            metrics=aggregate(journal.rows)
            full=Counter(r['scene'] for r in journal.rows)==FULL_COUNTS and len(journal.rows)==5004
            metrics.update(provenance,**tta_metadata(**TTA),test_scope='hard_subset_construction_full_scan',
                           official_full_test=full,official_test_scope=full,partial=not full,
                           weighted_score=weighted_scene_score(metrics) if full else None,
                           scene_miou={s:m['miou'] for s,m in metrics['scenes'].items()},
                           note='One frozen source model, complete test pass solely to construct the fixed hardest-image diagnostic subset')
        metrics.update(seconds=time.monotonic()-begin,resumed_images=start,per_image_jsonl=str(journal.path.resolve()),
                       per_image_jsonl_sha256=sha256_file(journal.path),per_image_confusion_reconstruction_verified=True)
        write_json(args.output,metrics)
        if args.build_hard_manifest and not metrics['partial']:
            selected=select_worst(journal.rows);full_by_id={identity(r):r for r in full_records}
            selected_records=[full_by_id[r['source_id']] for r in selected]
            document={'schema':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'full_test_manifest':str(full_path),'full_test_manifest_sha256':sha256_file(full_path),
                'full_test_record_count':5004,'full_test_ordered_records_sha256':canonical(full_records),
                'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha,'checkpoint_epoch':saved['epoch'],
                'selection_method':'Within each scene sort ascending per-image GT-present-class mIoU, source_id ascending resolves ties; fixed once',
                'selection_quotas':QUOTAS,'requested_image_ratio':{'Moon_2':1,'Moon_4':1,'Moon_9':6},
                'quota_rounding':'1500*(1,1,6)/8=(187.5,187.5,1125); extra tied image goes to Moon_2',
                'image_count':1500,'records':selected_records,'ordered_records_sha256':canonical(selected_records),
                'tta':TTA,'amp':args.amp,'hard_subset_score_weights':SCORE_WEIGHTS,
                'source_per_image_jsonl':str(journal.path.resolve()),'source_per_image_jsonl_sha256':sha256_file(journal.path),
                'source_full_metrics':str(args.output.resolve()),'source_full_metrics_sha256':sha256_file(args.output),
                'selected_image_scores':[{'source_id':r['source_id'],'scene':r['scene'],'source_model_image_miou':r['miou'],
                                          'within_scene_difficulty_rank':r['within_scene_difficulty_rank']} for r in selected],
                'test_labels_used_for_difficulty_selection':True,'validation_or_training_use':False,
                'official_full_test':False,'selection_is_model_dependent':True,'never_resample_per_epoch':True}
            if args.build_hard_manifest.exists():
                old=json.loads(args.build_hard_manifest.read_text())
                if (old['ordered_records_sha256']!=document['ordered_records_sha256']
                        or old['checkpoint_sha256']!=checkpoint_sha):
                    raise FileExistsError('Refusing to overwrite a different immutable hard-test manifest')
            else:write_json(args.build_hard_manifest,document)
            _,checked=load_hard_manifest(args.build_hard_manifest,full_path,full_records)
            assert len(checked)==1500
            source_subset=subset_metrics(selected,{**provenance,'subset_manifest_sha256':sha256_file(args.build_hard_manifest)})
            source_subset.update(derived_from_same_complete_scoring_pass=True,per_image_confusion_reconstruction_verified=True)
            write_json(args.build_hard_manifest.parent/'source_hard_subset_metrics.json',source_subset)
        print(json.dumps({'status':'complete','output':str(args.output),'image_count':metrics['image_count'],
                          'partial':metrics['partial'],'hard_subset_score':metrics.get('hard_subset_score'),
                          'build_manifest':str(args.build_hard_manifest) if args.build_hard_manifest and not metrics['partial'] else None}),flush=True)
    finally:
        journal.close()


if __name__=='__main__':main()
