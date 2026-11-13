"""Opt-in sampled original pool / former-test hard validation data protocol.

Training uses pre-existing ID masks. Validation retains the original test ZIP
palette decoder, with explicit provenance that test data now selects models.
Existing train/evaluate/data modules remain unchanged.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from vitseg.data import (CLASS_NAMES, TRAIN_SCENES, VAL_SCENES, TEST_SCENES,
                        LunarDataset, _canonical_sha256, sha256_file)

SCHEMA = 'lusnar_all_sources_hardval_v1'
PROTOCOL = 'sampled_original_pool_with_fixed_hard_test_validation'
HARD_COUNTS = {'Moon_2': 188, 'Moon_4': 187, 'Moon_9': 1125}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_path(root, value):
    require(isinstance(value, str) and value, 'Missing sample path')
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def provenance(source, key):
    value = source.get(key + '_manifest')
    require(isinstance(value, str) and Path(value).is_absolute(), f'{key} manifest path must be absolute')
    path = Path(value).resolve(); fingerprint = sha256_file(path)
    require(source.get(key + '_manifest_sha256') == fingerprint, f'{key} manifest SHA mismatch')
    return path, fingerprint, json.loads(path.read_text())


class HardValidationDataset(LunarDataset):
    def __init__(self, manifest, split, input_size=1024, horizontal_flip=False,
                 max_samples=None, selection_manifest=None):
        require(split in ('train', 'val'), 'Hard validation dataset supports only train/val')
        require(selection_manifest is None, 'Training selection is embedded in the fixed protocol manifest')
        require(type(input_size) is int and input_size >= 32 and input_size % 32 == 0, 'Invalid input size')
        require(type(horizontal_flip) is bool and (split == 'train' or not horizontal_flip), 'Validation cannot be augmented')
        require(max_samples is None or (type(max_samples) is int and max_samples > 0), 'Invalid max_samples')
        self.manifest = Path(manifest).resolve(); self.split = split; self.input_size = input_size
        self.horizontal_flip = horizontal_flip; self._archives = {}
        fingerprint = sha256_file(self.manifest); source = json.loads(self.manifest.read_text())
        require(source.get('schema') == SCHEMA and source.get('split_protocol') == PROTOCOL, 'Unexpected hard validation protocol')
        require(source.get('class_names') == list(CLASS_NAMES), 'Class order changed')
        require(type(source.get('seed')) is int and source['seed'] >= 0, 'Invalid split seed')
        families, splits = source.get('families'), source.get('splits')
        require(isinstance(families, dict) and set(families) == {'train_pool_source_ids', 'val_source_ids', 'remaining_test_source_ids'}, 'Unexpected source-family partitions')
        require(all(isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) == len(set(v)) for v in families.values()), 'Duplicate or invalid source-family identities')
        require(source.get('families_sha256') == _canonical_sha256(families), 'Family SHA mismatch')
        require(isinstance(splits, dict) and set(splits) == {'train', 'val'} and all(isinstance(x, list) and x for x in splits.values()), 'Missing literal split rows')
        require(source.get('splits_sha256') == _canonical_sha256(splits), 'Split SHA mismatch')
        ordered = source.get('ordered_samples_sha256')
        require(isinstance(ordered, dict) and set(ordered) == {'train', 'val'} and all(ordered[k] == _canonical_sha256(v) for k,v in splits.items()), 'Ordered sample SHA mismatch')
        source_path, source_sha, original_doc = provenance(source, 'source')
        aug_path, aug_sha, augmented_doc = provenance(source, 'augmented')
        test_path, test_sha, test_doc = provenance(source, 'test')
        hard_path, hard_sha, hard_doc = provenance(source, 'hard')
        require(original_doc.get('class_names') == list(CLASS_NAMES) and augmented_doc.get('class_names') == list(CLASS_NAMES), 'Provenance class order changed')
        require(Path(augmented_doc.get('source_manifest', '')).resolve() == source_path and augmented_doc.get('source_manifest_sha256') == source_sha, 'Augmentation provenance mismatch')
        require(augmented_doc.get('validation_or_test_used') is False, 'Existing augmentation source manifest was changed')
        require(hard_doc.get('schema') == 'lusnar_fixed_hard_test_v1' and hard_doc.get('image_count') == 1500, 'Unexpected fixed hard manifest')
        require(Path(hard_doc.get('full_test_manifest', '')).resolve() == test_path and hard_doc.get('full_test_manifest_sha256') == test_sha, 'Fixed hard/full test provenance mismatch')
        require(hard_doc.get('ordered_records_sha256') == _canonical_sha256(hard_doc['records']), 'Hard validation record SHA mismatch')
        require(hard_doc.get('full_test_ordered_records_sha256') == _canonical_sha256(test_doc['records']), 'Full test record SHA mismatch')
        originals, augments, full_test = {}, {}, {}
        image_paths, mask_paths = set(), set()
        for row in original_doc['samples']:
            scene, sid = row['scene'], str(row['sample_id']); source_id = f'{scene}/{sid}'
            expected = 'train' if scene in TRAIN_SCENES else 'val' if scene in VAL_SCENES else None
            require(expected is not None and row['split'] == expected and source_id not in originals, 'Original pool identity/split violation')
            image, mask = (canonical_path(source_path.parent, row[k]) for k in ('image','mask'))
            require(image != mask and image not in image_paths and image not in mask_paths and mask not in image_paths and mask not in mask_paths, 'Duplicate original paths')
            originals[source_id] = {'scene':scene,'sample_id':sid,'source_id':source_id,'source_kind':'original','image':image,'mask':mask,'split':'train'}
            image_paths.add(image); mask_paths.add(mask)
        require(len(originals) == 8002, 'Complete 8002-image original pool is required')
        for row in augmented_doc['samples']:
            scene, sid = row['source_scene'], str(row['sample_id']); source_id = f"{scene}/{row['source_sample_id']}"
            require(row.get('split') == 'train' and source_id in originals, 'Augmentation parent is outside the original pool')
            group = row.get('primary_group',row.get('focus_group')); key=(scene,sid)
            require(group in ('crater','mountain_rock','mixed') and key not in augments, 'Invalid augmentation identity/group')
            image, mask = (canonical_path(aug_path.parent, row[k]) for k in ('image','mask'))
            require(image != mask and image not in image_paths and image not in mask_paths and mask not in image_paths and mask not in mask_paths, 'Duplicate augmentation paths')
            augments[key]={'scene':scene,'sample_id':sid,'source_id':source_id,'source_kind':'augmented','augmentation_group':group,'image':image,'mask':mask,'split':'train'}
            image_paths.add(image); mask_paths.add(mask)
        # Reuse the exact original test archive/member rules without opening labels here.
        full_dataset = LunarDataset(test_path,'test',input_size=input_size)
        for record in full_dataset.records:
            row=dict(record); row['image']=str(Path(row['image']).resolve()); row['label_archive']=str(Path(row['label_archive']).resolve())
            row.update(split='val',source_kind='test_original')
            require(row['source_id'] not in full_test and row['image'] not in image_paths and row['image'] not in mask_paths, 'Duplicate test/source image')
            full_test[row['source_id']]=row; image_paths.add(row['image'])
        require(len(full_test)==5004, 'Full original test must contain 5004 unique images')
        hard_ids=[]
        for row in hard_doc['records']:
            sid=f"{row['scene']}/{row['sample_id']}"; candidate=full_test.get(sid)
            require(candidate is not None and canonical_path(hard_path.parent,row['image'])==candidate['image'], 'Hard row differs from full test provenance')
            hard_ids.append(sid)
        require(len(hard_ids)==1500 and len(set(hard_ids))==1500, 'Hard validation membership is not unique')
        require(Counter(full_test[s]['scene'] for s in hard_ids)==HARD_COUNTS,'Hard validation scene counts changed')
        train_pool=set(families['train_pool_source_ids']); val_ids=set(families['val_source_ids']); remaining=set(families['remaining_test_source_ids'])
        require(train_pool==set(originals) and val_ids==set(hard_ids) and remaining==set(full_test)-val_ids, 'Literal source partitions differ from provenance')
        require(not (train_pool & val_ids or train_pool & remaining or val_ids & remaining), 'Train/validation/remaining-test sources overlap')
        selected_ids, selected_images, selected_masks, original_ids = set(),set(),set(),set()
        actual_groups=Counter()
        for role, rows in splits.items():
            for row in rows:
                require(isinstance(row,dict) and row.get('split')==role,'Literal row split mismatch')
                key=(row.get('scene'),row.get('sample_id')); kind=row.get('source_kind')
                if role=='val':
                    expected=full_test.get(row.get('source_id'))
                    require(kind=='test_original' and row.get('source_id') in val_ids and 'mask' not in row, 'Validation must retain fixed test ZIP labels')
                elif kind=='original':
                    expected=originals.get(row.get('source_id')); original_ids.add(row.get('source_id'))
                else:
                    require(kind=='augmented' and row.get('augmentation_group') in ('crater','mountain_rock'),'Invalid selected training sample kind')
                    expected=augments.get(key); actual_groups[row.get('augmentation_group')]+=1
                require(expected is not None and all(row.get(k)==v for k,v in expected.items()), 'Literal row differs from source provenance')
                require(key not in selected_ids and row['image'] not in selected_images, 'Duplicate selected sample/image')
                selected_ids.add(key);selected_images.add(row['image'])
                if 'mask' in row:
                    require(row['mask'] not in selected_masks,'Duplicate selected mask');selected_masks.add(row['mask'])
        require([r['source_id'] for r in splits['val']]==hard_ids, 'Fixed validation order differs from original hard manifest')
        require(len(original_ids)==4000 and actual_groups=={'crater':1000,'mountain_rock':1000} and len(splits['train'])==6000, 'Training must contain 4000 originals + 1000 crater + 1000 mountain_rock')
        expected_counts={'original_pool':8002,'train_original':4000,'train_augmented':2000,'train_total':6000,'val_original':1500,'val_total':1500,'remaining_test':3504,'full_original_test':5004}
        counts=source.get('counts',{})
        require(all(type(counts.get(k)) is int and counts[k]==v for k,v in expected_counts.items()),'Declared counts differ from literal data')
        for path,sha in ((source_path,source_sha),(aug_path,aug_sha),(test_path,test_sha),(hard_path,hard_sha),(self.manifest,fingerprint)):
            require(sha256_file(path)==sha, 'A manifest changed during initialization')
        self.records=[dict(r) for r in splits[split]]; self.complete_count=len(self.records)
        if max_samples is not None:self.records=self.records[:max_samples]
        self.partial=len(self.records)!=self.complete_count
        self.identity={'manifest':str(self.manifest),'manifest_sha256':fingerprint,'schema':SCHEMA,'protocol':PROTOCOL,
            'split':split,'custom_split':True,'train_source':'embedded_sampled_original_pool' if split=='train' else None,
            'image_count':len(self.records),'complete_image_count':self.complete_count,'partial':self.partial,
            'unique_source_count':len({r['source_id'] for r in self.records}),
            'ordered_samples_sha256':_canonical_sha256(self.records),'literal_ordered_samples_sha256':ordered[split],
            'families_sha256':source['families_sha256'],'splits_sha256':source['splits_sha256'],'split_counts':counts,
            'source_manifest':str(source_path),'source_manifest_sha256':source_sha,'augmented_manifest':str(aug_path),'augmented_manifest_sha256':aug_sha,
            'test_manifest':str(test_path),'test_manifest_sha256':test_sha,'hard_manifest':str(hard_path),'hard_manifest_sha256':hard_sha,
            'validation_image_count':1500,'validation_scene_counts':HARD_COUNTS,'sample_scene_counts':dict(Counter(r['scene'] for r in self.records)),
            'selection_metric':'validation_weighted_score','selection_scope':'fixed_hard_validation_from_former_test',
            'validation_weights':{'Moon_2':1,'Moon_4':1,'Moon_9':2},'test_labels_used':True,
            'checkpoint_selection_uses_test':True,'official_test_used_for_selection':True,
            'remaining_test_image_count':3504,'remaining_test_independent':False,
            'validation_selected_using_model_predictions':True,'full_original_test_overlaps_validation':True,
            'fresh_training_required':True,'split_seed':source['seed']}
