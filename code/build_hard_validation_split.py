#!/usr/bin/env python3
"""Sample train originals from the complete 8002-image pool; reuse fixed hard val.

This is a new, explicit protocol: the former hardest 1500 test images become
validation for new experiments. Old manifests and running jobs are not modified.
The remaining 3504 test images form a separate, already-inspected evaluation set.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import zipfile
from PIL import Image

SCHEMA='lusnar_all_sources_hardval_v1'
PROTOCOL='sampled_original_pool_with_fixed_hard_test_validation'
CLASSES=['regolith','rock','crater','mountain','sky']
TRAIN_SCENES={'Moon_1','Moon_3','Moon_5','Moon_6','Moon_7','Moon_8'}
TEST_COUNTS={'Moon_2':1597,'Moon_4':1320,'Moon_9':2087}
VAL_COUNTS={'Moon_2':188,'Moon_4':187,'Moon_9':1125}
REMAINING_COUNTS={'Moon_2':1409,'Moon_4':1133,'Moon_9':962}
GROUPS=('crater','mountain_rock')


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def absolute(root,value):
    path=Path(value);return str((path if path.is_absolute() else root/path).resolve())


def sid(row):return f"{row['scene']}/{row['sample_id']}"


def original_row(row,root):
    if row['scene'] not in TRAIN_SCENES:raise ValueError('Test scene in original training pool')
    return {'split':'train','source_kind':'original','scene':row['scene'],'sample_id':str(row['sample_id']),
            'source_id':sid(row),'image':absolute(root,row['image']),'mask':absolute(root,row['mask'])}


def augmented_row(row,root):
    scene=row['source_scene'];source_id=f"{scene}/{row['source_sample_id']}"
    group=row.get('primary_group',row.get('focus_group'))
    if row.get('split')!='train' or group not in GROUPS:raise ValueError('Illegal augmentation group/split')
    return {'split':'train','source_kind':'augmented','scene':scene,'sample_id':str(row['sample_id']),
            'source_id':source_id,'image':absolute(root,row['image']),'mask':absolute(root,row['mask']),
            'augmentation_group':group}


def test_row(row,root,split):
    scene=row['scene'];sample=str(row['sample_id'])
    archive=(Path('/home/nas/lzc/lusnar_test_eval_20261102/labels/Moon_4_labels_verified.zip')
             if scene=='Moon_4' else Path(f'/home/nas/lzc/{scene}.zip/{scene}.zip'))
    return {'split':split,'source_kind':'test_original','scene':scene,'sample_id':sample,'source_id':sid(row),
            'image':absolute(root,row['image']),'label_archive':str(archive.resolve()),
            'label_member':f'{scene}/image0/label/{sample}.png'}


def verify_mask(row):
    with Image.open(row['image']) as image:image_size=image.size
    with Image.open(row['mask']) as mask:
        if mask.mode!='L' or mask.getextrema()[0]<0 or mask.getextrema()[1]>4:
            raise ValueError(f'Invalid ID mask: {row["mask"]}')
        if mask.size!=image_size:raise ValueError(f'Image/mask size mismatch: {row["source_id"]}')
    return image_size


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    seed=cfg.get('seed',42)
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer seed required')
    count=cfg['original_count'];aug_counts=cfg['augmentation_counts']
    if type(count) is not int or count<1 or count>8002:raise ValueError('Original count must be 1..8002')
    if set(aug_counts)!=set(GROUPS) or any(type(n) is not int or n<0 for n in aug_counts.values()):
        raise ValueError('Explicit nonnegative crater and mountain_rock counts are required')
    output=Path(cfg['output_manifest']);remaining_path=Path(cfg['remaining_test_manifest']);audit_dir=Path(cfg['audit_directory'])
    if output.exists() or remaining_path.exists():raise FileExistsError('New manifests are immutable; choose a new versioned output')
    audit_dir.mkdir(parents=True,exist_ok=True)
    paths={name:Path(cfg[name]).resolve() for name in ('source_manifest','augmented_manifest','hard_manifest','test_manifest','preferred_selection_manifest','old_split_manifest')}
    hashes={name:file_sha(path) for name,path in paths.items()}
    docs={name:json.loads(path.read_text()) for name,path in paths.items()}
    source,augdoc,hard,test=map(docs.get,('source_manifest','augmented_manifest','hard_manifest','test_manifest'))
    if source['class_names']!=CLASSES or augdoc['class_names']!=CLASSES:raise ValueError('Class order mismatch')
    if (augdoc.get('validation_or_test_used') is not False
            or augdoc.get('source_manifest_sha256')!=hashes['source_manifest']):raise ValueError('Augmentation source provenance differs')
    originals=sorted([original_row(r,paths['source_manifest'].parent) for r in source['samples']],key=lambda r:r['source_id'])
    pool={r['source_id'] for r in originals}
    if len(originals)!=8002 or len(pool)!=8002:raise ValueError('Incomplete/duplicate original pool')
    full_records=test['records'];full_by_id={sid(r):r for r in full_records}
    if len(full_records)!=5004 or len(full_by_id)!=5004 or Counter(r['scene'] for r in full_records)!=TEST_COUNTS:
        raise ValueError('Full former test source must have exactly 5004 unique images')
    if (hard.get('schema')!='lusnar_fixed_hard_test_v1' or hard.get('full_test_manifest_sha256')!=hashes['test_manifest']
            or canonical(hard['records'])!=hard.get('ordered_records_sha256')):raise ValueError('Fixed hard-validation source identity changed')
    if len(hard['records'])!=1500 or Counter(r['scene'] for r in hard['records'])!=VAL_COUNTS:raise ValueError('Hard-validation quotas changed')
    if any(r!=full_by_id.get(sid(r)) for r in hard['records']):raise ValueError('Hard-validation row differs from original test source')
    validation_ids={sid(r) for r in hard['records']}
    if len(validation_ids)!=1500 or validation_ids&pool:raise ValueError('Validation identity collision')
    remaining=[r for r in full_records if sid(r) not in validation_ids]
    if Counter(r['scene'] for r in remaining)!=REMAINING_COUNTS:raise ValueError('Remaining evaluation count mismatch')
    namespace=cfg.get('sampling_namespace','pool8002_train6000_hardval1500')
    random_streams={}
    for name in ('originals','augmentation_crater','augmentation_mountain_rock','training_order'):
        material=f'{namespace}:{name}:{seed}'
        hexdigest=hashlib.sha256(material.encode('utf-8')).hexdigest()
        random_streams[name]={'material':material,'sha256':hexdigest,'integer_seed':int(hexdigest,16)}
    original_rng=random.Random(random_streams['originals']['integer_seed'])
    selected_originals=original_rng.sample(originals,count)
    selected_original_ids={r['source_id'] for r in selected_originals}
    augmented=[augmented_row(r,paths['augmented_manifest'].parent) for r in augdoc['samples']
               if r.get('primary_group',r.get('focus_group')) in GROUPS]
    if any(r['source_id'] not in pool for r in augmented):raise ValueError('Augmentation parent outside 8002-source training pool')
    augmented_by_key={(r['scene'],r['sample_id']):r for r in augmented}
    if len(augmented_by_key)!=len(augmented):raise ValueError('Duplicate augmentation identities')
    preferred=docs['preferred_selection_manifest']['samples']
    selected_augmented=[];augmentation_audit={}
    for group in GROUPS:
        candidates=sorted([r for r in augmented if r['augmentation_group']==group],key=lambda r:(r['scene'],r['sample_id']))
        preferred_rows=[]
        for old in preferred:
            if old.get('augmentation_group')!=group:continue
            key=(old['scene'],old['sample_id']);row=augmented_by_key.get(key)
            if row is None or any(row[k]!=old[k] for k in ('source_id','image','mask','augmentation_group')):
                raise ValueError('Preferred C6000 augmentation row changed or disappeared')
            preferred_rows.append(row)
        preferred_rows.sort(key=lambda r:(r['scene'],r['sample_id']))
        if len({(r['scene'],r['sample_id']) for r in preferred_rows})!=len(preferred_rows):raise ValueError('Preferred augmentation list repeats samples')
        rng=random.Random(random_streams[f'augmentation_{group}']['integer_seed'])
        wanted=aug_counts[group]
        if wanted>len(candidates):raise ValueError(f'Requested {wanted} {group} but pool has {len(candidates)}')
        if wanted<len(preferred_rows):chosen=rng.sample(preferred_rows,wanted)
        else:
            preferred_keys={(r['scene'],r['sample_id']) for r in preferred_rows}
            extra=[r for r in candidates if (r['scene'],r['sample_id']) not in preferred_keys]
            chosen=preferred_rows+rng.sample(extra,wanted-len(preferred_rows))
        selected_augmented.extend(chosen)
        augmentation_audit[group]={'available':len(candidates),'requested':wanted,'preferred_available':len(preferred_rows),
            'preferred_preserved':len(set((r['scene'],r['sample_id']) for r in chosen)&set((r['scene'],r['sample_id']) for r in preferred_rows)),
            'selected':len(chosen),'selected_parents_without_selected_original':len({r['source_id'] for r in chosen}-selected_original_ids)}
    train=[*selected_originals,*selected_augmented]
    random.Random(random_streams['training_order']['integer_seed']).shuffle(train)
    val=[test_row(r,paths['test_manifest'].parent,'val') for r in hard['records']]
    splits={'train':train,'val':val}
    if len({r['image'] for r in train+val})!=len(train)+len(val):raise ValueError('Duplicate selected image files')
    if {r['source_id'] for r in train}&validation_ids:raise ValueError('Train/validation source collision')
    print(json.dumps({'stage':'verify_files','train_images':len(train),'validation_images':len(val),'remaining_evaluation_images':len(remaining)}),flush=True)
    with ThreadPoolExecutor(max_workers=cfg.get('verification_workers',4)) as executor:
        dimensions=Counter(executor.map(verify_mask,train))
    label_info={}
    for scene in TEST_COUNTS:
        subset=[r for r in val if r['scene']==scene];archive=Path(subset[0]['label_archive'])
        with zipfile.ZipFile(archive) as zip_file:
            for row in subset:
                if not Path(row['image']).is_file():raise FileNotFoundError(row['image'])
                info=zip_file.getinfo(row['label_member'])
                if info.file_size<=0:raise ValueError('Empty validation label member')
        st=archive.stat();label_info[scene]={'archive':str(archive),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'validated_member_count':len(subset)}
    if any(file_sha(paths[k])!=hashes[k] for k in paths):raise RuntimeError('Input provenance changed during construction')
    family={'train_pool_source_ids':sorted(pool),'val_source_ids':sorted(validation_ids),
            'remaining_test_source_ids':sorted(sid(r) for r in remaining)}
    remaining_document={'schema':'lusnar_remaining_test_v1','split_protocol':'former_full_test_minus_new_hard_validation',
        'full_test_manifest':str(paths['test_manifest']),'full_test_manifest_sha256':hashes['test_manifest'],
        'hard_manifest':str(paths['hard_manifest']),'hard_manifest_sha256':hashes['hard_manifest'],
        'records':remaining,'counts':dict(REMAINING_COUNTS),'image_count':len(remaining),
        'ordered_records_sha256':canonical(remaining),'validation_source_ids_sha256':canonical(sorted(validation_ids)),
        'validation_overlap':0,'complete_original_official_test':False,'previously_inspected_test_domain':True,
        'selection_is_complement_of_model_selected_validation':True,
        'interpretation':'3504-image remaining evaluation cohort after removing the model-selected 1500 validation images; not the complete original official 5004-image test'}
    write_json(remaining_path,remaining_document)
    old_val_ids=set(docs['old_split_manifest']['original_split']['val_source_ids'])
    counts={'original_pool':8002,'train_original':count,'train_augmented':len(selected_augmented),'train_total':len(train),
            'val_original':1500,'val_total':1500,'remaining_test':3504,'full_original_test':5004,'unselected_original':8002-count}
    policy={'test_labels_used_for_validation_selection':True,'former_test_samples_in_validation':1500,
        'full_original_test_validation_overlap':1500,'remaining_test_validation_overlap':0,
        'validation_used_for_checkpoint_selection':True,'checkpoint_selection_uses_former_test_labels':True,
        'validation_selection_model':{'checkpoint_epoch':hard['checkpoint_epoch'],'checkpoint_sha256':hard['checkpoint_sha256']},
        'source_hard_manifest_original_role':'diagnostic_test_only',
        'current_role_override':'validation_for_new_training_experiments_explicitly_requested_by_user',
        'requires_fresh_training':True,
        'reporting_note':'Original 5004-image test now contains 1500 validation images and is no longer independent of best selection; remaining 3504 is already inspected and excludes deliberately selected difficult cases.'}
    document={'schema':SCHEMA,'split_protocol':PROTOCOL,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'seed':seed,'base_seed':seed,'random_streams':random_streams,'random_stream_derivation':'SHA256 UTF-8 namespace:stream_name:base_seed, converted from full hexadecimal digest to integer; no Python hash()',
        'class_names':CLASSES,'sampling_algorithm':'Sort all 8002 originals by source_id; sample without replacement with the named originals SHA256-derived stream; preserve preferred C6000 augmentation identities when counts match; independent group streams extend/subsample if counts change; independent training_order stream shuffles final rows',
        'python_version':platform.python_version(),'builder_sha256':file_sha(__file__),'config':cfg,'config_sha256':file_sha(args.config),
        **{k:str(v) for k,v in paths.items()},**{k+'_sha256':v for k,v in hashes.items()},
        'remaining_test_manifest':str(remaining_path.resolve()),'remaining_test_manifest_sha256':file_sha(remaining_path),
        'counts':counts,'train_counts':{'original':count,**aug_counts},'families':family,'families_sha256':canonical(family),
        'splits':splits,'splits_sha256':canonical(splits),'ordered_samples_sha256':{k:canonical(v) for k,v in splits.items()},
        'selected_original_source_ids':sorted(selected_original_ids),'selected_original_source_ids_sha256':canonical(sorted(selected_original_ids)),
        'unselected_original_source_ids':sorted(pool-selected_original_ids),
        'unselected_original_source_ids_sha256':canonical(sorted(pool-selected_original_ids)),
        'actual_training_source_ids_sha256':canonical(sorted({r['source_id'] for r in train})),
        'augmentation_parent_need_selected_original':False,'augmentation_selection':augmentation_audit,
        'validation_scene_counts':dict(VAL_COUNTS),'validation_order_matches_fixed_hard_manifest':True,
        'label_validation':{'train_id_masks_checked':len(train),'required_mode':'L','class_id_range':[0,4],
                            'train_image_dimensions':{f'{w}x{h}':n for (w,h),n in dimensions.items()},'validation_archive_members':label_info},
        'protocol_policy':policy,**policy}
    write_json(output,document)
    audit={'status':'passed','manifest':str(output.resolve()),'manifest_sha256':file_sha(output),'schema':SCHEMA,'split_protocol':PROTOCOL,
        'counts':counts,'base_seed':seed,'random_streams':random_streams,'selected_original_scene_counts':dict(sorted(Counter(r['scene'] for r in selected_originals).items())),
        'original_pool_scene_counts':dict(sorted(Counter(r['scene'] for r in originals).items())),
        'originals_from_previous_1601_validation':len(selected_original_ids&old_val_ids),
        'old_validation_augmented_parents':len({r['source_id'] for r in selected_augmented}&old_val_ids),
        'unique_training_sources':len({r['source_id'] for r in train}),
        'augmentation_parents_without_selected_original':len({r['source_id'] for r in selected_augmented}-selected_original_ids),
        'validation_scene_counts':dict(VAL_COUNTS),'remaining_test_scene_counts':dict(REMAINING_COUNTS),
        'augmentation_selection':augmentation_audit,'old_hard_manifest_unchanged_sha256':hashes['hard_manifest'],
        'checks':['8002 original candidates complete and unique','fresh seed42 no-replacement original sample across full pool','fixed 1500 validation identities and ordering unchanged','training source families disjoint from validation and remaining test','augmentation parent IDs belong to the 8002 training candidate pool','remaining 3504 exactly complements validation within former 5004','all selected ID masks legal and paired image sizes equal','all validation image paths and archive members exist','input manifests unchanged'],
        'protocol_policy':policy}
    write_json(audit_dir/'audit.json',audit)
    print(json.dumps({'status':'complete','manifest':str(output),'counts':counts,'scene_counts':audit['selected_original_scene_counts'],'previous_validation_originals_selected':audit['originals_from_previous_1601_validation']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
