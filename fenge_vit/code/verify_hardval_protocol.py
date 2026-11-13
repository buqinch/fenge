#!/usr/bin/env python3
"""CPU verification of the opt-in hard-validation protocol and resume boundary."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import train_hardval as base
from hardval_data import HardValidationDataset, HARD_COUNTS
from vitseg.data import LunarDataset, sha256_file
from vitseg.metrics import ConfusionMatrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(base.ROOT/'config/c_adapter_masking_pool8002_train6000_hardval1500_103_4gpu.json'))
    parser.add_argument('--output',default=str(base.ROOT/'result/hardval_protocol_verification.json'))
    args=parser.parse_args()
    train_args=base.parse_args(['--config',args.config])
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    train=HardValidationDataset(train_args.manifest,'train');val=HardValidationDataset(train_args.manifest,'val')
    check('6000 train / 1500 validation',len(train)==6000 and len(val)==1500)
    check('training ratio',Counter((r['augmentation_group'] if r['source_kind']=='augmented' else 'original') for r in train.records)=={'original':4000,'crater':1000,'mountain_rock':1000})
    check('fixed scene counts',Counter(r['scene'] for r in val.records)==HARD_COUNTS)
    check('training-source isolation',not ({r['source_id'] for r in train.records}&{r['source_id'] for r in val.records}))
    check('full former-test provenance explicit',val.identity['checkpoint_selection_uses_test'] and val.identity['official_test_used_for_selection'] and not val.identity['remaining_test_independent'])
    original= LunarDataset(val.identity['test_manifest'],'test')
    old_indices={r['source_id']:i for i,r in enumerate(original.records)}
    samples=[]
    for scene in HARD_COUNTS:
        i=next(i for i,r in enumerate(val.records) if r['scene']==scene)
        x=val[i];y=original[old_indices[x['source_id']]]
        check(scene+' inherits exact ZIP label mapping',torch.equal(x['mask'],y['mask']) and torch.equal(x['image'],y['image']))
        check(scene+' native 1024 geometry',x['image'].shape==(3,1024,1024) and x['mask'].shape==(1024,1024))
        samples.append({'source_id':x['source_id'],'class_ids':x['mask'].unique().tolist()})
    for kind in ('original','crater','mountain_rock'):
        i=next(i for i,r in enumerate(train.records) if (r.get('augmentation_group') if r['source_kind']=='augmented' else 'original')==kind)
        x=train[i];check('training '+kind+' RGB/ID mask load',x['image'].shape==(3,1024,1024) and x['mask'].shape==(1024,1024) and bool(torch.all((x['mask']>=0)&(x['mask']<=4))))
    partial=HardValidationDataset(train_args.manifest,'val',max_samples=2)
    batch=next(iter(DataLoader(partial,batch_size=2,num_workers=0)))
    check('default collate',tuple(batch['image'].shape)==(2,3,1024,1024) and tuple(batch['mask'].shape)==(2,1024,1024))
    scenes={}
    for name,rate in zip(HARD_COUNTS,(.8,.9,.6)):
        cm=ConfusionMatrix();cm.matrix[0,0]=int(rate*100);cm.matrix[0,1]=100-int(rate*100)
        scenes[name]={**cm.compute(),'image_count':HARD_COUNTS[name]}
    metrics={'image_count':1500,'validation_unique_source_count':1500,'data_identity':val.identity,'scenes':scenes,'miou':.123,'partial':True,'test_scope':'partial_test','weighted_score':None}
    annotated=base.annotate_validation(metrics,2)
    check('weighted scene score distinct from pooled',abs(annotated['validation_weighted_score']-.725)<1e-12 and annotated['miou']==.123)
    check('complete 1500 validation not partial-test nor official S',not annotated['partial'] and annotated['test_scope']=='fixed_hard_validation' and annotated['weighted_score'] is None and not annotated['official_test_scope'])
    invalid=copy.deepcopy(metrics);invalid['image_count']=1499
    reject('reject incomplete coverage',lambda:base.annotate_validation(invalid,2))
    invalid=copy.deepcopy(metrics);invalid['scenes']['Moon_9']['image_count']=1124
    reject('reject wrong scene coverage',lambda:base.annotate_validation(invalid,2))
    invalid=copy.deepcopy(metrics);invalid['scenes'].pop('Moon_4')
    reject('reject missing full scene',lambda:base.validation_score(invalid))
    invalid=copy.deepcopy(metrics);invalid['data_identity']['hard_manifest_sha256']=None
    reject('reject absent fixed manifest identity',lambda:base.annotate_validation(invalid,2))
    part_metrics={'image_count':2,'validation_unique_source_count':2,'data_identity':partial.identity,'scenes':{'Moon_2':{**scenes['Moon_2'],'image_count':2}},'miou':.8,'partial':True}
    part=base.annotate_validation(part_metrics,2,partial=True)
    check('partial runtime score explicitly proxy',part['validation_score_is_partial_proxy'] and part['partial'])
    cadence=SimpleNamespace(validation_every=2,epochs=5)
    check('validate every two epochs plus odd final',[e for e in range(1,6) if base.should_validate(e,cadence)]==[2,4,5])
    tracker=base.PatienceTracker(patience=5,min_delta=.0005,start_epoch=4)
    untouched=tracker.state_dict();tracker.update(1,None)
    check('unvalidated epoch does not count',tracker.state_dict()==untouched)
    clone=base.PatienceTracker();clone.load_state_dict(json.loads(json.dumps(tracker.state_dict(),allow_nan=False)))
    check('unvalidated state JSON/resume',clone.state_dict()==tracker.state_dict())
    stops=[]
    for e in range(2,15):
        value=.8 if e%2==0 else None
        if tracker.update(e,value):stops.append(e)
    check('patience means five validation checks',stops==[14] and tracker.bad_epochs==5 and tracker.validation_checks==7)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);(out/'history.jsonl').write_bytes(b'{"epoch":1}\n{"epoch":2}\n{"epoch":')
        saved={'epoch':1,'best_epoch':None,'best_miou':None,'_resume_path':str(out/'last.pt')}
        base._validate_history(out,saved,True)
        check('recover unvalidated checkpoint and truncated history',(out/'history.jsonl').read_text()=='{"epoch":1}\n' and not (out/'best_smoke.pt').exists())
        (out/'history.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n')
        saved.update(epoch=2,best_epoch=2,best_miou=.725,model={'weight':torch.tensor([3.])})
        base._validate_history(out,saved,True)
        repaired=torch.load(out/'best_smoke.pt',map_location='cpu',weights_only=True)
        check('repair best from durable current-best last',repaired['epoch']==2 and repaired['best_miou']==.725 and '_resume_path' not in repaired)
    old=ast.parse((base.ROOT/'code/train.py').read_text());new=ast.parse((base.ROOT/'code/train_hardval.py').read_text())
    for name in ('PolynomialSchedule','_training_windows','_sample_keys','_sum_mask_stats','rng_state','restore_rng'):
        left=next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        right=next(n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        check('unchanged implementation '+name,ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False))
    baseline_text=(base.ROOT/'code/train.py').read_text()
    code_source=(base.ROOT/'code/train_hardval.py').read_text()
    for name,start,end_old,end_new in (
        ('model construction', '    model_config = ModelConfig(', '    configuration = _training_config(args)', '    configuration = _training_config(args)'),
        ('optimizer criterion and scheduler', '    model.to(device)', '    start_epoch, global_update, best, best_epoch', '    start_epoch, global_update, best, best_epoch'),
        ('intact and masking forward backward optimizer steps', '        for batch_index, batch, window_images, mask_setup in _training_windows(', '        validation = evaluate_loader(', '        validation, score = None, None')):
        left=baseline_text[baseline_text.index(start):baseline_text.index(end_old,baseline_text.index(start))]
        right=code_source[code_source.index(start):code_source.index(end_new,code_source.index(start))]
        check('exact baseline implementation '+name,left==right)
    baseline_sha={'code/train.py':'94bb978631d61ea77357000787dcdb9767369b80ccaf63f12e209726bda7e0bd',
        'code/train_distributed.py':'f61d391fb436322aa2d4ad9c5f9500baca3cfc9b20bedd5a1f253c83e37272c5',
        'code/distributed_training.py':'fe3d51b87898934a46fc1d7c5970e93e58f4d158f1ff919dac2f635423a53155',
        'code/vitseg/data.py':'d6bf1ab5e74c88f860a917db155253ca7eefef5be0d47d82537a507018bf0d42'}
    for name,fingerprint in baseline_sha.items():check('active source unchanged '+name,sha256_file(base.ROOT/name)==fingerprint)
    check('opt-in entry and loader source SHA captured',"Path(__file__)" in code_source and "with_name('hardval_data.py')" in code_source)
    base.write_json(args.output,{'status':'passed','checks':checks,'train_identity':train.identity,'validation_identity':val.identity,'sample_label_checks':samples,'formal_training_started':False,'gpu_used':False,'source_sha256':{str(p.relative_to(base.ROOT)):sha256_file(p) for p in [base.ROOT/'code/train_hardval.py',base.ROOT/'code/hardval_data.py',base.ROOT/'code/train.py',base.ROOT/'code/train_distributed.py',base.ROOT/'code/distributed_training.py',base.ROOT/'code/vitseg/data.py']}})
    print(json.dumps({'status':'passed','checks':len(checks),'manifest_sha256':train.identity['manifest_sha256'],'report':args.output},ensure_ascii=False))

if __name__=='__main__':main()
