#!/usr/bin/env python3
"""Evaluate an independent ViT checkpoint on the fixed validation or Moon_2/4/9."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from vitseg.data import LunarDataset, DEFAULT_TEST_MANIFEST, sha256_file
from vitseg.model import ModelConfig, build_model
from vitseg.metrics import evaluate_loader
from vitseg.diagnostics import TerrainDiagnostics
from vitseg.visualization import ComparisonExporter, write_json

ROOT=Path(__file__).resolve().parents[1]

def load_checkpoint(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format') not in (None,'lusnar_vit_checkpoint_v1'):
        raise ValueError('This checkpoint does not belong to the independent ViT experiments')
    config=ModelConfig.from_dict(saved['model_config'])
    model=build_model(config,pretrained=False)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved,config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--test-manifest',default=DEFAULT_TEST_MANIFEST)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--amp',choices=['none','bf16','fp16'],default='bf16')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--tta',choices=['none','hflip','multiscale_hflip'],default='none')
    parser.add_argument('--tta-sizes',nargs='+',type=int)
    parser.add_argument('--tta-weights',nargs='+',type=float)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-samples',type=int,default=None,help='Execution checks only; results explicitly marked partial')
    parser.add_argument('--object-diagnostics',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--large-crater-area',type=int,default=16384)
    parser.add_argument('--small-rock-area',type=int,default=256)
    parser.add_argument('--interior-erosion',type=int,default=5)
    parser.add_argument('--save-images',action='store_true')
    parser.add_argument('--image-output-root',type=Path,default=Path('/media/nas/lzc/fenge_result'))
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0 or (args.max_samples is not None and args.max_samples<1):parser.error('Invalid sample/batch/worker count')
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    model,saved,model_config=load_checkpoint(args.checkpoint,device)
    training=saved.get('train_config',saved.get('run_config'))
    if not isinstance(training,dict):raise ValueError('Checkpoint is missing its training configuration')
    manifest=training['manifest'] if args.split=='val' else args.test_manifest
    dataset=LunarDataset(manifest,args.split,model_config.input_size,horizontal_flip=False,max_samples=args.max_samples)
    expected=saved.get('data_identity',{}).get('val')
    if args.split=='val' and args.max_samples is None:
        if expected is None or dataset.identity!=expected:
            raise ValueError('Missing or changed validation identity; retain the fixed split saved by training')
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    tag=args.tta
    if args.tta_sizes:tag+='_'+'_'.join(map(str,args.tta_sizes))
    if args.tta_weights:tag+='_weights_'+'_'.join(str(x).replace('.','p') for x in args.tta_weights)
    if args.max_samples is not None:tag+=f'_partial{args.max_samples}'
    output=args.output or (args.checkpoint.parent/f'{args.split}_{tag}_metrics.json')
    metadata={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256_file(args.checkpoint),
              'checkpoint_epoch':saved['epoch'],'split':args.split,'tta':args.tta,
              'tta_sizes':args.tta_sizes,'tta_weights':args.tta_weights,'model_config':saved['model_config'],
              'checkpoint_partial_training':bool(saved.get('partial',False)),
              'score_used_for_ranking':'weighted_score for complete Moon_2/4/9; miou is the pooled pixel metric'}
    observer=TerrainDiagnostics(args.large_crater_area,args.small_rock_area,args.interior_erosion) if args.object_diagnostics else None
    exporter=ComparisonExporter(args.image_output_root/args.checkpoint.parent.name/f'{args.split}_{tag}',
        output.parent/f'{args.split}_{tag}_per_image',metadata) if args.save_images else None
    start=time.monotonic()
    metrics=evaluate_loader(model,loader,device,amp=args.amp,tta=args.tta,tta_sizes=args.tta_sizes,
        tta_weights=args.tta_weights,diagnostic_observer=observer,image_exporter=exporter,progress_every=25)
    metrics.update(metadata,seconds=time.monotonic()-start)
    if observer is not None:metrics['object_diagnostics']=observer.compute()
    if exporter is not None:exporter.finalize(metrics)
    write_json(output,metrics)
    print(json.dumps({'output':str(output),'image_count':metrics.get('image_count'),
                      'partial':metrics.get('partial'),'miou':metrics.get('miou'),
                      'weighted_score':metrics.get('weighted_score')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
