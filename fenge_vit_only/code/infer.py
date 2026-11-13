#!/usr/bin/env python3
"""Whole-image inference for pure DINOv3 ViT-S/16 Mask2Former checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from evaluate import load_checkpoint
from vitseg.metrics import autocast_context, predict_semantic_scores
from vitseg.visualization import PALETTE,write_json
from vitseg.data import sha256_file

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--amp',choices=['none','bf16','fp16'],default='bf16')
    parser.add_argument('--tta',choices=['none','hflip','multiscale_hflip'],default='none')
    parser.add_argument('--tta-sizes',nargs='+',type=int)
    parser.add_argument('--tta-weights',nargs='+',type=float)
    parser.add_argument('--overlay',action='store_true')
    args=parser.parse_args()
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    model,saved,config=load_checkpoint(args.checkpoint,device)
    source=args.input.resolve()
    if not source.exists():parser.error('Input path does not exist')
    suffixes={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}
    images=sorted(p for p in source.rglob('*') if p.suffix.lower() in suffixes and p.is_file()) if source.is_dir() else [source]
    if not images:parser.error('No input images found')
    output=(args.output or (Path('/media/nas/lzc/fenge_result')/args.checkpoint.parent.name/'inference')).resolve()
    if source.is_dir() and output.is_relative_to(source):parser.error('Output must be outside the input directory')
    source_root=source if source.is_dir() else source.parent
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    entries=[]
    for number,path in enumerate(images,1):
        with Image.open(path) as image:rgb=image.convert('RGB')
        array=np.array(rgb.resize((config.input_size,config.input_size),Image.Resampling.BILINEAR),dtype=np.float32)/255
        normalized=(torch.from_numpy(array).permute(2,0,1)[None].to(device)-mean)/std
        with autocast_context(device,args.amp):scores=predict_semantic_scores(model,normalized,tta=args.tta,tta_sizes=args.tta_sizes,tta_weights=args.tta_weights)
        # Average normalized class probabilities before resampling/argmax.
        scores=F.interpolate(scores.float(),size=(rgb.height,rgb.width),mode='bilinear',align_corners=False)
        prediction=scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
        relative=path.relative_to(source_root);folder=output/relative.parent/relative.name;folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(prediction).save(folder/'prediction_ids.png')
        color=PALETTE[prediction];Image.fromarray(color).save(folder/'prediction_color.png')
        if args.overlay:Image.fromarray((np.asarray(rgb)*.55+color*.45).round().astype(np.uint8)).save(folder/'overlay.png')
        entries.append({'input':str(path),'output':str(folder),'original_size_wh':[rgb.width,rgb.height]})
        if number%25==0:print(f'{number}/{len(images)} images',flush=True)
    metadata={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256_file(args.checkpoint),
              'checkpoint_epoch':saved['epoch'],'tta':args.tta,'tta_sizes':args.tta_sizes,
              'tta_weights':args.tta_weights,'ground_truth_available':False,'iou':None,'predictions':entries}
    write_json(output/'predictions.json',metadata)
    local_text=args.checkpoint.parent/'inference_predictions.json';write_json(local_text,metadata)
    print(json.dumps({'images':len(entries),'output':str(output),'local_metadata':str(local_text)}),flush=True)

if __name__=='__main__':main()
