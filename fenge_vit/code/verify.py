#!/usr/bin/env python3
"""Execute full-resolution pretrained A/B/C networks, gradients, and strict reload."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from vitseg.model import ModelConfig,build_model
from vitseg.losses import Mask2FormerCriterion
from vitseg.masking import make_block_masks,apply_block_mask,masked_semantic_nll
from vitseg.metrics import autocast_context,predict_semantic_scores
from vitseg.data import LunarDataset
from vitseg.visualization import write_json

ROOT=Path(__file__).resolve().parents[1]

def tensor_hash(parameters):
    digest=hashlib.sha256()
    for name,tensor in sorted(parameters):
        value=tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()

def gradient_l1(module):
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    result=sum(float(g.float().abs().sum()) for g in grads)
    assert result>0
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--size',type=int,default=1024)
    parser.add_argument('--variants',nargs='+',choices=['fusion','adapter','masked'],default=['fusion','adapter','masked'])
    parser.add_argument('--output',type=Path,default=ROOT/'result/verification_runtime.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    amp='bf16' if device.type=='cuda' else 'none'
    data=LunarDataset('/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_pool_20261112.json',
                      'train',args.size,selection_manifest='/home/lzc/workspace/fenge/result/selections/random80_4000_2000_2000_seed42_20261112.json',horizontal_flip=False)
    item=data[0];images=item['image'][None].to(device);target=item['mask'][None].to(device)
    key=str(item['source_id'])+'/'+str(item['sample_id'])
    rows=[];common_hash=None
    for name in args.variants:
        torch.manual_seed(42)
        config=ModelConfig(variant='fusion' if name=='fusion' else 'adapter',input_size=args.size,gradient_checkpointing=True)
        model=build_model(config,pretrained=True).to(device)
        current_hash=tensor_hash((n,p) for n,p in model.state_dict().items() if n.startswith(('encoder.','projection.','decoder.')))
        if common_hash is None:common_hash=current_hash
        assert current_hash==common_hash,'Common A/B/C initialization differs'
        assert len(model.encoder.vit.model.blocks)==12 and model.encoder.vit.num_prefix_tokens==5
        if name!='fusion':assert len(model.adapter.stages)==4
        criterion=Mask2FormerCriterion(num_points=4096).to(device)
        optimizer=torch.optim.AdamW([{'params':list(model.backbone_parameters()),'lr':5e-6},
                                     {'params':list(model.head_parameters()),'lr':1e-4}],weight_decay=.01)
        model.train();model.freeze_backbones(True)
        frozen_before=tensor_hash(model.encoder.state_dict().items())
        torch.cuda.reset_peak_memory_stats(device) if device.type=='cuda' else None
        started=time.monotonic()
        with autocast_context(device,amp):
            output=model(images);loss=criterion(output,target)['loss_total']
        assert output['pred_masks'].shape==(1,32,args.size//4,args.size//4)
        assert output['pred_logits'].shape==(1,32,6) and len(output['aux_outputs'])==6
        assert output['semantic_logits'].shape==(1,5,args.size,args.size)
        torch.testing.assert_close(output['semantic_logits'].float().exp().sum(1),torch.ones_like(target,dtype=torch.float32),atol=3e-6,rtol=3e-6)
        assert torch.isfinite(loss);loss.backward()
        frozen_grads={'decoder':gradient_l1(model.decoder)}
        assert all(p.grad is None for p in model.encoder.parameters())
        if model.adapter:
            for i,stage in enumerate(model.adapter.stages):
                frozen_grads[f'inject_{i}']=gradient_l1(stage.inject_attention)
                frozen_grads[f'extract_{i}']=gradient_l1(stage.extract_attention)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert tensor_hash(model.encoder.state_dict().items())==frozen_before
        del output,loss
        model.freeze_backbones(False);model.train()
        with autocast_context(device,amp):
            output=model(images);loss=criterion(output,target)['loss_total']
        assert torch.isfinite(loss);loss.backward();full_loss=float(loss.detach())
        del output,loss
        mask_stats=None
        if name=='masked':
            hidden=make_block_masks(target,[key],3001,ratio=.1,block_size=32)
            masked=apply_block_mask(images,hidden);seen=[]
            hooks=[branch.register_forward_pre_hook(lambda module,inputs:seen.append(inputs[0].data_ptr()))
                   for branch in [model.encoder.vit,model.encoder.convnext]]
            try:
                with autocast_context(device,amp):
                    masked_output=model(masked);aux,mask_stats=masked_semantic_nll(masked_output['semantic_logits'],target,hidden,.2)
                assert seen==[masked.data_ptr(),masked.data_ptr()],'Backbones did not receive the same masked tensor'
                assert torch.isfinite(aux);(.5*aux).backward()
                mask_stats['auxiliary_loss']=float(aux.detach())
            finally:
                for hook in hooks:hook.remove()
            del masked_output,aux,masked
        joint_grads={'vit':gradient_l1(model.encoder.vit),'convnext':gradient_l1(model.encoder.convnext),
                     'decoder':gradient_l1(model.decoder)}
        if model.adapter:joint_grads['adapter']=gradient_l1(model.adapter)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode(),autocast_context(device,amp):
            expected=model(images)['semantic_logits'].float().cpu()
            if args.size==1024:
                scores=predict_semantic_scores(model,images,tta='multiscale_hflip',tta_sizes=[1024,768])
                assert scores.shape==(1,5,1024,1024) and torch.isfinite(scores).all()
                torch.testing.assert_close(scores.sum(1),torch.ones_like(scores[:,0]),atol=3e-6,rtol=3e-6)
                del scores
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        state={n:p.detach().cpu() for n,p in model.state_dict().items()}
        del optimizer,criterion,model
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        restored=build_model(ModelConfig.from_dict(config.to_dict()),pretrained=False)
        restored.load_state_dict(state,strict=True);restored.to(device).eval()
        with torch.inference_mode(),autocast_context(device,amp):actual=restored(images)['semantic_logits'].float().cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        row={'variant':name,'passed':True,'input_size':args.size,'microbatch':1,'pretrained_backbones':'DINOv3 ViT-S/16 and ConvNeXt-Tiny',
             'common_initial_sha256':common_hash,'frozen_backbones_unchanged':True,'frozen_adapter_gradients':frozen_grads,
             'joint_gradients':joint_grads,'full_set_loss':full_loss,'masking':mask_stats,'strict_reload_equal':True,
             'multiscale_hflip_checked':args.size==1024,'peak_allocated_bytes':peak,'seconds':time.monotonic()-started}
        rows.append(row);write_json(args.output,{'passed':len(rows)==len(args.variants),'scope':'execution checks only; no trained segmentation score','device':str(device),'cases':rows})
        print(json.dumps({k:row[k] for k in ['variant','passed','input_size','peak_allocated_bytes','seconds']}),flush=True)
        del restored,state,actual,expected
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':main()
