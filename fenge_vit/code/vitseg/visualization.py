"""Export only RGB/GT/prediction triptychs and per-image IoU records."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .metrics import ConfusionMatrix
from .data import CLASS_NAMES

PALETTE = np.asarray([[187,70,156],[232,250,80],[120,0,200],[173,69,31],[34,201,248]],dtype=np.uint8)

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def safe_name(value):
    value=str(value)
    if not value or value in ('.','..') or any(c in value for c in '/\\\n\r\0'):
        raise ValueError('Invalid scene or sample path component')
    return value

def comparison_image(rgb, target, prediction, metrics):
    height,width=target.shape
    if width<256:
        size=(256,max(1,round(height*256/width)))
        rgb=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR))
        target=np.asarray(Image.fromarray(target.astype(np.uint8)).resize(size,Image.Resampling.NEAREST))
        prediction=np.asarray(Image.fromarray(prediction.astype(np.uint8)).resize(size,Image.Resampling.NEAREST))
        height,width=target.shape
    result=Image.new('RGB',(width*3,height+110),'white')
    valid=(target>=0)&(target<5)
    gt=np.zeros((*target.shape,3),dtype=np.uint8);gt[valid]=PALETTE[target[valid]]
    for i,array in enumerate((rgb,gt,PALETTE[prediction])):result.paste(Image.fromarray(array),(i*width,34))
    draw=ImageDraw.Draw(result)
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    for i,title in enumerate(('Input RGB','Ground truth','Prediction')):draw.text((i*width+8,5),title,font=font,fill='black')
    fmt=lambda x:'N/A' if x is None else f'{x:.2%}'
    draw.text((8,height+40),'mIoU (GT-present classes): '+fmt(metrics['miou']),font=font,fill='black')
    draw.text((8,height+70),'IoU: '+' | '.join(n+'='+fmt(metrics['class_iou'][n]) for n in CLASS_NAMES),font=font,fill='black')
    return result

class ComparisonExporter:
    def __init__(self, image_directory, text_directory, metadata):
        self.images=Path(image_directory);self.text=Path(text_directory)
        self.images.mkdir(parents=True,exist_ok=True);self.text.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.metadata=metadata
    def update(self,batch,targets,prediction,probabilities=None):
        truth=targets.detach().cpu().numpy();pred=prediction.detach().cpu().numpy()
        # Dataset images and GT share the same geometry; validation has no random transforms.
        mean=np.asarray([.485,.456,.406])[:,None,None];std=np.asarray([.229,.224,.225])[:,None,None]
        inputs=batch['image'].detach().cpu().float().numpy()
        for i,(scene,sid) in enumerate(zip(batch['scene'],batch['sample_id'])):
            scene,sid=safe_name(scene),safe_name(sid)
            counter=ConfusionMatrix();counter.update(truth[i],pred[i]);metrics=counter.compute()
            record={'scene':scene,'sample_id':sid,**metrics}
            destination=self.images/scene/sid;destination.mkdir(parents=True,exist_ok=True)
            rgb=np.clip((inputs[i]*std+mean)*255,0,255).round().astype(np.uint8).transpose(1,2,0)
            comparison_image(rgb,truth[i],pred[i],record).save(destination/'comparison.jpg',quality=92)
            write_json(destination/'metrics.json',record)
            write_json(self.text/scene/sid/'metrics.json',record)
            self.rows.append({'scene':scene,'sample_id':sid,'miou':metrics['miou'],**metrics['class_iou']})
    __call__=update
    def finalize(self,metrics=None):
        for directory in [self.images,self.text]:
            with (directory/'per_image_metrics.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['scene','sample_id','miou',*CLASS_NAMES]);writer.writeheader();writer.writerows(self.rows)
            write_json(directory/'index.json',{'metadata':self.metadata,'image_count':len(self.rows),
                       'image_root':str(self.images),'local_text_root':str(self.text),
                       'per_image_files':['comparison.jpg','metrics.json'],
                       'scene_metrics_must_use_accumulated_confusion_matrices':True})
