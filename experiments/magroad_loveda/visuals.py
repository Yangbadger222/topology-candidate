from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch


def save_visual(rgb,prob,path,threshold=.5,gt=None,valid=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rgb=np.asarray(rgb,dtype=np.uint8); prob=np.asarray(prob,dtype=np.float32)
    np.save(path.with_suffix('.npy'),prob)
    Image.fromarray(np.round(prob*65535).astype(np.uint16)).save(path.with_name(path.stem+'_probability.png'))
    pred=prob>=threshold; Image.fromarray(pred.astype(np.uint8)*255).save(path.with_name(path.stem+'_binary.png'))
    overlay=rgb.copy(); overlay[pred]=(rgb[pred]*.55+np.array([255,50,30])*.45).astype(np.uint8)
    heat=np.stack([prob,np.maximum(0,1-2*np.abs(prob-.5)),1-prob],-1)*255
    panels=[rgb]; names=['RGB']
    if gt is not None:
        target=np.repeat((np.asarray(gt)*255).astype(np.uint8)[...,None],3,-1)
        if valid is not None: target[~np.asarray(valid,dtype=bool)]=[255,0,255]
        panels.append(target); names.append('GT Road (ignore magenta)')
    panels.extend([heat.astype(np.uint8),np.repeat(pred[...,None].astype(np.uint8)*255,3,-1),overlay]); names.extend(['Probability (blue=0 red=1)',f'Prediction t={threshold:.2f}','Overlay'])
    h,w=prob.shape; out=Image.new('RGB',(w*len(panels),h+32))
    for i,(panel,name) in enumerate(zip(panels,names)):
        out.paste(Image.fromarray(panel),(i*w,32)); ImageDraw.Draw(out).text((i*w+8,10),name,fill='white')
    out.save(path.with_suffix('.jpg'))


@torch.inference_mode()
def target_inference(model,input_dir,output,threshold=.5,best_threshold=None,limit=8):
    files=sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))[:limit]
    if not files: raise ValueError(f'No target images in {input_dir}')
    model.eval(); dev=next(model.parameters()).device
    for file in files:
        rgb=np.array(Image.open(file).convert('RGB'))
        if rgb.shape!=(1024,1024,3): raise ValueError(f'Target must be 1024; no resize: {file}')
        with torch.autocast('cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
            prob=model.forward_road_only(torch.from_numpy(rgb.copy()).float().unsqueeze(0).to(dev)).float().sigmoid()[0].cpu().numpy()
        save_visual(rgb,prob,Path(output)/(file.stem+'_t050'),threshold)
        if best_threshold is not None and abs(best_threshold-threshold)>1e-6: save_visual(rgb,prob,Path(output)/(file.stem+'_valbest'),best_threshold)
