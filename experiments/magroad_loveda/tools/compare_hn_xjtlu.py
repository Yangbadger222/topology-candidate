#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from utils import load_config
from road_model import RoadOnlyMaGRoad,canonical,sha256

def infer(model,rgb):
    dev=next(model.parameters()).device
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
        return model.forward_road_only(torch.from_numpy(rgb.copy()).float().unsqueeze(0).to(dev)).float().sigmoid()[0].cpu().numpy()

def heat(prob):
    return np.stack([prob,np.maximum(0,1-2*np.abs(prob-.5)),1-prob],-1)*255

def overlay(rgb,prob,t):
    out=rgb.copy(); m=prob>=t; out[m]=(out[m]*.55+np.array([255,50,30])*.45).astype(np.uint8); return out

def cell(array,label,size=320):
    img=Image.fromarray(np.asarray(array,dtype=np.uint8)).resize((size,size)); out=Image.new('RGB',(size,size+26),'black'); out.paste(img,(0,26)); ImageDraw.Draw(out).text((6,7),label,fill='white'); return out

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--hn-checkpoint',required=True); p.add_argument('--input-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
cfg=load_config(a.config); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
model=RoadOnlyMaGRoad(cfg).cuda().eval(); files=sorted(Path(a.input_dir).glob('xjtlu_1_r03_c*.png')); assert len(files)==8
results=[]
for f in files:
    rgb=np.array(Image.open(f).convert('RGB')); pa=infer(model,rgb)
    raw=torch.load(a.hn_checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(canonical(raw['state_dict']),strict=True); model.eval(); ph=infer(model,rgb)
    # Restore A before the next image.
    model.load_initial_checkpoint(cfg.INIT_CHECKPOINT); model.eval()
    delta=ph-pa; t=float(raw['validation_history'][-1]['road_threshold'])
    panels=[cell(rgb,'RGB'),cell(heat(pa),'A probability'),cell(heat(ph),'HN probability'),cell(np.repeat((pa>=.5)[...,None],3,-1)*255,'A binary t=0.5'),cell(np.repeat((ph>=.5)[...,None],3,-1)*255,'HN binary t=0.5'),cell(overlay(rgb,pa,.5),'A overlay'),cell(overlay(rgb,ph,.5),'HN overlay')]
    canvas=Image.new('RGB',(4*320,2*346),'black')
    for i,panel in enumerate(panels): canvas.paste(panel,((i%4)*320,(i//4)*346))
    canvas.save(out/(f.stem+'_comparison.jpg'),quality=94)
    # Red means HN increased road probability; blue means it decreased.
    scale=max(float(np.percentile(np.abs(delta),99)),1e-4); vis=np.zeros((*delta.shape,3),dtype=np.uint8); vis[...,0]=np.clip(delta/scale,0,1)*255; vis[...,2]=np.clip(-delta/scale,0,1)*255
    Image.fromarray(vis).save(out/(f.stem+'_probability_difference.png'))
    Image.fromarray(np.rint(pa*65535).astype(np.uint16)).save(out/(f.stem+'_A_probability.png')); Image.fromarray(np.rint(ph*65535).astype(np.uint16)).save(out/(f.stem+'_HN_probability.png'))
    Image.fromarray((ph>=t).astype(np.uint8)*255).save(out/(f.stem+f'_HN_valbest_t{t:.2f}.png'))
    # Save three 256px crops with the largest mean |delta|, labeled as high-change only.
    candidates=[]
    for y in range(0,769,128):
        for x in range(0,769,128): candidates.append((float(np.mean(np.abs(delta[y:y+256,x:x+256]))),x,y))
    for rank,(score,x,y) in enumerate(sorted(candidates,reverse=True)[:3],1):
        trip=np.concatenate([rgb[y:y+256,x:x+256],overlay(rgb,pa,.5)[y:y+256,x:x+256],overlay(rgb,ph,.5)[y:y+256,x:x+256]],axis=1)
        Image.fromarray(trip).save(out/(f.stem+f'_high_change_{rank}_x{x}_y{y}.jpg'),quality=94)
    results.append({'image':f.stem,'a_mean_probability':float(pa.mean()),'hn_mean_probability':float(ph.mean()),'mean_delta':float(delta.mean()),'mean_abs_delta':float(np.abs(delta).mean()),'p99_abs_delta':float(np.percentile(np.abs(delta),99)),'hn_val_threshold':t})
(out/'xjtlu_probability_summary.json').write_text(json.dumps({'hn_checkpoint':str(Path(a.hn_checkpoint).resolve()),'hn_sha256':sha256(a.hn_checkpoint),'note':'XJTLU has no GT; changes are visual diagnostics only. High-change crops are not labeled false positives.','images':results},indent=2))
