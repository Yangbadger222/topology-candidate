#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from utils import load_config
from road_model import RoadOnlyMaGRoad,canonical,sha256

def infer(model,rgb):
    device=next(model.parameters()).device
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
        return model.forward_road_only(torch.from_numpy(rgb.copy()).float().unsqueeze(0).to(device)).float().sigmoid()[0].cpu().numpy()

def heat(prob): return (np.stack([prob,np.maximum(0,1-2*np.abs(prob-.5)),1-prob],-1)*255).astype(np.uint8)
def overlay(rgb,prob):
    out=rgb.copy(); mask=prob>=.5; out[mask]=(out[mask]*.55+np.array([255,50,30])*.45).astype(np.uint8); return out
def cell(array,label,size=300):
    image=Image.fromarray(np.asarray(array,dtype=np.uint8)).resize((size,size)); panel=Image.new('RGB',(size,size+26),'black'); panel.paste(image,(0,26)); ImageDraw.Draw(panel).text((6,7),label,fill='white'); return panel
def load_state(model,path):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    if 'state_dict' in raw:
        model.load_state_dict(canonical(raw['state_dict']),strict=True)
    elif 'adapter_state' in raw:
        result=model.load_state_dict(raw['adapter_state'],strict=False)
        if result.unexpected_keys or any('linear_' in key for key in result.missing_keys): raise ValueError('Adapter does not contain the complete LoRA state')
    else: raise ValueError(f'Unsupported checkpoint format: {path}')
    model.eval(); return raw

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--v1-checkpoint',required=True); p.add_argument('--v2-checkpoint',required=True); p.add_argument('--input-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
cfg=load_config(a.config); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
files=sorted(Path(a.input_dir).glob('xjtlu_1_r03_c*.png')); assert len(files)==8
rgbs={f.stem:np.array(Image.open(f).convert('RGB')) for f in files}
model=RoadOnlyMaGRoad(cfg).cuda().eval()
probs={'A':{name:infer(model,rgb) for name,rgb in rgbs.items()}}
v1raw=load_state(model,a.v1_checkpoint); probs['v1']={name:infer(model,rgb) for name,rgb in rgbs.items()}
v2raw=load_state(model,a.v2_checkpoint); probs['v2']={name:infer(model,rgb) for name,rgb in rgbs.items()}
def threshold(raw,fallback):
    history=raw.get('validation_history')
    if history: return float(history[-1]['road_threshold'])
    return float(raw.get('metrics',{}).get('road_threshold',fallback))
thresholds={'A':.30,'v1':threshold(v1raw,.20),'v2':threshold(v2raw,.20)}
rows=[]
for name,rgb in rgbs.items():
    pa,p1,p2=probs['A'][name],probs['v1'][name],probs['v2'][name]; delta=p2-pa
    arrays={
      'RGB':rgb,'A_probability':heat(pa),'v1_probability':heat(p1),'v2_probability':heat(p2),
      'A_binary_t0.5':np.repeat((pa>=.5)[...,None],3,-1)*255,'v1_binary_t0.5':np.repeat((p1>=.5)[...,None],3,-1)*255,'v2_binary_t0.5':np.repeat((p2>=.5)[...,None],3,-1)*255,
      'A_overlay':overlay(rgb,pa),'v2_overlay':overlay(rgb,p2),
    }
    for label,array in arrays.items(): Image.fromarray(np.asarray(array,dtype=np.uint8)).save(out/f'{name}_{label}.png')
    panels=[cell(array,label.replace('_',' ')) for label,array in arrays.items()]
    canvas=Image.new('RGB',(3*300,3*326),'black')
    for i,panel in enumerate(panels): canvas.paste(panel,((i%3)*300,(i//3)*326))
    canvas.save(out/f'{name}_comparison.jpg',quality=94)
    scale=max(float(np.percentile(np.abs(delta),99)),1e-4); diff=np.zeros((*delta.shape,3),dtype=np.uint8); diff[...,0]=np.clip(delta/scale,0,1)*255; diff[...,2]=np.clip(-delta/scale,0,1)*255
    Image.fromarray(diff).save(out/f'{name}_probability_difference.png')
    Image.fromarray(np.rint(pa*65535).astype(np.uint16)).save(out/f'{name}_A_probability_u16.png'); Image.fromarray(np.rint(p1*65535).astype(np.uint16)).save(out/f'{name}_v1_probability_u16.png'); Image.fromarray(np.rint(p2*65535).astype(np.uint16)).save(out/f'{name}_v2_probability_u16.png')
    rows.append({'image':name,'mean_delta':float(delta.mean()),'mean_abs_delta':float(np.abs(delta).mean()),'p10_delta':float(np.percentile(delta,10)),'p50_delta':float(np.percentile(delta,50)),'p90_delta':float(np.percentile(delta,90)),'a_mean_probability':float(pa.mean()),'v1_mean_probability':float(p1.mean()),'v2_mean_probability':float(p2.mean())})
summary={'checkpoints':{'v1':{'path':str(Path(a.v1_checkpoint).resolve()),'sha256':sha256(a.v1_checkpoint)},'v2':{'path':str(Path(a.v2_checkpoint).resolve()),'sha256':sha256(a.v2_checkpoint)}},'val_best_thresholds':thresholds,'note':'XJTLU has no GT; probability changes and visual observations are diagnostics only.','images':rows}
(out/'xjtlu_probability_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
