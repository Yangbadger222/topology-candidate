#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np, torch
from PIL import Image
from utils import load_config
from road_model import RoadOnlyMaGRoad,canonical,sha256

def infer(model,rgb):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        return model.forward_road_only(torch.from_numpy(rgb.copy()).float().unsqueeze(0).cuda()).float().sigmoid()[0].cpu().numpy()

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--a-checkpoint',required=True); p.add_argument('--hn-checkpoint',required=True); p.add_argument('--input-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True); cfg=load_config(a.config); model=RoadOnlyMaGRoad(cfg).cuda().eval()
files=sorted(Path(a.input_dir).glob('xjtlu_1_r03_c*.png')); assert len(files)==8
rgbs={f.stem:np.array(Image.open(f).convert('RGB')) for f in files}
def load(path):
    raw=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(canonical(raw['state_dict']),strict=True); return raw
load(a.a_checkpoint); pa={n:infer(model,r) for n,r in rgbs.items()}; load(a.hn_checkpoint); ph={n:infer(model,r) for n,r in rgbs.items()}
rows=[]
for n,rgb in rgbs.items():
    d=ph[n]-pa[n]
    for label,prob in [('A_reconstructed',pa[n]),('HN_v2',ph[n])]:
        Image.fromarray(np.rint(prob*65535).astype(np.uint16)).save(out/f'{n}_{label}_probability_u16.png')
        Image.fromarray(((prob>=.5).astype(np.uint8)*255)).save(out/f'{n}_{label}_binary_t050.png')
    Image.fromarray(np.clip((d+1)*127.5,0,255).astype(np.uint8)).save(out/f'{n}_HN_minus_A_difference.png')
    rows.append({'image':n,'mean_delta':float(d.mean()),'mean_abs_delta':float(np.abs(d).mean()),'p10_delta':float(np.percentile(d,10)),'p50_delta':float(np.percentile(d,50)),'p90_delta':float(np.percentile(d,90)),'a_mean_probability':float(pa[n].mean()),'hn_mean_probability':float(ph[n].mean())})
(out/'xjtlu_probability_summary.json').write_text(json.dumps({'checkpoints':{'A_reconstructed':{'path':str(Path(a.a_checkpoint).resolve()),'sha256':sha256(a.a_checkpoint)},'HN_v2_epoch0':{'path':str(Path(a.hn_checkpoint).resolve()),'sha256':sha256(a.hn_checkpoint)}},'images':rows,'note':'XJTLU has no GT; diagnostics only.'},indent=2)); print(json.dumps(rows,indent=2))
