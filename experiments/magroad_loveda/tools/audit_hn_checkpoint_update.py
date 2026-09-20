#!/usr/bin/env python3
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from road_model import canonical,category

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
p=argparse.ArgumentParser(); p.add_argument('--initial',required=True); p.add_argument('--trained',required=True); p.add_argument('--output',required=True); a=p.parse_args()
initial_raw=torch.load(a.initial,map_location='cpu',weights_only=False); trained_raw=torch.load(a.trained,map_location='cpu',weights_only=False)
initial=canonical(initial_raw['state_dict']); trained=canonical(trained_raw['state_dict']); counts={g:{'equal':0,'changed':0} for g in ('encoder_base','lora','map_decoder','toponet','other')}
for name,value in initial.items():
    assert name in trained,name; counts[category(name)]['equal' if torch.equal(value,trained[name]) else 'changed']+=1
assert counts['lora']['changed']>0
assert all(counts[g]['changed']==0 for g in ('encoder_base','map_decoder','toponet','other'))
ref=trained_raw.get('initial_checkpoint_reference',{}); assert ref.get('sha256')==sha(a.initial)
result={'initial_checkpoint':str(Path(a.initial).resolve()),'initial_sha256':sha(a.initial),'trained_checkpoint':str(Path(a.trained).resolve()),'comparison':counts,'audit':'passed'}
Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
