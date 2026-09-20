"""Compare real trained full checkpoint against original WildRoad tensors."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from road_model import canonical,category
p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--trained',required=True); p.add_argument('--mode',choices=['encoder','encoder_decoder'],required=True); p.add_argument('--output',required=True); a=p.parse_args()
base=canonical(torch.load(a.base,map_location='cpu',weights_only=False)['state_dict']); trained=canonical(torch.load(a.trained,map_location='cpu',weights_only=False)['state_dict']); counts={g:{'equal':0,'changed':0} for g in ('encoder_base','map_decoder','toponet','other')}
for name,value in base.items():
    if name=='mask_criterion.pos_weight': continue
    assert name in trained,name
    equal=torch.equal(value,trained[name]); counts[category(name)]['equal' if equal else 'changed']+=1
assert all(counts[g]['changed']==0 for g in ('encoder_base','toponet','other'))
assert counts['map_decoder']['changed']==0 if a.mode=='encoder' else counts['map_decoder']['changed']>0
lora_b=[v for n,v in trained.items() if '.qkv.linear_b_' in n]; assert lora_b and any(v.count_nonzero() for v in lora_b)
result={'mode':a.mode,'comparison':counts,'nonzero_lora_b':sum(int(v.count_nonzero()>0) for v in lora_b),'audit':'passed'}; Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
