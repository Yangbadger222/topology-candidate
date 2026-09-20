import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from utils import load_config
from road_model import RoadOnlyMaGRoad,canonical
from model import MaGRoad
from dataset import LoveDARoad
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=load_config(a.config)
adapt=RoadOnlyMaGRoad(cfg).cuda().eval()
original_cfg=load_config(a.config); original_cfg.ENCODER_LORA=False; original_cfg.MAGROAD_INIT_ONLY=True; original_cfg.BCE_POS_WEIGHT=10.
original=MaGRoad(original_cfg).cuda().eval(); original.load_state_dict(canonical(torch.load(cfg.PRETRAINED_MAGROAD_CKPT,map_location='cpu',weights_only=False)['state_dict']),strict=True)
batch=LoveDARoad(cfg.LOVEDA_ROOT,'val',cfg.LOVEDA_DOMAIN)[0]['rgb'].unsqueeze(0).cuda()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
 road=adapt.forward_road_only(batch); features,logits,scores=original.infer_masks_and_img_features(batch)
 difference=float((road-logits[:,1]).abs().max())
assert difference==0.,difference
print(json.dumps({'baseline_equivalence':'passed','max_abs_logit_difference':difference,'input_shape':list(batch.shape)}))
