import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from utils import load_config
from road_model import RoadOnlyMaGRoad
from visuals import target_inference
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint'); p.add_argument('--input_dir',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--threshold-json'); a=p.parse_args()
cfg=load_config(a.config); model=RoadOnlyMaGRoad(cfg)
if a.checkpoint:
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if 'adapter_state' in data: model.load_adapter(a.checkpoint)
    else:
        if data['base_reference']['sha256']!=model.base_reference['sha256']: raise ValueError('Base mismatch')
        model.load_state_dict(data['state_dict'],strict=True)
best=json.loads(Path(a.threshold_json).read_text())['road_threshold'] if a.threshold_json else None
model.cuda(); target_inference(model,a.input_dir,a.output_dir,.5,best)
