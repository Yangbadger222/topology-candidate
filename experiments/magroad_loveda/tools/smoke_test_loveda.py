"""Real WildRoad model CPU contract tests; no synthetic training PASS claims."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from utils import load_config
from road_model import RoadOnlyMaGRoad,load_base,is_lora,category
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=load_config(a.config)
model=RoadOnlyMaGRoad(cfg); groups=model.optimizer_groups(); names=[n for n,p in model.named_parameters() if p.requires_grad]
assert all(is_lora(n) or (cfg.TRAIN_MAP_DECODER and n.startswith('map_decoder.')) for n in names)
# B initialized zero -> base predictions exactly unchanged before adaptation.
assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if '.linear_b_' in n)
assert sum(p.numel() for g in groups for p in g['params'])==sum(p.numel() for p in model.parameters() if p.requires_grad)
path=Path(cfg.OUTPUT_DIR)/'contract_adapter.pt'; path.parent.mkdir(parents=True,exist_ok=True); torch.save(model.adapter_state(),path); model.load_adapter(path)
print(json.dumps({'contract_tests':'passed','trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'groups':[g['name'] for g in groups]}))
