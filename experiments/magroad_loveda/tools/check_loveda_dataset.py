import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, random
import numpy as np
from PIL import Image, ImageDraw
from dataset import discover
p=argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--domain', default='all'); p.add_argument('--output', default='outputs/loveda_sanity'); p.add_argument('--split', default='train'); p.add_argument('--train-rgb-root'); a=p.parse_args()
rows=discover(a.root,a.split,a.domain,a.train_rgb_root); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
road=other=ignored=0; hist=np.zeros(256,dtype=np.int64)
for row in rows:
    im=Image.open(row['image']); lab=np.array(Image.open(row['label']))
    assert im.size==(1024,1024) and lab.shape==(1024,1024), row
    hist+=np.bincount(lab.ravel(),minlength=256)
    assert set(np.unique(lab)).issubset(set(range(8))), f'Unexpected label IDs: {row}'
    road+=int((lab==3).sum()); other+=int(((lab>=1)&(lab<=7)&(lab!=3)).sum()); ignored+=int((lab==0).sum())
selection=random.Random(3407).sample(rows,min(20,len(rows)))
palette=np.array([[255,0,255],[80,80,80],[255,80,80],[255,255,255],[30,100,255],[180,140,70],[30,160,30],[180,230,50]],dtype=np.uint8)
for row in selection:
    rgb=Image.open(row['image']).convert('RGB'); lab=np.array(Image.open(row['label'])); binary=np.zeros((*lab.shape,3),dtype=np.uint8); binary[lab==3]=255; binary[lab==0]=[255,0,255]
    panel=Image.new('RGB',(3072,1056)); panel.paste(rgb,(0,32)); panel.paste(Image.fromarray(palette[lab]),(1024,32)); panel.paste(Image.fromarray(binary),(2048,32)); ImageDraw.Draw(panel).text((10,10),row['id']+' | RGB | Semantic IDs (road white) | Road binary (ignore magenta)',fill='white'); panel.save(out/(row['id']+'.jpg'))
stats={'images':len(rows),'road_pixels':road,'nonroad_pixels':other,'ignore_pixels':ignored,'road_ratio':road/(road+other),'suggested_pos_weight':other/max(road,1),'histogram':hist[:8].tolist(),'selection':selection}
(out/'statistics.json').write_text(json.dumps(stats,indent=2)); print(json.dumps({k:v for k,v in stats.items() if k!='selection'},indent=2))
