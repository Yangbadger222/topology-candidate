#!/usr/bin/env python3
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from dataset import discover,hard_negative_targets,LOVEDA_CLASSES,ROAD_CLASS_ID,BUILDING_CLASS_ID,IGNORE_CLASS_ID

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--train-rgb-root',required=True); p.add_argument('--output',required=True); p.add_argument('--count',type=int,default=20); p.add_argument('--width',type=int,default=5); p.add_argument('--building-weight',type=float,default=2.); p.add_argument('--boundary-weight',type=float,default=4.); a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
rows=discover(a.root,'train','all',a.train_rgb_root); rng=random.Random(3407); chosen=rng.sample(rows,a.count)
palette=np.array([[255,0,255],[80,80,80],[235,180,60],[255,255,255],[30,100,230],[180,140,90],[20,130,50],[180,230,80]],dtype=np.uint8)
stats={'class_mapping':LOVEDA_CLASSES,'IGNORE_CLASS_ID':IGNORE_CLASS_ID,'BUILDING_CLASS_ID':BUILDING_CLASS_ID,'ROAD_CLASS_ID':ROAD_CLASS_ID,'boundary_width':a.width,'building_weight':a.building_weight,'boundary_weight':a.boundary_weight,'samples':[],'road_boundary_overlap_pixels':0}
thumbs=[]
for row in chosen:
    rgb=np.array(Image.open(row['image']).convert('RGB')); sem=np.array(Image.open(row['label'])); t=hard_negative_targets(sem,a.width,building_weight=a.building_weight,boundary_weight=a.boundary_weight)
    road=t['road_mask'].numpy().astype(bool); building=t['building_mask'].numpy(); boundary=t['building_boundary'].numpy(); weight=t['loss_weight'].numpy()
    overlap=int(np.count_nonzero(road & boundary)); stats['road_boundary_overlap_pixels']+=overlap
    scale=max(a.building_weight,a.boundary_weight,1.)
    panels=[rgb,palette[sem],np.repeat(road[...,None],3,-1)*255,np.repeat(building[...,None],3,-1)*np.array([255,190,30]),np.repeat(boundary[...,None],3,-1)*np.array([255,40,40]),np.stack([weight/scale,weight/scale,weight/scale],-1)*255]
    names=['RGB','Semantic','Road GT','Building GT','Inner boundary','Loss weight']
    canvas=Image.new('RGB',(6*256,280),'black'); draw=ImageDraw.Draw(canvas)
    for i,(panel,name) in enumerate(zip(panels,names)):
        canvas.paste(Image.fromarray(np.asarray(panel,dtype=np.uint8)).resize((256,256)),(i*256,24)); draw.text((i*256+5,6),name,fill='white')
    canvas.save(out/(row['id']+'.jpg'),quality=92); thumbs.append(canvas.resize((768,140)))
    stats['samples'].append({'id':row['id'],'road_pixels':int(road.sum()),'building_pixels':int(building.sum()),'boundary_pixels':int(boundary.sum()),'road_boundary_overlap_pixels':overlap})
if stats['road_boundary_overlap_pixels']!=0: raise RuntimeError('Road positive overlaps hard-negative boundary')
sheet=Image.new('RGB',(768,140*len(thumbs)),'black')
for i,img in enumerate(thumbs): sheet.paste(img,(0,i*140))
sheet.save(out/'sanity_20.jpg',quality=92); (out/'statistics.json').write_text(json.dumps(stats,indent=2)); print(json.dumps(stats,indent=2))
