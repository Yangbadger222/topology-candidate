#!/usr/bin/env python3
"""Build M1A contact sheets without altering source imagery."""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

def read(path): return Image.open(path).convert("RGB")
def sheet(rows, root, output):
    cells=[]
    for row in rows.itertuples():
        prefix=root/f"{row.sample_id}"
        cells.append((row, read(prefix.with_name(prefix.name+"_original.jpg")), read(prefix.with_name(prefix.name+"_pca_shared.jpg"))))
    width,height=192,192; canvas=Image.new("RGB",(width*3,24+height*len(cells)),"white"); d=ImageDraw.Draw(canvas)
    for c,label in enumerate(("Original","DINOv3 blocked","DINOv2")): d.text((c*width+4,4),label,fill="black")
    for i,(row,original,pca) in enumerate(cells):
        y=24+i*height; canvas.paste(original.resize((width,height)),(0,y)); canvas.paste(Image.new("RGB",(width,height),"white"),(width,y)); canvas.paste(pca.resize((width,height)),(width*2,y))
    output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output,quality=90)

p=argparse.ArgumentParser(); p.add_argument("--samples",default="outputs/M1A/sample_selection.csv"); p.add_argument("--dinov2-root",default="outputs/M1A/dinov2_small/256"); p.add_argument("--output-dir",default="outputs/M1A/comparisons"); args=p.parse_args()
samples=pd.read_csv(args.samples); root=Path(args.dinov2_root); output=Path(args.output_dir)
sheet(samples,root,output/"side_by_side_256.jpg")
for dataset,scene in (("LoveDA","urban"),("LoveDA","rural"),("NAIP","campus_like"),("NAIP","wooded_roads")):
    sheet(samples[(samples.dataset==dataset)&(samples.domain_or_scene==scene)],root,output/f"{dataset.lower()}_{scene}_256.jpg")
