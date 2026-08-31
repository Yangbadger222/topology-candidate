#!/usr/bin/env python3
"""Extract a small M1A feature set and make per-image/shared PCA PNGs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from overhead_ssl.data import load_manifest, select_samples, resolve_path, load_rgb, pad_to_patch_multiple
from overhead_ssl.models import Dinov2SmallEncoder, Dinov3SatelliteEncoder
from overhead_ssl.pca import SharedPCA

def make_transform(size, satellite):
    mean, std = ((0.430,0.411,0.296),(0.213,0.156,0.143)) if satellite else ((0.485,0.456,0.406),(0.229,0.224,0.225))
    return v2.Compose([v2.ToImage(), v2.Resize((size,size), antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean,std)])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--manifest",required=True); p.add_argument("--model",choices=["dinov2_small","dinov3_sat"],required=True); p.add_argument("--size",type=int,default=256); p.add_argument("--output-dir",default="outputs/M1A"); args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda": raise SystemExit("M1A feature extraction requires CUDA")
    satellite=args.model == "dinov3_sat"
    model=(Dinov3SatelliteEncoder if satellite else Dinov2SmallEncoder)(device=device)
    selected=select_samples(load_manifest(args.manifest)); tfm=make_transform(args.size,satellite)
    rows=[]; originals=[]; features=[]
    for row in selected.itertuples():
        image=load_rgb(resolve_path(args.data_root,row.path)); x=pad_to_patch_multiple(tfm(image), model.patch_size).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16): out=model.encode(x)
        feat=out.features[0].float().cpu().numpy(); features.append(feat); originals.append(image)
        rows.append(row)
    pca=SharedPCA(3,False).fit(np.concatenate([f.reshape(-1,f.shape[-1]) for f in features]))
    root=Path(args.output_dir)/args.model/str(args.size); root.mkdir(parents=True,exist_ok=True)
    for row,image,feat in zip(rows,originals,features):
        image.save(root/f"{row.sample_id}_original.jpg")
        Image.fromarray(pca.transform_grid(feat)).save(root/f"{row.sample_id}_pca_shared.jpg")
        Image.fromarray(SharedPCA(3,False).fit(feat.reshape(-1,feat.shape[-1])).transform_grid(feat)).save(root/f"{row.sample_id}_pca_per_image.jpg")
    print(f"extracted {len(features)} images; feature shape={features[0].shape}; output={root}")

if __name__ == "__main__": main()
