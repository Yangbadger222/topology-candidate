"""Cache frozen deployed MaGRoad tile features without changing inference sources."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def sha(path: Path) -> str:
    """Stream SHA256 without loading a checkpoint into memory."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main() -> None:
    """Generate provenance, complete tile cache, and float32 fused probabilities."""
    p=argparse.ArgumentParser();p.add_argument('--magroad-root',type=Path,required=True)
    for name in ('image','checkpoint','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inference-size',type=int,nargs=2,required=True,metavar=('W','H'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    for name in ('magroad_root','image','checkpoint','config','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise FileExistsError('Use a new output directory')
    sys.path.insert(0,str(a.magroad_root.resolve()));os.chdir(a.magroad_root)
    import numpy as np
    import torch
    from PIL import Image
    from utils import load_config
    from model import MaGRoad
    config=load_config(str(a.config));assert config.MODEL_NAME=='MaGRoad' and config.PATCH_SIZE==512
    module=ast.parse((a.magroad_root/'inferencer.py').read_text())
    selected=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='get_patch_info_rectangular']
    namespace={'np':np};exec(compile(ast.Module(body=selected,type_ignores=[]),'patch_layout','exec'),namespace)
    w,h=a.inference_size
    if min(w,h)<512:raise ValueError('Image must fit an unpadded 512 patch')
    infos=namespace['get_patch_info_rectangular'](0,h,w,config.SAMPLE_MARGIN,512,config.INFER_PATCHES_PER_EDGE)
    image=Image.open(a.image).convert('RGB');original=list(image.size);rgb=np.asarray(image.resize((w,h),Image.Resampling.LANCZOS)).copy()
    net=MaGRoad(config);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['state_dict'],strict=True);del state
    net.eval().to(a.device);net.requires_grad_(False)
    fused=torch.zeros(2,h,w);count=torch.zeros(h,w);tiles=[];t=time.perf_counter()
    with torch.inference_mode():
        for i,(_, (x,y),(xx,yy)) in enumerate(infos):
            batch=torch.from_numpy(rgb[y:yy,x:xx].copy()).float().unsqueeze(0).to(a.device)
            feature,logits,scores=net.infer_masks_and_img_features(batch)
            if tuple(feature.shape)!=(1,256,32,32) or tuple(logits.shape)!=(1,2,512,512):raise ValueError('Unexpected feature shape')
            path=a.output/f'tile_{i:04d}.pt'
            torch.save({'encoder_feature':feature[0].cpu().half(),'mask_logits':logits[0].cpu().half(),
                        'metadata':{'origin_xy':[x,y],'image_size':[512,512],'stride':16}},path)
            fused[:,y:yy,x:xx]+=scores[0].cpu().permute(2,0,1);count[y:yy,x:xx]+=1
            tiles.append({'file':path.name,'origin_xy':[x,y],'bytes':path.stat().st_size,'sha256':sha(path)})
            if i%20==0:print(f'cache {i+1}/{len(infos)}',flush=True)
    if torch.any(count==0):raise ValueError('Patch layout leaves uncovered pixels')
    fused/=count
    np.save(a.output/'keypoint_probability.npy',fused[0].numpy());np.save(a.output/'road_probability.npy',fused[1].numpy())
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=a.magroad_root,capture_output=True,text=True)
    manifest={'coordinate_order':'x,y','units':'inference_pixel','image_size':[w,h],'original_image_size':original,
        'stride':16,'channels':256,'encoder_shape':[256,32,32],'dtype':'float16','tiles':tiles,
        'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'config':dict(config),'config_sha256':sha(a.config),
        'image_sha256':sha(a.image),'model_source_sha256':sha(a.magroad_root/'model.py'),
        'magroad_git_commit':commit.stdout.strip() if commit.returncode==0 else None,'git_available':commit.returncode==0,
        'elapsed_seconds':time.perf_counter()-t,'total_bytes':sum(x.stat().st_size for x in a.output.glob('*') if x.is_file()),
        'sampling_policy':'best tile margin, native sampler; no feature map fusion'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ('elapsed_seconds','total_bytes','encoder_shape')},indent=2))
if __name__=='__main__':main()
