#!/usr/bin/env python3
"""M1A frozen inference benchmark; no optimizer or training steps."""
import argparse, csv, time
from pathlib import Path
import torch
from torchvision.transforms import v2
from overhead_ssl.models import Dinov2SmallEncoder, Dinov3SatelliteEncoder
from overhead_ssl.data import pad_to_patch_multiple

def transform(size, satellite=False):
    mean, std = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)) if satellite else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])

def run(name, ctor, size, device):
    model = ctor(device=device)
    x = transform(size, "dinov3" in name)(torch.rand(3, 1024, 1024))
    x = pad_to_patch_multiple(x, model.patch_size).unsqueeze(0).to(device)
    for _ in range(3):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16): model.encode(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(10):
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16): out=model.encode(x)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t)*1000)
    return [name, size, x.shape[-1], sum(p.numel() for p in model.parameters()), "float16", device, sum(times)/len(times), sorted(times)[len(times)//2], torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20, tuple(out.features.shape)]

parser=argparse.ArgumentParser(); parser.add_argument("--output", default="outputs/M1A/benchmark.csv"); args=parser.parse_args()
if not torch.cuda.is_available(): raise SystemExit("CUDA is required for M1A benchmark")
rows=[]
for size in (256,512):
    for name, ctor in (("dinov2_small", Dinov2SmallEncoder), ("dinov3_sat", Dinov3SatelliteEncoder)):
        try: rows.append(run(name, ctor, size, "cuda")); print(rows[-1])
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower(): print(f"OOM,{name},{size}")
            elif name == "dinov3_sat" and "SAT493M requires" in message: print(f"BLOCKED_OFFICIAL_WEIGHTS,{name},{size},{message}")
            else: raise
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output,"w",newline="") as f:
    csv.writer(f).writerows([["model","content_resize","encoder_tensor_size","parameter_count","dtype","device","mean_ms","median_ms","peak_allocated_mib","peak_reserved_mib","feature_shape"]]+rows)
