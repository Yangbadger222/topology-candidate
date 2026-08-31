# overhead-ssl

Frozen foundation-encoder baselines for ordinary overhead/satellite RGB. M1A compares official DINOv3 ViT-L/16 SAT-493M and DINOv2 ViT-S/14 dense patch features. This repository does not contain datasets, labels, geospatial metadata, path planning, or training code.

## Scope and invariants

- Model input is only an RGB tensor `[B, 3, H, W]`; dataset, scene, GSD, coordinates, and provenance never enter the encoder.
- M1A is frozen inference: `eval()`, `requires_grad=False`, `torch.inference_mode()`, FP16 autocast on CUDA, and zero training steps.
- Public data and D3 manifests remain in the source data repository. Do not copy them into Git. Set `OVERHEAD_DATA_ROOT` (or pass `--data-root`) to an external processed-data root.
- Exact 80-image selection is deterministic with seed `20260831`; external target images are never included in PCA fitting.

## Environment

Use Python 3.10+ in an isolated environment. Install PyTorch using the wheel appropriate for the host CUDA driver, then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

DINOv2 loads the official `facebookresearch/dinov2` PyTorch Hub checkpoint. DINOv3 requires the official repository and the access-controlled SAT-493M checkpoint; provide them explicitly:

```bash
export DINOV3_REPO=/path/to/official/dinov3
export DINOV3_SAT493M_WEIGHTS=/external/weights/dinov3-vitl16-pretrain-sat493m.pth
```

If those variables or the accepted download are unavailable, DINOv3 is reported as blocked rather than replaced by another checkpoint.

## M1A commands

```bash
python scripts/run_m1a.py --data-root "$OVERHEAD_DATA_ROOT" \
  --manifest /path/to/rgb_model/data/splits/ssl_train.csv
python scripts/benchmark.py --output outputs/M1A/benchmark.csv
pytest -q
```

The benchmark resizes full scenes to 256x256 and then 512x512, batch size 1. Because DINOv2 ViT-S/14 requires dimensions divisible by 14, its normalized tensor is right/bottom padded (not cropped) to 266x266/518x518; both content and encoder tensor sizes are recorded. DINOv3 stays at 256x256/512x512. CUDA timing includes synchronization and warmup, mean/median latency, peak allocated/reserved memory, parameter count, and actual dense feature shape. A 512 OOM is recorded and does not trigger repeated retries. No 1024 benchmark is performed in M1A.

Outputs under `outputs/M1A/` are intentionally ignored by Git. The external campus target belongs in `assets/external_target/`; if absent, the run reports `external target image not provided`.
External target images are user-provided and excluded from repository version control. They must never be used for training or PCA fitting.

## Provenance

See `configs/models/` for checkpoint source, identifier, and official normalization. Record host package versions and benchmark results in `outputs/M1A/report.md` outside Git if desired. **No model training has been performed in M1A.**
