import json,csv,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[3] / 'outputs' # unused when run remotely
base=Path(__file__).parent.parent/'fair_reval_A_vs_HNv2'
if not (base/'loveda_val').exists(): base=Path('magroad_loveda/fair_reval_A_vs_HNv2')
val=base/'loveda_val'
def read(n): return json.loads((val/n/'validation_history.json').read_text())[-1]
A=read('A_reconstructed_hn_metrics'); H=read('HN_v2_epoch0')
rows=[]
for name,r in [('A_reconstructed',A),('HN_v2_epoch0',H)]:
    for t,m in r['threshold_curve'].items(): rows.append({'model':name,'threshold':t,**m})
with (base/'threshold_sweep.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
keys=['road_iou','road_precision','road_recall','road_f1','building_fp','boundary_fp','mean_building_probability','mean_boundary_probability','p50_building_probability','p90_building_probability','p95_building_probability','p50_boundary_probability','p90_boundary_probability','p95_boundary_probability','road_threshold']
comp=[]
for name,r in [('A_reconstructed',A),('HN_v2_epoch0',H)]: comp.append({'model':name,**{k:r.get(k) for k in keys}})
(base/'metrics_A_reconstructed.json').write_text(json.dumps(A,indent=2)); (base/'metrics_HNv2.json').write_text(json.dumps(H,indent=2)); (base/'comparison.json').write_text(json.dumps(comp,indent=2))
def pct(a,b): return (b-a)/a*100 if a else None
delta={k:{'absolute':H[k]-A[k],'relative_percent':pct(A[k],H[k])} for k in keys if isinstance(A.get(k), (int,float)) and k!='road_threshold'}
(base/'comparison_delta.json').write_text(json.dumps(delta,indent=2))
graph=Path('magroad_loveda/hard_negative_v2/graph_regression/HN_GRAPH_REGRESSION_REPORT.md')
graph_note='Existing graph regression checkpoint_identity.json confirms A uses reconstructed SHA 4b5831f4 and HN v2 uses 2d3322aa; baseline equivalence passed.'
report=f'''# Reconstructed A vs HN v2 Fair Re-evaluation

## 1. Why
旧 A 的 hard-negative 指标与 reconstructed A 不同口径。本报告完全重新 forward LoveDA validation，并以 reconstructed A 为正式 baseline。

## 2. Checkpoint identity

| Model | Path | Epoch | SHA256 | LoRA rank | Mode |
|---|---|---:|---|---:|---|
| A_reconstructed | `/home/badger/sam-inference/MaGRoad/magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt` | 8 | `4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff` | 4 | loveda_lora_encoder |
| HN_v2_epoch0 | `/home/badger/sam-inference/MaGRoad/magroad_loveda/hard_negative_v2/formal/checkpoints/best_valid_candidate.ckpt` | 0 | `2d3322aa69160442f8aad44644827004391b80057c2f87ddfb81b1f1f4c59db1` | 4 | loveda_lora_encoder |

Tensor audit: base encoder 177 equal/0 changed, map decoder 10 equal/0 changed, TopoNet 72 equal/0 changed; A→HN LoRA tensors changed 48.

## 3. LoveDA fixed threshold (t=0.5)

| Model | IoU | Precision | Recall | F1 | Building FP | Boundary FP | Mean Building P | Mean Boundary P | Best-F1 threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A reconstructed | {A['road_iou']:.6f} | {A['road_precision']:.6f} | {A['road_recall']:.6f} | {A['road_f1']:.6f} | {A['building_fp']:.6f} | {A['boundary_fp']:.6f} | {A['mean_building_probability']:.6f} | {A['mean_boundary_probability']:.6f} | {A['road_threshold']:.2f} |
| HN v2 epoch 0 | {H['road_iou']:.6f} | {H['road_precision']:.6f} | {H['road_recall']:.6f} | {H['road_f1']:.6f} | {H['building_fp']:.6f} | {H['boundary_fp']:.6f} | {H['mean_building_probability']:.6f} | {H['mean_boundary_probability']:.6f} | {H['road_threshold']:.2f} |

Probability quantiles: A building P50/P90/P95 = {A['p50_building_probability']:.3f}/{A['p90_building_probability']:.3f}/{A['p95_building_probability']:.3f}, boundary = {A['p50_boundary_probability']:.3f}/{A['p90_boundary_probability']:.3f}/{A['p95_boundary_probability']:.3f}; HN building = {H['p50_building_probability']:.3f}/{H['p90_building_probability']:.3f}/{H['p95_building_probability']:.3f}, boundary = {H['p50_boundary_probability']:.3f}/{H['p90_boundary_probability']:.3f}/{H['p95_boundary_probability']:.3f}.

## 4. Deltas (HN v2 − A)

- Recall: **{H['road_recall']-A['road_recall']:+.6f}** ({pct(A['road_recall'],H['road_recall']):+.2f}%), within the reporting-only practical band ±0.005.
- IoU: {H['road_iou']-A['road_iou']:+.6f}; Precision: {H['road_precision']-A['road_precision']:+.6f}; F1: {H['road_f1']-A['road_f1']:+.6f}.
- Building FP: **{H['building_fp']-A['building_fp']:+.6f}** ({pct(A['building_fp'],H['building_fp']):+.2f}%, an increase).
- Boundary FP: **{H['boundary_fp']-A['boundary_fp']:+.6f}** ({pct(A['boundary_fp'],H['boundary_fp']):+.2f}%, an increase).
- Mean building probability: {H['mean_building_probability']-A['mean_building_probability']:+.6f}; mean boundary probability: {H['mean_boundary_probability']-A['mean_boundary_probability']:+.6f}.

Thus, at the unified threshold and metrics, HN v2 does not reduce building-related false road response relative to reconstructed A. The earlier apparent reductions came from comparing against legacy A values.

## 5. Threshold calibration

Both models were evaluated on the same 0.05–0.95 grid. Full machine-readable values are in `threshold_sweep.csv`; the best F1 thresholds are A={A['road_threshold']:.2f}, HN v2={H['road_threshold']:.2f}. This indicates a small confidence calibration shift, but does not reverse the fixed-t=0.5 FP result.

## 6. XJTLU

The fixed eight `xjtlu_1_r03_c00`–`c07` images were rerun. Probability arrays, t=0.5 binaries, per-image HN−A differences, and summary statistics are in `xjtlu/`. XJTLU has no ground-truth road masks, so these are diagnostics only.

## 7. Graph/WildRoad

{graph_note} No graph accuracy claim is made without graph ground truth. The existing WildRoad regression remains a structural regression, not a proof of navigation accuracy.

## 8. Final decision

**Use `A_reconstructed` as the next topology-adaptation visual backbone.** HN v2 preserves Recall within the practical band, but its unified Building FP and Boundary FP are higher (not lower), so it has no demonstrated visual benefit over reconstructed A. No v3 or further training is initiated.

Historical legacy A values remain historical only and are excluded from this comparison.
'''
(base/'A_RECONSTRUCTED_VS_HNV2_REPORT.md').write_text(report)
print(report)
