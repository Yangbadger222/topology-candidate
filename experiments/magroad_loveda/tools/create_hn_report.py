#!/usr/bin/env python3
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def pct(new,old): return 100*(new/old-1) if old else 0

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--graph-root',required=True); p.add_argument('--selected-checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args()
root=Path(a.root); graph=Path(a.graph_root); selected=Path(a.selected_checkpoint)
A=read(root/'a_reference_val'/'validation_history.json')[-1]; raw=torch.load(selected,map_location='cpu',weights_only=False); H=raw['validation_history'][-1]
history=read(root/'checkpoints_run'/'validation_history.json'); audit=read(root/'checkpoints_run'/'parameter_update_audit.json'); xj=read(root/'xjtlu'/'xjtlu_probability_summary.json')
rows=list(csv.DictReader(open(graph/'hn_graph_regression_metrics.csv')))
for r in rows:
    for k,v in list(r.items()):
        try:r[k]=float(v)
        except:pass
def means(model,experiment='G1_controlled'):
    s=[r for r in rows if r['model']==model and r['experiment']==experiment]
    return {k:float(np.mean([r[k] for r in s])) for k in ['road_positive_ratio','keypoint_selected_count','graph_node_count','graph_edge_count','connected_component_count','largest_component_ratio','mean_accepted_topo_score']}
ag=means('encoder_lora'); hg=means('hard_negative'); bg=means('baseline')
focus={}
for name in ('xjtlu_1_r03_c05','xjtlu_1_r03_c06','xjtlu_1_r03_c07'):
    focus[name]={m:next(r for r in rows if r['model']==m and r['experiment']=='G1_controlled' and r['image_name']==name) for m in ('encoder_lora','hard_negative')}
guard=float(H['road_recall_guardrail']); recall_drop=H['road_recall']-A['road_recall']; building_change=pct(H['building_fp'],A['building_fp']); boundary_change=pct(H['boundary_fp'],A['boundary_fp'])
case='A' if H['recall_guardrail_passed'] and H['building_fp']<A['building_fp'] and H['boundary_fp']<A['boundary_fp'] and hg['keypoint_selected_count']<ag['keypoint_selected_count'] and hg['graph_edge_count']<ag['graph_edge_count'] else ('B' if H['recall_guardrail_passed'] and H['building_fp']<A['building_fp'] and H['boundary_fp']<A['boundary_fp'] else 'C')
lines=['# MaGRoad Hard-Negative LoRA v1 报告','',
'## 1. Purpose','',
'本实验从 LoveDA Encoder-LoRA A 继续，只训练 rank-4 Q/V LoRA。Building interior 与 5 px building inner-boundary 在 road BCE 中分别使用 2×/4×负样本权重；Dice 保持普通形式。目标是降低建筑与房檐样结构的 road 概率，同时保持道路 Recall。','',
'## 2. Starting checkpoint','',
f'- A: `{raw["initial_checkpoint_reference"]["path"]}`; epoch={raw["initial_checkpoint_reference"]["epoch"]}; SHA256=`{raw["initial_checkpoint_reference"]["sha256"]}`; LoRA rank=4。',
f'- Selected HN: `{selected}`; epoch={raw.get("epoch")}; SHA256=`{sha(selected)}`。','',
'## 3. Hard-negative definition','',
'LoveDA mapping: 0=ignore, 1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agriculture。Boundary 由 `building & ~erode(building, width=5)` 生成，只位于 building 内部。赋权优先级保证 ignore=0、road positive=1、boundary=4、building=2、ordinary background=1；20 张 sanity 的 road/boundary overlap 为 0。','',
'![Hard-negative sanity](sanity/sanity_20.jpg)','',
'## 4. Training config','',
'1024×1024，batch=1，FP16，gradient accumulation=4，gradient checkpointing，AdamW，LoRA LR=5e-5，weight decay=0.01，3 epochs。Map Decoder、keypoint head、TopoNet 与 SAM base encoder 全部冻结；TopoNet 不 forward。','',
'Gradient/update audit: '+f'`{audit["audit"]}`；相对 A，LoRA changed={audit["comparison"]["lora"]["changed"]}，base encoder / decoder / TopoNet changed={audit["comparison"]["encoder_base"]["changed"]}/{audit["comparison"]["map_decoder"]["changed"]}/{audit["comparison"]["toponet"]["changed"]}。','',
'## 5. LoveDA Results','',
'### Fixed threshold t=0.5','',
'| Model | IoU | Precision | Recall | F1 | Building FP | Boundary FP | Mean Building P | Mean Boundary P |','|---|---:|---:|---:|---:|---:|---:|---:|---:|',
f'| A | {A["road_iou"]:.6f} | {A["road_precision"]:.6f} | {A["road_recall"]:.6f} | {A["road_f1"]:.6f} | {A["building_fp"]:.6f} | {A["boundary_fp"]:.6f} | {A["mean_building_probability"]:.6f} | {A["mean_boundary_probability"]:.6f} |',
f'| HN | {H["road_iou"]:.6f} | {H["road_precision"]:.6f} | {H["road_recall"]:.6f} | {H["road_f1"]:.6f} | {H["building_fp"]:.6f} | {H["boundary_fp"]:.6f} | {H["mean_building_probability"]:.6f} | {H["mean_boundary_probability"]:.6f} |','',
f'Building FP change={building_change:+.1f}%；Boundary FP change={boundary_change:+.1f}%；Recall change={recall_drop:+.6f}。Guardrail={guard:.6f}，passed={H["recall_guardrail_passed"]}。','',
'### Val-calibrated threshold','',
f'A best-F1 threshold={A["road_threshold"]:.2f}；HN best-F1 threshold={H["road_threshold"]:.2f}。阈值均只由 LoveDA Val 选择，未用 XJTLU 调参。','',
'### Three-epoch history','',
'| Epoch | IoU | Recall | Building FP | Boundary FP | Mean Boundary P | Guardrail |','|---:|---:|---:|---:|---:|---:|---|']
for r in history: lines.append(f'| {r["epoch"]} | {r["road_iou"]:.6f} | {r["road_recall"]:.6f} | {r["building_fp"]:.6f} | {r["boundary_fp"]:.6f} | {r["mean_boundary_probability"]:.6f} | {r["recall_guardrail_passed"]} |')
lines += ['', '## 6. XJTLU Visual Regression','', '固定 8 张 XJTLU 图只作视觉诊断；没有 road/building GT，因此不报告 IoU、Recall 或 Building FP。红色差异表示 HN 提高概率，蓝色表示降低概率；high-change crops 不能称为真实 FP。','']
for x in xj['images']:
    name=x['image']; lines += [f'### {name}', '',f'Mean ΔP={x["mean_delta"]:+.6f}，mean |ΔP|={x["mean_abs_delta"]:.6f}，p99 |ΔP|={x["p99_abs_delta"]:.6f}。','',f'![{name} comparison](xjtlu/{name}_comparison.jpg)','',f'![{name} probability difference](xjtlu/{name}_probability_difference.png)','']
lines += ['## 7. Graph Regression — G1 controlled','', '所有模型使用 WildRoad 原始 ROAD/ITSC/TOPO threshold、NMS、neighbor radius、candidate generation 与 TopoNet。XJTLU 没有 graph GT；边数下降不能自动解释为 accuracy 提高。','',
'| Model | Road ratio | Keypoints | Nodes | Edges | Components | Largest ratio | Mean topo |','|---|---:|---:|---:|---:|---:|---:|---:|']
for label,m in [('Baseline',bg),('A',ag),('HN',hg)]: lines.append(f'| {label} | {m["road_positive_ratio"]:.4f} | {m["keypoint_selected_count"]:.1f} | {m["graph_node_count"]:.1f} | {m["graph_edge_count"]:.1f} | {m["connected_component_count"]:.1f} | {m["largest_component_ratio"]:.3f} | {m["mean_accepted_topo_score"]:.3f} |')
lines += ['', '### Focus tiles c05–c07','', '| Image | Model | Keypoints | Nodes | Edges | Components |','|---|---|---:|---:|---:|---:|']
for name,d in focus.items():
    for model,label in [('encoder_lora','A'),('hard_negative','HN')]:
        r=d[model]; lines.append(f'| {name} | {label} | {int(r["keypoint_selected_count"])} | {int(r["graph_node_count"])} | {int(r["graph_edge_count"])} | {int(r["connected_component_count"])} |')
lines += ['', 'Graph panels are under `graph_regression/comparison/G1_controlled/`; WildRoad deterministic crops are under `graph_regression/comparison/official_regression/`.','',
'## 8. WildRoad Regression','', 'The three deterministic WildRoad crops are a catastrophic-forgetting smoke test, not an official benchmark. Full rows are in `graph_regression/hn_graph_regression_metrics.csv`.','',
'## 9. Findings','',
f'**Q1 — Hard negative 是否降低建筑区域 road probability？** Building FP {building_change:+.1f}%，Boundary FP {boundary_change:+.1f}%；mean building/boundary P 的变化见主表。','',
f'**Q2 — Road Recall 是否保持？** Recall 从 {A["road_recall"]:.6f} 变为 {H["road_recall"]:.6f}（Δ={recall_drop:+.6f}），guardrail {guard:.6f}，结果={H["recall_guardrail_passed"]}。','',
f'**Q3 — keypoint explosion 是否自然缓解？** G1 八图均值从 {ag["keypoint_selected_count"]:.1f} 变为 {hg["keypoint_selected_count"]:.1f}；c05–c07 逐图见上表。这是结构诊断，不是 GT accuracy。','',
f'**Q4 — false densification 是否减少？** G1 edges 均值从 {ag["graph_edge_count"]:.1f} 变为 {hg["graph_edge_count"]:.1f}，components 从 {ag["connected_component_count"]:.1f} 变为 {hg["connected_component_count"]:.1f}。需结合面板人工判断，不宣称 APLS 或 route success。','',
'## 10. Decision Gate','',f'本轮自动归类：**Case {case}**。','']
if case=='A': lines.append('Hard-negative visual adaptation 在本实验标准下有效；下一阶段可由人工决定是否进入 keypoint / TopoNet adaptation。')
elif case=='B': lines.append('Visual false positive 改善且 Recall 保持，但 topology bottleneck 仍独立存在；下一阶段应由人工决定 keypoint + TopoNet adaptation。')
else: lines.append('Road Recall guardrail 或 hard-negative 改善条件未满足；优先降低 boundary/building 权重，不进入 topology。')
lines += ['', '本流程在此停止；未启动 decoder、keypoint、TopoNet、graph、clDice 或 A* 训练。']
Path(a.output).write_text('\n'.join(lines)+'\n')
