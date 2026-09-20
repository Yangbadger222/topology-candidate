"""Report real artifacts only; missing/running experiments remain explicitly pending."""
from pathlib import Path
import json,time,html
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]; out=root/'outputs'; report=out/'LOVEDA_LORA_REPORT.md'
def read(p,default=None): return json.loads(p.read_text()) if p.exists() else default
status=read(out/'experiment_status.json',{})
lines=['# LoveDA Road-only LoRA 实验报告','',f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}；流程状态：{status.get("stage","unknown")}','',
'本阶段只训练 road segmentation。没有 skeleton、graph、TopoNet forward/training 或 A* loss。','',
'LoveDA 官方 mapping：road=3；1..7 有效；0 ignore。[来源](https://github.com/Junjue-Wang/LoveDA#dataset-and-contest)。保持原生 1024×1024；RGB float32 0..255，经原 MaGRoad pixel_mean/std 标准化。',
'WildRoad base 用完整模型初始化；新 Q/V LoRA rank=4，B 矩阵为零。官方 ENCODER_LORA 默认仍训练 decoder/toponet，新增独立训练模式严格冻结。',
'原 WildRoad BASE_LR=1e-3、encoder factor=.1；本轮 LoRA/decoder 各 1e-4，AdamW decay=.01。BCE pos_weight=1、Dice weight=1。15 epochs，可选 patience=4；batch=1、FP16、accumulation=4。','',
'## 数据检查','']
s=read(out/'sanity/statistics.json',{})
reference=read(out/'baseline/base_reference.json',{})
lines += [f'WildRoad base: `{reference.get("path","pending")}`；SHA256=`{reference.get("sha256","pending")}`。']
lines += [f'Train：{s.get("images","pending")} 张；road pixels={s.get("road_pixels","pending")}；non-road={s.get("nonroad_pixels","pending")}；ignore={s.get("ignore_pixels","pending")}；road ratio={s.get("road_ratio","pending")}。统计 pos_weight 推荐值={s.get("suggested_pos_weight","pending")}，没有直接采用。',
'20 张固定随机 sanity 图片保存在 sanity/，RGB、semantic labels、road binary；ignore 显示洋红。','',
'## 性能（LoveDA 完整 Val，固定阈值 0.5）','',
'| 模型 | epoch | Val images | IoU | Precision | Recall | F1 | BCE | Dice loss | Total | trainable | peak alloc MiB | peak reserved MiB | time seconds |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
selected={}; summaries={}
for name,folder in [('WildRoad baseline','baseline'),('A: Encoder LoRA','encoder'),('B: LoRA + Decoder','encoder_decoder')]:
    p=out/folder; history=read(p/'validation_history.json',[]); run=read(p/'run_status.json',{}); stats=read(p/'parameter_statistics.json',{})
    if not history:
        lines.append(f'| {name} | pending | | | | | | | | | | | | |'); continue
    if run.get('status')!='complete': name += '（进行中，当前 best）'
    best=max(history,key=lambda r:r['road_iou']); selected[folder]=best; count=0 if folder=='baseline' else sum(x['trainable'] for x in stats.values())
    values=[best['epoch'],best['images'],*[f'{best[k]:.6f}' for k in ('road_iou','road_precision','road_recall','road_f1','bce_loss','dice_loss','loss')],count,f'{best["vram"]["peak_allocated_mib"]:.1f}',f'{best["vram"]["peak_reserved_mib"]:.1f}',f'{run.get("time_seconds",best["seconds_since_start"]):.1f}']
    lines.append('| '+name+' | '+' | '.join(map(str,values))+' |')
    recall=max(history,key=lambda r:r['road_recall']); summaries[folder]={'best_iou':best,'highest_recall':recall,'run_status':run,'parameter_statistics':stats}
    lines += []
lines+=['','每行适配模型选 best Val IoU epoch；另存 best_road_recall.ckpt，不能把不同 epoch 的 IoU/Recall 拼成同一模型表现。所有 validation search threshold 来自 LoveDA Val F1，不由 XJTLU 图片决定。','',
'## 阈值与 checkpoint','']
for folder,best in selected.items():
    history=read(out/folder/'validation_history.json',[]); recall=max(history,key=lambda r:r['road_recall'])
    lines.append(f'- {folder}: best IoU epoch={best["epoch"]}, validation-best F1 threshold={best["road_threshold"]}, best-threshold metrics={best["threshold_curve"][str(best["road_threshold"])]}; highest recall epoch={recall["epoch"]}, recall={recall["road_recall"]:.6f}。')
lines += ['','## 显存与验证','',
'实测 4070 Laptop 8GB：首次无 checkpointing、1024 FP16 backward OOM，峰值 allocated 6877.5 MiB / reserved 7028 MiB，额外分配 384 MiB 失败。','保留 1024，在独立 road-only forward 对 SAM blocks 使用 non-reentrant gradient checkpointing。10-step A：峰值 allocated 3275.6 MiB / reserved 4116 MiB；48 个 LoRA tensors 有梯度，其余无梯度。',
'A 的真实 checkpoint 与 WildRoad 对比：177 encoder、10 decoder、72 TopoNet tensors 全部逐元素不变；24 个 LoRA B tensors 从零更新。真实 B audit 在 smoke_B/parameter_update_audit.json、encoder_decoder/parameter_update_audit.json；未生成则尚未验证。','完整原推理 logits 与新 road-only 初始 logits 的 baseline_equivalence.log 必须 max difference=0。单元测试及合成 A* 不等价于真实任务成功率。','',
'## 固定 XJTLU 外域 before/after（无 GT）','',
'固定 8 张 1024 XJTLU 图只推理，未加入训练、没有 IoU/Recall/Route Success。保存 float32 probability NPY、16-bit probability PNG、binary、overlay/panel。默认 t=.5 与对应 epoch 的 validation-best threshold 分开输出。',
'为了比较一致，下方全部使用 t=.5，概率色标保持 0..1。观察到视觉变化也不能称为有 GT 的目标域 Recall 提升。','']
compare=out/'comparison'; compare.mkdir(exist_ok=True)
files=sorted((out/'baseline/target_domain/baseline').glob('*_t050.jpg'))
links=[]
for f in files:
    panels=[]; labels=[]
    for folder in ('baseline','encoder','encoder_decoder'):
        if folder not in selected: continue
        epoch=selected[folder]['epoch']; directory=out/folder/'target_domain'/('baseline' if folder=='baseline' else f'epoch_{epoch:03d}')
        image=directory/f.name
        if image.exists():
            im=Image.open(image); im.thumbnail((2048,600)); panels.append(im.copy()); labels.append(f'{folder}: epoch {epoch}, t=.5')
    if not panels: continue
    w=max(x.width for x in panels); h=sum(x.height+28 for x in panels); canvas=Image.new('RGB',(w,h)); y=0
    for im,label in zip(panels,labels): ImageDraw.Draw(canvas).text((8,y+8),label,fill='white'); canvas.paste(im,(0,y+28)); y+=im.height+28
    target=compare/f.name; canvas.save(target); lines.append(f'![{f.stem}](comparison/{f.name})'); links.append(f'<p>{html.escape(f.stem)}</p><img style="width:100%" src="comparison/{html.escape(f.name)}">')
lines+=['','## 决策门','']
if 'encoder_decoder' not in selected or summaries['encoder_decoder']['run_status'].get('status')!='complete':
    lines += ['完整实验仍在进行，不能判断域适配是否显著改善。A/B 完成后再比较 Val 与固定 XJTLU panels；不启动 topology。']
else:
    baseline=selected['baseline']; A=selected['encoder']; B=selected['encoder_decoder']
    lines += [f'A 相对 baseline：Val IoU Δ={A["road_iou"]-baseline["road_iou"]:+.6f}、Recall Δ={A["road_recall"]-baseline["road_recall"]:+.6f}。B：IoU Δ={B["road_iou"]-baseline["road_iou"]:+.6f}、Recall Δ={B["road_recall"]-baseline["road_recall"]:+.6f}。',
    '这些是单 seed、Val 用于选择的实验结果，未提供独立 test 或重复 seed 的显著性结论。目标域无 GT，尚不能量化明显道路 Recall。',
    '请审查固定外域 before/after：若明显道路主体连续而仅局部 gap，后续才讨论 centerline/topology；若大路仍漏检，先检查 GSD、zoom、normalization、来源与 decoder/label 定义。此流程在报告处停止，不自动启动 topology。']
lines += ['','## 已知限制与后续','',
'LoveDA road 标签边界/定义可能不覆盖校园全部步行路、停车场或建筑阴影；与 XJTLU GSD/zoom 的匹配尚需测量。B 只监督 road，decoder 共享层更新仍可能间接影响未监督 keypoint；后续 graph 使用前需单独回归验证。',
'A* 当前仅 non-autograd pixel evaluation，允许人工 GT Start/Goal 与 valid-path mask；合成 continuous/gap/alternative/low-probability shortcut/corner 测试。当前不正式报告 Route Success、Stretch、False Shortcut、APLS。']
report.write_text('\n'.join(lines)+'\n'); (out/'experiment_summary.json').write_text(json.dumps(summaries,indent=2)); (out/'review.html').write_text('<meta charset="utf-8"><title>LoveDA LoRA Comparison</title><h1>Fixed XJTLU t=0.5</h1>'+''.join(links))
print(report)
