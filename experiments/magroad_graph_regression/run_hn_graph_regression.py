#!/usr/bin/env python3
"""G1 graph regression for WildRoad baseline, LoveDA A, and Hard-Negative LoRA."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_graph_regression as gr

def dump(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2))
def load(path):
    raw=torch.load(path,map_location='cpu',weights_only=False); return raw,gr.canonical(raw.get('state_dict',raw))

def verify(output):
    identities={}; base_raw,base=load(gr.BASE_CHECKPOINT); assert gr.sha256(gr.BASE_CHECKPOINT)==gr.BASE_SHA256
    a_path=Path('magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt'); hn_path=Path(gr.MODELS['hard_negative']['checkpoint'])
    a_raw,a=load(a_path); hn_raw,hn=load(hn_path)
    for name,path,raw,state in [('baseline',gr.BASE_CHECKPOINT,base_raw,base),('encoder_lora',a_path,a_raw,a),('hard_negative',hn_path,hn_raw,hn)]:
        comparison=gr.tensor_group_comparison(base,state)
        identities[name]={'path':str(path.resolve()),'sha256':gr.sha256(path),'size_bytes':path.stat().st_size,'epoch':raw.get('epoch'),'train_mode':raw.get('experiment_config',{}).get('TRAIN_MODE','wildroad_baseline'),'lora_rank':raw.get('experiment_config',{}).get('LORA_RANK',0),'comparison_to_base':comparison}
    init_ref=hn_raw.get('initial_checkpoint_reference',{}); assert init_ref.get('sha256')==gr.sha256(a_path)
    relative=gr.tensor_group_comparison(a,hn)
    assert relative['encoder_base']['changed']==relative['map_decoder']['changed']==relative['toponet']['changed']==0
    assert any(not torch.equal(a[k],hn[k]) for k in a if gr.state_category(k)=='lora')
    selected=hn_raw['validation_history'][-1]; assert selected['recall_guardrail_passed']
    identities['hard_negative']['initial_checkpoint_reference']=init_ref; identities['hard_negative']['comparison_to_A']=relative; identities['hard_negative']['selected_validation']=selected
    dump(output/'checkpoint_identity.json',identities); return identities

def aggregate(rows):
    keys=['road_positive_ratio','keypoint_selected_count','graph_node_count','graph_edge_count','connected_component_count','largest_component_ratio','mean_accepted_topo_score']
    return {m:{k:float(np.mean([r[k] for r in rows if r['model']==m and r['experiment']==gr.G1])) for k in keys} for m in gr.MODEL_ORDER}

def report(output,identities,rows,equivalence):
    agg=aggregate(rows); labels={m:gr.MODELS[m]['label'] for m in gr.MODEL_ORDER}
    lines=['# Hard-Negative LoRA Graph Regression','', 'This is inference-only G1 controlled regression. All models use the unchanged WildRoad ROAD/ITSC/TOPO thresholds and the original candidate/TopoNet pipeline. XJTLU has no graph GT; structural changes are diagnostics, not accuracy.', '',f'Baseline equivalence: `{equivalence}`.','', '| Model | Road ratio | Keypoints | Nodes | Edges | Components | Largest ratio | Mean topo |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in gr.MODEL_ORDER:
        x=agg[m]; lines.append(f"| {labels[m]} | {x['road_positive_ratio']:.4f} | {x['keypoint_selected_count']:.1f} | {x['graph_node_count']:.1f} | {x['graph_edge_count']:.1f} | {x['connected_component_count']:.1f} | {x['largest_component_ratio']:.3f} | {x['mean_accepted_topo_score']:.3f} |")
    lines += ['', '## Per-image G1', '', '| Image | Model | Road ratio | Keypoints | Nodes | Edges | Components | Largest ratio |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in rows if r['experiment']==gr.G1),key=lambda x:(x['image_name'],gr.MODEL_ORDER.index(x['model']))): lines.append(f"| {r['image_name']} | {labels[r['model']]} | {r['road_positive_ratio']:.4f} | {r['keypoint_selected_count']} | {r['graph_node_count']} | {r['graph_edge_count']} | {r['connected_component_count']} | {r['largest_component_ratio']:.3f} |")
    lines += ['', '## WildRoad deterministic crops', '', 'Three unchanged crops are a catastrophic-forgetting smoke test, not an official benchmark.', '']
    official=[r for r in rows if r['experiment']==gr.OFFICIAL]
    for m in gr.MODEL_ORDER:
        sel=[r for r in official if r['model']==m]; lines.append(f"- {labels[m]}: road ratio={np.mean([r['road_positive_ratio'] for r in sel]):.4f}, keypoints={np.mean([r['keypoint_selected_count'] for r in sel]):.1f}, edges={np.mean([r['graph_edge_count'] for r in sel]):.1f}, components={np.mean([r['connected_component_count'] for r in sel]):.1f}.")
    lines += ['', '## Visual panels', '']
    for p in sorted((output/'comparison'/gr.G1).glob('*_graph_only.jpg')): lines += [f'![{p.stem}](comparison/{gr.G1}/{p.name})','']
    lines += ['## Interpretation','','The report records whether HN reduces keypoints, edges, components, and visible dense triangles relative to A. A decrease alone is not evidence of higher graph accuracy. No TopoNet, keypoint, graph, or A* training was performed.']
    (output/'HN_GRAPH_REGRESSION_REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--hn-checkpoint',required=True); p.add_argument('--hn-config',required=True); p.add_argument('--output-root',required=True); a=p.parse_args(); output=Path(a.output_root); output.mkdir(parents=True,exist_ok=True)
    gr.MODELS={
      'baseline':{'label':'WildRoad baseline','checkpoint':gr.BASE_CHECKPOINT,'config':Path('config/toponet_vitb_1024_wild_road.yaml'),'road_threshold':.1,'train_mode':'wildroad_baseline'},
      'encoder_lora':{'label':'A: Encoder LoRA','checkpoint':Path('magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt'),'config':Path('magroad_loveda/config/loveda/remote_encoder.yaml'),'road_threshold':.3,'train_mode':'loveda_lora_encoder'},
      'hard_negative':{'label':'HN: A + hard negatives','checkpoint':Path(a.hn_checkpoint),'config':Path(a.hn_config),'road_threshold':.3,'train_mode':'loveda_lora_encoder'},
    }; gr.MODEL_ORDER=['baseline','encoder_lora','hard_negative']
    identities=verify(output); inferencer,graph_extraction,MaGRoad,load_config=gr.import_magroad(); device=torch.device('cuda:0'); inferencer.args.device=device
    official_config=load_config(str(gr.MODELS['baseline']['config'])); dump(output/'inference_config.json',{'experiment':'G1 controlled only','ROAD_THRESHOLD':float(official_config.ROAD_THRESHOLD),'ITSC_THRESHOLD':float(official_config.ITSC_THRESHOLD),'TOPO_THRESHOLD':float(official_config.TOPO_THRESHOLD),'models':gr.MODEL_ORDER})
    base_net,base_cfg=gr.load_model('baseline',device,MaGRoad,load_config); equivalence=gr.baseline_equivalence(base_net,base_cfg,gr.XJTLU_DIR/'xjtlu_1_r03_c00.png',output,device,inferencer,graph_extraction); del base_net; torch.cuda.empty_cache()
    ximgs=sorted(gr.XJTLU_DIR.glob('xjtlu_1_r03_c*.png')); assert len(ximgs)==8
    official=[]
    for spec in gr.OFFICIAL_CROPS: official.append(gr.crop_official(spec,output/'official_inputs')[0])
    dump(output/'official_inputs'/'crops.json',gr.OFFICIAL_CROPS); rows=[]
    for model_name in gr.MODEL_ORDER:
        net,cfg=gr.load_model(model_name,device,MaGRoad,load_config)
        for image in ximgs: rows.append(gr.run_one(net,cfg,image,gr.G1,model_name,float(official_config.ROAD_THRESHOLD),output,device,inferencer,graph_extraction))
        for image in official: rows.append(gr.run_one(net,cfg,image,gr.OFFICIAL,model_name,float(official_config.ROAD_THRESHOLD),output,device,inferencer,graph_extraction))
        del net; torch.cuda.empty_cache()
    for image in ximgs: gr.create_panels(output,gr.G1,image.stem)
    for image in official: gr.create_panels(output,gr.OFFICIAL,image.stem)
    dump(output/'all_metrics.json',rows)
    with (output/'hn_graph_regression_metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=gr.CSV_COLUMNS,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    report(output,identities,rows,equivalence); dump(output/'run_status.json',{'status':'complete','finished_unix':time.time(),'rows':len(rows)})
if __name__=='__main__': main()
