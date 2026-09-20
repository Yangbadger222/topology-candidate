#!/usr/bin/env python3
"""Post-hoc topology diagnosis from the frozen MaGRoad regression artifacts.

This script never loads a model and never changes inference settings.  It reads
the stage artifacts produced by ``run_graph_regression.py`` and computes the
structural diagnostics requested for A_reconstructed and fixed WildRoad crops.
"""
from __future__ import annotations

import csv, json, math, shutil
from pathlib import Path
from collections import defaultdict

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/topology_diagnosis_A/graph_regression"
OUT = ROOT / "outputs/topology_diagnosis_A"
ITSC = 0.133
TOPO = 0.373
ROAD = 0.839

def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q)) if len(a) else 0.0

def sample_prob(im, x, y):
    h,w=im.shape
    return float(im[min(h-1,max(0,int(round(y)))), min(w-1,max(0,int(round(x))))])

def line_support(im, a, b):
    x0,y0=a; x1,y1=b
    length=float(math.hypot(x1-x0,y1-y0)); n=max(16,int(math.ceil(length/4)))
    xs=np.linspace(x0,x1,n); ys=np.linspace(y0,y1,n)
    vals=np.array([sample_prob(im,x,y) for x,y in zip(xs,ys)],dtype=float)
    low=vals<0.2; longest=cur=0
    for v in low:
        cur=cur+1 if v else 0; longest=max(longest,cur)
    return {"length_px":length,"mean_road_probability":float(vals.mean()),"median_road_probability":float(np.median(vals)),"min_road_probability":float(vals.min()),"p10_road_probability":pct(vals,10),"fraction_road_probability_gt_0.1":float((vals>0.1).mean()),"fraction_road_probability_gt_0.2":float((vals>0.2).mean()),"fraction_road_probability_gt_0.5":float((vals>0.5).mean()),"continuous_low_probability_gap_px":float(longest*length/max(1,n-1))}

def cluster(points, radius):
    p=np.asarray(points,float); n=len(p); seen=np.zeros(n,bool); sizes=[]
    for i in range(n):
        if seen[i]: continue
        stack=[i]; seen[i]=True; count=0
        while stack:
            j=stack.pop(); count+=1
            d=np.sqrt(((p-p[j])**2).sum(1))
            for k in np.where((d<=radius)&(~seen))[0]: seen[k]=True; stack.append(int(k))
        sizes.append(count)
    return sizes

def cluster_ids(points, radius=50):
    p=np.asarray(points,float); labels=np.full(len(p),-1,int); label=0
    for i in range(len(p)):
        if labels[i]>=0: continue
        labels[i]=label; stack=[i]
        while stack:
            j=stack.pop(); d=np.sqrt(((p-p[j])**2).sum(1))
            for k in np.where((d<=radius)&(labels<0))[0]: labels[k]=label; stack.append(int(k))
        label+=1
    return labels

def orientation(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def proper_cross(a,b,c,d):
    # strict crossing, excluding endpoint touches and collinear overlaps
    return orientation(a,b,c)*orientation(a,b,d)<0 and orientation(c,d,a)*orientation(c,d,b)<0

def heat(prob):
    u=np.clip(np.rint(prob*255),0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)

def graph_overlay(rgb,nodes,edges,color=(255,150,0),width=3):
    im=rgb.copy()
    for e in edges:
        a=nodes[e["source"]]; b=nodes[e["target"]]
        cv2.line(im,(round(a[0]),round(a[1])),(round(b[0]),round(b[1])),color,width,cv2.LINE_AA)
    return im

def key_overlay(rgb, nodes, kp):
    im=rgb.copy()
    for x,y in nodes:
        cv2.circle(im,(round(x),round(y)),4,(255,40,40),2,cv2.LINE_AA)
    return im

def cluster_overlay(rgb, points):
    im=rgb.copy(); colors=[(255,40,40),(40,255,40),(40,120,255),(255,220,40),(255,40,255),(40,255,255)]
    for (x,y),label in zip(points,cluster_ids(points)):
        cv2.circle(im,(round(x),round(y)),5,colors[int(label)%len(colors)],-1,cv2.LINE_AA)
    return im

def topo_overlay(rgb,nodes,edges):
    im=rgb.copy()
    for e in edges:
        a,b=nodes[e['source']],nodes[e['target']]; s=float(e['score'])
        cv2.line(im,(round(a[0]),round(a[1])),(round(b[0]),round(b[1])),(int(255*(1-s)),int(255*s),255),2,cv2.LINE_AA)
    return im

def panel(images, labels, out):
    cells=[]
    for im,label in zip(images,labels):
        im=Image.fromarray(im).resize((360,360),Image.Resampling.LANCZOS)
        c=Image.new('RGB',(360,388),'black'); c.paste(im,(0,28)); ImageDraw.Draw(c).text((6,7),label,fill='white',font=ImageFont.load_default()); cells.append(c)
    canvas=Image.new('RGB',(720,1164),'black')
    for i,c in enumerate(cells): canvas.paste(c,((i%2)*360,(i//2)*388))
    canvas.save(out,quality=94)

def analyze_case(case_dir, group, model_label, name, rows, edge_rows, kp_rows, tri_rows, support_rows):
    cg=json.loads((case_dir/'candidate_graph.json').read_text())
    d=json.loads((case_dir/'diagnostics.json').read_text())
    road=np.load(case_dir/'road_prob.npy'); kp=np.load(case_dir/'keypoint_prob.npy')
    rgb=np.array(Image.open(case_dir/'rgb.png').convert('RGB'))
    nodes=[(float(n['x']),float(n['y'])) for n in cg['nodes']]
    # The official extractor's graph points are the nodes used for KNN.  The
    # selected count is retained separately because the original code reports both.
    selected=[p for p in nodes if sample_prob(kp,*p)>ITSC]
    if len(selected)>1:
        sp=np.asarray(selected); distances=np.sqrt(((sp[:,None,:]-sp[None,:,:])**2).sum(-1)); distances[distances==0]=np.inf; nn=distances.min(1)
    else: nn=np.array([],float)
    cand=cg['candidate_edges']; final=json.loads((case_dir/'final_graph.json').read_text()).get('edges',[])
    und={tuple(sorted((int(e['source']),int(e['target'])))):e for e in cand if int(e['source'])!=int(e['target'])}
    g=nx.Graph(); g.add_nodes_from(range(len(nodes))); g.add_edges_from(und)
    deg=np.array([g.degree(i) for i in range(len(nodes))],float)
    lengths=[]; edge_metrics=[]
    for e in cand:
        a,b=nodes[int(e['source'])],nodes[int(e['target'])]
        s=line_support(road,a,b); lengths.append(s['length_px'])
        row={"group":group,"model_label":model_label,"image":name,"source":int(e['source']),"target":int(e['target']),"topology_probability":float(e['score']),"accepted":bool(float(e['score'])>TOPO),"endpoint_keypoint_score":float((sample_prob(kp,*a)+sample_prob(kp,*b))/2),**s}
        edge_metrics.append(row); support_rows.append(row)
    accepted=[e for e in edge_metrics if e['accepted']]
    low=[e for e in edge_metrics if e['mean_road_probability']<0.2 or e['fraction_road_probability_gt_0.2']<0.5]
    tri_c=sum(nx.triangles(g).values())//3
    fg=nx.Graph(); fg.add_nodes_from(range(len(nodes))); fg.add_edges_from((int(e['source']),int(e['target'])) for e in final)
    tri_f=sum(nx.triangles(fg).values())//3
    crossings=0; fe=list(und.values())
    for i,a in enumerate(fe):
        for b in fe[i+1:]:
            if set((a['source'],a['target']))&set((b['source'],b['target'])): continue
            if proper_cross(nodes[a['source']],nodes[a['target']],nodes[b['source']],nodes[b['target']]): crossings+=1
    f_und={tuple(sorted((int(e['source']),int(e['target'])))):e for e in final if int(e['source'])!=int(e['target'])}
    fc=0; fl=list(f_und.values())
    for i,a in enumerate(fl):
        for b in fl[i+1:]:
            if set((a['source'],a['target']))&set((b['source'],b['target'])): continue
            if proper_cross(nodes[a['source']],nodes[a['target']],nodes[b['source']],nodes[b['target']]): fc+=1
    for rad in (30,50,80):
        sizes=cluster(selected,rad); d[f'cluster_{rad}_count']=len(sizes); d[f'cluster_{rad}_largest']=max(sizes,default=0); d[f'cluster_{rad}_mean']=float(np.mean(sizes)) if sizes else 0.0
    d.update({"group":group,"model_label":model_label,"image":name,"selected_keypoint_count_recomputed":len(selected),"node_density":len(nodes)/road.size,"candidate_undirected_edge_count":len(und),"candidate_amplification_directed":len(cand)/max(1,len(nodes)),"candidate_amplification_undirected":len(und)/max(1,len(nodes)),"candidate_mean_degree":float(deg.mean()) if len(deg) else 0.0,"candidate_median_degree":pct(deg,50),"candidate_p90_degree":pct(deg,90),"candidate_max_degree":int(deg.max()) if len(deg) else 0,"candidate_triangle_count":tri_c,"final_triangle_count":tri_f,"triangles_per_candidate_node":tri_c/max(1,len(nodes)),"triangles_per_candidate_edge":tri_c/max(1,len(und)),"candidate_crossing_count":crossings,"final_crossing_count":fc,"candidate_length_p10":pct(lengths,10),"candidate_length_p25":pct(lengths,25),"candidate_length_p50":pct(lengths,50),"candidate_length_p75":pct(lengths,75),"candidate_length_p90":pct(lengths,90),"candidate_length_max":max(lengths,default=0.0),"candidate_lt30":sum(x<30 for x in lengths),"candidate_lt50":sum(x<50 for x in lengths),"candidate_lt100":sum(x<100 for x in lengths),"topo_accept_ratio":len(accepted)/max(1,len(cand)),"low_support_candidate_count":len(low),"low_support_accepted_count":sum(x['accepted'] for x in low),"low_support_accepted_fraction":sum(x['accepted'] for x in low)/max(1,len(low)),"accepted_mean_road_support":float(np.mean([x['mean_road_probability'] for x in accepted])) if accepted else 0.0,"rejected_mean_road_support":float(np.mean([x['mean_road_probability'] for x in edge_metrics if not x['accepted']])) if any(not x['accepted'] for x in edge_metrics) else 0.0})
    d.update({"keypoint_nn_mean":float(nn.mean()) if len(nn) else 0.0,"keypoint_nn_p10":pct(nn,10),"keypoint_nn_p50":pct(nn,50),"keypoint_nn_p90":pct(nn,90)})
    rows.append(d)
    # required node/edge CSVs
    for i,(x,y) in enumerate(nodes):
        if sample_prob(kp,x,y)<=ITSC: continue
        values={}
        yy,xx=np.ogrid[:road.shape[0],:road.shape[1]]
        for rad in (5,10,20): values[f'road_mean_radius_{rad}']=float(road[(xx-x)**2+(yy-y)**2<=rad**2].mean())
        kp_rows.append({"group":group,"model_label":model_label,"image":name,"node":i,"x":x,"y":y,"keypoint_probability":sample_prob(kp,x,y),"road_at_keypoint":sample_prob(road,x,y),**values})
    tri_rows.append({"group":group,"model_label":model_label,"image":name,"candidate_nodes":len(nodes),"candidate_undirected_edges":len(und),"candidate_triangles":tri_c,"candidate_triangles_per_node":tri_c/max(1,len(nodes)),"candidate_triangles_per_edge":tri_c/max(1,len(und)),"final_edges":len(f_und),"final_triangles":tri_f,"final_triangles_per_node":tri_f/max(1,len(nodes)),"candidate_crossings":crossings,"final_crossings":fc})
    # stage images / panel
    od=OUT/group/model_label/name; od.mkdir(parents=True,exist_ok=True)
    Image.fromarray(heat(kp)).save(od/'keypoint_probability_heatmap.png'); Image.fromarray(key_overlay(rgb,nodes,kp)).save(od/'keypoints_overlay.png')
    Image.fromarray(cluster_overlay(rgb,selected)).save(od/'keypoint_clusters.png')
    Image.fromarray(graph_overlay(rgb,nodes,cand,color=(255,210,0),width=2)).save(od/'candidate_graph_overlay.png')
    Image.fromarray(graph_overlay(rgb,nodes,final,color=(0,255,80),width=3)).save(od/'final_graph_overlay.png')
    panel([rgb,heat(road),key_overlay(rgb,nodes,kp),graph_overlay(rgb,nodes,cand,color=(255,210,0),width=2),topo_overlay(rgb,nodes,cand),graph_overlay(rgb,nodes,final,color=(0,255,80),width=3)],['RGB','Road probability','Keypoints','Candidate graph','Topo score graph','Final graph'],od/'pipeline_panel.png')

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def make_plots(support, rows):
    import matplotlib.pyplot as plt
    for group,model in [('xjtlu','A_reconstructed'),('wildroad','A_reconstructed'),('wildroad','WildRoad_baseline')]:
        ss=[r for r in support if r['group']==group and r['model_label']==model]
        if not ss: continue
        fig,ax=plt.subplots(1,3,figsize=(14,4))
        ax[0].scatter([float(r['mean_road_probability']) for r in ss],[float(r['topology_probability']) for r in ss],c=['tab:green' if r['accepted'] else 'tab:red' for r in ss],s=8,alpha=.55); ax[0].set(xlabel='mean road support',ylabel='TopoNet probability')
        ax[1].scatter([float(r['p10_road_probability']) for r in ss],[float(r['topology_probability']) for r in ss],c=['tab:green' if r['accepted'] else 'tab:red' for r in ss],s=8,alpha=.55); ax[1].set(xlabel='P10 road support',ylabel='TopoNet probability')
        ax[2].scatter([float(r['length_px']) for r in ss],[float(r['topology_probability']) for r in ss],c=['tab:green' if r['accepted'] else 'tab:red' for r in ss],s=8,alpha=.55); ax[2].set(xlabel='edge length (px)',ylabel='TopoNet probability')
        fig.suptitle(f'{group} / {model} (green accepted, red rejected)'); fig.tight_layout(); fig.savefig(OUT/'plots'/f'{group}_{model}_toponet_scatter.png',dpi=160); plt.close(fig)
    # XJTLU degree histogram
    xr=[r for r in rows if r['group']=='xjtlu']; fig,ax=plt.subplots(figsize=(7,4)); ax.hist([float(r['candidate_mean_degree']) for r in xr],bins=8,color='tab:orange'); ax.set(xlabel='candidate mean degree per image',ylabel='tile count'); fig.tight_layout(); fig.savefig(OUT/'plots/xjtlu_candidate_degree_histogram.png',dpi=160); plt.close(fig)

def main():
    rows=[]; edge=[]; kp=[]; tri=[]; support=[]
    for group,model,subdir in [('xjtlu','A_reconstructed','G1_controlled/encoder_lora'),('wildroad','A_reconstructed','official_regression/encoder_lora'),('wildroad','WildRoad_baseline','official_regression/baseline')]:
        root=SRC/subdir
        for case in sorted(root.iterdir()):
            if case.is_dir() and (case/'candidate_graph.json').exists(): analyze_case(case,group,model,case.name,rows,edge,kp,tri,support)
    edge.extend(support)
    write_csv(OUT/'csv/per_image_summary.csv',rows); write_csv(OUT/'csv/candidate_edges.csv',edge); write_csv(OUT/'csv/keypoints.csv',kp); write_csv(OUT/'csv/keypoint_road_support.csv',kp); write_csv(OUT/'csv/triangle_stats.csv',tri); write_csv(OUT/'csv/candidate_road_support.csv',support)
    make_plots(support, rows)
    # final edges and all candidate node table are machine-readable aliases
    finals=[]; nodes=[]
    for p in OUT.glob('csv/candidate_edges.csv'): pass
    for group,model,subdir in [('xjtlu','A_reconstructed','G1_controlled/encoder_lora'),('wildroad','A_reconstructed','official_regression/encoder_lora'),('wildroad','WildRoad_baseline','official_regression/baseline')]:
        for case in sorted((SRC/subdir).iterdir()):
            if not case.is_dir() or not (case/'final_graph.json').exists(): continue
            for e in json.loads((case/'final_graph.json').read_text()).get('edges',[]): finals.append({'group':group,'model':model,'image':case.name,**e})
    write_csv(OUT/'csv/final_edges.csv',finals)
    config=json.loads((SRC/'inference_config.json').read_text()); config['resolved_from']=str(SRC/'inference_config.json'); config['checkpoint']='A_reconstructed'; config['checkpoint_sha256']='4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff'; config['no_training']=True; json.dump(config,open(OUT/'resolved_graph_config.json','w'),indent=2)
    shutil.copy2(SRC/'checkpoint_identity.json',OUT/'checkpoint_identity.json'); shutil.copy2(SRC/'baseline_pipeline_equivalence.json',OUT/'baseline_equivalence.json')
    print(json.dumps({'images':len(rows),'edges':len(edge),'keypoints':len(kp),'triangles':len(tri)},indent=2))
if __name__=='__main__': main()
