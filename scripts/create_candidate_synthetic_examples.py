"""Reproducible synthetic proposal cases, including misleading parallel / blank maps."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
from PIL import Image,ImageDraw
from connection_candidates.geometry import Context,make_candidate
from connection_candidates.local import generate_local
from corridor_proposal import pixel_astar
from candidate_generator_v2 import parser
from graph_utils import write_json


def main():
    root=Path('outputs/connection_candidates_v2/synthetic');root.mkdir(parents=True,exist_ok=True);summary=[]
    for name in ('straight','parallel','t_junction','obstacle','blank'):
        p=np.zeros((120,180),float);p[57:64,10:170]=.9
        if name=='parallel':p[:]=0;p[37:44,:]=.9;p[77:84,:]=.9
        if name=='t_junction':p[60:110,87:94]=.9
        if name=='obstacle':p[40:80,75:105]=0;p[30:40,50:130]=.9;p[30:65,50:60]=.9;p[30:65,120:130]=.9
        if name=='blank':p[:]=0
        g=nx.Graph();nodes=[(10,60),(50,60),(130,60),(170,60)];edges=[(0,1),(2,3)]
        if name=='parallel':nodes=[(10,40),(80,40),(80,80),(170,80)]
        if name=='t_junction':nodes=[(10,60),(170,60),(90,85),(90,110)]
        for i,(x,y) in enumerate(nodes):g.add_node(i,x=x,y=y)
        for u,v in edges:g.add_edge(u,v,geometry=[nodes[u],nodes[v]],length=np.linalg.norm(np.array(nodes[u])-nodes[v]))
        ctx=Context(g,p);c=parser().parse_args(['--graph','x','--probability','p','--output','o']);local,_=generate_local(ctx,c);a=nodes[1];b=nodes[2];path,expanded=pixel_astar(1+8*(1-p)**2+4*(p<.2),a,b,3);row=make_candidate(ctx,'component_corridor',{'component_id':0},{'component_id':1},[a]+path+[b],{});row['status']='rejected_by_hard_gate' if row['probability_features']['mean']<.15 or row['probability_features']['p10']<.02 else 'proposed';im=Image.fromarray(np.uint8(np.stack([p,p,p],-1)*255));d=ImageDraw.Draw(im)
        for u,v in edges:d.line([nodes[u],nodes[v]],fill='cyan',width=3)
        d.line([tuple(q) for q in row['geometry']['polyline_xy']],fill='magenta',width=2);im.resize((720,480)).save(root/(name+'.png'));write_json(root/(name+'.json'),{'corridor':row,'local':local});summary.append({'case':name,'mean_probability':row['probability_features']['mean'],'p10':row['probability_features']['p10'],'detour':row['geometry_features']['detour_ratio'],'heuristic':row['heuristic_score'],'corridor_status':row['status'],'endpoint_edge_proposed':sum(r['type']=='endpoint_edge' and r['status']=='proposed' for r in local)})
    write_json(root/'summary.json',summary)
if __name__=='__main__':main()
