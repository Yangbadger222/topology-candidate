"""Compare graph statistics and user-selected A* pairs using edge projections."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import json
import networkx as nx
import numpy as np
from shapely.geometry import LineString,Point
from shapely.ops import substring
from graph_utils import load_graph,stats,xy,write_json,background,overlay


def project(g: nx.Graph,q: list,max_distance: float) -> tuple[int,dict]:
    """Project to nearest edge and insert a temporary node with split polyline."""
    pt=Point(q);candidates=[(LineString(d['geometry']).distance(pt),u,v,d) for u,v,d in g.edges(data=True)]
    if not candidates:raise ValueError('Graph has no edges')
    dist,u,v,data=min(candidates,key=lambda t:t[0])
    if dist>max_distance:raise ValueError(f'Projection distance {dist:.2f} exceeds limit')
    line=LineString(data['geometry']);t=line.project(pt);pos=line.interpolate(t);qxy=np.array(pos.coords[0])
    # Geometry orientation may differ from the undirected edge iterator.
    if np.linalg.norm(np.array(line.coords[0])-xy(g,u))>np.linalg.norm(np.array(line.coords[-1])-xy(g,u)):u,v=v,u
    info={'input_xy':q,'projected_xy':qxy.tolist(),'distance':dist,'edge':[u,v]}
    if t<1e-6:return u,info
    if line.length-t<1e-6:return v,info
    n=max(g.nodes,default=-1)+1;g.add_node(n,x=float(qxy[0]),y=float(qxy[1]));g.remove_edge(u,v)
    for a,b,part in [(u,n,substring(line,0,t)),(n,v,substring(line,t,line.length))]:
        g.add_edge(a,b,**{**data,'geometry':list(map(list,part.coords)),'length':float(part.length)})
    return n,info


def route(g: nx.Graph,pair: dict,max_projection: float,uncertainty_lambda: float=0) -> tuple[dict,nx.Graph,list]:
    """A* with geometric heuristic and length cost; uncertainty penalty opt-in."""
    h=copy.deepcopy(g);result={'success':False};lines=[]
    try:
        s,si=project(h,pair['start'],max_projection);t,ti=project(h,pair['goal'],max_projection)
        components={n:i for i,cs in enumerate(nx.connected_components(h)) for n in cs}
        result.update(start_projection=si,goal_projection=ti,connected_component={'start':components[s],'goal':components[t]})
        def cost(a: int,b: int,d: dict) -> float:return d['length']*(1+uncertainty_lambda*(1-d.get('confidence',1)))
        path=nx.astar_path(h,s,t,heuristic=lambda a,b:float(np.linalg.norm(xy(h,a)-xy(h,b))),weight=cost)
        result.update(success=True,path_nodes=path,path_length=sum(h.edges[a,b]['length'] for a,b in zip(path,path[1:])))
        lines=[(h.edges[a,b]['geometry'],'#ff00ff') for a,b in zip(path,path[1:])]
    except (nx.NetworkXNoPath,ValueError) as e:result['reason']=str(e)
    return result,h,lines


def main() -> None:
    """Evaluate genuine user pairs or write an empty, documented template."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--refined',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=Path);p.add_argument('--image',type=Path);p.add_argument('--max-projection-distance',type=float,default=40,help='inference pixels');p.add_argument('--uncertainty-lambda',type=float,default=0)
    c=p.parse_args()
    if c.uncertainty_lambda<0 or c.max_projection_distance<=0:raise ValueError('Nonnegative uncertainty penalty and positive projection distance required')
    c.output.mkdir(parents=True,exist_ok=True);graphs={'raw':load_graph(c.raw),'refined':load_graph(c.refined)}
    write_json(c.output/'navigation_comparison.json',{k:stats(g) for k,g in graphs.items()})
    if not c.pairs:
        write_json(c.output/'astar_pairs.json',[])
        write_json(c.output/'astar_pairs_example.json',[{'id':'replace_with_manual_pair','start':[100,100],'goal':[300,300]}])
        return
    pairs=json.loads(c.pairs.read_text());results=[]
    for pair in pairs:
        if not isinstance(pair['id'],str) or Path(pair['id']).name!=pair['id']:raise ValueError('Unsafe pair ID')
        r={'id':pair['id']}
        for name,g in graphs.items():
            answer,h,lines=route(g,pair,c.max_projection_distance,c.uncertainty_lambda);r[name]=answer
            overlay(h,background(c.image,g.graph['inference_shape']),c.output/(pair['id']+'_'+name+'.png'),lines=lines)
        results.append(r)
    write_json(c.output/'astar_comparison.json',results)

if __name__=='__main__':main()
