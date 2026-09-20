"""Conservative, auditable MaGRoad graph refinement. No model changes."""
from __future__ import annotations
import argparse
import copy
import logging
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.strtree import STRtree
from graph_utils import load_graph,save_graph,stats,xy,write_json,probability,sample_prob,background,overlay


def angle(a: np.ndarray,b: np.ndarray) -> float:
    """Directed angle in degrees; zero-length vectors cannot align."""
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a@b/den,-1,1)))) if den else 180.


def rebuild(g: nx.Graph,mapping: dict,positions: dict|None=None) -> nx.Graph:
    """Contract selected nodes, retaining polylines and avoiding loops."""
    h=nx.Graph();h.graph=copy.deepcopy(g.graph)
    for n,d in g.nodes(data=True):
        m=mapping.get(n,n)
        if m not in h:h.add_node(m,**copy.deepcopy(d))
    for n,q in (positions or {}).items():h.nodes[n].update(x=float(q[0]),y=float(q[1]))
    for u,v,d in g.edges(data=True):
        a,b=mapping.get(u,u),mapping.get(v,v)
        if a==b:continue
        e=copy.deepcopy(d);geom=np.array(e['geometry'],dtype=float)
        if np.linalg.norm(geom[0]-xy(g,u))>np.linalg.norm(geom[-1]-xy(g,u)):geom=geom[::-1]
        geom[0]=xy(h,a);geom[-1]=xy(h,b)
        e['geometry']=geom.tolist();e['length']=float(LineString(geom).length)
        if not h.has_edge(a,b) or e['length']>h.edges[a,b]['length']:h.add_edge(a,b,**e)
    return h


def clean(g: nx.Graph,c: argparse.Namespace) -> tuple[nx.Graph,dict]:
    """Merge only epsilon-identical coordinates through a spatial index."""
    ns=list(g);pairs=cKDTree([xy(g,n) for n in ns]).query_pairs(c.node_epsilon)
    uf=nx.utils.UnionFind(ns)
    for i,j in sorted(pairs):uf.union(ns[i],ns[j])
    groups=list(uf.to_sets());mapping={n:min(s) for s in groups for n in s}
    h=rebuild(g,mapping)
    return h,{'merged_nodes':len(g)-len(h),'removed_edges':g.number_of_edges()-h.number_of_edges(),**g.graph.get('parser_cleaning',{})}


def paths(g: nx.Graph) -> list[dict]:
    """Extract maximal degree-two chains; preserve closed chains as cycles."""
    used=set();result=[]
    for start in sorted(g,key=lambda n:g.degree(n)==2):
        for neighbor in g[start]:
            if frozenset((start,neighbor)) in used:continue
            ns=[start];geom=[];u,v=start,neighbor
            while True:
                used.add(frozenset((u,v)));e=g.edges[u,v];q=e['geometry']
                if np.linalg.norm(np.array(q[0])-xy(g,u))>np.linalg.norm(np.array(q[-1])-xy(g,u)):q=q[::-1]
                geom.extend(q if not geom else q[1:]);ns.append(v)
                if g.degree(v)!=2 or v==start:break
                w=next(n for n in g[v] if n!=u)
                if frozenset((v,w)) in used:break
                u,v=v,w
            result.append({'nodes':ns,'geometry':geom,'line':LineString(geom),'length':float(LineString(geom).length)})
    return result


def parallel(g: nx.Graph,p: np.ndarray|None,c: argparse.Namespace) -> tuple[nx.Graph,dict,list]:
    """Consolidate sustained straight corridors, preserving all external attachments.

    Keep the better supported continuous chain, project the redundant chain's
    endpoints onto it, split its edges, and reattach external branches there.
    This chooses evidence over an invented averaged road location.
    """
    h=g.copy();ps=paths(g);tree=STRtree([a['line'] for a in ps]);used=set();groups=[];debug=[];removed=0
    for i,a in enumerate(ps):
        if i in used or a['length']<c.parallel_min_length or a['nodes'][0]==a['nodes'][-1]:continue
        av=np.array(a['geometry'][-1])-a['geometry'][0]
        if np.linalg.norm(av)/a['length']<c.parallel_straightness:continue
        for j in sorted(map(int,tree.query(a['line'].buffer(c.parallel_distance)))):
            if j<=i or j in used:continue
            b=ps[j]
            shared=set(a['nodes'])&set(b['nodes'])
            if shared-set((a['nodes'][0],a['nodes'][-1])) or shared-set((b['nodes'][0],b['nodes'][-1])):continue
            if not all(h.has_edge(u,v) for t in (a,b) for u,v in zip(t['nodes'],t['nodes'][1:])):continue
            bv=np.array(b['geometry'][-1])-b['geometry'][0]
            if b['length']<c.parallel_min_length or b['nodes'][0]==b['nodes'][-1]:continue
            if np.linalg.norm(bv)/b['length']<c.parallel_straightness:continue
            err=min(angle(av,bv),angle(av,-bv))
            if err>c.parallel_angle:continue
            # Bidirectional corridor coverage, not just nearest edge distance.
            aq=np.array([a['line'].interpolate(t,normalized=True).coords[0] for t in np.linspace(0,1,40)])
            bq=np.array([b['line'].interpolate(t,normalized=True).coords[0] for t in np.linspace(0,1,40)])
            from shapely.geometry import Point
            ca=np.mean([b['line'].distance(Point(q))<=c.parallel_distance for q in aq])
            cb=np.mean([a['line'].distance(Point(q))<=c.parallel_distance for q in bq])
            # Require overlap of BOTH paths and substantial common length.
            if min(ca,cb)<c.parallel_overlap or min(ca*a['length'],cb*b['length'])<c.parallel_min_length:continue
            def rank(t: dict) -> tuple:
                conf=[g.edges[u,v].get('confidence') for u,v in zip(t['nodes'],t['nodes'][1:])]
                available=[v for v in conf if v is not None]
                evidence=sample_prob(t['geometry'],p).get('mean',float(np.mean(available)) if available else 0)
                junctions=sum(g.degree(n)>=3 for n in (t['nodes'][0],t['nodes'][-1]))
                return (evidence,junctions,t['length']) if p is not None or available else (junctions,t['length'],0)
            keep,drop=(a,b) if rank(a)>=rank(b) else (b,a)
            # Attach only endpoint branches. Interiors are degree two by construction.
            attaches={}
            for n in (drop['nodes'][0],drop['nodes'][-1]):
                q=keep['line'].interpolate(keep['line'].project(Point(xy(g,n))))
                target=np.array(q.coords[0]);best=None
                for u,v in zip(keep['nodes'],keep['nodes'][1:]):
                    if not h.has_edge(u,v):continue
                    line=LineString(h.edges[u,v]['geometry']);dist=line.distance(q)
                    if best is None or dist<best[0]:best=(dist,u,v,line)
                if best is None:continue
                _,u,v,line=best
                if np.linalg.norm(np.array(line.coords[0])-xy(h,u))>np.linalg.norm(np.array(line.coords[-1])-xy(h,u)):u,v=v,u
                if np.linalg.norm(target-xy(h,u))<=c.node_epsilon:t=u
                elif np.linalg.norm(target-xy(h,v))<=c.node_epsilon:t=v
                else:
                    from shapely.ops import substring
                    t=max(h.nodes,default=-1)+1;h.add_node(t,x=float(target[0]),y=float(target[1]));old=dict(h.edges[u,v]);h.remove_edge(u,v)
                    d=line.project(q)
                    for aa,bb,part in [(u,t,substring(line,0,d)),(t,v,substring(line,d,line.length))]:
                        data={**old,'geometry':list(map(list,part.coords)),'length':float(part.length)};h.add_edge(aa,bb,**data)
                attaches[n]=t
            if len(attaches)!=2:raise RuntimeError('Parallel attachment failed')
            count=0
            for u,v in zip(drop['nodes'],drop['nodes'][1:]):
                if h.has_edge(u,v):h.remove_edge(u,v);count+=1
            for n,t in attaches.items():
                if n!=t and n in h:
                    for nb,data in list(h[n].items()):
                        if nb==t:continue
                        geom=np.array(data['geometry'],float)
                        if np.linalg.norm(geom[0]-xy(h,n))<np.linalg.norm(geom[-1]-xy(h,n)):geom[0]=xy(h,t)
                        else:geom[-1]=xy(h,t)
                        if not h.has_edge(t,nb):h.add_edge(t,nb,**{**data,'geometry':geom.tolist(),'length':float(LineString(geom).length)})
                    h.remove_node(n)
            for n in drop['nodes'][1:-1]:
                if n in h and h.degree(n)==0:h.remove_node(n)
            groups.append({'group':len(groups),'kept_nodes':keep['nodes'],'removed_nodes':drop['nodes'],
                'angle':err,'coverage_a':float(ca),'coverage_b':float(cb),'removed_edges':count})
            import colorsys
            rgb=colorsys.hsv_to_rgb((len(groups)*.61803398875)%1,.8,1)
            group_color='#'+''.join(f'{int(x*255):02x}' for x in rgb)
            debug.extend([(keep['geometry'],group_color),(drop['geometry'],group_color)]);removed+=count;used.update((i,j));break
    return h,{'merged_parallel_groups':groups,'removed_duplicate_edges':removed},debug


def junction(g: nx.Graph,p: np.ndarray|None,c: argparse.Namespace) -> tuple[nx.Graph,dict,list]:
    """Merge directly connected nearby junctions with multiple direction families.

    Requiring a short existing edge avoids merging neighboring disconnected roads.
    """
    h=g.copy();ns=[n for n in g if g.degree(n)>=3];groups=[];debug=[];used=set()
    if len(ns)<2:return h,{'merged_junctions':0,'groups':[]},debug
    pairs=cKDTree([xy(g,n) for n in ns]).query_pairs(c.junction_radius)
    mapping={};pos={}
    for i,j in sorted(pairs):
        a,b=ns[i],ns[j]
        if a in used or b in used or not g.has_edge(a,b) or g.edges[a,b]['length']>c.junction_radius:continue
        evidence=sample_prob(g.edges[a,b]['geometry'],p)
        if p is not None and evidence['p10']<c.junction_probability:continue
        vectors=[xy(g,n)-xy(g,u) for u in (a,b) for n in g[u] if n not in (a,b)]
        if not any(c.junction_direction_angle<angle(v,w)<180-c.junction_direction_angle for v in vectors for w in vectors):continue
        center=(xy(g,a)+xy(g,b))/2;mapping[b]=a;pos[a]=center;used.update((a,b))
        groups.append([a,b]);debug.extend([([xy(g,a).tolist(),center.tolist()],'#ff00ff'),([xy(g,b).tolist(),center.tolist()],'#ff00ff')])
    return rebuild(h,mapping,pos),{'merged_junctions':len(groups),'groups':groups},debug


def tangent(g: nx.Graph,n: int) -> np.ndarray:
    """Outward endpoint tangent from actual edge geometry."""
    nb=next(iter(g[n]));q=np.array(g.edges[n,nb]['geometry'])
    if np.linalg.norm(q[0]-xy(g,n))<np.linalg.norm(q[-1]-xy(g,n)):q=q[::-1]
    return q[-1]-q[-2]


def connect(g: nx.Graph,p: np.ndarray|None,c: argparse.Namespace,snap: bool) -> tuple[nx.Graph,dict,list]:
    """Complete only facing endpoints supported throughout the gap.

    A probability image is required even for snapping: geometry alone cannot
    identify buildings or rivers. No-evidence mode records rejected candidates.
    """
    h=g.copy();ns=[n for n in g if g.degree(n)==1];records=[];debug=[];accepted=0;used=set()
    distance=c.snap_distance if snap else c.max_gap_distance
    if len(ns)<2:return h,{'accepted':0,'candidates':[]},debug
    pairs=cKDTree([xy(g,n) for n in ns]).query_pairs(distance)
    ordered=sorted(pairs,key=lambda ij:np.linalg.norm(xy(g,ns[ij[0]])-xy(g,ns[ij[1]])))
    geoms=[LineString(d['geometry']) for *_,d in g.edges(data=True)];elist=list(g.edges());tree=STRtree(geoms)
    for i,j in ordered:
        a,b=ns[i],ns[j];delta=xy(g,b)-xy(g,a);dist=float(np.linalg.norm(delta))
        if dist<=c.node_epsilon or g.has_edge(a,b):continue
        err=max(angle(tangent(g,a),delta),angle(tangent(g,b),-delta))
        line=LineString([xy(g,a),xy(g,b)]);prob=sample_prob(list(line.coords),p,c.gap_low_probability)
        crossing=any(not ({a,b}&set(elist[k])) and line.intersects(geoms[k]) for k in map(int,tree.query(line)))
        same=nx.has_path(h,a,b)
        ds=max(0.,1-dist/(2*distance));direction=max(0.,float(np.cos(np.radians(err))))
        ps=(prob.get('mean',0)+prob.get('p10',0))/2;ts=0. if crossing or same else 1.
        weights=np.array([c.weight_distance,c.weight_direction,c.weight_probability,c.weight_topology]);score=float(weights@np.array([ds,direction,ps,ts])/weights.sum())
        reasons=[]
        if a in used or b in used:reasons.append('endpoint_already_used')
        if err>c.snap_angle:reasons.append('not_facing')
        if crossing:reasons.append('crosses_existing_edge')
        if same:reasons.append('would_create_cycle')
        if p is None:reasons.append('no_probability_evidence')
        elif prob['mean']<c.gap_mean or prob['p10']<c.gap_p10 or prob['low_run_fraction']>c.gap_low_run:reasons.append('insufficient_probability')
        if score<c.gap_score:reasons.append('low_score')
        ok=not reasons
        r={'node_a':a,'node_b':b,'distance':dist,'angle_error':err,'probability':prob,
            'score_terms':{'distance':ds,'direction':direction,'probability':ps,'topology':ts},'score':score,'accepted':ok,'reasons':reasons,'kind':'snap' if snap else 'gap'}
        records.append(r);debug.append((list(map(list,line.coords)),'#00ff00' if ok else '#ff3333'))
        if ok:
            h.add_edge(a,b,geometry=list(map(list,line.coords)),length=dist,confidence=ps,refinement=r['kind']);used.update((a,b));accepted+=1
    return h,{'accepted':accepted,'candidates':records},debug


def prune(g: nx.Graph,p: np.ndarray|None,c: argparse.Namespace) -> tuple[nx.Graph,dict,list]:
    """Prune only short terminal chains with consistently low evidence."""
    h=g.copy();records=[];debug=[];components={n:len(s) for s in nx.connected_components(g) for n in s}
    for t in paths(g):
        ns=t['nodes'];terminal=g.degree(ns[0])==1 or g.degree(ns[-1])==1
        if not terminal or t['length']>=c.spur_length:continue
        ev=sample_prob(t['geometry'],p);conf=[g.edges[a,b].get('confidence') for a,b in zip(ns,ns[1:])];conf=[x for x in conf if x is not None]
        low=(p is not None and ev['mean']<c.spur_probability and ev['p10']<c.spur_p10)
        if conf and np.mean(conf)>=c.spur_confidence:low=False
        # Require attachment to a junction or an entire tiny component, in addition to low evidence.
        main=g.degree(ns[0])>=3 or g.degree(ns[-1])>=3;small=components[ns[0]]<=c.small_component_nodes
        if not low or not (main or small):continue
        records.append({'nodes':ns,'length':t['length'],'probability':ev,'component_nodes':components[ns[0]],'attached_to_junction':main})
        debug.append((t['geometry'],'#ff3333'))
        for a,b in zip(ns,ns[1:]):
            if h.has_edge(a,b):h.remove_edge(a,b)
        for n in ns:
            if n in h and h.degree(n)==0:h.remove_node(n)
    return h,{'removed_spurs':len(records),'spurs':records},debug


def parser() -> argparse.ArgumentParser:
    """All geometric thresholds are inference pixels, angles degrees, probabilities 0–1."""
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--graph',type=Path,required=True);a.add_argument('--image',type=Path);a.add_argument('--probability',type=Path);a.add_argument('--output',type=Path,required=True)
    defaults={'node-epsilon':.01,'parallel-distance':12.,'parallel-angle':12.,'parallel-overlap':.6,'parallel-min-length':60.,'parallel-straightness':.95,
        'junction-radius':10.,'junction-direction-angle':35.,'junction-probability':.25,'snap-distance':20.,'snap-angle':25.,'max-gap-distance':35.,
        'gap-mean':.45,'gap-p10':.25,'gap-low-run':.15,'gap-low-probability':.2,'gap-score':.55,'spur-length':25.,'spur-probability':.2,'spur-p10':.15,'spur-confidence':.5,
        'weight-distance':1.,'weight-direction':2.,'weight-probability':3.,'weight-topology':1.}
    for k,v in defaults.items():a.add_argument('--'+k,type=float,default=v,help='inference pixels / degrees / probability or score weight; see config_used.json')
    a.add_argument('--small-component-nodes',type=int,default=10);a.add_argument('--no-visuals',action='store_true')
    return a


def run(c: argparse.Namespace) -> dict:
    """Run ordered stages, save every graph, statistic, action and debug overlay."""
    if any(getattr(c,k)<=0 for k in ['parallel_distance','snap_distance','max_gap_distance','snap_angle']):raise ValueError('Distances and angle must be positive')
    for key in ['parallel_overlap','parallel_straightness','junction_probability','gap_mean','gap_p10','gap_low_run','gap_low_probability','gap_score','spur_probability','spur_p10','spur_confidence']:
        if not 0<=getattr(c,key)<=1:raise ValueError(f'{key} must be in [0,1]')
    if min(c.weight_distance,c.weight_direction,c.weight_probability,c.weight_topology)<0 or sum([c.weight_distance,c.weight_direction,c.weight_probability,c.weight_topology])<=0:raise ValueError('Score weights must be nonnegative with positive sum')
    if c.max_gap_distance>40:raise ValueError('First version limits gap distance to 40 inference pixels')
    c.output.mkdir(parents=True,exist_ok=True);write_json(c.output/'config_used.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(c).items()})
    raw=load_graph(c.graph);shape=raw.graph.get('inference_shape')
    if not shape:raise ValueError('inference_shape required for explicit pixel frame')
    p=probability(c.probability,shape);base=background(c.image,shape) if not c.no_visuals else None
    write_json(c.output/'raw_stats.json',stats(raw,c.small_component_nodes))
    if base:overlay(raw,base,c.output/'raw_graph_overlay.png')
    stages=[('stage1_clean',lambda g: (*clean(g,c),[])),('stage2_parallel_merge',lambda g:parallel(g,p,c)),
        ('stage3_junction_merge',lambda g:junction(g,p,c)),('stage4_endpoint_snap',lambda g:connect(g,p,c,True)),
        ('stage5_gap_completion',lambda g:connect(g,p,c,False)),('stage6_final_navigation_graph',lambda g:prune(g,p,c))]
    g=raw;summary=[];allgaps=[]
    for name,func in stages:
        before=stats(g,c.small_component_nodes);g,actions,debug=func(g);after=stats(g,c.small_component_nodes)
        summary.append({'stage':name,'before':before,'after':after,'actions':actions})
        save_graph(g,c.output/(name+'_graph.json'))
        if base:
            target=name+'.png' if name.startswith('stage6') else name+'_overlay.png'
            overlay(g,base,c.output/target,raw=raw if name.startswith('stage2') else None)
            overlay(g,base,c.output/(name+'_debug.png'),lines=debug)
        if 'candidates' in actions:allgaps.extend(actions['candidates'])
        if 'spurs' in actions:write_json(c.output/'pruned_spurs.json',actions['spurs'])
        logging.info('%s: nodes %d→%d edges %d→%d components %d→%d',name,before['nodes'],after['nodes'],before['edges'],after['edges'],before['components'],after['components'])
    save_graph(g,c.output/'refined_graph.json');write_json(c.output/'refined_stats.json',stats(g,c.small_component_nodes));write_json(c.output/'gap_candidates.json',allgaps)
    write_json(c.output/'experiment_summary.json',summary)
    if base:
        from PIL import Image,ImageDraw
        overlay(g,base,c.output/'final_navigation_graph.png')
        left=Image.open(c.output/'raw_graph_overlay.png');right=Image.open(c.output/'final_navigation_graph.png')
        left.thumbnail((1600,1200));right.thumbnail((1600,1200));im=Image.new('RGB',(left.width+right.width,left.height+40),'#202020');im.paste(left,(0,40));im.paste(right,(left.width,40));d=ImageDraw.Draw(im);d.text((10,10),'Raw MaGRoad',fill='white');d.text((left.width+10,10),'Refined navigation graph',fill='white');im.save(c.output/'refinement_comparison.png')
    return {'raw':stats(raw,c.small_component_nodes),'refined':stats(g,c.small_component_nodes),'stages':summary}

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s');run(parser().parse_args())
