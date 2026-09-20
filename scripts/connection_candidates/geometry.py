import math
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from graph_utils import xy


def angle(a,b):
    z=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/z,-1,1)))) if z else 180.


def oriented(g,u,v):
    q=np.asarray(g[u][v]['geometry'],float)
    return q if np.linalg.norm(q[0]-xy(g,u))<np.linalg.norm(q[-1]-xy(g,u)) else q[::-1]


def estimate_endpoint_tangent(g,n,lookback_length_px=30):
    points=[xy(g,n)];u=n;previous=None;remaining=lookback_length_px;visited={n}
    while remaining>0:
        neighbors=[v for v in g[u] if v!=previous and v not in visited]
        if not neighbors:break
        v=neighbors[0];q=oriented(g,u,v)
        for a,b in zip(q,q[1:]):
            length=np.linalg.norm(b-a)
            if length>remaining:
                points.append(a+(b-a)*remaining/length);remaining=0;break
            points.append(b);remaining-=length
        previous,u=u,v;visited.add(v)
        if g.degree(v)!=2:break
    q=np.asarray(points);q=q-q.mean(0)
    if len(q)<2:return np.zeros(2)
    _,_,vt=np.linalg.svd(q,full_matrices=False);t=vt[0]
    if np.dot(t,points[0]-points[-1])<0:t=-t
    return t


def probability_features(geometry,p):
    if p is None:
        return {'mean':0.5,'median':0.5,'p10':0.5,'minimum':0.5,'maximum':0.5,'std':0.0,
                'fraction_below_0.2':0.0,'fraction_below_0.3':0.0,'longest_low_probability_run_ratio':0.0,
                'sample_count':0,'evidence_available':False}
    line=LineString(geometry);q=np.asarray([line.interpolate(s).coords[0] for s in np.linspace(0,line.length,max(16,math.ceil(line.length)+1))])
    if (q<0).any() or (q[:,0]>p.shape[1]-1).any() or (q[:,1]>p.shape[0]-1).any():raise ValueError('Candidate outside probability frame')
    from scipy.ndimage import map_coordinates
    a=map_coordinates(p,[q[:,1],q[:,0]],order=1,mode='nearest');low=a<.2;longest=current=0
    for value in low:current=current+1 if value else 0;longest=max(longest,current)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'minimum':float(a.min()),'maximum':float(a.max()),'std':float(a.std()),'fraction_below_0.2':float((a<.2).mean()),'fraction_below_0.3':float((a<.3).mean()),'longest_low_probability_run_ratio':longest/len(a),'sample_count':len(a),'low_run_threshold':.2}


class Context:
    def __init__(self,g,p,lookback=30):
        self.g,self.p=g,p;self.ends=sorted(n for n in g if g.degree(n)==1);self.junctions=sorted(n for n in g if g.degree(n)>=3)
        self.components=list(nx.connected_components(g));self.component={n:i for i,c in enumerate(self.components) for n in c}
        self.tangents={n:estimate_endpoint_tangent(g,n,lookback) for n in self.ends}
        self.edges=list(g.edges());self.lines=[LineString(g[u][v]['geometry']) for u,v in self.edges];self.tree=STRtree(self.lines)
        self.component_info=[]
        for i,nodes in enumerate(self.components):
            q=np.asarray([xy(g,n) for n in sorted(nodes)]);edges=g.subgraph(nodes).edges(data=True)
            boundary=sorted(set([sorted(nodes)[int(np.argmin(q[:,j]))] for j in range(2)]+[sorted(nodes)[int(np.argmax(q[:,j]))] for j in range(2)]))
            self.component_info.append({'id':i,'size':len(nodes),'length':sum(e['length'] for *_,e in edges),'bbox_xyxy':[*q.min(0).tolist(),*q.max(0).tolist()],'centroid_xy':q.mean(0).tolist(),'endpoints':[n for n in sorted(nodes) if g.degree(n)==1],'boundary_nodes':boundary})
    def crossings(self,geom,source,target):
        line=LineString(geom);locations=[]
        for i in self.tree.query(line):
            i=int(i);intersection=line.intersection(self.lines[i])
            if intersection.is_empty:continue
            parts=list(intersection.geoms) if hasattr(intersection,'geoms') else [intersection]
            for part in parts:
                points=[part] if part.geom_type=='Point' else [Point(part.coords[0]),Point(part.coords[-1])]
                for pt in points:
                    q=list(pt.coords[0]);end=min(Point(source).distance(pt),Point(target).distance(pt))
                    if not any(np.linalg.norm(np.asarray(z['xy'])-q)<1e-4 and z['component_id']==self.component[self.edges[i][0]] for z in locations):locations.append({'xy':q,'edge_id':i,'component_id':self.component[self.edges[i][0]],'kind':'path_ends_at_graph' if end<2 else ('near_endpoint' if end<8 else 'interior'),'path_position':line.project(pt)})
        return locations


def make_candidate(ctx,kind,source,target,geom,features,flags=None):
    line=LineString(geom);features=dict(features)
    vectors=np.diff(np.asarray(geom,float),axis=0);vectors=vectors[np.linalg.norm(vectors,axis=1)>1e-8]
    turns=[angle(x,y) for x,y in zip(vectors,vectors[1:])]
    features.setdefault('mean_turning_angle',float(np.mean(turns)) if turns else 0)
    features.setdefault('max_turning_angle',max(turns,default=0))
    features.setdefault('curvature',sum(np.radians(turns))/max(line.length,1))
    errors=[]
    for seed,is_source in ((source,True),(target,False)):
        n=seed.get('node_id')
        if n in ctx.tangents:
            endpoint=np.asarray(geom[0] if is_source else geom[-1]);t=min(15,line.length/2);inner=np.asarray(line.interpolate(t if is_source else line.length-t).coords[0]);err=angle(ctx.tangents[n],inner-endpoint);errors.append(err)
            features.setdefault('outward_tangent_A' if is_source else 'outward_tangent_B',ctx.tangents[n].tolist())
    features.setdefault('max_angle_error',max(errors,default=0))
    a,b=source['component_id'],target['component_id'];crossings=ctx.crossings(geom,geom[0],geom[-1]);p=probability_features(geom,ctx.p)
    straight=float(np.linalg.norm(np.asarray(geom[-1])-geom[0]));detour=line.length/max(straight,1e-6)
    graph={'same_component':a==b,'component_size_A':ctx.component_info[a]['size'],'component_size_B':ctx.component_info[b]['size'],'component_length_A':ctx.component_info[a]['length'],'component_length_B':ctx.component_info[b]['length'],'crossing_count':len(crossings),'crossing_locations':crossings,'crossed_component_ids':sorted(set(z['component_id'] for z in crossings))}
    direction=max(0,1-features.get('max_angle_error',0)/90)
    subs={'distance_score':math.exp(-straight/(250 if kind=='component_corridor' else 80)),'direction_score':direction,'probability_score':.5*p['mean']+.25*p['p10']+.25*(1-p['longest_low_probability_run_ratio']),'detour_score':1/max(1,detour),'graph_context_score':1. if a!=b else .5}
    score=sum(subs[k]*v for k,v in [('distance_score',.15),('direction_score',.2),('probability_score',.4),('detour_score',.15),('graph_context_score',.1)])
    return {'type':kind,'source':source,'target':target,'geometry':{'polyline_xy':list(map(lambda q:list(map(float,q)),geom)),'length_px':line.length},'geometry_features':{'straight_distance_px':straight,'detour_ratio':detour,**features},'probability_features':p,'graph_features':graph,'ranking_subscores':subs,'heuristic_score':score,'rejection_flags':flags or [],'status':'rejected_by_hard_gate' if flags else 'proposed'}
