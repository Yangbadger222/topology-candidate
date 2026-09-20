"""Audited centerline GT, spatial splits, conservative labels and candidates."""
from __future__ import annotations
import json, math
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union, substring
from graph_utils import xy

def read_gt(path: Path, image_size: list[int], reviewed: bool=True) -> nx.Graph:
    """Refuse unreviewed or partial coverage for automatic negative labels."""
    d=json.loads(path.read_text())
    if d.get('coordinate_order')!='x,y' or d.get('units')!='inference_pixel' or d.get('image_size')!=image_size:raise ValueError('GT coordinate frame mismatch')
    if reviewed and (d.get('annotation_status')!='reviewed' or d.get('coverage_complete') is not True):raise ValueError('GT must be human reviewed and coverage complete')
    g=nx.Graph();g.graph.update(d)
    for n in d['nodes']:
        if n['id'] in g or not np.isfinite([n['x'],n['y']]).all() or not (0<=n['x']<=image_size[0] and 0<=n['y']<=image_size[1]):raise ValueError('Invalid GT node')
        g.add_node(n['id'],x=n['x'],y=n['y'])
    for e in d['edges']:
        u,v=e['u'],e['v'];geom=e['polyline']
        if u==v or u not in g or v not in g or g.has_edge(u,v):raise ValueError('Invalid GT edge')
        q=np.asarray(geom)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():raise ValueError('Invalid polyline')
        if (q<0).any() or (q[:,0]>image_size[0]).any() or (q[:,1]>image_size[1]).any():raise ValueError('GT polyline outside image')
        if np.linalg.norm(q[0]-xy(g,u))>1e-4 or np.linalg.norm(q[-1]-xy(g,v))>1e-4:raise ValueError('GT endpoints mismatch')
        g.add_edge(u,v,geometry=geom,length=LineString(geom).length)
    return g

def validate_splits(data: dict) -> None:
    """Require explicit geographic blocks and separation of full context extents."""
    regions=data['regions'];ids=set();buffer=data.get('min_cross_split_separation',512)
    if buffer<512:raise ValueError('Separation must cover at least a 512px context')
    for r in regions:
        if r['id'] in ids or r['split'] not in ('train','validation','test'):raise ValueError('Invalid region split')
        ids.add(r['id'])
        if r['units']!='inference_pixel' or len(r['bbox_xyxy'])!=4:raise ValueError('Explicit region extent required')
        x,y,xx,yy=r['bbox_xyxy']
        if not all(math.isfinite(v) for v in (x,y,xx,yy)) or xx<=x or yy<=y:raise ValueError('Invalid block extent')
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            if a['split']!=b['split'] and a['geographic_frame']==b['geographic_frame'] and box(*a['bbox_xyxy']).distance(box(*b['bbox_xyxy']))<buffer:raise ValueError('Spatial leakage: adjacent cross-split context blocks')

def points_on_line(geometry: list, n: int=32) -> np.ndarray:
    """Uniform arc-length samples on the actual polyline."""
    line=LineString(geometry)
    if line.length<=0:raise ValueError('Zero length sample')
    return np.array([line.interpolate(t).coords[0] for t in np.linspace(0,line.length,n)])

def edge_label(geometry: list, gt: nx.Graph, positive_distance: float=8, positive_overlap: float=.9, negative_distance: float=20) -> dict:
    """Match the full edge, retain ambiguous instead of forcing binary labels."""
    lines=[LineString(e['geometry']) for *_,e in gt.edges(data=True)]
    line=LineString(geometry);q=points_on_line(geometry,max(32,math.ceil(line.length)+1))
    if not lines:return {'label':0,'reason':'complete reviewed GT is empty','length':line.length}
    merged=unary_union(lines);dist=np.array([merged.distance(Point(p)) for p in q]);overlap=float(np.mean(dist<=positive_distance))
    local=[l for l in lines if l.distance(line)<=negative_distance]
    haus=float(np.percentile(dist,95)) # directed sampled Hausdorff-like; does not penalize GT outside prediction
    label=1 if overlap>=positive_overlap and dist.mean()<=positive_distance and haus<=positive_distance*2 else (0 if dist.mean()>=negative_distance and overlap<=.1 else -1)
    return {'label':label,'mean_distance':float(dist.mean()),'overlap':overlap,'directed_distance_p95':haus,'length':line.length,'local_gt_edges':len(local)}

def gap_candidates(g: nx.Graph,max_distance: float=120,min_distance: float=20,max_angle: float=45) -> list[dict]:
    """KD-tree endpoint pairs with outward tangent and distinct components."""
    ends=sorted(n for n in g if g.degree(n)==1);coords=np.array([xy(g,n) for n in ends]);cs={n:(i,len(c)) for i,c in enumerate(nx.connected_components(g)) for n in c}
    if len(ends)<2:return []
    rows=[]
    def tangent(n: int) -> np.ndarray:
        v=next(iter(g[n]));q=np.asarray(g[n][v]['geometry']);q=q if np.linalg.norm(q[0]-xy(g,n))<=np.linalg.norm(q[-1]-xy(g,n)) else q[::-1]
        t=q[0]-q[1];return t/max(np.linalg.norm(t),1e-9)
    for i,j in sorted(cKDTree(coords).query_pairs(max_distance)):
        u,v=ends[i],ends[j];delta=coords[j]-coords[i];distance=np.linalg.norm(delta)
        if distance<min_distance or cs[u][0]==cs[v][0]:continue
        angles=[math.degrees(math.acos(float(np.clip(np.dot(tangent(n),sign*delta/distance),-1,1)))) for n,sign in [(u,1),(v,-1)]]
        if max(angles)>max_angle:continue
        rows.append({'u':u,'v':v,'geometry':[coords[i].tolist(),coords[j].tolist()],'distance':float(distance),'angle_a':angles[0],'angle_b':angles[1],'component_size_a':cs[u][1],'component_size_b':cs[v][1],'different_components':True})
    return rows

def gap_label(row: dict,gt: nx.Graph,tolerance: float=8,max_detour: float=1.3) -> dict:
    """Project on GT edges; positive only if a short GT route follows the bridge."""
    if gt.number_of_edges()==0:return {'label':0,'reason':'complete reviewed GT is empty'}
    entries=[]
    for u,v,e in gt.edges(data=True):
        q=e['geometry'];q=q if np.linalg.norm(np.asarray(q[0])-xy(gt,u))<1e-4 else q[::-1]
        entries.append((u,v,LineString(q)))
    projections=[]
    for q in row['geometry']:
        p=Point(q);u,v,l=min(entries,key=lambda e:e[2].distance(p));projections.append((u,v,l,l.project(p),l.distance(p)))
    if any(p[4]>tolerance*2 for p in projections):return {'label':0,'reason':'endpoint far from GT'}
    if any(p[4]>tolerance for p in projections):return {'label':-1,'reason':'endpoint uncertain'}
    a,b=projections
    # route distance includes projection positions, not just nearest GT vertices.
    routes=[]
    if {a[0],a[1]}=={b[0],b[1]}:routes.append((abs(a[3]-b[3]),list(substring(a[2],a[3],b[3]).coords)))
    for na,da in ((a[0],a[3]),(a[1],a[2].length-a[3])):
        for nb,db in ((b[0],b[3]),(b[1],b[2].length-b[3])):
            if nx.has_path(gt,na,nb):
                path=nx.shortest_path(gt,na,nb,weight='length');geom=[]
                for u,v in zip(path,path[1:]):
                    q=gt[u][v]['geometry'];geom+=q if np.linalg.norm(np.asarray(q[0])-xy(gt,u))<1e-4 else q[::-1]
                routes.append((da+db+nx.path_weight(gt,path,'length'),geom))
    if not routes:return {'label':0,'reason':'GT components disconnected'}
    length,geom=min(routes,key=lambda r:r[0]);bridge=LineString(row['geometry'])
    corridor=max((bridge.distance(Point(q)) for q in geom),default=0)
    if length<=row['distance']*max_detour+2*tolerance and corridor<=tolerance*2:return {'label':1,'gt_route_length':length,'corridor_distance':corridor}
    return {'label':-1,'reason':'GT route detours or corridor uncertain','gt_route_length':length,'corridor_distance':corridor}
