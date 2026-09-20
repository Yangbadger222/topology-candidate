"""Coordinate-safe graph I/O, measurement and rendering in inference pixels."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx
from PIL import Image, ImageDraw
from shapely.geometry import LineString


def write_json(path: Path, data: object) -> None:
    """Write strict, human-readable JSON."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def load_graph(path: Path) -> nx.Graph:
    """Parse explicit MaGRoad rc or normalized xy schema without guessing axes."""
    d=json.loads(path.read_text()); g=nx.Graph()
    if 'nodes_rc' in d:
        if d.get('coordinate_order')!='row,column': raise ValueError('Expected explicit row,column convention')
        g.graph.update(inference_shape=d['inference_shape'],original_shape=d.get('original_shape'))
        for i,(y,x) in enumerate(d['nodes_rc']): g.add_node(i,x=float(x),y=float(y))
        edges=[{'u':e[0],'v':e[1]} for e in d['edges']]
    else:
        if d.get('coordinate_order')!='x,y': raise ValueError('Normalized schema must declare x,y')
        g.graph.update(d.get('metadata',{}))
        for n in d['nodes']: g.add_node(n['id'],x=float(n['x']),y=float(n['y']))
        edges=d['edges']
    invalid=loops=duplicates=0
    for e in edges:
        u,v=e['u'],e['v']
        if u not in g or v not in g: invalid+=1; continue
        if u==v: loops+=1; continue
        if g.has_edge(u,v): duplicates+=1; continue
        geom=e.get('geometry',[xy(g,u).tolist(),xy(g,v).tolist()])
        a={k:v for k,v in e.items() if k not in ('u','v')};a['geometry']=geom
        a['length']=float(LineString(geom).length)
        g.add_edge(u,v,**a)
    g.graph['parser_cleaning']={'invalid_edges':invalid,'self_loops':loops,'duplicate_edges':duplicates}
    if not all(np.isfinite(xy(g,n)).all() for n in g): raise ValueError('Nonfinite node coordinate')
    return g


def xy(g: nx.Graph,n: int) -> np.ndarray:
    """Return a node in x,y order."""
    return np.array([g.nodes[n]['x'],g.nodes[n]['y']],dtype=float)


def save_graph(g: nx.Graph,path: Path) -> None:
    """Export normalized coordinates with original-image transform metadata."""
    write_json(path,{'coordinate_order':'x,y','units':'inference_pixel','metadata':g.graph,
        'nodes':[{'id':n,**d} for n,d in g.nodes(data=True)],
        'edges':[{'u':u,'v':v,**d} for u,v,d in g.edges(data=True)]})


def stats(g: nx.Graph,small_nodes: int=10) -> dict:
    """Measure all components, including isolated nodes; cycles are cycle rank."""
    cs=sorted(nx.connected_components(g),key=len,reverse=True)
    components=[{'nodes':len(c),'edges':g.subgraph(c).number_of_edges(),
        'total_length':sum(d['length'] for *_,d in g.subgraph(c).edges(data=True))} for c in cs]
    return {'nodes':len(g),'edges':g.number_of_edges(),'components':len(cs),
        'largest_component_ratio':len(cs[0])/len(g) if cs else 0,
        'total_length':sum(d['length'] for *_,d in g.edges(data=True)),
        'degree_distribution':dict(Counter(dict(g.degree()).values())),
        'endpoints':sum(g.degree(n)==1 for n in g),'small_components':sum(len(c)<small_nodes for c in cs),
        'cycles':g.number_of_edges()-len(g)+len(cs),'component_details':components}


def probability(path: Path|None,shape: tuple) -> np.ndarray|None:
    """Read scalar probability; reject RGB heatmaps and size mismatches."""
    if path is None:return None
    a=np.load(path) if path.suffix=='.npy' else np.asarray(Image.open(path))
    if a.ndim!=2 or tuple(a.shape)!=tuple(shape):raise ValueError('Probability must be scalar and match inference_shape')
    a=a.astype(float)/(255 if a.dtype==np.uint8 else 1)
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('Probability outside [0,1]')
    return a


def sample_prob(points: list,p: np.ndarray|None,low_threshold: float=.2) -> dict:
    """Sample each polyline segment at no more than one pixel spacing."""
    if p is None:return {}
    samples=[]
    for a,b in zip(points,points[1:]):
        a,b=np.array(a),np.array(b);q=np.linspace(a,b,max(16,int(np.ceil(np.linalg.norm(b-a)))+1))
        if (q[:,0]<0).any() or (q[:,1]<0).any() or (q[:,0]>p.shape[1]-1).any() or (q[:,1]>p.shape[0]-1).any():raise ValueError('Geometry outside probability image')
        samples.extend(p[np.rint(q[:,1]).astype(int),np.rint(q[:,0]).astype(int)])
    a=np.array(samples); low=a<low_threshold; longest=current=0
    for v in low:current=current+1 if v else 0;longest=max(longest,current)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.percentile(a,10)),
        'minimum':float(a.min()),'low_run_fraction':longest/len(a)}


def background(path: Path|None,shape: tuple) -> Image.Image:
    """Resize RGB explicitly to the graph's inference coordinate frame."""
    h,w=shape
    return Image.open(path).convert('RGB').resize((w,h),Image.Resampling.LANCZOS) if path else Image.new('RGB',(w,h),'#202020')


def overlay(g: nx.Graph,base: Image.Image,path: Path,raw: nx.Graph|None=None,
            lines: list|None=None, color: str='#00ffff') -> None:
    """Draw graph and optional debug lines without altering geometry."""
    im=base.point(lambda x:int(x*.65));d=ImageDraw.Draw(im)
    if raw:
        for *_,e in raw.edges(data=True):d.line([tuple(q) for q in e['geometry']],fill='#777777',width=2)
    for *_,e in g.edges(data=True):d.line([tuple(q) for q in e['geometry']],fill=color,width=2)
    for n in g:
        if g.degree(n)!=2:
            x,y=xy(g,n);d.ellipse((x-2,y-2,x+2,y+2),fill='#ffff00')
    for pts,c in lines or []:d.line([tuple(q) for q in pts],fill=c,width=3)
    im.save(path)
