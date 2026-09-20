"""RGB / probability / graph review cards in the declared inference frame."""
import argparse,json
from pathlib import Path
import numpy as np
import networkx as nx
from PIL import Image,ImageDraw
from shapely.geometry import LineString,Point
from graph_utils import load_graph,background,probability,xy


def heat(p):
    return Image.fromarray(np.uint8(np.stack([p,np.sqrt(p),1-p],axis=-1)*255),'RGB')


def draw_graph(im,g,offset=(0,0),component=None):
    d=ImageDraw.Draw(im)
    for u,v,e in g.edges(data=True):
        q=[(x-offset[0],y-offset[1]) for x,y in e['geometry']];d.line(q,fill=component.get(u,'#00b7d7') if component else '#00b7d7',width=2)


def card(base,prob,g,row,path,point=None):
    geom=np.asarray(row['geometry']['polyline_xy']);q=geom if point is None else np.vstack([geom,point]);lo=np.maximum(0,np.floor(q.min(0)-40)).astype(int);hi=np.minimum(base.size,np.ceil(q.max(0)+41)).astype(int)
    # Review crops are bounded and preserve actual geometry scale until display.
    if point is None and (hi[0]-lo[0]>600 or hi[1]-lo[1]>600):
        center=np.asarray(point if point is not None else geom.mean(0));lo=np.maximum(0,center-300).astype(int);hi=np.minimum(base.size,lo+600).astype(int)
    bbox=(*lo,*hi);panels=[base.crop(bbox),prob.crop(bbox),base.crop(bbox)]
    colors={n:('#00ff66' if i==row['source']['component_id'] else '#ffff00') for i,nodes in enumerate(nx.connected_components(g)) if i in (row['source']['component_id'],row['target']['component_id']) for n in nodes}
    draw_graph(panels[2],g,lo,colors)
    if 'edge_nodes' in row['target']:
        u,v=row['target']['edge_nodes'];ImageDraw.Draw(panels[2]).line([(x-lo[0],y-lo[1]) for x,y in g[u][v]['geometry']],fill='yellow',width=4)
    for panel in panels:
        d=ImageDraw.Draw(panel);d.line([tuple(z-lo) for z in geom],fill='#ff3cff',width=3)
        for z,color in ((geom[0],'#00ff55'),(geom[-1],'#ffff00')):
            x,y=z-lo;d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        if point is not None:
            x,y=np.asarray(point)-lo;d.ellipse((x-6,y-6,x+6,y+6),outline='white',width=3)
    w,h=panels[0].size;out=Image.new('RGB',(w*3,max(h+90,130)),'#161616')
    for i,panel in enumerate(panels):out.paste(panel,(w*i,0))
    d=ImageDraw.Draw(out);f=row['probability_features'];z=row['geometry_features'];d.text((8,h+5),f"{row['candidate_id']} {row['type']}  RGB | probability | graph",fill='white');d.text((8,h+23),f"distance={z['straight_distance_px']:.1f} length={row['geometry']['length_px']:.1f} detour={z['detour_ratio']:.2f}",fill='white');d.text((8,h+41),f"mean={f['mean']:.3f} p10={f['p10']:.3f} rank={row['heuristic_score']:.3f} comp={row['source']['component_id']}->{row['target']['component_id']}",fill='white');d.text((8,h+59),','.join(row['rejection_flags'])[:150] or 'PROPOSAL ONLY - human road validity not established',fill='white')
    if point is not None:
        nearest_edge=min((LineString(e['geometry']) for *_,e in g.edges(data=True)),key=lambda line:line.distance(Point(point)))
        candidate_distance=LineString(row['geometry']['polyline_xy']).distance(Point(point))
        d.text((8,h+77),f'Click={point[0]:.1f},{point[1]:.1f} nearest graph={nearest_edge.distance(Point(point)):.1f}px candidate={candidate_distance:.1f}px',fill='white')
    out.save(path)


def main():
    p=argparse.ArgumentParser();
    for k in ('graph','probability','image','candidates','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--coverage',type=Path);a=p.parse_args();g=load_graph(a.graph);base=background(a.image,g.graph['inference_shape']);prob=heat(probability(a.probability,g.graph['inference_shape']));rows=json.loads(a.candidates.read_text())['candidates'];active=sorted([r for r in rows if r['status']=='proposed' and not r.get('budget_excluded')],key=lambda r:-r['heuristic_score']);a.output.mkdir(parents=True,exist_ok=True)
    for kind in ('endpoint_endpoint','endpoint_edge','endpoint_junction','component_corridor'):
        rr=[r for r in active if r['type']==kind];im=base.point(lambda x:int(x*.6));draw_graph(im,g);d=ImageDraw.Draw(im)
        for r in rr[:100]:
            q=r['geometry']['polyline_xy'];d.line([tuple(z) for z in q],fill='#ff3cff',width=3)
            if kind=='endpoint_edge':d.line([tuple(z) for z in g.edges[tuple(r['target']['edge_nodes'])]['geometry']],fill='#ffee00',width=4)
            if kind=='endpoint_junction':
                v=r['target']['node_id'];
                for u in g[v]:d.line([tuple(xy(g,v)),tuple(xy(g,u))],fill='#ffee00',width=4)
            for z,color in ((q[0],'#00ff66'),(q[-1],'#ffff00')):x,y=z;d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        im.save(a.output/('corridor_candidates_overlay.png' if kind=='component_corridor' else kind+'_overlay.png'));folder=a.output/('corridor_reviews' if kind=='component_corridor' else kind+'_reviews');folder.mkdir(exist_ok=True)
        for r in rr[:50]:card(base,prob,g,r,folder/(r['candidate_id']+'.png'))
        rejected=[r for r in rows if r['type']==kind and r['status']=='rejected_by_hard_gate'];folder=a.output/(kind+'_rejected_reviews');folder.mkdir(exist_ok=True)
        for r in sorted(rejected,key=lambda r:-r['heuristic_score'])[:10]:card(base,prob,g,r,folder/(r['candidate_id']+'.png'))
    if a.baseline and a.coverage:
        audit={r['pair_id']:r for r in json.loads(a.coverage.read_text())};folder=a.output/'coverage_failure_review';folder.mkdir(exist_ok=True)
        for r in json.loads(a.baseline.read_text()):
            pair=r['pair'];entry=audit[pair['id']]
            if entry['baseline_failure_type']!='coverage_failure':continue
            for key in ('start','goal'):
                pt=Point(pair[key]);corridors=[z for z in active if z['type']=='component_corridor'];nearest=min(corridors,key=lambda z:LineString(z['geometry']['polyline_xy']).distance(pt)) if corridors else None
                if nearest:card(base,prob,g,nearest,folder/f'{pair["id"]}_{key}.png',pair[key])
if __name__=='__main__':main()
