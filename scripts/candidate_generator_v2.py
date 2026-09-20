"""Generate ranked proposals only; graph and model files remain read-only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from graph_utils import load_graph,probability,write_json,stats
from connection_candidates.geometry import Context
from connection_candidates.local import generate_local
from corridor_proposal import generate_corridors


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True); p.add_argument('--probability',type=Path,default=None); p.add_argument('--output',type=Path,required=True)
    defaults={'endpoint-endpoint-radius':80,'endpoint-edge-radius':80,'endpoint-edge-angle':45,'tangent-lookback':30,'junction-snap-radius':8,'component-search-radius':250,'corridor-lambda-prob':8,'corridor-gamma':2,'low-prob-threshold':.2,'low-probability-penalty':4,'corridor-window-margin':64,'seed-radius':3,'max-corridor-length':250,'max-detour-ratio':2.5,'min-mean-probability':.15,'min-p10-probability':.02,'corridor-simplify-epsilon':2}
    for k,v in defaults.items():p.add_argument('--'+k,type=float,default=v)
    for k,v in {'max-candidates-per-endpoint':5,'max-corridors-per-component-pair':3,'max-component-neighbors':10}.items():p.add_argument('--'+k,type=int,default=v)
    return p


def distribution(values):
    return dict(zip(['min','p10','median','p90','max','mean'],map(float,[np.min(values),np.percentile(values,10),np.median(values),np.percentile(values,90),np.max(values),np.mean(values)]))) if values else {}


def run(c):
    if any(getattr(c,k)<=0 for k in ('endpoint_endpoint_radius','endpoint_edge_radius','component_search_radius','max_corridor_length','tangent_lookback','max_candidates_per_endpoint','max_corridors_per_component_pair','max_component_neighbors')):raise ValueError('Positive radii / budgets required')
    start=time.perf_counter();g=load_graph(c.graph); shape=g.graph.get('inference_shape',(int(max((g.nodes[n]['y'] for n in g),default=0))+2,int(max((g.nodes[n]['x'] for n in g),default=0))+2));p=probability(c.probability,shape) if c.probability else None;ctx=Context(g,p,c.tangent_lookback);t=time.perf_counter();local,audit=generate_local(ctx,c);localtime=time.perf_counter()-t;t=time.perf_counter();corridors,search=generate_corridors(ctx,c);corridortime=time.perf_counter()-t
    rows=local+corridors
    for i,row in enumerate(rows):row['candidate_id']=f'candidate_{i:06d}'
    active=[r for r in rows if r['status']=='proposed' and not r.get('budget_excluded')];groups={}
    for kind in ('endpoint_endpoint','endpoint_edge','endpoint_junction','component_corridor'):
        rr=[r for r in rows if r['type']==kind];aa=[r for r in active if r['type']==kind]
        groups[kind]={'total':len(rr),'hard_gate_rejected':sum(r['status']=='rejected_by_hard_gate' for r in rr),'budget_excluded':sum(r.get('budget_excluded',False) for r in rr),'proposed':len(aa),'distance_distribution':distribution([r['geometry_features']['straight_distance_px'] for r in aa]),'probability_distribution':distribution([r['probability_features']['mean'] for r in aa]),'score_distribution':distribution([r['heuristic_score'] for r in aa])}
    summary={'graph':stats(g),'endpoints':len(ctx.ends),'junctions':len(ctx.junctions),'types':groups,'endpoint_runtime_seconds':localtime,'corridor_runtime_seconds':corridortime,'total_runtime_seconds':time.perf_counter()-start,'local_audit':audit,**search,'active_candidates':len(active)}
    c.output.mkdir(parents=True,exist_ok=True)
    write_json(c.output/'config_used.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(c).items()})
    write_json(c.output/'provenance.json',{'coordinate_order':'x,y','units':'inference_pixel','inference_shape':list(shape),'graph_sha256':hashlib.sha256(c.graph.read_bytes()).hexdigest(),'probability_sha256':hashlib.sha256(c.probability.read_bytes()).hexdigest() if c.probability else None,'probability_optional':True,'probability_quantization':1/255 if c.probability and c.probability.suffix=='.png' else None,'status':'proposals_only','graph_mutated':False,'heuristic_is_label':False,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).parent/'corridor_proposal.py',*sorted(f for f in (Path(__file__).parent/'connection_candidates').glob('*') if f.is_file())]}})
    write_json(c.output/'all_candidates.json',{'coordinate_order':'x,y','units':'inference_pixel','candidates':rows});write_json(c.output/'candidate_statistics.json',summary);write_json(c.output/'component_inventory.json',ctx.component_info)
    for kind in groups:write_json(c.output/(('corridor' if kind=='component_corridor' else kind)+'_candidates.json'),[r for r in rows if r['type']==kind])
    print(json.dumps({k:v for k,v in summary.items() if k not in ('graph','search_log')},indent=2));return rows,summary
if __name__=='__main__':run(parser().parse_args())
