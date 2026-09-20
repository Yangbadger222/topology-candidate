"""Seven-run one-factor sensitivity; same source graph, no acceptance or training."""
import csv,json,copy,time
from pathlib import Path
from candidate_generator_v2 import parser,run
from evaluate_candidate_coverage import evaluate
from graph_utils import load_graph


def main():
    p=parser();p.add_argument('--baseline',type=Path,required=True);c=p.parse_args();root=c.output;g=load_graph(c.graph);baseline=json.loads(c.baseline.read_text());cases=[('baseline',{})]+[(f'{k}_{v}',{k:v}) for k,vals in [('endpoint_edge_radius',[40,120]),('component_search_radius',[150,350]),('max_corridor_length',[150,350])] for v in vals];records=[]
    for name,values in cases:
        a=copy.copy(c);a.output=root/'sweep'/name
        for k,v in values.items():setattr(a,k,v)
        print('SWEEP',name,flush=True)
        if name=='baseline' and (root/'all_candidates.json').exists():
            rows=json.loads((root/'all_candidates.json').read_text())['candidates'];s=json.loads((root/'candidate_statistics.json').read_text())
        else:rows,s=run(a)
        _,coverage=evaluate(g,rows,baseline);active=[r for r in rows if r['status']=='proposed' and not r.get('budget_excluded')]
        records.append({'case':name,'endpoint_edge_radius':a.endpoint_edge_radius,'component_search_radius':a.component_search_radius,'max_corridor_length':a.max_corridor_length,'candidate_count':len(active),'endpoint_edge':s['types']['endpoint_edge']['proposed'],'endpoint_junction':s['types']['endpoint_junction']['proposed'],'corridors':s['types']['component_corridor']['proposed'],'runtime_seconds':s['total_runtime_seconds'],'pixel_searches':s['pixel_searches'],'mean_probability':sum(r['probability_features']['mean'] for r in active)/max(1,len(active)),'coverage_failures_any_failed_endpoint':coverage['coverage_failures_with_candidate_near_at_least_one_failed_endpoint'],'connectivity_direct_pair':coverage['connectivity_failures_with_direct_component_pair_proposal'],'connectivity_adjacent':coverage['connectivity_failures_with_adjacent_component_proposal']})
        with (root/'candidate_sweep.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
if __name__=='__main__':main()
