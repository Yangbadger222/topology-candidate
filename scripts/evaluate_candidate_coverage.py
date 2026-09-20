"""Candidate proximity / component coverage is not navigation success."""
import argparse,json
from pathlib import Path
import networkx as nx
from shapely.geometry import Point,LineString
from graph_utils import load_graph,write_json


def evaluate(g,rows,baseline,radius=40):
    cs={n:i for i,c in enumerate(nx.connected_components(g)) for n in c};edges=[(u,v,LineString(e['geometry'])) for u,v,e in g.edges(data=True)];active=[r for r in rows if r['status']=='proposed' and not r.get('budget_excluded')];answer=[]
    for result in baseline:
        pair=result['pair'];raw=result['raw'];kind='success' if raw['success'] else ('connectivity_failure' if 'start_projection' in raw else 'coverage_failure');projections=[]
        for key in ('start','goal'):
            point=Point(pair[key]);u,v,line=min(edges,key=lambda e:e[2].distance(point));projections.append({'distance':line.distance(point),'component_id':cs[u],'projected_xy':list(line.interpolate(line.project(point)).coords[0])})
        near={key:[r['candidate_id'] for r in active if LineString(r['geometry']['polyline_xy']).distance(Point(pair[key]))<=radius] for key in ('start','goal')}
        components={z['component_id'] for z in projections};bridge=[r['candidate_id'] for r in active if {r['source']['component_id'],r['target']['component_id']}==components and len(components)==2];adjacent=[r['candidate_id'] for r in active if r['type'] in ('endpoint_edge','endpoint_junction','component_corridor') and components.intersection({r['source']['component_id'],r['target']['component_id']})]
        answer.append({'pair_id':pair['id'],'baseline_failure_type':kind,'near_start_candidates':len(near['start']),'near_goal_candidates':len(near['goal']),'near_start_ids':near['start'],'near_goal_ids':near['goal'],'component_bridge_candidates':len(bridge),'component_bridge_ids':bridge,'adjacent_component_candidate_ids':adjacent,'candidate_ids':sorted(set(near['start']+near['goal']+bridge)),'projections':dict(zip(('start','goal'),projections)),'coverage_failure_endpoint_diagnostic':{key:projections[i]['distance']>40 and bool(near[key]) for i,key in enumerate(('start','goal'))},'any_endpoint_proximity':bool(near['start'] or near['goal']),'direct_component_pair_proposal':bool(bridge)})
    failures=[r for r in answer if r['baseline_failure_type']=='coverage_failure'];connect=[r for r in answer if r['baseline_failure_type']=='connectivity_failure']
    summary={'valid_pairs':len(answer),'baseline_success':sum(r['baseline_failure_type']=='success' for r in answer),'coverage_failures':len(failures),'coverage_failures_with_candidate_near_at_least_one_failed_endpoint':sum(any(r['coverage_failure_endpoint_diagnostic'].values()) for r in failures),'coverage_failures_with_candidates_near_all_failed_endpoints':sum(all(r['coverage_failure_endpoint_diagnostic'][k] for k in ('start','goal') if r['projections'][k]['distance']>40) for r in failures),'connectivity_failures':len(connect),'connectivity_failures_with_direct_component_pair_proposal':sum(r['direct_component_pair_proposal'] for r in connect),'connectivity_failures_with_adjacent_component_proposal':sum(bool(r['adjacent_component_candidate_ids']) for r in connect),'proximity_radius_px':radius,'interpretation':'geometric proposal coverage only; no accepted graph, no new A* success, no GT recall'}
    return answer,summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r,s=evaluate(load_graph(a.graph),json.loads(a.candidates.read_text())['candidates'],json.loads(a.baseline.read_text()));a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/'pair_candidate_coverage.json',r);write_json(a.output/'pair_candidate_coverage_summary.json',s);print(json.dumps(s,indent=2))
if __name__=='__main__':main()
