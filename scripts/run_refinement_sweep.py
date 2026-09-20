"""One-factor-at-a-time sensitivity sweep: seven runs, not a huge grid."""
from __future__ import annotations
import csv
import copy
from graph_refine import parser,run


def main() -> None:
    """Vary 8/12/16 parallel, 10/20/30 snap and 20/30/40 gap pixels."""
    c=parser().parse_args();base=copy.deepcopy(c);base.parallel_distance=12;base.snap_distance=20;base.max_gap_distance=30;base.no_visuals=True
    experiments=[('baseline',base)]
    for key,values in [('parallel_distance',[8,16]),('snap_distance',[10,30]),('max_gap_distance',[20,40])]:
        for v in values:
            item=copy.deepcopy(base);setattr(item,key,v);experiments.append((f'{key}_{v}',item))
    rows=[]
    for name,item in experiments:
        item.output=c.output/'sweep'/name;r=run(item);s=r['refined']
        rows.append({'run':name,'parallel_distance':item.parallel_distance,'snap_distance':item.snap_distance,'max_gap':item.max_gap_distance,
            **{k:s[k] for k in ['nodes','edges','components','largest_component_ratio','endpoints']},
            'new_gap_edges':r['stages'][4]['actions']['accepted'],'snapped_pairs':r['stages'][3]['actions']['accepted'],'removed_spurs':r['stages'][5]['actions']['removed_spurs']})
    with (c.output/'sweep_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
