import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import networkx as nx
import numpy as np
from candidate_generator_v2 import parser
from connection_candidates.geometry import Context,probability_features,estimate_endpoint_tangent
from connection_candidates.local import generate_local
from corridor_proposal import pixel_astar,component_pairs


def graph(nodes,edges):
    g=nx.Graph()
    for n,q in enumerate(nodes):g.add_node(n,x=q[0],y=q[1])
    for u,v in edges:g.add_edge(u,v,geometry=[list(nodes[u]),list(nodes[v])],length=float(np.linalg.norm(np.array(nodes[u])-nodes[v])))
    return g


def config():return parser().parse_args(['--graph','x','--probability','p','--output','o'])


def test_t_junction_no_mutation():
    g=graph([(10,30),(90,30),(50,60),(50,90)],[(0,1),(2,3)]);before=copy.deepcopy(g);rows,_=generate_local(Context(g,np.ones((100,100))),config());rr=[r for r in rows if r['type']=='endpoint_edge' and r['source']['node_id']==2];assert len(rr)==1 and rr[0]['target']['xy']==[50,30] and rr[0]['status']=='proposed';assert nx.utils.graphs_equal(g,before)


def test_junction_and_dedup():
    g=graph([(50,30),(10,30),(90,30),(50,5),(50,60),(50,90)],[(0,1),(0,2),(0,3),(4,5)]);rows,_=generate_local(Context(g,np.ones((100,100))),config());rr=[r for r in rows if r['source']['node_id']==4 and np.linalg.norm(np.array(r['target']['xy'])-[50,30])<1];assert len(rr)==1 and rr[0]['type']=='endpoint_junction';assert rr[0]['geometry_features']['junction_direction_compatibility']==1


def test_parallel_not_junction():
    g=graph([(10,20),(90,20),(10,40),(90,40)],[(0,1),(2,3)]);rr,_=generate_local(Context(g,np.ones((100,100))),config());assert not any(r['type']=='endpoint_junction' for r in rr);assert all(r['status']=='rejected_by_hard_gate' for r in rr if r['geometry_features']['max_angle_error']>=90)


def test_probability_sampling_xy():
    p=np.zeros((20,50));p[7,:]=.8;f=probability_features([[2,7],[40,7]],p);assert abs(f['mean']-.8)<1e-6 and f['sample_count']>=39;assert f['fraction_below_0.2']==0


def test_probability_long_run():
    p=np.ones((5,40));p[:,10:25]=0;f=probability_features([[0,2],[39,2]],p);assert .3<f['longest_low_probability_run_ratio']<.5


def test_high_probability_path():
    p=np.zeros((30,50));p[12,3:47]=1;path,_=pixel_astar(1+8*(1-p)**2,[3,12],[46,12],0);assert all(y==12 for x,y in path)


def test_obstacle_and_crop_global():
    cost=np.ones((30,50));cost[10:20,20:30]=100;path,_=pixel_astar(cost,[105,115],[145,115],0,(100,100));q=np.array(path);assert q[0].tolist()==[105,115] and q[-1].tolist()==[145,115];assert not any(120<=x<130 and 110<=y<120 for x,y in path)


def test_component_pruning():
    g=graph([(1,1),(5,1),(20,1),(25,1),(900,1),(905,1)],[(0,1),(2,3),(4,5)]);c=config();c.component_search_radius=50;assert component_pairs(Context(g,np.ones((10,1000))),c)==[(0,1)]


def test_tangent_lookback():
    g=graph([(50,50),(49,50),(30,50),(10,50)],[(0,1),(1,2),(2,3)]);assert np.allclose(estimate_endpoint_tangent(g,0),[1,0])


def test_blank_region_gate():
    from connection_candidates.geometry import make_candidate
    g=graph([(1,5),(5,5),(30,5),(35,5)],[(0,1),(2,3)]);ctx=Context(g,np.zeros((10,40)));row=make_candidate(ctx,'component_corridor',{'component_id':0},{'component_id':1},[[5,5],[30,5]],{});assert row['probability_features']['mean']==0 and row['ranking_subscores']['probability_score']==0


def test_seed_region_offset():
    cost=np.ones((20,40));path,_=pixel_astar(cost,[2.4,5.4],[32.2,5.4],3);assert np.linalg.norm(np.array(path[0])-[2.4,5.4])<=3 and np.linalg.norm(np.array(path[-1])-[32.2,5.4])<=3


def test_native_same_optimal_cost_as_reference():
    from corridor_proposal import python_pixel_astar
    rng=np.random.default_rng(3);cost=(1+rng.random((20,35))*7).astype(np.float32)
    a,_=pixel_astar(cost,[3,6],[30,14],2,(0,0));b,_=python_pixel_astar(cost,[3,6],[30,14],2,(0,0))
    def pathcost(path):return sum(np.linalg.norm(np.array(v)-u)*(cost[u[1],u[0]]+cost[v[1],v[0]])*.5 for u,v in zip(path,path[1:]))
    assert abs(pathcost(a)-pathcost(b))<1e-5


def test_third_component_split_proposals():
    from corridor_proposal import generate_corridors
    g=graph([(5,30),(15,30),(85,30),(95,30),(50,10),(50,50)],[(0,1),(2,3),(4,5)]);c=config();c.component_search_radius=100;c.max_component_neighbors=3;c.corridor_window_margin=10;c.max_corridors_per_component_pair=1
    rows,_=generate_corridors(Context(g,np.ones((60,100))),c)
    parents=[r for r in rows if 'third_component_requires_split' in r['rejection_flags']]
    assert parents and any('split_from_component_pair' in r['geometry_features'] for r in rows)
    assert all(r['status']=='rejected_by_hard_gate' for r in parents)
