"""Check coordinate safety, obstacle rejection and navigational edge projection."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import networkx as nx
from graph_utils import load_graph
from graph_refine import parser,connect,parallel,rebuild
from evaluate_navigation_graph import route


def graph(segments):
    g=nx.Graph();g.graph['inference_shape']=[100,150]
    for a,b in segments:
        for n,q in (a,b):g.add_node(n,x=float(q[0]),y=float(q[1]))
        g.add_edge(a[0],b[0],geometry=[list(a[1]),list(b[1])],length=float(np.linalg.norm(np.array(a[1])-b[1])))
    return g


def config(tmp_path):
    return parser().parse_args(['--graph',str(tmp_path/'g.json'),'--output',str(tmp_path)])


def test_rc_parser(tmp_path):
    p=tmp_path/'g.json';p.write_text(json.dumps({'coordinate_order':'row,column','inference_shape':[100,150],'nodes_rc':[[3,40],[5,60]],'edges':[[0,1]]}))
    g=load_graph(p);assert g.nodes[0]['x']==40 and g.nodes[0]['y']==3


def test_gap_obstacle_and_facing(tmp_path):
    g=graph([((0,(10,50)),(1,(40,50))),((2,(60,50)),(3,(90,50)))])
    c=config(tmp_path);c.gap_score=.4
    p=np.ones((100,150))*.9
    h,a,_=connect(g,p,c,False);assert a['accepted']==1 and nx.has_path(h,0,3)
    p[:,45:56]=0
    h,a,_=connect(g,p,c,False);assert a['accepted']==0 and not nx.has_path(h,0,3)
    assert connect(g,None,c,False)[1]['accepted']==0


def test_projection_same_edge():
    g=graph([((0,(10,50)),(1,(100,50)))])
    r,_,_=route(g,{'start':[30,52],'goal':[70,49]},10)
    assert r['success'] and abs(r['path_length']-40)<1e-6
    assert len(g)==2 # immutable source


def test_parallel_attachments(tmp_path):
    g=graph([((0,(10,40)),(1,(100,40))),((2,(10,48)),(3,(100,48))),((3,(100,48)),(4,(100,80))),((3,(100,48)),(5,(130,48)))])
    h,a,_=parallel(g,np.ones((100,150))*.8,config(tmp_path))
    assert len(a['merged_parallel_groups'])==1
    assert 4 in h and h.degree(4)==1 and nx.number_connected_components(h)==1


def test_rebuild_preserves_orientation():
    g=graph([((0,(10,50)),(1,(100,50)))])
    g.edges[0,1]['geometry']=[[100,50],[50,40],[10,50]]
    h=rebuild(g,{})
    assert h.edges[0,1]['geometry']==[[10.,50.],[50.,40.],[100.,50.]]
