"""Synthetic tests check invariants, not training data or measured model accuracy."""
import sys,json
from pathlib import Path
import networkx as nx
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from learned_refiner_data import read_gt,validate_splits,edge_label,gap_label,gap_candidates
from convert_d_centerlines import convert

def road():
    g=nx.Graph();g.add_node(0,x=0,y=50);g.add_node(1,x=100,y=50);g.add_edge(0,1,geometry=[[0,50],[100,50]],length=100);return g

def test_reviewed_coverage_gate(tmp_path):
    d={'coordinate_order':'x,y','units':'inference_pixel','image_size':[100,100],'nodes':[],'edges':[],'annotation_status':'draft','coverage_complete':False}
    p=tmp_path/'gt.json';p.write_text(json.dumps(d))
    with pytest.raises(ValueError):read_gt(p,[100,100])

def test_spatial_leakage():
    r=lambda i,s,b:{'id':i,'split':s,'units':'inference_pixel','bbox_xyxy':b,'geographic_frame':'same'}
    with pytest.raises(ValueError):validate_splits({'regions':[r('a','train',[0,0,512,512]),r('b','test',[512,0,1024,512])]})
    validate_splits({'regions':[r('a','train',[0,0,512,512]),r('b','test',[1200,0,1712,512])]})

def test_full_geometry_labels():
    g=road()
    assert edge_label([[10,50],[90,50]],g)['label']==1
    assert edge_label([[10,90],[90,90]],g)['label']==0
    assert edge_label([[10,50],[50,90],[90,50]],g)['label']!=1
    assert edge_label([[10,62],[90,62]],g)['label']==-1

def test_gap_corridor_projection():
    g=road();assert gap_label({'geometry':[[30,50],[60,50]],'distance':30},g)['label']==1
    assert gap_label({'geometry':[[30,90],[60,90]],'distance':30},g)['label']==0
    h=nx.Graph();h.add_node(0,x=0,y=0);h.add_node(1,x=100,y=0);h.add_edge(0,1,geometry=[[0,0],[0,100],[100,100],[100,0]],length=300)
    assert gap_label({'geometry':[[0,0],[100,0]],'distance':100},h)['label']!=1

def test_candidate_geometry():
    g=nx.Graph()
    for n,x in enumerate((0,30,60,90)):g.add_node(n,x=x,y=50)
    g.add_edge(0,1,geometry=[[0,50],[30,50]],length=30);g.add_edge(2,3,geometry=[[60,50],[90,50]],length=30)
    rows=gap_candidates(g);assert [(r['u'],r['v']) for r in rows]==[(1,2)]

def test_d_conversion_does_not_approve():
    d=convert({'image_size':{'width':1024,'height':1024},'region_id':'r','image_id':'i','segments':[{'points':[[0,0],[10,0]],'path_type':'vehicle_road','edge_id':'e'}]})
    assert d['annotation_status']=='draft' and not d['coverage_complete']
