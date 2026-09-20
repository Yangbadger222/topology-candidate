import json
from topology_repair.graph_io import RoadGraph
from topology_repair.candidate_generation import generate_candidates
from topology_repair.derive_action import derive_action

def graph(edges,n=5):
    return RoadGraph.from_dict({"nodes":[{"id":i,"x":i*10.,"y":0.} for i in range(n)],"edges":[{"source":a,"target":b} for a,b in edges]})

def test_duplicate_edges_and_endpoints():
    g=graph([(0,1),(1,0),(1,2)]); assert g.edges==[(0,1),(1,2)]; assert set(g.endpoints({0,1,2}))=={0,2}

def test_false_component_overrides_connect():
    assert derive_action("FALSE","CONNECT")=="DELETE_COMPONENT"

def test_candidate_generation():
    g=graph([(0,1),(3,4)],5); rows=generate_candidates(g,radius=25,top_k=3); assert rows and rows[0].gap_px <= 25

def test_uncertain_review():
    assert derive_action("REAL","UNCERTAIN")=="REVIEW"
