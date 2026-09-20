import json
from PIL import Image
from topology_repair.graph_io import RoadGraph
from topology_repair.source_mining import mine_sources
from topology_repair.candidate_generation import generate_candidates, project_point_to_edge
from topology_repair.derive_action import derive_action
from topology_repair.annotation_schema import atomic_save, new_annotation, update_endpoint

def test_curved_polyline_is_preserved_and_length_is_arc_length():
    g=RoadGraph.from_dict({"nodes":[{"id":0,"x":0,"y":0},{"id":1,"x":10,"y":0}],"edges":[{"source":0,"target":1,"coordinates":[[0,0],[5,5],[10,0]]}]})
    assert g.edges[0].coordinates == ((0.,0.),(5.,5.),(10.,0.)); assert round(g.component_length({0,1}),4)==14.1421

def test_duplicate_edges_and_endpoint_detection():
    g=RoadGraph.from_dict({"nodes":[{"id":i,"x":i*10.,"y":0} for i in range(3)],"edges":[{"source":0,"target":1},{"source":1,"target":0},{"source":1,"target":2}]})
    assert g.edges == [(0,1),(1,2)]; assert g.endpoints({0,1,2}) == [0,2]

def test_projection_uses_polyline_segment():
    g=RoadGraph.from_dict({"nodes":[{"id":0,"x":0,"y":0},{"id":1,"x":10,"y":0}],"edges":[{"source":0,"target":1,"polyline":[[0,0],[5,5],[10,0]]}]})
    d,x,y=project_point_to_edge(5,4,g.edges[0]); assert y > 3 and d < 1.1

def branch_graph():
    return RoadGraph.from_dict({"nodes":[{"id":0,"x":0,"y":0},{"id":1,"x":10,"y":0},{"id":2,"x":20,"y":0},{"id":3,"x":10,"y":10},{"id":4,"x":10,"y":20},{"id":5,"x":100,"y":0}],"edges":[{"source":0,"target":1},{"source":1,"target":2},{"source":1,"target":3},{"source":3,"target":4},{"source":5,"target":5}]})

def test_main_component_leaf_branch_is_mined():
    sources=mine_sources(branch_graph(),min_branch_length_px=1); leaves=[s for s in sources if s.source_type=="leaf_branch"]
    assert leaves and any(s.attachment_node_id==1 and s.source_nodes=={3,4} for s in leaves)

def test_leaf_branch_delete_does_not_delete_junction():
    s=next(s for s in mine_sources(branch_graph(),min_branch_length_px=1) if s.source_nodes=={3,4}); assert s.attachment_node_id not in s.source_nodes
    assert derive_action("FALSE","NO_CONNECTION")=="DELETE_SOURCE_SUBGRAPH"

def test_multiple_endpoints_labels_are_separate():
    source={"source_subgraph_id":"source_1","source_type":"disconnected_component"}; a=new_annotation("s",source)
    a["component_validity"]="REAL"; update_endpoint(a,85,"candidate_1",candidate_ids=["candidate_1"]); update_endpoint(a,91,"NO_CONNECTION")
    assert a["component_validity"]=="REAL" and a["endpoint_labels"]["85"]["selection"]=="candidate_1" and a["endpoint_labels"]["91"]["derived_action"]=="KEEP"

def test_candidate_options_and_false_priority():
    assert derive_action("FALSE","UNCERTAIN")=="DELETE_SOURCE_SUBGRAPH"
    assert derive_action("REAL","CORRECT_TARGET_NOT_PROPOSED")=="CANDIDATE_MISS"

def test_leaf_branch_can_target_same_component_outside_source():
    rows=generate_candidates(branch_graph(),radius=25,top_k=3,mining_config={"min_branch_length_px":1}); assert any(c.target_component==c.source_component and c.target_edge_id for c in rows)

def test_atomic_annotation_save_reload(tmp_path):
    p=tmp_path/"labels.json"; atomic_save(p,{"schema_version":2,"annotations":[]}); assert json.loads(p.read_text())["schema_version"]==2

def test_generation_without_magroad():
    g=RoadGraph.from_dict({"nodes":[{"id":0,"x":0,"y":0},{"id":1,"x":10,"y":0},{"id":2,"x":30,"y":0}],"edges":[{"source":0,"target":1}]})
    assert generate_candidates(g,radius=30,top_k=3)
