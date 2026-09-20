import importlib.util

from topology_repair.annotation_logic import candidate_rank_from_id, is_candidate_selection, is_case_completed
from topology_repair.annotation_schema import new_annotation, update_endpoint, update_source
from topology_repair.derive_action import derive_action, derive_source_action

def _source():
    return {"source_subgraph_id": "source_00001", "source_type": "leaf_branch"}

def test_prefixed_candidate_id_and_action():
    value = "xjtlu_001_source_00001_ep_85_candidate_001"
    assert is_candidate_selection(value)
    assert candidate_rank_from_id(value) == 1
    assert derive_action("REAL", selection=value) == "ADD_CONNECTION"
    assert candidate_rank_from_id("candidate_001") == 1
    assert candidate_rank_from_id("road_005") is None

def test_update_endpoint_stores_rank_and_history_restores():
    a = new_annotation("s", _source())
    cid = "x_source_ep_85_candidate_003"
    update_source(a, "REAL")
    update_endpoint(a, 85, cid, candidate_ids=[cid])
    assert a["endpoint_labels"]["85"]["candidate_rank"] == 3
    update_source(a, "FALSE", ["ROOFTOP"])
    assert a["endpoint_labels"]["85"]["selection"] == cid
    assert a["endpoint_labels"]["85"]["derived_action"] == "DELETE_SOURCE_SUBGRAPH"
    update_source(a, "REAL")
    assert a["endpoint_labels"]["85"]["derived_action"] == "ADD_CONNECTION"

def test_no_connection_history_restores_keep():
    a = new_annotation("s", _source()); update_source(a, "REAL")
    update_endpoint(a, 85, "NO_CONNECTION")
    update_source(a, "FALSE"); update_source(a, "REAL")
    assert a["endpoint_labels"]["85"]["derived_action"] == "KEEP"

def test_case_completion_is_per_page():
    base = {"source_subgraph_id": "s", "source_endpoint_id": 85}
    assert not is_case_completed(base, None)
    a = new_annotation("s", _source())
    assert is_case_completed(base, a)  # new annotations default to UNCERTAIN/review
    update_source(a, "FALSE")
    assert is_case_completed(base, a)
    update_source(a, "REAL"); update_endpoint(a, 85, "NO_CONNECTION")
    assert is_case_completed(base, a)
    assert not is_case_completed({**base, "source_endpoint_id": 91}, a)

def test_source_only_completion_and_actions():
    a = new_annotation("s", {"source_subgraph_id": "loop", "source_type": "disconnected_component"})
    c = {"source_subgraph_id": "loop", "source_endpoint_id": None}
    assert is_case_completed(c, a)  # default UNCERTAIN is already a completed review state
    update_source(a, "REAL")
    assert is_case_completed(c, a)
    assert derive_source_action("REAL") == "KEEP"
    update_source(a, "FALSE")
    assert derive_source_action("FALSE") == "DELETE_SOURCE_SUBGRAPH"
    assert derive_source_action("UNCERTAIN") == "REVIEW"

def test_dynamic_topk_and_special_selections():
    a = new_annotation("s", _source()); update_source(a, "REAL")
    ids = [f"s_source_ep_85_candidate_{i:03d}" for i in range(1, 6)]
    for cid in ids: update_endpoint(a, 85, cid, candidate_ids=ids)
    assert a["endpoint_labels"]["85"]["candidate_rank"] == 5
    update_endpoint(a, 85, "CORRECT_TARGET_NOT_PROPOSED")
    assert a["endpoint_labels"]["85"]["derived_action"] == "CANDIDATE_MISS"
    update_endpoint(a, 85, "NO_CONNECTION")
    assert a["endpoint_labels"]["85"]["derived_action"] == "KEEP"

def test_summary_counts_prefixed_candidate_and_source_action():
    spec = importlib.util.spec_from_file_location("summary", "scripts/summarize_topology_annotations.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = {"scene_id": "s", "annotations": [
        {"source_subgraph_id": "loop", "source_type": "disconnected_component", "component_validity": "FALSE", "source_reason_codes": ["ROOFTOP"], "endpoint_labels": {}},
        {"source_subgraph_id": "x", "source_type": "leaf_branch", "component_validity": "REAL", "endpoint_labels": {"85": {"selection": "x_source_ep_85_candidate_001", "derived_action": "ADD_CONNECTION"}}},
    ]}
    result = mod.summarize(data)
    assert result["source_actions"]["DELETE_SOURCE_SUBGRAPH"] == 1
    assert result["candidate_selected_rank"]["T1"] == 1
