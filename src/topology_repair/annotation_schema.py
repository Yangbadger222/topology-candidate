"""Versioned normalized annotation schema and crash-safe persistence."""
from datetime import datetime, timezone
import hashlib, json, os, tempfile
from pathlib import Path

SCHEMA_VERSION=2
SOURCE_REASON_CODES=["ROOFTOP","BUILDING_EDGE","SHADOW","VEGETATION","NON_ROAD_LINEAR_STRUCTURE","AMBIGUOUS_SOURCE","OTHER"]
ENDPOINT_REASON_CODES=["TRUE_DEAD_END","PARALLEL_ROAD","WRONG_TARGET","OCCLUDED_ROAD","CORRECT_TARGET_MISSING","AMBIGUOUS_CONNECTION","OTHER"]
# Compatibility union used by older callers.
REASON_CODES=list(dict.fromkeys(SOURCE_REASON_CODES+ENDPOINT_REASON_CODES+["AMBIGUOUS_IMAGE"]))

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def new_annotation(scene_id, source, annotator="human", provenance=None):
    now=datetime.now(timezone.utc).isoformat()
    return {"schema_version":SCHEMA_VERSION,"scene_id":scene_id,"source_subgraph_id":source["source_subgraph_id"],
            "source_type":source["source_type"],"component_validity":"UNCERTAIN",
            "source_reason_codes":[],"source_notes":"","endpoint_labels":{},
            "annotator":annotator,"created_at":now,"updated_at":now,"provenance":provenance or {}}

def update_endpoint(annotation, endpoint_id, selection, reason_codes=None, notes="", candidate_ids=None):
    from .derive_action import derive_action
    if selection not in set(candidate_ids or []) | {"NO_CONNECTION","CORRECT_TARGET_NOT_PROPOSED","UNCERTAIN"}: raise ValueError("unknown endpoint selection")
    label={"selection":selection,"derived_action":derive_action(annotation["component_validity"],selection),
        "reason_codes":list(reason_codes or []),"notes":notes}
    if selection.startswith("candidate_"):
        import re
        match=re.search(r"(?:candidate[_-])(\d+)$", selection)
        if match: label["candidate_rank"]=int(match.group(1))
    annotation["endpoint_labels"][str(endpoint_id)]=label
    annotation["updated_at"]=datetime.now(timezone.utc).isoformat(); return annotation

def update_source(annotation, component_validity, reason_codes=None, notes=None):
    """Update source-level labels and synchronise derived actions for all endpoints."""
    if component_validity not in {"REAL", "FALSE", "UNCERTAIN"}: raise ValueError("invalid component validity")
    annotation["component_validity"] = component_validity
    annotation["source_reason_codes"] = list(reason_codes or [])
    if notes is not None: annotation["source_notes"] = notes
    from .derive_action import derive_action
    for label in annotation.get("endpoint_labels", {}).values():
        label["derived_action"] = derive_action(component_validity, selection=label.get("selection"))
    annotation["updated_at"] = datetime.now(timezone.utc).isoformat()
    return annotation

def atomic_save(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(payload,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_annotations(path):
    path=Path(path)
    if not path.exists(): return {}
    data=json.loads(path.read_text(encoding="utf-8"))
    rows=data if isinstance(data,list) else data.get("annotations",[])
    return {(r["source_subgraph_id"],str(ep)): {**r,"source_annotation":r} for r in rows for ep in (r.get("endpoint_labels") or {"_":{}})} if isinstance(rows,list) else {}
