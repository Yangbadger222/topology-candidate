"""Streamlit UI only; case construction lives in build_topology_annotation_cases.py."""
import json
from pathlib import Path
from .annotation_schema import atomic_save, new_annotation, update_endpoint, update_source, SOURCE_REASON_CODES, ENDPOINT_REASON_CODES
from .derive_action import derive_action
from .annotation_logic import is_case_completed

def _read_cases(path):
    data=json.loads(Path(path).read_text(encoding="utf-8")); return (data.get("cases",[]),data) if isinstance(data,dict) else (data,{})

def _read_labels(path, scene_id, provenance):
    if not Path(path).exists(): return {"schema_version":2,"scene_id":scene_id,"annotations":[],"provenance":provenance}
    data=json.loads(Path(path).read_text(encoding="utf-8")); return data if isinstance(data,dict) else {"schema_version":2,"scene_id":scene_id,"annotations":data,"provenance":provenance}

def _save_labels(path, payload): atomic_save(path,payload)

def run(cases_path, labels_path, annotator="human"):
    import streamlit as st
    cases,meta=_read_cases(cases_path)
    if not cases: st.warning("没有可标注的 source endpoint case。"); return
    scene_id=meta.get("scene_id",cases[0].get("scene_id","scene")); labels=_read_labels(labels_path,scene_id,meta.get("provenance",{})); records={r["source_subgraph_id"]:r for r in labels.get("annotations",[])}
    source_types=sorted({c.get("source_type") for c in cases}); validity_filter=st.sidebar.selectbox("Validity filter",["ALL","UNLABELED","REAL","FALSE","UNCERTAIN"]); type_filter=st.sidebar.selectbox("Source type",["ALL"]+source_types)
    filtered=[]
    for c in cases:
        r=records.get(c["source_subgraph_id"]); v=r.get("component_validity") if r else None
        if validity_filter=="UNLABELED" and is_case_completed(c, r): continue
        if validity_filter in ("REAL","FALSE","UNCERTAIN") and v != validity_filter: continue
        if type_filter!="ALL" and c.get("source_type")!=type_filter: continue
        filtered.append(c)
    if not filtered: st.info("当前筛选没有 case"); return
    ids=[c["source_subgraph_id"]+"/"+str(c["source_endpoint_id"]) for c in filtered]
    current=int(st.session_state.get("page",0)); current=min(current,len(filtered)-1)
    jump=st.text_input("Jump to index / sample id",value="")
    if jump:
        if jump.isdigit(): current=max(0,min(len(filtered)-1,int(jump)))
        elif jump in ids: current=ids.index(jump)
    c=filtered[current]; sid=c["source_subgraph_id"]; endpoint_id=c.get("source_endpoint_id"); eid=str(endpoint_id) if endpoint_id is not None else None; source=c["source_inventory"]; old=records.get(sid)
    if old is None: old=new_annotation(scene_id,source,annotator,meta.get("provenance",{})); records[sid]=old
    st.title("Gemini Road Graph Topology Annotation V2"); st.caption(f"Scene: {scene_id} · Source: {sid} · Type: {c['source_type']} · Endpoint: {eid or 'None'} · Progress: {current+1}/{len(filtered)}")
    tabs=st.tabs(["Overlay Local","Raw Local","Overlay Context","Raw Context"])
    for tab,name in zip(tabs,["overlay_local","raw_local","overlay_context","raw_context"]):
        with tab: st.image(c.get("views",{}).get(name),use_container_width=True)
    cv=st.radio("Q1 · Is this source subgraph a real road?",["REAL","FALSE","UNCERTAIN"],index=["REAL","FALSE","UNCERTAIN"].index(old.get("component_validity","UNCERTAIN")),horizontal=True,key=f"valid_{sid}")
    source_reasons=st.multiselect("Source reason codes",SOURCE_REASON_CODES,default=old.get("source_reason_codes",[]),key=f"source_reason_{sid}")
    source_notes=st.text_area("Source notes",old.get("source_notes",""),key=f"source_notes_{sid}")
    old_ep=old.get("endpoint_labels",{}).get(eid or "",{})
    options=[f"T{x['rank']}" for x in c.get("candidates",[])]+["NO_CONNECTION","CORRECT_TARGET_NOT_PROPOSED","UNCERTAIN"]
    candidate_by_display={f"T{x['rank']}":x["candidate_id"] for x in c.get("candidates",[])}
    candidate_rank_map={x["candidate_id"]: x["rank"] for x in c.get("candidates",[])}
    display_old = next((f"T{x['rank']}" for x in c.get("candidates",[]) if x.get("candidate_id") == old_ep.get("selection")), old_ep.get("selection","UNCERTAIN"))
    if endpoint_id is None or cv != "REAL":
        selection=None
        if endpoint_id is not None: st.info("只有 REAL source 才需要回答 endpoint connection；当前页面不会覆盖 endpoint 选择历史。")
    else:
        selection=st.radio("Q2 · Where should this endpoint connect?",options,index=options.index(display_old) if display_old in options else len(options)-1,horizontal=True,key=f"sel_{sid}_{eid}")
    if endpoint_id is not None:
        reasons=st.multiselect("Endpoint reason codes",ENDPOINT_REASON_CODES,default=old_ep.get("reason_codes",[]),key=f"reason_{sid}_{eid}"); notes=st.text_area("Endpoint notes",old_ep.get("notes",""),key=f"notes_{sid}_{eid}")
    else:
        reasons=[]; notes=""
    def persist():
        old["annotator"]=annotator; update_source(old,cv,source_reasons,source_notes)
        if endpoint_id is not None and cv == "REAL":
            choice=candidate_by_display.get(selection,selection)
            update_endpoint(old,eid,choice,reasons,notes,candidate_by_display.values(),candidate_rank_map)
            update_source(old,cv,source_reasons,source_notes)
        labels.update({"schema_version":2,"scene_id":scene_id,"annotations":list(records.values()),"provenance":meta.get("provenance",{})}); _save_labels(labels_path,labels)
    a,b,d,e=st.columns(4)
    if a.button("Previous"): persist(); st.session_state.page=max(0,current-1); st.rerun()
    if b.button("Save & Next"): persist(); st.session_state.page=min(len(filtered)-1,current+1); st.rerun()
    if d.button("Skip"): st.session_state.page=min(len(filtered)-1,current+1); st.rerun()
    if e.button("Next Unlabeled"):
        persist()
        nxt=current
        for j,x in enumerate(filtered[current+1:], current+1):
            if not is_case_completed(x, records.get(x["source_subgraph_id"])):
                nxt=j; break
        st.session_state.page=nxt; st.rerun()
    if st.button("Save"): persist(); st.success("已原子保存")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--cases",required=True); p.add_argument("--labels",required=True); p.add_argument("--annotator",default="human"); a=p.parse_args(); run(a.cases,a.labels,a.annotator)
