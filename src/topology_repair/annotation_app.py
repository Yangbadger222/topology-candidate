"""Run with: streamlit run -m topology_repair.annotation_app -- --cases cases.json"""
import json
from pathlib import Path

def run(cases_path, labels_path):
    import streamlit as st
    cases=json.load(open(cases_path,encoding="utf-8")); labels={r["sample_id"]:r for r in (json.load(open(labels_path,encoding="utf-8")) if Path(labels_path).exists() else [])}
    i=st.number_input("Case index",0,max(0,len(cases)-1),int(st.session_state.get("i",0))); c=cases[int(i)]
    if c.get("image") and Path(c["image"]).exists(): st.image(c["image"],caption=c["sample_id"],use_container_width=True)
    else: st.warning("此候选没有渲染图像；请先运行候选渲染脚本。")
    old=labels.get(c["sample_id"],{}); cv=st.radio("Component validity",["REAL","FALSE","UNCERTAIN"],index=["REAL","FALSE","UNCERTAIN"].index(old.get("component_validity","UNCERTAIN"))); cl=st.radio("Candidate connection",["CONNECT","NO_CONNECT","UNCERTAIN"],index=["CONNECT","NO_CONNECT","UNCERTAIN"].index(old.get("connection_label","UNCERTAIN"))); note=st.text_area("Optional note",old.get("notes",""))
    from .derive_action import derive_action
    def save():
        labels[c["sample_id"]]={**c,"component_validity":cv,"connection_label":cl,"derived_action":derive_action(cv,cl),"notes":note,"annotator":"human"}; Path(labels_path).parent.mkdir(parents=True,exist_ok=True); json.dump(list(labels.values()),open(labels_path,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    if st.button("Save"): save(); st.success("已保存")
    a,b,d=st.columns(3)
    if a.button("Previous"): save(); st.session_state.i=max(0,int(i)-1); st.rerun()
    if b.button("Save & Next"): save(); st.session_state.i=min(len(cases)-1,int(i)+1); st.rerun()
    if d.button("Skip"): st.session_state.i=min(len(cases)-1,int(i)+1); st.rerun()

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--cases",required=True); p.add_argument("--labels",default="data/labels/annotations.json"); a=p.parse_args(); run(a.cases,a.labels)
