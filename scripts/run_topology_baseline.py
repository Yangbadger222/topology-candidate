#!/usr/bin/env python
"""Generate candidates and a machine-readable manifest for a scene."""
import argparse, json
from topology_repair.graph_io import load_graph
from topology_repair.candidate_generation import generate_candidates
from topology_repair.rendering import render_candidate

p=argparse.ArgumentParser(); p.add_argument("--graph",required=True); p.add_argument("--image"); p.add_argument("--scene-id",default="scene"); p.add_argument("--out-dir",default="data"); a=p.parse_args()
g=load_graph(a.graph); rows=generate_candidates(g,a.scene_id)
import pathlib; root=pathlib.Path(a.out_dir); (root/"candidates").mkdir(parents=True,exist_ok=True)
for c in rows:
    c.image=str(root/"candidates"/(c.sample_id+".png"))
    if a.image: render_candidate(a.image,g,c,c.image)
json.dump([c.to_dict() for c in rows],open(root/"candidates"/(a.scene_id+".json"),"w",encoding="utf-8"),indent=2)
print(f"{len(rows)} candidates written to {root/'candidates'}")
