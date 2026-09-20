#!/usr/bin/env python
"""Build V2 source-endpoint annotation cases from only RGB + Gemini JSON."""
import argparse, json, subprocess, warnings
from pathlib import Path
import yaml
from PIL import Image
from topology_repair.graph_io import load_graph
from topology_repair.source_mining import mine_sources, inventory
from topology_repair.candidate_generation import generate_candidates
from topology_repair.rendering import render_source_views
from topology_repair.annotation_schema import sha256_file

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image",required=True); p.add_argument("--graph",required=True); p.add_argument("--scene-id",required=True); p.add_argument("--output",required=True); p.add_argument("--config"); p.add_argument("--radius",type=float); p.add_argument("--top-k",type=int); a=p.parse_args()
    defaults={"search_radius_px":128.0,"top_k":3,"local_crop_size_px":512,"context_crop_size_px":1024,
      "max_disconnected_component_nodes":64,"max_disconnected_component_length_px":2000.0,"max_leaf_branch_nodes":64,
      "max_leaf_branch_length_px":1000.0,"min_branch_length_px":4.0}
    if a.config: defaults.update(yaml.safe_load(Path(a.config).read_text()) or {})
    if a.radius is not None: defaults["search_radius_px"]=a.radius
    if a.top_k is not None: defaults["top_k"]=a.top_k
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); graph=load_graph(a.graph)
    with Image.open(a.image) as im: image_size=(im.width, im.height)
    meta=graph.metadata or {}; gw,gh=meta.get("image_width"),meta.get("image_height")
    verified=False
    if gw is not None and gh is not None:
        if (image_size[0],image_size[1]) != (int(gw),int(gh)):
            raise ValueError(f"Satellite image size {image_size[0]}x{image_size[1]} does not match Gemini graph metadata {int(gw)}x{int(gh)}. Use the original image used for Gemini graph generation.")
        verified=True
    else:
        warnings.warn("Gemini graph metadata image_width/image_height is missing; image size was not verified.")
    mining={k:defaults[k] for k in defaults if k.startswith("max_") or k=="min_branch_length_px"}
    sources=mine_sources(graph,**mining); candidates=generate_candidates(graph,a.scene_id,defaults["search_radius_px"],defaults["top_k"],sources=sources)
    renders=out/"renders"; pages=[]
    for source in sources:
        endpoints = source.source_endpoint_ids or [None]
        for endpoint in endpoints:
            cs=[c for c in candidates if c.source_subgraph_id==source.source_subgraph_id and c.source_endpoint_id==endpoint]
            views=render_source_views(a.image,graph,source,endpoint,cs,renders,defaults["local_crop_size_px"],defaults["context_crop_size_px"])
            pages.append({"scene_id":a.scene_id,"source_subgraph_id":source.source_subgraph_id,"source_type":source.source_type,
                "source_component_id":source.source_component_id,"source_endpoint_id":endpoint,"source_inventory":source.to_dict(),"candidates":[c.to_dict() for c in cs],"views":views})
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit=None
    provenance={"graph_sha256":sha256_file(a.graph),"image_sha256":sha256_file(a.image),"graph_image_size_verified":verified,"candidate_config":defaults,"generator_git_commit":commit,"generator_version":"topology_repair_v2"}
    (out/"source_inventory.json").write_text(json.dumps(inventory(sources),indent=2),encoding="utf-8")
    (out/"cases.json").write_text(json.dumps({"schema_version":2,"scene_id":a.scene_id,"provenance":provenance,"cases":pages},indent=2),encoding="utf-8")
    print(f"built {len(pages)} source cases ({sum(c['source_endpoint_id'] is not None for c in pages)} endpoint, {sum(c['source_endpoint_id'] is None for c in pages)} source-only) and {len(candidates)} candidates in {out}")
if __name__=="__main__": main()
