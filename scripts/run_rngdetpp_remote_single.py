#!/usr/bin/env python3
"""Run the official RNGDet++ graph detector on one prepared image.

This is intentionally a thin single-image adapter around the repository's
Agent class.  The model itself is unchanged; the image is letterboxed to a
square tile and the resulting graph is exported as JSON/CSV plus an overlay.
"""
import argparse
import csv
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from models.detr import build_model
from agent import Agent


def make_args(args):
    # Values mirror cityscale/bash/run_test_RNGDet++.sh and the checkpoint.
    return SimpleNamespace(
        savedir=args.savedir, dataroot=args.dataroot, lr_backbone=1e-5,
        backbone="resnet101", dilation=False, position_embedding="sine",
        hidden_dim=256, dropout=0, nheads=8, dim_feedforward=2048,
        enc_layers=6, dec_layers=6, pre_norm=False, num_queries=10,
        aux_loss=True, frozen_weights=None, multi_scale=True,
        instance_seg=True, eos_coef=.1, set_cost_class=1, set_cost_bbox=5,
        set_cost_giou=2, mask_loss_coef=1, dice_loss_coef=1,
        bbox_loss_coef=5, giou_loss_coef=2, masks=True,
        ROI_SIZE=args.roi_size, image_size=args.image_size,
        logit_threshold=args.logit_threshold,
        candidate_filter_threshold=args.candidate_filter_threshold,
        extract_candidate_threshold=args.extract_candidate_threshold,
        alignment_distance=args.alignment_distance,
        filter_distance=10, process_boundary=True,
        agent_savedir=os.path.join(args.savedir, "test"),
    )


def letterbox(src, size):
    im = Image.open(src).convert("RGB")
    scale = min(size / im.width, size / im.height)
    nw, nh = round(im.width * scale), round(im.height * scale)
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), (0, 0, 0))
    out.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--savedir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--roi-size", type=int, default=128)
    p.add_argument("--logit-threshold", type=float, default=.7)
    p.add_argument("--candidate-filter-threshold", type=int, default=30)
    p.add_argument("--extract-candidate-threshold", type=float, default=.7)
    p.add_argument("--alignment-distance", type=int, default=5)
    a = p.parse_args()

    # Agent expects cityscale's directory layout and a 3-channel PNG.
    image_dir = os.path.join(a.dataroot, "20cities")
    os.makedirs(image_dir, exist_ok=True)
    letterbox(a.input, a.image_size).save(os.path.join(image_dir, "region_single_sat.png"))
    os.makedirs(os.path.join(a.savedir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(a.savedir, "test", "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(a.savedir, "test", "json"), exist_ok=True)
    os.makedirs(os.path.join(a.savedir, "test", "skeleton"), exist_ok=True)

    cfg = make_args(a)
    model, _ = build_model(cfg)
    state = torch.load(a.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.cuda().eval()

    agent = Agent(cfg, model, "single")
    with torch.no_grad():
        while not agent.finish_current_image:
            agent.step_counter += 1
            sat, hist = agent.crop_ROI(agent.current_coord)
            sat = torch.from_numpy(sat).float().permute(2, 0, 1)[None].cuda() / 255.
            hist = torch.from_numpy(hist).float()[None, None].cuda() / 255.
            out = model(sat, hist)
            agent.step(out["pred_logits"], out["pred_boxes"], thr=cfg.logit_threshold)

    edges = agent.historical_edges
    # Deduplicate undirected edges and assign stable node ids.
    nodes = {}
    for u, v in edges:
        for pt in (u, v):
            nodes.setdefault(tuple(pt), f"n{len(nodes):03d}")
    unique = set()
    edge_rows = []
    for u, v in edges:
        ku, kv = tuple(u), tuple(v)
        key = tuple(sorted((ku, kv)))
        if key in unique:
            continue
        unique.add(key)
        edge_rows.append({"source": nodes[ku], "target": nodes[kv],
                          "source_x": ku[0], "source_y": ku[1],
                          "target_x": kv[0], "target_y": kv[1]})
    node_rows = [{"id": nid, "x": pt[0], "y": pt[1]}
                 for pt, nid in nodes.items()]
    result = {"image": a.input, "coordinate_system": "letterboxed image pixels",
              "image_size": [a.image_size, a.image_size],
              "nodes": node_rows, "edges": edge_rows}
    out_json = os.path.join(a.savedir, "road_graph.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.savedir, "road_graph_edges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=edge_rows[0].keys() if edge_rows else
                           ["source", "target"]); w.writeheader(); w.writerows(edge_rows)

    over = letterbox(a.input, a.image_size)
    draw = ImageDraw.Draw(over, "RGBA")
    for e in edge_rows:
        draw.line((e["source_x"], e["source_y"], e["target_x"], e["target_y"]),
                  fill=(0, 220, 255, 220), width=3)
    for n in node_rows:
        x, y = n["x"], n["y"]
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 40, 40, 230), outline=(255,255,255,255))
        draw.text((x+6, y-6), n["id"], fill=(255,255,0,255))
    over.save(os.path.join(a.savedir, "road_graph_overlay.png"))
    print(json.dumps({"nodes": len(node_rows), "edges": len(edge_rows), "output": out_json}))


if __name__ == "__main__":
    main()
