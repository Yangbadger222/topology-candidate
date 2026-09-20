#!/usr/bin/env python3
"""Controlled full-graph regression for WildRoad and LoveDA-adapted MaGRoad.

This module intentionally contains no training code.  It mirrors MaGRoad's
official ``inferencer.infer_one_img`` graph path and records the intermediate
values that the original entry point discards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
import scipy
import torch
from PIL import Image, ImageDraw, ImageFont
from rtree import index as rtree_index


REPO = Path(__file__).resolve().parents[2]
BASE_CHECKPOINT = Path("/home/badger/datasets/overhead_checkpoints/MaGRoad/wildroad_vitb.ckpt")
BASE_SHA256 = "095bf4f1688d7172604ff855a22960173c869eeee20061ed61d92a07540ef8a6"
XJTLU_DIR = Path("/home/badger/sam-inference/MaGRoad/xjtlu_inputs")
OUTPUT_ROOT = Path("/home/badger/sam-inference/MaGRoad/graph_regression")

MODELS = {
    "baseline": {
        "label": "WildRoad baseline",
        "checkpoint": BASE_CHECKPOINT,
        "config": Path("config/toponet_vitb_1024_wild_road.yaml"),
        "road_threshold": 0.1,
        "train_mode": "wildroad_baseline",
    },
    "encoder_lora": {
        "label": "A: Encoder LoRA",
        "checkpoint": Path("magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt"),
        "adapter": Path("magroad_loveda/outputs/encoder/adapter_epoch_008.pt"),
        "config": Path("magroad_loveda/config/loveda/remote_encoder.yaml"),
        "road_threshold": 0.3,
        "train_mode": "loveda_lora_encoder",
    },
    "encoder_decoder": {
        "label": "B: LoRA + Decoder",
        "checkpoint": Path("magroad_loveda/outputs/encoder_decoder/checkpoints/best_road_iou.ckpt"),
        "adapter": Path("magroad_loveda/outputs/encoder_decoder/adapter_epoch_008.pt"),
        "config": Path("magroad_loveda/config/loveda/remote_encoder_decoder.yaml"),
        "road_threshold": 0.4,
        "train_mode": "loveda_lora_encoder_decoder",
    },
}

OFFICIAL_CROPS = [
    {"name": "wildroad_data0_x2200_y1900", "source": "/home/badger/datasets/WildRoad/test/test/data0.jpg", "x": 2200, "y": 1900},
    {"name": "wildroad_data1_x3600_y1900", "source": "/home/badger/datasets/WildRoad/test/test/data1.jpg", "x": 3600, "y": 1900},
    {"name": "wildroad_data2_x4800_y2400", "source": "/home/badger/datasets/WildRoad/test/test/data2.jpg", "x": 4800, "y": 2400},
]

G1 = "G1_controlled"
G2 = "G2_road_calibrated"
OFFICIAL = "official_regression"
MODEL_ORDER = ["baseline", "encoder_lora", "encoder_decoder"]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in state.items():
        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError(f"checkpoint key collision: {key}")
        result[key] = value
    return result


def state_category(name: str) -> str:
    if name.startswith("image_encoder.") and ".qkv.linear_" in name:
        return "lora"
    if name.startswith("image_encoder."):
        return "encoder_base"
    if name.startswith("map_decoder."):
        return "map_decoder"
    if name.startswith("topo_net."):
        return "toponet"
    return "other"


def load_state_file(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = canonical(raw.get("state_dict", raw))
    return raw, state


def tensor_group_comparison(base: dict[str, torch.Tensor], adapted: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    result = {key: {"equal": 0, "changed": 0, "missing": 0} for key in ("encoder_base", "map_decoder", "toponet")}
    for key, base_value in base.items():
        category = state_category(key)
        if category not in result:
            continue
        if key not in adapted:
            result[category]["missing"] += 1
        elif torch.equal(base_value, adapted[key]):
            result[category]["equal"] += 1
        else:
            result[category]["changed"] += 1
    return result


def verify_checkpoints(output_root: Path) -> dict[str, Any]:
    baseline_raw, baseline_state = load_state_file(BASE_CHECKPOINT)
    baseline_hash = sha256(BASE_CHECKPOINT)
    if baseline_hash != BASE_SHA256:
        raise RuntimeError(f"WildRoad SHA mismatch: {baseline_hash}")
    identities: dict[str, Any] = {
        "baseline": {
            "path": str(BASE_CHECKPOINT),
            "size_bytes": BASE_CHECKPOINT.stat().st_size,
            "sha256": baseline_hash,
            "epoch": baseline_raw.get("epoch"),
            "train_mode": "wildroad_baseline",
            "lora_rank": 0,
            "comparison_to_base": {
                "encoder_base": {"equal": sum(state_category(k) == "encoder_base" for k in baseline_state), "changed": 0, "missing": 0},
                "map_decoder": {"equal": sum(state_category(k) == "map_decoder" for k in baseline_state), "changed": 0, "missing": 0},
                "toponet": {"equal": sum(state_category(k) == "toponet" for k in baseline_state), "changed": 0, "missing": 0},
            },
        }
    }
    for model_name in ("encoder_lora", "encoder_decoder"):
        spec = MODELS[model_name]
        checkpoint = Path(spec["checkpoint"])
        adapter_path = Path(spec["adapter"])
        raw, state = load_state_file(checkpoint)
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        experiment_config = raw.get("experiment_config", {})
        validation_history = raw.get("validation_history", [])
        best_epoch = max(validation_history, key=lambda row: row.get("road_iou", -1)).get("epoch") if validation_history else None
        comparison = tensor_group_comparison(baseline_state, state)
        adapter_state = canonical(adapter["adapter_state"])
        adapter_matches_full = all(key in state and torch.equal(value, state[key]) for key, value in adapter_state.items())
        lora_b = [value for key, value in state.items() if state_category(key) == "lora" and ".linear_b_" in key]
        identity = {
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "adapter_path": str(adapter_path.resolve()),
            "adapter_size_bytes": adapter_path.stat().st_size,
            "adapter_sha256": sha256(adapter_path),
            "epoch": raw.get("epoch"),
            "global_step": raw.get("global_step"),
            "best_road_iou_epoch_from_checkpoint_history": best_epoch,
            "train_mode": experiment_config.get("TRAIN_MODE", adapter.get("config", {}).get("TRAIN_MODE")),
            "lora_rank": experiment_config.get("LORA_RANK", adapter.get("config", {}).get("LORA_RANK")),
            "adapter_epoch": adapter.get("epoch"),
            "adapter_matches_full_checkpoint": adapter_matches_full,
            "comparison_to_base": comparison,
            "nonzero_lora_b": sum(bool(torch.count_nonzero(value)) for value in lora_b),
            "lora_tensor_count": sum(state_category(key) == "lora" for key in state),
        }
        if identity["epoch"] != 8 or identity["adapter_epoch"] != 8 or best_epoch != 8:
            raise RuntimeError(f"{model_name}: selected checkpoint is not best-IoU epoch 8")
        if identity["train_mode"] != spec["train_mode"] or identity["lora_rank"] != 4:
            raise RuntimeError(f"{model_name}: checkpoint metadata mismatch")
        if not adapter_matches_full or identity["nonzero_lora_b"] != 24:
            raise RuntimeError(f"{model_name}: adapter/full checkpoint mismatch")
        if comparison["encoder_base"]["changed"] or comparison["toponet"]["changed"]:
            raise RuntimeError(f"{model_name}: frozen base encoder or TopoNet changed")
        expected_decoder_changes = 0 if model_name == "encoder_lora" else 10
        if comparison["map_decoder"]["changed"] != expected_decoder_changes:
            raise RuntimeError(f"{model_name}: unexpected decoder change count")
        identities[model_name] = identity
        del raw, state, adapter, adapter_state, lora_b
    dump_json(output_root / "checkpoint_identity.json", identities)
    return identities


def import_magroad():
    # inferencer parses CLI arguments during import; supply a benign fixed argv.
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = ["inferencer.py", "--device", "0"]
    import graph_extraction  # type: ignore
    import inferencer  # type: ignore
    from model import MaGRoad  # type: ignore
    from utils import load_config  # type: ignore

    return inferencer, graph_extraction, MaGRoad, load_config


def clone_inference_settings(target: Any, source: Any) -> None:
    names = [
        "INFER_BATCH_SIZE", "SAMPLE_MARGIN", "INFER_PATCHES_PER_EDGE",
        "ITSC_THRESHOLD", "ROAD_THRESHOLD", "TOPO_THRESHOLD",
        "ITSC_NMS_RADIUS", "ROAD_NMS_RADIUS", "NEIGHBOR_RADIUS",
        "MAX_NEIGHBOR_QUERIES", "USE_FAST_NMS", "PATCH_SIZE",
        "TOPO_SAMPLE_NUM", "TOPONET", "TOPONET_VERSION", "NUM_INTERPOLATIONS",
        "USE_POINT_FEATURES", "USE_EDGE_BIAS", "USE_GEOMETRIC_FEATURES",
        "USE_PATH_FEATURES", "POOL_KERNEL_SIZES", "ANGLE_SIGMA",
        "LAMBDA_TURN", "LAMBDA_COMPETE", "SUBDIVIDE_RESOLUTION",
        "INTERESTING_RADIUS", "INTR_SAMPLE_WEIGHT", "NOISE_SCALE",
    ]
    for name in names:
        target[name] = source[name]


def load_model(model_name: str, device: torch.device, MaGRoad: Any, load_config: Any) -> tuple[Any, Any]:
    spec = MODELS[model_name]
    official_config = load_config(str(MODELS["baseline"]["config"]))
    config = load_config(str(spec["config"]))
    clone_inference_settings(config, official_config)
    if model_name != "baseline":
        config.MAGROAD_INIT_ONLY = True
    net = MaGRoad(config)
    raw = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = canonical(raw.get("state_dict", raw))
    result = net.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{model_name}: non-strict checkpoint load")
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    net.to(device)
    if any(parameter.requires_grad for parameter in net.parameters()):
        raise RuntimeError("inference model unexpectedly has trainable parameters")
    return net, config


def percentile_stats(probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    values = probability.astype(np.float64, copy=False).reshape(-1)
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "max": float(values.max()),
        "p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99),
        "above_original_keypoint_threshold": int(np.count_nonzero(values > threshold)),
    }


def selected_keypoints_from_graph(keypoint_u8: np.ndarray, graph_points_xy: np.ndarray, config: Any, graph_extraction: Any) -> tuple[np.ndarray, int]:
    candidates, scores = graph_extraction.get_points_and_scores_from_mask(keypoint_u8, config.ITSC_THRESHOLD * 255)
    selected = np.array([
        point for point in graph_points_xy
        if keypoint_u8[int(point[1]), int(point[0])] > config.ITSC_THRESHOLD * 255
    ], dtype=graph_points_xy.dtype).reshape(-1, 2)
    return selected, int(len(candidates))


def infer_detailed(net: Any, img: np.ndarray, config: Any, device: torch.device, inferencer: Any, graph_extraction: Any) -> dict[str, Any]:
    if img.shape[:2] != (config.PATCH_SIZE, config.PATCH_SIZE):
        raise ValueError(f"This controlled regression expects native {config.PATCH_SIZE} square images, got {img.shape}")
    all_patch_info = inferencer.get_patch_info_rectangular(
        0, img.shape[0], img.shape[1], config.SAMPLE_MARGIN,
        config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE,
    )
    batch_size = config.INFER_BATCH_SIZE
    batches = [all_patch_info[i:i + batch_size] for i in range(0, len(all_patch_info), batch_size)]
    fused_keypoint = torch.zeros(img.shape[:2], dtype=torch.float32, device=device)
    fused_road = torch.zeros(img.shape[:2], dtype=torch.float32, device=device)
    counter = torch.zeros(img.shape[:2], dtype=torch.float32, device=device)
    image_features: list[torch.Tensor] = []
    mask_logits: list[torch.Tensor] = []
    mask_logits_capture: list[np.ndarray] = []
    image_feature_norms: list[float] = []

    with torch.inference_mode():
        for batch_patch_info in batches:
            patches = inferencer.get_batch_img_patches(img, batch_patch_info).to(device)
            features, logits, scores = net.infer_masks_and_img_features(patches)
            image_feature_norms.extend(features.float().flatten(1).norm(dim=1).cpu().tolist())
            image_features.append(features.cpu())
            mask_logits.append(logits.cpu())
            mask_logits_capture.append(logits.float().cpu().numpy())
            for patch_index, (_, (x0, y0), (x1, y1)) in enumerate(batch_patch_info):
                keypoint_patch = scores[patch_index, :, :, 0]
                road_patch = scores[patch_index, :, :, 1]
                fused_keypoint[y0:y1, x0:x1] += keypoint_patch
                fused_road[y0:y1, x0:x1] += road_patch
                counter[y0:y1, x0:x1] += 1
    fused_keypoint /= counter
    fused_road /= counter
    keypoint_probability = fused_keypoint.cpu().numpy().astype(np.float32)
    road_probability = fused_road.cpu().numpy().astype(np.float32)
    keypoint_u8 = (fused_keypoint * 255).to(torch.uint8).cpu().numpy()
    road_u8 = (fused_road * 255).to(torch.uint8).cpu().numpy()
    del fused_keypoint, fused_road, counter

    graph_points_xy = graph_extraction.extract_graph_points(
        keypoint_u8, road_u8, config, use_fast_nms=config.USE_FAST_NMS,
    )
    selected_keypoints_xy, keypoint_candidate_count = selected_keypoints_from_graph(keypoint_u8, graph_points_xy, config, graph_extraction)
    graph_index = rtree_index.Index()
    for point_index, (x, y) in enumerate(graph_points_xy):
        graph_index.insert(point_index, (x, y, x, y))

    edge_score_sums: dict[tuple[int, int], float] = defaultdict(float)
    edge_counts: dict[tuple[int, int], float] = defaultdict(float)
    topo_score_capture: list[np.ndarray] = []
    point_feature_norms: list[float] = []
    for batch_index, batch_patch_info in enumerate(batches):
        topo_data: dict[str, list[np.ndarray]] = {"points": [], "pairs": [], "valid": []}
        index_maps: list[dict[int, int]] = []
        for _, (x0, y0), (x1, y1) in batch_patch_info:
            point_indices = list(graph_index.intersection((x0, y0, x1, y1)))
            index_maps.append({patch_index: all_index for patch_index, all_index in enumerate(point_indices)})
            patch_points = graph_points_xy[point_indices] - np.array([[x0, y0]], dtype=graph_points_xy.dtype)
            point_count = len(point_indices)
            if point_count:
                tree = scipy.spatial.KDTree(patch_points)
                _, neighbor_index = tree.query(
                    patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1,
                    distance_upper_bound=config.NEIGHBOR_RADIUS,
                )
                neighbor_index = neighbor_index[:, 1:]
                source_index = np.tile(np.arange(point_count)[:, None], (1, config.MAX_NEIGHBOR_QUERIES))
                valid = neighbor_index < point_count
                target_index = np.where(valid, neighbor_index, source_index)
                pairs = np.stack([source_index, target_index], axis=-1)
            else:
                pairs = np.zeros((0, config.MAX_NEIGHBOR_QUERIES, 2), dtype=np.int64)
                valid = np.zeros((0, config.MAX_NEIGHBOR_QUERIES), dtype=bool)
            topo_data["points"].append(patch_points)
            topo_data["pairs"].append(pairs)
            topo_data["valid"].append(valid)
        max_length = max((array.shape[0] for array in topo_data["points"]), default=0)
        if max_length == 0:
            continue
        collated = {}
        for key, arrays in topo_data.items():
            collated[key] = np.stack([
                np.pad(array, [(0, max_length - array.shape[0])] + [(0, 0)] * (array.ndim - 1))
                for array in arrays
            ])
        features = image_features[batch_index].to(device)
        logits = mask_logits[batch_index].to(device)
        points_tensor = torch.tensor(collated["points"], device=device)
        pairs_tensor = torch.tensor(collated["pairs"], device=device)
        valid_tensor = torch.tensor(collated["valid"], device=device)
        with torch.inference_mode():
            sampled_features = net.bilinear_sampler(features, points_tensor)
            valid_points = np.array([len(points) for points in topo_data["points"]])
            for bi, count in enumerate(valid_points):
                if count:
                    point_feature_norms.extend(sampled_features[bi, :count].float().norm(dim=-1).cpu().tolist())
            scores = net.infer_toponet(features, points_tensor, pairs_tensor, valid_tensor, logits)
        scores = torch.where(torch.isnan(scores), -100.0, scores).squeeze(-1).float().cpu().numpy()
        topo_score_capture.append(scores.copy())
        batch_count, sample_count, pair_count = scores.shape
        for bi in range(batch_count):
            for sample_index in range(sample_count):
                for pair_index in range(pair_count):
                    if not collated["valid"][bi, sample_index, pair_index]:
                        continue
                    source_patch, target_patch = collated["pairs"][bi, sample_index, pair_index]
                    source = index_maps[bi][int(source_patch)]
                    target = index_maps[bi][int(target_patch)]
                    edge_score_sums[(source, target)] += float(scores[bi, sample_index, pair_index])
                    edge_counts[(source, target)] += 1.0
        del features, logits, points_tensor, pairs_tensor, valid_tensor, sampled_features

    candidate_edges = []
    accepted_directed = []
    for (source, target), score_sum in edge_score_sums.items():
        score = score_sum / edge_counts[(source, target)]
        row = {"source": int(source), "target": int(target), "score": float(score), "observations": int(edge_counts[(source, target)])}
        candidate_edges.append(row)
        if score > config.TOPO_THRESHOLD:
            accepted_directed.append(row)
    accepted_undirected_map: dict[tuple[int, int], list[float]] = defaultdict(list)
    for edge in accepted_directed:
        source, target = sorted((edge["source"], edge["target"]))
        if source != target:
            accepted_undirected_map[(source, target)].append(edge["score"])
    accepted_undirected = [
        {"source": source, "target": target, "score": float(statistics.mean(scores))}
        for (source, target), scores in sorted(accepted_undirected_map.items())
    ]
    nodes_rc = graph_points_xy[:, ::-1]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes_rc)))
    graph.add_edges_from((edge["source"], edge["target"]) for edge in accepted_undirected)
    components = list(nx.connected_components(graph)) if len(nodes_rc) else []
    largest = max(components, key=len) if components else set()
    largest_subgraph = graph.subgraph(largest)
    degrees = [degree for _, degree in graph.degree()]
    accepted_scores = [edge["score"] for edge in accepted_directed]
    road_binary = road_u8 > config.ROAD_THRESHOLD * 255
    keypoint_binary = keypoint_u8 > config.ITSC_THRESHOLD * 255
    diagnostics = {
        "road_positive_pixels": int(road_binary.sum()),
        "road_positive_ratio": float(road_binary.mean()),
        "keypoint_candidate_count": keypoint_candidate_count,
        "keypoint_selected_count": int(len(selected_keypoints_xy)),
        "graph_point_count": int(len(nodes_rc)),
        "candidate_edge_count": int(len(candidate_edges)),
        "accepted_directed_edge_count": int(len(accepted_directed)),
        "graph_node_count": int(len(nodes_rc)),
        "graph_edge_count": int(len(accepted_undirected)),
        "connected_component_count": int(len(components)),
        "largest_component_nodes": int(len(largest)),
        "largest_component_edge_count": int(largest_subgraph.number_of_edges()),
        "largest_component_ratio": float(len(largest) / len(nodes_rc)) if len(nodes_rc) else 0.0,
        "isolated_node_count": int(nx.number_of_isolates(graph)) if len(nodes_rc) else 0,
        "mean_accepted_topo_score": float(np.mean(accepted_scores)) if accepted_scores else 0.0,
        "median_accepted_topo_score": float(np.median(accepted_scores)) if accepted_scores else 0.0,
        "min_accepted_topo_score": float(np.min(accepted_scores)) if accepted_scores else 0.0,
        "max_accepted_topo_score": float(np.max(accepted_scores)) if accepted_scores else 0.0,
        "mean_node_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "endpoint_count": int(sum(degree == 1 for degree in degrees)),
        "junction_count": int(sum(degree >= 3 for degree in degrees)),
        "keypoint_distribution": percentile_stats(keypoint_probability, config.ITSC_THRESHOLD),
        "image_feature_l2_mean": float(np.mean(image_feature_norms)) if image_feature_norms else 0.0,
        "image_feature_l2_std": float(np.std(image_feature_norms)) if image_feature_norms else 0.0,
        "point_feature_l2_mean": float(np.mean(point_feature_norms)) if point_feature_norms else 0.0,
        "point_feature_l2_std": float(np.std(point_feature_norms)) if point_feature_norms else 0.0,
    }
    return {
        "road_probability": road_probability,
        "keypoint_probability": keypoint_probability,
        "road_u8": road_u8,
        "keypoint_u8": keypoint_u8,
        "road_binary": road_binary,
        "keypoint_binary": keypoint_binary,
        "selected_keypoints_xy": selected_keypoints_xy,
        "nodes_rc": nodes_rc,
        "candidate_edges": candidate_edges,
        "accepted_directed": accepted_directed,
        "accepted_undirected": accepted_undirected,
        "diagnostics": diagnostics,
        "mask_logits_capture": mask_logits_capture,
        "topo_score_capture": topo_score_capture,
    }


def to_heatmap(probability: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(probability * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def overlay_graph(rgb: np.ndarray, nodes_rc: np.ndarray, edges: list[dict[str, Any]]) -> np.ndarray:
    image = rgb.copy()
    for edge in edges:
        a, b = nodes_rc[edge["source"]], nodes_rc[edge["target"]]
        cv2.line(image, (int(round(a[1])), int(round(a[0]))), (int(round(b[1])), int(round(b[0]))), (255, 150, 0), 3, cv2.LINE_AA)
    connected = sorted({edge[key] for edge in edges for key in ("source", "target")})
    for node_index in connected:
        row, column = nodes_rc[node_index]
        cv2.circle(image, (int(round(column)), int(round(row))), 3, (255, 255, 0), -1, cv2.LINE_AA)
    return image


def overlay_keypoints(rgb: np.ndarray, keypoints_xy: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    for x, y in keypoints_xy:
        cv2.circle(image, (int(x), int(y)), 5, (255, 30, 30), 2, cv2.LINE_AA)
    return image


def save_result(directory: Path, rgb: np.ndarray, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(directory / "rgb.png")
    np.save(directory / "road_prob.npy", result["road_probability"].astype(np.float32))
    np.save(directory / "keypoint_prob.npy", result["keypoint_probability"].astype(np.float32))
    np.save(directory / "edge_scores.npy", np.array([
        [edge["source"], edge["target"], edge["score"], edge["observations"]]
        for edge in result["candidate_edges"]
    ], dtype=np.float32).reshape(-1, 4))
    Image.fromarray(np.rint(result["road_probability"] * 65535).astype(np.uint16)).save(directory / "road_prob.png")
    Image.fromarray(to_heatmap(result["road_probability"])).save(directory / "road_heatmap.png")
    Image.fromarray((result["road_binary"].astype(np.uint8) * 255)).save(directory / "road_binary.png")
    Image.fromarray(np.rint(result["keypoint_probability"] * 65535).astype(np.uint16)).save(directory / "keypoint_prob.png")
    Image.fromarray((result["keypoint_binary"].astype(np.uint8) * 255)).save(directory / "keypoint_binary.png")
    Image.fromarray(overlay_keypoints(rgb, result["selected_keypoints_xy"])).save(directory / "keypoints.png")
    graph_overlay = overlay_graph(rgb, result["nodes_rc"], result["accepted_undirected"])
    Image.fromarray(graph_overlay).save(directory / "final_graph_overlay.png")
    nodes = []
    for node_id, (row, column) in enumerate(result["nodes_rc"]):
        score = max(float(result["road_probability"][int(round(row)), int(round(column))]), float(result["keypoint_probability"][int(round(row)), int(round(column))]))
        nodes.append({"id": node_id, "x": float(column), "y": float(row), "score": score})
    dump_json(directory / "candidate_graph.json", {
        "coordinate_order": "x,y", "nodes": nodes, "candidate_edges": result["candidate_edges"],
        "accepted_directed_edges": result["accepted_directed"], "metadata": metadata,
    })
    dump_json(directory / "final_graph.json", {
        "coordinate_order": "x,y", "nodes": nodes, "edges": result["accepted_undirected"], "metadata": metadata,
    })
    dump_json(directory / "diagnostics.json", {**metadata, **result["diagnostics"]})


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def crop_official(spec: dict[str, Any], official_input_dir: Path) -> tuple[Path, np.ndarray]:
    official_input_dir.mkdir(parents=True, exist_ok=True)
    source = Path(spec["source"])
    x, y = spec["x"], spec["y"]
    image = Image.open(source).convert("RGB")
    if x + 1024 > image.width or y + 1024 > image.height:
        raise ValueError(f"invalid official crop: {spec}")
    crop = image.crop((x, y, x + 1024, y + 1024))
    path = official_input_dir / f"{spec['name']}.png"
    crop.save(path)
    return path, np.array(crop)


def max_abs_between_lists(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    if len(left) != len(right) or any(a.shape != b.shape for a, b in zip(left, right)):
        return math.inf
    return max((float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) for a, b in zip(left, right)), default=0.0)


def baseline_equivalence(net: Any, config: Any, image_path: Path, output_root: Path, device: torch.device, inferencer: Any, graph_extraction: Any) -> dict[str, Any]:
    image = load_rgb(image_path)
    inferencer.args.device = device
    original_mask = net.infer_masks_and_img_features
    original_topo = net.infer_toponet
    official_mask_logits: list[np.ndarray] = []
    official_topo_scores: list[np.ndarray] = []

    def capture_mask(rgb: torch.Tensor):
        features, logits, scores = original_mask(rgb)
        official_mask_logits.append(logits.float().cpu().numpy())
        return features, logits, scores

    def capture_topo(*args: Any, **kwargs: Any):
        scores = original_topo(*args, **kwargs)
        official_topo_scores.append(torch.where(torch.isnan(scores), -100.0, scores).squeeze(-1).float().cpu().numpy())
        return scores

    net.infer_masks_and_img_features = capture_mask
    net.infer_toponet = capture_topo
    with torch.inference_mode():
        official_nodes, official_edges, official_keypoint, official_road = inferencer.infer_one_img(net, image, config)
    net.infer_masks_and_img_features = original_mask
    net.infer_toponet = original_topo
    detailed = infer_detailed(net, image, config, device, inferencer, graph_extraction)
    detailed_edges = np.array([[edge["source"], edge["target"]] for edge in detailed["accepted_directed"]], dtype=np.int64).reshape(-1, 2)
    checks = {
        "image": str(image_path),
        "road_uint8_exact": bool(np.array_equal(official_road, detailed["road_u8"])),
        "keypoint_uint8_exact": bool(np.array_equal(official_keypoint, detailed["keypoint_u8"])),
        "nodes_exact": bool(np.array_equal(official_nodes, detailed["nodes_rc"])),
        "directed_edges_exact": bool(np.array_equal(official_edges, detailed_edges)),
        "mask_logits_max_abs_difference": max_abs_between_lists(official_mask_logits, detailed["mask_logits_capture"]),
        "topology_scores_max_abs_difference": max_abs_between_lists(official_topo_scores, detailed["topo_score_capture"]),
    }
    checks["passed"] = bool(
        checks["road_uint8_exact"] and checks["keypoint_uint8_exact"] and checks["nodes_exact"]
        and checks["directed_edges_exact"] and checks["mask_logits_max_abs_difference"] <= 1e-6
        and checks["topology_scores_max_abs_difference"] <= 1e-6
    )
    dump_json(output_root / "baseline_pipeline_equivalence.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"baseline pipeline equivalence failed: {checks}")
    return checks


def make_labeled_cell(image: np.ndarray, label: str, size: tuple[int, int] = (360, 360)) -> Image.Image:
    rendered = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (size[0], size[1] + 28), "black")
    cell.paste(rendered, (0, 28))
    draw = ImageDraw.Draw(cell)
    draw.text((8, 7), label, fill="white", font=ImageFont.load_default())
    return cell


def hstack(cells: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (sum(cell.width for cell in cells), max(cell.height for cell in cells)), "black")
    x = 0
    for cell in cells:
        canvas.paste(cell, (x, 0)); x += cell.width
    return canvas


def vstack(rows: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y)); y += row.height
    return canvas


def create_panels(output_root: Path, experiment: str, image_name: str) -> None:
    base_dir = output_root / experiment
    loaded = {}
    for model_name in MODEL_ORDER:
        directory = base_dir / model_name / image_name
        loaded[model_name] = {
            "rgb": np.array(Image.open(directory / "rgb.png").convert("RGB")),
            "road": np.array(Image.open(directory / "road_heatmap.png").convert("RGB")),
            "binary": np.array(Image.open(directory / "road_binary.png").convert("RGB")),
            "keypoints": np.array(Image.open(directory / "keypoints.png").convert("RGB")),
            "graph": np.array(Image.open(directory / "final_graph_overlay.png").convert("RGB")),
        }
    labels = [MODELS[name]["label"] for name in MODEL_ORDER]
    rgb = loaded["baseline"]["rgb"]
    diagnostic = vstack([
        hstack([make_labeled_cell(rgb, "RGB", (1080, 360))]),
        hstack([make_labeled_cell(loaded[name]["road"], f"{label} | road probability") for name, label in zip(MODEL_ORDER, labels)]),
        hstack([make_labeled_cell(loaded[name]["keypoints"], f"{label} | selected keypoints") for name, label in zip(MODEL_ORDER, labels)]),
        hstack([make_labeled_cell(loaded[name]["graph"], f"{label} | final graph") for name, label in zip(MODEL_ORDER, labels)]),
    ])
    graph_only = hstack([make_labeled_cell(rgb, "RGB")] + [make_labeled_cell(loaded[name]["graph"], label) for name, label in zip(MODEL_ORDER, labels)])
    road_graph_rows = []
    for name, label in zip(MODEL_ORDER, labels):
        road_graph_rows.append(hstack([
            make_labeled_cell(loaded[name]["road"], f"{label} | road probability"),
            make_labeled_cell(loaded[name]["binary"], f"{label} | road binary"),
            make_labeled_cell(loaded[name]["graph"], f"{label} | final graph"),
        ]))
    comparison_dir = output_root / "comparison" / experiment
    comparison_dir.mkdir(parents=True, exist_ok=True)
    diagnostic.save(comparison_dir / f"{image_name}_diagnostic.jpg", quality=94)
    graph_only.save(comparison_dir / f"{image_name}_graph_only.jpg", quality=94)
    vstack(road_graph_rows).save(comparison_dir / f"{image_name}_road_vs_graph.jpg", quality=94)


CSV_COLUMNS = [
    "experiment", "model", "image_name", "road_threshold", "keypoint_threshold", "topo_threshold",
    "road_positive_pixels", "road_positive_ratio", "keypoint_candidate_count", "keypoint_selected_count",
    "candidate_edge_count", "accepted_directed_edge_count", "graph_node_count", "graph_edge_count",
    "connected_component_count", "largest_component_nodes", "largest_component_edge_count",
    "largest_component_ratio", "isolated_node_count", "mean_node_degree", "max_degree",
    "endpoint_count", "junction_count", "mean_accepted_topo_score", "median_accepted_topo_score",
    "min_accepted_topo_score", "max_accepted_topo_score", "inference_time_ms",
    "peak_allocated_vram_mb", "peak_reserved_vram_mb",
]


def read_all_diagnostics(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for experiment in (G1, G2, OFFICIAL):
        for model_name in MODEL_ORDER:
            model_dir = output_root / experiment / model_name
            if not model_dir.exists():
                continue
            for path in sorted(model_dir.glob("*/diagnostics.json")):
                rows.append(json.loads(path.read_text()))
    return rows


def write_csv(output_root: Path) -> list[dict[str, Any]]:
    rows = read_all_diagnostics(output_root)
    with (output_root / "graph_regression_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def aggregate_rows(rows: list[dict[str, Any]], experiment: str) -> dict[str, dict[str, float]]:
    metrics = [
        "road_positive_ratio", "keypoint_candidate_count", "keypoint_selected_count", "candidate_edge_count",
        "graph_node_count", "graph_edge_count", "connected_component_count", "largest_component_ratio",
        "isolated_node_count", "mean_node_degree", "endpoint_count", "junction_count",
        "mean_accepted_topo_score", "inference_time_ms", "peak_allocated_vram_mb", "peak_reserved_vram_mb",
    ]
    result = {}
    for model_name in MODEL_ORDER:
        selected = [row for row in rows if row["experiment"] == experiment and row["model"] == model_name]
        result[model_name] = {metric: float(np.mean([row[metric] for row in selected])) if selected else 0.0 for metric in metrics}
    return result


def detect_warnings(rows: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    warnings = []
    lookup = {(row["experiment"], row["image_name"], row["model"]): row for row in rows}
    for experiment in (G1, G2):
        names = sorted({row["image_name"] for row in rows if row["experiment"] == experiment})
        for image_name in names:
            baseline = lookup[(experiment, image_name, "baseline")]
            if baseline["graph_edge_count"] == 0:
                warnings.append({
                    "experiment": experiment, "image_name": image_name, "model": "baseline",
                    "type": "BASELINE EMPTY",
                    "details": "The unchanged official pipeline produced no final edge at the fixed experiment thresholds.",
                })
            for model_name in ("encoder_lora", "encoder_decoder"):
                adapted = lookup[(experiment, image_name, model_name)]
                def add(kind: str, details: str) -> None:
                    warnings.append({"experiment": experiment, "image_name": image_name, "model": model_name, "type": kind, "details": details})
                if baseline["keypoint_selected_count"] and adapted["keypoint_selected_count"] > 5 * baseline["keypoint_selected_count"]:
                    add("KEYPOINT EXPLOSION", f"selected keypoints {baseline['keypoint_selected_count']} -> {adapted['keypoint_selected_count']}")
                if baseline["keypoint_distribution"]["above_original_keypoint_threshold"] == 0 and adapted["keypoint_distribution"]["above_original_keypoint_threshold"] >= 100:
                    add("KEYPOINT DISTRIBUTION SHIFT", f"pixels above original threshold 0 -> {adapted['keypoint_distribution']['above_original_keypoint_threshold']}")
                if baseline["graph_edge_count"] and adapted["road_positive_ratio"] > baseline["road_positive_ratio"] and adapted["graph_edge_count"] < 0.2 * baseline["graph_edge_count"]:
                    add("GRAPH COLLAPSE", f"road ratio {baseline['road_positive_ratio']:.4f} -> {adapted['road_positive_ratio']:.4f}, edges {baseline['graph_edge_count']} -> {adapted['graph_edge_count']}")
                if adapted["connected_component_count"] > max(baseline["connected_component_count"] * 3, baseline["connected_component_count"] + 10):
                    add("FRAGMENTATION", f"components {baseline['connected_component_count']} -> {adapted['connected_component_count']}")
                if baseline["graph_edge_count"] and adapted["graph_edge_count"] > 5 * baseline["graph_edge_count"]:
                    add("FALSE DENSIFICATION", f"edges {baseline['graph_edge_count']} -> {adapted['graph_edge_count']}")
    dump_json(output_root / "warnings.json", warnings)
    return warnings


def markdown_table(aggregate: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Model | Road ratio | Keypoints | Nodes | Edges | Components | Largest component | Mean topo score | ms/image | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_ORDER:
        row = aggregate[model_name]
        lines.append(
            f"| {MODELS[model_name]['label']} | {row['road_positive_ratio']:.4f} | {row['keypoint_selected_count']:.1f} | "
            f"{row['graph_node_count']:.1f} | {row['graph_edge_count']:.1f} | {row['connected_component_count']:.1f} | "
            f"{row['largest_component_ratio']:.3f} | {row['mean_accepted_topo_score']:.3f} | "
            f"{row['inference_time_ms']:.1f} | {row['peak_allocated_vram_mb']:.1f} |"
        )
    return "\n".join(lines)


def keypoint_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Model | mean | std | p50 | p90 | p95 | p99 | above 0.133 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name in MODEL_ORDER:
        selected = [row["keypoint_distribution"] for row in rows if row["experiment"] == G1 and row["model"] == model_name]
        mean = lambda key: float(np.mean([row[key] for row in selected]))
        lines.append(f"| {MODELS[model_name]['label']} | {mean('mean'):.4f} | {mean('std'):.4f} | {mean('p50'):.4f} | {mean('p90'):.4f} | {mean('p95'):.4f} | {mean('p99'):.4f} | {mean('above_original_keypoint_threshold'):.1f} |")
    return "\n".join(lines)


def feature_drift_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Model | Image feature L2 | Point feature L2 |", "|---|---:|---:|"]
    for model_name in MODEL_ORDER:
        selected = [row for row in rows if row["experiment"] == G1 and row["model"] == model_name]
        image_norm = float(np.mean([row["image_feature_l2_mean"] for row in selected]))
        point_norm = float(np.mean([row["point_feature_l2_mean"] for row in selected]))
        lines.append(f"| {MODELS[model_name]['label']} | {image_norm:.4f} | {point_norm:.4f} |")
    return "\n".join(lines)


def per_image_table(rows: list[dict[str, Any]], experiment: str) -> str:
    lines = [
        "| Image | Model | Road ratio | Keypoints | Nodes | Edges | Components | Largest ratio | Mean topo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = sorted(
        (row for row in rows if row["experiment"] == experiment),
        key=lambda row: (row["image_name"], MODEL_ORDER.index(row["model"])),
    )
    for row in selected:
        lines.append(
            f"| {row['image_name']} | {MODELS[row['model']]['label']} | {row['road_positive_ratio']:.4f} | "
            f"{row['keypoint_selected_count']} | {row['graph_node_count']} | {row['graph_edge_count']} | "
            f"{row['connected_component_count']} | {row['largest_component_ratio']:.3f} | {row['mean_accepted_topo_score']:.3f} |"
        )
    return "\n".join(lines)


def create_report(output_root: Path, identities: dict[str, Any], equivalence: dict[str, Any], rows: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    g1 = aggregate_rows(rows, G1)
    g2 = aggregate_rows(rows, G2)
    official = aggregate_rows(rows, OFFICIAL)
    xjtlu_names = sorted({row["image_name"] for row in rows if row["experiment"] == G1})
    config = json.loads((output_root / "inference_config.json").read_text())
    b_road_gain = g1["encoder_decoder"]["road_positive_ratio"] - g1["baseline"]["road_positive_ratio"]
    b_component_change = g1["encoder_decoder"]["connected_component_count"] - g1["baseline"]["connected_component_count"]
    b_largest_change = g1["encoder_decoder"]["largest_component_ratio"] - g1["baseline"]["largest_component_ratio"]
    lines = [
        "# MaGRoad LoveDA LoRA Graph Regression",
        "",
        "## 1. Experiment Purpose",
        "",
        "LoveDA road-only adaptation substantially improved road segmentation. This inference-only experiment tests whether that change transfers to the unchanged WildRoad keypoint/TopoNet graph pipeline. No optimizer, backward, threshold tuning on XJTLU, graph optimization, skeletonization, or topology training was used.",
        "",
        "## 2. Model Checkpoints",
        "",
    ]
    for model_name in MODEL_ORDER:
        identity = identities[model_name]
        comparison = identity["comparison_to_base"]
        lora_note = f"; nonzero LoRA B={identity.get('nonzero_lora_b', 0)}" if model_name != "baseline" else ""
        lines.append(
            f"- **{MODELS[model_name]['label']}**: `{identity['path']}`; size={identity['size_bytes']} bytes; SHA256=`{identity['sha256']}`; "
            f"epoch={identity.get('epoch')}; mode={identity['train_mode']}; LoRA rank={identity['lora_rank']}{lora_note}; "
            f"base encoder changed={comparison['encoder_base']['changed']}, decoder changed={comparison['map_decoder']['changed']}, TopoNet changed={comparison['toponet']['changed']}."
        )
    lines += [
        "",
        "## 3. Inference Configuration",
        "",
        f"Baseline equivalence passed: `{equivalence}`.",
        "",
        f"- PATCH_SIZE={config['PATCH_SIZE']}; ITSC_THRESHOLD={config['ITSC_THRESHOLD']}; TOPO_THRESHOLD={config['TOPO_THRESHOLD']}; G1 ROAD_THRESHOLD={config['G1_ROAD_THRESHOLD']}.",
        f"- ITSC_NMS_RADIUS={config['ITSC_NMS_RADIUS']}; ROAD_NMS_RADIUS={config['ROAD_NMS_RADIUS']}; NEIGHBOR_RADIUS={config['NEIGHBOR_RADIUS']}; MAX_NEIGHBOR_QUERIES={config['MAX_NEIGHBOR_QUERIES']}; USE_FAST_NMS={config['USE_FAST_NMS']}.",
        "- G2 changes only road threshold: baseline=0.1, A=0.3, B=0.4, selected previously on LoveDA Val. Keypoint, topology, NMS, candidate generation, and pruning remain fixed.",
        "- Candidate edges are directed KNN/TopoNet queries. Final graph metrics use canonical undirected edges and include sampled isolated nodes.",
        "",
        "## 4. G1 Controlled Results",
        "",
        markdown_table(g1),
        "",
        per_image_table(rows, G1),
        "",
        "The same per-image rows are also stored in `graph_regression_metrics.csv`.",
        "",
        "## 5. G2 Road-Calibrated Results",
        "",
        markdown_table(g2),
        "",
        per_image_table(rows, G2),
        "",
        "## 6. Visual Comparison",
        "",
    ]
    for experiment in (G1, G2):
        lines += [f"### {experiment}", ""]
        for image_name in xjtlu_names:
            lines += [
                f"![{experiment} {image_name} diagnostic](comparison/{experiment}/{image_name}_diagnostic.jpg)", "",
                f"![{experiment} {image_name} graph only](comparison/{experiment}/{image_name}_graph_only.jpg)", "",
                f"![{experiment} {image_name} road vs graph](comparison/{experiment}/{image_name}_road_vs_graph.jpg)", "",
            ]
    lines += [
        "## 7. Keypoint Drift",
        "",
        keypoint_table(rows),
        "",
        "B updates the shared two-channel map decoder using road-only supervision, so its keypoint distribution is a diagnostic rather than a trained target.",
        "",
        "### TopoNet input feature drift",
        "",
        feature_drift_table(rows),
        "",
        "Image-feature norms are one value per 1024 patch; point-feature norms average sampled graph-point embeddings. They diagnose distribution shift but do not measure graph accuracy.",
        "",
        "## 8. Official WildRoad Regression",
        "",
        "Three deterministic native-resolution 1024×1024 crops were taken from official WildRoad test images. Crop coordinates and source paths are stored in `official_inputs/crops.json`.",
        "",
        markdown_table(official),
        "",
        "These samples are a catastrophic-forgetting smoke test, not an official benchmark.",
        "",
    ]
    for spec in OFFICIAL_CROPS:
        name = spec["name"]
        lines += [f"![Official regression {name}](comparison/{OFFICIAL}/{name}_graph_only.jpg)", ""]
    lines += [
        "## 9. Findings",
        "",
        f"**Observation:** Under G1, B road-positive ratio changes by {b_road_gain:+.4f}, connected components by {b_component_change:+.2f}, and largest-component ratio by {b_largest_change:+.3f} relative to baseline (8-image means).",
        "",
    ]
    if warnings:
        lines.append("Automated structural warnings:")
        for warning in warnings:
            lines.append(f"- **{warning['type']}** — {warning['experiment']} / {warning['image_name']} / {warning['model']}: {warning['details']}")
    else:
        lines.append("No configured keypoint-explosion, graph-collapse, fragmentation, or false-densification warning was triggered.")
    lines += [
        "",
        "**Visual review of all 16 graph-only panels:** Adapted road corridors do enter the candidate/final graph on several tiles, but wide road regions frequently become triangular meshes instead of a sparse centerline. c00 shows a B-only border graph without road support; c06 and c07 show strong over-connection for both adapted models. These are observations from fixed panels, not GT errors.",
        "",
        "## 10. Decision Gate",
        "",
        "- **Q1 — Did missed roads enter the graph?** Partly. Additional road corridors appear in A/B graphs, but many are represented as dense cross-road triangles or border connections, so the transfer is not a reliable clean-graph improvement.",
        "- **Q2 — Which adapted model is more stable?** A is more stable than B: under G1 it has fewer components (2.38 vs 7.13), a larger main-component ratio (0.657 vs 0.495), and less degradation on the three WildRoad crops. A still triggers keypoint expansion and severe densification on c07.",
        "- **Q3 — What is the current bottleneck?** The evidence places the bottleneck at keypoint/candidate generation and compatibility with the unchanged TopoNet. For B specifically, road-only decoder training also shifts the unsupervised keypoint channel. The overall result matches Case 2, with Case-3-style head/feature mismatch particularly evident for B.",
        "",
        "XJTLU has no graph GT. This report does not claim graph accuracy, APLS, route success, topology F1, or statistical significance. The workflow stops here and does not start topology adaptation or planning optimization.",
    ]
    (output_root / "GRAPH_REGRESSION_REPORT.md").write_text("\n".join(lines) + "\n")


def run_one(net: Any, config: Any, image_path: Path, experiment: str, model_name: str, road_threshold: float, output_root: Path, device: torch.device, inferencer: Any, graph_extraction: Any) -> dict[str, Any]:
    config.ROAD_THRESHOLD = float(road_threshold)
    image = load_rgb(image_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = infer_detailed(net, image, config, device, inferencer, graph_extraction)
    elapsed_ms = (time.perf_counter() - start) * 1000
    allocated = torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0
    reserved = torch.cuda.max_memory_reserved(device) / 2**20 if torch.cuda.is_available() else 0.0
    metadata = {
        "experiment": experiment, "model": model_name, "image_name": image_path.stem,
        "checkpoint": str(MODELS[model_name]["checkpoint"]),
        "road_threshold": float(config.ROAD_THRESHOLD), "keypoint_threshold": float(config.ITSC_THRESHOLD),
        "topo_threshold": float(config.TOPO_THRESHOLD), "inference_time_ms": elapsed_ms,
        "peak_allocated_vram_mb": allocated, "peak_reserved_vram_mb": reserved,
        "training": False, "autograd_enabled": False,
    }
    result["diagnostics"].update({"inference_time_ms": elapsed_ms, "peak_allocated_vram_mb": allocated, "peak_reserved_vram_mb": reserved})
    save_result(output_root / experiment / model_name / image_path.stem, image, result, metadata)
    return {**metadata, **result["diagnostics"]}


def run_smoke(output_root: Path, device: torch.device, inferencer: Any, graph_extraction: Any, MaGRoad: Any, load_config: Any) -> None:
    image_path = XJTLU_DIR / "xjtlu_1_r03_c00.png"
    smoke = {}
    for model_name in MODEL_ORDER:
        net, config = load_model(model_name, device, MaGRoad, load_config)
        row = run_one(net, config, image_path, "smoke", model_name, float(load_config(str(MODELS["baseline"]["config"])).ROAD_THRESHOLD), output_root, device, inferencer, graph_extraction)
        smoke[model_name] = row
        del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    dump_json(output_root / "smoke_test.json", smoke)


def run_all(output_root: Path, device: torch.device, inferencer: Any, graph_extraction: Any, MaGRoad: Any, load_config: Any) -> None:
    xjtlu_images = sorted(XJTLU_DIR.glob("xjtlu_1_r03_c*.png"))
    if len(xjtlu_images) != 8:
        raise RuntimeError(f"expected 8 fixed XJTLU images, found {len(xjtlu_images)}")
    official_input_dir = output_root / "official_inputs"
    official_images = []
    for spec in OFFICIAL_CROPS:
        path, _ = crop_official(spec, official_input_dir)
        official_images.append(path)
    dump_json(official_input_dir / "crops.json", OFFICIAL_CROPS)
    official_config = load_config(str(MODELS["baseline"]["config"]))
    all_rows = []
    for model_name in MODEL_ORDER:
        net, config = load_model(model_name, device, MaGRoad, load_config)
        for experiment, threshold in ((G1, float(official_config.ROAD_THRESHOLD)), (G2, float(MODELS[model_name]["road_threshold"]))):
            for image_path in xjtlu_images:
                print(f"RUN {experiment} {model_name} {image_path.name}", flush=True)
                all_rows.append(run_one(net, config, image_path, experiment, model_name, threshold, output_root, device, inferencer, graph_extraction))
        for image_path in official_images:
            print(f"RUN {OFFICIAL} {model_name} {image_path.name}", flush=True)
            all_rows.append(run_one(net, config, image_path, OFFICIAL, model_name, float(official_config.ROAD_THRESHOLD), output_root, device, inferencer, graph_extraction))
        del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    for experiment in (G1, G2):
        for image_path in xjtlu_images:
            create_panels(output_root, experiment, image_path.stem)
    for image_path in official_images:
        create_panels(output_root, OFFICIAL, image_path.stem)
    dump_json(output_root / "all_metrics.json", all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("verify", "smoke", "all", "report"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    args.output_root.mkdir(parents=True, exist_ok=True)
    identities = verify_checkpoints(args.output_root)
    if args.phase == "verify":
        return
    inferencer, graph_extraction, MaGRoad, load_config = import_magroad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer.args.device = device
    torch.backends.cudnn.benchmark = True
    official_config = load_config(str(MODELS["baseline"]["config"]))
    dump_json(args.output_root / "inference_config.json", {
        **{key: official_config[key] for key in [
            "PATCH_SIZE", "ITSC_THRESHOLD", "TOPO_THRESHOLD", "ITSC_NMS_RADIUS", "ROAD_NMS_RADIUS",
            "NEIGHBOR_RADIUS", "MAX_NEIGHBOR_QUERIES", "USE_FAST_NMS", "INFER_PATCHES_PER_EDGE",
            "INFER_BATCH_SIZE", "SAMPLE_MARGIN", "TOPONET", "TOPONET_VERSION", "NUM_INTERPOLATIONS",
            "USE_POINT_FEATURES", "USE_EDGE_BIAS", "USE_GEOMETRIC_FEATURES", "USE_PATH_FEATURES",
        ]},
        "G1_ROAD_THRESHOLD": float(official_config.ROAD_THRESHOLD),
        "G2_ROAD_THRESHOLDS": {name: MODELS[name]["road_threshold"] for name in MODEL_ORDER},
    })
    baseline_net, baseline_config = load_model("baseline", device, MaGRoad, load_config)
    equivalence = baseline_equivalence(
        baseline_net, baseline_config, XJTLU_DIR / "xjtlu_1_r03_c00.png",
        args.output_root, device, inferencer, graph_extraction,
    )
    del baseline_net
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.phase == "smoke":
        run_smoke(args.output_root, device, inferencer, graph_extraction, MaGRoad, load_config)
        return
    if args.phase == "all":
        run_all(args.output_root, device, inferencer, graph_extraction, MaGRoad, load_config)
    rows = write_csv(args.output_root)
    warnings = detect_warnings(rows, args.output_root)
    create_report(args.output_root, identities, equivalence, rows, warnings)
    dump_json(args.output_root / "run_status.json", {"status": "complete", "finished_unix": time.time(), "phase": args.phase})


if __name__ == "__main__":
    main()
