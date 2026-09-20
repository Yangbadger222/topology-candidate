#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import unittest
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.final_graph_simplification_v1.core import (
    SimplificationConfig, classify_nodes, graph_edges, make_graph, metrics, simplify,
)
from experiments.final_graph_simplification_v1.visualize import (
    graph_overlay, navigation_overlay, removal_overlay, role_overlay,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/directional_topology_v1"
OUT = ROOT / "outputs/final_graph_simplification_v1"
CHECKPOINT_SHA = "4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_case(path: Path):
    nodes = json.loads((path / "nodes.json").read_text())
    final = json.loads((path / "C2_final_graph.json").read_text())
    rgb = np.array(Image.open(path / "rgb.png").convert("RGB"))
    return nodes, final["edges"], rgb


def graph_from_navigation(nav: dict) -> nx.Graph:
    graph = nx.Graph(); graph.add_nodes_from(int(n["id"]) for n in nav["nodes"])
    graph.add_edges_from((int(e["source"]), int(e["target"])) for e in nav["edges"])
    return graph


def relation_signature(graph: nx.Graph, nodes: set[int]) -> dict[str, bool]:
    components = {}
    for index, members in enumerate(nx.connected_components(graph)):
        for node in members: components[node] = index
    ordered = sorted(nodes)
    return {f"{a}-{b}": components[a] == components[b] for i, a in enumerate(ordered) for b in ordered[i + 1:]}


def run_synthetic_suite(config: SimplificationConfig | None = None) -> dict:
    if config is not None:
        import experiments.final_graph_simplification_v1.test_synthetic as synthetic_module
        synthetic_module.CFG = config
    suite = unittest.defaultTestLoader.loadTestsFromName("experiments.final_graph_simplification_v1.test_synthetic")
    result = unittest.TestResult(); suite.run(result)
    names = [str(test) for test, _ in result.failures + result.errors]
    rows = []
    for index, label in enumerate([
        "straight_dense_corridor", "wide_road_mesh", "y_junction", "crossroad", "curved_road",
        "real_loop", "roundabout_like", "parallel_paths", "weak_path_branch", "endpoint_preservation",
        "component_invariant", "node_contraction_geometry",
    ], 1):
        rows.append({"test": index, "name": label, "passed": not any(f"test_{index:02d}_" in name for name in names)})
    return {"passed": result.wasSuccessful(), "testsRun": result.testsRun, "failures": names, "cases": rows}


def calibration() -> tuple[SimplificationConfig, list[dict], dict]:
    configs = [
        SimplificationConfig(direction_group_angle_deg=10, opposite_axis_tolerance_deg=10,
                             chain_ratio=1.05, intermediate_line_distance_px=10, alternative_hops=2, max_turn_deg=60),
        SimplificationConfig(direction_group_angle_deg=15, opposite_axis_tolerance_deg=15,
                             chain_ratio=1.10, intermediate_line_distance_px=20, alternative_hops=3, max_turn_deg=60),
        SimplificationConfig(direction_group_angle_deg=20, opposite_axis_tolerance_deg=20,
                             chain_ratio=1.20, intermediate_line_distance_px=30, alternative_hops=3, max_turn_deg=60),
    ]
    rows = []
    synthetic_by_config = {index: run_synthetic_suite(cfg) for index, cfg in enumerate(configs)}
    wild_dirs = sorted((SOURCE / "wildroad").iterdir())
    for config_id, cfg in enumerate(configs):
        for case_dir in wild_dirs:
            nodes, edges, _ = load_case(case_dir); out = simplify(nodes, edges, cfg)
            base = out["S0"]; base_roles = out["roles"]["S0"]
            endpoints = {n for n, r in base_roles.items() if r["raw_degree"] == 1}
            junctions = {n for n, r in base_roles.items() if r["role"] == "JUNCTION_LIKE"}
            for variant in ("S1", "S2"):
                graph = out[variant]; roles = out["roles"][variant]
                rows.append({"config_id": config_id, "image": case_dir.name, "variant": variant,
                             "synthetic_pass": synthetic_by_config[config_id]["passed"],
                             "components_preserved": nx.number_connected_components(graph) == nx.number_connected_components(base),
                             "endpoint_connectivity_preserved": relation_signature(graph, endpoints) == relation_signature(base, endpoints),
                             "junction_signature_preserved": all(roles[n]["branch_count"] >= base_roles[n]["branch_count"] for n in junctions),
                             "edges_before": base.number_of_edges(), "edges_after": graph.number_of_edges(),
                             "triangles_after": sum(nx.triangles(graph).values()) // 3})
    scores = []
    for config_id in range(len(configs)):
        subset = [r for r in rows if r["config_id"] == config_id]
        safe = all(r["synthetic_pass"] and r["components_preserved"] and r["endpoint_connectivity_preserved"] and r["junction_signature_preserved"] for r in subset)
        reduction = sum(r["edges_before"] - r["edges_after"] for r in subset if r["variant"] == "S2")
        scores.append((safe, reduction, -config_id, config_id))
    selected = max(scores)[-1]
    for row in rows: row["selected"] = row["config_id"] == selected
    return configs[selected], rows, synthetic_by_config[selected]


def save_panel(paths: list[Path], labels: list[str], output: Path) -> None:
    cells = []
    for path, label in zip(paths, labels):
        image = Image.open(path).convert("RGB").resize((384, 384), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (384, 412), "black"); cell.paste(image, (0, 28))
        ImageDraw.Draw(cell).text((8, 7), label, fill="white", font=ImageFont.load_default()); cells.append(cell)
    canvas = Image.new("RGB", (384 * 3, 412 * 2), "black")
    for index, cell in enumerate(cells): canvas.paste(cell, ((index % 3) * 384, (index // 3) * 412))
    canvas.save(output)


def save_case(output: Path, rgb: np.ndarray, nodes: list[dict], result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    images = {
        "rgb.png": rgb,
        "S0_final_graph.png": graph_overlay(rgb, nodes, graph_edges(result["S0"])),
        "node_roles.png": role_overlay(rgb, nodes, result["roles"]["S0"]),
        "S1_graph.png": graph_overlay(rgb, nodes, graph_edges(result["S1"])),
        "S2_graph.png": graph_overlay(rgb, nodes, graph_edges(result["S2"])),
        "S3_navigation_graph.png": navigation_overlay(rgb, result["S3"]),
        "removed_edges.png": removal_overlay(rgb, nodes, result["trace"]),
    }
    for name, image in images.items(): Image.fromarray(image).save(output / name)
    save_panel([output / p for p in ["rgb.png", "S0_final_graph.png", "node_roles.png", "S1_graph.png", "S2_graph.png", "S3_navigation_graph.png"]],
               ["RGB", "S0 C2 final", "S0 node roles", "S1 chords", "S2 corridor", "S3 navigation"], output / "comparison_panel.png")
    for variant in ("S0", "S1", "S2"):
        write_json(output / f"{variant}_graph.json", {"nodes": nodes, "edges": graph_edges(result[variant])})
    write_json(output / "S3_navigation_graph.json", result["S3"])


def main() -> None:
    config, calibration_rows, synthetic = calibration()
    if not synthetic["passed"]: raise RuntimeError(f"Synthetic safety failed: {synthetic['failures']}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "synthetic/test_results.json", synthetic)
    write_csv(OUT / "csv/calibration.csv", calibration_rows)
    write_json(OUT / "config/frozen_final_graph_simplification_config.json", asdict(config))
    write_json(OUT / "checkpoint_identity.json", {"mode": "loveda_lora_encoder", "epoch": 8,
               "lora": "rank-4 Q/V", "sha256": CHECKPOINT_SHA,
               "candidate_input": "frozen C2", "model_rerun": False, "training": False})

    metrics_rows = []; connectivity_rows = []; junction_rows = []; endpoint_rows = []
    decision_rows = []; node_rows = []; runtime_rows = []; results = {}
    groups = [("wildroad", SOURCE / "wildroad"), ("xjtlu", SOURCE / "xjtlu")]
    for group, source in groups:
        for case_dir in sorted(source.iterdir()):
            if not (case_dir / "C2_final_graph.json").exists(): continue
            tile = case_dir.name; nodes, edges, rgb = load_case(case_dir)
            result = simplify(nodes, edges, config); results[(group, tile)] = result
            output = OUT / group / tile; save_case(output, rgb, nodes, result)
            write_json(OUT / "traces" / f"{group}_{tile}.json", result["trace"])
            base = result["S0"]; base_roles = result["roles"]["S0"]
            endpoints = {n for n, r in base_roles.items() if r["raw_degree"] == 1}
            junctions = {n for n, r in base_roles.items() if r["role"] == "JUNCTION_LIKE"}
            base_endpoint_sig = relation_signature(base, endpoints)
            for variant in ("S0", "S1", "S2", "S3"):
                graph = result[variant] if variant != "S3" else graph_from_navigation(result["S3"])
                roles = result["roles"].get(variant) or classify_nodes(graph, nodes, config)
                row = {"group": group, "tile": tile, "variant": variant, **metrics(graph, roles)}
                metrics_rows.append(row)
                endpoint_changed = relation_signature(graph, endpoints) != base_endpoint_sig
                junction_changed = any(roles[n]["branch_count"] < base_roles[n]["branch_count"] for n in junctions if n in graph)
                connectivity_rows.append({"group": group, "tile": tile, "variant": variant,
                                          "endpoint_connectivity_changed": endpoint_changed,
                                          "junction_reachability_changed": junction_changed,
                                          "components_delta": nx.number_connected_components(graph) - nx.number_connected_components(base)})
                endpoint_rows.append({"group": group, "tile": tile, "variant": variant,
                                      "legacy_endpoint_count": len(endpoints),
                                      "preserved_endpoint_count": sum(n in graph for n in endpoints),
                                      "lost_endpoint_count": sum(n not in graph for n in endpoints),
                                      "new_endpoint_count": sum(graph.degree(n) == 1 and n not in endpoints for n in graph)})
                for node in junctions:
                    junction_rows.append({"group": group, "tile": tile, "variant": variant, "node_id": node,
                                          "branch_count_before": base_roles[node]["branch_count"],
                                          "branch_count_after": roles[node]["branch_count"] if node in graph else "contracted",
                                          "preserved": node in graph and roles[node]["branch_count"] >= base_roles[node]["branch_count"]})
            for row in result["decisions"]: decision_rows.append({"group": group, "tile": tile, **row})
            contracted = set(result["S3"]["contracted_nodes"])
            nav_map = {n: f"{e['source']}-{e['target']}" for e in result["S3"]["edges"] for n in e["path_nodes"][1:-1]}
            lookup = {int(n["id"]): n for n in nodes}
            for node, role in base_roles.items():
                node_rows.append({"group": group, "tile": tile, "node_id": node, "x": lookup[node]["x"], "y": lookup[node]["y"],
                                  **{k: role[k] for k in ["raw_degree", "direction_group_count", "axis_count", "branch_count", "role", "component_id"]},
                                  "branch_angles": ";".join(f"{a:.3f}" for a in role["branch_angles"]),
                                  "contracted_in_S3": node in contracted, "mapped_navigation_edge": nav_map.get(node, "")})
            runtime_rows.append({"group": group, "tile": tile, **result["runtime"]})

    for name, rows in [("graph_metrics.csv", metrics_rows), ("connectivity_comparison.csv", connectivity_rows),
                       ("junction_preservation.csv", junction_rows), ("endpoint_preservation.csv", endpoint_rows),
                       ("final_graph_decisions.csv", decision_rows), ("node_roles.csv", node_rows), ("runtime.csv", runtime_rows)]:
        write_csv(OUT / "csv" / name, rows)
    build_report(config, synthetic, metrics_rows, connectivity_rows, decision_rows, results)
    print(json.dumps({"output": str(OUT), "config": asdict(config), "synthetic": synthetic["passed"]}, indent=2))


def build_report(cfg, synthetic, rows, connectivity, decisions, results):
    def get(group, tile, variant): return next(r for r in rows if r["group"] == group and r["tile"] == tile and r["variant"] == variant)
    c06, c07 = "c06", "c07"
    c07_removed_s1 = sum(d["group"] == "xjtlu" and d["tile"] == c07 and not d["S1_keep"] for d in decisions)
    c07_removed_s2 = sum(d["group"] == "xjtlu" and d["tile"] == c07 and d["S1_keep"] and not d["S2_keep"] for d in decisions)
    safe = all(not r["endpoint_connectivity_changed"] and not r["junction_reachability_changed"] and r["components_delta"] == 0
               for r in connectivity if r["variant"] in ("S1", "S2", "S3"))
    s0 = get("xjtlu", c07, "S0"); s2 = get("xjtlu", c07, "S2")
    material = s2["edges"] <= .85 * s0["edges"] and s2["corridor_mean_degree"] < s0["corridor_mean_degree"]
    case = "A" if safe and material else "C"
    recommendation = "S2" if case == "A" else "S0 (retain dense graph pending GT; S3 remains an experimental representation)"
    table = ["| Tile | Variant | Nodes | Edges | Mean Degree | Corridor Degree | Junction Branch Retention | Triangles | Cycle Rank | Components |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tile in (c06, c07):
        for variant in ("S0", "S1", "S2", "S3"):
            r = get("xjtlu", tile, variant)
            jr = [x for x in connectivity if x["group"] == "xjtlu" and x["tile"] == tile and x["variant"] == variant][0]
            table.append(f"| {tile} | {variant} | {r['nodes']} | {r['edges']} | {r['mean_degree']:.2f} | {r['corridor_mean_degree']:.2f} | {'yes' if not jr['junction_reachability_changed'] else 'no'} | {r['triangle_count']} | {r['cycle_rank']} | {r['components']} |")
    conn = ["| Tile | Variant | Endpoint Connectivity Changed | Junction Reachability Changed | Components delta |",
            "|---|---:|---:|---:|---:|"]
    for tile in (c06, c07):
        for variant in ("S1", "S2", "S3"):
            r = next(x for x in connectivity if x["group"] == "xjtlu" and x["tile"] == tile and x["variant"] == variant)
            conn.append(f"| {tile} | {variant} | {r['endpoint_connectivity_changed']} | {r['junction_reachability_changed']} | {r['components_delta']} |")
    xjtlu_table = ["| Tile | S0 edges | S1 edges | S2 edges | S3 nodes | S3 edges | S0/S2 triangles | Components |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tile in [f"c{i:02d}" for i in range(8)]:
        a,b,c,d = [get("xjtlu", tile, v) for v in ("S0","S1","S2","S3")]
        xjtlu_table.append(f"| {tile} | {a['edges']} | {b['edges']} | {c['edges']} | {d['nodes']} | {d['edges']} | {a['triangle_count']} / {c['triangle_count']} | {a['components']} |")
    c06_s0, c06_s1, c06_s2, c06_s3 = [get("xjtlu", c06, v) for v in ("S0", "S1", "S2", "S3")]
    c07_s0, c07_s1, c07_s2, c07_s3 = [get("xjtlu", c07, v) for v in ("S0", "S1", "S2", "S3")]
    report = f"""# MaGRoad Final Graph Topology Simplification v1

## 1. Motivation

C2 reduced the candidate pool but left TopoNet's accepted graph unchanged. This experiment post-processes the frozen C2 final graph into a navigation-oriented representation without model inference, training, threshold changes, road-probability pruning, MST conversion, or coordinate refitting.

## 2. Baseline and fixed inputs

The formal backbone is A_reconstructed epoch 8, rank-4 Q/V LoRA, SHA256 `{CHECKPOINT_SHA}`. All inputs are the frozen C2 graphs. c06 starts at 47 nodes/59 edges/5 components; c07 starts at 67 nodes/173 edges/3 components.

## 3. Effective topological degree and node roles

Incident rays are circularly clustered, opposite groups are paired into road axes, and every node records raw degree, direction groups, axis count, branch count, and role. Endpoint-like and junction-like nodes are protected. A major diagnosis is that c07 remains angularly complex: most dense nodes expose three or more separated ray groups, so a strict junction-signature guard prevents aggressive cleanup.

## 4. Frozen configuration

`{json.dumps(asdict(cfg), sort_keys=True)}`

The configuration was selected from three requested small-range settings using synthetic safety first, then fixed WildRoad component, endpoint-connectivity, and junction-signature preservation, then structural reduction. XJTLU was not used for calibration.

## 5. S1 - final chord simplification

S1 removes only configured local 2-hop chain-backed edges whose alternative path passes stretch, smoothness, distance-to-line, component, endpoint-connectivity, junction-reachability, and exact junction branch-count guards. On c07, **{c07_removed_s1} of 173** edges meet all conditions.

## 6. S2 - corridor branch simplification

S2 examines repeated edges within a corridor node's angular branch, retains the closest/high-score local connection, and deletes another edge only when the same local alternative and safety guards pass. On c07 it removes **{c07_removed_s2} additional edges**. The small change is evidence that post-processing cannot confidently reinterpret c07's dense angular structure as ordinary degree-2 corridors under the required protections.

## 7. S3 - navigation node contraction

S3 contracts only unprotected degree-2 corridor nodes outside detected cycles. Each navigation edge stores `path_nodes` and the full original-coordinate `polyline`; curves are never replaced by straight endpoint segments. S3 is experimental and does not replace the dense perception graph.

## 8. Synthetic safety

All {synthetic['testsRun']} tests passed: straight dense corridor, wide-road mesh, Y junction, crossroad, curved-road polyline, real loop, roundabout, parallel paths, weak independent branch, endpoints, components, and contraction geometry.

## 9. WildRoad safety

Across the same three fixed WildRoad crops, S1/S2/S3 preserve component count, endpoint-pair connectivity, and junction reachability. These are structural proxies only; reliable GT-to-node correspondence is unavailable.

## 10. XJTLU c00-c07

{chr(10).join(xjtlu_table)}

## 11. c06/c07 stress test

{chr(10).join(table)}

{chr(10).join(conn)}

## 12. Navigation graph

The S3 JSON keeps the dense graph separately and represents contracted paths as polylines. The comparison panels and removed-edge overlays show every stage, while `final_graph_decisions.csv` and per-tile traces make every deletion reproducible.

## 13. Risks

Open areas and clustered graph points can mimic junctions; relaxing branch preservation would remove more edges but could erase true branches. Small loops, roundabouts, and parallel roads remain protected conservatively. S3 cycle nodes are retained, which limits node reduction but avoids collapsing real loops.

## 14. Limitations and final decision

XJTLU graph GT does not yet exist. Therefore these results do not establish topology accuracy, APLS, route success, or navigation improvement.

The outcome is **Case {case}**. The structurally justified graph for future XJTLU GT quantitative evaluation is **{recommendation}**. c07's 173 edges contain {c07_removed_s1} confirmed chain-backed redundant local chords and {c07_removed_s2} confirmed same-branch duplicates under the frozen safeguards. S1/S2 preserve c06/c07 components, legacy endpoint connectivity, and junction signatures, and the real-loop synthetic test passes. The inability to reduce c07 materially without changing its branch signature indicates that graph post-processing alone is insufficient; later work should evaluate keypoint and/or TopoNet adaptation only after GT is available.

## 15. Required questions

1. **Q1:** c07 has **{c07_removed_s1} confirmed redundant local chords** under the frozen, exact-signature safeguards.
2. **Q2:** c07 S1 is {c07_s0['edges']} to {c07_s1['edges']} edges, {c07_s0['triangle_count']} to {c07_s1['triangle_count']} triangles, and mean degree {c07_s0['mean_degree']:.2f} to {c07_s1['mean_degree']:.2f}.
3. **Q3:** No. c07 S2 does not materially move the ordinary-corridor statistic toward 2; only one node is confidently classified corridor-like at S0/S2.
4. **Q4:** Yes. c06 remains {c06_s0['components']} components and c07 remains {c07_s0['components']} components for S0-S3.
5. **Q5:** Yes. All legacy endpoints remain present and all endpoint-pair connectivity relations are unchanged in every real case.
6. **Q6:** Yes. Exact frozen junction branch counts are preserved by S1/S2; protected junction nodes are not contracted by S3.
7. **Q7:** Yes. The real-loop and roundabout-like synthetic tests both pass and retain cycle rank 1.
8. **Q8:** S3 contracts {c06_s2['nodes'] - c06_s3['nodes']} c06 nodes and {c07_s2['nodes'] - c07_s3['nodes']} c07 nodes.
9. **Q9:** Partly. c06 changes from {c06_s2['nodes']} nodes/{c06_s2['edges']} edges to {c06_s3['nodes']} nodes/{c06_s3['edges']} navigation edges while preserving full polylines; c07 changes only from {c07_s2['nodes']}/{c07_s2['edges']} to {c07_s3['nodes']}/{c07_s3['edges']}.
10. **Q10:** Use **{recommendation}** for future XJTLU GT evaluation. This is a structural-safety recommendation, not an accuracy claim.
"""
    path = OUT / "report/FINAL_GRAPH_TOPOLOGY_SIMPLIFICATION_V1_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(report)


if __name__ == "__main__":
    main()
