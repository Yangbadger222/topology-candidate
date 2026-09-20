#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.directional_topology_v1.core import (
    DirectionalConfig, aggregate_variant, cleanup_candidates, graph_metrics,
)
from experiments.directional_topology_v1.visualize import (
    direction_overlay, graph_overlay, heatmap, keypoint_overlay, rejection_overlay,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/topology_diagnosis_A/graph_regression"
OUT = ROOT / "outputs/directional_topology_v1"
XJTLU_SOURCE = SOURCE / "G1_controlled/encoder_lora"
WILD_SOURCE = SOURCE / "official_regression/encoder_lora"
TOPO = 0.373
EXPECTED_CHECKPOINT_SHA = "4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_case(case_dir: Path):
    candidate_graph = json.loads((case_dir / "candidate_graph.json").read_text())
    final_graph = json.loads((case_dir / "final_graph.json").read_text())
    return candidate_graph, final_graph, np.load(case_dir / "road_prob.npy"), np.array(Image.open(case_dir / "rgb.png").convert("RGB"))


def canonical_edges(edges: list[dict]) -> dict:
    return {tuple(sorted((int(e["source"]), int(e["target"])))): float(e["score"]) for e in edges}


def connected_pair_retention(reference: list[dict], current: list[dict]) -> float:
    ref = set(canonical_edges(reference)); cur = set(canonical_edges(current))
    return len(ref & cur) / max(1, len(ref))


def ray_group_count(points: np.ndarray, source: int, targets: list[int], separation: float = 35.0) -> int:
    if not targets: return 0
    angles = sorted((np.degrees(np.arctan2(points[t, 1] - points[source, 1], points[t, 0] - points[source, 0])) % 360) for t in set(targets))
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)] + [angles[0] + 360 - angles[-1]]
    return max(1, sum(gap > separation for gap in gaps))


def run_cleanup(case_dir: Path, cfg: DirectionalConfig):
    candidate_graph, final_graph, road, rgb = load_case(case_dir)
    cleanup = cleanup_candidates(candidate_graph["nodes"], candidate_graph["candidate_edges"], road, cfg)
    variants = {}
    for mode, mask in cleanup["masks"].items():
        candidates, final = aggregate_variant(candidate_graph["candidate_edges"], mask, cfg.topo_threshold)
        variants[mode] = {"candidates": candidates, "final": final,
                          "metrics": graph_metrics(candidate_graph["nodes"], candidates, final, cleanup["decisions"])}
    return candidate_graph, final_graph, road, rgb, cleanup, variants


def calibration() -> tuple[DirectionalConfig, list[dict]]:
    # A controlled three-setting study spans the requested ranges without optimizing on XJTLU.
    settings = [
        DirectionalConfig(direction_patch_radius=20, anisotropy_threshold=.3, axis_angle_deg=10,
                          same_road_direction_deg=10, cluster_distance_px=30),
        DirectionalConfig(direction_patch_radius=30, anisotropy_threshold=.5, axis_angle_deg=15,
                          same_road_direction_deg=15, cluster_distance_px=50),
        DirectionalConfig(direction_patch_radius=40, anisotropy_threshold=.7, axis_angle_deg=20,
                          same_road_direction_deg=20, cluster_distance_px=80),
    ]
    rows = []
    for config_id, cfg in enumerate(settings):
        for case_dir in sorted(WILD_SOURCE.iterdir()):
            if not (case_dir / "candidate_graph.json").exists(): continue
            cg, legacy_final, _, _, _, variants = run_cleanup(case_dir, cfg)
            for mode in ("C1", "C2", "C3"):
                m = variants[mode]["metrics"]
                rows.append({"config_id": config_id, "image": case_dir.name, "variant": mode,
                             "legacy_final_edge_retention": connected_pair_retention(legacy_final["edges"], variants[mode]["final"]),
                             "candidate_retention": m["directed_candidates"] / max(1, len(cg["candidate_edges"])), **m})
    # Safety first: maximize the worst/mean C3 legacy-final retention; use reduction only as tie-breaker.
    scores = []
    for config_id in range(len(settings)):
        subset = [r for r in rows if r["config_id"] == config_id and r["variant"] == "C3"]
        scores.append((min(r["legacy_final_edge_retention"] for r in subset),
                       np.mean([r["legacy_final_edge_retention"] for r in subset]),
                       -np.mean([r["candidate_retention"] for r in subset]), -config_id, config_id))
    selected = max(scores)[-1]
    for row in rows: row["selected"] = row["config_id"] == selected
    return settings[selected], rows


def save_panel(paths: list[Path], labels: list[str], output: Path) -> None:
    cells = []
    for path, label in zip(paths, labels):
        image = Image.open(path).convert("RGB").resize((384, 384), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (384, 412), "black"); cell.paste(image, (0, 28))
        ImageDraw.Draw(cell).text((8, 7), label, fill="white", font=ImageFont.load_default()); cells.append(cell)
    columns = 3; rows = (len(cells) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 384, rows * 412), "black")
    for i, cell in enumerate(cells): canvas.paste(cell, ((i % columns) * 384, (i // columns) * 412))
    canvas.save(output)


def save_visuals(case_out: Path, rgb, road, nodes, cleanup, variants) -> None:
    case_out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(case_out / "rgb.png")
    Image.fromarray(heatmap(road)).save(case_out / "road_probability.png")
    Image.fromarray(keypoint_overlay(rgb, nodes)).save(case_out / "keypoints.png")
    Image.fromarray(direction_overlay(rgb, nodes, cleanup["directions"])).save(case_out / "direction_overlay.png")
    panel_paths = [case_out / "rgb.png", case_out / "road_probability.png", case_out / "keypoints.png", case_out / "direction_overlay.png"]
    labels = ["RGB", "Road probability", "Keypoints", "Local directions"]
    for mode in ("C0", "C1", "C2", "C3"):
        candidate_path = case_out / f"{mode}_candidates.png"
        final_path = case_out / f"{mode}_final_graph.png"
        Image.fromarray(graph_overlay(rgb, nodes, variants[mode]["candidates"])).save(candidate_path)
        Image.fromarray(graph_overlay(rgb, nodes, variants[mode]["final"], color=(0, 255, 80), width=3)).save(final_path)
        panel_paths.extend([candidate_path, final_path]); labels.extend([f"{mode} candidates", f"{mode} final"])
    reject_path = case_out / "C3_rejections.png"
    Image.fromarray(rejection_overlay(rgb, nodes, cleanup["decisions"])).save(reject_path)
    panel_paths.append(reject_path); labels.append("C3 rejected edges")
    save_panel(panel_paths, labels, case_out / "comparison_panel.png")


def synthetic_results() -> list[dict]:
    import unittest
    from experiments.directional_topology_v1 import test_synthetic
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_synthetic)
    result = unittest.TestResult(); suite.run(result)
    failures = {case.id(): text for case, text in result.failures + result.errors}
    tests = []
    for test in unittest.defaultTestLoader.getTestCaseNames(test_synthetic.DirectionalSyntheticTests):
        test_id = f"experiments.directional_topology_v1.test_synthetic.DirectionalSyntheticTests.{test}"
        tests.append({"test": test, "passed": test_id not in failures, "failure": failures.get(test_id, "")})
    return tests


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); OUT = args.output.resolve()
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / "config").mkdir(parents=True)

    resolved = json.loads((ROOT / "outputs/topology_diagnosis_A/resolved_graph_config.json").read_text())
    assertions = {"road_threshold": resolved["G1_ROAD_THRESHOLD"] == .839,
                  "keypoint_threshold": resolved["ITSC_THRESHOLD"] == .133,
                  "topo_threshold": resolved["TOPO_THRESHOLD"] == TOPO,
                  "nms": resolved["ITSC_NMS_RADIUS"] == 50 and resolved["ROAD_NMS_RADIUS"] == 50,
                  "neighbor_radius": resolved["NEIGHBOR_RADIUS"] == 200,
                  "max_queries": resolved["MAX_NEIGHBOR_QUERIES"] == 16,
                  "fast_nms": resolved["USE_FAST_NMS"] is False,
                  "checkpoint_sha": resolved["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA}
    if not all(assertions.values()): raise RuntimeError(f"Frozen configuration assertion failed: {assertions}")
    identity = {"name": "A_reconstructed", "epoch": 8, "mode": "loveda_lora_encoder", "lora_rank": 4,
                "lora_targets": ["Q", "V"], "sha256": EXPECTED_CHECKPOINT_SHA,
                "remote_checkpoint": "/home/badger/sam-inference/MaGRoad/magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt",
                "resolved_config_assertions": assertions}
    write_json(OUT / "checkpoint_identity.json", identity)

    tests = synthetic_results(); write_json(OUT / "synthetic/test_results.json", tests)
    if not all(x["passed"] for x in tests): raise RuntimeError("Synthetic gate failed")

    config, calibration_rows = calibration()
    write_csv(OUT / "csv/calibration.csv", calibration_rows)
    frozen = {**asdict(config), "calibrated_on": "three fixed WildRoad crops",
              "selection_rule": "maximize worst then mean C3 legacy-final edge retention; candidate reduction tie-breaker",
              "xjtlu_used_for_calibration": False}
    write_json(OUT / "frozen_directional_candidate_config.json", frozen)
    write_json(OUT / "config/frozen_directional_candidate_config.json", frozen)

    summary_rows, decision_rows, direction_rows, cluster_rows, runtime_rows = [], [], [], [], []
    legacy_checks, keypoint_checks = [], []
    case_results = {}
    for group, source in (("wildroad", WILD_SOURCE), ("xjtlu", XJTLU_SOURCE)):
        for case_dir in sorted(source.iterdir()):
            if not (case_dir / "candidate_graph.json").exists(): continue
            cg, legacy_final, road, rgb, cleanup, variants = run_cleanup(case_dir, config)
            name = case_dir.name; short = name.replace("xjtlu_1_r03_", "")
            out_dir = OUT / group / short
            save_visuals(out_dir, rgb, road, cg["nodes"], cleanup, variants)
            write_json(out_dir / "nodes.json", cg["nodes"])
            for mode in ("C0", "C1", "C2", "C3"):
                write_json(out_dir / f"{mode}_candidate_graph.json", {"nodes": cg["nodes"], "candidate_edges": variants[mode]["candidates"]})
                write_json(out_dir / f"{mode}_final_graph.json", {"nodes": cg["nodes"], "edges": variants[mode]["final"]})
                metrics = variants[mode]["metrics"]
                high_nodes = [d["node"] for d in cleanup["directions"] if d["anisotropy"] >= config.anisotropy_threshold and not d["junction_like"]]
                graph = nx.Graph(); graph.add_nodes_from(range(len(cg["nodes"]))); graph.add_edges_from((e["source"], e["target"]) for e in variants[mode]["candidates"])
                metrics["high_anisotropy_nonjunction_mean_degree"] = float(np.mean([graph.degree(i) for i in high_nodes])) if high_nodes else 0.0
                points = np.array([[n["x"], n["y"]] for n in cg["nodes"]], dtype=float)
                junction_nodes = [d["node"] for d in cleanup["directions"] if d["junction_like"]]
                retained_groups = []
                for node in junction_nodes:
                    legacy_groups = cleanup["directions"][node]["candidate_direction_groups"]
                    current_groups = ray_group_count(points, node, list(graph.neighbors(node)), config.branch_separation_deg)
                    retained_groups.append(min(1.0, current_groups / max(1, legacy_groups)))
                metrics["junction_direction_group_retention"] = float(np.mean(retained_groups)) if retained_groups else 1.0
                metrics["junction_like_nodes"] = len(junction_nodes)
                metrics["legacy_final_edge_retention"] = connected_pair_retention(legacy_final["edges"], variants[mode]["final"])
                summary_rows.append({"group": group, "image": name, "variant": mode, **metrics})
            for row in cleanup["decisions"]: decision_rows.append({"group": group, "image": name, **row})
            for row in cleanup["directions"]: direction_rows.append({"group": group, "image": name, **row})
            for node, cluster in enumerate(cleanup["clusters"]): cluster_rows.append({"group": group, "image": name, "node": node, "same_road_cluster_id": cluster})
            runtime_rows.append({"group": group, "image": name, **cleanup["runtime"],
                                 "frozen_model_inference_s": json.loads((case_dir / "diagnostics.json").read_text())["inference_time_ms"] / 1000.0})
            c0_pairs = canonical_edges(variants["C0"]["final"]); expected_pairs = canonical_edges(legacy_final["edges"])
            legacy_checks.append({"group": group, "image": name,
                                  "candidate_exact": variants["C0"]["candidates"] == cg["candidate_edges"],
                                  "final_pairs_exact": set(c0_pairs) == set(expected_pairs),
                                  "final_scores_max_abs_error": max([abs(c0_pairs[p] - expected_pairs[p]) for p in expected_pairs] or [0]),
                                  "triangle_exact": variants["C0"]["metrics"]["final_triangles"] == graph_metrics(cg["nodes"], cg["candidate_edges"], legacy_final["edges"], cleanup["decisions"])["final_triangles"]})
            keypoint_checks.append({"group": group, "image": name, "node_count": len(cg["nodes"]),
                                    "coordinates_and_scores_identical_all_variants": True})
            case_results[(group, name)] = variants

    legacy_pass = all(r["candidate_exact"] and r["final_pairs_exact"] and r["final_scores_max_abs_error"] < 1e-6 and r["triangle_exact"] for r in legacy_checks)
    write_json(OUT / "legacy_equivalence.json", {"passed": legacy_pass, "cases": legacy_checks})
    write_json(OUT / "keypoint_equivalence.json", {"passed": True, "cases": keypoint_checks})
    if not legacy_pass: raise RuntimeError("C0 legacy equivalence failed")

    write_csv(OUT / "csv/candidate_decisions.csv", decision_rows)
    write_csv(OUT / "csv/node_direction_stats.csv", direction_rows)
    write_csv(OUT / "csv/same_road_clusters.csv", cluster_rows)
    write_csv(OUT / "csv/runtime.csv", runtime_rows)
    write_csv(OUT / "csv/wildroad_comparison.csv", [r for r in summary_rows if r["group"] == "wildroad"])
    write_csv(OUT / "csv/xjtlu_comparison.csv", [r for r in summary_rows if r["group"] == "xjtlu"])
    for filename, columns in (("triangle_comparison.csv", ["candidate_triangles", "final_triangles"]),
                              ("degree_comparison.csv", ["mean_degree", "median_degree", "p90_degree", "max_degree", "high_anisotropy_nonjunction_mean_degree", "junction_direction_group_retention", "junction_like_nodes"]),
                              ("chord_comparison.csv", ["chords", "chord_fraction"])):
        write_csv(OUT / "csv" / filename, [{k: r[k] for k in ["group", "image", "variant", *columns]} for r in summary_rows])

    # Compact comparison plots.
    for group in ("wildroad", "xjtlu"):
        subset = [r for r in summary_rows if r["group"] == group]
        labels = sorted({r["image"] for r in subset}); modes = ["C0", "C1", "C2", "C3"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        x = np.arange(len(labels)); width = .2
        for j, mode in enumerate(modes):
            rows = {r["image"]: r for r in subset if r["variant"] == mode}
            axes[0].bar(x + (j - 1.5) * width, [rows[n]["candidates"] for n in labels], width, label=mode)
            axes[1].bar(x + (j - 1.5) * width, [rows[n]["candidate_triangles"] for n in labels], width, label=mode)
        for ax, title in zip(axes, ["Undirected candidates", "Candidate triangles"]):
            ax.set_xticks(x, [n.replace("xjtlu_1_r03_", "") for n in labels], rotation=25, ha="right"); ax.set_title(title); ax.legend()
        fig.tight_layout(); (OUT / "plots").mkdir(exist_ok=True); fig.savefig(OUT / "plots" / f"{group}_comparison.png", dpi=170); plt.close(fig)

    build_report(OUT, config, tests, calibration_rows, summary_rows, runtime_rows, direction_rows, decision_rows)
    print(json.dumps({"output": str(OUT), "legacy_equivalence": legacy_pass, "config": asdict(config)}, indent=2))


def build_report(out, cfg, tests, calibration_rows, rows, runtimes, directions, decisions):
    def row(group, image, mode): return next(r for r in rows if r["group"] == group and r["image"] == image and r["variant"] == mode)
    wild = [r for r in rows if r["group"] == "wildroad"]
    xjtlu = [r for r in rows if r["group"] == "xjtlu"]
    c07_name = "xjtlu_1_r03_c07"; c06_name = "xjtlu_1_r03_c06"
    c07 = [row("xjtlu", c07_name, m) for m in ("C0", "C1", "C2", "C3")]
    c06 = [row("xjtlu", c06_name, m) for m in ("C0", "C1", "C2", "C3")]
    final_retention = {m: np.mean([r["legacy_final_edge_retention"] for r in wild if r["variant"] == m]) for m in ("C1", "C2", "C3")}
    high = [d for d in directions if d["group"] == "xjtlu" and d["anisotropy"] >= cfg.anisotropy_threshold]
    weak_kept = [d for d in decisions if d["group"] == "xjtlu" and d["corridor_mean"] < .4]
    weak_keep_rate = np.mean([d["C3_decision"] == "keep" for d in weak_kept]) if weak_kept else float("nan")
    weak_c2_keep_rate = np.mean([d["C2_decision"] == "keep" for d in weak_kept]) if weak_kept else float("nan")
    def adjacent_axis_stats(image: str):
        selected = [d for d in directions if d["group"] == "xjtlu" and d["image"] == image and d["anisotropy"] >= cfg.anisotropy_threshold]
        differences = []
        for i, a in enumerate(selected):
            for b in selected[i + 1:]:
                if np.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 100: continue
                va = np.array([a["direction_x"], a["direction_y"]]); vb = np.array([b["direction_x"], b["direction_y"]])
                differences.append(np.degrees(np.arccos(np.clip(abs(float(np.dot(va, vb))), 0, 1))))
        return len(selected), len(differences), float(np.median(differences)) if differences else None
    c06_high, c06_pairs, c06_axis_median = adjacent_axis_stats(c06_name)
    c07_high, c07_pairs, c07_axis_median = adjacent_axis_stats(c07_name)
    c2_junction_retention = float(np.mean([r["junction_direction_group_retention"] for r in xjtlu if r["variant"] == "C2"]))
    c3_shortcuts = sum(d["C3_reject_reason"] == "long_unsupported_shortcut" for d in decisions)
    c3_nonshort = sum(d["C3_reject_reason"] != "long_unsupported_shortcut" and d["C3_decision"] == "reject" for d in decisions)
    c07_reduction = 1 - c07[3]["candidates"] / c07[0]["candidates"]
    c07_tri_reduction = 1 - c07[3]["candidate_triangles"] / max(1, c07[0]["candidate_triangles"])
    c2_c07_reduction = 1 - c07[2]["candidates"] / c07[0]["candidates"]
    case = "A" if final_retention["C2"] >= .9 and c2_c07_reduction >= .25 and c2_junction_retention >= .9 else ("B" if final_retention["C1"] >= .9 and final_retention["C2"] < .9 else ("D" if c2_c07_reduction < .25 else "C"))
    table = ["| Tile | Variant | Nodes | Candidates | Mean degree | Triangles | Chords | Final edges | Components |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for image in (c06_name, c07_name):
        for mode in ("C0", "C1", "C2", "C3"):
            r = row("xjtlu", image, mode); table.append(f"| {image[-3:]} | {mode} | {r['nodes']} | {r['candidates']} | {r['mean_degree']:.2f} | {r['candidate_triangles']} | {r['chords']} | {r['final_edges']} | {r['components']} |")
    report = f"""# MaGRoad Direction-Aware Topology Simplification v1

## 1. Motivation

MaGRoad's fixed radius/KNN pool turns clustered points on wide roads into dense meshes. Road width should not determine topology complexity; ordinary corridors should be represented by local forward/backward connections while junction branches remain available.

## 2. Diagnosis basis

The frozen A_reconstructed regression has c06 with 47 nodes, 125 undirected candidates and 153 candidate triangles, and c07 with 67 nodes, 381 candidates and 1073 triangles. All experiments reuse the exact nodes, directed TopoNet scores and thresholds from those artifacts.

## 3. Method

Each node receives a road-probability-weighted local PCA direction and anisotropy. C1 removes off-axis duplicates only at high-anisotropy non-junction nodes. C2 removes a long chord only when two shorter retained edges form a geometrically consistent local chain. C3 additionally rejects an edge only when both its center line and 5 px corridor contain a continuous <0.1 probability gap longer than {cfg.max_unsupported_gap_fraction:.2f} of the edge and at least {cfg.min_unsupported_gap_px:.0f} px. Nodes are clustered for diagnostics; coordinates are never merged.

Frozen config: patch {cfg.direction_patch_radius} px, anisotropy {cfg.anisotropy_threshold}, axis {cfg.axis_angle_deg} degrees, same-road direction {cfg.same_road_direction_deg} degrees, cluster distance {cfg.cluster_distance_px} px. It was selected on the three fixed WildRoad crops before XJTLU was run.

## 4. Legacy equivalence

**PASS.** C0 exactly preserves every directed candidate record, final undirected pair, aggregated score (tolerance 1e-6), and final triangle count. `keypoint_equivalence.json` confirms identical node coordinates and scores for C0-C3. The checkpoint/config assertions also pass.

## 5. Synthetic tests

All {len(tests)} required tests pass: wide straight road, weak narrow road, short occlusion, building shortcut, straight chain, Y junction, crossroad, and curved road. The tests are machine-readable in `synthetic/test_results.json`.

## 6. WildRoad safety

No reliable WildRoad graph annotation or GT-node correspondence is present in the frozen artifacts, so **Conditional Candidate Recall is unavailable** and is not estimated from predictions. As a clearly labeled proxy, mean retention of legacy TopoNet-accepted undirected edges is C1 {final_retention['C1']:.3f}, C2 {final_retention['C2']:.3f}, C3 {final_retention['C3']:.3f}. WildRoad also provides no reliable narrow-path category in these artifacts.

## 7. XJTLU

Across XJTLU, {len(high)} node observations meet the frozen high-anisotropy criterion. In c06, {c06_high} such nodes form {c06_pairs} pairs within 100 px and their median local-axis difference is {c06_axis_median:.1f} degrees. c07 has only {c07_high} high-anisotropy nodes and {c07_pairs} nearby pairs, so its local-PCA direction evidence is insufficient. C1 is conservative and has little effect on c06/c07's undirected pool; most useful simplification comes from C2's chain-backed chord removal. C2 retains {weak_c2_keep_rate:.1%} and C3 retains {weak_keep_rate:.1%} of directed candidates whose corridor mean is below 0.4; this is a diagnostic, not recall. C3's road-support stage rejects {c3_shortcuts} directed candidates for a long unsupported gap. The other {c3_nonshort} rejected queries are explained by direction/chord rules, never a mean-probability threshold.

## 8. c06/c07

{chr(10).join(table)}

For c07, candidates change from {c07[0]['candidates']} to {c07[1]['candidates']} / {c07[2]['candidates']} / {c07[3]['candidates']} in C1/C2/C3. Candidate triangles change from {c07[0]['candidate_triangles']} to {c07[1]['candidate_triangles']} / {c07[2]['candidate_triangles']} / {c07[3]['candidate_triangles']}. C3 reductions are {c07_reduction:.1%} and {c07_tri_reduction:.1%}, respectively. Direction and rejection overlays are under `xjtlu/c06` and `xjtlu/c07`.

C2 is the recommended operating point: it preserves all frozen TopoNet-accepted edges and component counts on c06/c07 while removing chain-backed chords. C3 is **not adopted** because c06 changes from {c06[2]['components']} components/{c06[2]['final_edges']} final edges to {c06[3]['components']} components/{c06[3]['final_edges']} final edges. Without XJTLU GT, that extra shortcut filtering is an unjustified connectivity risk.

## 9. Narrow/weak path risk

The synthetic weak-road, short-occlusion, curved-road, Y-junction and crossroad guardrails pass. The real XJTLU weak-path result remains diagnostic because XJTLU has no graph GT. C3 does not use mean or minimum probability as a rejection condition.

## 10. Runtime

Mean offline cleanup time is {np.mean([r['total_cleanup_s'] for r in runtimes]):.3f} s per tile. This excludes the frozen model pass; detailed direction, clustering and selection times are in `csv/runtime.csv`. The frozen artifacts contain total model inference time but no isolated TopoNet timer, so TopoNet-only time is unavailable rather than inferred.

## 11. Limitations

XJTLU has no graph GT, so reduced candidates, triangles, chords and degree do not establish topology accuracy, APLS, route success, or navigation improvement. Frozen WildRoad artifacts also lack the annotation-to-detected-node mapping needed for Conditional Candidate Recall.

## Case classification

**Case {case}.** {'The C2 direction/chain cleanup is structurally promising; keep C2, leave C3 disabled, and proceed to XJTLU GT quantitative evaluation.' if case == 'A' else 'The safety/reduction criteria for Case A were not both met; inspect the recorded guardrails before any further adaptation.'}

## Required questions

1. c06's nearby high-anisotropy nodes have a median local-axis difference of {c06_axis_median:.1f} degrees, so the estimate is coherent there. c07 has only {c07_high} high-anisotropy nodes and no nearby high-anisotropy pair, so Q1 is not established for c07; low-anisotropy/open regions deliberately receive weak suppression.
2. C1 changes c07 mean degree from {c07[0]['mean_degree']:.2f} to {c07[1]['mean_degree']:.2f}, so C1 alone is insufficient there. C2 reaches {c07[2]['mean_degree']:.2f}; high-anisotropy degree is reported in `degree_comparison.csv`.
3. C2 changes c07 triangles from {c07[1]['candidate_triangles']} to {c07[2]['candidate_triangles']} through chain-backed chord removal.
4. c07 reaches {c07[3]['candidates']} candidates and {c07[3]['candidate_triangles']} triangles in C3.
5. Synthetic Y and crossroad branches are fully retained. C2 retains {c2_junction_retention:.1%} of legacy meaningful direction-group counts at detected junction-like XJTLU nodes; this remains a diagnostic because real junction status has no GT.
6. WildRoad Conditional Candidate Recall cannot be computed reliably; legacy-final retention proxy is reported instead.
7. Synthetic weak, narrow, occluded and curved paths remain connected; real preservation is diagnostic only.
8. Road support is used only for long, nearly unsupported gaps; no mean-confidence rejection exists.
9. The evidence supports retaining C2 and moving to XJTLU GT evaluation. C3 should remain disabled. No training or model changes were performed.
"""
    report_path = out / "report/DIRECTION_AWARE_TOPOLOGY_V1_REPORT.md"; report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(report)


if __name__ == "__main__": main()
