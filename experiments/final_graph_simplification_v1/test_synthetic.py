from __future__ import annotations

import unittest

import networkx as nx

from experiments.final_graph_simplification_v1.core import SimplificationConfig, metrics, simplify


CFG = SimplificationConfig(direction_group_angle_deg=20, opposite_axis_tolerance_deg=20,
                           chain_ratio=1.20, intermediate_line_distance_px=30,
                           alternative_hops=3, max_turn_deg=60)


def nodes(points):
    return [{"id": i, "x": float(x), "y": float(y), "score": 1.0} for i, (x, y) in enumerate(points)]


def edges(pairs, score=.9):
    return [{"source": a, "target": b, "score": score} for a, b in pairs]


def graph_from_navigation(nav):
    graph = nx.Graph(); graph.add_nodes_from(n["id"] for n in nav["nodes"])
    graph.add_edges_from((e["source"], e["target"]) for e in nav["edges"])
    return graph


class FinalGraphSyntheticTests(unittest.TestCase):
    def run_case(self, points, pairs):
        return simplify(nodes(points), edges(pairs), CFG)

    def test_01_straight_dense_corridor(self):
        out = self.run_case([(0, 0), (40, 0), (80, 0), (120, 0)],
                            [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)])
        self.assertEqual(3, out["S1"].number_of_edges())
        self.assertEqual(1, nx.number_connected_components(out["S2"]))

    def test_02_wide_road_mesh_reduces_chords(self):
        points = [(0, 0), (40, 0), (80, 0), (20, 8), (60, 8), (100, 8)]
        pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 2), (3, 5),
                 (0, 3), (1, 3), (1, 4), (2, 4), (2, 5)]
        out = self.run_case(points, pairs)
        self.assertLess(out["S2"].number_of_edges(), len(pairs))
        self.assertTrue(nx.has_path(out["S2"], 0, 5))

    def test_03_y_junction_preserves_three_branches(self):
        out = self.run_case([(50, 50), (0, 50), (100, 10), (100, 90), (25, 50)],
                            [(0, 1), (0, 2), (0, 3), (1, 4), (4, 0), (1, 0)])
        self.assertTrue(all(nx.has_path(out["S2"], 0, n) for n in (1, 2, 3)))
        self.assertGreaterEqual(out["roles"]["S2"][0]["branch_count"], 3)

    def test_04_crossroad_preserves_four_branches(self):
        out = self.run_case([(50, 50), (0, 50), (100, 50), (50, 0), (50, 100)],
                            [(0, 1), (0, 2), (0, 3), (0, 4)])
        self.assertEqual(4, out["S2"].degree(0))

    def test_05_curved_road_polyline_is_preserved(self):
        pts = [(0, 0), (30, 5), (55, 20), (75, 45)]
        out = self.run_case(pts, [(0, 1), (1, 2), (2, 3)])
        edge = out["S3"]["edges"][0]
        self.assertEqual(pts, [tuple(p) for p in edge["polyline"]])

    def test_06_real_loop_is_preserved(self):
        out = self.run_case([(0, 0), (50, 0), (50, 50), (0, 50)], [(0, 1), (1, 2), (2, 3), (3, 0)])
        self.assertEqual(1, metrics(out["S2"], out["roles"]["S2"])["cycle_rank"])

    def test_07_roundabout_and_branches_are_preserved(self):
        pts = [(0, 0), (40, 0), (40, 40), (0, 40), (-30, 0), (70, 0), (40, 70)]
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6)]
        out = self.run_case(pts, pairs)
        self.assertEqual(1, metrics(out["S2"], out["roles"]["S2"])["cycle_rank"])
        self.assertTrue(all(nx.has_path(out["S2"], 4, n) for n in (5, 6)))

    def test_08_parallel_paths_do_not_merge(self):
        out = self.run_case([(0, 0), (100, 0), (0, 15), (100, 15)], [(0, 1), (2, 3)])
        self.assertEqual(2, nx.number_connected_components(out["S2"]))

    def test_09_weak_path_branch_is_ignored_by_design(self):
        out = self.run_case([(0, 0), (50, 0), (100, 0), (50, 50)], [(0, 1), (1, 2), (1, 3)])
        self.assertTrue(out["S2"].has_edge(1, 3))
        self.assertTrue(all("probability" not in key.lower() and "confidence" not in key.lower()
                            for row in out["decisions"] for key in row))

    def test_10_endpoints_are_preserved(self):
        out = self.run_case([(0, 0), (30, 0), (60, 0), (90, 0)], [(0, 1), (1, 2), (2, 3), (0, 2)])
        self.assertTrue({0, 3}.issubset(out["S2"].nodes))
        self.assertTrue(nx.has_path(out["S2"], 0, 3))

    def test_11_component_invariant(self):
        out = self.run_case([(0, 0), (30, 0), (60, 0), (0, 50), (30, 50)],
                            [(0, 1), (1, 2), (0, 2), (3, 4)])
        self.assertEqual(nx.number_connected_components(out["S0"]), nx.number_connected_components(out["S1"]))
        self.assertEqual(nx.number_connected_components(out["S0"]), nx.number_connected_components(out["S2"]))

    def test_12_contraction_preserves_geometry_and_components(self):
        pts = [(0, 0), (25, 2), (50, 8), (75, 20), (100, 35)]
        out = self.run_case(pts, [(0, 1), (1, 2), (2, 3), (3, 4)])
        nav = graph_from_navigation(out["S3"])
        self.assertLess(nav.number_of_nodes(), out["S2"].number_of_nodes())
        self.assertEqual(nx.number_connected_components(nav), nx.number_connected_components(out["S2"]))
        self.assertEqual(pts, [tuple(p) for p in out["S3"]["edges"][0]["polyline"]])


if __name__ == "__main__":
    unittest.main()
