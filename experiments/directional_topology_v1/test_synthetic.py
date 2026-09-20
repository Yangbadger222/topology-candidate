import unittest

import cv2
import numpy as np

from experiments.directional_topology_v1.core import DirectionalConfig, cleanup_candidates


def nodes(points):
    return [{"id": i, "x": float(x), "y": float(y), "score": 1.0} for i, (x, y) in enumerate(points)]


def complete_candidates(count):
    return [{"source": i, "target": j, "score": 0.9, "observations": 1}
            for i in range(count) for j in range(count) if i != j]


class DirectionalSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DirectionalConfig(anisotropy_threshold=0.5)

    def run_case(self, points, road, candidates=None):
        candidates = candidates or complete_candidates(len(points))
        return cleanup_candidates(nodes(points), candidates, road, self.cfg)

    def test_a_wide_straight_road_reduces_mesh(self):
        road = np.zeros((160, 240), np.float32); road[55:105, :] = 0.9
        pts = [(30, 65), (30, 90), (90, 65), (90, 90), (150, 65), (150, 90), (210, 65), (210, 90)]
        out = self.run_case(pts, road)
        self.assertLess(out["masks"]["C2"].sum(), out["masks"]["C0"].sum() * 0.6)

    def test_b_narrow_weak_road_is_kept(self):
        road = np.zeros((120, 220), np.float32); road[58:63, :] = 0.3
        pts = [(20, 60), (100, 60), (200, 60)]
        out = self.run_case(pts, road)
        self.assertTrue(out["masks"]["C3"][0])

    def test_c_short_tree_occlusion_is_kept(self):
        road = np.zeros((120, 220), np.float32); road[55:66, :] = 0.7; road[55:66, 94:114] = 0.01
        out = self.run_case([(20, 60), (200, 60)], road)
        self.assertTrue(out["masks"]["C3"].all())

    def test_d_building_shortcut_is_rejected(self):
        road = np.zeros((120, 220), np.float32); road[55:66, :30] = .8; road[55:66, 190:] = .8
        out = self.run_case([(20, 60), (200, 60)], road)
        self.assertFalse(out["masks"]["C3"].any())

    def test_e_straight_chain_suppresses_chords(self):
        road = np.zeros((100, 240), np.float32); road[45:56, :] = .8
        out = self.run_case([(20, 50), (80, 50), (140, 50), (200, 50)], road)
        kept = {(r["source"], r["target"]) for r, k in zip(out["decisions"], out["masks"]["C2"]) if k}
        self.assertFalse((0, 2) in kept or (0, 3) in kept)

    def test_f_y_junction_keeps_branches(self):
        road = np.zeros((220, 220), np.float32)
        for p in [(20, 110), (110, 110), (190, 40), (190, 180)]: cv2.line(road, (110, 110), p, .8, 11)
        pts = [(110, 110), (20, 110), (190, 40), (190, 180)]
        c = [{"source": 0, "target": j, "score": .9, "observations": 1} for j in range(1, 4)]
        out = self.run_case(pts, road, c)
        self.assertEqual(3, int(out["masks"]["C1"].sum()))

    def test_g_crossroad_keeps_four_branches(self):
        road = np.zeros((220, 220), np.float32); road[104:117, :] = .8; road[:, 104:117] = .8
        pts = [(110, 110), (20, 110), (200, 110), (110, 20), (110, 200)]
        c = [{"source": 0, "target": j, "score": .9, "observations": 1} for j in range(1, 5)]
        out = self.run_case(pts, road, c)
        self.assertEqual(4, int(out["masks"]["C1"].sum()))

    def test_h_curved_road_preserves_local_chain(self):
        road = np.zeros((240, 240), np.float32)
        pts = [(20, 40), (70, 50), (115, 80), (150, 125), (175, 185)]
        cv2.polylines(road, [np.array(pts, np.int32)], False, .7, 13)
        c = []
        for i in range(len(pts) - 1):
            c.extend([{"source": i, "target": i + 1, "score": .9, "observations": 1},
                      {"source": i + 1, "target": i, "score": .9, "observations": 1}])
        out = self.run_case(pts, road, c)
        self.assertTrue(out["masks"]["C3"].all())


if __name__ == "__main__":
    unittest.main()
