from __future__ import annotations

import cv2
import numpy as np


def heatmap(prob: np.ndarray) -> np.ndarray:
    image = np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(image, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def graph_overlay(rgb, nodes, edges, color=(255, 190, 0), width=2):
    image = rgb.copy()
    for edge in edges:
        a, b = nodes[int(edge["source"])], nodes[int(edge["target"])]
        cv2.line(image, (round(a["x"]), round(a["y"])), (round(b["x"]), round(b["y"])), color, width, cv2.LINE_AA)
    for node in nodes:
        cv2.circle(image, (round(node["x"]), round(node["y"])), 3, (255, 40, 40), -1, cv2.LINE_AA)
    return image


def keypoint_overlay(rgb, nodes):
    image = rgb.copy()
    for node in nodes:
        cv2.circle(image, (round(node["x"]), round(node["y"])), 5, (255, 40, 40), 2, cv2.LINE_AA)
    return image


def direction_overlay(rgb, nodes, directions, length=24):
    image = rgb.copy()
    for node, direction in zip(nodes, directions):
        x, y = float(node["x"]), float(node["y"])
        dx, dy = direction["direction_x"] * length, direction["direction_y"] * length
        color = (0, 255, 80) if direction["anisotropy"] >= 0.7 else (255, 190, 0)
        cv2.line(image, (round(x - dx), round(y - dy)), (round(x + dx), round(y + dy)), color, 2, cv2.LINE_AA)
        cv2.circle(image, (round(x), round(y)), 3, (255, 40, 40), -1, cv2.LINE_AA)
    return image


def rejection_overlay(rgb, nodes, decisions):
    image = rgb.copy()
    colors = {"off_axis_redundant": (255, 80, 40), "same_branch_farther_neighbor": (255, 170, 0),
              "redundant_chord": (255, 0, 255),
              "long_unsupported_shortcut": (80, 80, 255)}
    for row in decisions:
        if row["reject_reason"] == "passed": continue
        a, b = nodes[int(row["source"])], nodes[int(row["target"])]
        cv2.line(image, (round(a["x"]), round(a["y"])), (round(b["x"]), round(b["y"])), colors[row["reject_reason"]], 2, cv2.LINE_AA)
    return image
