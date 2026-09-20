from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROLE_COLORS = {"ENDPOINT_LIKE": (255, 70, 70), "CORRIDOR_LIKE": (50, 220, 80),
               "JUNCTION_LIKE": (255, 180, 0), "AMBIGUOUS": (170, 80, 255)}
REMOVAL_COLORS = {"redundant_chord": (255, 0, 255), "same_branch_duplicate": (0, 180, 255),
                  "local_cycle_redundancy": (255, 100, 0), "contracted_node": (80, 230, 230)}


def _canvas(rgb):
    image = Image.fromarray(rgb).convert("RGB")
    return image, ImageDraw.Draw(image)


def _point(node):
    return round(float(node["x"])), round(float(node["y"]))


def _circle(draw, point, radius, fill, outline=None):
    x, y = point
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=outline)


def graph_overlay(rgb, nodes, edges, color=(30, 235, 70)):
    image, draw = _canvas(rgb); lookup = {int(n["id"]): n for n in nodes}
    for edge in edges:
        draw.line((_point(lookup[int(edge["source"])]), _point(lookup[int(edge["target"])])), fill=color, width=2)
    for node in nodes: _circle(draw, _point(node), 3, (255, 40, 40))
    return np.asarray(image)


def role_overlay(rgb, nodes, roles):
    image, draw = _canvas(rgb)
    for node in nodes: _circle(draw, _point(node), 7, ROLE_COLORS[roles[int(node["id"])]["role"]], (20, 20, 20))
    _legend(draw, list(ROLE_COLORS.items()))
    return np.asarray(image)


def removal_overlay(rgb, nodes, traces):
    image, draw = _canvas(rgb); lookup = {int(n["id"]): n for n in nodes}
    for trace in traces:
        a, b = [lookup[int(x)] for x in trace["edge"]]
        draw.line((_point(a), _point(b)), fill=REMOVAL_COLORS[trace["reason"]], width=3)
    _legend(draw, list(REMOVAL_COLORS.items()))
    return np.asarray(image)


def navigation_overlay(rgb, navigation):
    image, draw = _canvas(rgb)
    for edge in navigation["edges"]:
        draw.line([tuple(map(round, point)) for point in edge["polyline"]], fill=(40, 230, 80), width=3, joint="curve")
    for node in navigation["nodes"]: _circle(draw, _point(node), 5, (255, 50, 50))
    return np.asarray(image)


def _legend(draw, entries):
    font = ImageFont.load_default(); width = max(220, max((len(label) for label, _ in entries), default=0) * 8 + 36)
    draw.rectangle((8, 8, 8 + width, 20 + 22 * len(entries)), fill=(20, 20, 20))
    for index, (label, color) in enumerate(entries):
        y = 25 + 22 * index; _circle(draw, (22, y - 4), 5, color); draw.text((34, y - 9), label, fill="white", font=font)
