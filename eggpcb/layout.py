"""Outline-dependent placement of constrained parts: corner mounting holes and edge connectors."""
from __future__ import annotations

import math

import numpy as np

from .geometry import k_for_mouth
from .kicad_io import Problem

EDGES = ("left", "right", "top", "bottom")


def corner_inset(ol) -> float:
    """box-corner inset from both edges so a box sitting in a rounded corner keeps `edge` clearance."""
    if ol.r <= ol.edge:
        return ol.edge
    return max(ol.edge, ol.r - (ol.r - ol.edge) / math.sqrt(2))


def corner_centers(pr: Problem, ol):
    """{part index: (cx, cy)} for corner-pinned holes, in TL, TR, BL, BR order of their refs."""
    idx = [i for i, p in enumerate(pr.parts) if p.corner_hole]
    t = corner_inset(ol)
    out = {}
    for n, i in enumerate(idx[:4]):
        p = pr.parts[i]
        x = t + p.w / 2 if n in (0, 2) else ol.W - t - p.w / 2
        y = t + p.h / 2 if n in (0, 1) else ol.H - t - p.h / 2
        out[i] = (x, y)
    return out


def edge_k_table(pr: Problem):
    """(N,4) rotation index for each edge part on each edge (-1 for non-edge parts)."""
    kt = -np.ones((len(pr.parts), 4), dtype=np.int64)
    for i, p in enumerate(pr.parts):
        if p.fixed == "edge":
            for e, name in enumerate(EDGES):
                kt[i, e] = k_for_mouth(p.edge_mouth, name)
    return kt


def edge_place(p, e: int, along: float, ol):
    """centre + k of edge part p on edge e with its along-edge centre coordinate `along`."""
    k = k_for_mouth(p.edge_mouth, EDGES[e])
    w, h = (p.h, p.w) if k % 2 else (p.w, p.h)
    oh = p.edge_overhang
    if e == 0:
        return (-oh + w / 2, along), k
    if e == 1:
        return (ol.W + oh - w / 2, along), k
    if e == 2:
        return (along, -oh + h / 2), k
    return (along, ol.H + oh - h / 2), k


def nearest_edge(c, ol) -> int:
    x, y = c
    d = [x, ol.W - x, y, ol.H - y]
    return int(np.argmin(d))


def apply_constraints(pr: Problem, c, k, s, ol, edge_assign=None):
    """Project a free-form state onto the hard constraints: corner holes at corners, edge parts on
    their (assigned or nearest) edge with the mouth facing out, fixed-rotation parts at rot 0 / top.
    Returns (c, k, s, edge_assign)."""
    c = np.array(c, dtype=float); k = np.array(k, dtype=np.int64); s = np.array(s, dtype=float)
    ea = {} if edge_assign is None else dict(edge_assign)
    for i, xy in corner_centers(pr, ol).items():
        c[i] = xy; k[i] = 0; s[i] = 0
    for i, p in enumerate(pr.parts):
        if p.fixed == "edge":
            e = ea.get(i, nearest_edge(c[i], ol))
            ea[i] = e
            along = c[i, 1] if e in (0, 1) else c[i, 0]
            (cx, cy), kk = edge_place(p, e, along, ol)
            c[i] = (cx, cy); k[i] = kk; s[i] = 0
        elif not p.allow_rotate:
            k[i] = int(round(p.rot0 / 90)) % 4
        if not p.allow_bottom:
            s[i] = 0
    return c, k, s, ea
