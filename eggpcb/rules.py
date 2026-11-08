"""Apply the user's placement rules (configs/*.json) to a parsed Problem."""
from __future__ import annotations

import json
import numpy as np

from .kicad_io import Problem

MOUTH_AXIS = {"+x": (0, +1), "-x": (0, -1), "+y": (1, +1), "-y": (1, -1)}


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def apply_rules(pr: Problem, cfg: dict) -> Problem:
    pr.layers = int(cfg["layers"])
    pr.rules = cfg
    allow_bottom = set(cfg["sides"]["allow_bottom_kinds"]) if pr.layers >= 2 else set()
    hb = cfg["clearance"].get("hole_both_sides", 0.25)
    for p in pr.parts:
        p.allow_bottom = p.kind in allow_bottom
        p.allow_rotate = True
        p.fixed = "none"
        # through holes block the opposite side as well
        p.bottom_hole_keepout = [[hx - r - hb, hy - r - hb, hx + r + hb, hy + r + hb] for hx, hy, r in p.holes]

    for ref in cfg.get("fixed_abs", []):
        p = pr.parts[pr.index(ref)]
        p.fixed = "abs"; p.allow_rotate = False; p.allow_bottom = False

    for ref, spec in cfg.get("edge_parts", {}).items():
        p = pr.parts[pr.index(ref)]
        p.fixed = "edge"; p.allow_bottom = False; p.allow_rotate = False
        p.edge_mouth = spec["mouth"]
        ax, sg = MOUTH_AXIS[spec["mouth"]]
        half = (p.w, p.h)[ax] / 2
        if spec["edge_line"] == "courtyard":
            assert p.cy_box is not None, ref
            line = p.cy_box[ax + 2] if sg > 0 else p.cy_box[ax]          # already rel. box centre
        else:
            line = float(spec["edge_line"]) - p.box_off[ax]               # fp-local -> rel. box centre
        p.edge_line = float(line)
        p.edge_overhang = float(max(0.0, sg * (sg * half - line)))       # box beyond the edge line

    mh = cfg.get("mounting_holes", {})
    for ref in mh.get("refs", []):
        p = pr.parts[pr.index(ref)]
        p.allow_bottom = False; p.allow_rotate = False
        if mh.get("mode") == "corners":
            p.corner_hole = True; p.fixed = "corner"

    # net weights
    nw = cfg.get("net_weights", {})
    gnd = set(cfg.get("decoupling", {}).get("gnd_nets", ["GND"]))
    pwr = set(nw.get("power_nets", []))
    cap = nw.get("fanout_cap", 12)
    for n in pr.nets:
        w = 1.0
        if n["name"] in gnd:
            w = nw.get("gnd_weight", 0.15)
        elif n["name"] in pwr:
            w = nw.get("power_weight", 0.5)
        k = len(n["pins"])
        if k > cap:
            w *= cap / k * 2  # damp very high fan-out nets (pins only need to reach a pour/trunk)
        n["weight"] = float(min(w, 1.0))
    return pr


def decap_pairs(pr: Problem, cfg: dict):
    """Find decoupling caps: a C with one pad on GND and one on a power net shared with an IC on the
    same sheet. Returns list of (cap_part, cap_pad, [(ic_part, ic_pad), ...])."""
    dc = cfg.get("decoupling", {})
    gnd = set(dc.get("gnd_nets", ["GND"]))
    ics = set(dc.get("ic_kinds", ["U"]))
    net_name = {i: n["name"] for i, n in enumerate(pr.nets)}
    pads_on_net = {}
    for i, n in enumerate(pr.nets):
        pads_on_net[i] = n["pins"]
    out = []
    for ci, c in enumerate(pr.parts):
        if c.kind != dc.get("cap_kind", "C") or len(c.pad_net) != 2:
            continue
        nets = c.pad_net
        if any(n < 0 for n in nets):
            continue
        names = [net_name[n] for n in nets]
        if sum(nm in gnd for nm in names) != 1:
            continue
        k = 0 if names[1] in gnd else 1
        pnet = nets[k]
        cands = [(pi, pk) for (pi, pk) in pads_on_net[pnet]
                 if pr.parts[pi].kind in ics and pr.parts[pi].sheet == c.sheet]
        if not cands:
            cands = [(pi, pk) for (pi, pk) in pads_on_net[pnet] if pr.parts[pi].kind in ics]
        if cands and len(cands) <= 40:
            out.append((ci, k, cands))
    return out


# ---------------------------------------------------------------- paths / routing settings
def gnd_net(pr: Problem) -> str:
    return pr.rules.get("decoupling", {}).get("gnd_nets", ["GND"])[0]


def board_paths(pr: Problem):
    """(source .kicad_pcb, DRU text or None, runs directory) recorded by scripts/board_stats.py."""
    from pathlib import Path

    def resolve(key_abs, key_rel):
        p = Path(pr.rules[key_abs]) if pr.rules.get(key_abs) else None
        if (p is None or not p.exists()) and pr.rules.get(key_rel) and getattr(pr, "source", None):
            p = (pr.source.parent / pr.rules[key_rel]).resolve()
        return p
    src = resolve("board_abs", "board_rel")
    dru_p = resolve("dru_abs", "dru_rel")
    dru = dru_p.read_text() if dru_p and dru_p.exists() else None
    return src, dru, src.parent / "runs"


def routing_cfg(pr: Problem) -> dict:
    r = dict(router="1.9", gnd="pour", timeout=2700, netclass_overrides={})
    r.update(pr.rules.get("routing", {}))
    return r
