"""Synthetic PCB circuits matched to the target board's statistics (Phase-1 proposal).

generate(rng, lib) -> (Problem, centres, k, s, Outline, edge_assign)
  1. BOM: N ~ U(0.5, 1.3) x N_ref parts; class mix around the board's (2-pin 78 %, 3-9 pin 15 %,
     ICs 4-5 %, connectors 3 %); ~80 % real footprint templates from the board, ~20 % generic.
  2. Dense latent placement: aspect U(1,2), density U(0.45, 0.72); 3-8 cluster seeds ("sheets"),
     ICs at seeds, passives around an owner IC; packed with the greedy legalizer.
  3. Netlist on the latent placement: GND ~26 % of pins (IC GND pins, exposed pads, one pad of
     most caps), 1-4 power rails ~13 %, decoupling caps between IC power pins and GND, signal nets
     with the board's fan-out histogram, pins chosen with p ~ exp(-d / l).
The training label is produced afterwards by datagen.py (quick analytical placer + legalizer).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .energies import Outline
from .kicad_io import Part, Problem
from .layout import EDGES, apply_constraints, corner_centers, edge_place
from .legalize import legalize

SIGNAL_FANOUT = {2: 105, 3: 37, 4: 27, 6: 2, 7: 1, 8: 8, 10: 4, 12: 1}
CLASS_OF_KIND = {"R": "p2", "C": "p2", "L": "p2", "LED": "p2", "D": "p2", "SW": "p2",
                 "Q": "small", "Y": "small", "U": "ic", "J": "conn", "BH": "hole"}


class Library:
    """footprint templates harvested from the parsed target board."""

    def __init__(self, pr: Problem):
        self.templates = {"p2": [], "small": [], "ic": [], "conn": [], "hole": [], "edge": []}
        self.weights = {k: [] for k in self.templates}
        seen = {}
        for p in pr.parts:
            npads = sum(1 for n in p.pad_net if n >= 0) or len(p.pad_off)
            cls = CLASS_OF_KIND.get(p.kind, "small")
            if cls == "p2" and len(p.pad_off) != 2:
                cls = "small"
            if p.kind in ("L",) and len(p.pad_off) > 2:
                cls = "small"
            if p.kind == "U" and len(p.pad_off) <= 9:
                cls = "small"
            key = (cls, p.lib)
            if key not in seen:
                t = copy.deepcopy(p)
                t.pos0, t.rot0, t.side0, t.locked = [0, 0], 0.0, 0, False
                seen[key] = len(self.templates[cls])
                self.templates[cls].append(t); self.weights[cls].append(0)
                if p.fixed == "edge":
                    self.templates["edge"].append(t); self.weights["edge"].append(1)
            self.weights[cls][seen[key]] += 1
        for k in self.weights:
            w = np.array(self.weights[k], dtype=float)
            self.weights[k] = w / w.sum() if w.sum() else w
        self.n_ref = len(pr.parts)
        mix = {}
        for p in pr.parts:
            cls = CLASS_OF_KIND.get(p.kind, "small")
            if cls == "p2" and len(p.pad_off) != 2:
                cls = "small"
            if p.kind == "U" and len(p.pad_off) <= 9:
                cls = "small"
            mix[cls] = mix.get(cls, 0) + 1
        mix.pop("hole", None)
        tot = sum(mix.values())
        self.mix = {k: v / tot for k, v in mix.items()}
        # signal-net fan-out histogram of the target board (ground / power rails excluded)
        skip = set(pr.rules.get("decoupling", {}).get("gnd_nets", ["GND"])) | set(pr.rules.get("net_weights", {}).get("power_nets", []))
        fo = {}
        for n in pr.nets:
            if n["name"] not in skip and 2 <= len(n["pins"]) <= 16:
                fo[len(n["pins"])] = fo.get(len(n["pins"]), 0) + 1
        self.fanout = fo or dict(SIGNAL_FANOUT)


def generic_library():
    """Library without a target board: generic parts only, a typical mixed-signal class mix and
    fan-out, plus an M2-style corner hole. Used by the demo and the tests."""
    lib = Library.__new__(Library)
    lib.templates = {"p2": [], "small": [], "ic": [], "conn": [], "hole": [], "edge": []}
    lib.weights = {k: np.array([]) for k in lib.templates}
    hole = _part("BH", 4.7, 4.7, [[0.0, 0.0]], [(4.2, 4.2)], holes=[(0.0, 0.0, 1.1)])
    lib.templates["hole"].append(hole); lib.weights["hole"] = np.array([1.0])
    lib.n_ref = 200
    lib.mix = {"p2": 0.78, "small": 0.14, "ic": 0.045, "conn": 0.035}
    lib.fanout = dict(SIGNAL_FANOUT)
    return lib


# ------------------------------------------------------------------------------- generic parts
def _part(kind, w, h, pads, sizes, holes=(), ref="X"):
    pads = np.asarray(pads, dtype=float)
    return Part(ref=ref, lib=f"GEN_{kind}", value="", sheet="", kind=kind, w=float(w), h=float(h),
                box_off=[0.0, 0.0], pad_names=[str(i + 1) for i in range(len(pads))],
                pad_off=pads.tolist(), pad_size=[list(s) for s in sizes], pad_net=[-1] * len(pads),
                holes=[list(h_) for h_ in holes], pos0=[0, 0], rot0=0.0, side0=0, locked=False, courtyard_ok=True)


def gen_p2(rng):
    w = rng.uniform(0.9, 3.6); h = w * rng.uniform(0.42, 0.7)
    px = w * rng.uniform(0.24, 0.32); ps = (w * 0.22, h * 0.6)
    return _part(rng.choice(["R", "C", "L"]), w, h, [[-px, 0], [px, 0]], [ps, ps])


def gen_small(rng):
    n = int(rng.integers(3, 10)); w = rng.uniform(1.6, 4.6); h = rng.uniform(1.4, 4.0)
    left = (n + 1) // 2; right = n - left
    pitch = min(0.95, (h - 0.5) / max(left, 1))
    pads, sizes = [], []
    for side, cnt in ((-1, left), (1, right)):
        for i in range(cnt):
            pads.append([side * (w / 2 - 0.45), (i - (cnt - 1) / 2) * pitch]); sizes.append((0.5, pitch * 0.55))
    return _part("Q", w, h, pads, sizes)


def gen_ic(rng):
    per = int(rng.integers(3, 26)); pitch = rng.choice([0.4, 0.5, 0.65])
    body = per * pitch + rng.uniform(0.6, 1.6)
    qfp = rng.random() < 0.4 and per >= 8
    L = body + (2.4 if qfp else 0.9)
    pads, sizes = [], []
    r = body / 2 + (0.8 if qfp else 0.1)
    for side in range(4):
        for i in range(per):
            a = (i - (per - 1) / 2) * pitch
            p = [(-r, a), (a, r), (r, -a), (-a, -r)][side]
            pads.append(list(p)); sizes.append((1.0 if qfp else 0.6, pitch * 0.55) if side % 2 == 0 else (pitch * 0.55, 1.0 if qfp else 0.6))
    if not qfp and rng.random() < 0.7:
        pads.append([0.0, 0.0]); sizes.append((body * 0.6, body * 0.6))
    return _part("U", L, L, pads, sizes)


def gen_conn(rng):
    rows = int(rng.integers(1, 3)); n = int(rng.integers(2, 21)); pitch = rng.choice([0.5, 0.8, 1.0, 1.27])
    w = max(n * pitch + rng.uniform(1.5, 4.0), 4.0); h = rng.uniform(3.5, 9.0)
    pads, sizes = [], []
    for r_ in range(rows):
        y = (r_ - (rows - 1) / 2) * min(1.5, h / 3)
        for i in range(n):
            pads.append([(i - (n - 1) / 2) * pitch, y]); sizes.append((pitch * 0.55, 0.9))
    holes = []
    if rng.random() < 0.5:
        holes = [[-w / 2 + 0.9, h / 2 - 0.9, 0.4], [w / 2 - 0.9, h / 2 - 0.9, 0.4]]
    return _part("J", w, h, pads, sizes, holes)


def _template(lib: Library, cls, rng):
    T = lib.templates[cls]
    t = copy.deepcopy(T[rng.choice(len(T), p=lib.weights[cls])])
    t.pad_net = [-1] * len(t.pad_off)
    t.fixed = "none"; t.corner_hole = False; t.edge_mouth = ""; t.edge_overhang = 0.0; t.edge_line = 0.0
    return t


GEN = {"p2": gen_p2, "small": gen_small, "ic": gen_ic, "conn": gen_conn}


# ------------------------------------------------------------------------------- generator
def generate(rng: np.random.Generator, lib: Library, cfg: dict, n_scale=(0.5, 1.3), density=(0.55, 0.80),
             real_frac=0.8, grid=0.2):
    N = int(round(lib.n_ref * rng.uniform(*n_scale)))
    mix = {k: v * rng.uniform(0.75, 1.25) for k, v in lib.mix.items()}
    tot = sum(mix.values()); mix = {k: v / tot for k, v in mix.items()}
    classes = rng.choice(list(mix), size=N, p=list(mix.values()))
    parts: list[Part] = []
    for cls in classes:
        p = _template(lib, cls, rng) if (rng.random() < real_frac and lib.templates[cls]) else GEN[cls](rng)
        p.kind = p.kind if cls != "conn" else "J"
        parts.append(p)
    # constraints
    two_sided = rng.random() < 0.5
    for p in parts:
        p.allow_rotate = True
        p.allow_bottom = two_sided and p.kind in ("R", "C", "L") and len(p.pad_off) == 2
        hb = 0.25
        p.bottom_hole_keepout = [[hx - r - hb, hy - r - hb, hx + r + hb, hy + r + hb] for hx, hy, r in p.holes]
    conns = [i for i, p in enumerate(parts) if p.kind == "J"]
    n_edge = int(rng.integers(1, 4))
    # some edge parts are the target board's own edge connectors (with their mouths)
    for j in range(n_edge):
        if lib.templates["edge"] and rng.random() < 0.6:
            t = copy.deepcopy(lib.templates["edge"][rng.integers(len(lib.templates["edge"]))])
            t.pad_net = [-1] * len(t.pad_off); t.allow_bottom = False; t.allow_rotate = False
            t.bottom_hole_keepout = [[hx - r - 0.25, hy - r - 0.25, hx + r + 0.25, hy + r + 0.25] for hx, hy, r in t.holes]
            parts.append(t)
        elif conns:
            i = conns.pop(int(rng.integers(len(conns))))
            p = parts[i]; p.fixed = "edge"; p.allow_rotate = False; p.allow_bottom = False
            p.edge_mouth = str(rng.choice(["+x", "-x", "+y", "-y"]))
            ax = 0 if p.edge_mouth[1] == "x" else 1
            p.edge_overhang = float(rng.uniform(0.0, 1.0))
            sg = 1.0 if p.edge_mouth[0] == "+" else -1.0
            half = (p.w, p.h)[ax] / 2
            p.edge_line = sg * (half - p.edge_overhang)
    ics = [i for i, p in enumerate(parts) if p.kind == "U" and len(p.pad_off) >= 12]
    if ics and rng.random() < 0.5:
        i = max(ics, key=lambda i: parts[i].w * parts[i].h)
        parts[i].fixed = "abs"; parts[i].allow_rotate = False
    if rng.random() < 0.5 and lib.templates["hole"]:
        for _ in range(4):
            t = _template(lib, "hole", rng); t.corner_hole = True; t.fixed = "corner"; t.allow_rotate = False
            t.bottom_hole_keepout = [[hx - r - 0.25, hy - r - 0.25, hx + r + 0.25, hy + r + 0.25] for hx, hy, r in t.holes]
            parts.append(t)
    for n_, p in enumerate(parts):
        p.ref = f"{p.kind}{n_}"
    N = len(parts)

    # outline
    areas = np.array([p.w * p.h for p in parts])
    top_only = sum(a for a, p in zip(areas, parts) if not p.allow_bottom)
    flex = sum(a for a, p in zip(areas, parts) if p.allow_bottom)
    need = max(top_only, (top_only + flex) / 2) if two_sided else top_only + flex
    rho = rng.uniform(*density)
    A = need / rho
    asp = rng.uniform(1.0, 2.0)
    W = math.sqrt(A * asp); H = A / W
    if rng.random() < 0.5:
        W, H = H, W
    big = max(max(p.w, p.h) for p in parts)
    W = max(W, big + 11); H = max(H, big + 11)
    ol = Outline(float(W), float(H), 0.5, 0.2)

    # latent placement: clusters
    G = int(rng.integers(3, 9))
    seeds = np.stack([rng.uniform(0.15, 0.85, G) * W, rng.uniform(0.15, 0.85, G) * H], 1)
    owner = -np.ones(N, dtype=int)
    grp = rng.integers(G, size=N)
    ic_idx = [i for i, p in enumerate(parts) if p.kind == "U"]
    c = np.zeros((N, 2))
    for i in range(N):
        c[i] = seeds[grp[i]] + rng.normal(0, 0.06 * max(W, H), 2)
    for i, p in enumerate(parts):
        if p.kind in ("R", "C", "L", "LED", "D") and ic_idx:
            same = [j for j in ic_idx if grp[j] == grp[i]] or ic_idx
            owner[i] = same[int(rng.integers(len(same)))]
            c[i] = c[owner[i]] + rng.normal(0, 2.5, 2)
    k = np.array([int(rng.integers(4)) if (p.allow_rotate and p.fixed == "none") else 0 for p in parts])
    s = np.array([float(rng.random() < rng.uniform(0.1, 0.4)) if p.allow_bottom else 0.0 for p in parts])
    sheets = [f"/S{g}/" for g in range(G)]
    for i, p in enumerate(parts):
        p.sheet = sheets[grp[i]]
    pr = Problem(name="synth", parts=parts, nets=[], layers=2, rules=cfg)
    c, k, s, ea = apply_constraints(pr, np.clip(c, 1, [W - 1, H - 1]), k, s, ol)
    # pack: parts that fit at their cluster target stay, the rest go to the nearest feasible spot
    c, k, s, ea, info = legalize(pr, c, k, s, ol, ea, g=grid)
    if not info["ok"]:
        return None

    # ----------------------------------------------------------------- netlist on the placement
    pins = []  # (part, pad) of every copper pad
    for i, p in enumerate(parts):
        for kk in range(len(p.pad_off)):
            pins.append((i, kk))
    from .geometry import transform_offsets
    pw = np.array([c[i] + transform_offsets(np.array(parts[i].pad_off[kk]), np.array(k[i]), np.array(int(s[i]))) for i, kk in pins])
    P = len(pins)
    net_of = -np.ones(P, dtype=int)
    nets = []

    def new_net(name, members, weight=1.0):
        ni = len(nets)
        nets.append(dict(name=name, pins=[[int(pins[m][0]), int(pins[m][1])] for m in members], weight=weight))
        for m in members:
            net_of[m] = ni

    pin_ids_of = {}
    for m, (i, kk) in enumerate(pins):
        pin_ids_of.setdefault(i, []).append(m)
    # power rails
    R = int(rng.integers(1, 5))
    rail_members = [[] for _ in range(R)]
    gnd = []
    for i, p in enumerate(parts):
        ids = pin_ids_of.get(i, [])
        if not ids or p.corner_hole:
            if p.corner_hole and ids:
                gnd.append(ids[0])
            continue
        if len(ids) == 2 and p.kind == "C" and rng.random() < 0.65:
            gnd.append(ids[0]); rail_members[int(rng.integers(R))].append(ids[1])  # decoupling-like
            continue
        if len(ids) == 2:
            if rng.random() < 0.4:
                gnd.append(ids[int(rng.integers(2))])
            continue
        for m in ids:
            u = rng.random()
            sz = parts[i].pad_size[pins[m][1]]
            if sz[0] * sz[1] > 1.0 and p.kind == "U":
                gnd.append(m)  # exposed pad
            elif u < (0.24 if p.kind == "J" else 0.2):
                gnd.append(m)
            elif u < 0.28:
                rail_members[int(rng.integers(R))].append(m)
    new_net(cfg.get("decoupling", {}).get("gnd_nets", ["GND"])[0], gnd, cfg.get("net_weights", {}).get("gnd_weight", 0.15))
    for r_, mem in enumerate(rail_members):
        if len(mem) >= 2:
            new_net(f"PWR{r_}", mem, 0.5)
    # signal nets with the reference fan-out, spatially local
    fan_v = np.array(list(lib.fanout)); fan_p = np.array(list(lib.fanout.values()), float); fan_p /= fan_p.sum()
    ell = rng.uniform(0.05, 0.2) * math.hypot(W, H)
    free = [m for m in range(P) if net_of[m] < 0]
    rng.shuffle(free)
    pin_part = np.array([pp[0] for pp in pins])
    nc_frac = 0.05
    for m in free:
        if net_of[m] >= 0:
            continue
        p = parts[pins[m][0]]
        if len(p.pad_off) > 2 and rng.random() < nc_frac:
            continue
        f = int(rng.choice(fan_v, p=fan_p))
        cand = np.where((net_of < 0) & (pin_part != pins[m][0]))[0]
        if len(cand) == 0:
            break
        d = np.linalg.norm(pw[cand] - pw[m], axis=1)
        pr_ = np.exp(-d / ell); pr_ /= pr_.sum()
        take = rng.choice(cand, size=min(f - 1, len(cand)), replace=False, p=pr_)
        # at most one pin per part
        seen_parts = {pins[m][0]}; mem = [m]
        for t_ in take:
            if pins[t_][0] not in seen_parts:
                seen_parts.add(pins[t_][0]); mem.append(int(t_))
        if len(mem) >= 2:
            new_net(f"N{len(nets)}", mem)
    for ni, n in enumerate(nets):
        for (i, kk) in n["pins"]:
            parts[i].pad_net[kk] = ni
    # damp large fan-out like the real board
    cap = cfg.get("net_weights", {}).get("fanout_cap", 12)
    for n in nets:
        if len(n["pins"]) > cap:
            n["weight"] = float(min(1.0, n["weight"] * cap / len(n["pins"]) * 2))
    pr.nets = nets
    return pr, c, k, s, ol, ea
