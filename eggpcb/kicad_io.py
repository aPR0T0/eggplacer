"""KiCad board <-> placement problem.

Everything here is in millimetres in KiCad's board frame (x right, y DOWN).

Conventions (verified empirically against pcbnew 10.0.4, see tests/test_conventions.py):
  * top-side footprint with orientation r (deg):  world = pos + R(r) @ local
        R(r) = [[cos r,  sin r],
                [-sin r, cos r]]
  * bottom-side footprint (Flip LEFT_RIGHT then SetOrientation(r)):
        world = pos + R(r) @ (local.x, -local.y)
    i.e. a bottom part is its top-side local geometry mirrored in y, then rotated.

A "part" is described in its *top-local* frame relative to the centre of its keep-out box
(courtyard). The placement variable is that box centre, plus a rotation index k (r = 90k) and
a side (0 = top, 1 = bottom).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

NM = 1e-6  # KiCad internal units are nm


def rot_mat(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, s], [-s, c]])


def local_to_world(local: np.ndarray, pos, deg: float, bottom: bool) -> np.ndarray:
    """local: (...,2) top-local offsets -> world coords."""
    loc = np.array(local, dtype=float)
    if bottom:
        loc = loc * np.array([1.0, -1.0])
    return np.asarray(pos) + loc @ rot_mat(deg).T


def world_to_local(world: np.ndarray, pos, deg: float, bottom: bool) -> np.ndarray:
    w = np.asarray(world, dtype=float) - np.asarray(pos)
    loc = w @ rot_mat(-deg).T
    if bottom:
        loc = loc * np.array([1.0, -1.0])
    return loc


@dataclass
class Part:
    ref: str
    lib: str
    value: str
    sheet: str
    kind: str                      # R, C, L, U, J, ... (reference prefix)
    w: float                       # keep-out box size in top-local frame at rotation 0
    h: float
    box_off: list                  # box centre relative to footprint origin (top-local)
    pad_names: list
    pad_off: list                  # (P,2) pad centres relative to box centre (top-local)
    pad_size: list                 # (P,2)
    pad_net: list                  # net index or -1
    holes: list                    # (H,3) hole centre (rel. box centre) + radius
    pos0: list                     # current footprint origin (world)
    rot0: float                    # current orientation (deg)
    side0: int                     # 0 top, 1 bottom
    locked: bool
    courtyard_ok: bool             # True if the footprint's own courtyard covered pads/body
    # filled in by apply_rules():
    fixed: str = "none"            # none | abs | edge
    edge_mouth: str = ""           # for edge parts: local direction that must face outward (+x,-x,+y,-y)
    edge_overhang: float = 0.0     # how far the box may stick out past the board edge on the mouth side
    allow_bottom: bool = False
    allow_rotate: bool = True
    corner_hole: bool = False      # mounting hole pinned to an outline corner
    cy_box: list = None            # footprint's own courtyard bbox [xmin,ymin,xmax,ymax] rel. box centre
    edge_line: float = 0.0         # mouth-axis coordinate (rel. box centre) that must sit on the board edge
    bottom_hole_keepout: list = field(default_factory=list)  # boxes (rel. centre) blocked on the other side

    @property
    def area(self):
        return self.w * self.h


@dataclass
class Problem:
    name: str
    parts: list
    nets: list                      # list of dicts {name, pins: [[part, pad], ...], weight}
    layers: int = 2
    rules: dict = field(default_factory=dict)
    src_outline: list = field(default_factory=list)  # [xmin,ymin,xmax,ymax] of the source Edge.Cuts

    # ---------- flattened arrays (built lazily) ----------
    def pin_arrays(self):
        """Return (pin_part[int], pin_off[(N,2)], pin_net[int]) for every pad on a multi-pin net."""
        pp, po, pn = [], [], []
        for ni, net in enumerate(self.nets):
            for (pi, k) in net["pins"]:
                pp.append(pi)
                po.append(self.parts[pi].pad_off[k])
                pn.append(ni)
        return np.array(pp, dtype=np.int64), np.array(po, dtype=float), np.array(pn, dtype=np.int64)

    def to_json(self, path):
        d = dict(name=self.name, layers=self.layers, rules=self.rules, src_outline=self.src_outline,
                 nets=self.nets, parts=[asdict(p) for p in self.parts])
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path) -> "Problem":
        d = json.loads(Path(path).read_text())
        parts = [Part(**p) for p in d["parts"]]
        return Problem(name=d["name"], parts=parts, nets=d["nets"], layers=d["layers"],
                       rules=d.get("rules", {}), src_outline=d.get("src_outline", []))

    def index(self, ref):
        for i, p in enumerate(self.parts):
            if p.ref == ref:
                return i
        raise KeyError(ref)


# ----------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------

def _shape_points(shape, pcbnew):
    """Sample a footprint graphic's outline as world points (mm)."""
    pts = []
    st = shape.GetShape()
    if st == pcbnew.SHAPE_T_CIRCLE:
        c = shape.GetCenter(); r = shape.GetRadius()
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            pts.append((c.x + r * math.cos(a), c.y + r * math.sin(a)))
    elif st == pcbnew.SHAPE_T_POLY:
        poly = shape.GetPolyShape()
        for oi in range(poly.OutlineCount()):
            ol = poly.Outline(oi)
            for vi in range(ol.PointCount()):
                p = ol.CPoint(vi); pts.append((p.x, p.y))
    elif st == pcbnew.SHAPE_T_RECTANGLE:
        for c in shape.GetRectCorners():
            pts.append((c.x, c.y))
    elif st == pcbnew.SHAPE_T_ARC:
        s, m, e = shape.GetStart(), shape.GetArcMid(), shape.GetEnd()
        pts += [(s.x, s.y), (m.x, m.y), (e.x, e.y)]
    else:  # segment, bezier
        s, e = shape.GetStart(), shape.GetEnd()
        pts += [(s.x, s.y), (e.x, e.y)]
    return [(x * NM, y * NM) for x, y in pts]


def parse_board(pcb_path: str, name: str | None = None) -> Problem:
    import pcbnew  # KiCad's bundled module

    board = pcbnew.LoadBoard(str(pcb_path))
    parts: list[Part] = []
    pad_records = []  # (part_idx, pad_idx, netname)

    for fp in board.GetFootprints():
        ref = fp.GetReference()
        bottom = fp.IsFlipped()
        deg = fp.GetOrientationDegrees()
        pos = np.array([fp.GetPosition().x * NM, fp.GetPosition().y * NM])
        cy_layer = pcbnew.B_CrtYd if bottom else pcbnew.F_CrtYd
        fab_layer = pcbnew.B_Fab if bottom else pcbnew.F_Fab
        silk_layer = pcbnew.B_SilkS if bottom else pcbnew.F_SilkS

        # pads (top-local, relative to footprint origin)
        pad_names, pad_loc, pad_sz, pad_nets, holes = [], [], [], [], []
        pad_pts = []
        for pad in fp.Pads():
            wpos = np.array([pad.GetPosition().x * NM, pad.GetPosition().y * NM])
            loc = world_to_local(wpos, pos, deg, bottom)
            if pad.HasHole():
                d = max(pad.GetDrillSize().x, pad.GetDrillSize().y) * NM
                holes.append((loc[0], loc[1], d / 2))
            if not pad.IsOnCopperLayer():
                continue  # paste-only apertures, NPTH without copper
            bb = pad.GetBoundingBox()
            for (x, y) in [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetBottom()),
                           (bb.GetLeft(), bb.GetBottom()), (bb.GetRight(), bb.GetTop())]:
                pad_pts.append(world_to_local(np.array([x * NM, y * NM]), pos, deg, bottom))
            pad_names.append(pad.GetNumber())
            pad_loc.append(loc)
            sz = pad.GetSize(pcbnew.F_Cu) if hasattr(pad, "GetSize") else pad.GetSize()
            sx, sy = sz.x * NM, sz.y * NM
            pdeg = (pad.GetOrientationDegrees() - deg) % 180
            if abs(pdeg - 90) < 1:
                sx, sy = sy, sx
            pad_sz.append((sx, sy))
            pad_nets.append(pad.GetNetname() if pad.GetNetCode() > 0 else "")

        # courtyard
        cy_pts = []
        cy = fp.GetCourtyard(cy_layer)
        for oi in range(cy.OutlineCount()):
            ol = cy.Outline(oi)
            for vi in range(ol.PointCount()):
                p = ol.CPoint(vi)
                cy_pts.append(world_to_local(np.array([p.x * NM, p.y * NM]), pos, deg, bottom))
        # body outline. A valid courtyard (covers pads + fab) is authoritative. Otherwise fall back
        # to pads + fab + silkscreen + 0.25 mm IPC courtyard excess: some custom footprints only draw
        # the package body on silk (e.g. PP-1212-8), so pads alone underestimate it.
        fab_pts, silk_pts = [], []
        for g in fp.GraphicalItems():
            if g.GetClass() in ("PCB_TEXT", "PCB_FIELD", "PCB_TEXTBOX"):
                continue
            if g.GetLayer() in (fab_layer, silk_layer):
                if g.GetLayer() == silk_layer:
                    gb = g.GetBoundingBox()
                    if max(gb.GetWidth(), gb.GetHeight()) * NM < 0.8:
                        continue  # pin-1 dots / polarity marks are not package body
                tgt = fab_pts if g.GetLayer() == fab_layer else silk_pts
                for (x, y) in _shape_points(g, pcbnew):
                    tgt.append(world_to_local(np.array([x, y]), pos, deg, bottom))
        core = np.array(pad_pts + fab_pts) if (pad_pts or fab_pts) else np.zeros((1, 2))
        full = np.array(pad_pts + fab_pts + silk_pts) if (pad_pts or fab_pts or silk_pts) else np.zeros((1, 2))
        bmin, bmax = core.min(0), core.max(0)
        fmin, fmax = full.min(0), full.max(0)
        if cy_pts:
            c = np.array(cy_pts); cmin, cmax = c.min(0), c.max(0)
            ok = bool(np.all(cmin <= bmin + 1e-3) and np.all(cmax >= bmax - 1e-3))
            if ok:
                lo, hi = cmin, cmax
            else:  # broken courtyard
                lo, hi = np.minimum(cmin, fmin - 0.25), np.maximum(cmax, fmax + 0.25)
        else:
            ok = False
            lo, hi = fmin - 0.25, fmax + 0.25
        box_off = (lo + hi) / 2
        w, h = (hi - lo)
        cy_box = None
        if cy_pts:
            cy_box = [float(cmin[0] - box_off[0]), float(cmin[1] - box_off[1]),
                      float(cmax[0] - box_off[0]), float(cmax[1] - box_off[1])]

        sheet = fp.GetSheetname() if hasattr(fp, "GetSheetname") else ""
        kind = re.sub(r"\d+$", "", ref)
        p = Part(ref=ref, lib=str(fp.GetFPID().GetUniStringLibItemName()),
                 value=fp.GetValue(), sheet=sheet, kind=kind,
                 w=float(w), h=float(h), box_off=box_off.tolist(),
                 pad_names=pad_names,
                 pad_off=[(np.asarray(l) - box_off).tolist() for l in pad_loc],
                 pad_size=[list(s) for s in pad_sz],
                 pad_net=[-1] * len(pad_names),
                 holes=[[hx - box_off[0], hy - box_off[1], r] for hx, hy, r in holes],
                 pos0=pos.tolist(), rot0=float(deg), side0=int(bottom), locked=bool(fp.IsLocked()),
                 courtyard_ok=ok, cy_box=cy_box)
        pi = len(parts)
        parts.append(p)
        for k, nn in enumerate(pad_nets):
            if nn:
                pad_records.append((pi, k, nn))

    # nets: keep nets touching >= 2 distinct pads
    by_net: dict[str, list] = {}
    for pi, k, nn in pad_records:
        by_net.setdefault(nn, []).append([pi, k])
    nets = []
    for nn, pins in sorted(by_net.items()):
        if len(pins) < 2:
            continue
        ni = len(nets)
        nets.append(dict(name=nn, pins=pins, weight=1.0))
        for pi, k in pins:
            parts[pi].pad_net[k] = ni

    # source outline
    xs, ys = [], []
    for d in board.GetDrawings():
        if d.GetLayer() == pcbnew.Edge_Cuts:
            bb = d.GetBoundingBox()
            xs += [bb.GetLeft() * NM, bb.GetRight() * NM]; ys += [bb.GetTop() * NM, bb.GetBottom() * NM]
    src_outline = [min(xs), min(ys), max(xs), max(ys)] if xs else []
    return Problem(name=name or Path(pcb_path).stem, parts=parts, nets=nets, src_outline=src_outline)
