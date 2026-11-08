"""Greedy legalizer (paper Sec. 3.3 "Macro legalization", adapted to PCBs).

Grid occupancy per side (default 0.05 mm cells). Parts that are already legal keep their spot; the
rest are processed in descending area and moved to the nearest feasible grid point (nearest along
their edge for edge connectors). Rasterisation is conservative (every touched cell counts), so a
result that succeeds is exactly legal; it is re-checked with energies.legality_report anyway.
If a part has no feasible spot it may try another rotation (if allowed), the other side (if
allowed), or another edge (edge parts); otherwise legalization fails.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import box_dims, dims_rot, transform_offsets
from .layout import EDGES, corner_centers, edge_place
from .kicad_io import Problem


def _sat(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int32)
    s[1:, 1:] = a.astype(np.int32).cumsum(0).cumsum(1)
    return s


def _box_sums(sat, bh, bw):
    """sum of every bh x bw window (top-left anchored) -> (ny-bh+1, nx-bw+1)."""
    return sat[bh:, bw:] - sat[:-bh, bw:] - sat[bh:, :-bw] + sat[:-bh, :-bw]


class Grid:
    def __init__(self, ol, g=0.05):
        self.ol, self.g = ol, g
        self.nx = int(math.ceil(ol.W / g)); self.ny = int(math.ceil(ol.H / g))
        self.occ = np.zeros((2, self.ny, self.nx), dtype=bool)
        # cells not fully inside the rounded rectangle inset by the edge clearance
        xs0 = np.arange(self.nx) * g; ys0 = np.arange(self.ny) * g
        X0, Y0 = np.meshgrid(xs0, ys0); X1, Y1 = X0 + g, Y0 + g
        bad = np.zeros((self.ny, self.nx), dtype=bool)
        for (px, py) in ((X0, Y0), (X1, Y0), (X0, Y1), (X1, Y1)):
            bad |= self.sdf(px, py) + ol.edge > 1e-9
        self.outside = bad

    def sdf(self, px, py):
        ol = self.ol
        qx = np.abs(px - ol.W / 2) - (ol.W / 2 - ol.r)
        qy = np.abs(py - ol.H / 2) - (ol.H / 2 - ol.r)
        return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - ol.r

    def cells(self, lo, hi):
        """conservative cell index range [i0,i1) x [j0,j1) covering box lo..hi (clipped)."""
        g = self.g
        i0 = int(math.floor(lo[0] / g + 1e-9)); i1 = int(math.ceil(hi[0] / g - 1e-9))
        j0 = int(math.floor(lo[1] / g + 1e-9)); j1 = int(math.ceil(hi[1] / g - 1e-9))
        return max(i0, 0), min(i1, self.nx), max(j0, 0), min(j1, self.ny)

    def mark(self, side, lo, hi, val=True):
        i0, i1, j0, j1 = self.cells(lo, hi)
        if i1 > i0 and j1 > j0:
            if side == 2:
                self.occ[:, j0:j1, i0:i1] = val
            else:
                self.occ[side, j0:j1, i0:i1] = val

    def free(self, side, lo, hi, use_outside=True):
        i0, i1, j0, j1 = self.cells(lo, hi)
        if i1 <= i0 or j1 <= j0:
            return True
        blk = self.occ[side, j0:j1, i0:i1]
        if use_outside:
            blk = blk | self.outside[j0:j1, i0:i1]
        return not blk.any()


def legalize(pr: Problem, c, k, s, ol, edge_assign, g=0.05, clear=None, max_disp=None, verbose=False):
    clear = pr.rules.get("clearance", {}).get("courtyard", 0.05) if clear is None else clear
    N = len(pr.parts)
    c = np.array(c, dtype=float); k = np.array(k, dtype=np.int64)
    s = (np.asarray(s) > 0.5).astype(np.int64)
    ea = dict(edge_assign)
    wh0 = box_dims(pr)
    G = Grid(ol, g)
    hc = clear / 2

    def box(i, ci=None, ki=None):
        ci = c[i] if ci is None else ci
        ki = k[i] if ki is None else ki
        w, h = dims_rot(wh0[i], ki)
        return np.array([ci[0] - w / 2, ci[1] - h / 2]), np.array([ci[0] + w / 2, ci[1] + h / 2])

    def ko_boxes(i, ci=None, ki=None):
        p = pr.parts[i]
        ci = c[i] if ci is None else ci
        ki = k[i] if ki is None else ki
        out = []
        for b in p.bottom_hole_keepout:
            q = transform_offsets(np.array([[b[0], b[1]], [b[2], b[3]]]), np.array([ki, ki]), np.array([0, 0]))
            out.append((ci + q.min(0), ci + q.max(0)))
        return out

    def inboard(i, lo, hi, ki):
        """box minus mouth overhang + edge clearance (edge parts)."""
        p = pr.parts[i]
        lo = lo.copy(); hi = hi.copy()
        if p.fixed == "edge":
            ax = 0 if p.edge_mouth in ("+x", "-x") else 1
            sg = 1.0 if p.edge_mouth[0] == "+" else -1.0
            v = np.zeros(2); v[ax] = sg
            d = transform_offsets(v, np.array(ki), np.array(0))
            sh = p.edge_overhang + ol.edge
            if d[0] < -0.5: lo[0] += sh
            if d[0] > 0.5: hi[0] -= sh
            if d[1] < -0.5: lo[1] += sh
            if d[1] > 0.5: hi[1] -= sh
        return lo, hi

    def inside(i, lo, hi, ki):
        a, b = inboard(i, lo, hi, ki)
        pts = np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])
        return bool(np.all(G.sdf(pts[:, 0], pts[:, 1]) + ol.edge <= 1e-6))

    def place(i):
        lo, hi = box(i)
        side = 2 if pr.parts[i].corner_hole else s[i]
        G.mark(side, lo - hc, hi + hc)
        if side != 2:
            for (a, b) in ko_boxes(i):
                G.mark(1 - side, a, b)

    def fits_here(i):
        lo, hi = box(i)
        if not inside(i, lo, hi, k[i]):
            return False
        if not G.free(s[i], lo - hc, hi + hc, use_outside=pr.parts[i].fixed != "edge"):
            return False
        for (a, b) in ko_boxes(i):
            if not G.free(1 - s[i], a, b, use_outside=False):
                return False
        return True

    # 1) corner holes
    for i, xy in corner_centers(pr, ol).items():
        c[i] = xy; k[i] = 0; s[i] = 0
        place(i)
    order_all = sorted([i for i in range(N) if not pr.parts[i].corner_hole], key=lambda i: -wh0[i].prod())
    # 2) parts already legal keep their place (largest first)
    pending = []
    for i in order_all:
        if fits_here(i):
            place(i)
        else:
            pending.append(i)
    # 3) the rest: nearest feasible grid point
    failed = []
    moved = {}
    for i in pending:
        p = pr.parts[i]
        target = c[i].copy()
        best = None
        if p.fixed == "edge":
            edges = [ea.get(i, 0)] + [e for e in range(4) if e != ea.get(i, 0)]
            for e in edges:
                (cx0, cy0), kk = edge_place(p, e, 0.0, ol)
                w, h = dims_rot(wh0[i], kk)
                L = ol.H if e in (0, 1) else ol.W
                cand = np.arange(w / 2 if e > 1 else h / 2, L - (w / 2 if e > 1 else h / 2) + 1e-9, g) if True else None
                tgt_along = target[1] if e in (0, 1) else target[0]
                for a in cand[np.argsort(np.abs(cand - tgt_along))]:
                    ci = np.array([cx0, a]) if e in (0, 1) else np.array([a, cy0])
                    lo, hi = box(i, ci, kk)
                    if not inside(i, lo, hi, kk) or not G.free(0, lo - hc, hi + hc, use_outside=False):
                        continue
                    if any(not G.free(1, A, B, use_outside=False) for (A, B) in ko_boxes(i, ci, kk)):
                        continue
                    pen = abs(a - tgt_along) + (0 if e == edges[0] else 1e3)
                    best = (pen, ci, kk, 0, e)
                    break
                if best is not None:
                    break
        else:
            rots = [k[i]] + ([r for r in range(4) if r != k[i]] if p.allow_rotate else [])
            sides = [s[i]] + ([1 - s[i]] if p.allow_bottom else [])
            done = False
            for sd in sides:
                sat_main = _sat(G.occ[sd] | G.outside)   # once per side, shared by all rotations
                satb = None
                for kk in rots:
                    w, h = dims_rot(wh0[i], kk)
                    bw = int(math.ceil((w + clear) / g - 1e-9)); bh = int(math.ceil((h + clear) / g - 1e-9))
                    if bw > G.nx or bh > G.ny:
                        continue
                    sums = _box_sums(sat_main, bh, bw)
                    feas = sums == 0
                    kos = ko_boxes(i, np.zeros(2), kk)
                    if kos and feas.any():
                        satb = _sat(G.occ[1 - sd]) if satb is None else satb
                        for (a, b) in kos:
                            # ko box relative to the box top-left anchor
                            off_lo = a - np.array([-(w + clear) / 2, -(h + clear) / 2])
                            ko_w = int(math.ceil((b - a)[0] / g)) + 1; ko_h = int(math.ceil((b - a)[1] / g)) + 1
                            di = int(math.floor(off_lo[0] / g)); dj = int(math.floor(off_lo[1] / g))
                            ks = _box_sums(satb, ko_h, ko_w)
                            ok = np.zeros_like(feas)
                            H_, W_ = feas.shape
                            for jj in range(H_):
                                sj = jj + dj
                                if sj < 0 or sj >= ks.shape[0]:
                                    continue
                                i0 = max(0, -di); i1 = min(W_, ks.shape[1] - di)
                                if i1 > i0:
                                    ok[jj, i0:i1] = ks[sj, i0 + di:i1 + di] == 0
                            feas &= ok
                    if not feas.any():
                        continue
                    jj, ii = np.nonzero(feas)
                    cxs = ii * g + (w + clear) / 2; cys = jj * g + (h + clear) / 2
                    d2 = (cxs - target[0]) ** 2 + (cys - target[1]) ** 2
                    t = int(np.argmin(d2))
                    pen = math.sqrt(d2[t]) + (0 if (sd == s[i] and kk == k[i]) else 0.5) + (2.0 * sd if sd != s[i] else 0)
                    if best is None or pen < best[0]:
                        best = (pen, np.array([cxs[t], cys[t]]), kk, sd, None)
                    if best is not None and best[3] == s[i] and best[2] == k[i] and best[0] < max(1.0, 0.5 * min(w, h)):
                        done = True  # close spot without changing rotation/side: good enough
                        break
                if done:
                    break
        if best is None:
            failed.append(i)
            continue
        _, ci, kk, sd, e = best
        moved[i] = float(np.linalg.norm(ci - target))
        c[i] = ci; k[i] = kk; s[i] = sd
        if e is not None:
            ea[i] = e
        place(i)
    info = dict(ok=not failed, failed=[pr.parts[i].ref for i in failed], n_moved=len(moved),
                max_disp=max(moved.values()) if moved else 0.0, mean_disp=float(np.mean(list(moved.values()))) if moved else 0.0)
    if verbose:
        print("legalize:", info)
    return c, k, s.astype(float), ea, info
