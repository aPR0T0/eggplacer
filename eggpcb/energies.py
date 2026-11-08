"""Differentiable PCB placement energies (PyTorch), in millimetres.

Board frame: outline is the rounded rectangle [0,W] x [0,H] (y down, like KiCad), corner radius r.
State: centres c (N,2) of each part's keep-out box, rotation index k (N,) in {0..3}, side s (N,)
in [0,1] (0 = top, 1 = bottom; fractional values = soft side assignment used during guidance).

Terms (all scalar):
  wawl        weighted-average wirelength over every pad on every net (paper Eq. 9, pad offsets incl.
              rotation / mirroring)
  density     bin overflow (per side), exact box/bin overlap areas
  overlap     smooth pairwise overlap of keep-out boxes (softplus, Appendix D Eq. 11) incl. clearance,
              weighted by the probability that both parts are on the same side; plus through-hole
              keep-outs against parts on the opposite side
  contain     rounded-rectangle containment of every box corner incl. edge clearance (edge parts may
              overhang on their mouth side)
  keepout     overlap with user keep-out rectangles
  decap       soft-min distance from each decoupling cap's power pad to its IC power pads
  group       spread of each schematic sheet around its centroid
  compact     smooth-max bounding-box area of the movable parts
  bottom      number of parts on the bottom side
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import ROTS
from .kicad_io import Problem
from .rules import decap_pairs

AXIS = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}


@dataclass
class Outline:
    W: float
    H: float
    r: float = 0.5
    edge: float = 0.2      # box-to-edge clearance

    def as_dict(self, x0=0.0, y0=0.0):
        return dict(x0=x0, y0=y0, w=self.W, h=self.H, r=self.r)


class Ctx:
    """Static tensors for one problem (independent of outline size)."""

    def __init__(self, pr: Problem, cfg: dict | None = None, device="cpu", dtype=torch.float64):
        cfg = cfg or pr.rules
        self.pr, self.cfg, self.device, self.dtype = pr, cfg, device, dtype
        t = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        N = len(pr.parts)
        self.N = N
        self.wh0 = t([[p.w, p.h] for p in pr.parts])
        self.area = self.wh0[:, 0] * self.wh0[:, 1]
        self.fixed = [p.fixed for p in pr.parts]
        self.corner = t([p.corner_hole for p in pr.parts], torch.bool)
        self.edge = t([p.fixed == "edge" for p in pr.parts], torch.bool)
        self.movable = ~self.corner                     # edge parts move (along their edge)
        self.free = self.movable & ~self.edge           # fully free 2-D parts (incl. absolutely fixed parts, rotation frozen)
        self.allow_rot = t([p.allow_rotate for p in pr.parts], torch.bool)
        self.allow_bottom = t([p.allow_bottom for p in pr.parts], torch.bool)
        # edge parts: mouth axis / sign / overhang
        ax = np.zeros(N, dtype=np.int64); sg = np.zeros(N); oh = np.zeros(N)
        for i, p in enumerate(pr.parts):
            if p.fixed == "edge":
                a, s = AXIS[p.edge_mouth]; ax[i], sg[i], oh[i] = a, s, p.edge_overhang
        self.mouth_axis, self.mouth_sign, self.overhang = t(ax, torch.long), t(sg), t(oh)

        # pins
        pp, po, pn = pr.pin_arrays()
        self.pin_part, self.pin_off, self.pin_net = t(pp, torch.long), t(po), t(pn, torch.long)
        self.E = len(pr.nets)
        self.net_w = t([n["weight"] for n in pr.nets])
        self.R = t(ROTS)  # (4,2,2)

        # through-hole keep-outs (owner, box rel. owner centre in top-local frame)
        ko_owner, ko_box = [], []
        for i, p in enumerate(pr.parts):
            for b in p.bottom_hole_keepout:
                ko_owner.append(i); ko_box.append(b)
        self.ko_owner = t(ko_owner, torch.long) if ko_owner else torch.zeros(0, dtype=torch.long, device=device)
        self.ko_box = t(ko_box) if ko_box else torch.zeros(0, 4, dtype=dtype, device=device)

        # decoupling: flatten (pair, cap_pin_offset, cap_part, ic_part, ic_off)
        self.dc = None
        dps = decap_pairs(pr, cfg) if cfg else []
        if dps:
            cap_part, cap_off, cand_pair, ic_part, ic_off = [], [], [], [], []
            for pi_, (ci, ck, cands) in enumerate(dps):
                cap_part.append(ci); cap_off.append(pr.parts[ci].pad_off[ck])
                for (ii, ik) in cands:
                    cand_pair.append(pi_); ic_part.append(ii); ic_off.append(pr.parts[ii].pad_off[ik])
            self.dc = dict(cap_part=t(cap_part, torch.long), cap_off=t(cap_off),
                           pair=t(cand_pair, torch.long), ic_part=t(ic_part, torch.long),
                           ic_off=t(ic_off), n=len(dps))
        # sheets
        sheets = sorted(set(p.sheet for p in pr.parts))
        self.group = t([sheets.index(p.sheet) for p in pr.parts], torch.long)
        self.n_groups = len(sheets)

        # pair index (i<j) excluding pairs of two non-movable parts
        ii, jj = torch.triu_indices(N, N, 1, device=device)
        keep = self.movable[ii] | self.movable[jj]
        self.pi, self.pj = ii[keep], jj[keep]

        sides = cfg.get("sides", {}) if cfg else {}
        self.bottom_penalty = float(sides.get("bottom_penalty", 2.0))
        self.clear = float(cfg.get("clearance", {}).get("courtyard", 0.05)) if cfg else 0.05

    # ------------------------------------------------------------------ geometry helpers
    def dims(self, k):
        odd = (k % 2 == 1).unsqueeze(-1)
        return torch.where(odd, self.wh0.flip(-1), self.wh0)

    def xform(self, off, k, s):
        """off (...,2) top-local; k (...) long; s (...) float side prob -> world offsets."""
        R = self.R[k % 4]
        top = torch.einsum("...ij,...j->...i", R, off)
        bot = torch.einsum("...ij,...j->...i", R, off * off.new_tensor([1.0, -1.0]))
        s = s.unsqueeze(-1)
        return (1 - s) * top + s * bot

    def pins_world(self, c, k, s):
        p = self.pin_part
        return c[p] + self.xform(self.pin_off, k[p], s[p])

    def boxes(self, c, k):
        wh = self.dims(k)
        return c - wh / 2, c + wh / 2  # lo (x,y), hi (x,y)


# ---------------------------------------------------------------------- energy terms

def wawl(ctx: Ctx, c, k, s, gamma=1.0):
    q = ctx.pins_world(c, k, s)                       # q_{i,d}: absolute pin coords incl. offsets (P,2)
    net = ctx.pin_net
    E = ctx.E
    total = c.new_zeros(())
    # Eq. (18): WA_g(e,d) = sum q e^{q/g} / sum e^{q/g} - sum q e^{-q/g} / sum e^{-q/g};  WA-WL = sum_e w_e sum_d WA_g(e,d)
    for d in range(2):
        x = q[:, d]
        for sign in (1.0, -1.0):
            z = sign * x / gamma
            zmax = torch.full((E,), -1e30, dtype=x.dtype, device=x.device).scatter_reduce(0, net, z, "amax")
            e = torch.exp(z - zmax[net])
            se = torch.zeros(E, dtype=x.dtype, device=x.device).index_add(0, net, e)
            sxe = torch.zeros(E, dtype=x.dtype, device=x.device).index_add(0, net, x * e)
            total = total + sign * (ctx.net_w * sxe / se).sum()   # +max-like term / -min-like term, weighted by w_e
    return total


def hpwl(ctx: Ctx, c, k, s):
    q = ctx.pins_world(c, k, s)
    net = ctx.pin_net; E = ctx.E
    tot = c.new_zeros(())
    for d in range(2):
        x = q[:, d]
        mx = torch.full((E,), -1e30, dtype=x.dtype, device=x.device).scatter_reduce(0, net, x, "amax")
        mn = torch.full((E,), 1e30, dtype=x.dtype, device=x.device).scatter_reduce(0, net, x, "amin")
        tot = tot + (mx - mn).sum()
    return tot


def _sp(x, beta):
    return F.softplus(x, beta=beta)   # sp_b(x) = log(1 + exp(b x)) / b  (paper Eq. 20)


def overlap(ctx: Ctx, c, k, s, beta=20.0, clear=None):
    """smooth overlap area (mm^2) of keep-out boxes inflated by clearance/2 each."""
    clear = ctx.clear if clear is None else clear
    lo0, hi0 = ctx.boxes(c, k)
    lo = lo0 - clear / 2; hi = hi0 + clear / 2
    i, j = ctx.pi, ctx.pj
    # Eq. (20): A(Bi,Bj) = sp_b(min(r_i,r_j) - max(l_i,l_j)) * sp_b(min(q_i,q_j) - max(b_i,b_j))
    ox = _sp(torch.minimum(hi[i, 0], hi[j, 0]) - torch.maximum(lo[i, 0], lo[j, 0]), beta)
    oy = _sp(torch.minimum(hi[i, 1], hi[j, 1]) - torch.maximum(lo[i, 1], lo[j, 1]), beta)
    same = s[i] * s[j] + (1 - s[i]) * (1 - s[j])                   # PCB: only parts on the same side collide
    tot = (ox * oy * same).sum()
    if ctx.ko_owner.numel():
        o = ctx.ko_owner
        # keep-out boxes rotate with the owner (top side only)
        corners = torch.stack([ctx.ko_box[:, [0, 1]], ctx.ko_box[:, [2, 3]]], 1)  # (K,2,2)
        cw = ctx.xform(corners, k[o].unsqueeze(1).expand(-1, 2), torch.zeros_like(corners[..., 0]))
        klo = c[o] + torch.minimum(cw[:, 0], cw[:, 1]); khi = c[o] + torch.maximum(cw[:, 0], cw[:, 1])
        # against every other part, weighted by the probability that part is on the opposite side
        oside = s[o]  # owner side prob (0 for top owners)
        # keep-outs already carry their own margin: no courtyard clearance here
        ox = _sp(torch.minimum(khi[:, None, 0], hi0[None, :, 0]) - torch.maximum(klo[:, None, 0], lo0[None, :, 0]), beta)
        oy = _sp(torch.minimum(khi[:, None, 1], hi0[None, :, 1]) - torch.maximum(klo[:, None, 1], lo0[None, :, 1]), beta)
        opp = oside[:, None] * (1 - s[None, :]) + (1 - oside[:, None]) * s[None, :]
        notself = (o[:, None] != torch.arange(ctx.N, device=c.device)[None, :]).to(c.dtype)
        mov = (ctx.movable[o][:, None] | ctx.movable[None, :]).to(c.dtype)
        tot = tot + (ox * oy * opp * notself * mov).sum()
    return tot


def overlap_exact(ctx: Ctx, c, k, s, clear=None):
    """hard overlap area (sides rounded), for evaluation."""
    with torch.no_grad():
        sb = (s > 0.5).to(c.dtype)
        clear = ctx.clear if clear is None else clear
        lo, hi = ctx.boxes(c, k); lo = lo - clear / 2 + 1e-6; hi = hi + clear / 2 - 1e-6
        ii, jj = torch.triu_indices(ctx.N, ctx.N, 1, device=c.device)
        ox = (torch.minimum(hi[ii, 0], hi[jj, 0]) - torch.maximum(lo[ii, 0], lo[jj, 0])).clamp(min=0)
        oy = (torch.minimum(hi[ii, 1], hi[jj, 1]) - torch.maximum(lo[ii, 1], lo[jj, 1])).clamp(min=0)
        same = (sb[ii] == sb[jj]).to(c.dtype)
        a = ox * oy * same
        bad = [(int(ii[t]), int(jj[t]), float(a[t])) for t in torch.nonzero(a > 1e-9).flatten()]
        return float(a.sum()), bad


def _sdf_roundrect(p, W, H, r):
    """signed distance (negative inside) to the rounded rectangle [0,W]x[0,H] with corner radius r."""
    cx, cy = W / 2, H / 2
    qx = (p[..., 0] - cx).abs() - (W / 2 - r)
    qy = (p[..., 1] - cy).abs() - (H / 2 - r)
    outside = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-12)
    inside = torch.minimum(torch.maximum(qx, qy), qx.new_zeros(()))
    return outside + inside - r


def inboard_boxes(ctx: Ctx, c, k, extra=0.0):
    """keep-out boxes with edge-part overhang (+extra, the edge clearance the mouth side is exempt
    from) removed on the mouth side (world frame)."""
    lo, hi = ctx.boxes(c, k)
    if ctx.edge.any():
        # mouth direction in world after rotation k
        e = ctx.edge
        R = ctx.R[k[e] % 4]
        loc = torch.zeros(int(e.sum()), 2, dtype=c.dtype, device=c.device)
        loc[torch.arange(loc.shape[0]), ctx.mouth_axis[e]] = ctx.mouth_sign[e]
        d = torch.einsum("nij,nj->ni", R, loc)               # unit world dir of the mouth
        oh = ctx.overhang[e].unsqueeze(-1) + extra
        lo_e = lo[e] + (d < -0.5).to(c.dtype) * oh           # mouth facing -x/-y: raise lo
        hi_e = hi[e] - (d > 0.5).to(c.dtype) * oh
        lo = lo.clone(); hi = hi.clone()
        lo[e] = lo_e; hi[e] = hi_e
    return lo, hi


def contain(ctx: Ctx, c, k, ol: Outline, beta=20.0):
    lo, hi = inboard_boxes(ctx, c, k, ol.edge)
    m = ctx.movable
    corners = torch.stack([lo, torch.stack([hi[:, 0], lo[:, 1]], -1), hi, torch.stack([lo[:, 0], hi[:, 1]], -1)], 1)
    d = _sdf_roundrect(corners[m], ol.W, ol.H, ol.r) + ol.edge
    return (_sp(d, beta) ** 2).sum()


def contain_exact(ctx: Ctx, c, k, ol: Outline, tol=1e-4):
    with torch.no_grad():
        lo, hi = inboard_boxes(ctx, c, k, ol.edge)
        corners = torch.stack([lo, torch.stack([hi[:, 0], lo[:, 1]], -1), hi, torch.stack([lo[:, 0], hi[:, 1]], -1)], 1)
        d = _sdf_roundrect(corners, ol.W, ol.H, ol.r) + ol.edge
        worst = d.max(1).values
        bad = [(i, float(worst[i])) for i in range(ctx.N) if worst[i] > tol]
        return bad


def keepout(ctx: Ctx, c, k, zones, beta=20.0):
    if not zones:
        return c.new_zeros(())
    lo, hi = ctx.boxes(c, k)
    z = torch.as_tensor(zones, dtype=c.dtype, device=c.device)  # (Z,4) x0,y0,x1,y1
    ox = _sp(torch.minimum(hi[:, None, 0], z[None, :, 2]) - torch.maximum(lo[:, None, 0], z[None, :, 0]), beta)
    oy = _sp(torch.minimum(hi[:, None, 1], z[None, :, 3]) - torch.maximum(lo[:, None, 1], z[None, :, 1]), beta)
    return (ox * oy)[ctx.movable].sum()


def density(ctx: Ctx, c, k, s, ol: Outline, nbins=24, target=1.0):
    """sum over bins and sides of relu(occupied - target*binarea)^2 / binarea."""
    lo, hi = ctx.boxes(c, k)
    bx = torch.linspace(0, ol.W, nbins + 1, dtype=c.dtype, device=c.device)
    by = torch.linspace(0, ol.H, nbins + 1, dtype=c.dtype, device=c.device)
    ox = (torch.minimum(hi[:, None, 0], bx[None, 1:]) - torch.maximum(lo[:, None, 0], bx[None, :-1])).clamp(min=0)
    oy = (torch.minimum(hi[:, None, 1], by[None, 1:]) - torch.maximum(lo[:, None, 1], by[None, :-1])).clamp(min=0)
    ba = (ol.W / nbins) * (ol.H / nbins)
    tot = c.new_zeros(())
    both = ctx.corner.to(c.dtype)  # corner holes occupy both sides
    for side_w in ((1 - s) + both * s, s + both * (1 - s)):
        occ = torch.einsum("n,ni,nj->ij", side_w, ox, oy)
        tot = tot + (F.relu(occ - target * ba) ** 2).sum() / ba
    return tot


def decap(ctx: Ctx, c, k, s, tau=0.5):
    if ctx.dc is None:
        return c.new_zeros(())
    d = ctx.dc
    cp = c[d["cap_part"]] + ctx.xform(d["cap_off"], k[d["cap_part"]], s[d["cap_part"]])
    ip = c[d["ic_part"]] + ctx.xform(d["ic_off"], k[d["ic_part"]], s[d["ic_part"]])
    dist = torch.sqrt(((cp[d["pair"]] - ip) ** 2).sum(-1) + 1e-6)
    z = -dist / tau
    zmax = torch.full((d["n"],), -1e30, dtype=c.dtype, device=c.device).scatter_reduce(0, d["pair"], z, "amax")
    se = torch.zeros(d["n"], dtype=c.dtype, device=c.device).index_add(0, d["pair"], torch.exp(z - zmax[d["pair"]]))
    softmin = -tau * (torch.log(se) + zmax)
    return softmin.sum()


def group(ctx: Ctx, c):
    G = ctx.n_groups
    g = ctx.group
    n = torch.zeros(G, dtype=c.dtype, device=c.device).index_add(0, g, torch.ones_like(c[:, 0]))
    mu = torch.zeros(G, 2, dtype=c.dtype, device=c.device).index_add(0, g, c) / n[:, None]
    return ((c - mu[g]) ** 2).sum() / ctx.N


def compact(ctx: Ctx, c, k, tau=0.5):
    m = ctx.free
    lo, hi = ctx.boxes(c, k)
    lo, hi = lo[m], hi[m]
    xh = tau * torch.logsumexp(hi[:, 0] / tau, 0); xl = -tau * torch.logsumexp(-lo[:, 0] / tau, 0)
    yh = tau * torch.logsumexp(hi[:, 1] / tau, 0); yl = -tau * torch.logsumexp(-lo[:, 1] / tau, 0)
    return (xh - xl) * (yh - yl)


def bottom(ctx: Ctx, s):
    return s.sum()


TERMS = ("wawl", "density", "overlap", "contain", "keepout", "decap", "group", "compact", "bottom")


def all_terms(ctx: Ctx, c, k, s, ol: Outline, zones=(), gamma=1.0, beta=20.0, nbins=24):
    return dict(
        wawl=wawl(ctx, c, k, s, gamma),
        density=density(ctx, c, k, s, ol, nbins),
        overlap=overlap(ctx, c, k, s, beta),
        contain=contain(ctx, c, k, ol, beta),
        keepout=keepout(ctx, c, k, list(zones), beta),
        decap=decap(ctx, c, k, s),
        group=group(ctx, c),
        compact=compact(ctx, c, k),
        bottom=bottom(ctx, s),
    )


def legality_report(ctx: Ctx, c, k, s, ol: Outline):
    """exact check: returns dict(ok, overlaps, outside, hole_conflicts)."""
    c = torch.as_tensor(c, dtype=ctx.dtype); k = torch.as_tensor(k, dtype=torch.long)
    s = torch.as_tensor(s, dtype=ctx.dtype)
    ov_area, ov = overlap_exact(ctx, c, k, s)
    out = contain_exact(ctx, c, k, ol)
    hc = []
    if ctx.ko_owner.numel():
        with torch.no_grad():
            lo, hi = ctx.boxes(c, k)
            sb = s > 0.5
            for t_, o in enumerate(ctx.ko_owner.tolist()):
                b = ctx.ko_box[t_]
                cw = ctx.xform(torch.stack([b[[0, 1]], b[[2, 3]]]), k[o].repeat(2), torch.zeros(2, dtype=ctx.dtype))
                klo = c[o] + torch.minimum(cw[0], cw[1]); khi = c[o] + torch.maximum(cw[0], cw[1])
                for j in range(ctx.N):
                    if j == o or bool(sb[j]) == bool(sb[o]):
                        continue
                    ox = min(float(khi[0]), float(hi[j, 0])) - max(float(klo[0]), float(lo[j, 0]))
                    oy = min(float(khi[1]), float(hi[j, 1])) - max(float(klo[1]), float(lo[j, 1]))
                    if ox > 1e-6 and oy > 1e-6:
                        hc.append((o, j))
    return dict(ok=(not ov and not out and not hc), overlap_area=ov_area, overlaps=ov, outside=out, hole_conflicts=hc)
