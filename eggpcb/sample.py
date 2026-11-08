"""Energy-guided DDIM sampling (EGGPlacer Eq. 4-8, App. D) for PCB placement.

At every reverse step t:
    eps   = eps_theta(z_t, c, t)
    z0^   = (z_t - sqrt(1-ab_t) eps) / sqrt(ab_t)                          (Eq. 4)
    u(0)  = sg[z0^];  u(K) = Adam_K(u(0); E_EG)   K=10, lr=1e-2              (Eq. 6)
    g     = u(K) - u(0)                                                      (Eq. 7)
    z_t-1 = sqrt(ab_t-1) (z0^ + g) + sqrt(1-ab_t-1) eps                       (Eq. 8, DDIM eta=0)
E_EG = sum_soft lambda_term(t) E_term + lambda_leg(t) E_leg, where
  soft terms (WA-WL, density, compactness, decap, sheet grouping, bottom count) are each RMS-balanced
  against the legality gradient: lambda = clip(r * G_ref / (G_term + eps)), G_ref = max(lambda_leg,
  floor) * RMS(grad E_leg), EMA 0.9  (App. D Eq. 12);
  lambda_leg is a projected dual variable for E_leg <= 1e-4 (normalised area units), Adam ascent
  (lr 5e-4) while t < 100, capped at 15.
PCB additions: edge parts are projected onto their assigned edge after every inner step, corner holes
are fixed (and given to the model as fixed nodes); rotations, sides and edge assignments are chosen by
a discrete local search at a few checkpoints (DECISIONS.md), after which the conditioning is rebuilt.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .energies import Ctx, Outline, all_terms
from .features import build_cond, denormalize, normalize
from .layout import apply_constraints, corner_centers, edge_place, nearest_edge
from .sa import SAArrays, local_energy, weights_vec, DEFAULT_W

SOFT = ("wawl", "density", "compact", "decap", "group", "bottom")
DEFAULT_GUIDE = dict(
    K=10, lr=1e-2,
    r=dict(wawl=0.4, density=0.2, compact=0.2, decap=0.2, group=0.05, bottom=0.05),
    lam_min=1e-8, lam_max=1e3, ema=0.9,
    leg_target=1e-4, leg_lr=5e-4, leg_max=15.0, leg_floor=0.1, leg_start_t=100,
    gamma=1.0, beta=20.0,
    checkpoints=(150, 100, 60, 30, 10),
    terms=("wawl", "density", "compact", "decap", "group", "bottom"),
)


def _rms(g):
    return float(torch.sqrt((g ** 2).mean() + 1e-30))


class Guide:
    def __init__(self, ctx: Ctx, ol: Outline, cfg=None):
        self.ctx, self.ol = ctx, ol
        self.cfg = dict(DEFAULT_GUIDE); self.cfg.update(cfg or {})
        self.lam = {k: None for k in SOFT}
        self.lam_leg = 0.0
        self.dual_m = 0.0; self.dual_v = 0.0; self.dual_t = 0
        self.scale = torch.tensor([ol.W / 2, ol.H / 2], dtype=ctx.dtype)
        self.area_norm = 4.0 / (ol.W * ol.H)  # mm^2 -> normalised units

    def to_mm(self, u):
        return (u + 1.0) * self.scale

    def energies(self, u, k, s):
        T = all_terms(self.ctx, self.to_mm(u), k, s, self.ol, gamma=self.cfg["gamma"], beta=self.cfg["beta"])
        leg = (T["overlap"] + T["contain"] + T["keepout"]) * self.area_norm
        return T, leg

    def refine(self, z0, k, s, t, project):
        """K Adam steps on E_EG from u(0) = sg[z0]; returns u(K) (normalised)."""
        c = self.cfg
        u = z0.detach().clone().requires_grad_(True)                  # Eq. (6): u(0) = sg[z0^]
        opt = torch.optim.Adam([u], lr=c["lr"], foreach=False)
        mov = self.ctx.movable.to(u.dtype)[:, None]
        active = [n for n in c["terms"] if n in SOFT]
        last_leg = None
        for it in range(c["K"]):
            T, leg = self.energies(u, k, s)
            if it == 0:  # detached RMS-balanced targets, once per reverse step (EMA across steps)
                g_leg = torch.autograd.grad(leg, u, retain_graph=True)[0] * mov
                G_ref = max(self.lam_leg, c["leg_floor"]) * _rms(g_leg)   # Eq. (21): G_ref
                for n in active:
                    if not T[n].requires_grad:
                        continue
                    g = torch.autograd.grad(T[n], u, retain_graph=True)[0] * mov
                    tar = min(max(c["r"].get(n, 0.0) * G_ref / (_rms(g) + 1e-12), c["lam_min"]), c["lam_max"])  # Eq. (22)
                    self.lam[n] = tar if self.lam[n] is None else c["ema"] * self.lam[n] + (1 - c["ema"]) * tar  # EMA 0.9
            E = self.lam_leg * leg                                        # Eq. (4): E_EG = lam_leg E_leg + sum lam_n E_n
            for n in active:
                if self.lam[n] is not None and T[n].requires_grad:
                    E = E + self.lam[n] * T[n]
            opt.zero_grad()
            E.backward()
            u.grad *= mov
            opt.step()
            with torch.no_grad():
                project(u)
            last_leg = float(leg)
        # App. D: lambda_leg = projected dual variable for E_leg <= 1e-4 (Adam ascent, lr 5e-4, t < 100, cap 15)
        if t < c["leg_start_t"] and last_leg is not None:
            gdual = last_leg - c["leg_target"]
            self.dual_t += 1
            self.dual_m = 0.9 * self.dual_m + 0.1 * gdual
            self.dual_v = 0.999 * self.dual_v + 0.001 * gdual ** 2
            mh = self.dual_m / (1 - 0.9 ** self.dual_t); vh = self.dual_v / (1 - 0.999 ** self.dual_t)
            self.lam_leg = float(min(max(self.lam_leg + c["leg_lr"] * mh / (math.sqrt(vh) + 1e-8), 0.0), c["leg_max"]))
        return u.detach(), last_leg


class DiscreteSearch:
    """rotation / side / edge choice by local energy (numba kernels shared with SA)."""

    def __init__(self, ctx: Ctx, ol: Outline, weights=None, leg_weight=50.0):
        self.ctx, self.ol = ctx, ol
        self.A = SAArrays(ctx)
        w = dict(DEFAULT_W); w.update(weights or {})
        self.w = weights_vec(w)
        self.w[2] *= leg_weight; self.w[3] *= leg_weight; self.w[4] *= leg_weight
        self.bottom_w = w["bottom"]
        E = len(ctx.pr.nets)
        self.net_mark = np.zeros(E, np.int64); self.dc_mark = np.zeros(max(1, len(self.A.dc_cap_part)), np.int64)
        self.stamp = 0

    def local(self, i, cx, cy, k, s):
        A = self.A; ol = self.ol
        self.stamp += 1
        M = np.array([i, 0], np.int64)
        return local_energy(M, 1, cx, cy, k, s, A.wh0, A.ptype, A.maxis, A.msign, A.ohang,
                            A.net_ptr, A.net_pins, A.pin_part, A.pin_off, A.net_w, A.part_net_ptr, A.part_nets,
                            self.net_mark, self.stamp, A.ko_owner, A.ko_box, A.owner_ko_ptr, A.owner_ko,
                            A.dc_cap_part, A.dc_cap_off, A.dc_ptr, A.dc_cand_part, A.dc_cand_off, A.part_dc_ptr,
                            A.part_dc, self.dc_mark, ol.W, ol.H, ol.r, ol.edge, 1.0, 20.0, A.clear, 0.5, self.w) \
            + self.bottom_w * s[i]

    def run(self, c, k, s, ea, rng, sweeps=1):
        pr = self.ctx.pr
        cx, cy = c[:, 0].copy(), c[:, 1].copy()
        k = k.astype(np.int64).copy(); s = (np.asarray(s) > 0.5).astype(np.int64)
        order = rng.permutation(len(pr.parts))
        changed = 0
        for _ in range(sweeps):
            for i in order:
                p = pr.parts[i]
                if p.corner_hole:
                    continue
                if p.fixed == "edge":
                    best = None
                    for e in range(4):
                        along = cy[i] if e in (0, 1) else cx[i]
                        (x, y), kk = edge_place(p, e, along, self.ol)
                        ox, oy, okk = cx[i], cy[i], k[i]
                        cx[i], cy[i], k[i] = x, y, kk
                        val = self.local(i, cx, cy, k, s)
                        if best is None or val < best[0]:
                            best = (val, x, y, kk, e)
                        cx[i], cy[i], k[i] = ox, oy, okk
                    _, cx[i], cy[i], k[i], e = best
                    changed += int(ea.get(i) != e); ea[i] = e
                    continue
                opts = []
                rots = range(4) if p.allow_rotate else [k[i]]
                sides = (0, 1) if p.allow_bottom else (s[i],)
                for sd in sides:
                    for r in rots:
                        k0, s0 = k[i], s[i]
                        k[i], s[i] = r, sd
                        opts.append((self.local(i, cx, cy, k, s), r, sd))
                        k[i], s[i] = k0, s0
                val, r, sd = min(opts)
                changed += int(r != k[i] or sd != s[i])
                k[i], s[i] = r, sd
        return np.stack([cx, cy], 1), k, s.astype(float), ea, changed


@torch.no_grad()
def _model_eps(model, z, cond, t, dev):
    return model(z.unsqueeze(0).float().to(dev), cond, torch.tensor([t], device=dev)).squeeze(0).double().cpu()


def sample(model, pr, ol: Outline, seed=0, guided=True, guide_cfg=None, device="cpu", init=None, verbose=False,
           discrete=True):
    """returns dict(c, k, s, ea, stats). init: optional (k, s) starting rotations/sides."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ctx = Ctx(pr, dtype=torch.float64)
    N = ctx.N
    T = model.T
    ab = model.alpha_bar.double().cpu()
    k = np.array([int(round(p.rot0 / 90)) % 4 for p in pr.parts]) if init is None else np.asarray(init[0]).copy()
    s = np.zeros(N) if init is None else np.asarray(init[1], float).copy()
    corners = corner_centers(pr, ol)
    zc = {i: normalize(np.array(xy), ol.W, ol.H) for i, xy in corners.items()}
    edge_idx = [i for i, p in enumerate(pr.parts) if p.fixed == "edge"]
    ea = {}
    guide = Guide(ctx, ol, guide_cfg) if guided else None
    search = DiscreteSearch(ctx, ol) if discrete else None
    cfg = guide.cfg if guide else DEFAULT_GUIDE

    def project(u):
        for i, z in zc.items():
            u[i, 0], u[i, 1] = float(z[0]), float(z[1])
        for i in edge_idx:
            if i not in ea:
                continue
            cmm = denormalize(u[i].numpy(), ol.W, ol.H)
            e = ea[i]
            along = cmm[1] if e in (0, 1) else cmm[0]
            (x, y), _ = edge_place(pr.parts[i], e, along, ol)
            zz = normalize(np.array([x, y]), ol.W, ol.H)
            u[i, 0], u[i, 1] = float(zz[0]), float(zz[1])

    cond = build_cond(pr, k, s, ol.W, ol.H, device=device)
    z = torch.randn(N, 2, dtype=torch.float64)
    for i, zz in zc.items():
        z[i] = torch.tensor(zz)
    stats = dict(leg=[], lam_leg=[], changes=[])
    for t in range(T, 0, -1):
        eps = _model_eps(model, z, cond, t, device)
        a_t = ab[t - 1]; a_prev = ab[t - 2] if t > 1 else torch.tensor(1.0, dtype=torch.float64)
        z0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()             # Eq. (3): predicted clean placement
        z0 = z0.clamp(-1.5, 1.5)
        for i, zz in zc.items():
            z0[i] = torch.tensor(zz)
        if discrete and (t in cfg["checkpoints"]):
            cmm = denormalize(z0.numpy(), ol.W, ol.H)
            if not ea:
                for i in edge_idx:
                    ea[i] = nearest_edge(cmm[i], ol)
            cmm, k, s, ea = apply_constraints(pr, cmm, k, s, ol, ea)[:4]
            cmm, k, s, ea, ch = search.run(cmm, k, s, ea, rng)
            z0 = torch.tensor(normalize(cmm, ol.W, ol.H))
            cond = build_cond(pr, k, s, ol.W, ol.H, device=device)
            stats["changes"].append((t, ch))
        if guided:
            if not ea:
                cmm = denormalize(z0.numpy(), ol.W, ol.H)
                for i in edge_idx:
                    ea[i] = nearest_edge(cmm[i], ol)
            u, leg = guide.refine(z0, torch.tensor(k), torch.tensor(s), t, project)  # Eq. (6): u(K) = Adam_K(sg[z0]; E_EG)
            g = u - z0                                                                # Eq. (7): g_t = u(K) - u(0)
            stats["leg"].append(leg); stats["lam_leg"].append(guide.lam_leg)
        else:
            g = torch.zeros_like(z0)
        z = a_prev.sqrt() * (z0 + g) + (1 - a_prev).sqrt() * eps   # Eq. (8): guided DDIM step (eta = 0)
        if t == 1:
            z = z0 + g
    cmm = denormalize(z.numpy(), ol.W, ol.H)
    if not ea:
        for i in edge_idx:
            ea[i] = nearest_edge(cmm[i], ol)
    cmm, k, s, ea = apply_constraints(pr, cmm, k, s, ol, ea)[:4]
    if discrete:
        cmm, k, s, ea, ch = search.run(cmm, k, s, ea, rng)
    stats["seconds"] = time.time() - t0
    stats["lambdas"] = dict(guide.lam) if guided else {}
    return dict(c=cmm, k=k, s=s, ea=ea, stats=stats)
