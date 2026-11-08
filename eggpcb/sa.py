"""Simulated-annealing baseline on the *same* energy as energies.py (numba, incremental deltas).

Moves: shift, swap, rotate (90/270), flip side, slide an edge part along its edge, move an edge
part to another edge. Legality weights (overlap / containment / hole keep-outs) ramp up
geometrically over the schedule; the final state goes through the same greedy legalizer as the
diffusion placer.

tests/test_sa_energy.py checks that `full_energy` here equals energies.all_terms (torch) to 1e-9.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from .energies import Ctx, Outline
from .layout import EDGES, apply_constraints, corner_centers, edge_k_table

# term order shared with the python side
T_WL, T_DENS, T_OV, T_CONT, T_KO, T_DC, T_GRP, T_COMP, T_BOT = range(9)
TERM_NAMES = ("wawl", "density", "overlap", "contain", "keepout_holes", "decap", "group", "compact", "bottom")


# ---------------------------------------------------------------------------------------------
# primitives (must mirror energies.py exactly)
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def rot_off(ox, oy, k, s):
    if s == 1:
        oy = -oy
    k = k % 4
    if k == 0:
        return ox, oy
    elif k == 1:
        return oy, -ox
    elif k == 2:
        return -ox, -oy
    return -oy, ox


@njit(cache=True)
def softplus(x, beta):
    z = beta * x
    if z > 20.0:
        return x
    return math.log1p(math.exp(z)) / beta


@njit(cache=True)
def dims(wh0, i, k):
    if k % 2 == 1:
        return wh0[i, 1], wh0[i, 0]
    return wh0[i, 0], wh0[i, 1]


@njit(cache=True)
def sdf_rr(px, py, W, H, r):
    qx = abs(px - W / 2) - (W / 2 - r)
    qy = abs(py - H / 2) - (H / 2 - r)
    ox = qx if qx > 0 else 0.0
    oy = qy if qy > 0 else 0.0
    outside = math.sqrt(ox * ox + oy * oy + 1e-12)
    m = qx if qx > qy else qy
    inside = m if m < 0 else 0.0
    return outside + inside - r


@njit(cache=True)
def net_wawl(e, cx, cy, k, s, net_ptr, net_pins, pin_part, pin_off, gamma):
    a = net_ptr[e]; b = net_ptr[e + 1]
    tot = 0.0
    for d in range(2):
        mx = -1e300; mn = 1e300
        for t in range(a, b):
            p = net_pins[t]; i = pin_part[p]
            ox, oy = rot_off(pin_off[p, 0], pin_off[p, 1], k[i], s[i])
            v = cx[i] + ox if d == 0 else cy[i] + oy
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        sp = 0.0; sxp = 0.0; sn = 0.0; sxn = 0.0
        for t in range(a, b):
            p = net_pins[t]; i = pin_part[p]
            ox, oy = rot_off(pin_off[p, 0], pin_off[p, 1], k[i], s[i])
            v = cx[i] + ox if d == 0 else cy[i] + oy
            ep = math.exp((v - mx) / gamma); en = math.exp((mn - v) / gamma)
            sp += ep; sxp += v * ep; sn += en; sxn += v * en
        tot += sxp / sp - sxn / sn
    return tot


@njit(cache=True)
def pair_ov(i, j, cx, cy, k, s, wh0, ptype, clear, beta):
    if s[i] != s[j]:
        return 0.0
    if ptype[i] == 2 and ptype[j] == 2:
        return 0.0
    wi, hi = dims(wh0, i, k[i]); wj, hj = dims(wh0, j, k[j])
    ox = softplus(min(cx[i] + wi / 2, cx[j] + wj / 2) - max(cx[i] - wi / 2, cx[j] - wj / 2) + clear, beta)
    oy = softplus(min(cy[i] + hi / 2, cy[j] + hj / 2) - max(cy[i] - hi / 2, cy[j] - hj / 2) + clear, beta)
    return ox * oy


@njit(cache=True)
def ko_world(b, ko_owner, ko_box, cx, cy, k):
    o = ko_owner[b]
    x0, y0 = rot_off(ko_box[b, 0], ko_box[b, 1], k[o], 0)
    x1, y1 = rot_off(ko_box[b, 2], ko_box[b, 3], k[o], 0)
    return cx[o] + min(x0, x1), cy[o] + min(y0, y1), cx[o] + max(x0, x1), cy[o] + max(y0, y1)


@njit(cache=True)
def ko_pair(b, j, ko_owner, ko_box, cx, cy, k, s, wh0, ptype, beta):
    o = ko_owner[b]
    if j == o or s[j] == s[o]:
        return 0.0
    if ptype[o] == 2 and ptype[j] == 2:
        return 0.0
    lx, ly, hx, hy = ko_world(b, ko_owner, ko_box, cx, cy, k)
    wj, hj = dims(wh0, j, k[j])
    ox = softplus(min(hx, cx[j] + wj / 2) - max(lx, cx[j] - wj / 2), beta)
    oy = softplus(min(hy, cy[j] + hj / 2) - max(ly, cy[j] - hj / 2), beta)
    return ox * oy


@njit(cache=True)
def part_contain(i, cx, cy, k, wh0, ptype, maxis, msign, ohang, W, H, r, edge, beta):
    if ptype[i] == 2:
        return 0.0
    w, h = dims(wh0, i, k[i])
    lx = cx[i] - w / 2; hx = cx[i] + w / 2; ly = cy[i] - h / 2; hy = cy[i] + h / 2
    if ptype[i] == 1:
        ux, uy = (1.0, 0.0) if maxis[i] == 0 else (0.0, 1.0)
        ux *= msign[i]; uy *= msign[i]
        dx, dy = rot_off(ux, uy, k[i], 0)
        sh = ohang[i] + edge
        if dx < -0.5:
            lx += sh
        if dx > 0.5:
            hx -= sh
        if dy < -0.5:
            ly += sh
        if dy > 0.5:
            hy -= sh
    tot = 0.0
    for (px, py) in ((lx, ly), (hx, ly), (hx, hy), (lx, hy)):
        d = sdf_rr(px, py, W, H, r) + edge
        v = softplus(d, beta)
        tot += v * v
    return tot


@njit(cache=True)
def pair_decap(d, cx, cy, k, s, dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, tau):
    ci = dc_cap_part[d]
    ox, oy = rot_off(dc_cap_off[d, 0], dc_cap_off[d, 1], k[ci], s[ci])
    px = cx[ci] + ox; py = cy[ci] + oy
    zmax = -1e300
    for t in range(dc_ptr[d], dc_ptr[d + 1]):
        ii = dc_cand_part[t]
        qx, qy = rot_off(dc_cand_off[t, 0], dc_cand_off[t, 1], k[ii], s[ii])
        dist = math.sqrt((px - cx[ii] - qx) ** 2 + (py - cy[ii] - qy) ** 2 + 1e-6)
        z = -dist / tau
        if z > zmax:
            zmax = z
    se = 0.0
    for t in range(dc_ptr[d], dc_ptr[d + 1]):
        ii = dc_cand_part[t]
        qx, qy = rot_off(dc_cand_off[t, 0], dc_cand_off[t, 1], k[ii], s[ii])
        dist = math.sqrt((px - cx[ii] - qx) ** 2 + (py - cy[ii] - qy) ** 2 + 1e-6)
        se += math.exp(-dist / tau - zmax)
    return -tau * (math.log(se) + zmax)


@njit(cache=True)
def compact_val(cx, cy, k, wh0, ptype, tau):
    # free parts only (ptype 0)
    mxh = -1e300; mxl = -1e300; myh = -1e300; myl = -1e300
    n = cx.shape[0]
    for i in range(n):
        if ptype[i] != 0:
            continue
        w, h = dims(wh0, i, k[i])
        mxh = max(mxh, (cx[i] + w / 2) / tau); mxl = max(mxl, -(cx[i] - w / 2) / tau)
        myh = max(myh, (cy[i] + h / 2) / tau); myl = max(myl, -(cy[i] - h / 2) / tau)
    sxh = 0.0; sxl = 0.0; syh = 0.0; syl = 0.0
    for i in range(n):
        if ptype[i] != 0:
            continue
        w, h = dims(wh0, i, k[i])
        sxh += math.exp((cx[i] + w / 2) / tau - mxh); sxl += math.exp(-(cx[i] - w / 2) / tau - mxl)
        syh += math.exp((cy[i] + h / 2) / tau - myh); syl += math.exp(-(cy[i] - h / 2) / tau - myl)
    xh = tau * (math.log(sxh) + mxh); xl = -tau * (math.log(sxl) + mxl)
    yh = tau * (math.log(syh) + myh); yl = -tau * (math.log(syl) + myl)
    return (xh - xl) * (yh - yl)


@njit(cache=True)
def bin_range(lo, hi, L, nb):
    a = int(math.floor(lo / L * nb)); b = int(math.floor(hi / L * nb))
    return max(a, 0), min(b, nb - 1)


@njit(cache=True)
def density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, sign):
    w, h = dims(wh0, i, k[i])
    lx = cx[i] - w / 2; hx = cx[i] + w / 2; ly = cy[i] - h / 2; hy = cy[i] + h / 2
    bw = W / nb; bh = H / nb
    a0, a1 = bin_range(lx, hx, W, nb); b0, b1 = bin_range(ly, hy, H, nb)
    for a in range(a0, a1 + 1):
        ox = min(hx, (a + 1) * bw) - max(lx, a * bw)
        if ox <= 0:
            continue
        for b in range(b0, b1 + 1):
            oy = min(hy, (b + 1) * bh) - max(ly, b * bh)
            if oy <= 0:
                continue
            if ptype[i] == 2:
                occ[0, a, b] += sign * ox * oy; occ[1, a, b] += sign * ox * oy
            else:
                occ[s[i], a, b] += sign * ox * oy


@njit(cache=True)
def density_pen_range(occ, a0, a1, b0, b1, cap, ba):
    tot = 0.0
    for sd in range(2):
        for a in range(a0, a1 + 1):
            for b in range(b0, b1 + 1):
                v = occ[sd, a, b] - cap
                if v > 0:
                    tot += v * v / ba
    return tot


@njit(cache=True)
def part_bins(i, cx, cy, k, wh0, W, H, nb):
    w, h = dims(wh0, i, k[i])
    a0, a1 = bin_range(cx[i] - w / 2, cx[i] + w / 2, W, nb)
    b0, b1 = bin_range(cy[i] - h / 2, cy[i] + h / 2, H, nb)
    return a0, a1, b0, b1


# ---------------------------------------------------------------------------------------------
# full energy (for tests / bookkeeping)
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def full_terms(cx, cy, k, s, wh0, ptype, maxis, msign, ohang,
               net_ptr, net_pins, pin_part, pin_off, net_w,
               ko_owner, ko_box,
               dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off,
               grp, G, W, H, r, edge, nb, cap_frac,
               gamma, beta, clear, tau_dc, tau_comp):
    N = cx.shape[0]
    out = np.zeros(9)
    for e in range(net_ptr.shape[0] - 1):
        out[T_WL] += net_w[e] * net_wawl(e, cx, cy, k, s, net_ptr, net_pins, pin_part, pin_off, gamma)
    occ = np.zeros((2, nb, nb))
    for i in range(N):
        density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, 1.0)
    ba = (W / nb) * (H / nb)
    out[T_DENS] = density_pen_range(occ, 0, nb - 1, 0, nb - 1, cap_frac * ba, ba)
    for i in range(N):
        for j in range(i + 1, N):
            out[T_OV] += pair_ov(i, j, cx, cy, k, s, wh0, ptype, clear, beta)
        out[T_CONT] += part_contain(i, cx, cy, k, wh0, ptype, maxis, msign, ohang, W, H, r, edge, beta)
    for b in range(ko_owner.shape[0]):
        for j in range(N):
            out[T_KO] += ko_pair(b, j, ko_owner, ko_box, cx, cy, k, s, wh0, ptype, beta)
    for d in range(dc_cap_part.shape[0]):
        out[T_DC] += pair_decap(d, cx, cy, k, s, dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, tau_dc)
    # group
    S1 = np.zeros((G, 2)); S2 = np.zeros(G); n = np.zeros(G)
    for i in range(N):
        g = grp[i]; S1[g, 0] += cx[i]; S1[g, 1] += cy[i]; S2[g] += cx[i] ** 2 + cy[i] ** 2; n[g] += 1
    for g in range(G):
        if n[g] > 0:
            out[T_GRP] += S2[g] - (S1[g, 0] ** 2 + S1[g, 1] ** 2) / n[g]
    out[T_GRP] /= N
    out[T_COMP] = compact_val(cx, cy, k, wh0, ptype, tau_comp)
    for i in range(N):
        out[T_BOT] += s[i]
    return out


# ---------------------------------------------------------------------------------------------
# local energy of a moved set (pairs / per-part / per-net terms)
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def local_energy(M, m, cx, cy, k, s, wh0, ptype, maxis, msign, ohang,
                 net_ptr, net_pins, pin_part, pin_off, net_w, part_net_ptr, part_nets, net_mark, stamp,
                 ko_owner, ko_box, owner_ko_ptr, owner_ko,
                 dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, part_dc_ptr, part_dc, dc_mark,
                 W, H, r, edge, gamma, beta, clear, tau_dc, w):
    N = cx.shape[0]
    e_wl = 0.0; e_ov = 0.0; e_ct = 0.0; e_ko = 0.0; e_dc = 0.0
    for a in range(m):
        i = M[a]
        for t in range(part_net_ptr[i], part_net_ptr[i + 1]):
            e = part_nets[t]
            if net_mark[e] != stamp:
                net_mark[e] = stamp
                e_wl += net_w[e] * net_wawl(e, cx, cy, k, s, net_ptr, net_pins, pin_part, pin_off, gamma)
        for t in range(part_dc_ptr[i], part_dc_ptr[i + 1]):
            d = part_dc[t]
            if dc_mark[d] != stamp:
                dc_mark[d] = stamp
                e_dc += pair_decap(d, cx, cy, k, s, dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, tau_dc)
        for j in range(N):
            if j == i:
                continue
            inM = False
            for b in range(m):
                if M[b] == j:
                    inM = True
            if inM and j < i:
                continue
            e_ov += pair_ov(i, j, cx, cy, k, s, wh0, ptype, clear, beta)
        e_ct += part_contain(i, cx, cy, k, wh0, ptype, maxis, msign, ohang, W, H, r, edge, beta)
        # keep-outs owned by i, against everyone
        for t in range(owner_ko_ptr[i], owner_ko_ptr[i + 1]):
            b = owner_ko[t]
            for j in range(N):
                e_ko += ko_pair(b, j, ko_owner, ko_box, cx, cy, k, s, wh0, ptype, beta)
    # keep-outs owned by parts outside M, against parts in M
    for b in range(ko_owner.shape[0]):
        o = ko_owner[b]
        inM = False
        for a in range(m):
            if M[a] == o:
                inM = True
        if inM:
            continue
        for a in range(m):
            e_ko += ko_pair(b, M[a], ko_owner, ko_box, cx, cy, k, s, wh0, ptype, beta)
    return (w[T_WL] * e_wl + w[T_OV] * e_ov + w[T_CONT] * e_ct + w[T_KO] * e_ko + w[T_DC] * e_dc)


@njit(cache=True)
def group_val(S1, S2, n, g):
    if n[g] <= 0:
        return 0.0
    return S2[g] - (S1[g, 0] ** 2 + S1[g, 1] ** 2) / n[g]


@njit(cache=True)
def sa_core(cx, cy, k, s, ea, wh0, ptype, maxis, msign, ohang, kE, allow_rot, allow_bot,
            net_ptr, net_pins, pin_part, pin_off, net_w, part_net_ptr, part_nets,
            ko_owner, ko_box, owner_ko_ptr, owner_ko,
            dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, part_dc_ptr, part_dc,
            grp, G, W, H, r, edge, nb, cap_frac,
            gamma, beta, clear, tau_dc, tau_comp,
            w0, leg_start, leg_end, n_temps, moves_per_temp, T0, T_end, sig0, sig_min, seed):
    np.random.seed(seed)
    N = cx.shape[0]
    E = net_ptr.shape[0] - 1
    net_mark = np.zeros(E, dtype=np.int64); dc_mark = np.zeros(max(dc_cap_part.shape[0], 1), dtype=np.int64)
    stamp = 0
    occ = np.zeros((2, nb, nb))
    for i in range(N):
        density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, 1.0)
    ba = (W / nb) * (H / nb); cap = cap_frac * ba
    S1 = np.zeros((G, 2)); S2 = np.zeros(G); n = np.zeros(G)
    for i in range(N):
        g = grp[i]; S1[g, 0] += cx[i]; S1[g, 1] += cy[i]; S2[g] += cx[i] ** 2 + cy[i] ** 2; n[g] += 1
    comp = compact_val(cx, cy, k, wh0, ptype, tau_comp)

    free_idx = np.where(ptype == 0)[0]
    edge_idx = np.where(ptype == 1)[0]
    rot_idx = np.where((ptype == 0) & allow_rot)[0]
    bot_idx = np.where((ptype == 0) & allow_bot)[0]
    nf = free_idx.shape[0]
    w = w0.copy()
    alpha = (T_end / T0) ** (1.0 / max(n_temps - 1, 1))
    lr = (leg_end / leg_start) ** (1.0 / max(n_temps - 1, 1))
    M = np.zeros(2, dtype=np.int64)
    oc = np.zeros((2, 2)); ok_ = np.zeros(2, dtype=np.int64); os_ = np.zeros(2, dtype=np.int64)
    hist = np.zeros((n_temps, 3))
    T = T0; leg = leg_start
    L = max(W, H)
    sig = sig0
    for ts in range(n_temps):
        w[T_OV] = w0[T_OV] * leg; w[T_CONT] = w0[T_CONT] * leg; w[T_KO] = w0[T_KO] * leg
        acc = 0; tried = 0
        for it in range(moves_per_temp):
            u = np.random.random()
            mtype = 0
            if u < 0.62:
                mtype = 0  # shift
            elif u < 0.77:
                mtype = 1  # swap
            elif u < 0.87:
                mtype = 2  # rotate
            elif u < 0.93:
                mtype = 3  # flip
            elif u < 0.985:
                mtype = 4  # edge slide
            else:
                mtype = 5  # edge switch
            if mtype == 2 and rot_idx.shape[0] == 0:
                mtype = 0
            if mtype == 3 and bot_idx.shape[0] == 0:
                mtype = 0
            if (mtype == 4 or mtype == 5) and edge_idx.shape[0] == 0:
                mtype = 0
            m = 1
            if mtype == 0:
                M[0] = free_idx[np.random.randint(nf)]
            elif mtype == 1:
                M[0] = free_idx[np.random.randint(nf)]; M[1] = free_idx[np.random.randint(nf)]
                if M[0] == M[1]:
                    continue
                m = 2
            elif mtype == 2:
                M[0] = rot_idx[np.random.randint(rot_idx.shape[0])]
            elif mtype == 3:
                M[0] = bot_idx[np.random.randint(bot_idx.shape[0])]
            else:
                M[0] = edge_idx[np.random.randint(edge_idx.shape[0])]
            # save
            for a in range(m):
                i = M[a]; oc[a, 0] = cx[i]; oc[a, 1] = cy[i]; ok_[a] = k[i]; os_[a] = s[i]
            oea = ea[M[0]]
            stamp += 1
            e_old = local_energy(M, m, cx, cy, k, s, wh0, ptype, maxis, msign, ohang,
                                 net_ptr, net_pins, pin_part, pin_off, net_w, part_net_ptr, part_nets, net_mark, stamp,
                                 ko_owner, ko_box, owner_ko_ptr, owner_ko,
                                 dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, part_dc_ptr, part_dc, dc_mark,
                                 W, H, r, edge, gamma, beta, clear, tau_dc, w)
            # density / group before
            dpen = 0.0
            ga = 0.0
            for a in range(m):
                i = M[a]
                ga += group_val(S1, S2, n, grp[i])
            if m == 2 and grp[M[0]] == grp[M[1]]:
                ga = group_val(S1, S2, n, grp[M[0]])
            # ranges before
            ra = np.zeros((m, 4), dtype=np.int64)
            for a in range(m):
                a0, a1, b0, b1 = part_bins(M[a], cx, cy, k, wh0, W, H, nb)
                ra[a, 0] = a0; ra[a, 1] = a1; ra[a, 2] = b0; ra[a, 3] = b1
            # apply move
            if mtype == 0:
                i = M[0]
                cx[i] += np.random.normal() * sig; cy[i] += np.random.normal() * sig
                cx[i] = min(max(cx[i], 0.0), W); cy[i] = min(max(cy[i], 0.0), H)
            elif mtype == 1:
                i = M[0]; j = M[1]
                tx = cx[i]; ty = cy[i]; cx[i] = cx[j]; cy[i] = cy[j]; cx[j] = tx; cy[j] = ty
            elif mtype == 2:
                i = M[0]; k[i] = (k[i] + (1 if np.random.random() < 0.5 else 3)) % 4
            elif mtype == 3:
                i = M[0]; s[i] = 1 - s[i]
            else:
                i = M[0]
                e_new = ea[i]
                if mtype == 5:
                    e_new = (ea[i] + 1 + np.random.randint(3)) % 4
                kk = kE[i, e_new]
                ww, hh = dims(wh0, i, kk)
                if e_new == 0 or e_new == 1:
                    along = cy[i] if mtype == 4 else np.random.random() * H
                    if mtype == 4:
                        along += np.random.normal() * sig
                    along = min(max(along, hh / 2), H - hh / 2)
                    cy[i] = along
                    cx[i] = (-ohang[i] + ww / 2) if e_new == 0 else (W + ohang[i] - ww / 2)
                else:
                    along = cx[i] if mtype == 4 else np.random.random() * W
                    if mtype == 4:
                        along += np.random.normal() * sig
                    along = min(max(along, ww / 2), W - ww / 2)
                    cx[i] = along
                    cy[i] = (-ohang[i] + hh / 2) if e_new == 2 else (H + ohang[i] - hh / 2)
                k[i] = kk; ea[i] = e_new
            # density delta (sequential per part)
            for a in range(m):
                i = M[a]
                a0, a1, b0, b1 = part_bins(i, cx, cy, k, wh0, W, H, nb)
                A0 = min(a0, ra[a, 0]); A1 = max(a1, ra[a, 1]); B0 = min(b0, ra[a, 2]); B1 = max(b1, ra[a, 3])
                before = density_pen_range(occ, A0, A1, B0, B1, cap, ba)
                # remove old contribution
                nx = cx[i]; ny = cy[i]; nk = k[i]; ns = s[i]
                cx[i] = oc[a, 0]; cy[i] = oc[a, 1]; k[i] = ok_[a]; s[i] = os_[a]
                density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, -1.0)
                cx[i] = nx; cy[i] = ny; k[i] = nk; s[i] = ns
                density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, 1.0)
                after = density_pen_range(occ, A0, A1, B0, B1, cap, ba)
                dpen += after - before
            # group after
            for a in range(m):
                i = M[a]; g = grp[i]
                S1[g, 0] += cx[i] - oc[a, 0]; S1[g, 1] += cy[i] - oc[a, 1]
                S2[g] += cx[i] ** 2 + cy[i] ** 2 - oc[a, 0] ** 2 - oc[a, 1] ** 2
            gb = 0.0
            for a in range(m):
                gb += group_val(S1, S2, n, grp[M[a]])
            if m == 2 and grp[M[0]] == grp[M[1]]:
                gb = group_val(S1, S2, n, grp[M[0]])
            new_comp = compact_val(cx, cy, k, wh0, ptype, tau_comp)
            stamp += 1
            e_new_ = local_energy(M, m, cx, cy, k, s, wh0, ptype, maxis, msign, ohang,
                                  net_ptr, net_pins, pin_part, pin_off, net_w, part_net_ptr, part_nets, net_mark, stamp,
                                  ko_owner, ko_box, owner_ko_ptr, owner_ko,
                                  dc_cap_part, dc_cap_off, dc_ptr, dc_cand_part, dc_cand_off, part_dc_ptr, part_dc, dc_mark,
                                  W, H, r, edge, gamma, beta, clear, tau_dc, w)
            dbot = 0.0
            for a in range(m):
                dbot += s[M[a]] - os_[a]
            dE = (e_new_ - e_old) + w[T_DENS] * dpen + w[T_GRP] * (gb - ga) / N + w[T_COMP] * (new_comp - comp) + w[T_BOT] * dbot
            tried += 1
            if dE <= 0 or np.random.random() < math.exp(-dE / T):
                acc += 1
                comp = new_comp
            else:
                # revert density, group, state
                for a in range(m):
                    i = M[a]
                    density_add(occ, i, cx, cy, k, s, wh0, ptype, W, H, nb, -1.0)
                for a in range(m):
                    i = M[a]; g = grp[i]
                    S1[g, 0] -= cx[i] - oc[a, 0]; S1[g, 1] -= cy[i] - oc[a, 1]
                    S2[g] -= cx[i] ** 2 + cy[i] ** 2 - oc[a, 0] ** 2 - oc[a, 1] ** 2
                for a in range(m):
                    i = M[a]; cx[i] = oc[a, 0]; cy[i] = oc[a, 1]; k[i] = ok_[a]; s[i] = os_[a]
                ea[M[0]] = oea
                for a in range(m):
                    density_add(occ, M[a], cx, cy, k, s, wh0, ptype, W, H, nb, 1.0)
        rate = acc / max(tried, 1)
        # adaptive step: aim for ~0.3 acceptance on shifts
        if rate > 0.35:
            sig = min(sig * 1.08, L / 2)
        elif rate < 0.25:
            sig = max(sig / 1.08, sig_min)
        hist[ts, 0] = T; hist[ts, 1] = rate; hist[ts, 2] = sig
        T *= alpha; leg *= lr
    return hist


# ---------------------------------------------------------------------------------------------
# python side
# ---------------------------------------------------------------------------------------------
DEFAULT_W = dict(wawl=1.0, density=0.5, overlap=1.0, contain=1.0, keepout_holes=1.0, decap=2.0,
                 group=0.02, compact=0.02, bottom=10.0)


class SAArrays:
    """numpy views of a Ctx for the numba kernels."""

    def __init__(self, ctx: Ctx):
        pr = ctx.pr
        N = ctx.N
        self.N = N
        self.wh0 = ctx.wh0.cpu().numpy().astype(np.float64)
        self.ptype = np.array([2 if p.corner_hole else (1 if p.fixed == "edge" else 0) for p in pr.parts], dtype=np.int64)
        self.maxis = ctx.mouth_axis.cpu().numpy().astype(np.int64)
        self.msign = ctx.mouth_sign.cpu().numpy().astype(np.float64)
        self.ohang = ctx.overhang.cpu().numpy().astype(np.float64)
        self.kE = edge_k_table(pr)
        self.allow_rot = ctx.allow_rot.cpu().numpy()
        self.allow_bot = ctx.allow_bottom.cpu().numpy()
        pin_part = ctx.pin_part.cpu().numpy(); pin_net = ctx.pin_net.cpu().numpy()
        self.pin_part = pin_part.astype(np.int64)
        self.pin_off = ctx.pin_off.cpu().numpy().astype(np.float64)
        order = np.argsort(pin_net, kind="stable")
        self.net_pins = order.astype(np.int64)
        self.net_ptr = np.concatenate([[0], np.cumsum(np.bincount(pin_net, minlength=ctx.E))]).astype(np.int64)
        self.net_w = ctx.net_w.cpu().numpy().astype(np.float64)
        pn = [sorted(set(pin_net[pin_part == i].tolist())) for i in range(N)]
        self.part_net_ptr = np.concatenate([[0], np.cumsum([len(x) for x in pn])]).astype(np.int64)
        self.part_nets = np.array([e for x in pn for e in x], dtype=np.int64)
        self.ko_owner = ctx.ko_owner.cpu().numpy().astype(np.int64)
        self.ko_box = ctx.ko_box.cpu().numpy().astype(np.float64).reshape(-1, 4)
        ok = [np.where(self.ko_owner == i)[0].tolist() for i in range(N)]
        self.owner_ko_ptr = np.concatenate([[0], np.cumsum([len(x) for x in ok])]).astype(np.int64)
        self.owner_ko = np.array([b for x in ok for b in x], dtype=np.int64)
        if ctx.dc is not None:
            d = ctx.dc
            self.dc_cap_part = d["cap_part"].cpu().numpy().astype(np.int64)
            self.dc_cap_off = d["cap_off"].cpu().numpy().astype(np.float64)
            pair = d["pair"].cpu().numpy()
            self.dc_ptr = np.concatenate([[0], np.cumsum(np.bincount(pair, minlength=d["n"]))]).astype(np.int64)
            self.dc_cand_part = d["ic_part"].cpu().numpy().astype(np.int64)
            self.dc_cand_off = d["ic_off"].cpu().numpy().astype(np.float64)
            pd = [[] for _ in range(N)]
            for dd in range(d["n"]):
                parts = {int(self.dc_cap_part[dd])} | set(self.dc_cand_part[self.dc_ptr[dd]:self.dc_ptr[dd + 1]].tolist())
                for q in parts:
                    pd[q].append(dd)
        else:
            self.dc_cap_part = np.zeros(0, np.int64); self.dc_cap_off = np.zeros((0, 2))
            self.dc_ptr = np.zeros(1, np.int64); self.dc_cand_part = np.zeros(0, np.int64); self.dc_cand_off = np.zeros((0, 2))
            pd = [[] for _ in range(N)]
        self.part_dc_ptr = np.concatenate([[0], np.cumsum([len(x) for x in pd])]).astype(np.int64)
        self.part_dc = np.array([q for x in pd for q in x], dtype=np.int64)
        self.grp = ctx.group.cpu().numpy().astype(np.int64)
        self.G = int(ctx.n_groups)
        self.clear = ctx.clear


def weights_vec(wd):
    return np.array([wd["wawl"], wd["density"], wd["overlap"], wd["contain"], wd["keepout_holes"],
                     wd["decap"], wd["group"], wd["compact"], wd["bottom"]], dtype=np.float64)


def energy_terms_np(A: SAArrays, c, k, s, ol: Outline, nb=24, cap_frac=1.0, gamma=1.0, beta=20.0,
                    tau_dc=0.5, tau_comp=0.5):
    c = np.asarray(c, dtype=np.float64)
    return full_terms(c[:, 0].copy(), c[:, 1].copy(), np.asarray(k, np.int64), np.asarray(s, np.int64),
                      A.wh0, A.ptype, A.maxis, A.msign, A.ohang,
                      A.net_ptr, A.net_pins, A.pin_part, A.pin_off, A.net_w,
                      A.ko_owner, A.ko_box, A.dc_cap_part, A.dc_cap_off, A.dc_ptr, A.dc_cand_part, A.dc_cand_off,
                      A.grp, A.G, ol.W, ol.H, ol.r, ol.edge, nb, cap_frac, gamma, beta, A.clear, tau_dc, tau_comp)


def random_init(ctx: Ctx, ol: Outline, rng):
    pr = ctx.pr
    N = ctx.N
    c = np.stack([rng.uniform(0.15, 0.85, N) * ol.W, rng.uniform(0.15, 0.85, N) * ol.H], 1)
    k = np.array([rng.integers(4) if p.allow_rotate and p.fixed == "none" else int(round(p.rot0 / 90)) % 4 for p in pr.parts])
    s = np.zeros(N)
    ea = {i: int(rng.integers(4)) for i, p in enumerate(pr.parts) if p.fixed == "edge"}
    return apply_constraints(pr, c, k, s, ol, ea)


def run_sa(ctx: Ctx, ol: Outline, seed=0, weights=None, init=None, n_temps=300, moves_per_part=40,
           leg_start=0.3, leg_end=300.0, T0=None, T_end_frac=1e-4, nb=24, cap_frac=1.0, gamma=1.0, beta=20.0,
           verbose=True):
    A = SAArrays(ctx)
    wd = dict(DEFAULT_W); wd.update(weights or {})
    w0 = weights_vec(wd)
    rng = np.random.default_rng(seed)
    if init is None:
        c, k, s, ea = random_init(ctx, ol, rng)
    else:
        c, k, s, ea = apply_constraints(ctx.pr, *init[:3], ol, init[3] if len(init) > 3 else None)
    cx, cy = c[:, 0].copy(), c[:, 1].copy()
    k = k.astype(np.int64).copy(); s = (np.asarray(s) > 0.5).astype(np.int64)
    ea_arr = -np.ones(ctx.N, dtype=np.int64)
    for i, e in ea.items():
        ea_arr[i] = e
    if T0 is None:
        # scale temperature to the wirelength of the initial state
        t = energy_terms_np(A, np.stack([cx, cy], 1), k, s, ol, nb, cap_frac, gamma, beta)
        T0 = 0.02 * t[T_WL] / max(1, A.N) * 10
    n_free = int((A.ptype == 0).sum())
    t0 = time.time()
    hist = sa_core(cx, cy, k, s, ea_arr, A.wh0, A.ptype, A.maxis, A.msign, A.ohang, A.kE, A.allow_rot, A.allow_bot,
                   A.net_ptr, A.net_pins, A.pin_part, A.pin_off, A.net_w, A.part_net_ptr, A.part_nets,
                   A.ko_owner, A.ko_box, A.owner_ko_ptr, A.owner_ko,
                   A.dc_cap_part, A.dc_cap_off, A.dc_ptr, A.dc_cand_part, A.dc_cand_off, A.part_dc_ptr, A.part_dc,
                   A.grp, A.G, ol.W, ol.H, ol.r, ol.edge, nb, cap_frac,
                   gamma, beta, A.clear, 0.5, 0.5,
                   w0, leg_start, leg_end, n_temps, moves_per_part * n_free, T0, T0 * T_end_frac,
                   max(ol.W, ol.H) / 4, 0.05, seed)
    dt = time.time() - t0
    c = np.stack([cx, cy], 1)
    ea = {int(i): int(ea_arr[i]) for i in np.where(ea_arr >= 0)[0]}
    if verbose:
        print(f"SA seed={seed} {n_temps} temps x {moves_per_part * n_free} moves in {dt:.1f}s, final accept {hist[-1, 1]:.3f}")
    return c, k, s.astype(float), ea, dict(time=dt, hist=hist, T0=T0)
