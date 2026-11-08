"""Problem (+ rotation/side) -> model conditioning (GraphCond) and coordinate normalisation.

Coordinates are normalised per axis to the outline: z = 2 c / (W, H) - 1 in [-1, 1].
Node features: [2w/W, 2h/H, side, edge-part flag, log1p(#pins)/4] with w,h after rotation.
Edges: every net with 2..MAX_FANOUT pins becomes a star from the pin on the part with the most pins
(both directions); edge features = world pin offsets (rotation + mirroring applied) of source and
sink, normalised like coordinates. Higher fan-out nets (GND and the big rails) carry no positional
information and would collapse the graph, so they are left to the guidance energy.
Fixed nodes (is_fixed): corner-pinned mounting holes (their position is known from the outline).
"""
from __future__ import annotations

import numpy as np
import torch

from .geometry import box_dims, dims_rot, transform_offsets
from .model import GraphCond

MAX_FANOUT = 16


def normalize(c, W, H):
    return 2.0 * np.asarray(c) / np.array([W, H]) - 1.0


def denormalize(z, W, H):
    return (np.asarray(z) + 1.0) / 2.0 * np.array([W, H])


def build_cond(pr, k, s, W, H, device="cpu", dtype=torch.float32, flip=(1, 1), swap=False):
    """flip/swap: dihedral augmentation applied consistently to offsets (positions handled by caller)."""
    N = len(pr.parts)
    k = np.asarray(k); s = (np.asarray(s) > 0.5).astype(int)
    wh = dims_rot(box_dims(pr), k)
    scale = np.array([2.0 / W, 2.0 / H])
    npins = np.array([sum(1 for n in p.pad_net if n >= 0) for p in pr.parts])
    feats = np.stack([wh[:, 0] * scale[0], wh[:, 1] * scale[1], s.astype(float),
                      np.array([p.fixed == "edge" for p in pr.parts], float), np.log1p(npins) / 4.0], 1)
    src, dst, ea = [], [], []
    for n in pr.nets:
        pins = n["pins"]
        if not (2 <= len(pins) <= MAX_FANOUT):
            continue
        hub = max(range(len(pins)), key=lambda t: npins[pins[t][0]])
        hp, hk = pins[hub]
        ho = transform_offsets(np.array(pr.parts[hp].pad_off[hk]), np.array(k[hp]), np.array(s[hp])) * scale
        for t, (pp, kk) in enumerate(pins):
            if t == hub or pp == hp:
                continue
            po = transform_offsets(np.array(pr.parts[pp].pad_off[kk]), np.array(k[pp]), np.array(s[pp])) * scale
            src += [hp, pp]; dst += [pp, hp]
            ea += [np.concatenate([ho, po]), np.concatenate([po, ho])]
    ea = np.array(ea, dtype=float).reshape(-1, 4)
    feats = feats.copy()
    if swap:
        feats[:, [0, 1]] = feats[:, [1, 0]]
        ea = ea[:, [1, 0, 3, 2]]
    ea = ea * np.array([flip[0], flip[1], flip[0], flip[1]])
    fixed = np.array([p.corner_hole for p in pr.parts])
    return GraphCond(torch.tensor(feats, dtype=dtype, device=device),
                     torch.tensor(np.array([src, dst], dtype=np.int64).reshape(2, -1), device=device),
                     torch.tensor(ea, dtype=dtype, device=device),
                     torch.tensor(fixed, device=device))


def augment_xy(z, flip=(1, 1), swap=False):
    z = np.asarray(z).copy()
    if swap:
        z = z[:, [1, 0]]
    return z * np.array(flip)
