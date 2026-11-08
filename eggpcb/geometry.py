"""Per-part geometry under rotation / side, as numpy arrays (used by energies, SA, legalizer, render).

Rotation index k in {0,1,2,3} means KiCad orientation 90*k degrees.
Side s in {0 (top), 1 (bottom)}.
"""
from __future__ import annotations

import numpy as np

from .kicad_io import Problem, rot_mat

ROTS = np.stack([rot_mat(90 * k) for k in range(4)])  # (4,2,2)  world = R @ local
MIRROR = np.array([1.0, -1.0])


def box_dims(pr: Problem):
    """(N,2) w,h at rotation 0."""
    return np.array([[p.w, p.h] for p in pr.parts])


def dims_rot(wh: np.ndarray, k: np.ndarray):
    """box w,h after rotation k (90/270 swap)."""
    k = np.asarray(k)
    odd = (k % 2 == 1)[..., None]
    return np.where(odd, wh[..., ::-1], wh)


def transform_offsets(off: np.ndarray, k: np.ndarray, side: np.ndarray):
    """off (...,2) top-local offsets; k, side broadcastable (...) -> world offsets."""
    o = np.where(np.asarray(side)[..., None] == 1, off * MIRROR, off)
    R = ROTS[np.asarray(k) % 4]
    return np.einsum("...ij,...j->...i", R, o)


def mouth_world_dir(mouth: str, k: int):
    """unit vector (world) the connector mouth faces after rotation k (top side)."""
    v = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[mouth]
    return ROTS[k % 4] @ np.array(v, dtype=float)


def k_for_mouth(mouth: str, edge: str) -> int:
    """rotation index that makes `mouth` face board edge `edge` in {'left','right','top','bottom'}
    (KiCad y down: 'top' is -y)."""
    want = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[edge]
    for k in range(4):
        d = mouth_world_dir(mouth, k)
        if np.allclose(d, want):
            return k
    raise ValueError


def fp_origin(center, part, k, side):
    """footprint origin (KiCad position) for a part whose box centre is at `center`."""
    off = transform_offsets(np.array(part.box_off), np.array(k), np.array(side))
    return np.asarray(center) - off


def initial_state(pr: Problem):
    """centres, k, side from the board as parsed."""
    c, ks, ss = [], [], []
    for p in pr.parts:
        k = int(round(p.rot0 / 90.0)) % 4
        s = p.side0
        off = transform_offsets(np.array(p.box_off), np.array(k), np.array(s))
        c.append(np.array(p.pos0) + off)
        ks.append(k); ss.append(s)
    return np.array(c), np.array(ks), np.array(ss)
