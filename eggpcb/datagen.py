"""Training data: synthetic circuit -> quick analytical placement on its tight outline -> legalize.

Each record (pickled dict of numpy arrays, see pack()) holds the netlist geometry, rotations,
sides, outline and the legal label placement. Shards of `shard_size` records are written as
<out>/shard_XXXXX.pkl; finished shards are skipped on restart (resumable), and the output dir can be
a gs:// path (synced with gsutil/gcloud storage after each shard).

usage: python -m eggpcb.datagen --out data/v1 --n 5000 --workers 8 [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .energies import Ctx, Outline, all_terms, legality_report
from .kicad_io import Part, Problem
from .layout import apply_constraints, corner_centers, edge_place
from .legalize import legalize
from .rules import load_cfg
from .synth import Library, generate

ROOT = Path(__file__).resolve().parents[1]


def analytic_place(ctx: Ctx, c, k, s, ol: Outline, ea, steps=250, lr=0.25, seed=0, noise=1.0,
                   w=None, leg=(0.1, 500.0)):
    """Adam on WA-WL + density + decap + compact + group with a geometric legality ramp. Sides and
    rotations fixed; edge parts slide along their edges; corner holes fixed."""
    w = w or dict(wawl=1.0, density=0.5, decap=2.0, compact=0.005, group=0.02)
    g = torch.Generator().manual_seed(seed)
    c0 = torch.tensor(c, dtype=ctx.dtype)
    c0 = c0 + noise * torch.randn(c0.shape, generator=g, dtype=ctx.dtype) * ctx.movable[:, None]
    kt = torch.tensor(k); st = torch.tensor(s, dtype=ctx.dtype)
    x = c0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, foreach=False)
    mov = ctx.movable.to(ctx.dtype)[:, None]
    pr = ctx.pr
    edge_idx = [i for i, p in enumerate(pr.parts) if p.fixed == "edge"]
    for it in range(steps):
        a = it / max(steps - 1, 1)
        lam = leg[0] * (leg[1] / leg[0]) ** a
        T = all_terms(ctx, x, kt, st, ol, gamma=max(0.5, 2.0 * (1 - a)))
        E = (w["wawl"] * T["wawl"] + w["density"] * T["density"] + w["decap"] * T["decap"] +
             w["compact"] * T["compact"] + w["group"] * T["group"] + lam * (T["overlap"] + T["contain"]))
        opt.zero_grad(); E.backward()
        x.grad *= mov
        opt.step()
        for gp in opt.param_groups:
            gp["lr"] = lr * (1 - 0.8 * a)
        with torch.no_grad():  # keep inside the board, edge parts on their edge
            x[:, 0].clamp_(0, ol.W); x[:, 1].clamp_(0, ol.H)
            for i in edge_idx:
                e = ea[i]; p = pr.parts[i]
                along = float(x[i, 1] if e in (0, 1) else x[i, 0])
                (cx, cy), _ = edge_place(p, e, along, ol)
                x[i, 0] = cx; x[i, 1] = cy
    return x.detach().numpy()


def pack(pr: Problem, c, k, s, ol: Outline, ea, meta=None):
    """compact record (numpy) of a placed problem."""
    parts = pr.parts
    rec = dict(
        W=ol.W, H=ol.H, r=ol.r, edge=ol.edge,
        wh=np.array([[p.w, p.h] for p in parts], np.float32),
        kind=np.array([p.kind for p in parts]),
        ptype=np.array([2 if p.corner_hole else (1 if p.fixed == "edge" else (3 if p.fixed == "abs" else 0)) for p in parts], np.int8),
        allow_rot=np.array([p.allow_rotate for p in parts]), allow_bottom=np.array([p.allow_bottom for p in parts]),
        mouth=np.array([p.edge_mouth for p in parts]), overhang=np.array([p.edge_overhang for p in parts], np.float32),
        sheet=np.array([p.sheet for p in parts]),
        pad_ptr=np.cumsum([0] + [len(p.pad_off) for p in parts]).astype(np.int32),
        pad_off=np.array([o for p in parts for o in p.pad_off], np.float32).reshape(-1, 2),
        pad_size=np.array([o for p in parts for o in p.pad_size], np.float32).reshape(-1, 2),
        pad_net=np.array([n for p in parts for n in p.pad_net], np.int32),
        hole_ptr=np.cumsum([0] + [len(p.holes) for p in parts]).astype(np.int32),
        holes=np.array([h for p in parts for h in p.holes], np.float32).reshape(-1, 3),
        net_w=np.array([n["weight"] for n in pr.nets], np.float32),
        net_name=np.array([n["name"] for n in pr.nets]),
        pos=np.asarray(c, np.float32), k=np.asarray(k, np.int8), s=np.asarray(s, np.int8),
        ea=np.array([[i, e] for i, e in ea.items()], np.int32).reshape(-1, 2),
        meta=meta or {},
    )
    return rec


def unpack(rec, cfg) -> tuple:
    parts = []
    N = len(rec["wh"])
    for i in range(N):
        a, b = rec["pad_ptr"][i], rec["pad_ptr"][i + 1]
        ha, hb = rec["hole_ptr"][i], rec["hole_ptr"][i + 1]
        holes = rec["holes"][ha:hb].tolist()
        pt = int(rec["ptype"][i])
        p = Part(ref=f"P{i}", lib="", value="", sheet=str(rec["sheet"][i]), kind=str(rec["kind"][i]),
                 w=float(rec["wh"][i, 0]), h=float(rec["wh"][i, 1]), box_off=[0.0, 0.0],
                 pad_names=[str(j) for j in range(b - a)], pad_off=rec["pad_off"][a:b].tolist(),
                 pad_size=rec["pad_size"][a:b].tolist(), pad_net=rec["pad_net"][a:b].tolist(), holes=holes,
                 pos0=[0, 0], rot0=0.0, side0=0, locked=False, courtyard_ok=True)
        p.fixed = {0: "none", 1: "edge", 2: "corner", 3: "abs"}[pt]
        p.corner_hole = pt == 2
        p.allow_rotate = bool(rec["allow_rot"][i]); p.allow_bottom = bool(rec["allow_bottom"][i])
        p.edge_mouth = str(rec["mouth"][i]); p.edge_overhang = float(rec["overhang"][i])
        p.bottom_hole_keepout = [[hx - r - 0.25, hy - r - 0.25, hx + r + 0.25, hy + r + 0.25] for hx, hy, r in holes]
        parts.append(p)
    nets = [dict(name=str(nm), pins=[], weight=float(wt)) for nm, wt in zip(rec["net_name"], rec["net_w"])]
    for i, p in enumerate(parts):
        for kk, n in enumerate(p.pad_net):
            if n >= 0:
                nets[n]["pins"].append([i, kk])
    pr = Problem(name="rec", parts=parts, nets=nets, layers=2, rules=cfg)
    ol = Outline(float(rec["W"]), float(rec["H"]), float(rec["r"]), float(rec["edge"]))
    ea = {int(i): int(e) for i, e in rec["ea"]}
    return pr, rec["pos"].astype(np.float64), rec["k"].astype(np.int64), rec["s"].astype(np.float64), ol, ea


_G = {}


def cond_arrays(pr, c, k, s, ol):
    """model inputs precomputed in the worker (compact; training never needs the Problem)."""
    from .features import build_cond, normalize
    g = build_cond(pr, k, s, ol.W, ol.H)
    return dict(x=g.x.numpy().astype(np.float16), ei=g.edge_index.numpy().astype(np.int32),
                ea=g.edge_attr.numpy().astype(np.float16), fixed=g.is_fixed.numpy(),
                z=normalize(c, ol.W, ol.H).astype(np.float32))


def _init(problem_json, cfg_path):
    torch.set_num_threads(1)
    pr = Problem.from_json(problem_json)
    _G["cfg"] = load_cfg(cfg_path)
    _G["lib"] = Library(pr)


def make_one(seed):
    rng = np.random.default_rng(seed)
    cfg, lib = _G["cfg"], _G["lib"]
    for attempt in range(6):
        r = generate(rng, lib, cfg)
        if r is None:
            continue
        pr, c, k, s, ol, ea = r
        ctx = Ctx(pr)
        t0 = time.time()
        c2 = analytic_place(ctx, c, k, s, ol, ea, seed=seed)
        c3, k3, s3, ea3, info = legalize(pr, c2, k, s, ol, ea, g=0.1)
        if not info["ok"]:
            continue
        rep = legality_report(ctx, c3, k3, s3, ol)
        if not rep["ok"]:
            continue
        with torch.no_grad():
            T0 = all_terms(ctx, torch.tensor(c), torch.tensor(k), torch.tensor(s), ol)
            T1 = all_terms(ctx, torch.tensor(c3), torch.tensor(k3), torch.tensor(s3), ol)
        if float(T1["wawl"]) > float(T0["wawl"]) and legality_report(ctx, c, k, s, ol)["ok"]:
            # legalization undid the analytic gain: keep the (legal) latent placement
            c3, k3, s3, ea3, T1 = c, k, s, ea, T0
        meta = dict(seed=int(seed), latent_wawl=float(T0["wawl"]), label_wawl=float(T1["wawl"]),
                    place_s=time.time() - t0, n=len(pr.parts), legal_moves=info["n_moved"])
        rec = pack(pr, c3, k3, s3, ol, ea3, meta)
        rec["cond"] = cond_arrays(pr, c3, k3, s3, ol)
        return rec
    return None


def _gs_sync(local, remote):
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "-r", str(local), remote], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=250); ap.add_argument("--gcs", default="")
    ap.add_argument("--problem", default=str(ROOT / "out" / "problem.json"))
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "example_board.json"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_shards = math.ceil(a.n / a.shard)
    with Pool(a.workers, initializer=_init, initargs=(a.problem, a.cfg)) as pool:
        for sh in range(n_shards):
            f = out / f"shard_{sh:05d}.pkl"
            if f.exists():
                continue
            seeds = [a.seed * 10_000_000 + sh * a.shard + j for j in range(a.shard)]
            t0 = time.time()
            recs = [r for r in pool.imap_unordered(make_one, seeds, chunksize=2) if r is not None]
            tmp = f.with_suffix(".tmp"); pickle.dump(recs, open(tmp, "wb")); tmp.rename(f)
            wl = np.mean([r["meta"]["label_wawl"] / max(r["meta"]["latent_wawl"], 1e-9) for r in recs])
            print(f"shard {sh+1}/{n_shards}: {len(recs)} circuits in {time.time()-t0:.0f}s, label/latent WL {wl:.3f}", flush=True)
            _gs_sync(out, a.gcs)
    (out / "_DONE").write_text(time.ctime())
    _gs_sync(out, a.gcs)


if __name__ == "__main__":
    main()
