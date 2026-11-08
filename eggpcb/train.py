"""Train the denoiser (EGGPlacer App. C settings: cosine T=200, eps-MSE, Adam 1e-4, EMA 0.999).

Resumable: the latest checkpoint in --ckpt (local dir, mirrored to --gcs) is loaded on start.

usage: python -m eggpcb.train --data data/v1 --ckpt ckpt/run --steps 200000 [--gcs gs://.../ckpt/run]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .datagen import unpack
from .features import augment_xy, build_cond, normalize
from .model import EMA, Denoiser, GraphCond, n_params
from .rules import load_cfg

ROOT = Path(__file__).resolve().parents[1]
DIHEDRAL = [((fx, fy), sw) for fx in (1, -1) for fy in (1, -1) for sw in (False, True)]


def load_records(data_dirs, limit=None):
    recs = []
    for d in str(data_dirs).split(","):
        for f in sorted(glob.glob(str(Path(d) / "shard_*.pkl"))):
            recs += [r["cond"] if "cond" in r else r for r in pickle.load(open(f, "rb"))]
            if limit and len(recs) >= limit:
                return recs[:limit]
    return recs


class Example:
    """compact model inputs of one circuit; dihedral augmentation applied on the fly."""

    def __init__(self, rec, cfg):
        if "x" not in rec:  # old-style record: build the conditioning now
            from .datagen import cond_arrays
            pr, c, k, s, ol, ea = unpack(rec, cfg)
            rec = cond_arrays(pr, c, k, s, ol)
        self.x = torch.tensor(rec["x"].astype(np.float32)); self.ei = torch.tensor(rec["ei"].astype(np.int64))
        self.ea = torch.tensor(rec["ea"].astype(np.float32)); self.fixed = torch.tensor(rec["fixed"])
        self.z = torch.tensor(rec["z"])

    def get(self, aug, device):
        (fx, fy), swap = DIHEDRAL[aug]
        x, ea, z = self.x, self.ea, self.z
        if swap:
            x = x[:, [1, 0, 2, 3, 4]]; ea = ea[:, [1, 0, 3, 2]]; z = z[:, [1, 0]]
        ea = ea * torch.tensor([fx, fy, fx, fy], dtype=ea.dtype)
        z = z * torch.tensor([fx, fy], dtype=z.dtype)
        cond = GraphCond(x.to(device), self.ei.to(device), ea.to(device), self.fixed.to(device))
        return cond, z.to(device)


def gs(cmd):
    return subprocess.run(["gcloud", "storage"] + cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gcs", default=""); ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--circuits", type=int, default=4, help="circuits per step")
    ap.add_argument("--noise", type=int, default=16, help="noise draws per circuit (batch = circuits*noise)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val", type=int, default=250); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=5000); ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "example_board.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0, help="wall-clock budget (0 = none)")
    a = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(a.cfg)
    ck = Path(a.ckpt); ck.mkdir(parents=True, exist_ok=True)
    if a.gcs:
        gs(["rsync", "-r", a.gcs, str(ck)])
    recs = load_records(a.data, a.limit or None)
    random.Random(0).shuffle(recs)
    val_recs, tr_recs = recs[:a.val], recs[a.val:]
    print(f"device {dev}; {len(tr_recs)} train / {len(val_recs)} val circuits", flush=True)
    t0 = time.time()
    train = [Example(r, cfg) for r in tr_recs]
    val = [Example(r, cfg) for r in val_recs]
    print(f"unpacked in {time.time()-t0:.0f}s", flush=True)

    model = Denoiser(device=dev).to(dev)
    print("params", n_params(model), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ema = EMA(model, a.ema)
    step = 0
    cks = sorted(ck.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if cks:
        st = torch.load(cks[-1], map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); ema.shadow = st["ema"]
        step = st["step"]
        print(f"resumed from {cks[-1].name}", flush=True)
    log = open(ck / "log.jsonl", "a")
    run_loss, t_last = [], time.time()

    def evaluate():
        m = Denoiser(device=dev).to(dev); m.load_state_dict(ema.shadow); m.eval()
        g = torch.Generator(device=dev).manual_seed(123)
        tot = []
        with torch.no_grad():
            for ex in val:
                cond, z = ex.get(0, dev)
                t = torch.randint(1, m.T + 1, (a.noise,), device=dev, generator=g)
                tot.append(float(m.loss(z, cond, B=a.noise, t=t)))
        return float(np.mean(tot))

    while step < a.steps:
        if a.hours and time.time() - t_start > a.hours * 3600:
            a.steps = step  # time budget reached: save and stop
            f = ck / f"step_{step}.pt"
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema.shadow, step=step, args=vars(a)), f)
            torch.save(dict(ema=ema.shadow, step=step), ck / "ema_latest.pt")
            if a.gcs:
                gs(["rsync", "-r", str(ck), a.gcs])
            print(f"time budget reached at step {step}", flush=True)
            break
        model.train()
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad()
        loss_acc = 0.0
        for _ in range(a.circuits):
            ex = train[random.randrange(len(train))]
            cond, z = ex.get(random.randrange(8), dev)
            loss = model.loss(z, cond, B=a.noise) / a.circuits
            loss.backward(); loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update(model)
        step += 1
        run_loss.append(loss_acc)
        if step % a.log_every == 0:
            rec = dict(step=step, loss=float(np.mean(run_loss)), lr=lr, it_s=a.log_every / (time.time() - t_last))
            run_loss, t_last = [], time.time()
            if step % (a.log_every * 10) == 0:
                rec["val_loss_ema"] = evaluate()
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            f = ck / f"step_{step}.pt"
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema.shadow, step=step,
                            args=vars(a)), f)
            torch.save(dict(ema=ema.shadow, step=step), ck / "ema_latest.pt")
            for old in sorted(ck.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
                old.unlink()
            if a.gcs:
                gs(["rsync", "-r", "--delete-unmatched-destination-objects", str(ck), a.gcs])
    print("done", flush=True)


if __name__ == "__main__":
    main()
