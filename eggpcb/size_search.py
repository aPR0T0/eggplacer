"""Outer board-size search (Phase 4).

For each method (sa | unguided | guided) and aspect ratio, bisect the board area between --lo and
--hi (log scale). A candidate size passes if at least one seed's placement legalizes, routes 100%
with Freerouting at the user's layer count and has 0 DRC errors. Every (method, aspect, area, seed)
attempt is appended to results.jsonl (resumable, mirrored to GCS); jobs run in a process pool.

usage: python -m eggpcb.size_search --methods sa,guided --aspects 1,1.3,0.77 --seeds 3 \
           --lo 1850 --hi 3700 --steps 5 --out runs/ss --workers 48 [--ckpt ema.pt] [--gcs gs://...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dims(area, aspect):
    W = math.sqrt(area * aspect)
    return round(W, 2), round(area / W, 2)


def attempt(job):
    """place + legalize + write + route one candidate. Runs in a worker process."""
    import torch
    torch.set_num_threads(1)
    from .board_out import write_board
    from .energies import Ctx, Outline, hpwl, legality_report
    from .kicad_io import Problem
    from .legalize import legalize
    from .route import route_board
    from .rules import board_paths, gnd_net, routing_cfg
    t0 = time.time()
    method, W, H, seed, out = job["method"], job["W"], job["H"], job["seed"], Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    res = dict(job)
    try:
        pr = Problem.from_json(job["problem"])
        src_board, dru, runs_dir = board_paths(pr)
        rcfg = routing_cfg(pr)
        ol = Outline(W, H, pr.rules["outline"]["corner_radius"], pr.rules["clearance"]["edge"])
        ctx = Ctx(pr)
        if method == "sa":
            from .sa import run_sa
            c, k, s, ea, info = run_sa(ctx, ol, seed=seed, cap_frac=job.get("cap", 1.0), verbose=False,
                                       weights=job.get("weights"))
        else:
            from .model import Denoiser
            from .sample import sample
            m = Denoiser(); st = torch.load(job["ckpt"], map_location="cpu"); m.load_state_dict(st["ema"]); m.eval()
            r = sample(m, pr, ol, seed=seed, guided=(method == "guided"), guide_cfg=job.get("guide"))
            c, k, s, ea = r["c"], r["k"], r["s"], r["ea"]
        c2, k2, s2, ea2, li = legalize(pr, c, k, s, ol, ea, g=0.05)
        res["place_seconds"] = time.time() - t0
        res["legalize"] = li
        rep = legality_report(ctx, c2, k2, s2, ol)
        res["legal"] = bool(li["ok"] and rep["ok"])
        if not res["legal"]:
            res.update(passed=False, completion=0.0, reason="not legal")
            return res
        res["hpwl"] = float(hpwl(ctx, torch.tensor(c2), torch.tensor(k2), torch.tensor(s2)))
        res["bottom_parts"] = int((s2 > 0.5).sum())
        np.savez(out / "placement.npz", c=c2, k=k2, s=s2, ea=np.array([[i, e] for i, e in ea2.items()]))
        pcb = runs_dir / out.parents[2].name / out.relative_to(out.parents[2]) / "placed.kicad_pcb"
        wb = write_board(pr, c2, k2, s2, ol, src_board, pcb, dru_text=dru)
        res["board"] = wb
        rr = route_board(pcb, pcb.with_name("routed.kicad_pcb"), gnd=job.get("gnd") or rcfg["gnd"],
                         timeout=job.get("route_timeout") or rcfg["timeout"], router=rcfg["router"],
                         netclass=rcfg["netclass_overrides"], gnd_net=gnd_net(pr))
        res["route"] = {k: rr.get(k) for k in ("completion", "ok", "vias", "track_mm", "unrouted_baseline", "error")}
        res["route"]["drc"] = rr.get("drc")
        res["route"]["fr_passes"] = rr.get("freerouting", {}).get("passes")
        res["completion"] = rr.get("completion", 0.0)
        res["passed"] = bool(rr.get("ok"))
    except Exception as e:  # keep the search going; record the failure
        res.update(passed=False, completion=0.0, error=repr(e), tb=traceback.format_exc()[-2000:])
    res["seconds"] = time.time() - t0
    return res


class Bisect:
    """log-area bisection for one (method, aspect)."""

    def __init__(self, lo, hi, steps):
        self.lo, self.hi, self.steps = lo, hi, steps
        self.best_pass = None      # smallest passing area
        self.fail_max = None       # largest failing area
        self.queue = [hi]          # first verify the upper bound
        self.done = 0

    def next_area(self):
        if self.queue:
            return self.queue.pop(0)
        if self.done > self.steps:
            return None
        hi = self.best_pass if self.best_pass else None
        lo = self.fail_max if self.fail_max else self.lo
        if hi is None:
            return None            # even the upper bound failed
        if hi / lo < 1.02:
            return None
        return math.sqrt(hi * lo)

    def report(self, area, passed):
        self.done += 1
        if passed:
            self.best_pass = area if self.best_pass is None else min(self.best_pass, area)
        else:
            self.fail_max = area if self.fail_max is None else max(self.fail_max, area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="sa,guided"); ap.add_argument("--aspects", default="1.0")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True); ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ckpt", default=""); ap.add_argument("--gcs", default="")
    ap.add_argument("--route_timeout", type=int, default=0); ap.add_argument("--gnd", default="")
    ap.add_argument("--problem", default=str(ROOT / "out" / "problem.json"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    resf = out / "results.jsonl"
    done = {}
    if resf.exists():
        for l in resf.read_text().splitlines():
            r = json.loads(l); done[(r["method"], r["aspect"], round(r["area"], 1), r["seed"])] = r
    states = {(m, float(asp)): Bisect(a.lo, a.hi, a.steps) for m in a.methods.split(",") for asp in a.aspects.split(",")}
    # replay finished attempts into the bisection states
    for key, st in states.items():
        while True:
            area = st.next_area()
            if area is None:
                break
            rs = [done.get((key[0], key[1], round(area, 1), sd)) for sd in range(a.seeds)]
            if any(r is None for r in rs):
                st.queue.insert(0, area); break
            st.report(area, any(r["passed"] for r in rs))
    pool = ProcessPoolExecutor(a.workers)
    running = {}

    def submit(key, area):
        W, H = dims(area, key[1])
        futs = []
        for sd in range(a.seeds):
            if (key[0], key[1], round(area, 1), sd) in done:
                continue
            job = dict(method=key[0], aspect=key[1], area=area, W=W, H=H, seed=sd, ckpt=a.ckpt, problem=a.problem,
                       route_timeout=a.route_timeout, gnd=a.gnd,
                       out=str(out / key[0] / f"a{key[1]:.2f}" / f"A{area:.0f}_s{sd}"))
            futs.append(pool.submit(attempt, job))
        running[key] = (area, futs)

    for key, st in states.items():
        area = st.next_area()
        if area is not None:
            submit(key, area)
    while running:
        time.sleep(10)
        for key in list(running):
            area, futs = running[key]
            if not all(f.done() for f in futs):
                continue
            for f in futs:
                r = f.result(); done[(r["method"], r["aspect"], round(r["area"], 1), r["seed"])] = r
                with open(resf, "a") as fh:
                    fh.write(json.dumps(r, default=str) + "\n")
            rs = [done[(key[0], key[1], round(area, 1), sd)] for sd in range(a.seeds)]
            passed = any(r["passed"] for r in rs)
            states[key].report(area, passed)
            print(f"{key[0]:9s} aspect {key[1]:.2f} area {area:7.0f} -> {'PASS' if passed else 'fail'} "
                  f"(best completion {max(r['completion'] for r in rs):.3f})", flush=True)
            del running[key]
            nxt = states[key].next_area()
            if nxt is not None:
                submit(key, nxt)
            if a.gcs:
                subprocess.run(["gcloud", "storage", "rsync", "-r", str(out), a.gcs], capture_output=True)
    summary = {f"{m}@{asp}": dict(best_pass_area=st.best_pass, largest_fail=st.fail_max) for (m, asp), st in states.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
