"""SA baseline on the real board at one outline size: SA -> legalize -> exact legality check ->
render -> new .kicad_pcb (+ DRC) -> optional routing (pcb_router).

usage: python scripts/run_sa.py --W 58 --H 58 --seed 0 --out out/sa_58x58_s0 [--route --gnd plane_b]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eggpcb.board_out import run_drc, write_board  # noqa: E402
from eggpcb.energies import Ctx, Outline, hpwl, legality_report  # noqa: E402
from eggpcb.kicad_io import Problem  # noqa: E402
from eggpcb.legalize import legalize  # noqa: E402
from eggpcb.render import render  # noqa: E402
from eggpcb.rules import board_paths, route_placed  # noqa: E402
from eggpcb.sa import SAArrays, TERM_NAMES, energy_terms_np, run_sa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, required=True); ap.add_argument("--H", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--temps", type=int, default=250); ap.add_argument("--moves", type=int, default=40)
    ap.add_argument("--route", action="store_true"); ap.add_argument("--gnd", default=None)
    ap.add_argument("--route_timeout", type=int, default=None)
    ap.add_argument("--problem", default=str(ROOT / "out" / "problem.json"))
    ap.add_argument("--cap", type=float, default=1.0, help="bin density cap (fraction of bin area)")
    ap.add_argument("--compact", type=float, default=None, help="compactness weight override")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pr = Problem.from_json(a.problem)
    src_board, dru, runs_dir = board_paths(pr)
    ctx = Ctx(pr)
    ol = Outline(a.W, a.H, pr.rules["outline"]["corner_radius"], pr.rules["clearance"]["edge"])
    t0 = time.time()
    w = {} if a.compact is None else dict(compact=a.compact)
    c, k, s, ea, info = run_sa(ctx, ol, seed=a.seed, n_temps=a.temps, moves_per_part=a.moves, cap_frac=a.cap,
                               weights=w)
    pre = legality_report(ctx, c, k, s, ol)
    c2, k2, s2, ea2, li = legalize(pr, c, k, s, ol, ea, g=0.05)
    rep = legality_report(ctx, c2, k2, s2, ol)
    t_place = time.time() - t0
    A = SAArrays(ctx)
    terms = dict(zip(TERM_NAMES, map(float, energy_terms_np(A, c2, k2, s2, ol))))
    res = dict(W=a.W, H=a.H, seed=a.seed, method="SA", cap=a.cap, compact=a.compact, place_seconds=t_place, legal=rep["ok"],
               legalize=li, pre_legal_overlap_area=pre["overlap_area"],
               hpwl=float(hpwl(ctx, torch.tensor(c2), torch.tensor(k2), torch.tensor(s2))),
               bottom_parts=int((s2 > 0.5).sum()), terms=terms)
    np.savez(out / "placement.npz", c=c2, k=k2, s=s2, ea=np.array([[i, e] for i, e in ea2.items()]))
    render(pr, c2, k2, s2, outline=ol.as_dict(), path=out / "placement.png",
           title=f"SA {a.W:.1f}x{a.H:.1f} seed {a.seed}")
    if rep["ok"]:
        pcb = runs_dir / out.name / "placed.kicad_pcb"
        wb = write_board(pr, c2, k2, s2, ol, src_board, pcb, dru_text=dru)
        res["board"] = wb
        res["drc_placed"] = run_drc(pcb)
        if a.route:
            res["route"] = route_placed(pr, pcb, pcb.with_name("routed.kicad_pcb"), timeout=a.route_timeout,
                                        gnd=a.gnd)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("terms",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
