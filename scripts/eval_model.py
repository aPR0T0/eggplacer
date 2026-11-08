"""Phase-3 model check: sample held-out synthetic circuits (and the real board) with the trained
denoiser, unguided and guided, and compare raw-sample quality with the training labels and with
random placements.

usage: python scripts/eval_model.py --ckpt ckpt/small/ema_latest.pt --data data/v1_5k --n 16 --out out/eval_small
"""
import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eggpcb.datagen import unpack  # noqa: E402
from eggpcb.energies import Ctx, Outline, hpwl, legality_report  # noqa: E402
from eggpcb.kicad_io import Problem  # noqa: E402
from eggpcb.legalize import legalize  # noqa: E402
from eggpcb.model import Denoiser  # noqa: E402
from eggpcb.render import render  # noqa: E402
from eggpcb.rules import load_cfg  # noqa: E402
from eggpcb.sample import sample  # noqa: E402


def metrics(pr, c, k, s, ol, legal=True, ea=None):
    ctx = Ctx(pr)
    rep = legality_report(ctx, c, k, s, ol)
    area = sum(p.w * p.h for p in pr.parts)
    out = dict(overlap_frac=rep["overlap_area"] / area, n_overlap=len(rep["overlaps"]), n_outside=len(rep["outside"]),
               hpwl=float(hpwl(ctx, torch.tensor(c), torch.tensor(k), torch.tensor(s))))
    if legal:
        c2, k2, s2, ea2, li = legalize(pr, c, k, s, ol, ea or {})
        out.update(legal_ok=li["ok"], legal_moved=li["n_moved"], legal_mean_disp=li["mean_disp"],
                   hpwl_legal=float(hpwl(ctx, torch.tensor(c2), torch.tensor(k2), torch.tensor(s2))) if li["ok"] else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--board_size", type=float, nargs=2, default=[54, 54])
    ap.add_argument("--problem", default=str(ROOT / "out" / "problem.json"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Problem.from_json(a.problem).rules
    m = Denoiser(); st = torch.load(a.ckpt, map_location="cpu"); m.load_state_dict(st["ema"]); m.eval()
    files = sorted(glob.glob(str(Path(a.data) / "shard_*.pkl")))
    recs = pickle.load(open(files[-1], "rb"))[-a.n:]   # last shard's tail: never in the first train shards
    rows = []
    rng = np.random.default_rng(0)
    for j, rec in enumerate(recs):
        pr, c, k, s, ol, ea = unpack(rec, cfg)
        r = dict(i=j, n=len(pr.parts))
        r["label"] = metrics(pr, c, k, s, ol, legal=False)
        crand = np.stack([rng.uniform(0, ol.W, len(c)), rng.uniform(0, ol.H, len(c))], 1)
        r["random"] = metrics(pr, crand, k, s, ol, legal=False)
        t0 = time.time()
        smp = sample(m, pr, ol, seed=j, guided=False, init=(k, s), discrete=False)
        r["unguided"] = metrics(pr, smp["c"], smp["k"], smp["s"], ol, ea=smp["ea"]); r["unguided"]["sec"] = time.time() - t0
        if j < 4:
            render(pr, smp["c"], k, s, outline=ol.as_dict(), path=out / f"synth{j}_unguided.png", title=f"unguided {j}")
            render(pr, c, k, s, outline=ol.as_dict(), path=out / f"synth{j}_label.png", title=f"label {j}")
        rows.append(r)
        print(json.dumps({kk: (v if not isinstance(v, dict) else {x: round(y, 3) if isinstance(y, float) else y for x, y in v.items()}) for kk, v in r.items()}), flush=True)
    summ = {}
    for key in ("label", "random", "unguided"):
        for mtr in ("overlap_frac", "n_outside"):
            summ[f"{key}_{mtr}"] = float(np.mean([r[key][mtr] for r in rows]))
        summ[f"{key}_hpwl_rel_label"] = float(np.mean([r[key]["hpwl"] / r["label"]["hpwl"] for r in rows]))
    summ["unguided_legal_rate"] = float(np.mean([r["unguided"]["legal_ok"] for r in rows]))
    summ["unguided_hpwl_legal_rel_label"] = float(np.mean([r["unguided"]["hpwl_legal"] / r["label"]["hpwl"] for r in rows if r["unguided"]["hpwl_legal"]]))
    # real board
    pr = Problem.from_json(a.problem)
    ol = Outline(a.board_size[0], a.board_size[1], 0.5, 0.2)
    board = {}
    for guided in (False, True):
        for seed in range(2):
            t0 = time.time()
            smp = sample(m, pr, ol, seed=seed, guided=guided)
            mt = metrics(pr, smp["c"], smp["k"], smp["s"], ol, legal=False)
            c2, k2, s2, ea2, li = legalize(pr, smp["c"], smp["k"], smp["s"], ol, smp["ea"])
            ctx = Ctx(pr)
            mt.update(legal_ok=li["ok"], legal_moved=li["n_moved"], sec=time.time() - t0,
                      hpwl_legal=float(hpwl(ctx, torch.tensor(c2), torch.tensor(k2), torch.tensor(s2))) if li["ok"] else None,
                      bottom=int((s2 > 0.5).sum()))
            board[f"{'guided' if guided else 'unguided'}_s{seed}"] = mt
            render(pr, c2, k2, s2, outline=ol.as_dict(), path=out / f"board_{'guided' if guided else 'unguided'}_s{seed}.png",
                   title=f"{'guided' if guided else 'unguided'} seed {seed} (legalized)")
            print("board", guided, seed, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in mt.items()}), flush=True)
    res = dict(ckpt=a.ckpt, step=int(st.get("step", -1)), synthetic=summ, board=board, rows=rows)
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(dict(synthetic=summ, board=board), indent=1, default=str))


if __name__ == "__main__":
    main()
