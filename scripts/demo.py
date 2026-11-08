"""Two-minute demo, no KiCad needed: generate a synthetic PCB circuit, place it with the SA baseline
(and, given a checkpoint, with energy-guided diffusion), legalize, and render PNGs.

usage: python scripts/demo.py [--seed 0] [--ckpt ema_latest.pt] [--out out/demo]
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
from eggpcb.energies import Ctx, hpwl, legality_report  # noqa: E402
from eggpcb.legalize import legalize  # noqa: E402
from eggpcb.render import render  # noqa: E402
from eggpcb.rules import load_cfg  # noqa: E402
from eggpcb.sa import run_sa  # noqa: E402
from eggpcb.synth import generate, generic_library  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="out/demo"); ap.add_argument("--parts", type=float, default=0.6,
                                                                  help="circuit size as a fraction of 200 parts")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(ROOT / "configs" / "example_board.json")
    rng = np.random.default_rng(a.seed)
    r = None
    while r is None:
        r = generate(rng, generic_library(), cfg, n_scale=(a.parts, a.parts))
    pr, c_label, k, s, ol, ea = r
    ctx = Ctx(pr)
    H = lambda c, k_, s_: float(hpwl(ctx, torch.tensor(c), torch.tensor(k_), torch.tensor(s_)))
    print(f"circuit: {len(pr.parts)} parts, {len(pr.nets)} nets, outline {ol.W:.1f} x {ol.H:.1f} mm")
    rows = {"generator's latent packing": dict(hpwl=H(c_label, k, s), legal=True)}
    render(pr, c_label, k, s, outline=ol.as_dict(), path=out / "latent.png", title="latent packing")

    t0 = time.time()
    c, k2, s2, ea2, _ = run_sa(ctx, ol, seed=a.seed, n_temps=120, moves_per_part=20, verbose=False)
    c, k2, s2, ea2, li = legalize(pr, c, k2, s2, ol, ea2)
    rows["simulated annealing"] = dict(hpwl=H(c, k2, s2), legal=legality_report(ctx, c, k2, s2, ol)["ok"],
                                       seconds=round(time.time() - t0, 1))
    render(pr, c, k2, s2, outline=ol.as_dict(), path=out / "sa.png", title="SA baseline (legalized)")

    if a.ckpt:
        from eggpcb.model import Denoiser
        from eggpcb.sample import sample
        m = Denoiser(); m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["ema"]); m.eval()
        for guided in (False, True):
            t0 = time.time()
            smp = sample(m, pr, ol, seed=a.seed, guided=guided)
            c3, k3, s3, ea3, li = legalize(pr, smp["c"], smp["k"], smp["s"], ol, smp["ea"])
            name = "diffusion + energy guidance" if guided else "diffusion, unguided"
            rows[name] = dict(hpwl=H(c3, k3, s3) if li["ok"] else None, legal=li["ok"],
                              seconds=round(time.time() - t0, 1))
            render(pr, c3, k3, s3, outline=ol.as_dict(), path=out / f"{'guided' if guided else 'unguided'}.png", title=name)
    print(json.dumps(rows, indent=1))
    print(f"renders in {out}/")


if __name__ == "__main__":
    main()
