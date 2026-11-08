"""Energy tests: torch == numba (same energy for SA), and gradients match finite differences."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggpcb.energies import Ctx, Outline, all_terms  # noqa: E402
from eggpcb.kicad_io import Problem  # noqa: E402
from eggpcb.layout import apply_constraints  # noqa: E402
from eggpcb.sa import SAArrays, energy_terms_np, TERM_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROB = ROOT / "out" / "problem.json"


def load_problem():
    """your parsed board (out/problem.json) if present, otherwise a synthetic circuit."""
    if PROB.exists():
        return Problem.from_json(PROB)
    from eggpcb.rules import load_cfg
    from eggpcb.synth import generate, generic_library
    cfg = load_cfg(ROOT / "configs" / "example_board.json")
    rng = np.random.default_rng(0)
    while True:
        r = generate(rng, generic_library(), cfg, n_scale=(0.4, 0.5))
        if r is not None:
            return r[0]


def random_state(pr, ol, rng, soft=False):
    N = len(pr.parts)
    c = np.stack([rng.uniform(0, ol.W, N), rng.uniform(0, ol.H, N)], 1)
    k = np.array([rng.integers(4) if p.allow_rotate and p.fixed == "none" else 0 for p in pr.parts])
    s = np.array([float(rng.random() < 0.4) if p.allow_bottom else 0.0 for p in pr.parts])
    c, k, s, ea = apply_constraints(pr, c, k, s, ol)
    if soft:
        s = np.array([rng.uniform(0.1, 0.9) if p.allow_bottom else 0.0 for p in pr.parts])
    return c, k, s


def test_torch_equals_numba():
    pr = load_problem()
    ctx = Ctx(pr)
    A = SAArrays(ctx)
    ol = Outline(48.0, 44.0, 0.5, 0.2)
    rng = np.random.default_rng(1)
    for trial in range(3):
        c, k, s = random_state(pr, ol, rng)
        tt = all_terms(ctx, torch.tensor(c), torch.tensor(k), torch.tensor(s), ol)
        tn = energy_terms_np(A, c, k, s, ol)
        names = dict(wawl="wawl", density="density", overlap="overlap", contain="contain", decap="decap",
                     group="group", compact="compact", bottom="bottom")
        # torch 'overlap' includes the through-hole keep-outs; numba keeps them as a separate term
        tor = {kk: float(v) for kk, v in tt.items()}
        num = dict(zip(TERM_NAMES, tn))
        num["overlap"] = num["overlap"] + num.pop("keepout_holes")
        for kk in names:
            a, b = tor[kk], num[kk]
            assert abs(a - b) <= 1e-7 * max(1.0, abs(a)), (kk, a, b)
        print("trial", trial, {kk: round(tor[kk], 4) for kk in names})


def test_gradients_fd():
    pr = load_problem()
    ctx = Ctx(pr)
    ol = Outline(48.0, 44.0, 0.5, 0.2)
    rng = np.random.default_rng(2)
    c, k, s = random_state(pr, ol, rng, soft=True)
    c = torch.tensor(c, requires_grad=True); kt = torch.tensor(k); st = torch.tensor(s, requires_grad=True)
    terms = all_terms(ctx, c, kt, st, ol)
    v = torch.randn_like(c); vs = torch.randn_like(st) * ctx.allow_bottom.double()
    h = 1e-5
    for name, val in terms.items():
        if not val.requires_grad:
            continue
        gc, gs = torch.autograd.grad(val, (c, st), allow_unused=True)
        gc = torch.zeros_like(c) if gc is None else gc
        gs = torch.zeros_like(st) if gs is None else gs
        an = float((gc * v).sum() + (gs * vs).sum())
        with torch.no_grad():
            ep = all_terms(ctx, c + h * v, kt, st + h * vs, ol)[name]
            em = all_terms(ctx, c - h * v, kt, st - h * vs, ol)[name]
        fd = float((ep - em) / (2 * h))
        rel = abs(an - fd) / max(1e-6, abs(fd), abs(an))
        print(f"  grad {name:8s} analytic {an: .6e}  fd {fd: .6e}  rel {rel:.2e}")
        assert rel < 1e-4 or abs(an - fd) < 1e-6, name


def test_gradcheck_small():
    """torch.autograd.gradcheck on a 12-part sub-problem (all terms)."""
    pr = load_problem()
    corner = [i for i, p in enumerate(pr.parts) if p.corner_hole]
    edge = [i for i, p in enumerate(pr.parts) if p.fixed == "edge"][:1]
    rest = [i for i in range(len(pr.parts)) if i not in corner and i not in edge]
    keep = corner + edge + rest[: 12 - len(corner) - len(edge)]
    sub = Problem(name="sub", parts=[pr.parts[i] for i in keep], nets=[], layers=2, rules=pr.rules)
    remap = {o: n for n, o in enumerate(keep)}
    for n in pr.nets:
        pins = [[remap[a], b] for a, b in n["pins"] if a in remap]
        if len(pins) >= 2:
            sub.nets.append(dict(name=n["name"], pins=pins, weight=n["weight"]))
    for p in sub.parts:
        p.pad_net = [-1] * len(p.pad_net)
    for ni, n in enumerate(sub.nets):
        for a, b in n["pins"]:
            sub.parts[a].pad_net[b] = ni
    ctx = Ctx(sub)
    ol = Outline(20.0, 16.0, 0.5, 0.2)
    rng = np.random.default_rng(3)
    c, k, s = random_state(sub, ol, rng, soft=True)
    kt = torch.tensor(k)
    for name in ["wawl", "density", "overlap", "contain", "decap", "group", "compact"]:
        def f(cc, ss, name=name):
            return all_terms(ctx, cc, kt, ss, ol)[name]
        ok = torch.autograd.gradcheck(f, (torch.tensor(c, requires_grad=True), torch.tensor(s, requires_grad=True)),
                                      eps=1e-6, atol=1e-5, rtol=1e-4)
        print("  gradcheck", name, ok)


if __name__ == "__main__":
    test_torch_equals_numba(); test_gradients_fd(); test_gradcheck_small(); print("ALL OK")
