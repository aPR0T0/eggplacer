"""eggplacer: EGGPlacer for PCBs (energy-guided placement, `eggpcb/`).

`method: sa` (default) runs simulated annealing on the EGGPlacer energies (pad wirelength, overlap,
outline containment, decoupling distance, sheet grouping, compactness); `guided` / `unguided` sample
the trained diffusion model (`checkpoint`), with or without energy guidance. Every proposal is
legalized; if none is legal at the current outline, the outline grows by `grow` and it tries again.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..base import Placer, register
from ..kicad_env import import_pcbnew

DEFAULT_RULES = dict(
    layers=2,
    outline=dict(shape="rect", aspect="free", corner_radius=0.5, max_w=None, max_h=None),
    clearance=dict(edge=0.2, courtyard=0.05, hole_both_sides=0.25),
    mounting_holes={},
    fixed_abs=[],
    edge_parts={},
    sides=dict(allow_bottom_kinds=[], bottom_penalty=10.0),
    rotation=dict(allowed=[0, 90, 180, 270]),
    decoupling=dict(cap_kind="C", ic_kinds=["U"], gnd_nets=None),
    net_weights=dict(gnd_weight=0.15, power_nets=None, power_weight=0.5, fanout_cap=12),
)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _is_gnd(name: str) -> bool:
    n = name.rsplit("/", 1)[-1].upper()
    return n in ("GND", "GNDA", "GNDD", "AGND", "DGND", "PGND", "VSS", "0V") or n.startswith("GND")


def _is_power(name: str) -> bool:
    import re
    n = name.rsplit("/", 1)[-1].upper()
    return n.startswith("+") or bool(re.match(r"^(VCC|VDD|VBUS|VIN|VBAT|V\d|\d+V\d*)", n))


@register("eggplacer")
class EggPlacer(Placer):
    summary = "EGGPlacer: energy-guided placement (simulated annealing, or diffusion with a trained checkpoint)"
    isolate = True

    @dataclass
    class Options:
        method: str = "sa"                 # sa | guided | unguided
        checkpoint: str | None = None      # trained denoiser (.pt) for guided / unguided
        guide: dict = field(default_factory=dict)       # guided-sampling settings (eggpcb.sample)
        width: float | None = None         # board outline in mm; None = from the parts' area
        height: float | None = None
        density: float = 0.35              # auto size: sum of courtyard areas / board area
        aspect: float = 1.0                # auto size: width / height
        grow: float = 1.08                 # outline growth when no legal placement is found
        fit_outline: bool = True           # shrink the outline to the placed parts (+ fit_margin)
        fit_margin: float = 1.0            # mm of routing room between the outermost parts and the edge
        tidy_silk: bool = True             # move reference designators off pads / other silkscreen
        max_tries: int = 8
        seeds: list = field(default_factory=lambda: [0])   # best legal HPWL over these seeds
        temps: int = 250                   # SA temperatures
        moves: int = 40                    # SA moves per part per temperature
        cap: float = 1.0                   # SA bin density cap
        compact: float | None = None       # SA compactness weight override
        weights: dict = field(default_factory=dict)     # SA energy weights override
        layers: int | None = None          # copper layers (default: the board's)
        corner_radius: float = 0.5
        clearance: dict = field(default_factory=dict)   # edge, courtyard, hole_both_sides (mm)
        mounting_holes: dict = field(default_factory=dict)   # {refs: [...], mode: corners, edge_gap}
        fixed: list = field(default_factory=list)       # refs that keep their coordinates (+ locked parts)
        edge_parts: dict = field(default_factory=dict)  # {J1: {mouth: -x, edge_line: courtyard}}
        allow_bottom_kinds: list = field(default_factory=list)   # e.g. [R, C, L]; [] = single-sided
        bottom_penalty: float = 10.0
        decoupling: dict = field(default_factory=dict)  # cap_kind, ic_kinds, gnd_nets
        net_weights: dict = field(default_factory=dict)  # gnd_weight, power_nets, power_weight, fanout_cap
        rules_file: str | None = None      # an EGGPlacer board config (JSON/YAML, GUIDE.md step 3)
        verbose: bool = False

    def __init__(self, options: dict | None = None):
        super().__init__(options)
        self.given = set(options or {})

    # ------------------------------------------------------------------ rules
    def rules(self, pr, board_layers: int, locked: list[str]) -> dict:
        o = self.opts
        cfg = copy.deepcopy(DEFAULT_RULES)
        if o.rules_file:
            p = Path(o.rules_file)
            import yaml
            cfg = _merge(cfg, yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text()))
            cfg.pop("board", None)
        g = self.given
        if "layers" in g or not o.rules_file:
            cfg["layers"] = o.layers or board_layers
        if "corner_radius" in g or not o.rules_file:
            cfg["outline"]["corner_radius"] = o.corner_radius
        cfg["clearance"] = _merge(cfg["clearance"], o.clearance)
        if o.mounting_holes:
            cfg["mounting_holes"] = o.mounting_holes
        if o.edge_parts:
            cfg["edge_parts"] = o.edge_parts
        if "allow_bottom_kinds" in g or not o.rules_file:
            cfg["sides"]["allow_bottom_kinds"] = list(o.allow_bottom_kinds)
        if "bottom_penalty" in g:
            cfg["sides"]["bottom_penalty"] = o.bottom_penalty
        cfg["decoupling"] = _merge(cfg["decoupling"], o.decoupling)
        cfg["net_weights"] = _merge(cfg["net_weights"], o.net_weights)
        names = [n["name"] for n in pr.nets]
        if not cfg["decoupling"].get("gnd_nets"):
            cfg["decoupling"]["gnd_nets"] = sorted((n for n in names if _is_gnd(n)), key=len) or ["GND"]
        if cfg["net_weights"].get("power_nets") is None:
            cfg["net_weights"]["power_nets"] = [n for n in names if _is_power(n)]
        fixed = set(cfg.get("fixed_abs", [])) | set(o.fixed) | set(locked)
        fixed -= set(cfg.get("edge_parts", {})) | set(cfg.get("mounting_holes", {}).get("refs", []))
        cfg["fixed_abs"] = sorted(fixed)
        return cfg

    def auto_size(self, pr, cfg) -> tuple[float, float]:
        cl = cfg["clearance"].get("courtyard", 0.05)
        edge = cfg["clearance"].get("edge", 0.2)
        top = sum((p.w + cl) * (p.h + cl) for p in pr.parts if not p.allow_bottom)
        flex = sum((p.w + cl) * (p.h + cl) for p in pr.parts if p.allow_bottom)
        need = max(top, (top + flex) / 2) if flex else top
        area = need / self.opts.density
        W = math.sqrt(area * self.opts.aspect)
        H = area / W
        longest = max(max(p.w, p.h) for p in pr.parts) + 2 * edge
        return round(max(W, longest), 1), round(max(H, longest), 1)

    # ------------------------------------------------------------------ proposals
    def _propose(self, ctx, pr, ol, seed):
        o = self.opts
        if o.method == "sa":
            from eggpcb.sa import run_sa
            w = dict(o.weights)
            if o.compact is not None:
                w["compact"] = o.compact
            c, k, s, ea, _ = run_sa(ctx, ol, seed=seed, n_temps=o.temps, moves_per_part=o.moves, cap_frac=o.cap,
                                    weights=w, verbose=o.verbose)
            return c, k, s, ea
        if o.method in ("guided", "unguided"):
            import torch
            from eggpcb.model import Denoiser
            from eggpcb.sample import sample
            if not o.checkpoint:
                raise ValueError(f"method {o.method!r} needs `checkpoint:` (a trained denoiser .pt)")
            if not hasattr(self, "_model"):
                m = Denoiser()
                st = torch.load(o.checkpoint, map_location="cpu")
                m.load_state_dict(st["ema"])
                self._model = m.eval()
            r = sample(self._model, pr, ol, seed=seed, guided=o.method == "guided", guide_cfg=o.guide or None)
            return r["c"], r["k"], r["s"], r["ea"]
        raise ValueError(f"eggplacer method {o.method!r}: use sa, guided or unguided")

    def _fit(self, pr, cfg, best):
        """Trim the outline to the parts' bounding box + margin (not with edge parts or corner holes,
        which are tied to the outline). Relative positions don't change, so legality is kept."""
        import numpy as np
        from eggpcb.energies import Outline
        from eggpcb.geometry import box_dims, dims_rot
        if any(p.fixed in ("edge", "corner") for p in pr.parts):
            return None
        c = np.asarray(best["c"], float)
        half = dims_rot(box_dims(pr), np.asarray(best["k"])) / 2
        lo, hi = (c - half).min(0), (c + half).max(0)
        m = cfg["clearance"]["edge"] + self.opts.fit_margin
        W, H = round(float(hi[0] - lo[0] + 2 * m), 2), round(float(hi[1] - lo[1] + 2 * m), 2)
        if W >= best["W"] and H >= best["H"]:
            return None
        before = [best["W"], best["H"]]
        best["c"] = c - lo + m
        best["W"], best["H"] = min(W, best["W"]), min(H, best["H"])
        best["ol"] = Outline(best["W"], best["H"], best["ol"].r, best["ol"].edge)
        return before

    # ------------------------------------------------------------------ placer
    def place(self, board_in: Path, board_out: Path, work_dir: Path) -> dict:
        import numpy as np
        import torch
        from eggpcb.board_out import write_board
        from eggpcb.energies import Ctx, Outline, hpwl, legality_report
        from eggpcb.kicad_io import parse_board
        from eggpcb.legalize import legalize
        from eggpcb.render import render
        from eggpcb.rules import apply_rules

        pcbnew = import_pcbnew()
        b = pcbnew.LoadBoard(str(board_in))
        board_layers = b.GetCopperLayerCount()
        locked = [f.GetReference() for f in b.GetFootprints() if f.IsLocked()]
        del b
        pr = parse_board(board_in)
        refs = [p.ref for p in pr.parts]
        dups = sorted({r for r in refs if refs.count(r) > 1})
        if dups:
            raise ValueError(f"duplicate references on the board: {dups}; annotate the schematic")
        cfg = self.rules(pr, board_layers, locked)
        apply_rules(pr, cfg)
        (work_dir / "eggplacer_rules.json").write_text(json.dumps(cfg, indent=1))
        if not pr.src_outline:
            pr.src_outline = [50.0, 50.0, 50.0, 50.0]   # new board: put it on the drawing sheet
        if self.opts.width and self.opts.height:
            W, H = float(self.opts.width), float(self.opts.height)
        else:
            W, H = self.auto_size(pr, cfg)
        ctx = Ctx(pr)
        r, edge = cfg["outline"]["corner_radius"], cfg["clearance"]["edge"]
        tries = []
        best = None
        t0 = time.time()
        for _ in range(self.opts.max_tries):
            ol = Outline(W, H, r, edge)
            for seed in self.opts.seeds:
                c, k, s, ea = self._propose(ctx, pr, ol, seed)
                c2, k2, s2, ea2, li = legalize(pr, c, k, s, ol, ea, g=0.05)
                rep = legality_report(ctx, c2, k2, s2, ol)
                legal = bool(li["ok"] and rep["ok"])
                h = float(hpwl(ctx, torch.tensor(c2), torch.tensor(k2), torch.tensor(s2))) if legal else None
                tries.append(dict(W=W, H=H, seed=seed, legal=legal, hpwl=h, failed=li.get("failed", [])[:10]))
                if legal and (best is None or h < best["hpwl"]):
                    best = dict(c=c2, k=k2, s=s2, ea=ea2, hpwl=h, seed=seed, ol=ol, W=W, H=H)
            if best:
                break
            W, H = round(W * self.opts.grow, 2), round(H * self.opts.grow, 2)
        if best is None:
            raise RuntimeError(f"no legal placement up to {W / self.opts.grow:.1f} x {H / self.opts.grow:.1f} mm; "
                               "lower `density`, raise `max_tries`, or set width/height")
        place_s = time.time() - t0
        fitted = self._fit(pr, cfg, best) if self.opts.fit_outline else None
        wb = write_board(pr, best["c"], best["k"], best["s"], best["ol"], board_in, board_out)
        # the old routing and copper zones belong to the old placement
        from ..kicad import run_module
        stripped = run_module("pcb_component_placer.kicad_scripts.strip_tracks", board_out, board_out, "--zones")
        if self.opts.tidy_silk:
            stripped.update(run_module("pcb_component_placer.kicad_scripts.tidy_silk", board_out, board_out))
        np.savez(work_dir / "placement.npz", c=best["c"], k=best["k"], s=best["s"],
                 ea=np.array([[i, e] for i, e in best["ea"].items()]))
        png = work_dir / "placement.png"
        render(pr, best["c"], best["k"], best["s"], outline=best["ol"].as_dict(), path=png,
               title=f"eggplacer {self.opts.method} {best['W']:.1f}x{best['H']:.1f} mm")
        return dict(method=self.opts.method, width=best["W"], height=best["H"], seed=best["seed"],
                    hpwl_mm=round(best["hpwl"], 1), fitted_from=fitted, place_seconds=round(place_s, 1), tries=tries,
                    pad_error_mm=wb.get("pad_error_mm"), layers=cfg["layers"], fixed=cfg["fixed_abs"],
                    gnd_nets=cfg["decoupling"]["gnd_nets"], image=str(png), **stripped)
