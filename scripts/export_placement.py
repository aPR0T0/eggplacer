"""Write a saved placement (placement.npz from run_sa.py / size_search) as a NEW KiCad board next to
the source board (so its footprint/symbol library tables resolve), with the project file and the
2-layer DRU, then run DRC.

usage: python scripts/export_placement.py out/sa_58x58_s0/placement.npz --W 58 --H 58 \
           --name myboard_sa_58x58
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eggpcb.board_out import run_drc, write_board  # noqa: E402
from eggpcb.energies import Ctx, Outline, legality_report  # noqa: E402
from eggpcb.kicad_io import Problem  # noqa: E402
from eggpcb.rules import board_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("placement"); ap.add_argument("--W", type=float, required=True); ap.add_argument("--H", type=float, required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--problem", default=str(ROOT / "out" / "problem.json"))
    a = ap.parse_args()
    pr = Problem.from_json(a.problem)
    src, dru, _ = board_paths(pr)
    d = np.load(a.placement)
    ol = Outline(a.W, a.H, pr.rules["outline"]["corner_radius"], pr.rules["clearance"]["edge"])
    rep = legality_report(Ctx(pr), d["c"], d["k"], d["s"], ol)
    assert rep["ok"], "placement is not legal"
    out = src.parent / f"{a.name}.kicad_pcb"
    wb = write_board(pr, d["c"], d["k"], d["s"], ol, src, out, dru_text=dru)
    res = dict(board=wb, drc=run_drc(out))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
