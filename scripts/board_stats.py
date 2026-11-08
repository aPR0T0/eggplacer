"""Phase 1: parse the board, apply rules, print statistics and the area lower bound.

usage: python scripts/board_stats.py configs/myboard.json out/
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggpcb.kicad_io import parse_board  # noqa: E402
from eggpcb.rules import apply_rules, decap_pairs, load_cfg  # noqa: E402


def main(cfg_path, out_dir):
    cfg = load_cfg(cfg_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    board = (Path(cfg_path).parent / cfg["board"]).resolve()
    pr = apply_rules(parse_board(board), cfg)
    # absolute paths so every later step finds the source board and its 2-layer DRC rules
    pr.rules["board_abs"] = str(board)
    pr.rules["config_abs"] = str(Path(cfg_path).resolve())
    if cfg.get("dru"):
        pr.rules["dru_abs"] = str((Path(cfg_path).parent / cfg["dru"]).resolve())
    pr.to_json(out / "problem.json")

    S = {}
    S["n_parts"] = len(pr.parts)
    S["n_nets"] = len(pr.nets)
    S["n_pins"] = int(sum(len(n["pins"]) for n in pr.nets))
    kinds = collections.defaultdict(list)
    for p in pr.parts:
        kinds[p.kind].append(p)
    S["by_kind"] = {k: dict(n=len(v), area=round(sum(p.area for p in v), 1),
                            mean_w=round(float(np.mean([max(p.w, p.h) for p in v])), 2),
                            mean_h=round(float(np.mean([min(p.w, p.h) for p in v])), 2),
                            pins=int(sum(sum(1 for n in p.pad_net if n >= 0) for p in v)))
                    for k, v in sorted(kinds.items(), key=lambda kv: -sum(p.area for p in kv[1]))}
    areas = np.array([p.area for p in pr.parts])
    S["box_area_total"] = round(float(areas.sum()), 1)
    S["area_quantiles"] = {q: round(float(np.quantile(areas, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    pins_per_part = np.array([sum(1 for n in p.pad_net if n >= 0) for p in pr.parts])
    S["pins_per_part_hist"] = dict(collections.Counter(int(min(x, 40)) for x in pins_per_part))
    fan = [len(n["pins"]) for n in pr.nets]
    S["net_fanout_hist"] = dict(sorted(collections.Counter(int(min(x, 30)) for x in fan).items()))
    S["fixed"] = {p.ref: p.fixed for p in pr.parts if p.fixed != "none"}
    S["edge_parts"] = {p.ref: dict(mouth=p.edge_mouth, overhang=round(p.edge_overhang, 3), box=[round(p.w, 2), round(p.h, 2)])
                       for p in pr.parts if p.fixed == "edge"}
    S["sheets"] = dict(collections.Counter(p.sheet for p in pr.parts))
    dps = decap_pairs(pr, cfg)
    S["decap_pairs"] = len(dps)

    # ---------------- area lower bound ----------------
    cl = cfg["clearance"]["courtyard"]
    def eff(p):  # in-board area incl. half the courtyard gap, minus edge overhang
        a = (p.w + cl) * (p.h + cl)
        if p.fixed == "edge":
            a -= p.edge_overhang * (p.h if p.edge_mouth in ("+x", "-x") else p.w)
        return a
    holes = [p for p in pr.parts if p.corner_hole]
    gap = cfg["mounting_holes"].get("edge_gap", 0.2)
    a_holes = sum((p.w + gap) * (p.h + gap) for p in holes)          # corners occupy both sides
    a_top_only = sum(eff(p) for p in pr.parts if not p.allow_bottom and not p.corner_hole)
    a_flex = sum(eff(p) for p in pr.parts if p.allow_bottom)
    a_hole_kos = sum((b[2] - b[0]) * (b[3] - b[1]) for p in pr.parts if not p.corner_hole for b in p.bottom_hole_keepout)
    # both sides: the top must take every top-only part; flexible parts can fill the bottom.
    need_top = a_top_only + a_holes
    need_bot = a_holes + a_hole_kos
    two_side = max(need_top, (need_top + need_bot + a_flex) / 2)
    one_side = need_top + a_flex
    S["area_terms"] = dict(top_only=round(a_top_only, 1), bottom_allowed=round(a_flex, 1),
                           corner_holes=round(a_holes, 1), through_hole_keepouts=round(a_hole_kos, 1))
    lb = {}
    # packing density = sum(courtyard area) / board area that a 2-layer, fine-pitch board can reach.
    for rho, label in ((1.0, "absolute (rho=1.0)"), (0.80, "optimistic (rho=0.80)"),
                       (0.65, "realistic 2L dense (rho=0.65)"), (0.50, "comfortable (rho=0.50)")):
        lb[label] = dict(two_sided=round(two_side / rho, 1), one_sided=round(one_side / rho, 1),
                         square_side_two_sided=round(float(np.sqrt(two_side / rho)), 2))
    S["area_lower_bound_mm2"] = lb
    # the outline must also contain the largest parts side by side with the fixed edge parts
    big = sorted(pr.parts, key=lambda p: -p.area)[:4]
    S["largest_parts"] = [(p.ref, round(p.w, 2), round(p.h, 2)) for p in big]
    min_side = max(max(min(p.w, p.h) for p in pr.parts), 0) + 2 * (holes[0].w + gap if holes else 0)
    S["min_side_from_holes_and_biggest_part"] = round(min_side, 2)
    (out / "board_stats.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))
    return pr, S


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out")
