"""Draw footprints with their derived keep-out boxes (green dashed), pads (red), holes, fab (grey),
silk (orange), courtyard (magenta) and Dwgs.User lines/text (blue, e.g. "PCB Edge" markers).
Use it to check the parsed boxes and to decide each edge connector's mouth direction.

usage: python scripts/plot_footprints.py out/problem.json J1,J2,J3 out/footprints.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggpcb.kicad_io import NM, Problem, _shape_points, world_to_local  # noqa: E402


def main(problem, refs, out_png):
    import pcbnew
    pr = Problem.from_json(problem)
    b = pcbnew.LoadBoard(pr.rules["board_abs"])
    refs = refs.split(",")
    fig, axs = plt.subplots(1, len(refs), figsize=(5 * len(refs), 5))
    col = {pcbnew.F_Fab: "gray", pcbnew.F_CrtYd: "magenta", pcbnew.F_SilkS: "orange", pcbnew.Dwgs_User: "blue",
           pcbnew.Cmts_User: "cyan"}
    for ax, ref in zip(np.atleast_1d(axs), refs):
        fp = [f for f in b.GetFootprints() if f.GetReference() == ref][0]
        p = pr.parts[pr.index(ref)]
        pos, deg, bot = np.array(p.pos0), p.rot0, bool(p.side0)
        for g in fp.GraphicalItems():
            if g.GetClass() in ("PCB_TEXT", "PCB_FIELD"):
                if g.GetLayer() == pcbnew.Dwgs_User:
                    q = world_to_local(np.array([g.GetPosition().x * NM, g.GetPosition().y * NM]), pos, deg, bot)
                    ax.text(q[0], q[1], g.GetText(), color="blue", fontsize=7)
                continue
            if g.GetLayer() in col:
                pts = np.array([world_to_local(np.array(q), pos, deg, bot) for q in _shape_points(g, pcbnew)])
                ax.plot(pts[:, 0], pts[:, 1], "-", color=col[g.GetLayer()], lw=1)
        for o, s in zip(p.pad_off, p.pad_size):
            x, y = np.array(o) + p.box_off
            ax.add_patch(plt.Rectangle((x - s[0] / 2, y - s[1] / 2), s[0], s[1], color="red", alpha=0.5))
        for hx, hy, r in p.holes:
            ax.add_patch(plt.Circle((hx + p.box_off[0], hy + p.box_off[1]), r, fill=False, color="k"))
        bx, by = p.box_off
        ax.add_patch(plt.Rectangle((bx - p.w / 2, by - p.h / 2), p.w, p.h, fill=False, color="green", lw=2, ls="--"))
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.3); ax.autoscale()
        ax.set_title(f"{ref} {p.lib[:28]}\nbox {p.w:.2f}x{p.h:.2f}  courtyard_ok={p.courtyard_ok}", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=90)
    print(out_png)


if __name__ == "__main__":
    main(*sys.argv[1:4])
