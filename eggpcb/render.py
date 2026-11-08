"""PNG rendering of a placement + outline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .geometry import box_dims, dims_rot, transform_offsets  # noqa: E402

KIND_COLORS = {"U": "#4c72b0", "J": "#8c564b", "R": "#c4c4c4", "C": "#e5c07b", "L": "#55a868",
               "Q": "#937860", "LED": "#dd8452", "BH": "#555555", "Y": "#64b5cd", "D": "#da8bc3",
               "SW": "#8172b3"}


def render(pr, centers, ks, sides, outline=None, path="placement.png", title="", show_nets=True,
           show_pads=True, label_big=True, dpi=110):
    """outline: dict(x0,y0,w,h,r) in board coordinates (y down)."""
    wh = dims_rot(box_dims(pr), ks)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), sharex=True, sharey=True)
    for side_i, ax in enumerate(axs):
        if outline:
            ax.add_patch(FancyBboxPatch((outline["x0"] + outline["r"], outline["y0"] + outline["r"]),
                                        outline["w"] - 2 * outline["r"], outline["h"] - 2 * outline["r"],
                                        boxstyle=f"round,pad={outline['r']}", fill=False, ec="k", lw=1.6))
        for i, p in enumerate(pr.parts):
            on_side = sides[i] == side_i
            through = p.corner_hole or (p.holes and not on_side)
            if not on_side and not through:
                continue
            cx, cy = centers[i]; w, h = wh[i]
            col = KIND_COLORS.get(p.kind, "#999999")
            if on_side:
                ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=col, ec="k", lw=0.4, alpha=0.75))
            if p.holes:
                ho = transform_offsets(np.array([[hx, hy] for hx, hy, _ in p.holes]), ks[i], sides[i])
                for (hx, hy), (_, _, r) in zip(ho, p.holes):
                    ax.add_patch(plt.Circle((cx + hx, cy + hy), r, fill=False, ec="k", lw=0.5))
            if show_pads and on_side and len(p.pad_off):
                po = transform_offsets(np.array(p.pad_off), ks[i], sides[i])
                ax.plot(cx + po[:, 0], cy + po[:, 1], ".", ms=1.2, color="#b22222")
            if label_big and on_side and (w * h > 6 or p.fixed != "none"):
                ax.text(cx, cy, p.ref, ha="center", va="center", fontsize=5.5)
        if show_nets and side_i == 0:
            pin_part, pin_off, pin_net = pr.pin_arrays()
            pw = centers[pin_part] + transform_offsets(pin_off, ks[pin_part], sides[pin_part])
            for ni, n in enumerate(pr.nets):
                if len(n["pins"]) > 12:
                    continue
                q = pw[pin_net == ni]
                c = q.mean(0)
                for x, y in q:
                    ax.plot([c[0], x], [c[1], y], "-", lw=0.25, color="#1f77b4", alpha=0.35)
        ax.set_aspect("equal"); ax.invert_yaxis() if side_i == 0 else None
        ax.set_title(("TOP" if side_i == 0 else "BOTTOM (viewed from top)") + f"  {title}", fontsize=9)
        ax.grid(alpha=0.15)
    pts = np.concatenate([centers - wh / 2, centers + wh / 2])
    if outline:
        pts = np.concatenate([pts, [[outline["x0"], outline["y0"]], [outline["x0"] + outline["w"], outline["y0"] + outline["h"]]]])
    lo, hi = pts.min(0) - 2, pts.max(0) + 2
    axs[0].set_xlim(lo[0], hi[0]); axs[0].set_ylim(hi[1], lo[1])
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close(fig)
    return path
