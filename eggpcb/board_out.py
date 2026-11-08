"""Write a placement + new outline to a NEW .kicad_pcb (never the source), and run KiCad DRC."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .geometry import transform_offsets
from .kicad_io import NM, Problem

MM = 1e6


def anchor_translation(pr: Problem, c, k, s):
    """translation that puts the absolutely-fixed part(s) back on their locked KiCad coordinates."""
    for i, p in enumerate(pr.parts):
        if p.fixed == "abs":
            off = transform_offsets(np.array(p.box_off), np.array(k[i]), np.array(int(s[i] > 0.5)))
            origin = np.asarray(c[i]) - off
            return np.array(p.pos0) - origin
    # nothing fixed: keep the board near the source outline
    return np.array(pr.src_outline[:2]) if pr.src_outline else np.zeros(2)


def write_board(pr: Problem, c, k, s, ol, src_pcb, out_pcb, dru_text=None, verify=True):
    import pcbnew

    src_pcb = Path(src_pcb).resolve(); out_pcb = Path(out_pcb).resolve()
    assert src_pcb != out_pcb, "refusing to overwrite the source board"
    c = np.asarray(c, float); k = np.asarray(k, int); s = (np.asarray(s) > 0.5).astype(int)
    T = anchor_translation(pr, c, k, s)
    board = pcbnew.LoadBoard(str(src_pcb))
    idx = {p.ref: i for i, p in enumerate(pr.parts)}
    for fp in board.GetFootprints():
        i = idx[fp.GetReference()]
        p = pr.parts[i]
        want_bottom = bool(s[i])
        if fp.IsFlipped() != want_bottom:
            fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        fp.SetOrientationDegrees(float(90 * (k[i] % 4)))
        off = transform_offsets(np.array(p.box_off), np.array(k[i]), np.array(s[i]))
        org = c[i] - off + T
        locked = fp.IsLocked()
        fp.SetLocked(False)
        fp.SetPosition(pcbnew.VECTOR2I(int(round(org[0] * MM)), int(round(org[1] * MM))))
        fp.SetLocked(locked)
    # drop the old outline and group labels
    for d in list(board.GetDrawings()):
        if d.GetLayer() == pcbnew.Edge_Cuts or (d.GetLayer() == pcbnew.Cmts_User and d.GetClass() == "PCB_TEXT"):
            board.Remove(d)
    x0, y0 = T; W, H, r = ol.W, ol.H, ol.r

    def pt(x, y):
        return pcbnew.VECTOR2I(int(round((x0 + x) * MM)), int(round((y0 + y) * MM)))

    def seg(a, b):
        sh = pcbnew.PCB_SHAPE(board); sh.SetShape(pcbnew.SHAPE_T_SEGMENT)
        sh.SetStart(pt(*a)); sh.SetEnd(pt(*b)); sh.SetLayer(pcbnew.Edge_Cuts); sh.SetWidth(int(0.05 * MM))
        board.Add(sh)

    def arc(cx, cy, a0):
        import math
        sh = pcbnew.PCB_SHAPE(board); sh.SetShape(pcbnew.SHAPE_T_ARC)
        st = (cx + r * math.cos(a0), cy + r * math.sin(a0))
        md = (cx + r * math.cos(a0 + math.pi / 4), cy + r * math.sin(a0 + math.pi / 4))
        en = (cx + r * math.cos(a0 + math.pi / 2), cy + r * math.sin(a0 + math.pi / 2))
        sh.SetArcGeometry(pt(*st), pt(*md), pt(*en)); sh.SetLayer(pcbnew.Edge_Cuts); sh.SetWidth(int(0.05 * MM))
        board.Add(sh)

    import math
    if r > 0:
        seg((r, 0), (W - r, 0)); seg((W, r), (W, H - r)); seg((W - r, H), (r, H)); seg((0, H - r), (0, r))
        arc(W - r, r, -math.pi / 2); arc(W - r, H - r, 0); arc(r, H - r, math.pi / 2); arc(r, r, math.pi)
    else:
        seg((0, 0), (W, 0)); seg((W, 0), (W, H)); seg((W, H), (0, H)); seg((0, H), (0, 0))
    board.SetCopperLayerCount(int(pr.layers))
    out_pcb.parent.mkdir(parents=True, exist_ok=True)
    pcbnew.SaveBoard(str(out_pcb), board)
    # project file (netclasses) + 2-layer DRU next to the new board
    pro = src_pcb.with_suffix(".kicad_pro")
    if pro.exists():
        shutil.copyfile(pro, out_pcb.with_suffix(".kicad_pro"))
    if dru_text is not None:
        out_pcb.with_suffix(".kicad_dru").write_text(dru_text)
    err = None
    if verify:
        err = verify_pads_subprocess(pr, c, k, s, T, out_pcb)
    return dict(path=str(out_pcb), translation=T.tolist(), pad_error_mm=err,
                outline=dict(x0=float(x0), y0=float(y0), w=W, h=H, r=r))


def verify_pads(pr: Problem, c, k, s, T, pcb_path):
    """max |pad position in the saved file - model prediction| over all copper pads (mm)."""
    import pcbnew
    b = pcbnew.LoadBoard(str(pcb_path))
    idx = {p.ref: i for i, p in enumerate(pr.parts)}
    worst = 0.0
    for fp in b.GetFootprints():
        i = idx[fp.GetReference()]
        p = pr.parts[i]
        names = [pd.GetNumber() for pd in fp.Pads() if pd.IsOnCopperLayer()]
        pos = [np.array([pd.GetPosition().x * NM, pd.GetPosition().y * NM]) for pd in fp.Pads() if pd.IsOnCopperLayer()]
        pred = c[i] + T + transform_offsets(np.array(p.pad_off), np.full(len(p.pad_off), k[i]), np.full(len(p.pad_off), s[i]))
        for q, pp in zip(pos, pred):
            worst = max(worst, float(np.abs(q - pp).max()))
        if fp.IsFlipped() != bool(s[i]):
            worst = max(worst, 1e9)
    return worst


def verify_pads_subprocess(pr: Problem, c, k, s, T, pcb_path):
    """pcbnew can't reliably re-load a board it just saved in the same process: verify in a child."""
    import sys
    with tempfile.TemporaryDirectory() as td:
        prob = Path(td) / "p.json"; st = Path(td) / "s.npz"
        pr.to_json(prob)
        np.savez(st, c=c, k=k, s=s, T=T)
        code = ("import sys,numpy as np; sys.path.insert(0,%r);"
                "from eggpcb.kicad_io import Problem; from eggpcb.board_out import verify_pads;"
                "d=np.load(%r); print('PADERR', verify_pads(Problem.from_json(%r), d['c'], d['k'], d['s'], d['T'], %r))"
                % (str(Path(__file__).resolve().parents[1]), str(st), str(prob), str(pcb_path)))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("PADERR"):
                return float(line.split()[1])
    return None


def run_drc(pcb_path, timeout=600):
    """kicad-cli DRC -> dict(errors, warnings, unconnected, by_type, report)."""
    pcb_path = Path(pcb_path)
    rep = pcb_path.with_suffix(".drc.json")
    cmd = ["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all", "--units", "mm", "-o", str(rep), str(pcb_path)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    d = json.loads(rep.read_text())
    by_type = {}
    err = warn = 0
    for v in d.get("violations", []):
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        if v.get("severity") == "error":
            err += 1
        elif v.get("severity") == "warning":
            warn += 1
    unc = len(d.get("unconnected_items", []))
    return dict(errors=err, warnings=warn, unconnected=unc, by_type=by_type, report=str(rep),
                n_violations=len(d.get("violations", [])))
