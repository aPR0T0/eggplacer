"""Move every visible reference designator on the silkscreen to a spot where it overlaps no pad,
no other silkscreen and stays on the board: its current spot, then above / below / left / right of
the part, then the part's centre. A label with no free spot is hidden (the fab-layer copy stays).

usage: python -m pcb_component_placer.kicad_scripts.tidy_silk in.kicad_pcb out.kicad_pcb -> @@{json}
"""
import json
import sys

from pcb_component_placer.kicad_env import import_pcbnew

pcbnew = import_pcbnew()
MM = 1e6
GAP = int(0.15 * MM)
b = pcbnew.LoadBoard(sys.argv[1])
SILK = {pcbnew.F_SilkS, pcbnew.B_SilkS}


def rect(bb, grow=0):
    return (bb.GetLeft() - grow, bb.GetTop() - grow, bb.GetRight() + grow, bb.GetBottom() + grow)


def hit(a, c):
    return a[0] < c[2] and c[0] < a[2] and a[1] < c[3] and c[1] < a[3]


edge = b.GetBoardEdgesBoundingBox()
board = rect(edge, -int(0.3 * MM))
fps = list(b.GetFootprints())
# fixed obstacles: copper pads (both sides matter for silk on that side) and footprint silk graphics
pads = {fp.GetReference(): [rect(p.GetBoundingBox(), GAP) for p in fp.Pads() if p.IsOnCopperLayer()] for fp in fps}
silk = {}
for fp in fps:
    silk[fp.GetReference()] = [(g.GetLayer(), rect(g.GetBoundingBox(), GAP)) for g in fp.GraphicalItems()
                               if g.GetLayer() in SILK and g.GetClass() not in ("PCB_TEXT", "PCB_FIELD")]
placed = []   # (layer, rect) of labels already settled
moved = hidden = kept = 0


def free(layer, r, own):
    if not (board[0] <= r[0] and board[1] <= r[1] and r[2] <= board[2] and r[3] <= board[3]):
        return False
    for ref, rs in pads.items():
        if any(hit(r, q) for q in rs):
            return False
    for ref, rs in silk.items():
        if ref != own and any(lyr == layer and hit(r, q) for lyr, q in rs):
            return False
    return not any(lyr == layer and hit(r, q) for lyr, q in placed)


for fp in sorted(fps, key=lambda f: -f.GetBoundingBox(False).GetArea()):
    t = fp.Reference()
    if not t.IsVisible() or t.GetLayer() not in SILK:
        continue
    ref = fp.GetReference()
    layer = t.GetLayer()
    if free(layer, rect(t.GetBoundingBox()), ref):
        placed.append((layer, rect(t.GetBoundingBox())))
        kept += 1
        continue
    if hasattr(t, "SetTextAngleDegrees"):
        t.SetTextAngleDegrees(0)
    body = fp.GetBoundingBox(False)
    tb = t.GetBoundingBox()
    tw, th = tb.GetWidth(), tb.GetHeight()
    cx, cy = body.GetCenter().x, body.GetCenter().y
    cands = [(cx, body.GetTop() - GAP - th // 2), (cx, body.GetBottom() + GAP + th // 2),
             (body.GetLeft() - GAP - tw // 2, cy), (body.GetRight() + GAP + tw // 2, cy), (cx, cy)]
    done = False
    for x, y in cands:
        t.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
        r = rect(t.GetBoundingBox())
        if free(layer, r, ref):
            placed.append((layer, r))
            moved += 1
            done = True
            break
    if not done:
        t.SetVisible(False)
        hidden += 1
pcbnew.SaveBoard(sys.argv[2], b)
print("@@" + json.dumps(dict(labels_kept=kept, labels_moved=moved, labels_hidden=hidden)))
