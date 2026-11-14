"""Placement check: outline present, every footprint's copper pads inside it.

usage: python -m pcb_component_placer.kicad_scripts.check_placement board.kicad_pcb -> @@{json}
"""
import json
import sys

from pcb_component_placer.kicad_env import import_pcbnew

pcbnew = import_pcbnew()
MM = 1e6
b = pcbnew.LoadBoard(sys.argv[1])
poly = pcbnew.SHAPE_POLY_SET()
has_outline = any(d.GetLayer() == pcbnew.Edge_Cuts for d in b.GetDrawings())


def outlines(board, poly):
    try:
        return board.GetBoardPolygonOutlines(poly, False)   # KiCad 10
    except TypeError:
        return board.GetBoardPolygonOutlines(poly)


closed = bool(has_outline and outlines(b, poly) and poly.OutlineCount() > 0)
outside = []
for fp in b.GetFootprints():
    pads = [p for p in fp.Pads() if p.IsOnCopperLayer()]
    if not pads:
        continue
    if not closed:
        outside.append(fp.GetReference())
        continue
    for p in pads:
        bb = p.GetBoundingBox()
        corners = [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()),
                   (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())]
        if not all(poly.Contains(pcbnew.VECTOR2I(int(x), int(y))) for x, y in corners):
            outside.append(fp.GetReference())
            break
bb = poly.BBox() if closed else b.GetBoardEdgesBoundingBox()   # without the Edge.Cuts line width
fps = list(b.GetFootprints())
print("@@" + json.dumps(dict(
    outline=closed,
    outline_mm=[round(bb.GetWidth() / MM, 3), round(bb.GetHeight() / MM, 3)] if has_outline else None,
    footprints=len(fps), bottom=sum(1 for f in fps if f.IsFlipped()),
    copper_layers=b.GetCopperLayerCount(), outside=sorted(outside))))
