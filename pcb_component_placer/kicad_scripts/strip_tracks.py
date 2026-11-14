"""Remove unlocked tracks and vias (and with --zones, every copper zone) from a board.

usage: python -m pcb_component_placer.kicad_scripts.strip_tracks in.kicad_pcb out.kicad_pcb [--zones]
"""
import json
import sys

from pcb_component_placer.kicad_env import import_pcbnew

pcbnew = import_pcbnew()
b = pcbnew.LoadBoard(sys.argv[1])
tracks = [t for t in b.GetTracks() if not t.IsLocked()]
zones = []
if "--zones" in sys.argv[3:]:
    zones = [z for z in (b.GetArea(i) for i in range(b.GetAreaCount())) if not z.GetIsRuleArea()]
# keep the removed items referenced until the board is saved (SWIG may free them otherwise)
for item in tracks + zones:
    b.Remove(item)
pcbnew.SaveBoard(sys.argv[2], b)
print("@@" + json.dumps(dict(tracks_removed=len(tracks), zones_removed=len(zones))))
