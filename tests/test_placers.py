"""Placer registry and options (no KiCad needed), plus eggplacer / user_placed on a small generated
board when KiCad is installed."""
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from pcb_component_placer import Placer, get_placer, list_placers, place, register

LIB = Path("/usr/share/kicad/footprints/Resistor_SMD.pretty")


def test_builtin_placers_registered():
    names = list_placers()
    assert "eggplacer" in names and "user_placed" in names


def test_unknown_option_rejected():
    with pytest.raises(ValueError, match="unknown option"):
        get_placer("eggplacer", {"densty": 0.3})


def test_register_custom_placer():
    @register("copy_test")
    class Copy(Placer):
        summary = "copies the board"

        @dataclass
        class Options:
            note: str = ""

        def place(self, board_in, board_out, work_dir):
            shutil.copyfile(board_in, board_out)
            return {}

    assert get_placer("copy_test", {"note": "x"}).opts.note == "x"


BOARD = r'''
import sys
sys.path.insert(0, %(root)r)
from pcb_component_placer.kicad_env import import_pcbnew
pcbnew = import_pcbnew()
MM = 1e6
b = pcbnew.NewBoard(sys.argv[1])
def seg(a, c):
    s = pcbnew.PCB_SHAPE(b); s.SetShape(pcbnew.SHAPE_T_SEGMENT); s.SetLayer(pcbnew.Edge_Cuts)
    s.SetStart(pcbnew.VECTOR2I(int(a[0]*MM), int(a[1]*MM))); s.SetEnd(pcbnew.VECTOR2I(int(c[0]*MM), int(c[1]*MM)))
    s.SetWidth(int(0.05*MM)); b.Add(s)
P = [(100, 100), (130, 100), (130, 120), (100, 120)]
for i in range(4):
    seg(P[i], P[(i + 1) %% 4])
nets = {n: pcbnew.NETINFO_ITEM(b, n) for n in ("A", "B", "C", "GND")}
for n in nets.values():
    b.Add(n)
parts = [("R1", "A", "B"), ("R2", "B", "C"), ("R3", "C", "GND"), ("R4", "A", "GND")]
for i, (ref, n1, n2) in enumerate(parts):
    fp = pcbnew.FootprintLoad(%(lib)r, "R_0603_1608Metric"); fp.SetReference(ref)
    fp.SetPosition(pcbnew.VECTOR2I(int((105 + 6 * i) * MM), int(110 * MM))); b.Add(fp)
    for pad, n in zip(sorted(fp.Pads(), key=lambda p: p.GetNumber()), (n1, n2)):
        pad.SetNet(nets[n])
pcbnew.SaveBoard(sys.argv[1], b)
'''


def _board(tmp_path):
    pcb = tmp_path / "small.kicad_pcb"
    code = BOARD % dict(root=str(Path(__file__).resolve().parents[1]), lib=str(LIB))
    subprocess.run([sys.executable, "-c", code, str(pcb)], check=True, capture_output=True)
    return pcb


kicad = pytest.mark.skipif(not LIB.exists() or shutil.which("kicad-cli") is None, reason="KiCad not installed")


@kicad
def test_user_placed_keeps_placement(tmp_path):
    res = place(_board(tmp_path), tmp_path / "out" / "small.kicad_pcb", algorithm="user_placed")
    assert res.ok and res.footprints == 4 and res.outline_mm == [30.0, 20.0]


@kicad
def test_eggplacer_places_legally(tmp_path):
    res = place(_board(tmp_path), tmp_path / "out" / "small.kicad_pcb", algorithm="eggplacer",
                options={"temps": 60, "moves": 20})
    assert res.ok, (res.outside, res.placement_errors)
    assert res.footprints == 4 and not res.outside
    assert res.details["width"] < 30 and res.details["height"] < 30   # fitted to the parts
