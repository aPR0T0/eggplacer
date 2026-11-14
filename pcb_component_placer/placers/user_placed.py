"""user_placed: keep the placement the user drew in KiCad.

The board is copied unchanged; `Placer.run()` then checks it like any other placement (closed
outline, every part's pads on the board, no courtyard overlaps). Parts added from the schematic
since the user last placed sit outside the outline and are reported by reference.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..base import Placer, register


@register("user_placed")
class UserPlaced(Placer):
    summary = "keep the user's placement from the KiCad board (checked, not changed)"

    @dataclass
    class Options:
        keep_tracks: bool = True      # False: drop existing tracks/vias so the router starts clean

    def place(self, board_in: Path, board_out: Path, work_dir: Path) -> dict:
        if self.opts.keep_tracks:
            shutil.copyfile(board_in, board_out)
            return {"tracks": "kept"}
        from ..kicad import run_module
        return run_module("pcb_component_placer.kicad_scripts.strip_tracks", board_in, board_out)
