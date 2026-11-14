"""Placer interface and registry.

Every placer is a subclass of `Placer` in `pcb_component_placer/placers/<name>.py`, registered with
`@register("<name>")`. It declares its YAML options as a dataclass `Options` and implements
`place(board_in, board_out, work_dir)`. `Placer.run()` adds what every placer gets the same way:
sidecar files and the placement check (outline, parts on the board, courtyard overlaps via DRC).
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import pkgutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import kicad

REGISTRY: dict[str, type["Placer"]] = {}
# DRC types that make a placement unusable (routing problems are the router's job)
PLACEMENT_ERRORS = ("courtyards_overlap", "malformed_courtyard", "invalid_outline", "pth_inside_courtyard",
                    "npth_inside_courtyard", "copper_edge_clearance")


def register(name: str):
    """Class decorator: make a Placer available as `algorithm: <name>`."""
    def deco(cls):
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"placer {name!r} registered twice")
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def _load_builtin():
    from . import placers
    for m in pkgutil.iter_modules(placers.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{placers.__name__}.{m.name}")


def list_placers() -> dict[str, str]:
    _load_builtin()
    return {n: c.summary for n, c in sorted(REGISTRY.items())}


def get_placer(name: str, options: dict | None = None) -> "Placer":
    _load_builtin()
    if name not in REGISTRY:
        raise KeyError(f"unknown placer {name!r}; available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](options)


def parse_options(cls, options: dict | None):
    """YAML mapping -> the placer's Options dataclass; unknown keys are an error."""
    options = dict(options or {})
    names = {f.name for f in dataclasses.fields(cls)}
    bad = sorted(set(options) - names)
    if bad:
        raise ValueError(f"unknown option(s) {bad} for {cls.__qualname__}; valid: {sorted(names)}")
    return cls(**options)


@dataclass
class PlaceResult:
    ok: bool                      # outline closed, every part on the board, no placement DRC errors
    algorithm: str
    board: str
    seconds: float
    footprints: int = 0
    bottom_footprints: int = 0
    outline_mm: list | None = None
    outside: list = field(default_factory=list)       # refs with pads off the board
    placement_errors: dict = field(default_factory=dict)
    drc: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)        # placer-specific

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=1, default=str))


class Placer:
    name: str = ""
    summary: str = ""
    # run place() in a child process: set for placers that load and save boards with pcbnew in-process
    isolate: bool = False

    @dataclass
    class Options:
        pass

    def __init__(self, options: dict | None = None):
        self.options = dict(options or {})
        self.opts = parse_options(self.Options, options)

    # ---- implement this -------------------------------------------------------------------
    def place(self, board_in: Path, board_out: Path, work_dir: Path) -> dict[str, Any]:
        """Place every footprint of `board_in` (synced from the schematic: footprints + nets) and
        save the result, with a closed Edge.Cuts outline, as `board_out`. Never modify `board_in`.
        Return placer-specific details (shown in the report)."""
        raise NotImplementedError

    # ---- common driver --------------------------------------------------------------------
    def run(self, board_in, board_out, work_dir=None, drc: bool = True) -> PlaceResult:
        board_in, board_out = Path(board_in).resolve(), Path(board_out).resolve()
        if board_in == board_out:
            raise ValueError("refusing to overwrite the input board")
        if not board_in.exists():
            raise FileNotFoundError(board_in)
        work_dir = Path(work_dir or board_out.parent / f"{board_out.stem}_work").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        board_out.parent.mkdir(parents=True, exist_ok=True)
        kicad.copy_sidecars(board_in, board_out)
        t0 = time.time()
        if self.isolate:
            job = work_dir / f"{self.name}_job.json"
            job.write_text(json.dumps(dict(algorithm=self.name, options=self.options, board_in=str(board_in),
                                           board_out=str(board_out), work_dir=str(work_dir)), default=str))
            details = kicad.run_module("pcb_component_placer._child", job, timeout=24 * 3600)
        else:
            details = self.place(board_in, board_out, work_dir) or {}
        seconds = time.time() - t0
        if not board_out.exists():
            raise RuntimeError(f"placer {self.name!r} did not write {board_out}")
        kicad.copy_sidecars(board_in, board_out)
        chk = kicad.run_module("pcb_component_placer.kicad_scripts.check_placement", board_out)
        res = PlaceResult(ok=False, algorithm=self.name, board=str(board_out), seconds=round(seconds, 1),
                          footprints=chk["footprints"], bottom_footprints=chk["bottom"],
                          outline_mm=chk["outline_mm"], outside=chk["outside"], details=details)
        ok = chk["outline"] and not chk["outside"]
        if drc:
            d = kicad.run_drc(board_out)
            res.drc = d
            res.placement_errors = {k: v for k, v in d["by_type"].items() if k in PLACEMENT_ERRORS}
            ok = ok and not res.placement_errors
        res.ok = bool(ok)
        res.to_json(board_out.with_suffix(".place.json"))
        return res
