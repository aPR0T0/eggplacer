"""pcb-placer: place a KiCad board with any registered placer.

  pcb-placer list
  pcb-placer place board.kicad_pcb -o placed.kicad_pcb [-a eggplacer] [-c options.yaml] [-s density=0.4]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

from . import get_placer, list_placers


def _options(path: str | None, sets: list[str]) -> dict:
    opts = {}
    if path:
        opts.update(yaml.safe_load(Path(path).read_text()) or {})
    for s in sets:
        k, _, v = s.partition("=")
        opts[k.strip()] = yaml.safe_load(v)
    return opts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcb-placer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="available placers and their options")
    p = sub.add_parser("place", help="place a board")
    p.add_argument("board", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-a", "--algorithm", default="eggplacer")
    p.add_argument("-c", "--options", help="YAML file with the placer's options")
    p.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE", help="one placer option")
    p.add_argument("--no-drc", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        from .base import REGISTRY
        for name, summary in list_placers().items():
            print(f"{name:12s} {summary}")
            for f in dataclasses.fields(REGISTRY[name].Options):
                default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
                print(f"    {f.name} = {default!r}")
        return 0
    res = get_placer(a.algorithm, _options(a.options, a.set)).run(a.board, a.output, drc=not a.no_drc)
    keep = ("ok", "algorithm", "board", "seconds", "footprints", "bottom_footprints", "outline_mm",
            "outside", "placement_errors")
    print(json.dumps({k: getattr(res, k) for k in keep}, indent=1))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
