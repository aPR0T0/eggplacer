"""KiCad helpers shared by every placer: kicad-cli DRC, pcbnew child processes, sidecar files."""
from __future__ import annotations

import functools
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIDECARS = (".kicad_pro", ".kicad_dru")


@functools.lru_cache(maxsize=None)
def kicad_cli() -> str:
    for c in (os.environ.get("KICAD_CLI"), shutil.which("kicad-cli"),
              "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"):
        if c and Path(c).exists():
            return c
    hits = sorted(glob.glob(r"C:\Program Files\KiCad\*\bin\kicad-cli.exe"), reverse=True)
    if hits:
        return hits[0]
    raise RuntimeError("kicad-cli not found; install KiCad or set KICAD_CLI")


def run_module(module: str, *args, timeout: float = 1800) -> dict:
    """Run `python -m <module> args` in a child (pcbnew can't reliably reload a board it saved in
    the same process); the module prints one '@@{json}' line."""
    r = subprocess.run([sys.executable, "-m", module, *map(str, args)], capture_output=True, text=True,
                       timeout=timeout)
    out = [ln for ln in r.stdout.splitlines() if ln.startswith("@@")]
    if r.returncode != 0 or not out:
        raise RuntimeError(f"{module} failed (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return json.loads(out[-1][2:])


def copy_sidecars(src_pcb: Path, dst_pcb: Path) -> None:
    for suf in SIDECARS:
        s, d = Path(src_pcb).with_suffix(suf), Path(dst_pcb).with_suffix(suf)
        if s.exists() and s.resolve() != d.resolve():
            shutil.copyfile(s, d)


def run_drc(pcb: Path, refill: bool = False, timeout: float = 1800) -> dict:
    """kicad-cli DRC -> {errors, warnings, unconnected, by_type, report}."""
    pcb = Path(pcb)
    rep = pcb.with_suffix(".drc.json")
    cmd = [kicad_cli(), "pcb", "drc", "--format", "json", "--severity-all", "--units", "mm", "-o", str(rep)]
    if refill:
        cmd.append("--refill-zones")
    r = subprocess.run(cmd + [str(pcb)], capture_output=True, text=True, timeout=timeout)
    if not rep.exists():
        raise RuntimeError(f"kicad-cli drc failed on {pcb}:\n{r.stdout}\n{r.stderr}")
    d = json.loads(rep.read_text())
    by_type: dict[str, int] = {}
    err = warn = 0
    for v in d.get("violations", []):
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        err += v.get("severity") == "error"
        warn += v.get("severity") == "warning"
    return dict(errors=int(err), warnings=int(warn), unconnected=len(d.get("unconnected_items", [])),
                by_type=by_type, report=str(rep))
