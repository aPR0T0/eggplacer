"""Make KiCad's `pcbnew` module importable from a uv/pip virtualenv.

KiCad installs pcbnew for the system Python (e.g. /usr/lib/python3/dist-packages on Debian/Ubuntu).
Instead of exposing all of dist-packages to the venv, a private directory with links to just
`pcbnew.py` and `_pcbnew.so` is appended to sys.path. The venv must use the same Python minor
version KiCad was built for (`uv venv --python /usr/bin/python3`). Set $KICAD_PCBNEW_DIR to the
directory holding pcbnew.py if it is somewhere else.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

_DONE = False


def _candidates():
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    if os.environ.get("KICAD_PCBNEW_DIR"):
        yield os.environ["KICAD_PCBNEW_DIR"]
    yield from ("/usr/lib/python3/dist-packages", f"/usr/lib/python{v}/site-packages",
                f"/usr/lib64/python{v}/site-packages", "/usr/local/lib/python3/dist-packages",
                f"/usr/local/lib/python{v}/site-packages")
    yield from glob.glob(f"/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/*/lib/python{v}/site-packages")
    yield from glob.glob(r"C:\Program Files\KiCad\*\bin\Lib\site-packages")


def ensure_pcbnew() -> None:
    """Idempotent; raises ImportError with instructions if pcbnew cannot be found."""
    global _DONE
    if _DONE or importlib.util.find_spec("pcbnew") is not None:
        _DONE = True
        return
    for d in _candidates():
        src = Path(d) / "pcbnew.py"
        if not src.exists():
            continue
        shim = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pcb_pipeline" / \
            f"pcbnew-py{sys.version_info.major}{sys.version_info.minor}"
        shim.mkdir(parents=True, exist_ok=True)
        for f in [src, *Path(d).glob("_pcbnew*")]:
            link = shim / f.name
            target = f.resolve()
            if link.is_symlink() and link.resolve() == target:
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
        if str(shim) not in sys.path:
            sys.path.append(str(shim))
        _DONE = True
        return
    raise ImportError("KiCad's pcbnew module was not found. Install KiCad (10.x), create the venv with "
                      "KiCad's Python (`uv venv --python /usr/bin/python3`), or set KICAD_PCBNEW_DIR.")


def import_pcbnew():
    ensure_pcbnew()
    import pcbnew
    return pcbnew
