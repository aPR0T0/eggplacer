"""Freerouting round trip: placed .kicad_pcb -> DSN -> Freerouting (headless) -> SES -> routed
.kicad_pcb -> kicad-cli DRC (zones refilled) -> metrics.

Every pcbnew step runs in a fresh child process (pcbnew can't reliably reload a board it saved).

CLI:  python -m eggpcb.route placed.kicad_pcb --out routed.kicad_pcb [--gnd plane_b] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = os.environ.get("EGG_JAVA", str(Path.home() / "tools/jdk/bin/java"))
FR_JAR = os.environ.get("EGG_FREEROUTING", str(Path.home() / "tools/fr.jar"))
FR19_JAR = os.environ.get("EGG_FREEROUTING19", str(Path.home() / "tools/fr19.jar"))
XVFB = os.environ.get("EGG_XVFB", str(Path.home() / "tools/xvfb/root/usr/bin/Xvfb"))
XDISPLAY = os.environ.get("EGG_DISPLAY", ":97")
PY = sys.executable


def _child(code: str, timeout=600):
    r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, %r)\n" % str(ROOT) + code],
                       capture_output=True, text=True, timeout=timeout)
    out = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    if r.returncode != 0 or not out:
        raise RuntimeError("child failed:\n" + r.stdout[-2000:] + r.stderr[-3000:])
    return json.loads(out[-1][2:])


def set_netclasses(pro_path, overrides: dict):
    """overrides: {classname: {track_width, clearance, ...}} applied to a *copied* .kicad_pro."""
    d = json.loads(Path(pro_path).read_text())
    for c in d["net_settings"]["classes"]:
        if c["name"] in overrides:
            c.update(overrides[c["name"]])
    Path(pro_path).write_text(json.dumps(d, indent=2))


PREP_CODE = r'''
import pcbnew, json
MM = 1e6
b = pcbnew.LoadBoard(%(src)r)
gnd_mode = %(gnd)r
made = 0
if gnd_mode in ("plane_b", "plane_both"):
    bb = b.GetBoardEdgesBoundingBox()
    net = b.FindNet(%(net)r)
    layers = [pcbnew.B_Cu] + ([pcbnew.F_Cu] if gnd_mode == "plane_both" else [])
    for L in layers:
        z = pcbnew.ZONE(b)
        z.SetLayer(L)
        z.SetNet(net)
        ol = z.Outline(); ol.NewOutline()
        for (x, y) in [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()), (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())]:
            ol.Append(int(x), int(y))
        z.SetMinThickness(int(0.15 * MM))
        z.SetThermalReliefGap(int(0.2 * MM))
        z.SetThermalReliefSpokeWidth(int(0.2 * MM))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
        z.SetAssignedPriority(0)
        b.Add(z); made += 1
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
pcbnew.SaveBoard(%(dst)r, b)
print("@@" + json.dumps(dict(zones=made)))
'''

DSN_CODE = r'''
import pcbnew, json
b = pcbnew.LoadBoard(%(src)r)
ok = pcbnew.ExportSpecctraDSN(b, %(dsn)r)
print("@@" + json.dumps(dict(ok=bool(ok))))
'''

SES_CODE = r'''
import pcbnew, json
b = pcbnew.LoadBoard(%(src)r)
ok = pcbnew.ImportSpecctraSES(b, %(ses)r)
if b.Zones():
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
pcbnew.SaveBoard(%(dst)r, b)
tl = 0.0; nv = 0
for t in b.GetTracks():
    if t.GetClass() == "PCB_VIA":
        nv += 1
    else:
        tl += t.GetLength() / 1e6
print("@@" + json.dumps(dict(ok=bool(ok), track_mm=tl, vias=nv)))
'''


def run_freerouting(dsn, ses, timeout=900, passes=100, optimize=False, log=None, threads=1):
    """Headless Freerouting. Its own job timeout (timeout - 60 s) makes it stop gracefully and still
    write the session; the log streams to `log` so progress can be watched."""
    env = dict(os.environ); env.pop("DISPLAY", None)
    jt = max(60, int(timeout) - 60)
    hms = f"{jt // 3600:02d}:{(jt % 3600) // 60:02d}:{jt % 60:02d}"
    cmd = [JAVA, "-Xmx3g", "-Djava.awt.headless=true", "-jar", FR_JAR, "--gui.enabled=false",
           "--router.fanout.enabled=false", f"--router.optimizer.enabled={'true' if optimize else 'false'}",
           f"--router.autorouter.max_passes={passes}", f"--router.max_threads={threads}",
           f"--router.job_timeout={hms}", "--router.automatic_neckdown=false", "-de", str(dsn), "-do", str(ses)]
    t0 = time.time()
    log = Path(log) if log else Path(str(ses) + ".log")
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            p.wait(timeout=timeout + 180)
            timed_out = False
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); timed_out = True
    txt = log.read_text(errors="ignore")
    passes_done = re.findall(r"Auto-routing pass #(\d+).*?\((\d+) unrouted", txt)
    return dict(seconds=time.time() - t0, timed_out=timed_out, ses_exists=Path(ses).exists(),
                passes=[(int(a), int(b)) for a, b in passes_done])


def inflate_npth(dsn_path, extra_um=200.0):
    """JLC wants 0.2 mm from unplated holes to copper, but that rule only lives in the KiCad DRU.
    KiCad exports NPTH holes as netless pins named '@<n>' whose padstack is the hole outline; give
    those pins copies of their padstacks grown by `extra_um` (diameter / path width) so Freerouting's
    0.1 mm default clearance becomes 0.2 mm from the hole edge."""
    txt = Path(dsn_path).read_text()
    npth_stacks = set(re.findall(r'\(pin (\S+) @\d+ ', txt))
    for ps in npth_stacks:
        m = re.search(r'\(padstack ' + re.escape(ps) + r'\n(.*?)\n    \)', txt, re.S)
        if not m:
            continue
        body = m.group(1)
        body2 = re.sub(r'\(circle (\S+) ([\d.]+)', lambda q: f"(circle {q.group(1)} {float(q.group(2)) + extra_um:.1f}", body)
        body2 = re.sub(r'\(path (\S+) ([\d.]+)', lambda q: f"(path {q.group(1)} {float(q.group(2)) + extra_um:.1f}", body2)
        new = ps + "_NPTHCLR"
        txt = txt.replace(m.group(0), m.group(0) + f"\n    (padstack {new}\n{body2}\n    )")
        txt = re.sub(r'\(pin ' + re.escape(ps) + r' (@\d+) ', lambda q: f"(pin {new} {q.group(1)} ", txt)
    # round NPTH holes are exported as keep-out circles (hole diameter) on every copper layer
    txt, n_ko = re.subn(r'\(keepout "" \(circle (\S+) ([\d.]+)([ )])',
                        lambda q: f'(keepout "" (circle {q.group(1)} {float(q.group(2)) + extra_um:.1f}{q.group(3)}', txt)
    Path(dsn_path).write_text(txt)
    return len(npth_stacks) + n_ko


def ensure_xvfb():
    """Freerouting 1.9 needs a display; start a private Xvfb (user space, no root) if needed."""
    lock = Path(f"/tmp/.X{XDISPLAY.lstrip(':')}-lock")
    if lock.exists():
        return
    subprocess.Popen([XVFB, XDISPLAY, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if lock.exists():
            return
        time.sleep(0.1)


def run_freerouting19(dsn, ses, timeout=1800, passes=40, log=None):
    """Freerouting 1.9.0 (better completion on this board than 2.5.0; DECISIONS #38)."""
    ensure_xvfb()
    env = dict(os.environ); env["DISPLAY"] = XDISPLAY
    cmd = [JAVA, "-Xmx3g", "-jar", FR19_JAR, "-de", str(dsn), "-do", str(ses), "-mp", str(passes), "-oit", "50"]
    t0 = time.time()
    log = Path(log) if log else Path(str(ses) + ".log")
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(Path(dsn).parent))
        try:
            p.wait(timeout=timeout); timed_out = False
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); timed_out = True
    return dict(seconds=time.time() - t0, timed_out=timed_out, ses_exists=Path(ses).exists(), passes=[], router="1.9.0")


def shrink_boundary(dsn_path, inset_um=100.0):
    """Freerouting keeps only its default clearance (0.1 mm) from the outline; JLC wants 0.2 mm."""
    from shapely.geometry import Polygon
    txt = Path(dsn_path).read_text()
    m = re.search(r"\(boundary\s*\(path pcb 0\s+([-\d.\s]+)\)\s*\)", txt)
    if not m:
        return False
    v = [float(x) for x in m.group(1).split()]
    poly = Polygon(list(zip(v[0::2], v[1::2]))).buffer(-inset_um, join_style=2)
    pts = list(poly.exterior.coords)
    body = "  ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    txt = txt[:m.start()] + f"(boundary\n      (path pcb 0  {body})\n    )" + txt[m.end():]
    Path(dsn_path).write_text(txt)
    return True


def strip_net(dsn_path, net="GND"):
    """Remove a net from the DSN netlist so Freerouting leaves it unrouted (its pads stay obstacles)."""
    txt = Path(dsn_path).read_text()
    txt, n = re.subn(r'\n\s*\(net ' + re.escape(net) + r'\b[^()]*\(pins[^)]*\)\s*\)', "", txt)
    txt = re.sub(r'(\(class [^\n]*?)\s' + re.escape(net) + r'(?=[\s)])', r"\1", txt)
    Path(dsn_path).write_text(txt)
    return n


POUR_CODE = r'''
import pcbnew, json, math
MM = 1e6
b = pcbnew.LoadBoard(%(src)r)
net = b.FindNet(%(net)r)
bb = b.GetBoardEdgesBoundingBox()
outline = pcbnew.SHAPE_POLY_SET()
b.GetBoardPolygonOutlines(outline)
zones = []
for L in (pcbnew.F_Cu, pcbnew.B_Cu):
    z = pcbnew.ZONE(b); z.SetLayer(L); z.SetNet(net)
    ol = z.Outline(); ol.NewOutline()
    for (x, y) in [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()), (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())]:
        ol.Append(int(x), int(y))
    z.SetMinThickness(int(0.15 * MM)); z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS); z.SetAssignedPriority(0)
    b.Add(z); zones.append(z)
filler = pcbnew.ZONE_FILLER(b)
filler.Fill(b.Zones())
# stitching vias wherever a 0.5 mm via (+0.1 mm margin) sits inside both GND fills
step = %(step)f
r = 0.25 + 0.1
fills = [z.GetFilledPolysList(z.GetLayer()) for z in zones]
def inside_all(x, y):
    for a in range(8):
        px = int((x + r * math.cos(a * math.pi / 4)) * MM); py = int((y + r * math.sin(a * math.pi / 4)) * MM)
        for f in fills:
            if not f.Contains(pcbnew.VECTOR2I(px, py)):
                return False
    return all(f.Contains(pcbnew.VECTOR2I(int(x * MM), int(y * MM))) for f in fills)
nv = 0
x0, y0, x1, y1 = bb.GetLeft() / MM, bb.GetTop() / MM, bb.GetRight() / MM, bb.GetBottom() / MM
y = y0 + step / 2
while y < y1:
    x = x0 + step / 2
    while x < x1:
        if inside_all(x, y):
            v = pcbnew.PCB_VIA(b); v.SetPosition(pcbnew.VECTOR2I(int(x * MM), int(y * MM)))
            v.SetWidth(int(0.5 * MM)); v.SetDrill(int(0.3 * MM)); v.SetNet(net); v.SetViaType(pcbnew.VIATYPE_THROUGH)
            v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
            b.Add(v); nv += 1
        x += step
    y += step
filler.Fill(b.Zones())
pcbnew.SaveBoard(%(dst)r, b)
print("@@" + json.dumps(dict(stitch_vias=nv)))
'''


def drc(pcb, refill=True, timeout=900):
    rep = Path(pcb).with_suffix(".drc.json")
    cmd = ["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all", "--units", "mm", "-o", str(rep)]
    if refill:
        cmd.append("--refill-zones")
    subprocess.run(cmd + [str(pcb)], capture_output=True, text=True, timeout=timeout)
    d = json.loads(rep.read_text())
    by_type, err, warn = {}, 0, 0
    for v in d.get("violations", []):
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        err += v.get("severity") == "error"; warn += v.get("severity") == "warning"
    return dict(errors=int(err), warnings=int(warn), unconnected=len(d.get("unconnected_items", [])),
                by_type=by_type, report=str(rep))


ROUTING_NETCLASS = {}  # CLI default: no netclass overrides (see configs/example_board.json "routing")


def route_board(placed_pcb, out_pcb, gnd="track", timeout=900, passes=100, optimize=False,
                netclass=None, unrouted_baseline=None, workdir=None, router="1.9", gnd_net="GND"):
    """Route a placed board. Returns metrics dict; writes out_pcb (+ .kicad_pro/.kicad_dru copies)."""
    placed_pcb = Path(placed_pcb).resolve(); out_pcb = Path(out_pcb).resolve()
    assert placed_pcb != out_pcb
    work = Path(workdir or out_pcb.parent / (out_pcb.stem + "_work")); work.mkdir(parents=True, exist_ok=True)
    # project + rules travel with every intermediate board
    def sidecars(dst_pcb):
        for suf in (".kicad_pro", ".kicad_dru"):
            s = placed_pcb.with_suffix(suf)
            if s.exists():
                shutil.copyfile(s, Path(dst_pcb).with_suffix(suf))
        if netclass and Path(dst_pcb).with_suffix(".kicad_pro").exists():
            set_netclasses(Path(dst_pcb).with_suffix(".kicad_pro"), netclass)
    prep = work / "prep.kicad_pcb"
    sidecars(prep)
    _child(PREP_CODE % dict(src=str(placed_pcb), dst=str(prep), gnd=gnd, net=gnd_net))
    if unrouted_baseline is None:
        unrouted_baseline = drc(prep, refill=True)["unconnected"]
    dsn, ses = work / "board.dsn", work / "board.ses"
    if ses.exists():
        ses.unlink()
    _child(DSN_CODE % dict(src=str(prep), dsn=str(dsn)))
    inflate_npth(dsn)
    shrink_boundary(dsn)
    if gnd == "pour":
        strip_net(dsn, gnd_net)
    if router == "1.9":
        fr = run_freerouting19(dsn, ses, timeout=timeout, passes=min(passes, 40), log=work / "freerouting.log")
    else:
        fr = run_freerouting(dsn, ses, timeout=timeout, passes=passes, optimize=optimize, log=work / "freerouting.log")
    res = dict(placed=str(placed_pcb), routed=str(out_pcb), gnd=gnd, freerouting=fr,
               unrouted_baseline=unrouted_baseline)
    if not fr["ses_exists"]:
        res.update(completion=0.0, ok=False, error="no session file (timeout before first save)")
        return res
    sidecars(out_pcb)
    imp = _child(SES_CODE % dict(src=str(prep), ses=str(ses), dst=str(out_pcb)))
    if gnd == "pour":
        pour = _child(POUR_CODE % dict(src=str(out_pcb), dst=str(out_pcb), step=2.0, net=gnd_net))
        imp["vias"] = imp["vias"] + pour["stitch_vias"]; res["stitch_vias"] = pour["stitch_vias"]
    d = drc(out_pcb, refill=True)
    comp = 1.0 - d["unconnected"] / max(1, unrouted_baseline)
    res.update(track_mm=imp["track_mm"], vias=imp["vias"], drc=d, completion=comp,
               ok=(d["unconnected"] == 0 and d["errors"] == 0))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("placed"); ap.add_argument("--out", required=True)
    ap.add_argument("--gnd", default="track", choices=["track", "plane_b", "plane_both", "pour"])
    ap.add_argument("--timeout", type=int, default=900); ap.add_argument("--passes", type=int, default=100)
    ap.add_argument("--optimize", action="store_true"); ap.add_argument("--router", default="1.9", choices=["1.9", "2.5"])
    ap.add_argument("--gnd_net", default="GND")
    ap.add_argument("--netclass", default=json.dumps(ROUTING_NETCLASS),
                    help='JSON netclass overrides for routing, e.g. \'{"Power": {"track_width": 0.25}}\'')
    a = ap.parse_args()
    r = route_board(a.placed, a.out, a.gnd, a.timeout, a.passes, a.optimize, netclass=json.loads(a.netclass),
                    router=a.router, gnd_net=a.gnd_net)
    print(json.dumps(r, indent=1))
