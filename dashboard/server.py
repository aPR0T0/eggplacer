"""Live training dashboard (local). Polls GCS every 60 s with your gcloud login:
  * gs://<bucket>/logs/<vm>.log        - the VM's job log (mirrored by the VM every 60 s)
  * gs://<bucket>/ckpt/<run>/          - checkpoint files + their upload times (step/time anchors)
  * gcloud compute instances list      - VM status
  * gcp/ledger.py                      - estimated spend
and serves index.html + /api/state on http://127.0.0.1:8765.

usage: source gcp/env.sh && python dashboard/server.py [--port 8765] [--vm egg-gpu-train] [--run full] [--budget-hours 6]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gcp"))
import ledger  # noqa: E402

import os
BUCKET = os.environ.get("BUCKET", "gs://your-bucket-eggplacer")   # same variable as gcp/env.sh
STATE = {"updated": None, "error": None}
LOCK = threading.Lock()
EVENT_RE = re.compile(r"device cuda|resumed from|time budget|^done$|wait_for|JOB EXIT|Traceback|Error|params \d+|nvidia-smi")


def sh(cmd, timeout=60):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def parse_log(txt):
    """split the job log into runs (each 'device cuda;' line starts one) of step records + events."""
    runs, events = [], []
    cur = None
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r"device cuda; (\d+) train / (\d+) val circuits", s)
        if m:
            cur = dict(train=int(m.group(1)), val=int(m.group(2)), resume=0, records=[])
            runs.append(cur)
        m = re.match(r"resumed from step_(\d+)\.pt", s)
        if m and cur is not None:
            cur["resume"] = int(m.group(1))
        if s.startswith('{"step"') and cur is not None:
            try:
                cur["records"].append(json.loads(s))
            except json.JSONDecodeError:
                pass
        if EVENT_RE.search(s) and not s.startswith("{"):
            events.append(s[:160])
    for i, r in enumerate(runs):
        r["name"] = "small" if (len(runs) > 1 and i == 0) else ("full" if r["train"] > 20000 else f"run{i}")
    return runs, events[-12:]


def checkpoints(run):
    out = sh(["gcloud", "storage", "ls", "-l", f"{BUCKET}/ckpt/{run}/"])
    cks = []
    for line in out.splitlines():
        m = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(\S+)$", line.strip())
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
            name = m.group(2).rsplit("/", 1)[-1]
            sm = re.match(r"step_(\d+)\.pt", name)
            cks.append(dict(name=name, time=t, step=int(sm.group(1)) if sm else None))
    return sorted(cks, key=lambda c: c["time"])


def vms():
    out = sh(["gcloud", "compute", "instances", "list", "--filter=labels.project=eggplacer",
              "--format=json(name,status,machineType,zone)"])
    try:
        return [dict(name=v["name"], status=v["status"], machine=v["machineType"].rsplit("/", 1)[-1],
                     zone=v["zone"].rsplit("/", 1)[-1]) for v in json.loads(out or "[]")]
    except json.JSONDecodeError:
        return []


def spend(running_names):
    d = ledger.load()
    changed = False
    for e in d:
        if e["end"] is None and e["name"] not in running_names:
            e["end"] = time.time(); changed = True
    if changed:
        ledger.save(d)
    return dict(spent=ledger.spend(d), worst=ledger.spend(d, worst=True), cap=ledger.CAP)


def poll(args):
    try:
        txt = sh(["gcloud", "storage", "cat", f"{BUCKET}/logs/{args.vm}.log"], timeout=90)
        runs, events = parse_log(txt)
        cks = checkpoints(args.run)
        vm = vms()
        sp = spend({v["name"] for v in vm})
        run = next((r for r in runs if r["name"] == args.run), runs[-1] if runs else None)
        timing = {}
        if run and run["records"]:
            # step/time anchors: checkpoint uploads (step_N.pt mtime ~ when step N was reached)
            anchors = [(c["step"], c["time"]) for c in cks if c["step"] and c["step"] > run["resume"]]
            rates = [r["it_s"] for r in run["records"][-20:]]
            rate = sum(rates) / len(rates)
            if len(anchors) >= 2:
                (s0, t0), (s1, t1) = anchors[-2], anchors[-1]
                rate = (s1 - s0) / max(t1 - t0, 1)
            start = (anchors[-1][1] - (anchors[-1][0] - run["resume"]) / rate) if anchors else None
            last = run["records"][-1]["step"]
            if start:
                end = start + args.budget_hours * 3600
                timing = dict(start=start, end=end, now=time.time(), rate=rate, last_step=last,
                              est_final_step=int(run["resume"] + rate * args.budget_hours * 3600),
                              budget_hours=args.budget_hours)
        with LOCK:
            STATE.update(updated=time.time(), error=None, runs=runs, events=events, checkpoints=cks, vms=vm,
                         spend=sp, timing=timing, run=args.run, vm=args.vm, bucket=BUCKET)
    except Exception as e:  # keep serving the last good state
        with LOCK:
            STATE.update(error=repr(e), updated_error=time.time())


def poller(args):
    while True:
        poll(args)
        time.sleep(args.interval)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            with LOCK:
                self._send(json.dumps(STATE).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--vm", default="egg-gpu-train")
    ap.add_argument("--run", default="full"); ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    poll(a)
    threading.Thread(target=poller, args=(a,), daemon=True).start()
    print(f"dashboard on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
