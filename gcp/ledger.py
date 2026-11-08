"""Local GCP cost ledger + guard (a billing budget alert can't be created: the account lacks
permission on the billing account, or as a second guard). Rates are conservative on-demand estimates
incl. boot disk, in USD/hour.

  ledger.py check MACHINE HOURS     -> exit 1 if worst-case committed spend + this job > CAP
  ledger.py start NAME MACHINE HOURS
  ledger.py status                  -> marks VMs that no longer exist as finished, prints spend
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import os
CAP = float(os.environ.get("EGG_BUDGET_USD", "100"))
RATES = {"g2-standard-4": 0.95, "g2-standard-8": 1.15, "g2-standard-16": 1.6,
         "n2d-highcpu-32": 1.15, "n2d-highcpu-48": 1.7, "n2d-highcpu-64": 2.25, "n2d-highcpu-96": 3.35,
         "c2d-highcpu-56": 2.7, "c2d-highcpu-112": 5.4, "e2-standard-4": 0.17}
LEDGER = Path(os.environ.get("EGG_LEDGER", Path(__file__).resolve().parents[1] / "logs" / "gcp_ledger.json"))


def load():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def save(d):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, indent=1))


def spend(d, worst=False):
    now = time.time(); tot = 0.0
    for e in d:
        end = e.get("end") or (e["start"] + e["max_h"] * 3600 if worst else now)
        tot += RATES.get(e["machine"], 5.0) * max(0.0, min(end, e["start"] + e["max_h"] * 3600) - e["start"]) / 3600
    return tot


def running():
    r = subprocess.run(["gcloud", "compute", "instances", "list", "--filter=labels.project=eggplacer",
                        "--format=value(name,status)"], capture_output=True, text=True)
    return {l.split()[0]: l.split()[1] for l in r.stdout.splitlines() if l.strip()}


def main():
    cmd = sys.argv[1]
    d = load()
    if cmd == "check":
        mt, h = sys.argv[2], float(sys.argv[3])
        worst = spend(d, worst=True) + RATES.get(mt, 5.0) * h
        print(f"worst-case committed spend incl. this job: ${worst:.2f} (cap ${CAP:.0f})")
        sys.exit(0 if worst <= CAP else 1)
    if cmd == "start":
        d.append(dict(name=sys.argv[2], machine=sys.argv[3], max_h=float(sys.argv[4]), start=time.time(), end=None))
        save(d)
    if cmd in ("status", "start"):
        run = running()
        for e in d:
            if e["end"] is None and e["name"] not in run:
                e["end"] = time.time()
        save(d)
        print(f"spent so far ~${spend(d):.2f}; worst case ${spend(d, worst=True):.2f}; running: {run}")


if __name__ == "__main__":
    main()
