"""Stop a training VM cleanly once a given step's checkpoint is safely in GCS.

Polls <ckpt>/ until step_<N>.pt exists, verifies the EMA weights of step N (from ema_latest.pt
when it is exactly step N, otherwise from step_<N>.pt itself), uploads them as ema_step<N>.pt
(a name later syncs never touch), then deletes the VM and updates the cost ledger.

usage: python gcp/stop_at_step.py --vm egg-gpu-train --zone asia-south1-b \
           --ckpt gs://bucket/ckpt/full --step 80000
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sh(*cmd, check=False):
    return subprocess.run(list(cmd), capture_output=True, text=True, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", required=True); ap.add_argument("--zone", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true", help="verify and upload, but don't delete the VM")
    a = ap.parse_args()
    import torch
    target = f"{a.ckpt.rstrip('/')}/step_{a.step}.pt"
    print(f"waiting for {target}", flush=True)
    while sh("gcloud", "storage", "ls", target).returncode != 0:
        time.sleep(a.interval)
    time.sleep(30)  # let the trainer's rsync finish uploading ema_latest.pt alongside it
    with tempfile.TemporaryDirectory() as td:
        ema = Path(td) / "ema_latest.pt"
        sh("gcloud", "storage", "cp", f"{a.ckpt.rstrip('/')}/ema_latest.pt", str(ema), check=True)
        st = torch.load(ema, map_location="cpu")
        if st.get("step") == a.step:
            src = "ema_latest.pt"
        else:  # the trainer already moved on: take the EMA stored inside the full step-N checkpoint
            full = Path(td) / "full.pt"
            sh("gcloud", "storage", "cp", target, str(full), check=True)
            ck = torch.load(full, map_location="cpu")
            st = dict(ema=ck["ema"], step=ck["step"])
            torch.save(st, ema)
            src = f"step_{a.step}.pt"
        assert st["step"] == a.step, st["step"]
        out = f"{a.ckpt.rstrip('/')}/ema_step{a.step}.pt"
        sh("gcloud", "storage", "cp", str(ema), out, check=True)
        print(f"verified EMA weights at step {st['step']} (from {src}) -> {out}", flush=True)
    if a.dry_run:
        return
    r = sh("gcloud", "compute", "instances", "delete", a.vm, "--zone", a.zone, "--quiet")
    print(f"deleted VM {a.vm}: rc={r.returncode} {r.stderr.strip()[-200:]}", flush=True)
    ledger = Path(__file__).resolve().parent / "ledger.py"
    print(sh(sys.executable, str(ledger), "status").stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
