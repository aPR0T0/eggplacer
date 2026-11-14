"""Run one placer's place() in a fresh process (pcbnew can't reliably load boards again after it
saved one in the same process).  usage: python -m pcb_component_placer._child job.json -> @@{json}"""
import json
import os
import sys
from pathlib import Path

from pcb_component_placer import get_placer

job = json.loads(Path(sys.argv[1]).read_text())
placer = get_placer(job["algorithm"], job["options"])
details = placer.place(Path(job["board_in"]), Path(job["board_out"]), Path(job["work_dir"]))
print("@@" + json.dumps(details or {}, default=str), flush=True)
sys.stderr.flush()
# the board is saved: skip interpreter teardown, where pcbnew (KiCad 10.0.8) can segfault
os._exit(0)
