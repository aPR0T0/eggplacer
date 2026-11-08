# How to run EGGPlacer-for-PCBs on any KiCad board

Step by step, with the exact commands, run from the repository root with the venv activated.
Tested on KiCad 10.0.4 only; KiCad 8/9 Python APIs differ in places such as pad and arc helpers.

Status: steps 1–11 are implemented and verified on our test board (241 parts, 2 copper layers,
LQFP-100 MCU, QFNs, four USB-C receptacles). Steps 12–13 (size search and final export of the
smallest board) are written but not yet validated end to end. On that board, Freerouting doesn't
reach 100% completion yet; see "Known limits".

---

## 0. One-time setup
```bash
# Python (system site packages so KiCad's pcbnew module is importable)
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch      # or a CUDA build for training
pip install -r requirements.txt
bash scripts/fetch_chipdiffusion.sh        # denoiser backbone (fetched from upstream, not redistributed)
# Routers, no root needed: Java 25 (Freerouting 2.5 needs it), Freerouting 2.5.0 + 1.9.0, a private Xvfb
mkdir -p ~/tools/jdk ~/tools/xvfb
#   Zulu JDK 25 tarball -> ~/tools/jdk (api.azul.com), freerouting-2.5.0.jar -> ~/tools/fr.jar,
#   freerouting-1.9.0.jar -> ~/tools/fr19.jar (GitHub release assets)
(cd ~/tools/xvfb && apt-get download xvfb && dpkg -x xvfb_*.deb root)
gcloud auth login            # only needed for GCP training
```
Override locations with `EGG_JAVA`, `EGG_FREEROUTING`, `EGG_FREEROUTING19`, `EGG_XVFB`.

## 1. Work on a copy of the KiCad project
```bash
rsync -a --exclude .history --exclude '~*.lck' /path/to/myboard/ ../myboard_egg/
```
Keep the copy at the same directory depth if your `fp-lib-table` uses `${KIPRJMOD}/../…` paths.
Every output is written next to this copy under a new name; the original is never touched.

## 2. In KiCad (on the copy), lock what must not move
Lock (`L`) every footprint that must stay at its exact coordinates. Locked parts become
`fixed_abs`: they keep position and rotation, and the new outline is built around them.

## 3. Write the rules config
Start from `configs/example_board.json` and save it as `configs/myboard.json`. Every field:

| Key | Meaning |
|---|---|
| `board` | path to the copy's `.kicad_pcb`, **relative to this config file** |
| `layers` | copper layer count (you decide it; the tool never changes it) |
| `outline` | `shape: rect`, `aspect: free` or a number, `aspect_candidates`, `corner_radius`, `max_w` / `max_h` |
| `clearance` | `edge` (box to board edge, mm), `courtyard` (box to box), `hole_both_sides` (keep-out margin around holes on the other side) |
| `mounting_holes` | refs + `mode: corners` (pinned to the outline corners, move with it) or omit to leave them free |
| `fixed_abs` | refs of absolutely fixed parts (your locked parts) |
| `edge_parts` | per ref: `mouth` (`+x`/`-x`/`+y`/`-y` in the footprint's own frame: the side that must face out) and `edge_line` (footprint-local coordinate on that axis that sits on the board edge, or `"courtyard"`) |
| `sides` | `allow_bottom_kinds` (reference prefixes allowed on the bottom, e.g. `["R","C","L"]`), `bottom_penalty` |
| `decoupling` | `cap_kind`, `ic_kinds`, `gnd_nets` (**first entry = your ground net name**) |
| `net_weights` | `gnd_weight`, `power_nets`, `power_weight`, `fanout_cap` |
| `dru` | KiCad DRC rules file for the output board (fab limits for your layer count), relative to the config |
| `routing` | `router` (`1.9` recommended), `gnd` (`track` or `pour`), `timeout` (s), `netclass_overrides` (step 6) |

## 4. Parse the board, get statistics and the area lower bound
```bash
python scripts/board_stats.py configs/myboard.json out
```
This writes `out/problem.json` (parts, keep-out boxes, pads, nets, rules, absolute paths) and
`out/board_stats.json`, which includes `area_lower_bound_mm2` at several packing densities. Every
later step reads `out/problem.json`; pass `--problem` to use another file.

## 5. Check footprints and edge-connector mouths
```bash
# refs whose courtyard was missing/broken (their box was rebuilt from pads + fab + silk + 0.25 mm)
python3 -c "import json;print([p['ref'] for p in json.load(open('out/problem.json'))['parts'] if not p['courtyard_ok']])"
python scripts/plot_footprints.py out/problem.json J1,J2,J3,Q1 out/footprints.png
```
- Every green box must cover the whole package. If one doesn't, fix the footprint's courtyard in the
  copy and redo step 4.
- For each edge connector, find the side that must face out: a "PCB Edge" line on Dwgs.User, the
  card or plug outline on Fab, or the 3D model's opening. Put that `mouth` and `edge_line` in the
  config and redo step 4. Wrong mouths silently produce useless placements.

## 6. Make the rules routable before placing anything
- **Netclasses:** for every netclass, the track plus clearance on both sides must fit the finest-pitch
  pin on its nets: `track/2 + clearance <= pitch - pad_width/2`. If it doesn't (on our test board a
  0.4 / 0.15 mm Power class couldn't reach 0.4 mm-pitch QFN pins), add
  `routing.netclass_overrides`. That override only affects the routing copy; widen high-current
  traces after routing.
- **DRU:** write a `.kicad_dru` with the fab's limits for your layer count (min track and space, via
  drill and annular ring, hole and edge clearances). See `configs/jlcpcb_2layer.kicad_dru`.

## 7. Verify the energies on your board
```bash
python tests/test_energy.py      # torch == numba energies; gradients vs finite differences
```

## 8. SA baseline at one generous size (legal placement + KiCad file + DRC [+ routing])
```bash
python scripts/run_sa.py --W 60 --H 60 --seed 0 --out out/sa_60 [--route]
python scripts/export_placement.py out/sa_60/placement.npz --W 60 --H 60 --name myboard_sa_60x60
```
- `run_sa.py` writes `placement.npz`, `placement.png`, `result.json` (legality, HPWL, DRC, routing
  completion) and a board under `<copy>/runs/`.
- `export_placement.py` writes `<copy>/myboard_sa_60x60.kicad_pcb` with its `.kicad_pro` and
  `.kicad_dru`, re-checks every pad position, and runs DRC.
- Options: `--cap 0.6 --compact 0` spreads parts to leave routing room.

## 9. Synthetic training data (matched to your board automatically)
The generator takes footprint templates, the class mix and the signal fan-out histogram from
`out/problem.json`.
```bash
python -m eggpcb.datagen --out /tmp/data --n 48 --workers 6     # local smoke test
bash gcp/push_code.sh                                                          # code + out/problem.json -> GCS
gcp/launch.sh egg-dg-a n2d-highcpu-96 cpu 3 '$PY -m eggpcb.datagen --out /opt/egg/data/v1_20k_a --n 20000 --workers 94 --shard 500 --seed 1 --gcs $BUCKET/data/v1_20k_a'
```
Edit `gcp/env.sh` first (project, region, zone, bucket). Each VM has a hard `--max-run-duration`,
an idle watchdog, and powers off when its job ends; delete finished VMs with
`gcloud compute instances delete <name>`. `python3 gcp/ledger.py status` shows estimated spend.

## 10. Train the denoiser on one GPU
Edit the dataset names in `gcp/gpu_job.sh`, then:
```bash
gcp/launch.sh egg-gpu-train g2-standard-8 gpu 8 'bash gcp/gpu_job.sh'
```
It runs a short validation run on the first set, then the full run (resumable; checkpoints in
`$BUCKET/ckpt/{small,full}/`).

## 11. Check the model
```bash
gcloud storage cp $BUCKET/ckpt/full/ema_latest.pt /tmp/ema.pt
python scripts/eval_model.py --ckpt /tmp/ema.pt --data /tmp/heldout_shards --n 12 --out out/eval
```
This compares raw samples with labels and random placements on held-out synthetic circuits, then
samples your board with and without guidance and legalizes the result.

## 12. Outer size search (written, not yet validated end to end)
Run it on a high-CPU VM with KiCad 10, Java 25, Freerouting and Xvfb installed (not scripted yet):
```bash
python -m eggpcb.size_search --methods sa,unguided,guided --ckpt /tmp/ema.pt \
    --aspects 1.0,1.3,0.77,1.6,0.625 --seeds 3 --lo <0.80-density bound> --hi <generous area> \
    --steps 5 --workers 90 --out runs/ss --gcs $BUCKET/size_search
```
A size passes only when some seed legalizes, routes 100% and has 0 DRC errors. Results go to
`runs/ss/results.jsonl` (resumable), with the smallest passing area per method and aspect in
`summary.json`.

## 13. Export the smallest passing board
`export_placement.py <its placement.npz> --W .. --H .. --name myboard_eggplacer`. Its routed version
is next to it under `<copy>/runs/…/routed.kicad_pcb`.


## Using eggplacer as a git submodule (recommended for a board project)

Keep this repo generic and put everything board-specific in your own repository, next to it:

```
myproject/
  eggplacer/              git submodule -> github.com/aPR0T0/eggplacer
  myboard/                copy of your KiCad project (outputs are written here under new names)
  eggplacer_myboard/      your files, committed in YOUR repo
    configs/myboard.json  "board": "../../myboard/myboard.kicad_pcb", "dru": "myboard_2layer.kicad_dru"
    configs/myboard_2layer.kicad_dru
    gcp.env               export PROJECT=... REGION=... ZONE=... BUCKET=gs://... EGG_LEDGER=$PWD/eggplacer_myboard/logs/gcp_ledger.json
    out/ logs/ reports/   problem.json, results, cost ledger, notes
```

From `myproject/`:
```bash
git submodule add https://github.com/aPR0T0/eggplacer eggplacer     # once; others: git clone --recursive
(cd eggplacer && bash scripts/fetch_chipdiffusion.sh)
P=eggplacer_myboard/out/problem.json
python eggplacer/scripts/board_stats.py eggplacer_myboard/configs/myboard.json eggplacer_myboard/out
EGG_PROBLEM=$P python eggplacer/tests/test_energy.py
python eggplacer/scripts/plot_footprints.py $P J1,J2 eggplacer_myboard/out/footprints.png
python eggplacer/scripts/run_sa.py --problem $P --W 60 --H 60 --out eggplacer_myboard/out/sa_60 [--route]
python eggplacer/scripts/export_placement.py eggplacer_myboard/out/sa_60/placement.npz --problem $P --W 60 --H 60 --name myboard_sa_60x60
PYTHONPATH=eggplacer python -m eggpcb.datagen --problem $P --cfg eggplacer_myboard/configs/myboard.json --out /tmp/data --n 48
source eggplacer_myboard/gcp.env && bash eggplacer/gcp/push_code.sh $P        # bundle code + YOUR problem for the VMs
eggplacer/gcp/launch.sh egg-gpu-train g2-standard-8 gpu 8 'FULL=v1_5k,v1_20k_a HOURS=6 bash gcp/gpu_job.sh'
python eggplacer/dashboard/server.py --bucket $BUCKET --ledger $EGG_LEDGER         # live training dashboard
python eggplacer/scripts/eval_model.py --problem $P --ckpt /tmp/ema.pt --data /tmp/heldout --out eggplacer_myboard/out/eval
```
`problem.json` stores the board and DRU paths both absolutely and relative to itself, so the project
keeps working after a fresh clone. Update the code with
`git submodule update --remote eggplacer` and commit the new submodule pointer.

---

## Known limits (from our test board)
- **Routing is the bottleneck.** Freerouting 2.5.0 stalls after its first pass; 1.9.0 does better
  (68% vs 47% on the same 58 × 58 board) but not 100%. The `gnd: pour` mode (GND poured and
  stitched after routing) is new and under test.
- Don't pour copper before routing: Freerouting treats zones as obstacles for every other net.
- The greedy legalizer can fail above roughly 70% keep-out-box density, because a large part finds
  no hole in fragmented free space (docs/DESIGN.md).
- Freerouting doesn't know DRU-only rules. NPTH hole clearance and edge clearance are patched into
  the DSN automatically (`inflate_npth`, `shrink_boundary`); other DRU-only rules aren't.
- Rotation, side and edge choice during guided sampling come from a discrete local search, not the
  model (see docs/DESIGN.md).
- The ChipDiffusion backbone has no licence upstream, so it is fetched, not redistributed
  (`scripts/fetch_chipdiffusion.sh`).
