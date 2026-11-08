# Design notes: EGGPlacer adapted to PCBs

These are the places where this implementation departs from the paper or fills in a detail the paper
leaves open. Equation numbers refer to arXiv:2610.33099.

## Problem representation
- **Every footprint is a macro.** There are no standard cells, so the paper's analytical-placer
  response (Eq. 9–11) is replaced by a direct WA-WL over every pad (Eq. 18, with pad offsets after
  rotation and mirroring) plus a bin-density term.
- **Keep-out box = courtyard bounding box.** When a footprint's courtyard is missing or doesn't
  cover its pads and fab outline, the box is the union of courtyard, pads, fab and silkscreen outlines
  plus 0.25 mm (IPC courtyard excess). Silk marks under 0.8 mm (pin-1 dots) are ignored.
- **Placement variable** = centre of the keep-out box, plus a rotation index (0/90/180/270) and a side.
- **KiCad conventions** (verified against pcbnew 10):
  - top side: `world = pos + R(r) @ local`, with `R(r) = [[cos, sin], [-sin, cos]]`;
  - bottom side: local y is mirrored first.
  - Written boards are re-checked pad by pad (< 1e-6 mm).

## Energies (`eggpcb/energies.py`)
- **WA-WL**, γ = 1 mm.
- **Density:** 24 × 24 bins per side, exact box/bin overlap.
- **Smooth overlap** (Eq. 20) with β = 20 /mm. That's the paper's β = 100 expressed per mm for a
  ~10 mm normalised half-width; β = 100 /mm would be a hard step. Includes courtyard clearance.
  Two parts collide only with probability P(same side). Through-hole keep-outs also block the
  opposite side.
- **Outline containment:** signed distance of every box corner to the rounded-rectangle outline,
  with edge clearance. Edge connectors are exempt on their mouth side, where they may overhang by a
  set amount.
- **PCB soft terms:**
  - decoupling: soft-min distance from each cap's power pad to its IC's power pads;
  - sheet grouping: variance within each schematic sheet;
  - compactness: log-sum-exp smooth-max bounding-box area;
  - bottom-side count.
- The SA baseline (`eggpcb/sa.py`) reimplements the identical energy in numba with incremental
  deltas. The test suite asserts torch == numba to 1e-7 and checks every gradient against finite
  differences.

## Guidance (`eggpcb/sample.py`)
- Eq. 3 and 6–8 as written: K = 10 inner Adam steps at lr 1e-2 in normalised units.
- **One RMS-balanced λ per soft term** (Eq. 21–22): r = WL 0.4, density 0.2, compactness 0.2,
  decap 0.2, group 0.05, bottom 0.05.
  - Targets are computed once per reverse step at u(0) and smoothed with EMA 0.9.
  - λ bounds are [1e-8, 1e3]. The paper's [1e-12, 1e-3] is tied to DREAMPlace's wirelength units.
- **λ_leg** is the projected dual variable for E_leg ≤ 1e-4 (normalised area), Adam ascent at
  lr 5e-4 while t < 100, capped at 15, floor 0.1 (App. D).
- **Hard constraints are projections, applied after every inner step:** corner holes are pinned,
  edge connectors are snapped to their assigned edge, and absolutely fixed parts keep rotation and side.
- **Rotation, side and edge choice** come from a discrete local search (shared with SA) at
  t ∈ {150, 100, 60, 30, 10} and after the last step; the denoiser's conditioning is rebuilt after
  each search. The denoiser stays purely positional.
- **Thermal guidance** (Eq. 12–13, 23–26) is not implemented: it needs per-part power data.

## Denoiser (`eggpcb/model.py`, ChipDiffusion backbone)
- **Architecture:** AttGNN with encoder width 64 and 2 hybrid blocks, each GATv2 message passing
  (4 heads, width 192) plus global attention (width 64, 4 heads), with FiLM timestep conditioning.
  **567,938 parameters** (paper: 556,354; it doesn't give the inner widths).
- **Conditioning:**
  - node features: w, h normalised to the canvas, side, edge-connector flag, log1p(#pins)/4;
  - edges: each net with 2–16 pins becomes a star from the pin on the part with the most pins,
    carrying both pin offsets;
  - higher fan-out nets (ground, big rails) are left to the guidance energy.
- **Training:** cosine schedule, T = 200, masked ε-MSE (Eq. 15), Adam 1e-4, EMA 0.999, batch 64
  (4 circuits × 16 noise draws), and the 8 flip/transpose symmetries of the canvas as augmentation.
  Circuits are never merged into one graph, because GroupNorm and global attention would leak
  between them.

## Synthetic data (`eggpcb/synth.py`, `eggpcb/datagen.py`)
- **Matched to the target board:** about 80% copies of its footprints, its class mix ±25%, and its
  signal fan-out histogram.
- **Ground** gets about 26% of pins, plus 1–4 power rails and decoupling caps.
- **Outline:** aspect U(1, 2), density U(0.55, 0.80), edge connectors, optional corner holes and
  optional two-sided assembly.
- **Placement-first** (the ChipDiffusion v1 idea): pack parts into clusters, generate the netlist
  with distance-decaying pin choice, then make the label with a quick analytical placement on the
  same energy plus the legalizer. If legalizing loses the wirelength gain, the label falls back to
  the legal latent packing.

## Routing (`eggpcb/route.py`)
- **Headless Freerouting.** Version 1.9.0 (under a user-space Xvfb) routed much better than 2.5.0 on
  our test board: 68% vs 47%.
- **DRU-only rules patched into the DSN:** NPTH holes are inflated for 0.2 mm hole clearance, and
  the boundary is shrunk 0.1 mm for 0.2 mm edge clearance.
- **Never pour copper before routing:** Freerouting treats zones as obstacles for every other net.
- **`gnd: pour` mode:** strip the ground net from the DSN, route everything else, then pour ground
  on both layers, add stitching vias and let DRC judge connectivity.
- **Netclass overrides** in the config apply only to the routing copy. A wide power class that
  can't physically reach fine-pitch pins makes 100% completion impossible.

## Size search (`eggpcb/size_search.py`)
- Log-area bisection per (method, aspect ratio).
- A size passes only if some seed legalizes, routes 100% and has 0 DRC errors.
- Resumable `results.jsonl`.
