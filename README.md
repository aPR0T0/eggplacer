# pcb_component_placer

Component placement for KiCad boards. Every placer takes a board synced from its schematic (all
footprints and nets, no placement needed) and writes a placed board with a closed board outline.
All placers share one interface, so [pcb_design_pipeline](https://github.com/aPR0T0/pcb_design_pipeline)
can pick any of them by name from YAML.

| Placer | What it does |
|---|---|
| `eggplacer` | EGGPlacer for PCBs: energy-guided placement by simulated annealing, or by a trained diffusion model with energy guidance ([docs/EGGPLACER.md](docs/EGGPLACER.md), [GUIDE.md](GUIDE.md)) |
| `user_placed` | Keeps the placement you drew in KiCad. The board is checked but not changed |

## Usage

```bash
uv venv --python /usr/bin/python3 && uv sync          # KiCad's pcbnew is built for the system Python
uv run pcb-placer list                                # placers and their options
uv run pcb-placer place board.kicad_pcb -o placed.kicad_pcb -a eggplacer -s density=0.4
uv run pcb-placer place board.kicad_pcb -o placed.kicad_pcb -a user_placed
```

```python
from pcb_component_placer import place
res = place("board.kicad_pcb", "out/placed.kicad_pcb", algorithm="eggplacer", options={"seeds": [0, 1]})
print(res.ok, res.outline_mm, res.outside, res.placement_errors)
```

## Adding a placer

1. Create `pcb_component_placer/placers/<name>.py`. Every module in that folder is imported
   automatically.
2. Subclass `Placer` and register it with `@register("<name>")`. That name is the `algorithm:` in YAML.
3. Declare the YAML options as a nested `@dataclass class Options` with defaults. Unknown keys are
   rejected.
4. Implement `place(board_in, board_out, work_dir) -> dict`:
   - read `board_in` and never modify it;
   - write `board_out` with every footprint placed inside a closed `Edge.Cuts` outline;
   - put scratch files in `work_dir`;
   - return details for the report.
   - Get `pcbnew` with `import_pcbnew()`.
   - If your placer loads and saves boards with pcbnew in-process, set `isolate = True` on the
     class. pcbnew can't reliably load boards again after saving one in the same process.
5. Don't check your own result. `Placer.run()` does that the same way for every placer: closed
   outline, every pad on the board, no courtyard-overlap DRC errors. It writes `<board>.place.json`.

```python
from dataclasses import dataclass
from pcb_component_placer.base import Placer, register
from pcb_component_placer.kicad_env import import_pcbnew

@register("grid")
class Grid(Placer):
    summary = "parts on a grid, largest first"

    @dataclass
    class Options:
        pitch: float = 10.0

    def place(self, board_in, board_out, work_dir):
        pcbnew = import_pcbnew()
        b = pcbnew.LoadBoard(str(board_in))
        ...  # set footprint positions, draw the Edge.Cuts outline
        pcbnew.SaveBoard(str(board_out), b)
        return {"pitch": self.opts.pitch}
```

Then add a test in `tests/` and a line to the table above.

## Citation

```bibtex
@article{zhou2026eggplacer,
  title   = {EGGPlacer: Energy-Guided Generative Chip Placement with Diffusion Models},
  author  = {Zhou, Qi and Liu, Guojun and Zhao, Juanhong and Qi, Guangzhi and Chen, Zehao and Lu, Ming},
  journal = {arXiv preprint arXiv:2610.33099},
  year    = {2026}
}
@article{lee2024chipdiffusion,
  title   = {Chip Placement with Diffusion Models},
  author  = {Lee, Vint and Nguyen, Minh and Elzeiny, Leena and Deng, Chun and Abbeel, Pieter and Wawrzynek, John},
  journal = {arXiv preprint arXiv:2407.12282},
  year    = {2024}
}
```
