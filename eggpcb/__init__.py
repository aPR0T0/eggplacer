"""EGGPlacer for PCBs (energy-guided diffusion / SA placement). See pcb_component_placer for the
placer interface used by pcb_design_pipeline."""
try:  # make KiCad's pcbnew importable inside a uv/pip virtualenv
    from pcb_component_placer.kicad_env import ensure_pcbnew as _ensure_pcbnew
    _ensure_pcbnew()
except ImportError:
    pass
