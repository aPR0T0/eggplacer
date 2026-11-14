"""pcb_component_placer: PCB component placement algorithms for KiCad boards behind one interface."""
from .base import PlaceResult, Placer, get_placer, list_placers, register


def place(board_in, board_out, algorithm: str = "eggplacer", options: dict | None = None,
          work_dir=None, drc: bool = True) -> PlaceResult:
    """Place a KiCad board (synced from its schematic) with the named placer and check the result."""
    return get_placer(algorithm, options).run(board_in, board_out, work_dir=work_dir, drc=drc)


__all__ = ["PlaceResult", "Placer", "get_placer", "list_placers", "place", "register"]
