# controller/decide.py
# Phase enum and transition helper for TerraScout OODA loop.
# Lives here (not in ooda_backend.py) to avoid circular imports.

from enum import Enum, auto


class Phase(Enum):
    IDLE          = auto()   # startup — zero rotors, init subsystems
    TAKEOFF       = auto()   # climb to scan altitude under PID
    DESCEND_SCAN  = auto()   # descend slowly while LiDAR classifies terrain below
    APPROACH      = auto()   # PID transit to 5m directly above committed zone
    LAND          = auto()   # MPC final descent from 5m to touchdown
    ABORT         = auto()   # no viable zone found — return to origin and land


def transition_phase(current: Phase, next_phase: Phase) -> Phase:
    """
    Log and return the new phase.  Add any guard conditions here later.
    """
    print(f"[OODA] {current.name} -> {next_phase.name}")
    return next_phase
