"""Roundabout middle-density moderately complex interaction with replay attack."""

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "simple_interaction"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from roundabout_attack_common import run_roundabout_attack_movements


CASE_NAME = "Roundabout_5_Moderate_ReplayAttack"
INTERACTION_COMPLEXITY = "moderately_complex_interaction"
MOVEMENT_NAMES = (
    "north_to_west",
    "north_to_east",
    "north_to_south",
    "west_to_north",
    "west_to_south",
)


if __name__ == "__main__":
    run_roundabout_attack_movements(CASE_NAME, "replay", MOVEMENT_NAMES, INTERACTION_COMPLEXITY)
