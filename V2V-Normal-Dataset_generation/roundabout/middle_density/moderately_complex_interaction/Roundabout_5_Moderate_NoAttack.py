"""Roundabout middle-density moderately complex interaction without attack."""

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "simple_interaction"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import SouthEntries_5_NoAttack as roundabout_normal


roundabout_normal.CASE_NAME = "Roundabout_5_Moderate_NoAttack"
roundabout_normal.INTERACTION_COMPLEXITY = "moderately_complex_interaction"
roundabout_normal.MOVEMENT_NAMES = (
    "south_to_east",
    "south_to_west",
    "south_to_north",
    "east_to_south",
    "east_to_north",
)


if __name__ == "__main__":
    roundabout_normal.main()
