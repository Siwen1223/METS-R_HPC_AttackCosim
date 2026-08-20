"""Roundabout middle-density moderately complex interaction with false information injection attack."""

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "simple_interaction"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from roundabout_attack_common import run_roundabout_attack_movements


CASE_NAME = "Roundabout_5_Moderate_FalseInformationInjection"
INTERACTION_COMPLEXITY = "moderately_complex_interaction"
MOVEMENT_NAMES = (
    "south_to_east",
    "south_to_west",
    "south_to_north",
    "east_to_south",
    "east_to_north",
)


if __name__ == "__main__":
    run_roundabout_attack_movements(CASE_NAME, "false_information_injection", MOVEMENT_NAMES, INTERACTION_COMPLEXITY)
