"""Roundabout middle-density simple interaction: south-entry vehicles with false information injection attack."""

from roundabout_attack_common import run_roundabout_attack


CASE_NAME = "Roundabout_5_FalseInformationInjection"


if __name__ == "__main__":
    run_roundabout_attack(CASE_NAME, "false_information_injection", "south")
