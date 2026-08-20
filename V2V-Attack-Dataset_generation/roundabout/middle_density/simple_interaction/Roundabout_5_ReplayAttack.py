"""Roundabout middle-density simple interaction: north-entry vehicles with replay attack."""

from roundabout_attack_common import run_roundabout_attack


CASE_NAME = "Roundabout_5_ReplayAttack"


if __name__ == "__main__":
    run_roundabout_attack(CASE_NAME, "replay", "north")
