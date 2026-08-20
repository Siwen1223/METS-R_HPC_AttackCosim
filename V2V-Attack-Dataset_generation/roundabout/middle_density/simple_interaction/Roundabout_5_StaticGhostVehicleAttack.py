"""Roundabout middle-density simple interaction: west-entry vehicles with static ghost vehicle attack."""

from roundabout_attack_common import run_roundabout_attack


CASE_NAME = "Roundabout_5_StaticGhostVehicleAttack"


if __name__ == "__main__":
    run_roundabout_attack(CASE_NAME, "static_ghost_vehicle", "west")
