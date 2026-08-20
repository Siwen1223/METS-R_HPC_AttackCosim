"""Roundabout middle-density simple interaction: south-entry vehicles with obstacle ghost vehicle attack."""

from roundabout_attack_common import run_roundabout_attack


CASE_NAME = "Roundabout_5_ObstacleGhostVehicleAttack"


if __name__ == "__main__":
    run_roundabout_attack(CASE_NAME, "obstacle_ghost_vehicle", "south")
