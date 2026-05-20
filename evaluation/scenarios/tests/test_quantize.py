import math

from evaluation.scenarios.quantize import (
    cell_index_to_label,
    yaw_rad_to_bin,
    z_height_to_bin,
)


def test_yaw_rad_to_bin_cardinal_directions():
    assert yaw_rad_to_bin(0.0) == "E"
    assert yaw_rad_to_bin(math.pi / 2) == "N"
    assert yaw_rad_to_bin(math.pi) == "W"
    assert yaw_rad_to_bin(-math.pi) == "W"          # wraparound
    assert yaw_rad_to_bin(-math.pi / 2) == "S"
    assert yaw_rad_to_bin(math.pi / 4) == "NE"
    assert yaw_rad_to_bin(-3 * math.pi / 4) == "SW"


def test_yaw_rad_to_bin_handles_real_log_values():
    # The dominant gripper yaw across PRIME_LOGS is ~-0.26 rad ≈ -15° from E.
    # That should fall inside the "E" bin (centered on 0, ±22.5°).
    assert yaw_rad_to_bin(-0.26) == "E"
    assert yaw_rad_to_bin(-0.49) == "SE"            # past the -22.5° boundary
    assert yaw_rad_to_bin(0.21) == "E"


def test_z_height_to_bin():
    assert z_height_to_bin(0.41) == "HIGH"
    assert z_height_to_bin(0.30) == "MID"
    assert z_height_to_bin(0.10) == "LOW"
    assert z_height_to_bin(0.35) == "HIGH"          # boundary
    assert z_height_to_bin(0.18) == "MID"           # boundary


def test_cell_index_to_label():
    assert cell_index_to_label(0) == "A1"
    assert cell_index_to_label(4) == "B2"
    assert cell_index_to_label(8) == "C3"
