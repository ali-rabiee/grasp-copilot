from data_generator import yaw


def test_yaw_wraparound_and_move_toward():
    assert yaw.cyclic_distance_steps("N", "S") == 4
    assert yaw.cyclic_distance_steps("NW", "N") == 1
    assert yaw.move_toward("NW", "N", steps=1) == "N"
    assert yaw.move_toward("N", "NW", steps=1) == "NW"
    assert yaw.move_toward("N", "N", steps=3) == "N"

