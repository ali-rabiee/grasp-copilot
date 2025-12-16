import grid


def test_grid_neighbors_and_manhattan():
    assert set(grid.neighbors("A1")) == {"A2", "B1"}
    assert set(grid.neighbors("B2")) == {"A2", "B1", "B3", "C2"}
    assert grid.manhattan("A1", "C3") == 4
    assert grid.step_toward("A1", "C3") == "B1"
    assert grid.step_toward("B1", "C3") == "C1"

