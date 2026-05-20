from evaluation.scenarios.log_parser import RawObject, dedup_objects


def _obj(id_: str, label: str, cell: str, held: bool = False) -> RawObject:
    return RawObject(id=id_, label=label, grid_cell=0, grid_label=cell, is_held=held)


def test_dedup_collapses_duplicate_label_cell_pairs():
    pre = [
        _obj("obj_1", "coffee_can", "A1"),
        _obj("obj_2", "coffee_can", "A1"),   # duplicate detection of the same object
        _obj("obj_3", "mug", "B2"),
    ]
    out = dedup_objects(pre)
    assert [(o.label, o.grid_label) for o in out] == [("coffee_can", "A1"), ("mug", "B2")]
    assert [o.id for o in out] == ["obj_1", "obj_2"]


def test_dedup_keeps_distinct_cells_for_same_label():
    pre = [_obj("a", "mug", "A1"), _obj("b", "mug", "B2")]
    out = dedup_objects(pre)
    assert len(out) == 2
    assert {o.grid_label for o in out} == {"A1", "B2"}


def test_dedup_is_held_OR():
    pre = [_obj("a", "mug", "A1", held=False), _obj("b", "mug", "A1", held=True)]
    out = dedup_objects(pre)
    assert len(out) == 1
    assert out[0].is_held is True
