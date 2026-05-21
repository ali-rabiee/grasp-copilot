import random

from evaluation.rollouts.noise import NoiseInjector, NoiseProfile, STANDARD_CONDITIONS


def _injector(profile):
    return NoiseInjector(profile=profile, rng=random.Random(0))


def test_profile_rejects_out_of_range():
    import pytest
    with pytest.raises(ValueError):
        NoiseProfile("bad", p_dir=1.5)


def test_clean_profile_is_identity():
    inj = _injector(NoiseProfile("clean"))
    cmd = {"axis": "x", "direction": 1, "mode": "translation"}
    for _ in range(50):
        assert inj.direction(cmd) == cmd
        assert inj.dropout(cmd) == cmd
        assert inj.selection(0, 3) == 0
    assert inj.latency_sec() == 0.0
    assert inj.stats()["direction_perturbed"] == 0


def test_direction_perturbation_changes_command():
    inj = NoiseInjector(profile=NoiseProfile("all_dir", p_dir=1.0), rng=random.Random(0))
    cmd = {"axis": "x", "direction": 1, "mode": "translation"}
    seen = set()
    for _ in range(20):
        out = inj.direction(dict(cmd))
        seen.add((out["axis"], out["direction"]))
    # We expect either an axis swap or a direction flip at p=1.0.
    assert seen != {(cmd["axis"], cmd["direction"])}
    assert inj.stats()["direction_perturbed"] == 20


def test_gripper_command_passes_through_direction_noise():
    inj = NoiseInjector(profile=NoiseProfile("all_dir", p_dir=1.0), rng=random.Random(0))
    cmd = {"axis": "", "direction": 0, "mode": "gripper"}
    for _ in range(10):
        assert inj.direction(cmd) == cmd


def test_dropout_returns_none_at_rate():
    inj = NoiseInjector(profile=NoiseProfile("all_drop", p_drop=1.0), rng=random.Random(0))
    cmd = {"axis": "x", "direction": 1, "mode": "translation"}
    n_dropped = sum(1 for _ in range(50) if inj.dropout(cmd) is None)
    assert n_dropped == 50


def test_selection_noise_picks_different_index():
    inj = NoiseInjector(profile=NoiseProfile("all_sel", p_sel=1.0), rng=random.Random(0))
    for _ in range(30):
        new_idx = inj.selection(0, n_choices=4)
        assert new_idx != 0
        assert 0 <= new_idx < 4


def test_selection_noiseless_with_one_choice():
    inj = NoiseInjector(profile=NoiseProfile("all_sel", p_sel=1.0), rng=random.Random(0))
    # Only one valid choice → can't flip.
    assert inj.selection(0, n_choices=1) == 0


def test_latency_bounded_when_enabled():
    inj = NoiseInjector(profile=NoiseProfile("late", latency=True), rng=random.Random(0))
    for _ in range(10):
        v = inj.latency_sec()
        assert 0.10 <= v <= 0.50


def test_standard_conditions_count_and_levels():
    names = {p.name for p in STANDARD_CONDITIONS}
    assert {"clean", "dir_low", "dir_high", "sel_low", "sel_high", "compound_mid"} <= names
    for p in STANDARD_CONDITIONS:
        assert 0.0 <= p.p_dir <= 1.0
        assert 0.0 <= p.p_sel <= 1.0
        assert 0.0 <= p.p_drop <= 1.0
