from evaluation.scenarios.user_priors import compute_user_priors


def _cmd(t, axis="x", mode="translation", direction=1, stamp=0.0, active=True):
    return {
        "type": t,
        "active": active,
        "axis": axis,
        "direction": direction,
        "mode": mode,
        "stamp": stamp,
    }


def test_empty_gui_events_returns_zeros():
    p = compute_user_priors([])
    assert p.total_commands == 0
    assert p.translation_share == 0.0
    assert p.total_active_time_sec == 0.0


def test_single_translation_burst():
    events = [
        _cmd("cartesian_velocity", mode="translation", stamp=0.0, active=True),
        {"type": "stop", "reason": "release", "axis": "x", "mode": "translation", "stamp": 0.5},
        _cmd("cartesian_velocity", mode="translation", direction=1, stamp=1.0, active=True),
        {"type": "stop", "reason": "release", "axis": "x", "mode": "translation", "stamp": 1.4},
    ]
    p = compute_user_priors(events)
    assert p.total_commands == 2
    assert p.translation_share == 1.0
    assert 0.0 < p.mean_active_burst_sec < 1.0


def test_mode_switch_counted():
    events = [
        _cmd("cartesian_velocity", axis="x", mode="translation", stamp=0.0),
        {"type": "stop", "reason": "release", "mode": "translation", "axis": "x", "stamp": 0.1},
        _cmd("cartesian_velocity", axis="z", mode="rotation", stamp=0.2),
        {"type": "stop", "reason": "release", "mode": "rotation", "axis": "z", "stamp": 0.3},
    ]
    p = compute_user_priors(events)
    # One switch from translation → rotation across the two active commands.
    assert p.mode_switches_per_sec > 0
    assert p.rotation_share > 0


def test_direction_reversal_counted():
    events = [
        _cmd("cartesian_velocity", axis="x", direction=1, stamp=0.0),
        {"type": "stop", "reason": "release", "axis": "x", "mode": "translation", "stamp": 0.1},
        _cmd("cartesian_velocity", axis="x", direction=-1, stamp=0.2),
        {"type": "stop", "reason": "release", "axis": "x", "mode": "translation", "stamp": 0.3},
    ]
    p = compute_user_priors(events)
    assert p.direction_reversals_per_sec > 0
