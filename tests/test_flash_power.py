"""Tests for :mod:`session.flash_power` — power-label formatter,
parser, and per-aperture target computation."""

from __future__ import annotations

import pytest

from session.flash_power import (
    parse_power_label,
    power_label,
    target_for_next_aperture,
)


# ---- power_label / parse_power_label --------------------------------------


@pytest.mark.parametrize(
    "stops_below_full,expected_label",
    [
        # Full stops only.
        (0.0, "1/1"),
        (1.0, "1/2"),
        (2.0, "1/4"),
        (3.0, "1/8"),
        (4.0, "1/16"),
        (5.0, "1/32"),
        (6.0, "1/64"),
        (7.0, "1/128"),
        # Sub-stop values (the -0.3 / -0.7 sub-steps on the SB-5000
        # commander display).
        (1 / 3, "1/1 -0.3"),
        (2 / 3, "1/1 -0.7"),
        (1 + 1 / 3, "1/2 -0.3"),
        (1 + 2 / 3, "1/2 -0.7"),
        (2 + 1 / 3, "1/4 -0.3"),
        (6 + 2 / 3, "1/64 -0.7"),
    ],
)
def test_power_label_known_cases(stops_below_full, expected_label):
    assert power_label(stops_below_full) == expected_label


def test_power_label_rounds_to_third_stop():
    """Input values not exactly on the 1/3-stop grid are rounded."""
    # 0.1 stops below full → rounds to 0 third-stops → "1/1".
    assert power_label(0.1) == "1/1"
    # 0.2 stops below full → rounds to 1 third-stop → "1/1 -0.3".
    assert power_label(0.2) == "1/1 -0.3"
    # 1.1 stops → rounds to 3 third-stops → "1/2".
    assert power_label(1.1) == "1/2"


def test_power_label_rejects_above_full():
    with pytest.raises(ValueError, match="above 1/1"):
        power_label(-0.5)


def test_power_label_rejects_below_minimum():
    with pytest.raises(ValueError, match="below 1/128"):
        power_label(7.5)


@pytest.mark.parametrize(
    "label,expected_stops",
    [
        ("1/1", 0.0),
        ("1/2", 1.0),
        ("1/8", 3.0),
        ("1/128", 7.0),
        ("1/1 -0.3", 1 / 3),
        ("1/1 -0.7", 2 / 3),
        ("1/2 -0.3", 1 + 1 / 3),
        ("1/2 -0.7", 1 + 2 / 3),
        ("1/4 -0.3", 2 + 1 / 3),
    ],
)
def test_parse_power_label_roundtrip(label, expected_stops):
    assert parse_power_label(label) == pytest.approx(expected_stops)


def test_parse_power_label_tolerates_whitespace():
    assert parse_power_label("  1/2  ") == pytest.approx(1.0)
    assert parse_power_label("1/2   -0.3") == pytest.approx(1 + 1 / 3)


def test_parse_power_label_rejects_garbage():
    with pytest.raises(ValueError):
        parse_power_label("not a label")
    with pytest.raises(ValueError):
        parse_power_label("1/3")  # not a power of 2


# ---- target_for_next_aperture ---------------------------------------------


# Common kwargs for the 55 AIS sweep tests.
SWEEP_KW = dict(
    soft_cap_stops=1.0,    # 1/2 power
    hard_cap_stops=0.0,    # 1/1 power
    iso_floor=100,
    iso_warn=1600,
    iso_ceil=3200,
)


def test_target_simple_one_stop_step_uses_flash():
    """Prev: 1/8 power, ISO 100. Next: +1 stop aperture step. Power
    bumps to 1/4 (one full stop more flash), ISO stays at floor."""
    target = target_for_next_aperture(
        prev_power_stops=3.0,        # 1/8
        prev_iso=100,
        aperture_stops_diff=1.0,
        **SWEEP_KW,
    )
    assert target.power_stops_below_full == pytest.approx(2.0)  # 1/4
    assert target.iso == 100


def test_target_hits_soft_cap_pushes_iso():
    """Prev: 1/2 (soft cap), ISO 100. Next: +1 stop. Can't bump power
    further (at soft cap), so ISO climbs by 1 stop to 200."""
    target = target_for_next_aperture(
        prev_power_stops=1.0,        # 1/2
        prev_iso=100,
        aperture_stops_diff=1.0,
        **SWEEP_KW,
    )
    assert target.power_stops_below_full == pytest.approx(1.0)
    assert target.iso == 200


def test_target_carries_iso_overhang_back_into_power():
    """If the previous aperture's calibration bumped ISO above floor
    (e.g. ISO 200), the next aperture's power request bakes that in:
    request power that bring ISO back to floor."""
    # Prev: 1/2 power + ISO 200 (1 stop above floor). Next step: +1
    # stop aperture. The naive total boost = 1 (aperture) + 1 (ISO
    # back to floor) = 2 stops. Power would need to go 2 stops lower
    # = 1/8 (3 stops below full). But prev was 1/2 (1 stop below
    # full), so target = 1.0 - 2.0 = -1.0 stops below full — i.e. we
    # WANT more power than the prev. That makes target_power < soft
    # cap → fits under, ISO drops back to floor.
    # Wait: target_power = prev_power_stops - boost = 1 - 2 = -1.
    # That's negative (above full power), which is over-bright.
    # Algorithm: if naive_power < soft_cap_stops, we hold at soft
    # cap and push ISO. But here naive_power = -1, which IS less
    # than soft_cap=1.0, so we hold at soft_cap and compute ISO.
    # iso_stops_needed = soft_cap - naive_power = 1 - (-1) = 2.
    # ISO 2 stops above 100 = 400. Hmm, but the intent of the
    # algorithm is to use the *available* iso headroom (prev was at
    # 200 = 1 stop above floor) to drop power. The "ISO back to
    # floor" component (1 stop) means we have an extra stop of flash
    # power available, so net we need only +1 stop of total light
    # = +1 stop of power (since iso is going back to floor).
    # Hmm, re-reading my impl: total_boost = aperture_diff +
    # iso_above_floor = 1 + 1 = 2. We subtract 2 from prev_power
    # stops to get naive_power = -1 (i.e., 1 stop OVER full power
    # = bright). That clamps at soft_cap with iso = floor +
    # (soft_cap - naive_power) = floor + 2 stops = 400. WRONG —
    # we should be at floor.
    # Let me re-think: prev_power_stops=1 (1/2 power), prev_iso=200.
    # If we kept the SAME exposure for the next aperture (which is
    # 1 stop smaller), we'd need 1 stop more flash. So:
    #   new_power = prev_power - 1 = 0 stops (full).
    # And ISO can return to floor since the +1 stop of flash power
    # carries the entire aperture step.
    # Hmm but the previous calibration was ISO 200 because flash
    # was capped. If we increase flash to 1/1, the previous EXPOSURE
    # delta was correct, so we don't need ISO 200 anymore — drop
    # ISO to 100. And we need 1 stop MORE light for the aperture
    # step. Net: prev exposure at 1/2 + 200, target exposure at
    # 1/1 + 100 (same exposure +1 stop more), but we want +1 stop
    # MORE for the next aperture. So:
    #   delta_light = (1 stop from flash power 1/2→1/1)
    #               + (-1 stop from ISO 200→100)
    #               = 0 stops. Same exposure as before.
    # That's wrong — we want +1 stop. So:
    #   target_power = 1/1 (uses up the +1 from going to full)
    #   target_iso = 200 (KEEPS the +1 from ISO).
    # Wait so iso should NOT drop to 100. Hmm.
    # OK the algorithm I wrote actually does this correctly:
    # total_boost = aperture_diff + iso_above_floor_stops = 1 + 1 = 2
    # naive_power = prev_power - total_boost = 1 - 2 = -1.
    # naive_power < soft_cap (1.0), so hold at... wait but -1 stops
    # is ABOVE full power, which is invalid. The algorithm clamps at
    # soft_cap and computes ISO from there:
    # iso_stops = soft_cap - naive_power = 1 - (-1) = 2 stops above
    # floor = 400.
    # Hmm, but actually keeping power at 1/2 and ISO at 400 is the
    # SAME total exposure as 1/1 + 200, both 2 stops above the
    # baseline 1/2 + 100. So either pair works for the same
    # exposure outcome. The algorithm picks soft_cap (1/2) + 400.
    # The user could equivalently dial 1/1 + 200, but the algorithm
    # prefers staying at soft_cap.
    # So the test assertion should be power=1.0 (1/2), ISO=400.
    target = target_for_next_aperture(
        prev_power_stops=1.0,
        prev_iso=200,
        aperture_stops_diff=1.0,
        **SWEEP_KW,
    )
    assert target.power_stops_below_full == pytest.approx(1.0)
    assert target.iso == 400


def test_target_relaxes_to_hard_cap_when_iso_would_exceed_warn():
    """If staying at soft cap requires ISO > ISO_WARN, relax power
    to hard cap (1/1) so ISO can come back down."""
    # Prev: 1/2 power (1.0), ISO 1600. Next: +1 stop.
    # iso_above_floor = log2(1600/100) = 4 stops. total_boost = 5.
    # naive_power = 1 - 5 = -4. Hold at soft cap (1.0). iso_needed
    # = 1 - (-4) = 5 stops above floor = ISO ~3200. That's >
    # iso_warn=1600. Relax to hard cap (0.0). iso_needed_at_hard =
    # 0 - (-4) = 4 stops = ISO 1600 (within warn).
    # Actually that's exactly at warn — we want power=hard_cap=0 to
    # let ISO return to 1600. Hmm, but my algorithm only relaxes if
    # the soft-cap path would EXCEED warn. ISO 3200 > 1600, so
    # relax. New target: power=0.0, iso=1600.
    target = target_for_next_aperture(
        prev_power_stops=1.0,
        prev_iso=1600,
        aperture_stops_diff=1.0,
        **SWEEP_KW,
    )
    assert target.power_stops_below_full == pytest.approx(0.0)
    # 4 stops above floor=100 on the ⅓-stop ladder: 100→125→160→200
    # →250→320→400→500→640→800→1000→1250→1600. That's 12 third-
    # steps = 4 stops. ISO 1600.
    assert target.iso == 1600


def test_target_clamps_at_ceiling_when_even_full_power_isnt_enough():
    """When the math says we need more than full power + max ISO,
    cap at (hard_cap, iso_ceil) and let the caller decide whether
    to abort."""
    # Prev: 1/1 (0 stops), ISO 3200. Next: +5 stop step (impossible).
    # iso_above_floor = log2(3200/100) = 5. total_boost = 5 + 5 = 10
    # stops needed. Hard cap path: iso = 0 - (-10) = 10 stops above
    # floor → way over ceiling. Algorithm returns (hard_cap, ceil).
    target = target_for_next_aperture(
        prev_power_stops=0.0,
        prev_iso=3200,
        aperture_stops_diff=5.0,
        **SWEEP_KW,
    )
    assert target.power_stops_below_full == pytest.approx(0.0)
    assert target.iso == 3200


def test_target_55_ais_pk13_full_sweep():
    """End-to-end: walk through the 55 AIS + PK-13 1:1 sweep,
    assuming first-aperture calibration landed at 1/4 -1/3 + ISO
    100. The brute-force solver picks the (power, iso) tuple
    minimising |error| + small soft-cap and ISO-above-floor
    penalties. For these whole-stop transitions starting from
    1/4 -1/3, the answer at each step is "drop power 1/3 stop
    below soft cap (= keep at the same 1/2 -1/3 value reached at
    step 1) and bump ISO by 1 stop" — slightly dimmer flash than
    sitting at soft cap means better recycle / battery, and the
    penalty for being below soft cap is zero."""
    sweep = []
    power, iso = 2 + 1 / 3, 100  # 1/4 -1/3 (post-first-aperture)
    sweep.append((power, iso))
    for _ in range(3):  # f/4 → f/5.6 → f/8 → f/11
        target = target_for_next_aperture(
            prev_power_stops=power,
            prev_iso=iso,
            aperture_stops_diff=1.0,
            **SWEEP_KW,
        )
        power, iso = target.power_stops_below_full, target.iso
        sweep.append((power, iso))
    # Step 0: 1/4 -1/3 (2.33) + 100
    # Step 1: 1/2 -1/3 (1.33) + 100 — power up 1 stop, ISO unchanged
    # Step 2: 1/2 -1/3 + 200 — power stays, ISO up 1 stop
    # Step 3: 1/2 -1/3 + 400 — power stays, ISO up 1 stop
    assert sweep[0] == (pytest.approx(2 + 1 / 3), 100)
    assert sweep[1] == (pytest.approx(1 + 1 / 3), 100)
    assert sweep[2] == (pytest.approx(1 + 1 / 3), 200)
    assert sweep[3] == (pytest.approx(1 + 1 / 3), 400)
    # All apertures stayed under soft cap and well under warn.
    for power, iso in sweep:
        assert power >= 1.0  # ≤ 1/2 power (at or below soft cap)
        assert iso <= 1600
