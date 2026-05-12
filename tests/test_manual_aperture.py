"""Unit tests for the pure helpers in :mod:`session.manual_aperture`.

Two helpers, both pure:

* :func:`parse_lens_slug` — extract ``focal_length_mm`` and ``f_ring``
  via regex; tests run against every slug in the 2026-05 reshoot plan
  to confirm the runner never falls back to interactive prompts on
  the actual workload.
* :func:`round_to_1_3_stop_label` — log-distance snap to the 1/3-stop
  scale; tests against hand-computed cases.
* :func:`aperture_sequence_for_manual_mode` — synthesised sequence
  for non-CPU lenses, walks ring whole stops.
"""

from __future__ import annotations

import pytest

import config
from session.manual_aperture import (
    aperture_sequence_for_manual_mode,
    parse_lens_slug,
    round_to_1_3_stop_label,
)


# ---- parse_lens_slug -------------------------------------------------------


@pytest.mark.parametrize(
    "slug,expected_focal,expected_fring",
    [
        # AI-S manual lens with serial — first 'mm' is the focal length.
        (
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            55.0, 2.8,
        ),
        (
            "Nikon Micro-Nikkor 55mm f_2.8 AIS — serial 287318",
            55.0, 2.8,
        ),
        # Pre-AI f/3.5 Micro-Nikkor.
        (
            "Nikon Micro-Nikkor Auto 55mm f_3.5 209088",
            55.0, 3.5,
        ),
        # Z MC 105 — straight lens name from the reshoot plan.
        (
            "NIKKOR Z MC 105mm f/2.8 VR S",
            105.0, 2.8,
        ),
        # Z MC 105 with diopters appended (close-up filters, no
        # additional 'mm' token).
        (
            "NIKKOR Z MC 105mm f/2.8 VR S + 6T + 5T",
            105.0, 2.8,
        ),
        # Z MC 105 with extension tube — extra 'mm' token; we want
        # the lens focal length (105), not the tube (61).
        (
            "NIKKOR Z MC 105mm f/2.8 VR S + 61mm extension",
            105.0, 2.8,
        ),
        # Zoom: 24-120mm — only one 'mm' (after the range end).
        # The regex sees "100" with "-" after, not "mm", so the only
        # match is "120".
        (
            "NIKKOR Z 24-120mm f/4 S",
            120.0, 4.0,
        ),
        (
            "NIKKOR Z 24-120mm f/4 S + 26mm extension",
            120.0, 4.0,
        ),
        # 100-400mm — same pattern, only the long end matches.
        (
            "NIKKOR Z 100-400mm f/4.5-5.6 VR S",
            400.0, 4.5,
        ),
        (
            "NIKKOR Z 100-400mm f/4.5-5.6 VR S + Nikon 5T",
            400.0, 4.5,
        ),
        (
            "NIKKOR Z 100-400mm f/4.5-5.6 VR S Z TC-2.0x",
            400.0, 4.5,
        ),
        # AF-D lens — "f/4D" suffix should still parse to f_ring=4.0.
        # This was the original bug in the lookahead-based regex.
        (
            "Nikon AF Micro-Nikkor 200mm f_4D IF-ED",
            200.0, 4.0,
        ),
        (
            "Nikon AF Micro-Nikkor 200mm f_4D IF-ED + 6T + 5T",
            200.0, 4.0,
        ),
        # AF zoom with decimal f-stops.
        (
            "Nikon AF Zoom-Micro Nikkor 70-180mm f_4.5-5.6D ED",
            180.0, 4.5,
        ),
        # CPU macro from a different brand.
        (
            "Sigma APO Macro 150mm F2.8 EX DG OS HSM",
            150.0, 2.8,
        ),
        (
            "Sigma Macro 105mm F2.8 EX DG OS HSM + 6T + 5T",
            105.0, 2.8,
        ),
        # Third-party Z lens — different "EVO" suffix shouldn't trip
        # the f-ring regex (no false match on EVO's "E").
        (
            "Viltrox 85mm f/2 EVO",
            85.0, 2.0,
        ),
        (
            "Viltrox 85mm f/2 EVO + 26mm extension",
            85.0, 2.0,
        ),
    ],
)
def test_parse_lens_slug_reshoot_plan(slug, expected_focal, expected_fring):
    focal, f_ring = parse_lens_slug(slug)
    assert focal == expected_focal, (
        f"focal mismatch for slug={slug!r}: got {focal}, want {expected_focal}"
    )
    assert f_ring == expected_fring, (
        f"f_ring mismatch for slug={slug!r}: got {f_ring}, want {expected_fring}"
    )


def test_parse_lens_slug_missing_components():
    """Slugs without focal-length or f-stop markings return None for
    the missing component. Caller falls back to interactive prompts."""
    focal, f_ring = parse_lens_slug("Some random lens name")
    assert focal is None
    assert f_ring is None


def test_parse_lens_slug_focal_only():
    focal, f_ring = parse_lens_slug("Some 50mm lens with no f-stop")
    assert focal == 50.0
    assert f_ring is None


def test_parse_lens_slug_fring_only():
    focal, f_ring = parse_lens_slug("Lens at f/2.8 with no focal marking")
    assert focal is None
    assert f_ring == 2.8


def test_parse_lens_slug_ignores_af_capital_F():
    """The 'F' in 'AF' must not match the f-ring regex. The slug
    'Nikon AF Nikkor 50mm' has no f-stop marking — parse should return
    f_ring=None even though the slug contains 'F'."""
    focal, f_ring = parse_lens_slug("Nikon AF Nikkor 50mm")
    assert focal == 50.0
    assert f_ring is None


# ---- round_to_1_3_stop_label ----------------------------------------------


@pytest.mark.parametrize(
    "f_number,expected",
    [
        # Bellows-corrected wide-open for 55 AIS + PK-13 at 1:1 →
        # 2.8 × 1.5 = 4.2. Between scale entries 4.0 and 4.5;
        # log distance to 4.0 is 0.0704 stops, to 4.5 is 0.0986 →
        # closer to 4.0.
        (4.2, "4"),
        # Ring f/4 at 1:1 + PK-13 → 4 × 1.5 = 6.0. Between 5.6 and
        # 6.3; log distance to 5.6 is 0.103, to 6.3 is 0.072 → 6.3.
        (6.0, "6.3"),
        # Ring f/5.6 at 1:1 + PK-13 → 5.6 × 1.5 = 8.4. Between 8 and
        # 9; log distance to 8 is 0.070, to 9 is 0.100 → 8.
        (8.4, "8"),
        # Ring f/8 at 1:1 + PK-13 → 8 × 1.5 = 12.0. Between 11 and
        # 13; log distance to 11 is 0.123, to 13 is 0.115 → 13.
        (12.0, "13"),
        # Ring f/11 at 1:1 + PK-13 → 11 × 1.5 = 16.5. Between 16
        # and 18; log distance to 16 is 0.044, to 18 is 0.125 → 16.
        (16.5, "16"),
        # Exact scale values map to themselves.
        (2.8, "2.8"),
        (4.0, "4"),
        (5.6, "5.6"),
        (8.0, "8"),
        (11.0, "11"),
        (16.0, "16"),
    ],
)
def test_round_to_1_3_stop_label_known_cases(f_number, expected):
    label = round_to_1_3_stop_label(
        f_number, scale=config.ONE_THIRD_STOP_SCALE
    )
    assert label == expected


def test_round_to_1_3_stop_label_clamps_low():
    """Below the scale's minimum, snaps to the minimum (no extrapolation
    below the supported range)."""
    label = round_to_1_3_stop_label(
        0.5, scale=config.ONE_THIRD_STOP_SCALE
    )
    assert label == "1.4"


def test_round_to_1_3_stop_label_clamps_high():
    """Above the scale's maximum, snaps to the maximum."""
    label = round_to_1_3_stop_label(
        100.0, scale=config.ONE_THIRD_STOP_SCALE
    )
    assert label == "32"


def test_round_to_1_3_stop_label_rejects_nonpositive():
    with pytest.raises(ValueError):
        round_to_1_3_stop_label(0.0, scale=config.ONE_THIRD_STOP_SCALE)
    with pytest.raises(ValueError):
        round_to_1_3_stop_label(-1.0, scale=config.ONE_THIRD_STOP_SCALE)


def test_round_to_1_3_stop_label_rejects_empty_scale():
    with pytest.raises(ValueError):
        round_to_1_3_stop_label(4.0, scale=[])


# ---- aperture_sequence_for_manual_mode ------------------------------------


def test_manual_sequence_55_ais_pk13_1_to_1():
    """55 AIS f/2.8 + PK-13 at 1:1 (m_indicated=0.5 since lens is at
    its own 1:2 MFD).

    Ring walks 2.8, 4, 5.6, 8, 11; ring f/16 would yield effective
    24 → label "25" > MAX_FSTOP=16, so it's dropped. Ring f/11's
    effective 16.5 rounds to label "16" (= MAX_FSTOP), so it is
    included — the bound is on the rounded *label*, not the raw
    effective."""
    seq = aperture_sequence_for_manual_mode(
        f_ring_max=2.8,
        m_indicated=0.5,
        whole_stop_scale=config.WHOLE_STOP_SCALE,
        max_fstop=config.MAX_FSTOP,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    # 2.8 × 1.5 = 4.2 → "4"
    # 4.0 × 1.5 = 6.0 → "6.3"
    # 5.6 × 1.5 = 8.4 → "8"
    # 8.0 × 1.5 = 12.0 → "13"
    # 11.0 × 1.5 = 16.5 → label "16" (≤ MAX_FSTOP)
    # 16.0 × 1.5 = 24.0 → label "25" → dropped
    assert seq == ["4", "6.3", "8", "13", "16"]


def test_manual_sequence_55_ais_no_tubes_1_to_2():
    """55 AIS at its own 1:2 MFD without tubes: m_indicated == m_total
    == 0.5. Same effective labels as the 1:1 + PK-13 case because the
    lens itself is in the same internal configuration."""
    seq = aperture_sequence_for_manual_mode(
        f_ring_max=2.8,
        m_indicated=0.5,
        whole_stop_scale=config.WHOLE_STOP_SCALE,
        max_fstop=config.MAX_FSTOP,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    assert seq == ["4", "6.3", "8", "13", "16"]


def test_manual_sequence_55_f35_pk13_equivalent():
    """55 f/3.5 + K2+K3+K5 (~30 mm tube) at 1:1 — lens at its own
    1:2 MFD, m_indicated=0.5.

    Note: 3.5 isn't on the WHOLE_STOP_SCALE, so f_ring_max=3.5 is
    the widest emitted, then the scale takes over from 4.0 upward.
    """
    seq = aperture_sequence_for_manual_mode(
        f_ring_max=3.5,
        m_indicated=0.5,
        whole_stop_scale=config.WHOLE_STOP_SCALE,
        max_fstop=config.MAX_FSTOP,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    # 3.5 × 1.5 = 5.25 → "5"
    # 4.0 × 1.5 = 6.0 → "6.3"
    # 5.6 × 1.5 = 8.4 → "8"
    # 8.0 × 1.5 = 12.0 → "13"
    # 11.0 × 1.5 = 16.5 → "16"
    assert seq == ["5", "6.3", "8", "13", "16"]


def test_manual_sequence_no_extension_unit_focus():
    """Lens at infinity / no extension: m_indicated=0. Effective ==
    geometric ring. Sequence matches WHOLE_STOP_SCALE walked from
    f_ring_max upward."""
    seq = aperture_sequence_for_manual_mode(
        f_ring_max=2.8,
        m_indicated=0.0,
        whole_stop_scale=config.WHOLE_STOP_SCALE,
        max_fstop=config.MAX_FSTOP,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    assert seq == ["2.8", "4", "5.6", "8", "11", "16"]


def test_manual_sequence_rejects_bad_inputs():
    with pytest.raises(ValueError):
        aperture_sequence_for_manual_mode(
            f_ring_max=0.0,
            m_indicated=0.5,
            whole_stop_scale=config.WHOLE_STOP_SCALE,
            max_fstop=config.MAX_FSTOP,
            one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
        )
    with pytest.raises(ValueError):
        aperture_sequence_for_manual_mode(
            f_ring_max=2.8,
            m_indicated=-2.0,  # (1+m) <= 0
            whole_stop_scale=config.WHOLE_STOP_SCALE,
            max_fstop=config.MAX_FSTOP,
            one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
        )


def test_manual_sequence_dedups_labels():
    """If two ring stops round to the same effective label, dedupe."""
    seq = aperture_sequence_for_manual_mode(
        f_ring_max=2.8,
        m_indicated=0.0,
        whole_stop_scale=(2.8, 2.9),  # both round to "2.8" at m=0
        max_fstop=16.0,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    assert seq == ["2.8"]
