"""Pure helpers used by manual-aperture mode (non-CPU lenses).

Two utilities, both pure (no I/O) so they unit-test cleanly:

* :func:`parse_lens_slug` — extract ``focal_length_mm`` and ``f_ring``
  from a lens-config slug via regex. The slugs in
  ``analyse-lens-test/reshoot-plan-2026-05.md`` and the
  ``test-raw-data`` directory tree all follow ``<...>Nmm<...>fN<...>``
  or ``<...>Nmm<...>F_N<...>``; this returns ``(focal, f_ring)`` or
  ``(None, None)`` on miss so the runner can fall back to prompts.

* :func:`round_to_1_3_stop_label` — snap a float f-number to the nearest
  entry on :data:`config.ONE_THIRD_STOP_SCALE` by log-distance. Used
  to compute file-label apertures for non-CPU mode: the user reads
  ``m_indicated`` off the lens barrel, the rig computes
  ``f_ring × (1 + m_indicated)``, this rounds the result to the
  standard scale (matching what gphoto reports for CPU lenses).
"""

from __future__ import annotations

import math
import re
from typing import Iterable


# ---- Slug parsing ---------------------------------------------------------

# Focal length: integer or decimal, immediately followed by "mm" (case
# insensitive). Uses the first match — slugs that mention multiple
# focal lengths (zoom + tube length, e.g. "Z 24-120mm + 26mm extension")
# need a smarter rule, see below.
_FOCAL_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)

# Max aperture marked on the lens: "f/2.8", "f_2.8", "F2.8", "F/2.8".
# Anchored on F/f followed by an optional slash or underscore so it
# matches the slug form "f_2.8" as well as the canonical "f/2.8".
# The digit group greedily captures the f-number; for zoom ranges like
# "f/4.5-5.6" we take the first number (the widest end of the zoom —
# the rig captures at the longest focal length where the lens is at
# its narrowest max). No lookahead so AF-D lenses' "f/4D" suffix
# doesn't break the match.
_FRING_RE = re.compile(r"(?<![A-Za-z])[Ff][/_]?(\d+(?:\.\d+)?)")


def parse_lens_slug(slug: str) -> tuple[float | None, float | None]:
    """Extract ``(focal_length_mm, f_ring)`` from a lens-config slug.

    The regex looks for the **last** ``<N>mm`` token, not the first.
    Slugs like ``"Nikon Z 24-120mm f/4 S + 26mm extension"`` mention
    a 26 mm extension tube *after* the lens's own focal length; we
    want the lens's focal length (120 here, the longer end of the
    zoom that the rig is using), not the tube. Picking the last
    match is wrong for the zoom-with-tube case but right for the
    prime-with-tube case. The user can override via the fallback
    prompt in :mod:`ui.prompts` either way.

    Returns ``(None, None)`` or a partial tuple if either component
    isn't found; caller falls back to prompts.
    """
    focal = _last_focal(slug)
    f_ring = _first_fring(slug)
    return focal, f_ring


def _last_focal(slug: str) -> float | None:
    matches = _FOCAL_RE.findall(slug)
    if not matches:
        return None
    # Prefer the *first* focal in the slug — that's the lens itself.
    # Tubes / extensions almost always follow the lens name. For zoom
    # lenses like "24-120mm" the first regex match grabs "120" because
    # the lookbehind on (?<![\d.]) rejects "24" once a "-" hasn't been
    # consumed by another match. Empirically, take the first match.
    try:
        return float(matches[0])
    except ValueError:
        return None


def _first_fring(slug: str) -> float | None:
    match = _FRING_RE.search(slug)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# ---- Aperture-label rounding ----------------------------------------------


def round_to_1_3_stop_label(
    f_number: float, *, scale: Iterable[float],
) -> str:
    """Snap a float f-number to the nearest entry on ``scale`` by
    log-distance, return as a label string formatted like the
    existing CPU EXIF-derived labels (no leading 'F', no trailing
    zero on whole stops, single decimal otherwise).

    Examples (assuming ``scale = config.ONE_THIRD_STOP_SCALE``)::

        round_to_1_3_stop_label(4.2)  -> "4"      # 4.0 is closest by log dist
        round_to_1_3_stop_label(6.0)  -> "6.3"
        round_to_1_3_stop_label(8.4)  -> "8"
        round_to_1_3_stop_label(12.0) -> "11"
        round_to_1_3_stop_label(2.8)  -> "2.8"

    Tie behaviour: log-equidistant rounds *down* (smaller f-number),
    which is the conservative direction for the aperture sequence
    start in manual mode (don't skip a usable wider aperture).
    """
    scale_list = list(scale)
    if not scale_list:
        raise ValueError("scale must be non-empty")
    if f_number <= 0:
        raise ValueError(f"f_number must be > 0; got {f_number}")
    log_target = math.log2(f_number)
    # Find smallest log-distance; on tie prefer the smaller f-number
    # (earlier index in a sorted-ascending scale).
    sorted_scale = sorted(scale_list)
    best_value = sorted_scale[0]
    best_dist = abs(math.log2(best_value) - log_target)
    for value in sorted_scale[1:]:
        dist = abs(math.log2(value) - log_target)
        if dist < best_dist:
            best_value = value
            best_dist = dist
    return _format_label(best_value)


def _format_label(value: float) -> str:
    """Format an aperture float as it appears in the existing data
    folders: whole stops without trailing zero (``"4"``, ``"8"``),
    half/third stops with one decimal (``"4.5"``, ``"5.6"``)."""
    if value == int(value):
        return str(int(value))
    # One decimal place; strip a trailing zero if it slipped in.
    text = f"{value:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return text


# ---- Synthetic aperture sequence (manual mode) ----------------------------


def ring_and_label_sequence(
    *,
    f_ring_max: float,
    m_indicated: float,
    whole_stop_scale: Iterable[float],
    max_fstop: float,
    one_third_stop_scale: Iterable[float],
) -> list[tuple[float, str]]:
    """Build the (ring_stop, effective_label) pair list for manual mode.

    Walks the ring whole stops from ``f_ring_max`` upward, stopping
    when the next ring stop's effective aperture would exceed
    ``max_fstop``. For each surviving ring stop, the effective label
    is computed as ``f_ring × (1 + m_indicated)`` snapped to the
    1/3-stop scale. Labels are deduped (rare but possible when two
    nearby ring stops round to the same effective).

    Used by the runner to drive per-aperture flash-power
    calculations (the runner needs the geometric ring f-number to
    compute EV stops, separate from the effective label that goes
    in the file path).
    """
    if f_ring_max <= 0:
        raise ValueError(f"f_ring_max must be > 0; got {f_ring_max}")
    factor = 1.0 + m_indicated
    if factor <= 0:
        raise ValueError(
            f"m_indicated yields non-positive bellows factor; "
            f"got m={m_indicated}"
        )
    scale_sorted = sorted(whole_stop_scale)
    one_third_list = list(one_third_stop_scale)
    ring_stops: list[float] = [f_ring_max]
    for stop in scale_sorted:
        if stop > f_ring_max:
            ring_stops.append(stop)
    out: list[tuple[float, str]] = []
    seen_labels: set[str] = set()
    for ring_stop in ring_stops:
        effective = ring_stop * factor
        label = round_to_1_3_stop_label(effective, scale=one_third_list)
        # Bound the sequence by the *rounded* label rather than the
        # raw effective value: at the 55 AIS f/2.8 + PK-13 1:1 case
        # ring f/11 has effective f/16.5 which would over-shoot
        # max_fstop=16, but rounds to label "16" (≤ max_fstop). The
        # user's intent is "cap the file labels at f/16", so checking
        # the label is the principled bound.
        try:
            if float(label) > max_fstop:
                break
        except ValueError:
            # Unparseable label (shouldn't happen). Fall back to raw
            # effective check.
            if effective > max_fstop:
                break
        if label in seen_labels:
            continue
        seen_labels.add(label)
        out.append((ring_stop, label))
    return out


def aperture_sequence_for_manual_mode(
    *,
    f_ring_max: float,
    m_indicated: float,
    whole_stop_scale: Iterable[float],
    max_fstop: float,
    one_third_stop_scale: Iterable[float],
) -> list[str]:
    """Build the aperture sequence for a non-CPU manual-ring lens.

    Unlike :func:`session.runner.aperture_sequence` (which iterates
    *camera-reported choices*), this iterates **ring whole stops**
    starting from ``f_ring_max``. For each ring stop, the
    *effective* aperture is ``f_ring × (1 + m_indicated)``; the rig
    captures at that effective value and stores it as the file
    label, snapped to the 1/3-stop scale.

    Stops when the next ring stop would yield an effective aperture
    exceeding ``max_fstop``. Returns a list of label strings (no
    leading ``"f/"``) in widest-first order, matching the convention
    used by :data:`session.log.SHOT_COLUMNS`'s ``aperture`` column.

    Example for the 55 AIS f/2.8 at 1:1 + PK-13
    (f_ring_max=2.8, m_indicated=0.5, max_fstop=16):

    * ring 2.8 → eff 4.2 → label "4"
    * ring 4.0 → eff 6.0 → label "6.3"
    * ring 5.6 → eff 8.4 → label "8"
    * ring 8.0 → eff 12.0 → label "11"
    * ring 11.0 → eff 16.5 → > 16, stop.

    Returned: ``["4", "6.3", "8", "11"]``.

    With ``f_ring_max < min(whole_stop_scale)`` (rare; e.g. an f/1.2
    lens with WHOLE_STOP_SCALE starting at 1.4), the f_ring_max
    itself is emitted as the widest label.
    """
    pairs = ring_and_label_sequence(
        f_ring_max=f_ring_max,
        m_indicated=m_indicated,
        whole_stop_scale=whole_stop_scale,
        max_fstop=max_fstop,
        one_third_stop_scale=one_third_stop_scale,
    )
    return [label for _, label in pairs]
