"""Tunable constants for macro lens sharpness test sessions.

Anything that might reasonably change between rigs, camera bodies, or
test campaigns lives here. The runner imports values from this module
directly; no env-var or CLI override layer in v1.
"""

from pathlib import Path

# ---- Aperture sequence ------------------------------------------------------

MAX_FSTOP = 16.0
WHOLE_STOP_SCALE = (1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0)

# Standard ⅓-stop scale used by manual-aperture mode's label rounding.
# Matches the values gphoto2 reports for CPU lenses (camera f-number
# settings). Computed-effective apertures (f_ring × (1 + m_indicated))
# are snapped to the closest entry here by log-distance so file labels
# stay drop-in compatible with the CPU EXIF convention.
ONE_THIRD_STOP_SCALE = (
    1.4, 1.6, 1.8,
    2.0, 2.2, 2.5,
    2.8, 3.2, 3.5,
    4.0, 4.5, 5.0,
    5.6, 6.3, 7.1,
    8.0, 9.0, 10.0,
    11.0, 13.0, 14.0,
    16.0, 18.0, 20.0,
    22.0, 25.0, 29.0,
    32.0,
)

# Per-aperture range extension. (iso, recycle_seconds).
# ISO is the absolute base ISO at that aperture (NOT a stop bump).
# Recycle delay absorbs flash recycle and any aperture-mechanism settle.
RANGE_EXTENSION: dict[float, tuple[int, float]] = {
    2.0:  (100, 3.0),
    2.8:  (100, 3.0),
    4.0:  (100, 3.0),
    5.6:  (100, 3.0),
    8.0:  (100, 4.0),
    11.0: (200, 5.0),
    16.0: (400, 6.0),
    22.0: (800, 7.0),
}

# ---- ISO --------------------------------------------------------------------

ISO_FLOOR = 100
ISO_WARN = 1600
ISO_CEIL = 3200
ISO_STEP_EV = 1.0 / 3.0

# ---- Manual-aperture mode (non-CPU lens, flash-M cascade) -----------------
# Used only when --manual-aperture is enabled. The flash is in M mode
# on the SB-5000 commander; FEC is inert; ISO and flash power are the
# two exposure knobs. The rig prompts the user to dial in flash power
# on the commander between apertures; ISO is set programmatically as
# fine-tune and ceiling compensator.

MANUAL_ISO_FLOOR = 100
"""Floor ISO used by manual-aperture mode. ISO 100 is the lowest
non-extended sensitivity on the Z7; staying at-or-above 100 avoids
the Lo 1 (extended-low ISO 50) which has slightly worse colour
response. The cascade prefers to absorb over-exposure via flash
power drops (down to 1/128) before considering ISO. If even 1/128
is still too bright, the cascade aborts with a "reduce ambient
light" message rather than silently falling to ISO 50 — that's a
rig-setup problem the operator should fix, not the algorithm."""

MANUAL_FLASH_SOFT_CAP_STOPS = 1.0
"""Preferred maximum flash power as stops below 1/1. 1.0 = 1/2 power.
Keeps the flash in its comfortable operating zone (faster recycle,
less battery drain, headroom for droop). The rig prefers bumping ISO
up to MANUAL_FLASH_ISO_RELAX_THRESHOLD before exceeding this cap."""

MANUAL_FLASH_HARD_CAP_STOPS = 0.0
"""Maximum-allowed flash power as stops below 1/1. 0.0 = 1/1 full.
Reached only when ISO would otherwise exceed ISO_WARN. Above this
the cascade can only push ISO further (up to ISO_CEIL)."""

MANUAL_FLASH_INITIAL_POWER_STOPS = 5.0
"""Starting flash power for the coarse search on the first aperture.
5.0 = 1/32 power. Biased low so the first capture is more likely
under-exposed than over: ISO has more room to go up (50→3200, 6
stops) than down (50→50, 0 stops on the Z7's extended Lo 1 range),
and the coarse search will halve / double from here. Empirically
1/16 is roughly correct for the 55 AIS f/2.8 + PK-13 at 1:1 —
starting at 1/32 lands one stop under, which the +1-stop coarse-
search adjustment corrects in a single iteration."""

MANUAL_FLASH_SEARCH_TOLERANCE_EV = 1.0
"""Coarse-search exit threshold. Once the test shot's EV delta from
baseline is within ±this value, the first-aperture path stops
halving / doubling power and switches to ISO fine-tune."""

MANUAL_FLASH_BUMP_SANITY_EV = 0.5
"""After the user sets the rig-prescribed power for a subsequent
aperture, the first capture's EV delta from baseline must lie within
±this value. Wider deviation suggests the user dialled the wrong
power or didn't turn the ring; the cascade prompts the user to
verify before proceeding."""

MANUAL_FLASH_MAX_SEARCH_ITERATIONS = 15
"""Safety termination for the first-aperture coarse search. The
SB-5000 range is 7 stops (1/1..1/128); 15 iterations is enough for
bidirectional search with margin. Exceeding this aborts with a
'subject too bright/dim — adjust lighting' prompt."""

MANUAL_FLASH_TOLERANCE_EV = 0.2
"""Per-shot exposure tolerance for the manual-aperture flash-M
cascade — tighter than :data:`TEST_SHOT_TOLERANCE_EV` (0.5 EV, used
by auto-mode AA calibration where the closed-loop FEC absorbs most
error). Manual mode's predictive solver can miss by ~0.25 EV when
the simple `(1+m)²` bellows model or the flash IGBT timing drift
from linear assumptions; this tolerance triggers an ISO fine-tune
retry when that happens. Cost is at most one extra capture per
aperture; benefit is consistent exposure across the sweep."""

# ---- FEC (Z7 body range, SB-5000 commander) ---------------------------------

FEC_MIN_EV = -3.0
FEC_MAX_EV = +1.0
FEC_STEP_EV = 1.0 / 3.0

# ---- EC (camera exposure compensation, ⅓-stop) -----------------------------
# On the Z7 with custom function e3 set to "Entire frame", camera EC
# stacks additively with FEC for flash output. Calibration uses EC as
# *fallback* once FEC saturates at FEC_MAX_EV but the rig still hasn't
# reached the calibration target — extending the camera-side flash
# headroom by another EC_MAX_EV stops before falling back to ISO.
EC_MIN_EV = -3.0
EC_MAX_EV = +3.0
EC_STEP_EV = 1.0 / 3.0

# ---- Plateau detection (when ISO is actually useful) ----------------------
# In AA flash mode, ISO doesn't change image brightness — it just shifts
# the flash's auto-quench point. ISO only buys real brightness when the
# flash is at its mechanical maximum (firing at full power, full duration,
# can't put out more). The calibrator detects that condition by checking
# whether luminosity tracks the cumulative FEC+EC bumps it asked for.
# Cumulative response < threshold × cumulative request, after at least
# min_ev of bumping, → plateau → ISO range-extend.
PLATEAU_DETECT_THRESHOLD = 0.5  # ratio
PLATEAU_DETECT_MIN_EV = 1.0     # don't fire on a single ⅓-stop bump

# Flash-headroom swap. After calibration converges *and* ISO was
# bumped (a positive plateau detection — the flash was at its
# mechanical maximum during cascade), the calibrator does a final
# bias-↔-ISO swap: trade ``ISO_HEADROOM_EV`` of FEC/EC bias for the
# same EV in ISO. The image stays at the same brightness, but the
# flash now fires below its mechanical max during the sweep, which
# leaves margin for battery droop and reduces stress on the units.
# A verification capture confirms the swap landed within tolerance;
# otherwise the calibrator reverts.
ISO_HEADROOM_EV = 1.0

# ---- Linear AA cascade -----------------------------------------------------
# Replaces the legacy iterative cascade (greedy ``new_fec = fec - delta``
# loop with plateau detection and FEC saturation fallback chain) with a
# 3-5 shot linear-model cascade: probe at FEC=0, apply -delta, headroom-
# probe, then either fine-tune or bump ISO. See
# :meth:`Calibrator._calibrate_aperture_linear` for the algorithm.
#
# Enabled by default. The legacy iterative cascade is preserved in
# :meth:`Calibrator._calibrate_aperture` and is reachable by setting
# this flag to ``False``.
LINEAR_CASCADE_ENABLED = True

# ---- Sample planner --------------------------------------------------------
# Minimum rail positions for a usable sweep. The advisor's bell-curve
# detector needs at least ~10 points across the captured range to land
# a confident peak / truncation verdict at the widest aperture's narrow
# DOF. If the planner's step size × jogged range yields fewer positions
# than this, the runner refuses and prompts the user to re-jog wider.
MIN_RAIL_POSITIONS = 10

# ---- Camera startup ---------------------------------------------------------

MIN_CONFIG_TREE_SIZE = 100
IMAGE_QUALITY = "JPEG Fine"
SHUTTER_SPEED_CANDIDATES = ("1/200", "0.0050s", "0.005s")
FEC_SANITY_LOW_EV = -2.0
FEC_SANITY_MIN_DROP_EV = 1.0
"""Minimum scene-light EV drop expected between FEC=0 and FEC=-2.0
shots, computed from the gamma-linearised central-crop medians.
Default 1.0 leaves generous margin for ambient-light fill at the
central crop while still catching a totally non-responsive setup
(SU-800 mounted instead of SB-5000 commander, or flashes not in
remote mode), which would show drop ≈ 0."""

# ---- Calibration baseline ----------------------------------------------------

BASELINE_REFERENCE_IMAGE: "Path | None" = Path(
    "/home/alson/shared/files/pics/digikam/test-raw-data/2026-01-29/"
    "Nikon AF Micro-Nikkor 200mm f_4D IF-ED/1_1/F5.6/"
    "00188|Nikon AF Micro-Nikkor 200mm f_4D IF-ED|1_1|200.0 mm|F5.6|"
    "alm-20260129-160552-188.jpg"
)
"""Path to a reference JPEG used as the absolute calibration target.
The runner measures its central-crop median (using the same
CROP_*_FRAC values below) at session start and uses that value as
the baseline for the FEC closed-loop calibration on every lens
config.

Without this, each session falls back to "first shot at the widest
aperture of the first lens config of the session" as the baseline —
which is fine for within-session comparability but makes inter-
session comparability lens-config-dependent (e.g. a session that
opens with a 105 mm macro at 1:1 + extension tubes will land at a
different baseline than one that opens with no extension, even
though the wafer reflectance is the same).

Pick a representative shot from a previous successful session that
captures the wafer brightness you want all future sessions to
target. The runner will read its central-crop median once and that
becomes the absolute reference."""

# ---- Exposure analysis (central crop on the wafer) --------------------------

CROP_W_FRAC = 0.20
CROP_H_FRAC = 0.20
CROP_X_FRAC = 0.00
CROP_Y_FRAC = 0.15  # positive = below image centre
GAMMA = 2.2

TEST_SHOT_TOLERANCE_EV = 0.5
"""Per-shot tolerance for FEC/EC closed-loop calibration. The
sharpness analysis depends on inter-aperture exposure parity, not
absolute baseline match, so ½-stop is a comfortable floor —
tighter than this just consumes retries on diminishing-returns
cascade steps."""

TEST_SHOT_MAX_RETRIES = 10
"""Each retry advances the cascade by one nudge (FEC up, EC up,
ISO up on plateau, or any combination). With a 3 EV initial
deficit the cascade easily consumes 6+ steps; 10 leaves room for
a plateau-driven ISO bump to settle."""

# ---- Droop detector ---------------------------------------------------------

ROLLING_WINDOW = 10
# Droop trip criteria — evaluated in order; whichever fires first
# wins. The detector pools pushes session-wide (across every
# aperture in the sweep), so consecutive under-exposed shots at
# *different* apertures still trip the moderate criterion. Severe
# fires on a single shot — typical "flash refused to fire / hit
# its mechanical-min" failure mode.
DROOP_CRIT_A = (1, -2.0)   # 1 severe shot ≤ -2 EV — fire immediately
DROOP_CRIT_B = (2, -0.7)   # 2 of last 10 ≤ -0.7 EV — gradual drift

# ---- Range-too-narrow auto-extension ---------------------------------------

RANGE_CHECK_ENABLED = True
RANGE_EXTENSION_FRACTION = 0.5
RANGE_CHECK_PROBE_STEPS = 2

# ---- Coarse-to-fine focus probe --------------------------------------------
# Two-pass adaptive sweep: a sparse phase-A probe at the widest aperture
# locates the per-crop peak band, then a narrower phase-C window captures
# all apertures at full density. Replaces the uniform user-jog → full-sweep
# flow when enabled. The post-sweep RangeChecker (phase D) is unchanged
# and handles residual high-FC outliers.
#
# See ~/.claude/plans/imperative-sprouting-biscuit.md and
# tools/probe_simulator.py (calibration tooling).
PROBE_ENABLED = True
PROBE_PHASE_A_POINTS = 9
"""Phase-A probe density across the user-jogged range (single
aperture, single replica). Validated against two regimes:

**Live wide-jog regime** (3 calibration sessions, p95-max field
curvature, jog ≈ 9-20 × FWHM):

* N=3: 8 % rel-loss (unsafe).
* N=5-7: 2 % rel-loss on a p95 corner.
* **N=9 (chosen)**: best savings on the worst-FC clean case
  (44 % vs 38 % at N=7); same loss profile.
* N=11+: marginally worse savings (-2 to -3 %) for no safety gain.

**Older corpus regime** (136 scenarios from 2025-07 → 2026-02 in
``test-raw-data/``, validated via ``tools/probe_replay_corpus.py``,
typical jog ≈ 5 × FWHM — tighter than live):

* Phase A goes inconclusive on 51-80 % of scenarios (sparse phase-A
  samples on a tight jog often can't pass the dynamic-range gate).
  Algorithm degrades gracefully: falls back to today's full uniform
  sweep + the wasted phase-A captures.
* Cross-aperture amortized overhead at N=9: ~-2 % on the median
  scenario — bounded, not catastrophic.
* Success cases (top ~30 % of corpus) save 30-40 % regardless of N.
* Rel-loss > 2 %: 2 / 136 at N=9 (caught by phase D extension).

So phase A only pays off when users actually jog wide. The design
intent assumes users will (since jogging-wide is now cheap), but if
they jog tight the cost is bounded ~2 % per lens config.
"""

PROBE_PHASE_C_FWHM_PADDING = 1.5
"""Padding × FWHM_widest on each side of the observed crop-peak
bracket. Phase C window =
``[min_peak − PADDING × FWHM, max_peak + PADDING × FWHM]``.
1.5 was calibrated against a 327-scenario field-curvature corpus
(``tools/curvature_survey.py``): holds rel-loss ≤ 2 % on every
measured live session, projects ~74 % median savings, fires phase D
on ~0.3 % of corpus configs."""

# ---- Sharpness pipeline ----------------------------------------------------

SHARPNESS_MAX_WORKERS = 8
"""Concurrent crop workers per LiveCollector. Production uses a
ProcessPoolExecutor (true parallel libjpeg encode across cores), so
this should match the laptop's logical core count. The bench-rig
i5-8350U has 8 logical cores; 4 was the previous ThreadPool default
and bottlenecked at ~3× speedup because of GIL contention between
encodes."""

SHOTS_PER_STEP_APERTURE = 3
"""Number of frames captured at each (rail step, aperture) pair.

Replica 0 is saved as ``step_NNNN.jpg`` (no suffix) so existing
analysis tooling that expects single-shot filenames keeps working.
Replicas 1..N-1 are saved as ``step_NNNN_b.jpg``, ``step_NNNN_c.jpg``,
… The advisor aggregates by-(step, aperture) median across replicas
before peak / truncation detection so JPEG-encoding noise doesn't
swamp a weak focus signal. Set to 1 to revert to single-shot."""

SHARPNESS_CROPPED_DIRNAME = "cropped"
"""Subdirectory name under each <lens>/<aperture>/ directory where the
17 crops are persisted. Matches the user's offline-analysis layout."""

SHARPNESS_CROP_DYNAMIC_RANGE_MIN = 0.25
"""Upper threshold for the per-crop signal classifier as
``(max - min) / max`` over the rail series. Crops at or above this
dynamic range are treated as high-signal: a real peak in the middle
of the range becomes :data:`CropStatus.OK`.

The wafer doesn't necessarily fill the frame, so corner crops can
fall off onto a low-contrast background; their JPEG sizes barely
move across the focus sweep because there's nothing to focus on.
A defocused wafer crop, by contrast, swings dramatically (typical
on-target peak/trough ratios of 1.5–1.8 ⇒ dynamic range 0.33–0.44).
0.25 sits comfortably below that band and well above empty-frame
noise (typically ≤ 0.10).

Crops with dynamic range between
:data:`SHARPNESS_CROP_OFF_TARGET_RANGE` and this value are classified
as ``LOW_SIGNAL`` — a real but weak bell. The aggregator treats those
like ``OK`` for the per-aperture verdict (a peak was found), while
still reporting the count so the operator can spot rigs where every
corner is barely above the noise floor."""

SHARPNESS_CROP_OFF_TARGET_RANGE = 0.05
"""Lower threshold for the per-crop signal classifier. Crops with
``(max - min) / max`` strictly below this value are short-circuited to
``OFF_TARGET`` without peak / truncation detection — the window almost
certainly fell off the wafer onto featureless background.

Empirical motivation: the Viltrox 85mm f/2 EVO + 61mm extension combo
(2026-05-10 rig session) produced corner crops with max/min ratios of
1.04–1.22 (dynamic range 0.04–0.18) and clearly-visible bell shapes.
The old single-threshold logic (everything below 0.25 → OFF_TARGET)
classified every such corner as off-wafer, hiding the genuine soft
bells. Splitting the threshold into a low ``OFF_TARGET`` floor and a
higher ``OK`` ceiling lets soft-but-real bells be reported as
``LOW_SIGNAL`` instead. 0.05 sits comfortably above empty-frame JPEG
noise while still capturing the dimmest legitimate corners observed
on the rig."""

SHARPNESS_PARTIAL_FRAME_FACTOR = 0.80
"""Cutoff for the absolute-size peer-compare partial-frame detector.
A crop whose peak file size is below this factor times its peer-group
median (4 corners vs 4 corners, 4 edges vs 4 edges) is classified
as :data:`CropStatus.PARTIAL_FRAME` — the wafer / target edge is
cutting through the crop window so part of the area is dark
background that dilutes the JPEG-size metric.

Distinct from :data:`SHARPNESS_CROP_OFF_TARGET_RANGE` (which catches
"no wafer at all" via low dynamic range). Partial-frame crops still
have real focus signal in the on-wafer portion, so dynamic range
remains roughly normal; the giveaway is *absolute* peak size falling
well below positionally similar peers. Operator remedy is
reframing, not re-jogging.

Empirical motivation: the 24-120 + 35 mm extension session
(2026-05-11_185420) framed the wafer's lower-left corner partly
outside the bottom-left crop window. bottom-left peaked at ~32 KB
while the other three corners peaked at 41-49 KB, giving
BL/peer-median ratios of 0.72-0.79 across all five apertures.
Non-partial corners on the same session and the sibling 24-120 +
45 mm session clustered at ≥ 0.95 of peer-group median, so 0.80
cleanly separates the two regimes — partial-frame fires
consistently across apertures while normal lens asymmetry stays
below the flag."""

# ---- Step-numbering for capture filenames ----------------------------------

STEP_NUMBER_OFFSET = 100
"""Numeric offset applied to the step index before formatting the
filename. Reserves 100 slots below the main sweep so near-side
extensions (negative step indices) become positive filenames like
``step_0099.jpg``, ``step_0098.jpg``."""

STEP_NUMBER_WIDTH = 4
"""Zero-padded width for the step number in the filename. With offset
100 and width 4 the maximum representable step index is 9899
(filename ``step_9999.jpg``)."""

# ---- StackShot motion -------------------------------------------------------

STACKSHOT_BAUD = 38400
STACKSHOT_INTERFRAME_S = 0.002
STACKSHOT_POLL_INTERVAL_S = 0.10
STACKSHOT_MOVE_TIMEOUT_S = 30.0
JOG_FINE_MM = 0.05
JOG_COARSE_MM = 1.0
POSITION_TOLERANCE_MM = 0.02

# ---- Paths ------------------------------------------------------------------

OUTPUT_ROOT = Path.home() / "macro-lens-test-output"
READLINE_HISTORY = Path("data/readline_history")
ALERT_WAV = Path("data/alert.wav")

RESHOOT_PLAN_PATH: "Path | None" = (
    Path(__file__).resolve().parent.parent / "analyse-lens-test"
    / "reshoot-plan-2026-05.md"
)
"""Path to the curated lens-config menu source. The runner offers
each row as a preset at the lens-config prompt, with a ``[done]``
marker for slugs that were successfully acquired in a past session.
Set to ``None`` (or rename / move the file) to disable the menu —
the runner falls through to the existing free-text + magnification
prompts."""

# ---- Notification -----------------------------------------------------------

NTFY_TOPIC = "macro-lens-H1wpeWs6Fxi1gudY"
NTFY_BASE_URL = "https://ntfy.sh"
