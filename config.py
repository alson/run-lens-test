"""Tunable constants for macro lens sharpness test sessions.

Anything that might reasonably change between rigs, camera bodies, or
test campaigns lives here. The runner imports values from this module
directly; no env-var or CLI override layer in v1.
"""

from pathlib import Path

# ---- Aperture sequence ------------------------------------------------------

MAX_FSTOP = 16.0
WHOLE_STOP_SCALE = (1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0)

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
FEC_SANITY_MIN_RATIO = 0.5  # FEC=-2 median must be ≤ 50% of FEC=0 median

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
DROOP_CRIT_A = (3, -0.7)   # 3 of last 10 ≤ -0.7 EV
DROOP_CRIT_B = (2, -2.0)   # 2 of last 10 ≤ -2.0 EV

# ---- Range-too-narrow auto-extension ---------------------------------------

RANGE_CHECK_ENABLED = True
RANGE_EXTENSION_FRACTION = 0.5
RANGE_CHECK_PROBE_STEPS = 2

# ---- Sharpness pipeline ----------------------------------------------------

SHARPNESS_MAX_WORKERS = 4
"""Concurrent crop workers per LiveCollector. Tune for the laptop's
core count vs flash recycle: 4 keeps a single 45 MP JPEG cropping run
just under typical 3 s recycle on the test rig."""

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
