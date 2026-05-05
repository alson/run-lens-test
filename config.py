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

# ---- Camera startup ---------------------------------------------------------

MIN_CONFIG_TREE_SIZE = 100
IMAGE_QUALITY = "JPEG Fine"
SHUTTER_SPEED_CANDIDATES = ("1/200", "0.0050s", "0.005s")
FEC_SANITY_LOW_EV = -2.0
FEC_SANITY_MIN_RATIO = 0.5  # FEC=-2 median must be ≤ 50% of FEC=0 median

# ---- Exposure analysis (central crop on the wafer) --------------------------

CROP_W_FRAC = 0.20
CROP_H_FRAC = 0.20
CROP_X_FRAC = 0.00
CROP_Y_FRAC = 0.15  # positive = below image centre
GAMMA = 2.2

TEST_SHOT_TOLERANCE_EV = 1.0 / 3.0
TEST_SHOT_MAX_RETRIES = 6

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

# ---- Notification -----------------------------------------------------------

NTFY_TOPIC = "macro-lens-H1wpeWs6Fxi1gudY"
NTFY_BASE_URL = "https://ntfy.sh"
