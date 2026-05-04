# Macro Lens Sharpness Test Automation — Implementation Plan

This plan is intended to be handed to Claude Code for implementation. It describes a Python program that controls a Nikon Z7 and an original (single-axis, text-LCD) StackShot focus rail to run automated lens sharpness tests at successive apertures, with battery-droop monitoring and remote notification to a MacBook.

---

## 1. Hardware & environment

- **Camera**: Nikon Z7 (1st gen), tethered by USB. Capture is triggered over USB by the program, not by the StackShot.
- **Rail**: Original single-axis StackShot controller, USB serial (FTDI bridge). The program drives the rail directly via the serial protocol — no auto-step front-panel programming, no shutter cable.
- **Lighting**: Two remote flashes (SB-800 + SB-5000) controlled wirelessly by an **SB-5000 mounted on the camera hot shoe as commander** (master output `--`). Remote groups in **A mode** (auto aperture).
  - **The SU-800 cannot be used as commander on Z7.** The Z7 dropped the in-body CLS forwarding path that older Nikon DSLRs used to relay camera-side FEC/EC commands to an SU-800. Camera FEC and EC writes succeed at the camera level but never reach the flash. Empirically confirmed.
  - The SB-5000 commander does forward FEC and EC correctly. Both produce monotonic exposure changes matching the requested EV.
  - The SB-800 also supports commander mode but was not validated in this project; if SB-5000 is unavailable, SB-800 commander is worth retesting before falling back to a different design.
- **Controlling computer**: Linux. Must be able to reach the user's MacBook over the local network for notifications.
- **Test target**: silicon wafer, fixed.

## 2. Key design decisions and tradeoffs

### 2.1 How AA mode actually works (and why it matters)

In Nikon AA (Auto Aperture; on SB-5000 commander menu, called `A` per group) flash mode, the **flash's own external sensor** measures reflected light during the main pulse and quenches the flash via IGBT when its threshold is reached. The camera communicates aperture and ISO to the flash so the threshold is set correctly; that's what differentiates AA from plain non-TTL A. AA does fire a preflash but that's used by the camera for white balance and FV Lock — **not** for flash power calculation. This is fundamentally different from i-TTL, which uses through-the-lens metering of the preflash to set main flash power.

Implications for this project:

- **AA does not compensate for bellows factor.** Extension tubes reduce light reaching the sensor through the lens, but the flash's external sensor sees the scene independently and doesn't know about the loss.
- **AA is unreliable on a specular silicon wafer target.** Small changes in target tilt or subject distance cause large swings in what the flash sensor sees.
- **Within the flash's working range, raising ISO does not change image brightness.** AA recalculates a lower threshold, the flash quenches sooner, exposure is held constant. ISO is only useful as a **range extender** — when the flash is at its power ceiling or in battery droop, raising ISO lowers the demand below what the flash can deliver, and the loop closes again.
- **The fine-tune lever in AA is FEC** (or EC, which does effectively the same thing with the SB-5000 commander mounted). FEC is conceptually cleaner because it only biases AA's flash target; EC also affects ambient if any leaks in.

### 2.2 Exposure adjustment lever — Path A confirmed

Empirical testing on this rig (Z7 V3.80 firmware, libgphoto2 2.5.33, python-gphoto2 2.6.0, **SB-5000 on hot shoe as commander**) confirmed:

- **FEC is writable and reaches the flash.** Property `flashexposurecompensation`, Type `RANGE`, range −3.0 to +1.0 EV, step ⅓ EV. Sweep from −3 to +1 produces JPEG p50 values that halve per stop (143, 100, 63, 37, 22), exactly matching expected linear-light scaling.
- **EC is also writable and produces equivalent flash effects.** Range −5 to +5 in ⅓-stop steps. Behaves identically to FEC at every tested value (within 0.02 EV).
- **ISO is writable.** Range 32–51200 in ⅓-stop steps.
- **Aperture is writable.** Standard ⅓-stop scale.
- **Shutter is writable in M mode.** `expprogram` is ro=1; the user must set the mode dial physically.
- **The SU-800 cannot be used as commander.** Camera FEC/EC writes succeed but do not reach the flash. SB-5000 (or SB-800 if validated) is required.

**Decision: Path A — FEC for closed-loop fine-tuning, ISO for range extension at small apertures, SB-5000 as commander.**

The architecture has two distinct exposure concerns:

- **Range extension** (deterministic): a per-aperture pre-emptive ISO bump and post-shot delay table for small apertures. Configured at session start.
- **Fine-tune** (closed-loop): test shot at each aperture, measure central crop, nudge FEC until within tolerance of baseline. Bellows factor and specular-target measurement quirks are absorbed here.

If FEC saturates (hits −3 or +1) and we're still off baseline, fall back to ISO bump as a last resort before logging a "flash power limit" warning.

- **ISO step size**: ⅓ stop. Floor: ISO 100. Warn at ISO 1600. Hard stop at ISO 3200.
- **FEC step size**: ⅓ stop.
- **FEC range**: −3.0 to +1.0 EV (Z7 body's FEC range; SB-5000 commander group FEC and remote-flash FEC stack on top, normally left at 0 by program but the user may dial in a session-wide offset on the SB-5000 if needed for the test target's reflectivity).

### 2.3 Camera control

- **Library**: `python-gphoto2` (2.6.0 confirmed working). **One persistent connection** opened at session start, closed at session end. Do NOT shell out to the `gphoto2` CLI per command — each invocation re-handshakes the camera and on Z7 the property tree drops to a 23-line minimal MTP view, which breaks everything.
- **Capture mode**: `capturetarget=Internal RAM` (a.k.a. SDRAM). Capture returns a transient file like `capt0000.jpg` that the program saves to its own filesystem path with the program's naming scheme.
- **Camera mode**: Manual (M). `expprogram` is read-only via gphoto2; the user sets the mode dial physically. Program checks `expprogram` at startup and aborts if not `M`.
- **Image quality**: `imagequality=JPEG Basic` (or Fine). RAW+JPEG complicates capture-and-download because the camera alternates which file is presented in the SDRAM slot — sometimes you get the JPEG, sometimes the NEF. JPEG-only avoids this.
- **Aperture sequence**: starts at the lens's widest aperture, then jumps to the next whole stop and continues in whole stops up to a session-wide max stop defined as a constant near the top of the program.

### 2.3.1 Camera operational constraints (from empirical testing)

These are hard requirements baked into the program flow:

- **Camera must be in M mode** on the mode dial (or U2 if U2 is configured for M). Program verifies and aborts otherwise.
- **Camera must come up in full Nikon-PTP mode**, not the cut-down Microsoft-MTP view. The reliable way to force this on Z7 V3.80:
  1. Power off the camera.
  2. **Remove the battery and wait at least 10 seconds.** A power-switch-off alone does not drop the USB state machine — only a battery pull does.
  3. Reinsert battery, power on. Confirm the rear screen shows a live image (proves the camera came up in shooting mode, not minimal-host-companion mode).
  4. Connect USB.
  5. Half-press the shutter to confirm the camera is awake.
- The program checks at startup that the config tree has more than ~100 properties (a healthy session has ~417). If fewer, abort with an error message instructing the user to perform the battery-pull recovery.
- **U2 / banks reset on power-off** for some settings (aperture, ISO at minimum). The program must re-apply session config after every reconnect: capturetarget, imagequality, base ISO, FEC=0, current aperture.
- **Lens change flow involves a power cycle** (because the user typically powers the camera off to swap lenses). The program's lens-change recovery prompt must walk the user through the battery-pull sequence above, then reconnect and re-apply session config.

### 2.4 StackShot control

- **Library**: Custom thin Python wrapper around the documented Cognisys protocol (PDF: `StackShotCommands_1_2.pdf`). FTDI bridge appears as a USB serial port on Linux at 38400 baud. `pyserial` for transport.
- **Architecture**: The program drives the rail directly, one move at a time. The StackShot's auto-step mode is unused. The shutter cable is unused — capture is triggered over USB by the program through gphoto2.
- **Protocol frame**: `0x55 CMD_MSB CMD_LSB Action Length Data... Checksum`. 2 ms inter-frame delay. Slave responds with `0xAA`. Multibyte values are little-endian. F32 floats are IEEE 754 LSB-first.
- **Commands the program uses**:
  - `CC_RAIL_MOVE` (0x1000) — relative move from the current target position. Direction (FWD/BACK), units (mm), backlash compensation (true), distance (F32). Backlash compensation must be true on direction reversal.
  - `CC_RAIL_STATUS` (0x1005) — returns bitfield: IDLE / MOVING / SHUTTER / USER_ABORTED. Polled every ~100 ms after a move command to wait for IDLE.
  - `CC_RAIL_POSITION_CURRENT` (0x1002) — read the current absolute position in mm (signed, relative to power-on origin or last `ZERO`).
  - `CC_RAIL_POSITION_ZERO` (0x1003) — set current position as the new origin. Used at the start of each lens config so positions in logs are relative to the chosen start.
  - `CC_RAIL_STOP` (0x1006) — emergency stop, used for droop alerts and Ctrl-C.
  - `CC_SOFTWARE_ID` / `CC_HARDWARE_ID` — read at startup for the rig session.json.
- **Move-and-wait pattern**: every motion call is `move(...)` followed by `wait_until_idle()` which polls status every 100 ms with a 30 s safety timeout. After each move, optionally read back position via `get_position()` and log if it differs from the requested target by more than a configurable tolerance.

### 2.5 Notification path

- **Tool**: `ntfy.sh`. Free, HTTP POST-based. User installs the ntfy app on macOS and subscribes to a private topic name. The Linux program POSTs to `https://ntfy.sh/<topic>` for both alerts (battery low) and lens-change prompts. The same topic also drives a sound on the Mac. Topic name in config.
- **Local sound on the controlling Linux machine**: Play a `.wav` via `paplay` when an alert fires, in addition to remote notification. (Useful if the user is at the rig, not the Mac.)

### 2.6 Shot-order architecture (apertures inside, rail steps outside)

The program inverts the natural "one stack per aperture" order. Instead of running K independent stacks of N steps each (one stack per aperture), it runs **a single sweep of N rail steps**, and at each rail position takes K shots — one per aperture.

Why this is better:

- **N rail moves total instead of K×N.** For typical N=30 steps and K=5 apertures, that's 30 moves vs 150. Less mechanical wear, less time spent moving.
- **All apertures sampled at exactly the same focus position.** Sharpness comparisons across apertures aren't biased by mechanical drift between separate sequences.
- **Battery droop affects all apertures uniformly.** Droop signal is cleaner because it builds gradually across the whole sweep, not concentrated in the last aperture's sequence.
- **FEC calibration done once per lens config.** At step 0, the program runs a test-shot loop per aperture to determine the FEC value that hits the baseline. Those FEC values are then reused for all 29 remaining steps. With sequential per-aperture sequences we'd recalibrate FEC at the start of each aperture's sequence — same data, more time.

Tradeoffs accepted:

- **Per-rail-step time goes up proportionally to K.** With 5 apertures at recycle times 3–6 s each = ~20–25 s per rail step plus ~1 s for the move. 30 steps ≈ 11–13 minutes per lens config. Comparable to the per-aperture approach total since flash recycle dominates either way.
- **Aperture changes are essentially free.** The Z7's electromagnetic aperture switches in well under 50 ms; any tiny mechanism settle is absorbed by the multi-second flash recycle that follows.

The session flow (§4) is restructured around this: outer loop is rail steps, inner loop is apertures. FEC calibration is a one-time per-lens-config step that happens at rail position 0.

## 3. File and directory layout

```
macro_lens_test/
├── README.md
├── pyproject.toml
├── config.py                        # constants: max F-stop, ISO floor/ceiling, paths, ntfy topic, exposure thresholds, jog increments
├── stackshot/
│   ├── __init__.py
│   ├── protocol.py                  # binary frame encode/decode per StackShotCommands_1_2.pdf
│   ├── controller.py                # high-level: connect, move, wait_idle, get_position, zero, stop, info
│   └── jog.py                       # interactive jog session (terminal keys → moves) for setting start/end
├── camera/
│   ├── __init__.py
│   └── nikon_z7.py                  # gphoto2 wrapper: persistent connection, set_aperture, set_iso, set_fec, capture_to, list_apertures, verify_healthy
├── exposure/
│   ├── __init__.py
│   ├── analyse.py                   # median of central crop, EV math
│   └── droop.py                     # battery-droop detector (rolling window per aperture)
├── notify/
│   ├── __init__.py
│   ├── ntfy.py                      # POST to ntfy.sh
│   └── sound.py                     # paplay wrapper
├── ui/
│   ├── __init__.py
│   └── prompts.py                   # readline-based input with persistent history
├── session/
│   ├── __init__.py
│   ├── runner.py                    # main orchestration loop
│   ├── calibrate.py                 # per-lens-config FEC calibration at step 0
│   └── log.py                       # CSV + JSON shot-log writer
├── data/
│   ├── readline_history             # auto-saved prompt history
│   ├── alert.wav                    # local alert sound
│   └── z7_diagnostic.py             # rig health-check script (bundled for ops use)
└── tests/
    ├── test_protocol.py             # offline tests of StackShot framing/checksum
    ├── test_droop.py                # synthetic exposure series → expected alerts
    └── test_exposure_analyse.py     # known-luminance test JPEGs
```

Output layout per session, under a configurable root directory:

```
<output_root>/
└── <session_timestamp>/
    ├── session.json                 # session metadata: camera, lens-config history, ntfy topic, rail info
    ├── shots.csv                    # one row per shot
    └── <lens_config_slug>/
        ├── calibration.json         # per-aperture FEC values determined at step 0
        └── f<aperture>/             # e.g. f4_0, f5_6, f8_0
            └── step_NNN.jpg         # NNN = zero-padded rail step index
```

`shots.csv` columns:
`timestamp,session_id,lens_config_slug,rail_step,rail_position_mm,aperture,iso,shutter_speed,fec,median_central_luminance,median_ev_relative_to_baseline,filename,phase,notes`

`phase` is one of `calibration` (test shots during the per-aperture FEC tuning at step 0), `sweep` (regular shot during the rail sweep), `aborted` (shot taken just before a droop abort).

`lens_config_slug` is generated from the free-text lens-config field by replacing each `:` and `/` with `_`. No other transformation.

## 4. Main session flow

```
1.  startup
    - load config.py
    - open StackShot serial port; read software/hardware ID; log them
    - open python-gphoto2 camera connection (persistent for session)
    - verify config tree has > 100 properties; if not, abort with a
      message instructing the user to do a battery-pull recovery
    - verify expprogram == M; abort otherwise
    - apply session config: capturetarget=Internal RAM,
      imagequality=JPEG Fine, iso=ISO_FLOOR, flashexposurecompensation=0,
      shutterspeed=1/200, autofocus=Off, imagesize=max
    - run one-time FEC sanity check (FEC=0 vs FEC=-2 capture, abort if
      median ratio < 50% — catches SU-800 mistakenly mounted)
    - create session directory under output_root

2.  for each lens configuration (loop until user enters empty line):
    a. prompt: "Lens config (free text, blank to quit):"
       - readline with persistent history
       - generate slug (substitute : and / with _)
       - mkdir for the slug under session dir
    b. JOG mode: hybrid start/end definition
       - prompt: "Jog rail to START position. Keys: a/d = back/forward
                  fine, A/D = back/forward coarse, s = set START, q = abort"
       - on each key press, issue a relative move at JOG_FINE_MM or
         JOG_COARSE_MM, wait_until_idle, redraw position
       - on 's', call CC_RAIL_POSITION_ZERO so START becomes 0.000 mm
       - prompt: "Jog rail to END position. Same keys, e = set END"
       - on 'e', call get_position; END_MM = current position
       - prompt: "Number of rail steps for this lens config:" (integer ≥ 2)
       - compute step_size_mm = END_MM / (N - 1)
       - move rail back to position 0 (issue back-move with backlash=true)
       - confirm: "About to do N steps of step_size mm each from
         position 0 to END. Press ENTER to begin."
    c. read available apertures from camera; build aperture sequence:
       widest → next whole stop → … → MAX_FSTOP
    d. CALIBRATION phase (rail still at position 0):
       - for each aperture in sequence:
           * set aperture
           * apply RANGE_EXTENSION[aperture]: bump ISO by configured stops
           * reset FEC to 0
           * test-shot loop (max TEST_SHOT_MAX_RETRIES iterations):
               - capture, download, measure median central luminance
               - if baseline is None (first lens config of session):
                   set baseline = this measurement, break
               - delta_ev = log2(median / baseline)
               - if |delta_ev| <= TEST_SHOT_TOLERANCE_EV: break
               - adjust FEC by -delta_ev, clamped to [FEC_MIN, FEC_MAX]
               - if FEC saturated and still under: raise ISO ⅓ stop
                 (clamped to ISO_CEIL)
               - log this test shot to CSV with phase=calibration
           * record final (FEC, ISO) tuple for this aperture in
             calibration.json under <slug>/
           * if final exposure still off at ISO_CEIL: send ntfy
             "flash power limit at f/X" warning, continue anyway
       - calibration complete; print summary table
    e. SWEEP phase: outer rail-step loop
       - for k = 0 .. N-1:
           * if k > 0:
               - move forward by step_size_mm (backlash=false; we're
                 only ever going forward in the sweep)
               - wait_until_idle
               - read position; warn if differs from expected by
                 more than POSITION_TOLERANCE_MM
           * for aperture in sequence:
               - set aperture, ISO, FEC to the calibrated values for
                 this aperture
               - capture, download to <slug>/f<aperture>/step_NNN.jpg
               - measure median central luminance
               - update per-aperture rolling droop window
               - if droop criteria met:
                   - send ntfy alert + local sound
                   - mark this and remaining shots in this step as
                     phase=aborted in CSV
                   - prompt user: "Battery droop detected. Swap
                     batteries on flashes, wait for steady ready light,
                     press ENTER to resume from rail step k."
                   - on resume: rail STAYS at current position (not
                     reset to 0). Re-do the current rail step k from
                     the start of the aperture inner loop.
                     Note: this means the user does not re-program the
                     StackShot — the program retains rail position
                     across the pause.
               - sleep RECYCLE_S (per-aperture configurable, longer
                 for small apertures)
       - after the loop: move rail back to position 0 for the next
         lens config (backlash compensation enabled)
    f. when all rail steps done:
       - send ntfy notification "Change lens config: was <slug>"
       - play local sound
       - prompt user with lens-change recovery:
         * "Power off camera. Change lens. Remove battery for at least
            10 seconds. Reinsert. Power on. Confirm rear screen shows
            live view. Press ENTER when ready."
       - close gphoto2 connection
       - wait for ENTER
       - reopen gphoto2 connection
       - verify config tree has > 100 properties; if not, prompt user
         to redo battery-pull (max 3 retries)
       - re-apply session config
       - loop back to (2a)

3.  on empty lens-config input → graceful shutdown:
    - move rail back to position 0 (for the user's convenience)
    - close gphoto2 connection
    - close StackShot port
    - write final session.json
```

## 5. Exposure analysis details

- **Central crop**: small offset crop on the wafer. Default 20% × 20% in image area, offset 15% below image centre. Configurable via constants and tunable at runtime.
- **Conversion**: convert JPEG to grayscale via Pillow (`ITU-R 601-2 luma`), take the median of the cropped pixels. Range 0–255.
- **EV calculation**: linearise the 8-bit median value with gamma 2.2, then `EV = log2(linear_median / baseline_linear_median)`. The gamma model is approximate at very low values but adequate for the ±1.5 EV range that matters for fine-tune and droop decisions. Below median ~30, treat the EV calculation as a clipping warning rather than a precise value.
- **Tolerance for fine-tune target**: ±⅓ EV (0.333 stops). Configurable.
- **Baseline**: the median of the **first calibration shot at the widest aperture of the very first lens config** of the session. Persisted in session.json so a resumed session can use the same baseline.

## 6. Battery-droop detector

- **Rolling window per aperture**: each aperture has its own rolling window of the last ROLLING_WINDOW (default 10) sweep-phase shots' EV-relative-to-baseline values. Calibration-phase shots and aborted shots do not enter the window.
- **Trip criteria** (whichever fires first, evaluated after each new sweep-phase shot at any aperture):
  - 3 of last 10 shots at a given aperture ≤ −0.7 EV, OR
  - 2 of last 10 shots at a given aperture ≤ −2.0 EV
- **Action on trip**:
  1. Mark this and any remaining shots in the current rail step as `phase=aborted` in the CSV.
  2. Send ntfy alert with the aperture, shot indices, and EV values.
  3. Play local sound.
  4. Pause: prompt the user to swap flash batteries, wait for steady ready light, then press ENTER.
     - The rail **stays at its current position**. With direct rail control, there is no "pre-programmed sequence" to restart — when we resume we simply re-run the aperture inner loop for the current rail step, then continue with rail step k+1 as normal.
     - The user does not need to re-program the StackShot or rejog. The program retains rail position across the pause.
  5. Reset the rolling window for all apertures so we don't immediately re-trip on stale low-EV samples from the droop event.

This is a meaningful improvement over the per-aperture-sequence design: a droop event mid-sweep loses at most one rail step's worth of shots, not an entire aperture's sequence.

## 7. Configuration constants (top of `config.py`)

```python
# Aperture
MAX_FSTOP = 16.0                # session-wide stop ceiling
WHOLE_STOP_SCALE = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11, 16, 22, 32]

# Range-extension table — applied BEFORE fine-tune at each aperture
# (recycle_s is also used as the per-shot recycle delay during sweep,
# and absorbs any small aperture-mechanism settle time)
RANGE_EXTENSION = {
    # f-stop : (iso_bump_stops, recycle_s)
    2.0:  (0,    3.0),
    2.8:  (0,    3.0),
    4.0:  (0,    3.0),
    5.6:  (0,    3.0),
    8.0:  (1,    4.0),
    11.0: (1,    5.0),
    16.0: (2,    6.0),
    22.0: (3,    7.0),
}

# ISO
ISO_FLOOR = 100
ISO_WARN = 1600
ISO_CEIL = 3200
ISO_STEP_EV = 1/3

# FEC (Z7 body range; with SB-5000 commander)
FEC_MIN_EV = -3.0
FEC_MAX_EV = +1.0
FEC_STEP_EV = 1/3

# Camera startup
MIN_CONFIG_TREE_SIZE = 100      # battery-pull recovery threshold
IMAGEQUALITY = "JPEG Fine"

# Exposure analysis (small offset crop on the wafer)
CROP_W_FRAC = 0.20
CROP_H_FRAC = 0.20
CROP_X_FRAC = 0.00              # offset of crop centre from image centre, x
CROP_Y_FRAC = 0.15              # offset y; positive = below centre (wafer sits low)
TEST_SHOT_TOLERANCE_EV = 1/3
TEST_SHOT_MAX_RETRIES = 6
ROLLING_WINDOW = 10
DROOP_CRIT_A = (3, -0.7)
DROOP_CRIT_B = (2, -2.0)

# StackShot motion
STACKSHOT_POLL_INTERVAL_S = 0.10        # 10 Hz status polling during moves
STACKSHOT_MOVE_TIMEOUT_S = 30.0         # safety: abort if a move never reports IDLE
JOG_FINE_MM = 0.05                      # a/d keys
JOG_COARSE_MM = 1.0                     # A/D keys
POSITION_TOLERANCE_MM = 0.02            # warn if get_position differs from expected by more
BACKLASH_COMPENSATION_DEFAULT = True

# Paths
OUTPUT_ROOT = "/path/to/output"
READLINE_HISTORY = "data/readline_history"
ALERT_WAV = "data/alert.wav"

# Notification
NTFY_TOPIC = "macro-lens-test-<random-suffix>"
NTFY_BASE_URL = "https://ntfy.sh"
```

## 8. Camera control specifics (python-gphoto2)

Single persistent `gp.Camera()` connection for the session. Reopen on lens change after the user does the battery-pull recovery.

Properties used:

- `capturetarget` — set to `Internal RAM` at session start and after every reconnect.
- `imagequality` — `JPEG Basic` (or Fine). NEF-only or NEF+JPEG complicate the SDRAM transfer because the camera alternates which file is presented; sticking to JPEG-only avoids this entirely.
- `expprogram` — verified == `M` at startup, read-only.
- `iso` — RADIO widget, set by string value (e.g. `"100"`).
- `f-number` — RADIO widget, set by string value (e.g. `"f/4"`).
- `flashexposurecompensation` — RANGE widget, set by float in EV (e.g. `-1.333`).
- `shutterspeed` — set once at session start (likely `1/200s` for max sync), then untouched.

Capture pattern:
```python
fp = camera.capture(gp.GP_CAPTURE_IMAGE)
cf = camera.file_get(fp.folder, fp.name, gp.GP_FILE_TYPE_NORMAL)
cf.save(dest_path)
```

Setting a config value is always a fresh `get_config` → mutate widget → `set_config`:
```python
cfg = camera.get_config()
w = find_widget(cfg, "flashexposurecompensation")
w.set_value(-1.333)
camera.set_config(cfg)
```

The startup config-tree health check:
```python
cfg = camera.get_config()
if count_widgets(cfg) < MIN_CONFIG_TREE_SIZE:
    raise CameraStateError(
        "Camera came up in minimal-MTP mode. Power off, remove "
        "battery for 10 seconds, reinsert, power on, wait for live "
        "view, then connect USB."
    )
```

Save `data/z7_config_dump.txt` from a healthy session for reference (the captured 417-line dump from the working diagnostic run).

## 9. StackShot control specifics

The protocol is fully documented in `StackShotCommands_1_2.pdf`. The original StackShot enumerates as an FTDI USB serial device at 38400 baud.

**Frame format**: `0x55 CMD_MSB CMD_LSB Action Length Data... Checksum`.
- `Action`: `0x00` for read, `0x01` for write.
- `Length`: number of bytes in the data field (may be 0).
- Multibyte values in the data field are **little-endian** (LSB first).
- F32 floats are IEEE 754, also LSB first.
- Slave responses begin with `0xAA` and use the same frame layout, with Action holding the response code (`COMM_ACTION_RSP_OK` = 0x02 on success).
- 2 ms minimum inter-frame delay required for resync.

**Commands the program uses** (from §1.2 of the protocol PDF):

| Command | ID | Purpose |
|--|--|--|
| `CC_RAIL_MOVE` | `0x1000` | Relative move from current target. Data: dir (1B), units (1B), backlash (1B), distance F32 (4B). |
| `CC_RAIL_POSITION_CURRENT` | `0x1002` | Read current absolute position, signed. |
| `CC_RAIL_POSITION_ZERO` | `0x1003` | Set current as origin. Used after jog-to-start. |
| `CC_RAIL_STATUS` | `0x1005` | Read 32-bit status bitfield. Bits: 0=MOVING, 1=SHUTTER, 2=USER_ABORTED. IDLE = 0. |
| `CC_RAIL_STOP` | `0x1006` | Emergency stop. |
| `CC_SOFTWARE_ID`, `CC_HARDWARE_ID` | `0x1102`, `0x1103` | Identification, logged once at session start. |

**Move-and-wait pattern**:
```python
def move_and_wait(self, direction, distance_mm, backlash=True):
    self._write(CC_RAIL_MOVE, dir=direction, units=METRIC,
                backlash=backlash, distance=distance_mm)
    deadline = time.monotonic() + STACKSHOT_MOVE_TIMEOUT_S
    while time.monotonic() < deadline:
        st = self._read(CC_RAIL_STATUS)
        if st & RAIL_STATUS_USER_ABORTED:
            raise StackShotUserAborted()
        if (st & RAIL_STATUS_MOVING) == 0:
            return
        time.sleep(STACKSHOT_POLL_INTERVAL_S)
    self.stop()
    raise StackShotTimeout(f"move did not complete within {STACKSHOT_MOVE_TIMEOUT_S}s")
```

**Jog session**: a small terminal UI using `termios` to put stdin in cbreak mode. Each keypress maps to a relative move call. Position is read after each move and printed on the same line (carriage return, no newline).

| Key | Action |
|--|--|
| `a` / `d` | Back / Forward fine (`JOG_FINE_MM`) |
| `A` / `D` | Back / Forward coarse (`JOG_COARSE_MM`) |
| `s` | Set START: `CC_RAIL_POSITION_ZERO`, exit jog session, return |
| `e` | Set END: read position, exit jog session, return position |
| `q` | Abort: raise to caller |

Backlash compensation is enabled on every move during jog (the user often reverses direction); during the sweep we move strictly forward so it can be disabled for sweep moves.

## 10. Notification implementation (ntfy)

- Linux side: `requests.post(f"{NTFY_BASE_URL}/{NTFY_TOPIC}", data=msg, headers={"Title": title, "Priority": prio, "Tags": tags})`.
- Mac side (one-time setup, document in README): install ntfy from https://ntfy.sh, subscribe to the topic, enable notifications + sound.
- Distinct alert types:
  - `lens_change_required` — title "Change lens config", priority `default`.
  - `battery_droop` — title "Battery droop detected", priority `high`, tag `warning`.
  - `flash_power_limit` — title "Flash power limit reached", priority `default`.

## 11. Implementation order for Claude Code

### Step 0 — Verify rig is in working state (5 min)

Path A is locked in (FEC writable, capture works, full ~417-line config tree, **SB-5000 commander forwards FEC to remotes**). Before each fresh implementation session, verify the camera is in a healthy state by running `data/z7_diagnostic.py`.

Camera prep checklist:
- Power off, **remove battery, wait 10 seconds, reinsert** (this is the only reliable way to drop the USB state machine on Z7 V3.80).
- Mode dial = M (or U2 with M bank).
- **SB-5000 mounted on hot shoe as commander** (not SU-800), master output `--`, remote groups in A mode. Standby timeout on SB-5000 set to 300 s.
- Remote flashes (SB-800 + SB-5000) switched on, set to remote-slave, matching channel/group.
- Power on the camera, wait for live view to appear on the rear screen.
- THEN connect USB.
- Half-press shutter to wake.
- `python3 z7_diagnostic.py 2>&1 | tee /tmp/diag.log; wc -l /tmp/z7_full_config.txt`

Healthy state: ~400 lines in the config dump and both test JPEGs captured. Bad state: ~23 lines in the config dump — redo the battery-pull.

### Steps 1–10 — Implementation (each ends in something runnable)

1. **Skeleton + config.py + README.md**. Include the operating-procedure section in the README (battery-pull, M mode, USB-after-power-on order, SB-5000 as commander, post-reconnect reconfig).
2. **StackShot protocol module** (`stackshot/protocol.py`) — frame encode/decode, checksum, byte-ordering helpers. Unit tests with hand-computed example frames from §9.
3. **StackShot controller** (`stackshot/controller.py`) — open serial, software/hardware ID, `move_and_wait`, `get_position`, `zero_position`, `stop`, `wait_until_idle` with timeout. CLI: `python -m stackshot.controller --info` prints IDs and current position. **User tests on rig.**
4. **StackShot jog session** (`stackshot/jog.py`) — terminal cbreak loop with the key map from §9. CLI: `python -m stackshot.jog` runs an interactive jog session and prints the final position. **User tests on rig.**
5. **Camera wrapper** (`camera/nikon_z7.py`) using python-gphoto2 with persistent connection. Methods: `connect()`, `disconnect()`, `verify_healthy()` (counts widgets, checks `expprogram == M`), `apply_session_config()`, `set_aperture()`, `set_iso()`, `set_fec()`, `capture_to(dest_path)` with event drain and retry, `list_apertures()`, `fec_sanity_check()`. CLI: `python -m camera.nikon_z7 --capture /tmp/test.jpg`. **User tests.**
6. **Exposure analysis** (`exposure/analyse.py`) — small offset crop, median, gamma=2.2 EV math. Unit tests with synthetic JPEGs.
7. **Droop detector** (`exposure/droop.py`) — per-aperture rolling-window class with the two trip criteria. Unit tests with synthetic series demonstrating both criteria firing.
8. **Notifications** (`notify/ntfy.py`, `notify/sound.py`) — ntfy POST + paplay. CLI test. **User tests delivery to Mac.**
9. **Prompts with readline history** (`ui/prompts.py`) — free-text lens config field, slug = substitute `:` and `/` with `_`.
10. **Session runner** (`session/runner.py`, `session/calibrate.py`, `session/log.py`) — orchestrate per §4: jog start/end, calibrate FEC per aperture at step 0, sweep with rail-step outer / aperture inner loop, droop detection across all apertures, lens-change recovery. Add `--dry-run` mode that replays a directory of pre-captured JPEGs without touching hardware.
11. **Polish** — graceful Ctrl-C (issue StackShot Stop), atomic CSV writes, final session.json with full session metadata.

## 12. Open items the program should validate at startup

- python-gphoto2 sees exactly one Z7. If zero or multiple, error.
- Camera config tree has > MIN_CONFIG_TREE_SIZE (~100) properties. If fewer, the camera came up in minimal-MTP mode — instruct user to do a battery-pull recovery.
- StackShot serial port opens and `GetSoftwareID` returns a sensible value.
- Camera is in M mode (`expprogram` should report `M`). If not, error.
- Camera shutter speed is fixed and reported. Print it for the user to verify.
- A flash is detected on the hot shoe (`externalflash` widget value, or via the fact that `flashexposurecompensation` is non-readonly). If not, warn the user that the SB-5000 commander is required for FEC to reach the flash.
- ntfy topic is not the default placeholder. If it is, error.
- Output root directory exists and is writable.

The program should also do a **one-time FEC sanity check** on the first session of any rig session: take two test shots at FEC=0 and FEC=−2, compare central-crop p50 ratio, and abort if they differ by less than 50% (i.e., if FEC isn't reaching the flash). This catches the case where the user accidentally puts the SU-800 back on the hot shoe.

## 13. Things explicitly NOT in v1

- Multi-camera support.
- Wi-Fi connection to StackShot 3X (different model).
- Live-view streaming.
- Automatic SU-800 / flash power adjustment (not possible — physical buttons only).
- Sharpness analysis itself — this script only collects the data; sharpness measurement is a downstream tool the user already has.
- A GUI. Terminal only.
