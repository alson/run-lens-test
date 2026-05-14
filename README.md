# Macro Lens Sharpness Test Automation

Automated lens sharpness testing on a silicon wafer target, using a Nikon Z7,
StackShot focus rail, and SB-5000 / SB-800 wireless flashes. Drives the rail
and camera directly from a Linux laptop; sends notifications to a paired
MacBook over the network.

## What it does

For each lens configuration the user defines:

1. The user jogs the rail to a chosen start and end position.
2. At rail position 0, the program runs a per-aperture FEC calibration: it
   takes test shots, measures wafer brightness in a small central crop, and
   nudges flash exposure compensation until each aperture matches the
   session-wide reference brightness within ⅓ EV.
3. The program then sweeps the rail through N steps. At each step, it shoots
   the wafer at every aperture from the lens's widest to a configurable max,
   downloading each JPEG to disk and measuring exposure.
4. Throughout the sweep, a per-aperture rolling-window droop detector watches
   for the flashes losing power as their batteries deplete. If detected, the
   program sends a notification to the MacBook, plays a local alert sound,
   and pauses for the user to swap batteries — without losing rail position
   or progress.
5. When all rail steps are done for one lens config, the program prompts the
   user to change lenses (and walks them through a battery-pull camera
   recovery, since the Z7 firmware needs that to come up cleanly).

## Hardware setup

| Component | Connection |
|--|--|
| Nikon Z7 | USB to laptop. Used for capture and all camera control. **Requires a USB 3.0 (5 Gbps) or faster cable.** USB 2.0 cables negotiate the link at 480M, which can't reliably sustain the 12 MB/frame bulk transfer — capture appears to succeed (shutter fires, flashes go off) but the post-capture file retrieval errors with `PTP General Error / I/O problem` and wedges the gphoto2 USB session. Verified working: 5 Gbps USB-A-to-C cable; 40 Gbps USB-C-to-C cable. Check post-plug with `lsusb -t` — the Z7's interface should show `5000M` (or higher), not `480M`. |
| StackShot rail | USB serial to laptop. Used for motion control. |
| SB-5000 | Hot shoe on Z7, **commander mode**, master output `--`. |
| SB-800 + SB-5000 (remotes) | Optical slaves, A mode, channel/group matching commander. |
| Laptop | Linux, with `gphoto2` 2.5.21+ and python-gphoto2 installed. |
| MacBook | Same network as laptop, with the [ntfy](https://ntfy.sh) app installed and subscribed to the configured topic. |

The shutter cable between the StackShot and the camera is **not used** in this
project. The program triggers the camera over USB.

## Operating procedure

Read this section before each test session.

### Camera prep (every session start)

1. **Power off the camera.**
2. **Remove the battery for at least 10 seconds, then reinsert.**
   This is the only reliable way to drop the Z7's USB state machine. A
   power-switch-off alone does not work, and a different USB device class
   may persist from a previous session.
3. Set the **mode dial to M** (or U2 if U2 is configured for M). Do not use
   anything else — the program checks `expprogram == M` and aborts otherwise.
4. Set the **focus mode to Manual** (Fn1 button + sub-command-dial on the Z7).
   AF-S / AF-C / AF-F can refocus between captures and ruin the focus-bracketed
   sweep. The focus-mode widget is read-only over PTP on Z lenses (focus-by-
   wire), so the runner can only detect the wrong state and abort — it can't
   set MF for you. Setting `autofocus = Off` in software only suppresses
   AF-on-shutter; it doesn't change the lens focus mode.
5. Confirm custom function **e3 = "Entire frame"** (not the default "Background
   only"). The calibrator uses camera EC as a fallback when FEC saturates,
   relying on EC stacking with FEC for flash output. With e3 = "Background
   only" the EC fallback wouldn't reach the flash, leaving you 3 EV less
   headroom in the closed loop.
6. Mount the **SB-5000 on the hot shoe in commander mode**, master output `--`,
   group A in `A` (auto aperture) mode, group FEC = 0.
7. Switch on the remote flashes (SB-800 + SB-5000), check ready light.
8. **Power on the camera.** Wait for the rear screen to show live view.
9. Half-press the shutter to confirm the camera is awake.
10. **Now connect the USB cable from the camera to the laptop.** Plugging in
    before power-on tends to make the camera enumerate as a minimal MTP
    device and the program will refuse to start.

If at any point the program reports "config tree too small — battery-pull
recovery needed", that means the camera dropped into the minimal-MTP state
and you need to repeat steps 1–10.

### Rig prep

1. Make sure the StackShot is powered.
2. Plug in the StackShot's USB cable.
3. Position the wafer target. Aim the lens roughly so the wafer falls in
   the lower-centre of the frame (the default crop region).

### Health check

Before each session run the diagnostic script:

```bash
python3 data/z7_diagnostic.py 2>&1 | tee /tmp/diag.log
wc -l /tmp/z7_full_config.txt
```

The config dump should be ~400 lines and both test JPEGs should be captured.
A 23-line dump means the camera came up in minimal-MTP mode — redo the
battery-pull procedure.

### Running the program

```bash
python -m session.runner
```

Optional CLI flags (all skip the matching interactive prompt; useful
for tail-extension and re-shoot scenarios):

| Flag | Effect |
| --- | --- |
| `--no-probe` | Disables Phase-A probe for this run only (equivalent to `PROBE_ENABLED = False` in config). Use for tail-extension sessions whose user-jogged range deliberately lies outside the widest-aperture peak. |
| `--lens TEXT` | Skip the lens-config prompt and use the supplied string. Match the verbatim text used in prior sessions so the slug stays consistent for offline analysis. |
| `--magnification M` | Skip the magnification prompt (e.g. `1.0` for 1:1, `0.5` for 1:2). |
| `--rail-start MM` | Signed mm offset from the rail's current hardware position to the session START. The rail moves there, and that position becomes session-zero. Usually `0` (= "use the rail's current position as START"). Required with `--rail-end`. |
| `--rail-end MM` | Signed mm offset from session START to session END. Negative for backward sweeps (the common case). Required with `--rail-start`. |
| `--manual-aperture` | Enable manual-aperture mode for non-CPU lenses (DESIGN_NOTES §6.31). |

Typical tail-extension recipe:

```bash
# 1. Manually position the rail at the intended START with the
#    rig's panel buttons or a previous interactive jog.
# 2. Launch with the full setup baked in:
python -m session.runner \
  --no-probe \
  --lens "Viltrox 85mm f/2 EVO + 45mm extension" \
  --magnification 0.66 \
  --rail-start 0 \
  --rail-end -2.0
```

Skips all interactive prompts, runs one lens config, exits.

The program will:

1. Open the StackShot and camera.
2. Offer a freeform jog before the FEC sanity check (`[ENTER]`
   continue, `j` jog, `q` abort). Useful if the rail is at a stale
   position from a previous run, the wafer is out of focus, or the
   StackShot was left in USB-control mode after a crash and you
   need to release the panel buttons.
3. Validate camera state (M mode, focus mode = Manual, full config
   tree, FEC sanity test).
4. Prompt for the first lens configuration as a free-text line.
5. Prompt for the **magnification** in ratio (`1:1`, `1.5:1`, `1:1.5`,
   `2:1`) or float form (`0.5`, `1.0`). The runner uses this plus the
   lens's widest aperture to compute the rail step size from a thin-lens
   DOF model — you no longer pick a step count manually.

   **Magnification math caveat**: total magnification with an
   extension tube is `m_native + L_tube / F_eff`, where `m_native`
   is the lens's max magnification at minimum focus and `F_eff` is
   the focal length at that focus (which is *not* the barrel
   nominal for internal-focus macros). The Z MC 105 mm f/2.8 VR S
   drops to ~49 mm F_eff at 1:1 (per photomacrography.net optical-
   bench measurements); the Z 24-120 stays close to its zoom-
   position nominal (3 % breathing); the Viltrox 85 mm f/2 EVO
   reaches only 0.13× max and breathes little. The reshoot-plan
   menu carries pre-computed estimates corrected per-lens. Over-
   estimating M makes the planner pick a smaller rail step, which
   over-samples the bell — the safe error direction — but if you
   want a tighter rail range, enter the corrected value.
6. Drop you into a jog session: use `a`/`d` to step the rail back/forward
   in fine increments, `A`/`D` for coarse, `s` to mark the start position,
   `e` to mark the end, `q` to abort. The current position is shown live.
   The runner prints the planned rail step + position count once you've
   set END; if the jogged range yields fewer than `MIN_RAIL_POSITIONS`
   (default 10) positions, you'll be prompted to re-jog wider.
7. Run FEC calibration at step 0 (one test shot per aperture, possibly more).
8. Run the rail sweep.
9. Notify on completion and prompt for the next lens config.

### Lens change procedure

When the program prompts for a lens change:

1. **Power off the camera** with the power switch.
2. Change lenses.
3. **Remove the battery for at least 10 seconds and reinsert.** This is
   non-negotiable — the camera's USB state must be fully reset. Bank U2
   resets some settings on power-off, and the program re-applies them on
   reconnect.
4. **Power on the camera, wait for live view.**
5. Press ENTER in the program. It reconnects, verifies state, and resumes.

### Battery droop pause

If the program pauses with a "Battery droop detected" alert:

1. Swap the AA batteries in the affected flash(es).
2. Wait for steady ready lights on the flash.
3. Press ENTER in the program.

The runner then **rewinds the rail** to the earliest under-exposed step
(across all apertures, taken from the rolling-window samples that tripped
the criterion) and **reshoots that step through the trigger step** with
fresh power. The under-exposed JPEGs on disk are overwritten in place;
`shots.csv` carries new rows with `phase=reshoot` for the affected steps,
and the original `phase=sweep` rows for those steps are dropped to keep
the metadata clean.

If droop fires *during* the reshoot — i.e., a flash you didn't catch on
the first swap is also weak — the runner pauses again, prompts again,
and recurses into another reshoot covering the new earliest under-exposed
step through the trigger step. This nests as many times as needed.

You do **not** need to re-jog the rail, the StackShot is being driven by
the program throughout.

### Range-adequacy prompts

After the main sweep, the runner runs a sharpness analysis to check
whether the captured rail span actually covered the focus peak.

* **Truncated bell on one side** → the runner shoots a probe in that
  direction; if it brackets the peak it then fills out the rest of an
  extension sweep automatically and continues. No prompt.
* **Truncated bell that the probe didn't bracket** → the runner prompts:
  `[ENTER] extend by ~N mm on the failing side, [s] skip and continue
  to lens change, [q] abort`. ENTER to keep extending until the peak
  brackets; `s` to accept the partial data and move on; `q` to abort.
* **All apertures' curves are flat (`OFF_TARGET`)** → the rail span
  almost certainly missed the focus zone, or the wafer position lacks
  micro-detail. The runner sends a high-priority ntfy alert + chirp
  and prompts: `[ENTER] re-jog START / END and re-sweep,
  [s] skip and continue with this data, [q] abort`. ENTER reruns the
  jog UI and redoes the sweep at a new range with the existing
  calibration preserved (no re-calibration burn).

### Quitting

Type an empty line, `q`, `quit`, or anything shorter than 5 characters
at the lens-config prompt to quit cleanly. The program moves the rail
back to position 0, closes the camera and rail, and writes the final
session log.

Ctrl-C also works — the program issues a StackShot stop, closes
connections, and exits cleanly without dumping a traceback. Any
in-progress shot may be lost.

### Manual-aperture mode (non-CPU lenses)

For AI-S, pre-AI, and other lenses where the diaphragm is set by a
physical aperture ring (e.g., Micro-Nikkor 55mm f/2.8 AIS,
Micro-Nikkor Auto 55mm f/3.5 + K-tubes), launch the runner with:

```bash
uv run macro-lens-test --manual-aperture
```

Differences from the default (CPU-lens) flow:

* **Flash mode = Manual** on the SB-5000 commander remote groups
  (M, not AA / TTL). The body can't communicate aperture to the
  flash for non-CPU lenses on the Z7, so AA quenching is broken.
  Group C is the left flash; group A is the right flash and
  should be one third stop lower than C (subject-distance offset).
  The rig calculates the per-aperture power target and prompts
  you to dial it on the commander.
* **You turn the aperture ring** between captures. The runner
  prompts: `"Turn lens ring to f/X, then press [ENTER]. [q] to
  abort:"` — dial that ring stop and press ENTER. The runner
  walks ring whole stops from the lens's marked widest up to
  whichever stop yields effective f-number ≤ `MAX_FSTOP`.
* **You dial flash power** when the runner asks. The cascade
  uses a deterministic per-aperture (power, ISO) target — it
  knows the previous aperture's result and computes the next
  power level that will land at baseline. The prompt format
  matches the SB-5000 commander display, e.g.
  ```
  group A (right) : 1/4 -0.3
  group C (left)  : 1/4
  [ENTER] to confirm, [q] to abort:
  ```
* **Lens-config prompts**: the runner parses `focal_length_mm`
  and `f_ring` (marked widest) from the slug regex, then asks
  for `magnification_indicated` (the value the lens barrel
  shows, e.g. `1:2` for 55 AIS at MFD). With an extension tube,
  this is the lens-only contribution; the rig does the
  bellows-factor math.
* **ISO floor = 100** (not 50, which is auto-mode's default —
  manual mode prefers staying at the native floor; the cascade
  uses flash power to absorb most of the range and only bumps
  ISO when it must).
* **Tolerance = 0.2 EV** (tighter than auto-mode's 0.5) — the
  predictive solver can miss by ~0.2-0.3 EV when the simple
  bellows model or the flash IGBT timing diverge from linear
  assumptions.

See `macro_lens_test_plan.md` §14 for the full design;
`DESIGN_NOTES.md` §6.31 for the decision rationale.

### Tail-extension workflow (multi-session merging)

When a lens-config's data lands with some crops' peaks past either end
of the captured range (e.g. soft corners on a non-macro lens + heavy
extension tubes), the practical remedy is a *short follow-up session*
that covers just the missing tail with small overlap into the prior
session — not a full re-do.

Recipe:

1. Skip out of the current session (`s` at the range-inadequate prompt
   or `q` at the next lens-config prompt).
2. Restart `python -m session.runner` so any config edits / code
   changes load. **Set `PROBE_ENABLED = False` in `config.py`** for
   tail-extension scenarios — phase-A's bracket assumes the widest
   aperture peaks somewhere inside the user-jogged range, which is
   false when you're deliberately jogging *outside* that range.
   Re-enable `PROBE_ENABLED = True` afterward for normal sweeps.
3. New lens-config name (≥5 chars). Reuse the same name across the
   prior session and the follow-up if you want a single dataset, or
   suffix with `extension`, `tail`, etc. for clearer audit.
4. At the START prompt, use the numeric jog (`:` then a typed mm
   target) to place the rail with overlap into the prior session's
   coverage. *Sign convention*: in a backward-jog session (START is
   less-negative, END is more-negative), more-negative target positions
   are deeper into the captured range. To extend **deeper than the
   previous END**, set the new START a few hundred µm *before* (i.e.
   *less* negative than) the previous END so phase-D's first frames
   overlap. To extend **shallower than the previous START** (i.e. closer
   to the camera than the previous START), set the new START a few
   hundred µm *past* the previous START in the away-from-camera
   direction.
5. Press `s` to zero at the new START. The new session's coordinate
   system has its own origin; you have to track the offset to the
   prior session manually.
6. At the END prompt, use `:` again to type the relative target. **The
   END sign must match the jog direction**: typing `-1.0` with `s` at
   step 4 zeroing-then-jogging-backward will end the new range 1 mm
   deeper than START; typing `+1.0` will end 1 mm shallower. Mis-signed
   END is the most common operator mistake here — re-verify the sign
   before pressing `e`.
7. Same magnification as the prior session.
8. Calibration re-runs (~30–60 s — unavoidable; calibration data is
   per-lens-config). Phase A, if enabled, will probably go inconclusive
   because the widest aperture's bell is already past peak in the new
   range; that's fine — the runner falls back to uniform sweep.

**Range-adequacy prompts can flag single-session truncations that
your *prior* session already covers**. If the prompt says
`peak_side=both` but your other session covers one of those sides
densely, accepting the proposed extension means duplicating data on
the already-covered side. Either accept once and skip on the second
round, or skip immediately and live with the single-session "truncation"
that you know is captured elsewhere.

Merge offline: combine the JPEGs / shots.csv across sessions per
`(lens_config_slug, step_index, aperture, replica)`. The per-session
rail-zero is tracked in `session.json` only implicitly (the rail was
wherever the user left it before pressing `s`); record the offsets
in your own notes or a sidecar file.

## Output

Per-session directory under `OUTPUT_ROOT`, named with a timestamp:

```
~/macro-lens-test-output/
└── <session_id>/                          e.g. 2026-05-05_172105
    ├── session.json                       session-level metadata
    ├── shots.csv                          one row per capture
    ├── fec_sanity_baseline.jpg            FEC sanity test shots
    ├── fec_sanity_low.jpg
    └── <lens_config_slug>/                free-text name, slugified
        ├── calibration.json               per-aperture (FEC, ISO) tuples
        ├── range_check.json               peak-side, extension applied, etc.
        ├── calibration/
        │   └── cal_f_<X>_<NN>.jpg         calibration test shots
        └── <aperture>/                    e.g. f_4_5, f_5_6, f_8, f_11, f_16
            ├── step_<NNNN>.jpg            replica 0 (legacy filename)
            ├── step_<NNNN>_b.jpg          replica 1 (when SHOTS_PER_STEP_APERTURE > 1)
            ├── step_<NNNN>_c.jpg          replica 2
            └── cropped/                   17 crops per source JPEG
                ├── center/step_<NNNN>.jpg
                ├── top-left/step_<NNNN>.jpg
                └── …
```

* **Filename → step_index**: `STEP_NUMBER_OFFSET = 100`, so
  `step_0100.jpg` = step_index 0, `step_0099.jpg` = step_index −1
  (start-side extension, before step 0), `step_0107.jpg` =
  step_index 7 (end-side extension, past the original step count).
  Phase (sweep / probe / extension / reshoot) lives in
  `shots.csv`'s `phase` column, not the filename — the same
  `step_NNNN.jpg` is reused across phases (file gets overwritten in
  place on a reshoot).
* **Replica suffix**: replica 0 keeps the legacy unsuffixed name so
  existing offline-analysis tooling sees no change; replicas 1, 2, …
  use `_b`, `_c`, …
* **Per-aperture subsampling**: smaller apertures may have *fewer*
  `step_NNNN.jpg` files than wider ones. The planner's
  `subsample_factors` (in `session.json` under each lens config)
  records the integer skip stride per aperture — e.g. `f/4.5: 1`
  means "every step", `f/16: 3` means "every third step". The
  planner picks the factors so the FWHM density at every aperture
  is comparable, even though smaller apertures shoot less. The
  runner now honours these factors at capture time, saving
  20-35 % of the wall-clock per lens config at typical
  magnifications.
* **`session.json`** records `lens_configs`, `baseline_median_luminance`,
  StackShot software / hardware IDs, ntfy topic, `aborted` flag,
  and `ended_at`.
* **`shots.csv`** rows: rail step, aperture, ISO, FEC, measured
  median luminance, EV-relative-to-baseline, filename, phase. Stale
  rows are *dropped* from the file when a reshoot or off-target
  re-jog overwrites their data — only the canonical, latest row per
  (step, aperture, replica) is kept.

## Configuration

All tunable constants are at the top of `config.py`. Notable ones:

- `MAX_FSTOP` — the largest f-number to test (session-wide).
- `WHOLE_STOP_SCALE` — the canonical f-stop ladder; the per-lens
  aperture sequence is the lens's widest aperture plus each whole
  stop on this ladder up to `MAX_FSTOP`.
- `RANGE_EXTENSION` — per-aperture (base ISO, flash recycle seconds).
  The recycle wait is *subtracted* by elapsed time since the previous
  capture, so set_config + capture during the recycle window count
  against the budget.
- `BASELINE_REFERENCE_IMAGE` — `Path` to a reference JPEG; the runner
  measures its central-crop median once and uses that value as the
  absolute calibration target across every session. Set to `None` to
  fall back to "first widest-aperture shot of the first lens config"
  per-session baseline.
- `SHOTS_PER_STEP_APERTURE` — replicas captured at each (rail step,
  aperture) pair (default 3). The advisor / range-checker collapse
  replicas via per-crop median before peak detection. Set to 1 to
  revert to single-shot.
- `STEP_NUMBER_OFFSET` / `STEP_NUMBER_WIDTH` — filename layout for
  start-side extensions where step indices go negative (default
  offset 100, width 4 → `step_0099.jpg` for step −1,
  `step_0100.jpg` for step 0, etc.).
- `SHARPNESS_MAX_WORKERS` — concurrent crop workers (default 8;
  matches a typical laptop's logical-core count). Production uses
  a `ProcessPoolExecutor` with these workers and direct
  libjpeg-turbo via `PyTurboJPEG`; per-shot crop time on the
  bench rig is ~89 ms for 17 crops at 45 MP.
- `SHARPNESS_CROPPED_DIRNAME` — subdir name under each
  `<lens>/<aperture>/` where the 17 crops are persisted.
- `SHARPNESS_CROP_DYNAMIC_RANGE_MIN` — high-signal floor for the
  per-crop dynamic range `(max - min) / max` (default 0.25). At or
  above this threshold, a crop with a peak in the middle of the
  rail series is classified as `OK`. Below it (but above
  `SHARPNESS_CROP_OFF_TARGET_RANGE`) the crop is `LOW_SIGNAL` —
  a real-but-weak bell, common on soft lenses (e.g. fast primes
  with external extension tubes where corner acutance is poor).
  The advisor treats `LOW_SIGNAL` like `OK` for the per-aperture
  verdict but surfaces the count in the summary so the operator
  can spot soft-rig regimes.
- `SHARPNESS_CROP_OFF_TARGET_RANGE` — off-wafer floor for the
  per-crop dynamic range (default 0.05). Strictly below this
  value, the crop's window almost certainly fell off the wafer
  onto featureless background; the advisor short-circuits to
  `OFF_TARGET` without running peak detection. Empirical
  motivation: the Viltrox 85mm f/2 EVO + 61mm extension produced
  legitimate corner crops with dynamic range 0.04–0.22 that the
  pre-2026-05 single-threshold logic falsely rejected as off-wafer;
  splitting the threshold lets soft-but-real bells be reported
  as `LOW_SIGNAL` instead.
- `SHARPNESS_PARTIAL_FRAME_FACTOR` — absolute-size peer-compare
  threshold for the `PARTIAL_FRAME` detector (default 0.80). A
  crop whose peak file size is below this factor times the
  peer-group median (4 full-frame corners vs each other, 4
  full-frame edges vs each other) is flagged `PARTIAL_FRAME`,
  meaning the wafer / target edge cuts through the crop window.
  Distinct from `OFF_TARGET` (no wafer at all) because the
  operator remedy is reframing, not re-jogging. See DESIGN_NOTES
  §6.30.
- `LINEAR_CASCADE_ENABLED` — enable the 3-5 shot linear AA
  calibration cascade (default `True`). When `False`, falls back
  to the legacy iterative 10-cap cascade preserved in
  `_calibrate_aperture`. See DESIGN_NOTES §6.32.
- `PROBE_ENABLED` — enable Phase-A's coarse-to-fine focus probe
  (default `True`). Flip to `False` for tail-extension sessions
  (or use the `--no-probe` CLI flag for per-invocation override).
  DESIGN_NOTES §6.26 / §6.28 / §6.29.
- `OUTPUT_ROOT` — where session directories are created.
- `NTFY_TOPIC` / `NTFY_BASE_URL` — ntfy.sh topic + endpoint. Set the
  topic once per laptop.
- `ALERT_WAV` — local audible alert file (paplay) for droop /
  off-target events.
- `JOG_FINE_MM` / `JOG_COARSE_MM` — rail jog increments.
- `CROP_*_FRAC` — central-crop position and size on the wafer.
- `ROLLING_WINDOW`, `DROOP_CRIT_A`, `DROOP_CRIT_B` — droop detector
  window length + the two trip criteria
  (default: 1 severe shot ≤ −2.0 EV OR 2-of-10 ≤ −0.7 EV).
  The detector pools shots session-wide so consecutive
  under-exposed shots at *different* apertures still trip the
  moderate criterion. On any trip the runner reshoots at least
  the current step's apertures (rewinding further when earlier
  steps in the window are also under-exposed).
- `RANGE_CHECK_ENABLED`, `RANGE_EXTENSION_FRACTION`,
  `RANGE_CHECK_PROBE_STEPS` — auto-extension behaviour. The
  multi-crop advisor checks all 17 crops per aperture; field
  curvature where corners truncate on opposite sides maps to
  `peak_side="both"` and triggers bilateral extension. The
  extension target prefers the advisor's per-aperture
  `extra_steps` projection (max across truncated apertures, capped
  at the original sweep length); `RANGE_EXTENSION_FRACTION` is the
  fallback budget when the advisor returns 0.

## Troubleshooting

| Symptom | Likely cause | Fix |
|--|--|--|
| "Config tree too small" at startup | Camera came up in minimal-MTP mode | Battery-pull recovery (camera prep §1–8). |
| `PTP General Error` / `I/O problem` on capture (camera fires but file transfer fails) | USB 2.0 cable on the Z7; can't sustain 12 MB bulk transfer | Use a USB 3.0+ cable. Confirm with `lsusb -t` shows the Z7 at `5000M`, not `480M`. |
| "Camera not in M mode" | Mode dial wrong, or U2 bank reset | Set dial to M. |
| "focusmode is 'AF-S'" (or AF-C / AF-F) | Lens focus mode isn't Manual | Set MF on the body (Fn1 + sub-command-dial on the Z7). The widget is read-only over PTP; the runner can detect but not fix it. |
| FEC sanity check fails (drop < 1 EV) | SU-800 mounted instead of SB-5000, lens cap on, or remotes not in remote mode | Check commander; confirm flashes fire; press `j` at the pre-sanity-check pause to jog the rail onto a focused subject before re-running. |
| All test shots have similar luminance | Crop is on a saturated highlight, or ambient overwhelming flash | Verify overlay JPEG; close room blinds; check wafer position. |
| Rail moves but no captures arrive | gphoto2 USB session corrupted | Stop program, unplug-replug camera USB, rerun. |
| StackShot move never completes | Cable issue or rail jam | Check StackShot LCD for error; power-cycle StackShot. |
| Lens-change reconnect fails 3 times | Camera USB state stuck | Full battery-pull camera recovery, then ENTER. |
| Notifications not arriving on Mac | ntfy topic mismatch, or Mac not subscribed | Check `NTFY_TOPIC`; open ntfy app, verify subscription. |
| ntfy entries arrive in the Mac app's history but no banner / sound | The macOS "ntfy" App-Store app is actually the iPad app via "Designed for iPad" / Mac Catalyst. It doesn't reliably hold a streaming subscription in the background and macOS doesn't expose it as a background activity. Reliable alternatives: (1) the iOS / Android apps (real APNs/FCM push); (2) the **web client** at `https://ntfy.sh/<topic>` in a pinned browser tab — entries arrive live and `paplay` on bumblebee gives you the audible chime at the rig regardless. macOS browser banners can be enabled per-site but are also unreliable for some users. |

## Project background

See `DESIGN_NOTES.md` for the full history of design decisions, alternatives
considered, and the empirical findings that shaped this plan. That document
exists primarily for future maintenance work — it explains *why* the design
is the way it is, including several substantial detours (SU-800
incompatibility, libgphoto2 quirks, USB enumeration issues) that don't show
up in the final code but matter if you're trying to extend or debug.
