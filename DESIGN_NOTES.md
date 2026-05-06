# Design Notes — Macro Lens Sharpness Test Automation

This document records the design history of the project: what was tried,
what worked, what didn't, and why the architecture ended up as it did. It
exists for future maintainers (including the original author after a few
months away from the code) and for any subsequent AI-assisted work that
needs to understand context the final plan does not show.

The plan itself is in `macro_lens_test_plan.md`. The user-facing
operating procedure is in `README.md`. This document covers the *why*.

---

## 1. Goals and starting point

The user manually tests macro lenses for sharpness. The setup:

- Nikon Z7 (1st generation) on a StackShot focus rail.
- Silicon wafer test target.
- Two off-camera flashes (SB-800 + SB-5000) wirelessly controlled.
- For each lens configuration (extension tubes, diopters, focus distance),
  the user wants to take 3–4 shots at multiple apertures (f/4 through f/16).
- Battery droop in the flashes is a real problem during long sessions: the
  flashes start under-exposing as recycle slows.

The starting workflow had the StackShot trigger the camera via shutter sync
cable, with the user programming the StackShot's auto-step sequence and
manually changing camera settings between sequences. The goal was to
automate everything from a script.

## 2. What changed during planning

Major architectural decisions that ended up different from the initial sketch:

### 2.1 Direct rail control instead of pre-programmed sequences

**Initial plan**: the user programs the StackShot's auto-step mode for start,
end, and step count. The script triggers the StackShot's start, polls for
sequence completion, and downloads JPEGs as they arrive from the camera. The
sequence is repeated once per aperture.

**Final plan**: the script drives the StackShot directly via its USB serial
protocol, one move at a time. At each rail step the script takes one shot
per aperture. This means a single rail sweep produces shots for all apertures.

**Why we changed it**:

1. The Cognisys protocol PDF (`StackShotCommands_1_2.pdf`) documents
   `CC_RAIL_MOVE`, `CC_RAIL_STATUS`, `CC_RAIL_POSITION_*`, and `CC_RAIL_STOP`.
   These are sufficient for direct control without involving the StackShot's
   auto-step state machine at all.
2. With apertures sampled at every rail step, all apertures share the same
   focus position — so any sharpness comparison across apertures is not
   biased by mechanical drift between independent sequences.
3. Total rail moves drop from K×N to N (where K = apertures, N = steps). For
   K=5, N=30 that's 150 → 30. Less wear.
4. Battery droop affects all apertures uniformly through the sweep, instead
   of biasing whichever aperture's sequence runs last.
5. FEC calibration only needs to happen once per lens configuration (at rail
   step 0), and the calibrated FEC values are reused across all 30 steps.
   With per-aperture sequences we'd recalibrate for each aperture's sequence.
6. A droop pause loses at most one rail step's worth of shots, and the rail
   simply stays put while the user swaps batteries — no re-programming.

### 2.2 SB-5000 commander instead of SU-800 (forced change)

**Initial plan**: continue using the user's existing rig with the SU-800 as
hot-shoe commander, in AA mode.

**Final plan**: SB-5000 mounted on the hot shoe as commander, master output
`--`. SU-800 is incompatible with this project on the Z7.

**Why we changed it**: empirical testing. With the SU-800 on the hot shoe,
every camera-side FEC and EC change wrote successfully to the camera and
read back correctly, but had **zero effect on the actual flash output** —
verified by sweeping FEC from −3 to +1 EV and seeing the wafer's median
luminance stay flat at ~180. With the SB-5000 commander, the same sweep
produced JPEG p50 values of 22 / 37 / 63 / 100 / 143, almost exactly halving
per stop, which is the linear-light response we expected.

This matches Thom Hogan's "Complete Guide to the Nikon Z6 & Z7" stating the
SU-800 is incompatible with Z6/Z7 (with the practical caveat that it does
trigger remotes but cannot accept compensation from the camera body). The
Z7 dropped the in-body CLS forwarding path that older Nikon DSLRs used to
relay camera-side compensation to an SU-800.

The SB-800 supports commander mode and is generationally similar to the
SU-800, but was not validated in this project. If the user ever loses
SB-5000 access, they should retest with SB-800 commander before assuming
they need to redesign the project.

### 2.3 FEC as fine-tune lever, not ISO

**Considered**: using ISO as the only exposure adjustment lever, since ISO
is reliably writable via gphoto2 in M mode.

**Rejected because**: in AA flash mode, raising ISO causes the flash's own
sensor (with its threshold communicated by the camera) to quench *sooner*,
not the camera to amplify. Within the flash's working range, ISO produces
no change in image brightness — it's a range-extender, not a tuner. The
only true fine-tune lever in AA mode is something that biases the flash's
target: FEC or EC.

**Final approach**: FEC for closed-loop fine-tune (with EC as a fallback if
FEC is read-only on a given firmware), ISO for range extension at small
apertures and as a last resort when FEC saturates.

### 2.4 python-gphoto2 with persistent connection, not gphoto2 CLI

**Considered**: shelling out to the `gphoto2` command-line tool per command.
Simpler, no Python binding dependency.

**Rejected because**: each gphoto2 invocation re-handshakes the camera over
PTP, which on Z7 takes ~30 seconds of driver-loading per call. More
critically, the property tree returned right after a fresh handshake is
sometimes the cut-down 23-property MTP tree instead of the full 417-property
Nikon tree, depending on the Z7's USB enumeration state.

**Final approach**: open one `gp.Camera()` connection at session start, keep
it for the whole session, drain events between captures, explicitly delete
captured files from the SDRAM buffer after download.

### 2.5 JPEG-only, not RAW+JPEG

**Considered**: capture RAW+JPEG, download only the JPEG for analysis,
leave RAW on the SD card.

**Rejected because**: with `capturetarget=Internal RAM` (which we need for
fast tethered transfer), the Z7 alternates which file gets put in the SDRAM
slot when imagequality is RAW+JPEG. Sometimes you get the JPEG, sometimes
you get the NEF. Filtering by extension on download works but adds
complexity for no real benefit — the user does sharpness analysis on JPEG
in the existing workflow.

**Final approach**: `imagequality=JPEG Fine` for the whole session. RAW
capture is out of scope.

### 2.6 ntfy.sh notifications

**Considered**: iMessage, Pushover, email, custom HTTP server.

**Final approach**: ntfy.sh. Free, no account, native macOS app, single
HTTP POST per alert. The user installs the ntfy app on the Mac and
subscribes to a private random topic name; the Linux laptop POSTs to the
same topic. Sound and notification both come from the Mac app's settings.

## 3. Empirical findings during planning

These are the discoveries that shaped specific operational requirements.
They're listed here so future debugging starts from informed ground.

### 3.1 Z7 USB enumeration is fragile

The Z7 sometimes enumerates as a generic Microsoft MTP device (returning a
23-line config tree with only basic file-transfer ops) and sometimes as a
full Nikon PTP device (returning a 417-line tree with all the camera
controls). The trigger for the bad state is unclear but seems related to
USB state retained across power cycles.

The reliable way to force the good state:

1. Power off the camera with the power switch.
2. **Remove the battery and wait at least 10 seconds.**
3. Reinsert the battery.
4. Power on. Wait for live view to appear on the rear screen.
5. THEN connect the USB cable.

Power-switch-off alone does not work. USB cable disconnect alone does not
work. Battery removal is the only reliable reset.

This is now baked into the operating procedure as a hard requirement and
the program checks the config tree size at startup.

### 3.2 USB cable rabbit hole (eventually a red herring)

Significant time was spent suspecting the USB cable, including verifying
USB 3 negotiation at 5000M vs 480M. Tests passed at both speeds
inconsistently. The cable was *not* the cause — the camera's own
enumeration state was — but the diagnostic process is documented in case
similar symptoms come up later. lsusb's `-v` showed the device correctly
advertising Imaging class 6 subclass 1 throughout, which ruled out the USB
layer.

### 3.3 SDRAM buffer staleness

`gp.Camera.capture()` returns whichever object is currently in the SDRAM
buffer, not necessarily one taken in response to that call. If the buffer
has a stale frame from earlier (e.g. from a half-press shutter wake, or a
spontaneous capture during config changes), `capture()` returns that frame.

This caused several confusing test results (a "no flash" reference shot
that looked identical to a flash-on shot, FEC sweeps that appeared flat).

The fix is `drain_events()` before each capture: call
`camera.wait_for_event()` in a loop with a short timeout to swallow any
pending events and stale `GP_EVENT_FILE_ADDED` notifications. Also
explicitly delete the captured file from the camera filesystem after
download so the buffer doesn't accumulate.

### 3.4 Property values transiently return '?'

After a burst of `DevicePropChanged` events (which the camera emits
liberally — turning off a flash can produce 21 of them), an immediate
`get_config()` call may return `'?'` for some property values while the
camera is still refreshing its internal state.

The fix: when a check expects a specific value (like `expprogram == M`),
re-fetch up to 5 times with 200ms sleeps before failing. A `'?'` is a
transient and should be tolerated; only a definite non-matching value
should abort.

### 3.5 EXIF flash compensation tag is unreliable on Z7

The EXIF `FlashExposureComp` field is not written by the Z7 in JPEGs
(empirically, exiftool does not find it regardless of FEC value). Don't
use EXIF readback to verify FEC — measure pixel brightness instead.

The EXIF `Flash` field *is* reliable: it correctly reports
`Off, Did not fire` when no flash light reached the sensor and `On, Return
not detected` when it did. Useful as a smoke test.

### 3.6 AA mode external flash sensor

Initial confusion was around how AA mode actually works. Definitive
answer: the flash unit's own external sensor (the small thyristor window
on the front) measures reflected light during the main pulse and quenches
the flash via IGBT when the threshold is reached. The threshold is set
based on aperture and ISO communicated by the camera. The preflash that
fires in AA mode is for white balance and FV Lock, not for flash power
calculation. This is fundamentally different from i-TTL where the
camera's metering sensor reads preflash light through the lens.

Implications:

- **AA does not compensate for bellows factor.** Extension tubes reduce
  light through the lens but the flash sensor sees the scene
  independently and doesn't know about the loss.
- **AA is unreliable on specular targets** (like silicon wafers). Small
  tilt changes cause large swings in what the sensor sees.
- **ISO within working range is invisible to AA.** Raising ISO tells the
  flash to quench earlier, image stays the same brightness.

### 3.7 EV math at low values

The exposure analysis converts JPEG pixel values to linear with gamma 2.2
before computing EV deltas. This is approximate at very low pixel values
because the Z7's JPEG tone curve is steeper in the shadows than gamma
2.2 implies, and the floor at value=1 to avoid log(0) caps the sensitivity.

In practice: at FEC −3 EV, the measured ΔEV came out as −4.71 (expected
−3) because of this. The flash output ratio was actually correct
(p50 22/100 ≈ ½⁵ in linear sensor units, which is consistent with linear
flash quenching).

For fine-tune we operate within ±1.5 EV of baseline where the
approximation is fine. For droop detection we look at sustained
under-exposure relative to baseline at a known-stable aperture, where the
approximation is also fine. The error only shows up if the user tries to
diagnose the system at extreme FEC values.

### 3.8 "Battery droop" is actually flash-specific

The problem the user wants to detect is the AA-controlled flashes losing
power as their AA batteries deplete. This causes:

1. Slower recycle (longer between shots).
2. Eventually, under-exposure when the flash can't deliver the demanded
   output.

Symptom (1) is hidden by our recycle settle delays. Symptom (2) is what
the rolling-window detector watches for, against a baseline established
at session start. The detector lives per-aperture so that a drift in one
aperture's exposure isn't masked by another aperture's correct shots.

## 4. Things considered but not implemented

- **WiFi to a StackShot 3X.** The user has the original (text LCD) model;
  the protocol differs in places. If the user upgrades to a 3X, the
  protocol module would need updating but the rest of the architecture
  is unchanged.
- **Live view streaming.** Not needed for this workflow.
- **Automatic SU-800 / flash power adjustment.** Not possible — physical
  buttons only. Moot since SB-5000 commander is now the design.
- **Sharpness analysis itself.** This program collects data; the user
  has a separate workflow (in digiKam) for sharpness measurement.
- **GUI.** Terminal only. Adds significant complexity for a tool used
  by one person.
- **Multi-camera support.** Not needed.
- **Resume from a session crash mid-sweep.** The session.json + shots.csv
  contain enough state to do this in principle, but it's not implemented
  in v1. A crash means restart from the current lens config.
- **Nikon SDK on Linux.** Nikon SDK is Mac/Win only. Considered briefly
  to potentially get FEC writability that gphoto2 might lack, but
  empirical testing showed gphoto2 + SB-5000 commander gives us
  everything we need.

## 5. On-rig discoveries (2026-05-05)

Validated and refined on the rig over a full afternoon of integration
testing. The session log of changes:

### 5.1 Z7 needs USB 3.0+ cable

A USB 2.0 cable enumerates the Z7 at 480 Mbps. Capture appears to
succeed (shutter fires, flashes go off) but the post-capture file
retrieval throws `PTP General Error / I/O problem` and wedges the
gphoto2 USB session. The 12 MB JPEG bulk transfer simply can't
sustain through a 480M link. Use a 5 Gbps+ cable; verify with
`lsusb -t` showing the Z7 at `5000M`. Documented in the README
troubleshooting table.

### 5.2 Capture timing — batching + recycle subtraction

Profiled with `tools/profile_capture.py`. With three separate
`set_aperture` / `set_iso` / `set_fec` calls per shot, each
`get_config → mutate → set_config → drain_events` round-trip costs
~1.5–3 s on the Z7 — total ~6.6 s of PTP overhead per shot.
Batched into a single `set_aperture_iso_fec` (one `get_config`,
mutate three widgets, one `set_config`, one drain) it drops to
~2.5 s. On a 5-aperture step the saving is ~20 s; on a 750-shot
session it's ~17 minutes.

The flash recycle wait (per-aperture `RANGE_EXTENSION[ap][1]`,
3–6 s) is now subtracted by elapsed time since the previous
shutter — `set_aperture_iso_fec` runs *during* the recycle window
rather than on top of it. The recycle is keyed on the *previous*
shot's aperture (small apertures = more flash power discharged =
longer recycle), not the upcoming one, so the f/16 → next-step
f/4.5 wait happens after the f/16 capture and is mostly absorbed
by the rail move.

### 5.3 BASELINE_REFERENCE_IMAGE

Each session was previously calibrating against the first
widest-aperture shot of the first lens config — fine for
within-session comparability but lens-config-dependent for
inter-session comparability (a session that opens with 105 mm + 1:1
+ extension tubes lands at a different baseline than one that
opens with no extension, even though the wafer reflectance is the
same). `BASELINE_REFERENCE_IMAGE` lets you point at one canonical
JPEG; the runner reads its central-crop median once at session
start and that becomes the absolute reference for every session.

The median needs to be reachable for the rig. If the chosen
reference is too bright for the current lens / extension /
flash-distance combo, the calibrator can't close the gap and every
sweep starts under-exposed. Pick a representative previous shot
at a similar magnification.

### 5.4a Lever cascade: FEC → EC then plateau-triggered ISO

Originally the calibrator's lever cascade was FEC → EC → ISO with
ISO bumping unconditionally as soon as both FEC and EC reached
their camera-side limits. That's wrong for AA flash: the flash's
auto-quench is calibrated to the camera's (aperture, ISO, FEC/EC
bias). Raising ISO without a *mechanical* flash plateau just
makes the flash quench sooner — image brightness unchanged. ISO
is purely a **range-extender**: it only buys real brightness when
the flash is firing at its mechanical maximum (full power, can't
fire harder, can't fire longer).

Detection rule: track cumulative positive FEC+EC bumps applied
since the last reset and the median we had at the start of that
push. If after ≥ 1 EV of cumulative bumping the luminosity hasn't
moved by at least 50% of the requested EV, the flash has plateaued
and an ISO bump becomes useful. Otherwise (FEC+EC saturate while
the flash *was* responding stop-for-stop), the calibrator accepts
the result without an ISO bump — the flash had room, only the
camera's bias dial ran out, so ISO would just shift the auto-
quench earlier and gain nothing.

Initial ISO per aperture stays in the existing
``RANGE_EXTENSION`` table (e.g. f/11 → 200, f/16 → 400) — that's
range-extension upfront for apertures the flash can't comfortably
reach at base ISO.

### 5.4 Calibrator FEC-saturation off-by-one (fixed mid-session)

Real bug surfaced when calibration was 1.7 EV under target with
FEC headroom available: the saturation check used `new_fec`
(the clamped *target*) instead of `fec` (the current value).
First shot at FEC=0, dim by 1.7 EV → `new_fec = clamp(+1.667, …)
= +1.0 == FEC_MAX_EV` → misread as "FEC already at max" → ISO
bumped instead of FEC set. Loop never escaped FEC=0; calibration
exhausted retries with all 6 ISO bumps and gave up. Fixed
2026-05-05; check uses *current* `fec`.

### 5.5 Multiple shots per (rail step, aperture)

JPEG-encoding noise on a single shot can mask a weak focus signal:
on a 7-step sweep where the bell amplitude is small, file-size
max/min landed below the OFF_TARGET threshold (1.15) even when the
data was real. `SHOTS_PER_STEP_APERTURE = 3` captures three replicas
per (step, aperture); the advisor and `SharpnessRangeChecker` call
`CollectedDataset.aggregated()` to take a per-crop median across
replicas before peak detection. Replica 0 keeps the legacy filename
(`step_NNNN.jpg`) so existing offline-analysis tooling works
unchanged; replicas 1, 2, … get `_b`, `_c`, … suffixes.

### 5.6 Droop reshoot from earliest under-exposed step

Droop pause originally just reset the rolling window and continued
forward — but the under-exposed shots before the trip stayed in
the dataset. Now on resume the runner finds the earliest step
across all apertures whose EV ≤ the firing trip threshold, rewinds
the rail there, and reshoots that step through the trigger step
inclusive. The reshoot's EVs feed the droop detector again; if a
flash you didn't catch on the first swap is still weak, droop
re-fires and `_handle_droop` recurses into another reshoot. CSV
stays clean across nesting: each `_reshoot_steps` start drops
`phase ∈ {sweep, aborted, reshoot}` rows for its range so a nested
reshoot's clean rows replace its parent's bad ones.

### 5.7 Off-target detection + re-jog

The `Verdict.OFF_TARGET` aperture verdict (no truncated *and* no
`CropStatus.OK` crops) propagates as `RangeResult.all_off_target`
when *every* aperture lands there. The runner alerts (ntfy +
chirp), prompts `[ENTER] re-jog / [s] skip / [q] abort`, and on
re-jog reruns the jog UI without re-calibrating; the prior sweep's
shots.csv rows are dropped. Triggers when the rail span misses
the focus zone or the wafer position lacks micro-detail — both
concrete failure modes seen on the rig.

### 5.8 Per-aperture cropped-output layout

The original cropper wrote to `<lens>/cropped/<crop>/<source>`,
and since every aperture's source files share the
`step_NNNN.jpg` naming, only the *last* aperture's crops survived
on disk (the in-memory dataset was unaffected, so the live
advisor's decisions were correct, but offline analysis lost 4/5
of the data). Now production wiring sets
`crops_under_source=True` so crops land at
`<lens>/<aperture>/cropped/<crop>/<source>`. Documented in §8.3.5
of the implementation plan; matches the user's existing
offline-analysis layout.

### 5.9 Magnification-driven step size; subsample factors deferred to Phase B

The runner used to ask the user for an explicit step count, then derive
`step_size_mm = end_mm / (steps - 1)`. Step size was one constant for
the whole sweep regardless of which aperture was being shot.

The user built `sharpness/planner.py` to do the math properly. Given
the lens's widest aperture (the worst case for sampling — narrowest
DOF, sharpest peak) and the magnification, `plan_sweep` returns a
physically-motivated `rail_step_mm` from the thin-lens DOF formula
(`DOF = 2·c·N·(1+M)/M²`, c = 30 µm, FX sensor) and density of 5
positions across FWHM (8 if M ≥ 1.5).

The runner now prompts for **magnification** right after the
lens-config name (accepts ratio forms like `1:1`, `1.5:1`, `1:1.5`,
`2:1`, plus bare floats), reads the **widest aperture** from the
camera, and after the jog calls the planner with `rail_range_mm =
abs(end_mm)`. The result drives `(rail_step_mm, rail_positions)`.

If the resulting position count is below `MIN_RAIL_POSITIONS`
(default 10), the runner refuses with a re-jog prompt — 10 is the
floor for reliable peak detection at the widest aperture's narrow
DOF.

The `SamplePlan` (including `subsample_factors` per aperture) is
persisted in `session.json`'s `lens_configs` list — which is now
`list[dict]` instead of `list[str]`. Each entry carries `slug`,
`magnification`, `widest_aperture`, and the full plan. Downstream
sharpness analysis reads the subsample factors to skip every Nth
position at smaller apertures (where the broader DOF doesn't
require the same density).

**Capture-time subsampling is deferred (Phase B)**. This phase
captures every aperture at every rail step regardless of subsample
factor. Skipping at capture time would save rig time but requires
adapting:
- the advisor's `peak_index` / `extra_steps_needed` math (currently
  assumes contiguous indices),
- the droop detector's rolling window (currently 10 shots; with
  subsampled f/16 those 10 shots span 5× more rail time, dilating
  the droop signal),
- the probe / extension shooting grid (today shoots all apertures
  at each new step; with subsampling it should honour per-aperture
  factors).

The data format is forward-compatible: subsample_factors stored in
session.json today are exactly what Phase B will honour at capture
time tomorrow.

### 5.10 Sample-planner calibration: where the constants come from

The planner's defaults (`POSITIONS_ACROSS_FWHM_DEFAULT = 5`,
`POSITIONS_ACROSS_FWHM_HIGH_MAG = 8`, `HIGH_MAG_THRESHOLD = 1.5`,
`DEFAULT_COC_MM = 0.030`, `MIN_RAIL_STEP_MM = 0.005`) are not
arbitrary — every one was derived from real data in
`analyse-lens-test/rail-coverage/output/scan-cache.json` (the
2026-01-29 + 2026-02-18 sessions, 545 scenarios). The calibration
notes live alongside the dataset in
`analyse-lens-test/ANALYSIS_NOTES.md`; this section summarises
which constants come from which evidence.

**Circle of confusion (`DEFAULT_COC_MM = 0.030`).** Back-derived
from the historical 25-position-per-scenario regime. At M=1, N=8
the median observed FWHM was 12 rail positions out of 25 (so the
sweep covered ~2 × FWHM). With `DOF = 2·c·N·(1+M)/M²` and that
sweep ratio, c ≈ 30 µm reproduces the observed step size — which is
also the standard FX sensor circle-of-confusion. The two values
matching is a coincidence that lets us use the literature value
without explicit calibration.

**`POSITIONS_ACROSS_FWHM_DEFAULT = 5`.** A resampling experiment on
the existing 25-position bell curves (Sigma 150 at 1:1, F5.6/F8/F11
center crops; see `ANALYSIS_NOTES.md §3`) showed peak detection
within 1.5 % of the fitted maximum at k = 8 sub-samples across the
whole sweep, which translates to ~5 positions across FWHM. Going
from k=8 to k=15 gave no further improvement; going from 25 to 50
positions also gave no further improvement. So 5 positions across
FWHM is enough at moderate magnifications.

**`POSITIONS_ACROSS_FWHM_HIGH_MAG = 8` and `HIGH_MAG_THRESHOLD = 1.5`.**
The same experiment with a 200 mm + 5T+6T at 2:1 / F8 scenario (a
high-mag, narrow-peak case) showed clear loss at k < 15 — k=10 was
4 % off the fitted peak; k=15 was within 1 %. At that scenario's
FWHM, 15 sub-samples translate to ~8 across FWHM. We trigger at
M ≥ 1.5 because the 1:1-to-1.6:1 magnification band was
empirically the "sweet spot" in `rail-coverage/output/report.html`
(58 % of scenarios in that band were "good", the highest of any
band) and high-mag-specific behaviour kicks in just past it.

**`MIN_RAIL_STEP_MM = 0.005`.** The Cognisys StackShot accepts
arbitrary float distances over the wire but the leadscrew + motor
assembly's mechanical resolution is ~2-5 µm. Below that, requested
moves disappear into mechanical noise. At extreme regimes (e.g.
M=3, widest f/1.4) the raw step recommendation drops below the
floor; the planner clamps and sets `step_clamped=True` so the
caller knows the bell will be under-sampled and there's no
recourse without different hardware.

**Aperture subsample factors.** DOF scales linearly with f-number,
so at aperture N the bell is `N / N_min` times wider than at the
widest aperture. Analysing every `floor(N / N_min)`-th captured
step keeps the same bell-curve density at every aperture. The
captured data has every aperture at every rail step (Phase B will
add capture-time subsampling per §5.9); the per-aperture
`subsample_factors` dict is what the analyser reads to honour the
optimal density at analysis time.

**Why "low-mag" and "small-aperture" bumps are NOT applied.** An
earlier draft had `+5` positions at M < 0.4 and `+3` at N ≥ 11 on
the assumption that broader, less-sharp bells benefit from more
samples. The resampling experiment showed the opposite: broader
bells are *more* tolerant of sparse sampling because the FWHM
spans more rail positions even at the same step. Those bumps would
have been counterproductive, so the planner has no per-band-hint
beyond the high-mag transition.

### 5.10 Phase B: capture-time subsampling

Phase A persisted `SamplePlan.subsample_factors` in `session.json`
but the sweep still captured every aperture at every rail step.
Phase B wires the factors into the runner's three capture loops
(`_run_one_step`, `_shoot_at_position`, `_reshoot_steps`) so an
aperture A at step S is shot only when `S % factor[A] == 0`. The
trip dropped one new helper (`_should_capture`) into each loop
and a per-lens-config lookup (`_subsample_factors`) on the runner.

**Why subsampling is sound analytically.** The planner sizes
`rail_step_mm` from the *widest* aperture's DOF — the narrowest
peak, the worst case for sampling. Wider DOF (smaller aperture)
→ broader FWHM → fewer rail positions needed for the same
peak-detection accuracy. The factors are computed as
`floor(N / N_widest)` so each aperture ends up with roughly the
same number of rail positions across its FWHM as the widest gets
across its (narrower) FWHM. The user's resampling experiment on
the existing 25-position dataset confirmed this empirically.

**Reshoot honours factors.** When droop fires and the rail
rewinds to overwrite under-exposed frames, the reshoot loop also
applies `_should_capture`. The reshoot grid mirrors the original
sweep's exactly — f/16 at factor 5 stays at every-fifth-step
inside the reshoot range, not every step. Series stays uniform
end-to-end.

**Skipped captures are silent.** No `phase=skipped` row in
shots.csv; no shots.csv row at all for the skipped (step,
aperture) pair. Sparse series shows up as absent rows;
downstream tooling (advisor, range checker) already iterates
actual rows, so no filtering required.

**Droop window dilation, accepted.** The droop detector keeps
a fixed 10-sample rolling window per aperture. With f/16 sampled
every fifth step, those 10 samples span 50 sweep iterations —
detection is 5× delayed in real time at f/16. Documented and
intentional: full-sampling apertures (f/4.5, f/5.6) trip first
because they're 5× more sensitive in real time, so the dilation
at smaller apertures doesn't matter on the rig. If false
negatives surface in practice, the fix is to pass per-aperture
`window // factor` into `DroopDetector` at construction time.

**Advisor sparse-safety: free.** Re-exploration during Phase B
planning confirmed `sharpness/advisor.py:advise()` already
iterates apertures independently, never compares lengths across
them; `sharpness/range_check.py:_build_curves` uses the real
`rail_step` from `dataset.crop_series_for(...)`, not array
indices. The runner uses `RANGE_EXTENSION_FRACTION` for the
extension-shot count, not the advisor's `extra_steps_needed`,
so the advisor's array-index-relative output (which is
technically wrong on sparse data) is harmless. Phase B added no
code in those modules.

**Throughput.** Typical session at M=1.58, widest f/4.5, 100
positions × 5 apertures × 3 replicas ≈ 1500 captures. With
Phase B factors `{4.5: 1, 5.6: 1, 8.0: 1, 11.0: 2, 16.0: 3}`,
captures drop to ~1150. At wider widest-apertures (f/2.8) the
factors fan out further (f/16: 5) for ~30 % savings. Plan
announcement now prints the per-aperture breakdown + savings
percentage so the user can sanity-check before committing.

### 5.11 Droop detector: session-level pooling + loosened criteria

Original design was per-aperture: each aperture had its own rolling
window, isolated from the others. Two trip criteria fired
independently per aperture (3 of last 10 ≤ -0.7 EV; 2 of last 10
≤ -2.0 EV).

The user flagged this as too insensitive. Real flash droop tends
to spread across consecutive captures regardless of aperture —
two adjacent shots at f/5.6 and f/8 both ≤ -0.7 EV is already
strong evidence of droop, but per-aperture pooling missed it
because neither aperture had been pushed against twice.

Refactored to **session-level**: a single rolling window across
all apertures. Criteria loosened:
* Severe (1 of last 10 ≤ -2 EV) — fires immediately on any single
  catastrophically dim shot. Catches "flash refused to fire / hit
  mechanical-min" failure modes.
* Moderate (2 of last 10 ≤ -0.7 EV) — was 3 of 10. Combined with
  pooling, a moderate trip now fires on the second moderate shot
  in the sweep regardless of which aperture it landed at.

Backwards-compat preserved: `window_for(aperture)` and
`window_entries_for(aperture)` filter the session-level deque by
aperture, returning what callers used to get from the
per-aperture deque. New `session_window_entries()` exposes the
unfiltered pooled view.

`earliest_underexposed_step(threshold)` still works correctly —
it iterates the session-level window and finds the lowest
step_index whose ev_value ≤ threshold, which is exactly what the
runner's reshoot rewind needs.

## 6. What would be different if redoing the project

- **Test the SB-5000 commander earlier.** The first half of the planning
  conversation assumed the user's existing SU-800 setup would work, and
  we burned a lot of effort debugging why FEC wasn't propagating. The
  Thom Hogan note that the SU-800 is "incompatible with the Z6 and Z7"
  is right there in dpreview threads from 2018 — but interpreted as
  "doesn't trigger remotes" rather than "drops camera-side FEC". A
  five-minute experiment swapping commanders would have caught it
  immediately.
- **Build the small empirical test scripts before the plan.** The
  diagnostic and FEC sweep scripts ended up doing most of the actual
  fact-finding. They should have been Step 0 of the project, not Step 0
  of the implementation.
- **Read the StackShot protocol PDF before assuming auto-step is the
  only option.** The direct-control architecture is materially better
  than what was originally planned. The protocol documents the moves
  and status reads we need; we never had to use auto-step at all.

## 6. Discoveries from on-rig integration (2026-05-05 session)

### 6.1 Per-shot PTP overhead dominates wall-clock time

The Z7's per-call PTP `set_config` cycle costs ~1.5–3 s wall-clock —
each one is a full get-config tree → mutate → set-config → drain-events
round-trip. The original sweep loop did three of these per shot
(aperture, ISO, FEC), plus the capture itself, which on the rig
measured 8.2 s/shot median. Batching aperture/ISO/FEC into a single
`set_aperture_iso_fec` call (one get_config, mutate three widgets,
one set_config + drain) dropped that to 4.0 s/shot — a 51 % win,
about 17 minutes saved on a 5-config × 30-step session.

The profiler that produced these numbers is checked in at
`tools/profile_capture.py` and can be re-run after firmware / cable
changes.

### 6.2 Recycle delay should subtract elapsed work

The per-aperture `RANGE_EXTENSION[ap].recycle_s` is a *minimum*
between consecutive shutter fires, not a sleep on top of work
already done. The runner now records `_last_capture_started_at`
just before each `capture_to`, and the wait — moved to *before* the
next capture — sleeps `max(0, recycle_s − elapsed)` so the
~2.5 s set_config of the next shot counts against the recycle
budget. Across thousands of frames this is hours saved.

The recycle is also keyed on the *previous* aperture's value (the
flash that just discharged), not the upcoming one — a small
aperture's longer recycle is paid *after* its capture, not before
the next.

### 6.3 Replicas-for-noise-averaging beat tighter thresholds

Initial on-target sweeps came back with `max/min < 1.15` per
aperture across the focus stack — i.e., the curves looked flat to
the advisor's `is_flat` heuristic. The fix wasn't to lower the
threshold (which risks treating real noise as a peak) but to
average multiple replicas at each (step, aperture) before peak
detection. `SHOTS_PER_STEP_APERTURE = 3` plus a per-crop median in
`CollectedDataset.aggregated()` gives a much cleaner sharpness
signal at the cost of 3× capture time per shot.

### 6.4 Droop reshoot must rewind, not skip-and-continue

First implementation reset the rolling window after droop and let
the sweep continue forward — the under-exposed frames at the trip
step and the previous samples that contributed to the trip stayed
in `shots.csv` and on disk as the canonical data. Wrong: every
sample ≤ the firing trip threshold should be overwritten with a
fresh capture. The runner now finds the earliest under-exposed
step across all apertures' droop windows, rewinds the rail, and
reshoots that step through the trigger step inclusive
(filenames are deterministic from step_index, so reshoot
overwrites in place). Stale `phase=sweep` rows for the reshot
range are dropped from `shots.csv`.

### 6.5 Droop must be re-detected during reshoot

Initial reshoot ran with droop disabled (window cleared, captures
not pushed back in). If the user missed a flash on the first swap,
or another battery was also weak, the reshoot itself produced
under-exposed frames that landed in the canonical filenames —
silently. Now reshoot pushes every capture's EV through the
detector; on a nested trip, `_handle_droop` recurses into
another `_reshoot_steps` call covering `[new_earliest,
current_step_index]`. Each reshoot's start-of-method drop wipes
earlier `phase=reshoot` rows in its range so the freshest
reshoot's rows are the ones that survive in the CSV.

### 6.6 Off-target ≠ truncated bell

A flat curve at every aperture is *not* the same as "needs
extension" — extending wouldn't add signal if there's no signal
to find. The cause is usually one of: (a) the rail span misses
the focus zone entirely, or (b) the wafer position lacks
micro-detail at the chosen crops. Both resolve via re-jog. The
advisor now returns `Verdict.OFF_TARGET` per aperture when no
crop has a real peak, and the SharpnessRangeChecker exposes
`RangeResult.all_off_target` when every aperture is off. The
runner branches on this flag *before* the extension flow and
prompts the user to re-jog.

### 6.7 In-memory dataset and on-disk crops have separate failure modes

The `LiveCollector` keys its in-memory `CollectedDataset` by
`(rail_step, aperture, replica_index)` — the advisor sees per-
aperture data correctly even when the on-disk crop output had a
bug. Concretely: the cropper's `out_dir` was lens-level
(`<lens>/cropped/<crop>/<source>`), so every aperture writing
`step_0100.jpg` overwrote the previous on disk; only the last
aperture's crops survived. The advisor was unaffected (in-memory
data is keyed by aperture), but offline analysis lost 4/5 of the
data. Fix is `crops_under_source=True` →
`<lens>/<aperture>/cropped/<crop>/<source>`. Worth keeping in
mind that the live decision and the persisted artefacts can drift
apart — verify both when investigating.

### 6.8 BASELINE_REFERENCE_IMAGE and the calibration ceiling

The first sessions set baseline from "first widest-aperture shot
of the first lens config" — fine for within-session comparability
but lens-config-dependent across sessions (a 105 mm + extension
opens at a different brightness than a 200 mm at 1:1). Fixing the
baseline to a designated reference JPEG's central-crop median
(default 129/255 for the user's reference) makes inter-session
comparisons valid. **Caveat**: if the current lens setup is
significantly dimmer than the reference (different focal length,
extension, or working distance) the FEC closed-loop calibrator can
hit FEC=0 + ISO < ceiling and still report `hit_tolerance=False`
— it gives up before pushing FEC to +1 EV. Worth a follow-up to
confirm the calibrator escalates fully through FEC max → ISO
ceiling before declaring failure.

### 6.9 set_ec silent no-op on positive / fractional EC (bug)

The Z7 EC widget reports its choices as **unsigned numeric thirds**
(e.g. `"0"`, `"1"`, `"3"`, `"1.333"`, `"-1.333"`), with the 1/3 and
2/3 fractions truncated rather than rounded:

```
Choice: 16 0.333
Choice: 17 0.666     ← truncated, NOT "0.667"
Choice: 18 1
```

The original `_format_ec_label` emitted the human-readable
`"+1"` / `"+1 1/3"` form, which the Z7 widget's choice list does
*not* contain. `set_ec` looked up the formatted label and silently
returned without doing anything when no match was found — no error,
no log line. Negative integer labels (`"-1"`, `"-3"`) happened to
match by coincidence; everything else silently failed.

Symptom: the calibrator's cascade reported "EC=+3" but EXIF on
saved JPEGs showed `ExposureCompensation: 0`. The cascade then
fell back to ISO bumps which, in AA flash mode, just make the
flash quench earlier and don't actually brighten the scene unless
the flash is at mechanical max. So calibration plateaued
inexplicably at f/4.5 with an apparent EV gap that never closed.

The 2/3-stop rounding discrepancy was a separate landmine: even
after fixing the format to use unsigned decimals, formatting
`2/3 ≈ 0.6667` with `:.3f` produces `"0.667"` (rounded up) but the
widget label is `"0.666"` (truncated). The whole-stop rounded
fallback then silently set EC=1.0 instead of EC=0.667. The fixed
formatter hard-codes `.333` / `.666` for the two third-stops.

`set_ec` now raises `CameraStateError` if neither the requested
nor the rounded fallback label is in the widget's choice list, so
future widget-format drift fails loudly. Coverage:
`tests/test_nikon_z7.py::TestSetEC`.

### 6.10 EC must be reset and re-applied per aperture

Two related runner bugs surfaced after the EC label fix:

1. `apply_session_config` reset FEC=0 at session start and on
   lens-change reconnect, but never touched EC. A session
   inherited whatever EC the camera was last left at — typically 0,
   but if any standalone tool or prior cascade had left positive EC
   the sanity-check baseline was that much brighter and the rig's
   true FEC response got mis-measured. Fixed: `apply_session_config`
   takes an `ec=0.0` kwarg and calls `set_ec` after `set_fec`.

2. `_set_capture_settings` (the per-shot wrapper) configured
   aperture / ISO / FEC but never set EC. After the calibrator
   converged f/4.5 on EC=+0.667 and f/16 on EC=0, the camera was
   left at EC=0 from the final calibrated aperture, so when the
   sweep started at f/4.5 every shot fired ~0.7 EV darker than
   calibration and triggered severe droop on the first frame.
   Fixed: the wrapper calls `set_ec(cal.ec_ev)` per shot.

Both bugs were latent for as long as the calibrator landed on
EC=0 across all apertures (the typical case before the rig drifted
into needing positive bias on the widest stop).

### 6.11 FEC sanity check: pixel-ratio threshold was too tight

The `fec_sanity_check` threshold was `low_median / baseline_median
≤ 0.5` in pixel units. With γ=2.2, pixel-ratio 0.5 corresponds to
a 2.20 EV drop in scene light, **tighter than the requested -2.0
EV drop itself**. So a perfectly-responding rig that delivered
exactly the requested -2.0 EV (pixel ratio 0.547) falsely failed.

Symptom: 148 → 78 medians, ratio 0.527 (= -2.02 EV in linear
light, i.e. perfect FEC response). Earlier sessions only passed
because the central crop happened to over-respond by ~0.84 EV
(less ambient fill, brighter spec at the crop position).

Fixed by switching to gamma-linearised EV math: `drop_ev =
log2(linear(baseline) / linear(low))`, threshold `min_drop_ev=1.0`.
Half the requested -2 EV drop, leaves margin for ambient-light
fill at the central crop while still catching a totally
non-responsive setup (drop ≈ 0).

### 6.12 Focus mode is readonly via gphoto2 on Z lenses

Z lenses are focus-by-wire and the camera body's focus mode (set
via the Fn1 + sub-command-dial combination on the Z7) isn't
writable over PTP. gphoto2 exposes `focusmode` as `Readonly: 1` —
we can only *observe* whether it's `Manual` / `AF-S` / `AF-C` /
`AF-F`.

The `autofocus` widget (`/main/settings/autofocus`) is writable
and the runner sets it `Off`, but that only suppresses AF
triggering on shutter — the lens focus mode itself can still drift
under AF-S. So `verify_healthy` now refuses to start the session
unless `focusmode == "Manual"`, with a CameraStateError pointing
the user at the body-side focus-mode toggle.

### 6.13 close() must tolerate a powered-off camera

Lens-change recovery instructs the user to power off the camera,
swap the lens, pull the battery, reinsert, power back on, *then*
press ENTER. By that point the camera is gone from USB. The
runner's old `close()` did `try: camera.exit() finally: handle =
None` — the `finally` cleared the handle but `GPhoto2Error -52
("Could not find the requested device on the USB port")`
propagated out of the function and killed the session. Fixed:
`close()` catches `GPhoto2Error`, logs at info, and proceeds to
`_camera = None` so the subsequent `open()` runs cleanly against
the freshly-powered-on camera.

### 6.14 Optional freeform jog at session start

The StackShot retains its rail position across program crashes
and across lens swaps, so the rail (and target focus) is often
nowhere useful when a fresh runner launches. Two consequences:

1. The FEC sanity check fired against a defocused central crop,
   producing flaky ratios that drifted near the threshold edge.
2. After a previous crash, the StackShot can be left in
   USB-control mode — the panel buttons are inert until a process
   releases USB, but there's no way to release without entering
   the jog UI.

Added `jog_freeform` injection + `_offer_jog_pause(where)` helper
called at two new points: right before the FEC sanity check, and
at the top of each lens-config prompt iteration. Prompts
`[ENTER] continue, [j] jog rail, [q] abort:`; pressing `j` enters
the freeform jog (which already releases the camera for live view
via SPACE → on_release callback) and re-prompts on exit. Default
`None` = no-op, so existing test fixtures stay backward-compatible.

### 6.15 Rig flash headroom envelope (commander +1.7 EV)

Measured 2026-05-06 with the macro rig in the user's standard
configuration (Z7 + Z MC 105 + 61 mm extension, SB-5000 commander
in AA mode with +1.7 EV group bias on A & B, one SB-800 in each).
FEC and EC each track linearly until the SB-800s hit mechanical
full-dump (~6 s recycle, blinking ready light); below that they
stack additively (`FEC=-1 + EC=+1 ≈ FEC=0 + EC=0` to within
0.04 EV). Mechanical max lands at:

* **f/4.5**: > +3.5 EV above (FEC=0, EC=0) baseline. The cascade
  max (FEC=+1, EC=+3) doesn't quite reach it — JPEG starts
  highlight-clipping (~p50=237, 2.4% clip) before the flash runs
  out of headroom. Confirmed by 3.8 s recycle and *no* underexposure
  blink at this corner.
* **f/8**: ≈ +2.6 EV. Cascade hits this at FEC=+1, EC=+2; adding
  EC=+3 produces zero further output (p50 stays at 183-184).
  Confirmed full-dump regime by user (6 s recycle, blinking).

ISO 400 vs ISO 100 at f/8 confirms the limit is **flash physical
output**, not AA-loop / commander metering saturation: at ISO 400
the same FEC=+1 + EC=+2 reaches +3.00 EV (vs +2.64 at ISO 100),
because at higher ISO the AA target needs less photon output and
the SB-800 has 2 more stops of mechanical headroom. Below the cap
ISO 100 and ISO 400 are identical (FEC=+1 + EC=+1 → p50=158 at
both ISOs) — the textbook signature of an output limit.

Implication for calibration: the existing plateau-detect → ISO
range-extend mechanism is doing the right thing; ISO is the only
lever that genuinely adds light to the sensor when the flash is
mechanically maxed.

The +1.7 EV commander bias uses ~half the apparent f/8 envelope.
Removing it (or reducing it) would give the calibrator ~1.7 EV
more cascade headroom at every aperture. Worth re-evaluating if
sessions repeatedly fail to reach baseline at smaller apertures.

### 6.16 Per-replica droop short-circuit

Originally `_capture_one` captured all N replicas and pushed only
the median to the droop detector. With `SHOTS_PER_STEP_APERTURE=3`,
a single severely under-exposed replica still wasted two more
shots before the failure was even evident — particularly painful
on the severe criterion (1 of 10 ≤ -2 EV) which fires on a single
replica. `_capture_one` now pushes per-replica and breaks the loop
on the first trip event, returning `(median_ev, droop_event)` so
the caller can react.

### 6.17 Sweep progress: replica counter + per-lens-config ETA

Two UX additions to the sweep progress bar:

1. **Replica counter**: `rep N/M` appended when
   `SHOTS_PER_STEP_APERTURE > 1` so the bar advances during the
   3-replica burst at each (step, aperture) instead of looking
   frozen for several seconds.
2. **ETA**: tail-end `eta M:SS` (or `H:MM:SS`) computed from
   `remaining_shots × interval_average`. `remaining_shots` walks
   the (step, aperture, replica) grid forward from the current
   position honouring per-aperture subsample factors. The EWMA was
   later replaced by a cumulative mean per lens config (see commit
   cba4b83); cumulative is more stable across the sweep's natural
   rhythm of fast (subsample-skipped) and slow (full-aperture)
   steps. Battery-droop pauses invalidate the next interval via
   `_eta_invalidate()` so a multi-minute swap doesn't poison the
   average. Reshoots aren't anticipated by the grid math, so the
   ETA runs slightly optimistic after a droop recovery —
   acceptable for guidance purposes.

### 6.18 F16-only-truncated entries surfaced in the lens-config menu

The rail-coverage analyser's reshoot list (see
`analyse-lens-test/ANALYSIS_NOTES.md` and
`reshoot-plan-2026-05.md`) tags six lens configs as
**F16-only-truncated**: F4-F11 sweeps are all complete with on-target
peaks, only F16 is truncated. Since the user has stated F16 is the
least important aperture, those six configs are optional to reshoot.

The plan-file `Status` column carries this as
`reshoot (F16-only)`. The lens-config picker in `ui/prompts.py`
surfaces it in two ways:

1. **Marker**: `[F16-opt]` instead of the empty leading whitespace
   used for normal entries. The done marker (`[done]`, set when a
   completed slug is in `completed`) takes precedence when both
   apply.
2. **Sort order**: F16-only entries sink to the bottom of their
   group via a stable partition. Group order is preserved; within
   each bucket (normal vs F16-only) the original plan order is
   preserved.

Detection is a case-insensitive substring match on `f16-only` in
the loaded `PlanEntry.status`, so future plan files can use
`reshoot (F16-only)`, `(F16-only)`, or `F16-only optional`
interchangeably without code changes.

Picking an F16-only entry from the menu does not change session
behaviour — the runner shoots all apertures as usual. The marker
is purely informational so the user can defer those configs when
time is tight.
