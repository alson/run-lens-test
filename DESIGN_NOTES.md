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

## 5. What would be different if redoing the project

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
