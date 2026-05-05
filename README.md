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
4. Mount the **SB-5000 on the hot shoe in commander mode**, master output `--`,
   group A in `A` (auto aperture) mode, group FEC = 0.
5. Switch on the remote flashes (SB-800 + SB-5000), check ready light.
6. **Power on the camera.** Wait for the rear screen to show live view.
7. Half-press the shutter to confirm the camera is awake.
8. **Now connect the USB cable from the camera to the laptop.** Plugging in
   before power-on tends to make the camera enumerate as a minimal MTP
   device and the program will refuse to start.

If at any point the program reports "config tree too small — battery-pull
recovery needed", that means the camera dropped into the minimal-MTP state
and you need to repeat steps 1–8.

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

The program will:

1. Open the StackShot and camera.
2. Validate camera state (M mode, full config tree, FEC sanity test).
3. Prompt for the first lens configuration as a free-text line.
4. Drop you into a jog session: use `a`/`d` to step the rail back/forward
   in fine increments, `A`/`D` for coarse, `s` to mark the start position,
   `e` to mark the end, `q` to abort. The current position is shown live.
5. Ask for a step count.
6. Run FEC calibration at step 0 (one test shot per aperture, possibly more).
7. Run the rail sweep.
8. Notify on completion and prompt for the next lens config.

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
3. Press ENTER in the program. The rail stays at its current position; the
   program redoes the current rail step and continues.

You do **not** need to re-jog the rail, the StackShot is being driven by
the program throughout.

### Quitting

Enter an empty line at the lens-config prompt to quit cleanly. The program
moves the rail back to position 0, closes the camera and rail, and writes
the final session log.

Ctrl-C also works — the program issues a StackShot stop, closes connections,
and exits. Any in-progress shot may be lost.

## Output

Per-session directory under `OUTPUT_ROOT`, named with a timestamp.
Inside it:

- `session.json` — session metadata including chosen lens configs, calibrated
  FEC values per aperture per lens, ntfy topic, StackShot software ID, etc.
- `shots.csv` — one row per shot taken, with rail step, aperture, ISO, FEC,
  measured median luminance, EV-relative-to-baseline, filename, phase.
- `<lens_config_slug>/calibration.json` — per-aperture (FEC, ISO) tuples
  determined during the calibration phase.
- `<lens_config_slug>/f<aperture>/step_NNN.jpg` — the JPEGs.

## Configuration

All tunable constants are at the top of `config.py`. Notable ones:

- `MAX_FSTOP` — the largest f-number to test (session-wide).
- `RANGE_EXTENSION` — per-aperture ISO bumps and recycle delays.
- `OUTPUT_ROOT` — where session directories are created.
- `NTFY_TOPIC` — the ntfy.sh topic string. Set this once per laptop.
- `JOG_FINE_MM` / `JOG_COARSE_MM` — rail jog increments.
- `CROP_*_FRAC` — central-crop position and size on the wafer.

## Troubleshooting

| Symptom | Likely cause | Fix |
|--|--|--|
| "Config tree too small" at startup | Camera came up in minimal-MTP mode | Battery-pull recovery (camera prep §1–8). |
| `PTP General Error` / `I/O problem` on capture (camera fires but file transfer fails) | USB 2.0 cable on the Z7; can't sustain 12 MB bulk transfer | Use a USB 3.0+ cable. Confirm with `lsusb -t` shows the Z7 at `5000M`, not `480M`. |
| "Camera not in M mode" | Mode dial wrong, or U2 bank reset | Set dial to M. |
| FEC sanity check fails | SU-800 mounted instead of SB-5000, or lens cap on | Check commander; confirm flashes fire. |
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
