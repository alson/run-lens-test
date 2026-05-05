# Hardware integration & end-to-end testing

Pick up here after rebooting for Windows. Each section is a single
test or troubleshooting probe; do them in order. Each step lists the
command, what success looks like, and what to try if it fails.

The goal is a clean run of the full session pipeline against the real
rig, but step 1 (StackShot connectivity) is the current blocker —
work through that first.

---

## 1. StackShot — protocol troubleshooting

### What we know so far (Linux)

- USB enumerates: `lsusb` shows `0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC`.
- udev creates the symlink: `/dev/serial/by-id/usb-Cognisys_StackShot_A700C08T-if00-port0 -> /dev/ttyUSB0`.
- Front panel shows `PC` in the top-right corner (PC-control mode active).
- Sending any frame (GetSoftwareID / GetStatus / GetPosition) at any baud rate (9600 / 19200 / 38400 / 57600 / 115200) yields **0 bytes** back.
- DTR / RTS toggle has no effect.
- No unsolicited bytes on the wire when we just listen.
- No CTS / DSR / RI / CD signals from the device.
- Serial BREAK (250 ms) before the frame: no effect.
- Replacing the cable with a plain USB-A → mini-USB (no adapters): no effect.
- `ModemManager` confirmed inactive.
- `lpvo_usb_gpib` (an FTDI-based GPIB driver in the Linux staging tree)
  was being autoloaded for any FT232R via the alias
  `usb:v0403p6001…` in `/lib/modules/$(uname -r)/modules.alias`. The
  staging warning made it suspect, so we blacklisted it:
  ```
  sudo sh -c 'echo "blacklist lpvo_usb_gpib" > /etc/modprobe.d/blacklist-lpvo-gpib.conf'
  sudo rmmod lpvo_usb_gpib gpib_common
  ```
  Confirmed gone (`lsmod | grep gpib` returns nothing) but the
  StackShot still does not respond. Driver was a red herring on
  this rig.
- **`pyftdi` / libusb-direct also returns 0 bytes** (`tools/pyftdi_probe.py`,
  run via `sudo`). Bypasses ftdi_sio entirely; bytes reach the wire
  (`write_data` confirms 6 bytes written) but the device emits nothing.

### Step 1.0a — The actual fix: SET_BITMODE 0x20FF

After all that, the answer was in the Cognisys reference C++ source
(`StackshotUSB_app.zip`, `Form1.h:2335-2342`):

```c++
ftStatus = FT_SetBitMode(*handle, 0xFF, 0x20);
// Sleep required here for slave device to debounce pins and enter USB control
Thread::Sleep(500);
```

Translated to a USB control transfer: vendor request `0x0B` with
``wValue = 0x20FF`` (mode = 0x20 = CBUS bit bang, mask = 0xFF = all
pins as outputs). Then a 500 ms sleep so the StackShot's MCU sees
the CBUS state change and enables its serial command listener.

**This is undocumented in the protocol PDF.** Without sending it,
the StackShot's MCU stays in standalone / front-panel mode and
silently ignores every byte of the documented protocol. With it,
the device responds normally on the very next bulk-OUT.

`tools/pyftdi_probe.py` now mirrors the recipe from
`StackshotUSB_app.zip` exactly: SET_BITMODE → 500 ms sleep → reset
cycles → BAUDRATE/LINE/FLOW. Validated end-to-end on this rig: a
`CC_SOFTWARE_STRING` (cmd 0x1102) read returns `'SW Rev:1.0.09'`.

### Step 1.0 — Factory reset on the StackShot

Per manual page 45. Likely fix if the firmware reflash last year left
PC-mode in an unusable state.

1. Power off the StackShot.
2. Hold **DOWN** button.
3. Apply power.
4. When the splash screen appears, release DOWN.

This loads factory defaults but does **not** erase saved settings.
After the splash screen clears, plug USB back in (front panel should
still show `PC` in the corner) and re-run:

```bash
.venv/bin/python -m stackshot.controller --info --port /dev/ttyUSB0
```

### Step 1.1 — Confirm the device works at all (Windows)

**Important — get the right ZIP.** There are two separate downloads:

* `Reflash.zip` — bootloader-mode reflasher only. Useless for sniffing
  protocol traffic; needs `config + power-on` to enter bootloader. This
  is what you'll find first if you Google "StackShot Windows utility".
* `StackshotUSB_app.zip` — the actual C++ protocol example referenced
  in the protocol PDF §1. Issues real `Get Software ID` / `Move`
  commands over the documented wire format. **This** is what you want:
  ```
  http://www.cognisys-inc.com/downloads/stackshot/StackshotUSB_app.zip
  ```

Plug the StackShot in (front panel still showing `PC`). Run the app.

**Success criteria**: app shows the StackShot, Get Software ID returns a non-zero number, a 1 mm Move succeeds and the rail visibly moves.

If the app **fails to talk to the rig**, the issue is hardware — try:
- A different USB cable
- A different USB port (rear vs front of the laptop, USB 2 vs USB 3)
- Cycle the StackShot power
- Is the StackShot's firmware revision the one the protocol PDF documents (1.2)? Front panel → "About" or similar should report it.

If the app **does talk to the rig**, the device is fine and the issue is on the Linux side. Continue to step 1.2.

### Step 1.2 — Capture a working frame on Windows

Two options, ordered by ease.

#### Option A: USB sniff with Wireshark + USBPcap (preferred)

Wireshark on Windows can capture USB traffic via USBPcap.

1. Install Wireshark with the USBPcap component during setup.
2. Run Wireshark, pick the USBPcap interface that matches the FTDI bridge.
3. Send a Get Software ID command from the StackShot reference app.
4. In Wireshark, find the URB_BULK_OUT (host → device) and the URB_BULK_IN (device → host) for that exchange.
5. **Capture the exact bytes** of each. Save the .pcapng so you can compare offline.

Compare against what our code sends. Our protocol expects:

| Direction | Bytes (hex) | Meaning |
|--|--|--|
| Host → Slave (Get Software ID) | `55 11 03 00 00 69` | Sync 0x55, cmd 0x1103, action READ, len 0, checksum |
| Slave → Host | `AA 11 03 02 04 LSB MSB MSB MSB CK` | Sync 0xAA, cmd echo, RSP_OK=0x02, len 4, U32 LSB-first, checksum |

Things to verify on the captured working frame:
- Is the host command byte-for-byte identical to ours?
- Does the slave response start with `0xAA`, or some other byte?
- Are there extra bytes (e.g. a leading `0x00` or `0xFF`) before the response?
- Does the cmd ID echo back as `11 03` (LSB first) or `03 11` (raw)?
- Length field — is it 4 (just the U32) or different?

Note any difference. The most likely surprises:
- Cmd ID byte order in the response is opposite to what the PDF implies.
- A leading start-of-frame escape byte (some FTDI-based protocols use 0x7E etc.).
- Different baud rate than 38400 (FT232 supports non-standard rates; the Cognisys app might be programming the FTDI to a custom rate).

#### Option B: app log / verbose mode

If the Cognisys app exposes a debug or verbose-logging mode in its menus, enable it and capture the bytes from there. Less reliable than USBPcap but quicker if available.

### Step 1.3 — Reproduce on Linux

Back on Linux, with the bytes captured in step 1.2:

```bash
# Replace <hex bytes> with what Wireshark showed.
.venv/bin/python -c "
import serial, time
s = serial.Serial('/dev/ttyUSB0', 38400, timeout=2.0)
s.dtr = True; s.rts = True
time.sleep(2.0)
s.reset_input_buffer()
frame = bytes.fromhex('<hex bytes>')   # <-- the working host frame from Windows
print('sending:', frame.hex())
s.write(frame); s.flush()
time.sleep(0.5)
data = s.read(64)
print('received:', data.hex(), '(', len(data), 'bytes)')
s.close()
"
```

If Linux now gets a response from the same byte sequence: the device is happy, our framing is correct, but our code's first frame on the open port is being lost or mangled. Common causes:
- The FT232 needs a settling delay after `Serial()` open before writes are accepted. We currently sleep 0.002 s pre-frame; bump to 200 ms in `stackshot/transport.py:SerialTransport.open` and retest.
- DTR pulse on open resets some FTDI devices; try opening with `dsrdtr=False, rtscts=False` and explicit `dtr = True; rts = True` *before* the first write.
- udev rule auto-runs ModemManager on tty creation; ModemManager probes serial devices with random AT-style strings that confuse the StackShot. Verify with `systemctl status ModemManager` and either stop it or add an udev exclusion:
  ```
  # /etc/udev/rules.d/77-mm-cognisys-blacklist.rules
  ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ENV{ID_MM_DEVICE_IGNORE}="1"
  ```
  Reload: `sudo udevadm control --reload && sudo udevadm trigger`.

If Linux still gets 0 bytes back even with the same byte sequence that worked on Windows: the issue is at the FT232 / USB layer (driver, latency timer, line-discipline). Check:
- `cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` — default 16 ms, sometimes 1 ms is needed for short responses. Try `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`.
- `dmesg -w` while sending — anything from `ftdi_sio` showing up?

### Step 1.4 — Once the rig responds

Run our smoke-test:

```bash
.venv/bin/python -m stackshot.controller --info --port /dev/ttyUSB0
```

Expected: `software_id = 0xNN…`, `hardware_id = 0xNN…`, `position = +0.0000 mm`.

Then a small move:

```bash
.venv/bin/python -m stackshot.controller --move 1.0 --port /dev/ttyUSB0
```

Expected: rail moves visibly by ~1 mm; stdout reports the new position.

Then a backward move:

```bash
.venv/bin/python -m stackshot.controller --move 1.0 --back --port /dev/ttyUSB0
```

Expected: rail moves back; new position close to (but not exactly) starting position because of backlash compensation.

If steps 1.4 pass, **mark task #3 (StackShot controller, rig test) complete**.

---

## 2. StackShot jog UI

After step 1 passes:

```bash
.venv/bin/python -m stackshot.jog --port /dev/ttyUSB0
```

You should see:

```
Jog rail to START. a/d=fine, A/D=coarse, s=set START, q=abort
[START] position =  +0.000 mm
```

Test sequence:
1. Press `d` a few times — rail steps forward in 0.05 mm increments; line redraws each time.
2. Press `A` once — rail steps back 1 mm.
3. Press `s` — `[START set (zeroed)]` shows; prompt moves to END.
4. Press `D` until you reach a few mm — line updates.
5. Press `e` — `[END set]`; final printout shows START=0.000 mm and END=<your value>.

If keys aren't registering, check that the terminal is in a real TTY (not piped through anything that buffers stdin — e.g. Claude Code's web shell may not deliver keystrokes byte-by-byte; run from a normal terminal).

---

## 3. Camera (Nikon Z7)

Prerequisites:
- Camera prep per `README.md` (battery pull ≥10 s, mode dial = M, USB connected after power-on).
- SB-5000 mounted on hot shoe in commander mode, master output `--`, group A in A.
- Remote flashes (SB-800 + SB-5000) on, ready light steady.

### Step 3.1 — Health check

```bash
.venv/bin/python -m camera.nikon_z7 --health
```

Expected:
- `config tree widgets:` ≥ 100 (a healthy Z7 reports ~417).
- `expprogram = 'M'`.
- `iso`, `f-number`, `shutterspeed`, `flashexposurecompensation` all show plausible values.

If the widget count is ~23, the camera came up in minimal-MTP mode. Power off, remove battery 10+ s, reinsert, power on, wait for live view, then reconnect USB. Re-run.

### Step 3.2 — Single capture

```bash
.venv/bin/python -m camera.nikon_z7 --capture /tmp/test.jpg
```

Expected: shutter fires (you hear the flash), JPEG written to `/tmp/test.jpg` (open it; should show the wafer scene). Stdout reports a non-zero byte count.

If capture fails: re-run with the diagnostic from `tools/`:

```bash
.venv/bin/python tools/z7_diagnostic.py 2>&1 | tee /tmp/diag.log
```

The diagnostic will dump the full config tree to `/tmp/z7_full_config.txt` and run two test captures with FEC = 0 and FEC = -2. Compare medians: they should differ noticeably (FEC -2 ≈ ¼ the brightness of FEC 0 in linear-light terms).

### Step 3.3 — FEC sanity check via the production wrapper

The session runner runs an FEC sanity check at startup. To exercise it standalone:

```bash
.venv/bin/python -c "
from camera.nikon_z7 import NikonZ7
from exposure.analyse import CropBox, crop_box, measure_crop
from PIL import Image

cam = NikonZ7()
cam.open()
try:
    cam.verify_healthy()
    cam.apply_session_config()
    # measure_crop adapter that the wrapper expects
    box = None
    def measure(p):
        global box
        if box is None:
            with Image.open(p) as im:
                box = crop_box(*im.size, 0.20, 0.20, 0.0, 0.15)
        return measure_crop(p, box).p50
    base, low = cam.fec_sanity_check(measure=measure, capture_dir='/tmp')
    print(f'FEC=0 median {base:.1f}, FEC=-2 median {low:.1f}, ratio {low/base:.2f}')
finally:
    cam.close()
"
```

Expected: ratio < 0.5 (the FEC=-2 shot is at most half the brightness of FEC=0). If ratio > 0.5, you've likely got the SU-800 mounted by mistake — it doesn't propagate FEC on the Z7. Swap to the SB-5000 commander.

---

## 4. Notifications

### Step 4.1 — Set the topic

In `config.py`, change `NTFY_TOPIC` from the default placeholder to a private random string. On the Mac (or phone), open the ntfy app and subscribe to the same topic.

### Step 4.2 — Send a test message

```bash
.venv/bin/python -m notify.ntfy --test
```

Expected: `delivered`. The Mac/phone should buzz with a notification within a couple of seconds.

If `failed`: check network reachability (`curl -fSs https://ntfy.sh`) and that the topic name doesn't include any forbidden characters.

### Step 4.3 — Local sound

```bash
.venv/bin/python -c "
from notify.sound import play_alert
print(play_alert('data/alert.wav'))
"
```

Drop a small `.wav` at `data/alert.wav` first (any system .wav works for testing). Expected: speakers chirp; stdout shows `True`.

---

## 5. End-to-end smoke test (one lens config)

After steps 1-4 all pass.

### Setup

- Camera prep (battery pull, M mode, etc.).
- StackShot prep: powered on, in PC mode, USB connected.
- `OUTPUT_ROOT` in `config.py` points to writable storage with a few GB free.
- Wafer target visible in frame (lower-centre of frame per the default crop region).

### Run

```bash
.venv/bin/python -m session.runner
```

The runner will:
1. Open hardware, run `verify_healthy`, apply session config, run FEC sanity check.
2. Prompt: `Lens config (free text, blank to quit):` — enter something descriptive, e.g. `Sigma 105 macro test 1`.
3. Drop into the START jog. Use `a/d` (fine, 0.05 mm) and `A/D` (coarse, 1 mm) to position the rail at the start of focus, then press `s`.
4. END jog. Position the rail at the deepest focus point. Press `e`.
5. Prompt: `Number of rail steps for this lens config (>=2):` — enter `5` for a deliberately quick test.
6. Confirmation prompt; ENTER to begin.
7. Calibration phase: one test shot per aperture at rail position 0. Watch the terminal — it should converge each aperture's FEC within ≤ 3 shots.
8. Sweep phase: 5 rail steps × N apertures. You'll hear the rail step and the flashes fire per shot. Cropping happens in the background per shot.
9. Range check at end: with the real `SharpnessRangeChecker` it'll either `ok` (write `range_check.json` and move on) or trigger the probe + extension flow. With a 5-step deliberately-narrow range it'll likely trigger one side.
10. Lens change prompt: ENTER to continue (or `q` to abort gracefully).
11. Empty lens-config name → graceful shutdown.

### Verify

In `<OUTPUT_ROOT>/<session_timestamp>/`:

```
session.json                            ← session metadata
shots.csv                               ← one row per shot
<lens_config_slug>/
  calibration.json                       ← per-aperture (FEC, ISO)
  range_check.json                       ← decision + extension info
  calibration/cal_f_*.jpg                ← test shots from FEC tuning
  f_4/
    step_0100.jpg                        ← main sweep at offset 100
    step_0101.jpg
    …
    step_0099.jpg                        ← only present if NEAR extension fired
    step_0103.jpg                        ← only present if FAR probe / extension fired
    cropped/
      center/
        step_0100.jpg                    ← 17 crops per main / probe / extension shot
        step_0101.jpg
        …
      left-edge/
      …
      dx-bottom-right/
  f_8/
  …
```

Open one of the JPEGs to confirm the wafer is in frame. Open one of the `cropped/center/step_0100.jpg` to confirm the cropper produced sensible output.

### Spot-check the sharpness pipeline ran

```bash
# Count crops produced for one aperture
ls <session>/<lens>/f_4/cropped/center/ | wc -l       # should == steps + extensions
ls <session>/<lens>/f_4/cropped/ | wc -l              # should == 17
```

If `cropped/` is empty: cropping failed. Look at `session.json` for any `notes`; rerun with `PYTHONLOGGING=DEBUG` or check stderr for `sharpness collector shutdown failed` messages.

### Spot-check `shots.csv`

```bash
column -t -s, <session>/shots.csv | head -20
```

The `phase` column should show `calibration` first, then `sweep` rows for each step × aperture, optionally `probe` and `extension` rows.

---

## 6. Battery droop simulation

After step 5 passes once cleanly.

### Run

Start a fresh session as in step 5. During the sweep phase, **physically pull a battery** from one of the off-camera flashes mid-sweep (around step 2 or 3 of a 5-step sweep).

### Expected

- Within a few captures, terminal shows `Battery droop on f/X` with a window snapshot.
- Mac/phone receives a `Battery droop detected` ntfy notification (priority high).
- Local speaker chirps (if `ALERT_WAV` is configured).
- Prompt: `Swap flash batteries, wait for steady ready light, then press ENTER to resume. q = abort`.
- Reinsert the battery. Press ENTER.
- Sweep continues from the **current** rail step (the rail did not move during the pause).
- The rolling-window detector resets so the just-low samples don't immediately re-trip.

### If it doesn't trigger

The droop criteria are `3 of last 10 ≤ -0.7 EV` or `2 of last 10 ≤ -2.0 EV`. If the flash recovers fast enough that the EV drop never reaches threshold, the detector won't fire — that's correct behaviour. Try pulling AND keeping the battery out for 3+ shots.

---

## 7. Lens-change recovery

After step 5 passes.

### Run

Continue the session past step 5's lens change prompt.

### Procedure when prompted

1. Power off camera with the power switch.
2. Change lens.
3. **Remove battery for at least 10 seconds.** This is non-negotiable; the Z7 USB state machine doesn't reset otherwise.
4. Reinsert battery.
5. Power on. Wait for live view to appear on the rear screen.
6. Press ENTER in the terminal.

### Expected

- Runner reopens the camera, runs `verify_healthy` (config tree ≥ 100 widgets, `expprogram == M`).
- Reapplies session config.
- Prompts for the next lens-config name.

### If reconnect fails

The runner retries up to 3 times. On the 3rd failure: high-priority ntfy alert "Camera reconnect failed", and the session aborts with `aborted=True` in `session.json`. If this happens unexpectedly, redo the battery-pull procedure and start a new session.

---

## 8. Range-check + auto-extension on rig

After step 5 passes once.

### Setup

For the real `SharpnessRangeChecker` to produce a `NEEDS_RESHOOT` verdict, the captured rail range must be **deliberately too narrow** for the lens. Aim to cover only one side of the lens's focus bell.

Concretely: jog the START to a position where the wafer is visibly in focus, then jog END to a position only ~half-way deeper than where the lens's natural depth-of-field ends. With 5 steps over 3 mm at high magnification, the bell will likely be truncated.

### Run

Start a session per step 5; pick the deliberately-narrow START/END.

### Expected

After the main sweep:
- Runner emits `Range still inadequate` notes only if the probe also fails. Otherwise:
- Runner shoots 1-2 grid-aligned probe shots in the extension (filenames `step_0103.jpg` etc. for far, `step_0099.jpg` for near).
- If probes bracket the peak: extension sweep continues, additional `step_NNNN.jpg` files appear.
- `<lens>/range_check.json` records `{"peak_side": "far"|"near"|"both", "extension_applied": true|false, "extension_step_count": {"far": N, "near": M}}`.

### Verify the cropping kept up

```bash
# Total source JPEGs in the per-aperture dir
n=$(ls <session>/<lens>/f_4/step_*.jpg | wc -l)
# Cropped JPEGs per crop name (should equal n)
ls <session>/<lens>/f_4/cropped/center/ | wc -l
```

The two should match. If `cropped/` has fewer files, the collector's queue didn't drain in time — `wait_for_pending()` should prevent this; if it doesn't, capture the issue with `dmesg` and the runner's stderr.

---

## 9. Graceful Ctrl-C

Standalone test, anytime after step 5 passes.

### Run

Start a session per step 5. Mid-sweep, press `Ctrl-C` in the terminal.

### Expected

- StackShot rail stops where it is (`CC_RAIL_STOP` issued).
- Camera closes cleanly.
- `session.json` finalised with `aborted: true` and `ended_at` set.
- Any in-flight crops drain via the collector's shutdown path.
- Process exits with a `KeyboardInterrupt` traceback (or non-zero status — both are fine).

If the rail keeps moving after Ctrl-C: the StackShot didn't acknowledge `CC_RAIL_STOP`. Check `dmesg` and the StackShot front panel.

---

## 10. Checklist before declaring "rig integration done"

- [ ] StackShot reference app talks to the rig on Windows (step 1.1)
- [ ] We've captured a working frame (step 1.2)
- [ ] Linux-side replay of the working frame succeeds (step 1.3)
- [ ] `python -m stackshot.controller --info` returns sensible IDs and position (step 1.4)
- [ ] `python -m stackshot.controller --move 1.0` moves the rail (step 1.4)
- [ ] `python -m stackshot.jog` lets you set START / END (step 2)
- [ ] `python -m camera.nikon_z7 --health` reports ≥100 widgets, M mode (step 3.1)
- [ ] `python -m camera.nikon_z7 --capture` writes a JPEG (step 3.2)
- [ ] FEC sanity check ratio < 0.5 (step 3.3)
- [ ] `python -m notify.ntfy --test` delivers (step 4)
- [ ] `python -m session.runner` runs one full lens config end-to-end (step 5)
- [ ] Battery-droop pause+resume works (step 6)
- [ ] Lens-change recovery works (step 7)
- [ ] Range-check + extension fires on a deliberately narrow range (step 8)
- [ ] Ctrl-C aborts cleanly (step 9)
