#!/usr/bin/env python3
"""
Z7 + gphoto2 sweep test:
  1. Reference shot with flash disabled (ambient only)
  2. Reference shot with flash enabled, EC=0, FEC=0
  3. FEC sweep
  4. EC sweep

Run after rig prep (battery pull, M mode, USB after power-on).

  python3 z7_ec_test.py
  python3 z7_ec_test.py --crop-y 0.20 --skip-init
  python3 z7_ec_test.py --no-flash-ref
"""

import argparse
import os
import subprocess
import sys
import time

try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed."); sys.exit(1)

try:
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError:
    print("Pillow and numpy required."); sys.exit(1)


# ---- helpers ---------------------------------------------------

def find(root, name):
    if root.get_name() == name:
        return root
    for i in range(root.count_children()):
        r = find(root.get_child(i), name)
        if r is not None:
            return r
    return None


def count_widgets(root):
    n = 1
    for i in range(root.count_children()):
        n += count_widgets(root.get_child(i))
    return n


def widget_choices(w):
    return [w.get_choice(i) for i in range(w.count_choices())]


INIT_SETTINGS = [
    ("capturetarget",             ["Internal RAM"], "SDRAM"),
    ("imagequality",              ["JPEG Fine"],    "JPEG Fine"),
    ("iso",                       ["100"],          "base ISO"),
    ("f-number",                  ["f/8", "f/5.6", "f/4"], "aperture"),
    ("shutterspeed",              ["1/200", "0.0050s", "0.005s"], "1/200s"),
    ("autofocus",                 ["Off"],          "manual focus"),
    ("flashexposurecompensation", [0.0],            "FEC = 0"),
    ("exposurecompensation",      ["0"],            "EC = 0"),
]


def set_setting(camera, name, candidates, description):
    cfg = camera.get_config()
    w = find(cfg, name)
    if w is None:
        print(f"  - {name:30s} NOT IN TREE")
        return False
    wtype = w.get_type()
    chosen = None
    if wtype in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        choices = widget_choices(w)
        for cand in candidates:
            if cand in choices:
                chosen = cand
                break
        if chosen is None:
            print(f"  ! {name:30s} none of {candidates!r} in {choices[:8]}...")
            return False
    else:
        chosen = candidates[0]
    try:
        w.set_value(chosen)
        camera.set_config(cfg)
    except gp.GPhoto2Error as e:
        print(f"  ! {name:30s} set failed: {e}")
        return False
    cfg2 = camera.get_config()
    actual = find(cfg2, name).get_value()
    ok = "OK " if str(actual) == str(chosen) else " ? "
    print(f"  [{ok}] {name:30s} = {actual!r}")
    return True


def crop_box(img_w, img_h, cw_f, ch_f, x_f, y_f):
    cw = int(img_w * cw_f); ch = int(img_h * ch_f)
    cx = img_w // 2 + int(img_w * x_f)
    cy = img_h // 2 + int(img_h * y_f)
    left = max(0, cx - cw // 2); top = max(0, cy - ch // 2)
    right = min(img_w, left + cw); bottom = min(img_h, top + ch)
    return left, top, right, bottom


def measure_crop(jpg_path, box):
    img = Image.open(jpg_path).convert("L")
    arr = np.array(img)
    l, t, r, b = box
    crop = arr[t:b, l:r]
    p10, p50, p90 = np.percentile(crop, [10, 50, 90])
    return {
        "p10": float(p10), "p50": float(p50), "p90": float(p90),
        "mean": float(np.mean(crop)),
        "clip_hi_pct": float(np.mean(crop >= 254)) * 100,
        "clip_lo_pct": float(np.mean(crop <= 1)) * 100,
        "shape": crop.shape,
    }


def save_overlay(jpg_path, box, overlay_path, max_w=1200):
    img = Image.open(jpg_path).convert("RGB")
    iw, ih = img.size
    scale = min(1.0, max_w / iw)
    if scale < 1.0:
        img = img.resize((int(iw * scale), int(ih * scale)))
    draw = ImageDraw.Draw(img)
    l, t, r, b = box
    sb = (int(l * scale), int(t * scale), int(r * scale), int(b * scale))
    for w in range(3):
        draw.rectangle((sb[0]-w, sb[1]-w, sb[2]+w, sb[3]+w), outline=(255, 0, 0))
    img.save(overlay_path, quality=85)


def linearise(v_0_255, gamma=2.2):
    return (max(v_0_255, 1.0) / 255.0) ** gamma


def read_exif(jpg_path):
    try:
        out = subprocess.run(
            ["exiftool", "-s", "-s", "-s",
             "-FlashExposureComp", "-ExposureCompensation",
             "-FNumber", "-ISO", "-ExposureTime", "-Flash",
             jpg_path],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return {}
        keys = ["FlashExposureComp", "ExposureCompensation",
                "FNumber", "ISO", "ExposureTime", "Flash"]
        vals = out.stdout.strip().split("\n")
        # exiftool only prints lines for tags it found
        # so we can't zip directly; parse the verbose form instead
        out2 = subprocess.run(
            ["exiftool", "-FlashExposureComp", "-ExposureCompensation",
             "-FNumber", "-ISO", "-ExposureTime", "-Flash",
             jpg_path],
            capture_output=True, text=True, timeout=10,
        )
        d = {}
        for line in out2.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip().replace(" ", "")] = v.strip()
        return d
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def snapshot_state(camera, label=""):
    """Print a concise snapshot of camera state for debugging."""
    try:
        cfg = camera.get_config()
    except gp.GPhoto2Error as e:
        print(f"    [snapshot{' '+label if label else ''}] get_config FAILED: {e}")
        return
    interesting = [
        "expprogram", "shutterspeed", "f-number", "iso",
        "flashexposurecompensation", "exposurecompensation",
        "flashmode", "capturetarget", "imagequality", "autofocus",
        "batterylevel",
    ]
    parts = []
    for name in interesting:
        w = find(cfg, name)
        if w is None:
            continue
        try:
            v = w.get_value()
        except gp.GPhoto2Error as e:
            v = f"<err: {e}>"
        parts.append(f"{name}={v!r}")
    print(f"    [state{' '+label if label else ''}] " + "  ".join(parts))


def drain_events(camera, timeout_ms=500):
    """Drain any pending events from the camera so subsequent capture
    returns a fresh frame, not a stale one from the SDRAM buffer."""
    drained = 0
    while True:
        try:
            evt_type, evt_data = camera.wait_for_event(timeout_ms)
        except gp.GPhoto2Error:
            break
        if evt_type == gp.GP_EVENT_TIMEOUT:
            break
        drained += 1
        if evt_type == gp.GP_EVENT_FILE_ADDED:
            try:
                camera.file_delete(evt_data.folder, evt_data.name)
            except gp.GPhoto2Error:
                pass
        if drained > 20:
            break
    return drained


def capture_to(camera, dest, max_retries=3):
    """Capture a fresh frame, draining stale buffered frames first.
    Retry on transient errors with diagnostic output."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            drained = drain_events(camera, timeout_ms=200)
            if drained:
                print(f"    drained {drained} stale events before capture")
            fp = camera.capture(gp.GP_CAPTURE_IMAGE)
            cf = gp.CameraFile()
            camera.file_get(fp.folder, fp.name, gp.GP_FILE_TYPE_NORMAL, cf)
            cf.save(dest)
            try:
                camera.file_delete(fp.folder, fp.name)
            except gp.GPhoto2Error:
                pass
            if attempt > 1:
                print(f"    capture succeeded on attempt {attempt}")
            return os.path.getsize(dest)
        except gp.GPhoto2Error as e:
            last_err = e
            err_code = getattr(e, "code", None)
            print(f"    ! capture attempt {attempt} failed: code={err_code} {e}")
            snapshot_state(camera, f"after-fail-attempt-{attempt}")

            # try to drain whatever's pending
            try:
                n = drain_events(camera, timeout_ms=1000)
                print(f"    recovery: drained {n} pending events")
            except Exception as e2:
                print(f"    recovery: drain raised {e2}")

            # progressive backoff
            backoff = 1.0 * attempt
            print(f"    recovery: waiting {backoff:.1f}s before retry")
            time.sleep(backoff)

            # on second retry, also nudge an unrelated config to force the
            # camera out of any stuck state it may be in
            if attempt == 2:
                try:
                    cfg = camera.get_config()
                    af = find(cfg, "autofocus")
                    if af is not None and af.get_type() in (gp.GP_WIDGET_RADIO,
                                                            gp.GP_WIDGET_MENU):
                        cur = af.get_value()
                        # toggle to other value and back
                        choices = widget_choices(af)
                        other = next((c for c in choices if c != cur), None)
                        if other:
                            af.set_value(other)
                            camera.set_config(cfg)
                            time.sleep(0.3)
                            af.set_value(cur)
                            camera.set_config(cfg)
                            print(f"    recovery: toggled autofocus {cur} -> {other} -> {cur}")
                except gp.GPhoto2Error as e2:
                    print(f"    recovery: autofocus toggle failed: {e2}")

    print(f"    ! capture failed after {max_retries} attempts; "
          f"last error: {last_err}")
    raise last_err


def set_value(camera, name, value):
    cfg = camera.get_config()
    w = find(cfg, name)
    if w is None:
        return None
    if w.get_type() in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        w.set_value(str(value))
    else:
        w.set_value(float(value))
    camera.set_config(cfg)
    # drain any DevicePropChanged events triggered by the set
    drain_events(camera, timeout_ms=300)
    cfg2 = camera.get_config()
    return find(cfg2, name).get_value()


def take_one(camera, label, dest, args, box_holder):
    """Capture, measure, save overlay. Returns (stats, exif, box)."""
    overlay = dest.replace(".jpg", "_overlay.jpg")

    # Sanity check: camera should still be in M mode and at the shutter we set.
    # The Z7 sometimes returns transient '?' values for properties immediately
    # after a burst of DevicePropChanged events. Retry a few times.
    ep_val, ss_val = "?", "?"
    for attempt in range(5):
        cfg = camera.get_config()
        ep = find(cfg, "expprogram")
        ss = find(cfg, "shutterspeed")
        ep_val = ep.get_value() if ep else "?"
        ss_val = ss.get_value() if ss else "?"
        if ep_val != "?" and ss_val != "?":
            break
        time.sleep(0.2)
    if ep_val == "?":
        print(f"    ! expprogram still '?' after retries; assuming M and proceeding")
    elif ep_val != "M":
        print(f"    ! camera left M mode (expprogram={ep_val!r}); aborting")
        snapshot_state(camera, "left-M")
        return None, None, box_holder[0]
    if ss_val not in ("0.0050s", "1/200", "?"):
        print(f"    ! shutterspeed changed to {ss_val!r}; resetting to 1/200")
        try:
            ss.set_value("0.0050s")
            camera.set_config(cfg)
        except gp.GPhoto2Error as e:
            print(f"    ! could not restore shutterspeed: {e}")

    try:
        size = capture_to(camera, dest)
    except gp.GPhoto2Error as e:
        # capture_to already logged details and tried recovery
        print(f"    [{label}] giving up on this shot")
        return None, None, box_holder[0]

    if box_holder[0] is None:
        with Image.open(dest) as im:
            iw, ih = im.size
        box_holder[0] = crop_box(iw, ih, args.crop_w, args.crop_h,
                                 args.crop_x, args.crop_y)
        print(f"    image {iw}x{ih}, crop box {box_holder[0]}")

    stats = measure_crop(dest, box_holder[0])
    save_overlay(dest, box_holder[0], overlay)
    exif = read_exif(dest)
    print(f"    [{label}] p10={stats['p10']:6.1f}  p50={stats['p50']:6.1f}  "
          f"p90={stats['p90']:6.1f}  mean={stats['mean']:6.1f}  "
          f"clip-hi={stats['clip_hi_pct']:.2f}%")
    if exif:
        print(f"    EXIF: Flash={exif.get('Flash','?')}  "
              f"FEC={exif.get('FlashExposureComp','?')}  "
              f"EC={exif.get('ExposureCompensation','?')}  "
              f"f={exif.get('FNumber','?')}  ISO={exif.get('ISO','?')}  "
              f"shutter={exif.get('ExposureTime','?')}")
    print(f"    saved {dest}, overlay {overlay}")
    return stats, exif, box_holder[0]


# ---- sweep -----------------------------------------------------

def sweep_property(camera, prop_name, values, args, box_holder, results_out):
    print()
    print("=" * 78)
    print(f"sweep: {prop_name}")
    print("=" * 78)

    cfg = camera.get_config()
    w = find(cfg, prop_name)
    if w is None:
        print(f"  ! {prop_name} not in tree; skipping")
        return
    print(f"  type={w.get_type()} ro={w.get_readonly()}")
    if w.get_type() in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        choices = widget_choices(w)
        print(f"  choices: {choices}")
        # filter values to existing choices
        kept = []
        for v in values:
            sv = str(v)
            if sv in choices:
                kept.append(sv)
            else:
                # look for closest numeric match
                try:
                    target = float(sv)
                    nums = []
                    for c in choices:
                        try: nums.append((c, float(c)))
                        except ValueError: pass
                    if nums:
                        best = min(nums, key=lambda x: abs(x[1] - target))
                        print(f"  {sv!r} not in choices, using {best[0]!r}")
                        kept.append(best[0])
                except ValueError:
                    pass
        values = kept
    print()

    for v in values:
        actual = set_value(camera, prop_name, v)
        print(f"  {prop_name} = {v}  (read back: {actual!r})")
        time.sleep(args.settle)
        tag = str(v).replace("+", "p").replace("-", "m").replace(".", "_")
        dest = f"/tmp/{prop_name}_{tag}.jpg"
        stats, exif, _ = take_one(camera, prop_name, dest, args, box_holder)
        if stats:
            results_out.append((prop_name, v, actual, stats, exif, dest))
        print()

    # reset to 0
    if w.get_type() in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        set_value(camera, prop_name, "0")
    else:
        set_value(camera, prop_name, 0.0)
    print(f"  reset {prop_name} to 0")


def report(results, ref_label):
    if not results:
        print("  no captures")
        return
    # find the row matching ref_label
    ref = None
    for r in results:
        if r[0] == "ref" and r[1] == ref_label:
            ref = r
            break
    if ref is None:
        print(f"  no reference row matching {ref_label!r}")
        return
    ref_lin = linearise(ref[3]["p50"])
    print(f"  reference: {ref_label}  p50={ref[3]['p50']:.1f}  linear={ref_lin:.4f}")
    print()
    print(f"  {'group':>20}  {'value':>6}  {'p10':>6}  {'p50':>6}  {'p90':>6}  "
          f"{'ΔEV(p50)':>9}  {'expected':>9}  flag")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  ----")
    for r in results:
        group, v, actual, stats, exif, _ = r
        lin = linearise(stats["p50"])
        dev = float(np.log2(lin / ref_lin)) if ref_lin > 0 else float("nan")
        try:
            expected = float(v) - (float(ref[1]) if isinstance(ref[1], (int, float, str))
                                   and str(ref[1]).replace("-","").replace(".","").isdigit()
                                   else 0.0)
        except (ValueError, TypeError):
            expected = float("nan")
        if not np.isnan(expected):
            diff = dev - expected
            if abs(diff) < 0.5: flag = "OK"
            elif abs(diff) < 1.0: flag = "?"
            else: flag = "FAIL"
        else:
            flag = "ref"
        clip = ""
        if stats["clip_hi_pct"] > 5: clip = " HI-CLIP"
        elif stats["clip_lo_pct"] > 5: clip = " LO-CLIP"
        print(f"  {group:>20}  {str(v):>6}  {stats['p10']:6.1f}  {stats['p50']:6.1f}  "
              f"{stats['p90']:6.1f}  {dev:+9.2f}  "
              f"{expected if not np.isnan(expected) else float('nan'):+9.2f}  "
              f"{flag}{clip}")


# ---- main ------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-w", type=float, default=0.20)
    p.add_argument("--crop-h", type=float, default=0.20)
    p.add_argument("--crop-x", type=float, default=0.00)
    p.add_argument("--crop-y", type=float, default=0.15)
    p.add_argument("--fec-values", nargs="+",
                   default=["-3", "-2", "-1", "0", "1"])
    p.add_argument("--ec-values", nargs="+",
                   default=["-3", "-2", "-1", "0", "1"])
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--skip-init", action="store_true")
    p.add_argument("--no-flash-ref", action="store_true",
                   help="skip the flash-disabled reference shot")
    args = p.parse_args()

    print("=== opening camera ===")
    camera = gp.Camera()
    try:
        camera.init()
    except gp.GPhoto2Error as e:
        print(f"  init failed: {e}"); sys.exit(1)

    box_holder = [None]
    results = []

    try:
        cfg = camera.get_config()
        n = count_widgets(cfg)
        print(f"  config tree has {n} widgets")
        if n < 100:
            print("  ! tree too small; battery-pull recovery needed"); sys.exit(2)
        ep = find(cfg, "expprogram")
        if ep is None or ep.get_value() != "M":
            print(f"  ! expprogram = {ep.get_value() if ep else None}"); sys.exit(2)
        print(f"  expprogram = M (good)")

        if not args.skip_init:
            print("\n=== applying init settings ===")
            for name, cands, desc in INIT_SETTINGS:
                set_setting(camera, name, cands, desc)

        # Reference 1: flash disabled (manual — Z7 won't allow >1/200 with
        # a CLS commander on the hot shoe, and disabling flashmode via
        # gphoto2 doesn't actually stop the SU-800 from triggering remotes)
        if not args.no_flash_ref:
            print("\n=== reference: flash disabled ===")
            print("  Please TURN OFF the SU-800 (and any other flashes), then")
            input("  press ENTER to take the no-flash reference shot... ")
            # aggressive drain — anything could have happened during the pause
            n = drain_events(camera, timeout_ms=500)
            if n:
                print(f"  drained {n} pending events after prompt")
            time.sleep(args.settle)
            stats, exif, _ = take_one(camera, "no-flash",
                                      "/tmp/ref_no_flash.jpg", args, box_holder)
            if stats:
                results.append(("ref", "no-flash", None, stats, exif,
                                "/tmp/ref_no_flash.jpg"))
            print()
            print("  Now TURN THE SU-800 (and flashes) BACK ON, wait for them")
            print("  to finish recycling (steady ready light), then")
            input("  press ENTER to continue with the flash-on tests... ")
            n = drain_events(camera, timeout_ms=500)
            if n:
                print(f"  drained {n} pending events after prompt")
            time.sleep(args.settle)

        # Reference 2: flash on, EC=0, FEC=0
        print("\n=== reference: flash on, EC=0, FEC=0 ===")
        set_value(camera, "exposurecompensation", "0")
        set_value(camera, "flashexposurecompensation", 0.0)
        time.sleep(args.settle)
        stats, exif, _ = take_one(camera, "flash-on-ref",
                                  "/tmp/ref_flash_on.jpg", args, box_holder)
        if stats:
            results.append(("ref", "0", "0", stats, exif, "/tmp/ref_flash_on.jpg"))

        # FEC sweep
        sweep_property(camera, "flashexposurecompensation",
                       [float(v) for v in args.fec_values], args, box_holder, results)

        # back to FEC 0 before EC sweep
        set_value(camera, "flashexposurecompensation", 0.0)

        # EC sweep
        sweep_property(camera, "exposurecompensation",
                       args.ec_values, args, box_holder, results)

        # Final report
        print()
        print("=" * 78)
        print("FINAL REPORT (reference = flash on, EC=0, FEC=0)")
        print("=" * 78)
        report(results, "0")

    finally:
        camera.exit()
        print("\n=== closed ===")


if __name__ == "__main__":
    main()
