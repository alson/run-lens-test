#!/usr/bin/env python3
"""
Z7 + gphoto2: configure camera for macro lens testing, then verify
that flash exposure compensation actually controls flash output, by
sweeping FEC across its range and measuring the luminance of a small
crop on the wafer test target.

Run after the standard rig prep:
  - Power off, remove battery 10 seconds, reinsert
  - Mode dial = M
  - SU-800 + flashes mounted, AA mode, switched on
  - Power on, wait for live view to appear
  - THEN connect USB
  - Half-press shutter to wake

The crop is small and offset to land on the wafer. Defaults assume the
wafer is slightly below image centre. Tune with --crop-w / --crop-h /
--crop-y / --crop-x. Each capture saves a *_overlay.jpg thumbnail with
a red rectangle marking the sampled region — open one before trusting
the EV numbers.

Background ambient is fine as long as the crop is on the wafer; flash
typically dominates ambient at close macro distances.

Usage:
  python3 z7_settings_and_fec_test.py
  python3 z7_settings_and_fec_test.py --crop-w 0.15 --crop-y 0.20
  python3 z7_settings_and_fec_test.py --fec-values -3 -1 0 1
  python3 z7_settings_and_fec_test.py --skip-init   # don't reapply settings
"""

import argparse
import os
import sys
import time

try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed.")
    print("  sudo pacman -S python-gphoto2")
    print("  or: pip install --user --break-system-packages gphoto2")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError:
    print("Pillow and numpy required for FEC luminance analysis.")
    print("  pip install --user --break-system-packages pillow numpy")
    sys.exit(1)


# ---- helpers -----------------------------------------------------

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


# ---- settings ----------------------------------------------------

INIT_SETTINGS = [
    ("capturetarget",             ["Internal RAM"],
     "save captures to SDRAM, not card"),
    ("imagequality",              ["JPEG Fine"],
     "JPEG Fine, no RAW"),
    ("iso",                       ["100"],
     "base ISO"),
    ("f-number",                  ["f/8", "f/5.6", "f/4"],
     "starting aperture"),
    ("shutterspeed",              ["1/200", "0.0050s", "0.005s"],
     "1/200s flash sync"),
    ("autofocus",                 ["Off"],
     "manual focus"),
    ("flashexposurecompensation", [0.0],
     "FEC = 0 baseline"),
]

VR_CANDIDATES = ["vr", "vrenable", "vibrationreduction",
                 "vibrationreductiontype", "imagestabilization"]


def set_setting(camera, name, candidates, description):
    cfg = camera.get_config()
    w = find(cfg, name)
    if w is None:
        print(f"  - {name:30s} NOT IN TREE  [{description}]")
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
            short = choices[:6]
            more = "..." if len(choices) > 6 else ""
            print(f"  ! {name:30s} none of {candidates!r} in choices: {short}{more}")
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
    w2 = find(cfg2, name)
    actual = w2.get_value() if w2 is not None else "<gone>"
    ok = "OK " if str(actual) == str(chosen) else " ? "
    print(f"  [{ok}] {name:30s} = {actual!r}  [{description}]")
    return True


def try_disable_vr(camera):
    cfg = camera.get_config()
    for name in VR_CANDIDATES:
        w = find(cfg, name)
        if w is None:
            continue
        wtype = w.get_type()
        try:
            if wtype in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
                choices = widget_choices(w)
                for off_label in ("Off", "OFF", "off", "Disabled", "0"):
                    if off_label in choices:
                        w.set_value(off_label)
                        camera.set_config(cfg)
                        print(f"  [OK ] VR ({name}) = {off_label}")
                        return True
                print(f"  !    VR ({name}): no Off-like choice in {choices}")
                return False
            else:
                w.set_value(0)
                camera.set_config(cfg)
                print(f"  [OK ] VR ({name}) = 0")
                return True
        except gp.GPhoto2Error as e:
            print(f"  !    VR ({name}): set failed: {e}")
            continue
    print(f"  -    VR: none of {VR_CANDIDATES} present (skipped)")
    return False


def try_max_imagesize(camera):
    cfg = camera.get_config()
    w = find(cfg, "imagesize")
    if w is None:
        print(f"  -    imagesize NOT IN TREE (skipped)")
        return False
    choices = widget_choices(w)
    if not choices:
        return False
    target = choices[0]
    try:
        w.set_value(target)
        camera.set_config(cfg)
        print(f"  [OK ] imagesize = {target!r}  (choices: {choices})")
        return True
    except gp.GPhoto2Error as e:
        print(f"  !    imagesize: set failed: {e}")
        return False


# ---- crop math ---------------------------------------------------

def crop_box(img_w, img_h, crop_w_frac, crop_h_frac, x_off_frac, y_off_frac):
    """
    Compute (left, top, right, bottom) for a crop centred on
    (image centre + offset). x_off_frac and y_off_frac are fractions of
    image dimension; positive y_off_frac moves crop downward in pixels.
    """
    cw = int(img_w * crop_w_frac)
    ch = int(img_h * crop_h_frac)
    cx = img_w // 2 + int(img_w * x_off_frac)
    cy = img_h // 2 + int(img_h * y_off_frac)
    left = max(0, cx - cw // 2)
    top = max(0, cy - ch // 2)
    right = min(img_w, left + cw)
    bottom = min(img_h, top + ch)
    return left, top, right, bottom


def measure_crop(jpg_path, box):
    img = Image.open(jpg_path).convert("L")
    arr = np.array(img)
    left, top, right, bottom = box
    crop = arr[top:bottom, left:right]
    return float(np.median(crop)), float(np.mean(crop)), crop.shape


def save_overlay(jpg_path, box, overlay_path, max_w=1200):
    img = Image.open(jpg_path).convert("RGB")
    iw, ih = img.size
    scale = min(1.0, max_w / iw)
    if scale < 1.0:
        img = img.resize((int(iw * scale), int(ih * scale)))
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = box
    sb = (int(left * scale), int(top * scale),
          int(right * scale), int(bottom * scale))
    for w in range(3):
        draw.rectangle((sb[0] - w, sb[1] - w, sb[2] + w, sb[3] + w),
                       outline=(255, 0, 0))
    img.save(overlay_path, quality=85)


def linearise_srgb(v_0_255, gamma=2.2):
    v = max(v_0_255, 1.0)
    return (v / 255.0) ** gamma


def capture_to(camera, dest):
    fp = camera.capture(gp.GP_CAPTURE_IMAGE)
    cf = camera.file_get(fp.folder, fp.name, gp.GP_FILE_TYPE_NORMAL)
    cf.save(dest)
    return os.path.getsize(dest)


def set_fec(camera, value):
    cfg = camera.get_config()
    w = find(cfg, "flashexposurecompensation")
    if w is None:
        return None
    w.set_value(float(value))
    camera.set_config(cfg)
    cfg2 = camera.get_config()
    return find(cfg2, "flashexposurecompensation").get_value()


def fec_sweep_test(camera, fec_values, args):
    print()
    print("=" * 70)
    print("FEC sweep test")
    print("=" * 70)
    print(f"  crop {args.crop_w*100:.0f}% x {args.crop_h*100:.0f}%, "
          f"offset x={args.crop_x*100:+.0f}% y={args.crop_y*100:+.0f}% "
          f"(positive y = below centre)")
    print(f"  settle {args.settle:.1f}s between shots")
    print()

    results = []
    box = None
    for fec in fec_values:
        actual = set_fec(camera, fec)
        if actual is None:
            print(f"  ! FEC widget gone; aborting")
            return
        print(f"  FEC requested {fec:+.2f}  actual {actual:+.2f}  "
              f"(waiting {args.settle:.1f}s)")
        time.sleep(args.settle)

        tag = f"{fec:+.2f}".replace("+", "p").replace("-", "m")
        dest = f"/tmp/fec_test_{tag}.jpg"
        overlay = f"/tmp/fec_test_{tag}_overlay.jpg"
        try:
            size = capture_to(camera, dest)
        except gp.GPhoto2Error as e:
            print(f"    ! capture failed: {e}")
            continue

        # Compute crop box on the first frame; reuse same pixel box for all
        if box is None:
            with Image.open(dest) as im:
                iw, ih = im.size
            box = crop_box(iw, ih, args.crop_w, args.crop_h,
                           args.crop_x, args.crop_y)
            print(f"    image {iw}x{ih}, crop box (l,t,r,b) = {box} "
                  f"({box[2]-box[0]}x{box[3]-box[1]} px)")

        median, mean, shape = measure_crop(dest, box)
        save_overlay(dest, box, overlay)
        results.append((fec, actual, median, mean, dest, overlay))
        print(f"    saved {dest} ({size} bytes), overlay {overlay}")
        print(f"    crop {shape}  median={median:.1f}  mean={mean:.1f}")

    set_fec(camera, 0.0)
    print(f"\n  reset FEC = 0")

    print()
    print("=" * 70)
    print("FEC sweep results")
    print("=" * 70)
    if not results:
        print("  no successful captures")
        return

    ref = min(results, key=lambda r: abs(r[0]))
    ref_lin = linearise_srgb(ref[2])
    print(f"  reference: FEC={ref[0]:+.2f}  median={ref[2]:.1f}  "
          f"linear={ref_lin:.4f}\n")

    print(f"  {'FEC':>6}  {'median':>7}  {'mean':>7}  "
          f"{'measured ΔEV':>14}  {'expected':>10}  flag")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*14}  {'-'*10}  ----")
    for fec, actual, median, mean, _, _ in results:
        lin = linearise_srgb(median)
        delta_ev = float(np.log2(lin / ref_lin)) if ref_lin > 0 else float("nan")
        expected = fec - ref[0]
        diff = delta_ev - expected
        if abs(diff) < 0.5:
            flag = "OK"
        elif abs(diff) < 1.0:
            flag = "?"
        else:
            flag = "FAIL"
        print(f"  {fec:+6.2f}  {median:7.1f}  {mean:7.1f}  "
              f"{delta_ev:+14.2f}  {expected:+10.2f}  {flag}  (Δ={diff:+.2f})")

    print()
    print("  Interpretation")
    print("    OK    measured ΔEV within ±0.5 of expected — FEC is working")
    print("    ?     within ±1.0 — partial effect; ambient leak or flash ceiling")
    print("    FAIL  > ±1.0 — FEC not reaching flash, or saturation")
    print()
    print("  IMPORTANT: open one of the *_overlay.jpg files and confirm the")
    print("  red rectangle lands on the wafer. If it doesn't, rerun with")
    print("  adjusted --crop-w / --crop-h / --crop-x / --crop-y values.")


# ---- main --------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-w", type=float, default=0.20,
                   help="crop width as fraction of image width (default 0.20)")
    p.add_argument("--crop-h", type=float, default=0.20,
                   help="crop height as fraction of image height (default 0.20)")
    p.add_argument("--crop-x", type=float, default=0.00,
                   help="crop centre x offset, fraction of width; "
                        "positive = right (default 0.00)")
    p.add_argument("--crop-y", type=float, default=0.15,
                   help="crop centre y offset, fraction of height; "
                        "positive = below (default 0.15)")
    p.add_argument("--fec-values", type=float, nargs="+",
                   default=[-3.0, -2.0, -1.0, 0.0, 1.0],
                   help="FEC values to sweep (default -3 -2 -1 0 1)")
    p.add_argument("--settle", type=float, default=2.0,
                   help="seconds between shots for flash recycle (default 2.0)")
    p.add_argument("--skip-init", action="store_true",
                   help="skip applying init settings, go straight to FEC test")
    args = p.parse_args()

    print("=== opening camera ===")
    camera = gp.Camera()
    try:
        camera.init()
    except gp.GPhoto2Error as e:
        print(f"  init failed: {e}")
        sys.exit(1)

    try:
        cfg = camera.get_config()
        n = count_widgets(cfg)
        print(f"  config tree has {n} widgets")
        if n < 100:
            print("  ! tree too small — camera in minimal-MTP mode.")
            print("  ! Power off, REMOVE BATTERY 10s, reinsert, power on,")
            print("  ! wait for live view, THEN connect USB.")
            sys.exit(2)

        ep = find(cfg, "expprogram")
        if ep is None or ep.get_value() != "M":
            cur = ep.get_value() if ep is not None else "<missing>"
            print(f"  ! expprogram is {cur!r}; set mode dial to M and rerun.")
            sys.exit(2)
        print(f"  expprogram = {ep.get_value()!r} (good)")

        if not args.skip_init:
            print("\n=== applying initial settings ===")
            for name, cands, desc in INIT_SETTINGS:
                set_setting(camera, name, cands, desc)
            print("\n=== imagesize ===")
            try_max_imagesize(camera)
            print("\n=== VR ===")
            try_disable_vr(camera)

        fec_sweep_test(camera, args.fec_values, args)

    finally:
        camera.exit()
        print("\n=== closed ===")


if __name__ == "__main__":
    main()
