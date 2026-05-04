#!/usr/bin/env python3
"""
Z7 + gphoto2 diagnostic. Single persistent connection.

Run after:
  - Camera mode dial = M (or U2 if U2 is configured for M)
  - Camera awake (half-press shutter once)
  - SU-800 + at least one SB-900/800 in AA mode mounted and switched on
  - USB cable plugged into the laptop (Bumblebee)

Then:
  python3 z7_diagnostic.py
"""

import sys
import os
try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed. Try:")
    print("  sudo pacman -S python-gphoto2          # if available in repos")
    print("  pip install --user --break-system-packages gphoto2")
    sys.exit(1)


TYPE_NAMES = {
    gp.GP_WIDGET_WINDOW: "WINDOW",
    gp.GP_WIDGET_SECTION: "SECTION",
    gp.GP_WIDGET_TEXT: "TEXT",
    gp.GP_WIDGET_RANGE: "RANGE",
    gp.GP_WIDGET_TOGGLE: "TOGGLE",
    gp.GP_WIDGET_RADIO: "RADIO",
    gp.GP_WIDGET_MENU: "MENU",
    gp.GP_WIDGET_BUTTON: "BUTTON",
    gp.GP_WIDGET_DATE: "DATE",
}


def walk(widget, indent=0, out=None):
    if out is None:
        out = []
    name = widget.get_name()
    label = widget.get_label()
    wtype = widget.get_type()
    tname = TYPE_NAMES.get(wtype, str(wtype))
    pad = "  " * indent
    if wtype in (gp.GP_WIDGET_WINDOW, gp.GP_WIDGET_SECTION):
        out.append(f"{pad}[{tname}] {name} ({label})")
    else:
        try:
            val = widget.get_value()
        except gp.GPhoto2Error as e:
            val = f"<error: {e}>"
        ro = widget.get_readonly()
        out.append(f"{pad}{name} [{tname}] ro={ro} value={val!r}")
    for i in range(widget.count_children()):
        walk(widget.get_child(i), indent + 1, out)
    return out


def find(root, name):
    if root.get_name() == name:
        return root
    for i in range(root.count_children()):
        r = find(root.get_child(i), name)
        if r is not None:
            return r
    return None


def set_widget(camera, name, value):
    """Set a config value via fresh get_config -> set_config. Returns True on success."""
    cfg = camera.get_config()
    w = find(cfg, name)
    if w is None:
        print(f"  ! widget {name!r} not in config tree")
        return False
    try:
        w.set_value(value)
        camera.set_config(cfg)
        print(f"  set {name} = {value!r}")
        return True
    except gp.GPhoto2Error as e:
        print(f"  ! set {name}={value!r} failed: {e}")
        return False


def get_widget_value(camera, name):
    cfg = camera.get_config()
    w = find(cfg, name)
    if w is None:
        return None, None
    try:
        return w.get_value(), w.get_readonly()
    except gp.GPhoto2Error as e:
        return f"<error: {e}>", None


def capture_to(camera, dest):
    try:
        fp = camera.capture(gp.GP_CAPTURE_IMAGE)
        print(f"  captured: {fp.folder}/{fp.name}")
        cf = camera.file_get(fp.folder, fp.name, gp.GP_FILE_TYPE_NORMAL)
        cf.save(dest)
        print(f"  saved {dest} ({os.path.getsize(dest)} bytes)")
        return True
    except gp.GPhoto2Error as e:
        print(f"  ! capture failed: {e}")
        return False


def main():
    print("=== opening camera ===")
    camera = gp.Camera()
    try:
        camera.init()
    except gp.GPhoto2Error as e:
        print(f"init failed: {e}")
        sys.exit(1)

    try:
        # 1. dump abilities
        print("\n=== abilities ===")
        ab = camera.get_abilities()
        ops = ab.operations
        print(f"  model: {ab.model}")
        print(f"  operations bitfield: {ops}")
        print(f"  GP_OPERATION_CAPTURE_IMAGE supported: {bool(ops & gp.GP_OPERATION_CAPTURE_IMAGE)}")
        print(f"  GP_OPERATION_CAPTURE_PREVIEW supported: {bool(ops & gp.GP_OPERATION_CAPTURE_PREVIEW)}")
        print(f"  GP_OPERATION_CONFIG supported: {bool(ops & gp.GP_OPERATION_CONFIG)}")

        # 2. dump config tree
        print("\n=== dumping config tree to /tmp/z7_full_config.txt ===")
        cfg = camera.get_config()
        lines = walk(cfg)
        with open("/tmp/z7_full_config.txt", "w") as f:
            f.write("\n".join(lines))
        print(f"  {len(lines)} lines written")

        # 3. probe key widgets
        print("\n=== key widgets ===")
        for name in ("expprogram", "capturetarget", "recordingmedia",
                     "imagequality", "iso", "f-number", "shutterspeed",
                     "exposurecompensation", "flashexposurecompensation",
                     "flashmode", "controlmode"):
            v, ro = get_widget_value(camera, name)
            if v is None and ro is None:
                print(f"  {name}: NOT IN TREE")
            else:
                print(f"  {name}: ro={ro} value={v!r}")

        # 4. try to set capturetarget (Z7 needs Internal RAM/SDRAM for generic capture)
        print("\n=== setting capturetarget ===")
        # try by name first, then by index
        ok = set_widget(camera, "capturetarget", "Internal RAM")
        if not ok:
            ok = set_widget(camera, "capturetarget", "Memory card")

        # 5. first capture (FEC = 0)
        print("\n=== set FEC = 0 ===")
        set_widget(camera, "flashexposurecompensation", 0.0)
        print("=== capture #1 -> /tmp/fec_zero.jpg ===")
        capture_to(camera, "/tmp/fec_zero.jpg")

        # 6. second capture (FEC = -2)
        print("\n=== set FEC = -2 ===")
        set_widget(camera, "flashexposurecompensation", -2.0)
        print("=== capture #2 -> /tmp/fec_minus2.jpg ===")
        capture_to(camera, "/tmp/fec_minus2.jpg")

        # 7. reset FEC
        print("\n=== reset FEC = 0 ===")
        set_widget(camera, "flashexposurecompensation", 0.0)

    finally:
        camera.exit()
        print("\n=== closed ===")


if __name__ == "__main__":
    main()
