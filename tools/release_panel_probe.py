"""One-off probe: test whether SET_BITMODE 0x00 releases the
StackShot's front panel so the hardware buttons can be used.

Sequence:
  1. Open libusb, claim FT232R.
  2. Acquire USB control (SET_BITMODE 0x20FF + 500 ms settle).
  3. Read position once — proves we're in PC control.
  4. Release USB control (SET_BITMODE wValue=0x00FF; mask=all,
     mode=0x00 = Reset / UART mode).
  5. Pause so the user can poke the front-panel buttons.
  6. ENTER → re-acquire (SET_BITMODE 0x20FF + 500 ms).
  7. Read position — should reflect wherever the user left the
     rail.

If the front panel responds during step 5, we have our answer:
``stackshot.jog`` can offer a hardware-jog mode that drops to
0x00 while the user uses the rail's buttons, then snaps the
position back via the protocol.
"""

from __future__ import annotations

import sys
import time

import usb.core
import usb.util


VID, PID = 0x0403, 0x6001
BMREQ_VENDOR_OUT = 0x40
BMREQ_VENDOR_IN = 0xC0
SIO_RESET = 0x00
SIO_SET_BAUDRATE = 0x03
SIO_SET_LINE_PROPERTY = 0x04
SIO_SET_FLOW_CTRL = 0x02
SIO_SET_BIT_MODE = 0x0B
EP_OUT, EP_IN = 0x02, 0x81

# Bit-mode wValues: lower byte = pin mask, upper byte = mode.
# Mode 0x20 = CBUS bit-bang. Mask is which CBUS pins are outputs.
# Acquire: mask 0xFF (all 4 CBUS pins driven by host).
# Release: mask 0xF0 (top 4 outputs, bottom 4 high-impedance inputs).
# Per Form1.h:2311, the bottom-4 release is what lets the StackShot
# MCU see "host has let go" and switch back to front-panel control.
BIT_MODE_CBUS_BANG = 0x20FF
BIT_MODE_CBUS_RELEASE = 0x20F0

# CC_RAIL_POSITION_CURRENT = 0x1002, action READ, units METRIC=0x01.
GET_POS_FRAME = bytes.fromhex("55100200010169")

# CC_CLOSE = 0x110E (Form1.h:2313 sends this just before FT_Close).
# Action WRITE, length 0, sentinel checksum 0x69.
CC_CLOSE_FRAME = bytes([0x55, 0x11, 0x0E, 0x01, 0x00, 0x69])


def vendor_out(dev, request: int, value: int) -> None:
    dev.ctrl_transfer(BMREQ_VENDOR_OUT, request, value, 0, 0)


def reset_cycle(dev) -> None:
    vendor_out(dev, SIO_RESET, 0x0000)
    for _ in range(6):
        vendor_out(dev, SIO_RESET, 0x0001)
    vendor_out(dev, SIO_RESET, 0x0002)


def acquire(dev) -> None:
    vendor_out(dev, SIO_SET_BIT_MODE, BIT_MODE_CBUS_BANG)
    time.sleep(0.5)
    reset_cycle(dev)
    vendor_out(dev, SIO_SET_BAUDRATE, 0xC04E)
    vendor_out(dev, SIO_SET_LINE_PROPERTY, 0x0008)
    vendor_out(dev, SIO_SET_FLOW_CTRL, 0x0000)
    reset_cycle(dev)


def _read_response(dev, *, timeout_s: float = 3.0) -> bytes:
    """Read one StackShot response frame from bulk-IN. Returns the
    decoded frame body (without the FT232 status header).
    """
    deadline = time.time() + timeout_s
    buf = b""
    while time.time() < deadline:
        try:
            chunk = bytes(dev.read(EP_IN, 64, timeout=200))
        except usb.core.USBError:
            time.sleep(0.05)
            continue
        if len(chunk) > 2:
            buf += chunk[2:]
        if len(buf) >= 6:
            length = buf[4]
            if len(buf) >= 5 + length + 1:
                return buf[: 5 + length + 1]
    raise RuntimeError(f"no response within {timeout_s}s; buf={buf.hex()}")


def release_panel(dev) -> None:
    """Mirror Form1.h:2311-2313:
       1. SET_BITMODE wValue=0x20F0 (CBUS bang, bottom 4 pins released).
       2. Send CC_CLOSE (0x110E) WRITE.
    """
    vendor_out(dev, SIO_SET_BIT_MODE, BIT_MODE_CBUS_RELEASE)
    dev.write(EP_OUT, CC_CLOSE_FRAME, timeout=2000)
    rsp = _read_response(dev)
    print(f"  CC_CLOSE response: {rsp.hex()}")


def read_position(dev) -> float:
    dev.write(EP_OUT, GET_POS_FRAME, timeout=2000)
    deadline = time.time() + 3.0
    buf = b""
    while time.time() < deadline:
        try:
            chunk = bytes(dev.read(EP_IN, 64, timeout=200))
        except usb.core.USBError:
            time.sleep(0.05)
            continue
        if len(chunk) > 2:
            buf += chunk[2:]
        if len(buf) >= 6:
            length = buf[4]
            if len(buf) >= 5 + length + 1:
                # response: aa 10 02 02 05 <units> <F32 mm> <cs>
                import struct
                if length >= 5:
                    units = buf[5]
                    pos = struct.unpack("<f", buf[6:10])[0]
                    return pos
                break
    raise RuntimeError(f"no position response; buf={buf.hex()}")


def main() -> int:
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        print("StackShot not found", file=sys.stderr)
        return 1
    if dev.is_kernel_driver_active(0):
        dev.detach_kernel_driver(0)
    cfg = dev.get_active_configuration()
    if cfg is None or cfg.bConfigurationValue != 1:
        dev.set_configuration()
    usb.util.claim_interface(dev, 0)
    try:
        print("acquiring USB control...")
        acquire(dev)
        pos_before = read_position(dev)
        print(f"  position now: {pos_before:+.4f} mm  (front panel should show 'PC')")

        print("\nreleasing USB control (SET_BITMODE 0x00)...")
        release_panel(dev)
        print("  Try the front-panel buttons NOW.")
        print("  Does the rail respond? Does 'PC' disappear from the display?")
        input("  Move the rail somewhere using the buttons, then press ENTER: ")

        print("\nre-acquiring USB control...")
        acquire(dev)
        pos_after = read_position(dev)
        print(f"  position now: {pos_after:+.4f} mm")
        print(f"  delta: {pos_after - pos_before:+.4f} mm")
        return 0
    finally:
        usb.util.release_interface(dev, 0)
        usb.util.dispose_resources(dev)


if __name__ == "__main__":
    sys.exit(main())
