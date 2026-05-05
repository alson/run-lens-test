"""StackShot probe via raw pyusb. Replicates Zerene Stacker's exact
sequence captured with usbmon — including the double-RESET cycle,
the post-init delay, the BAUDRATE→LINE→FLOW order, the constant
``0x69`` sentinel checksum, and pre-arming bulk-IN reads before the
first command.

Run with sudo (libusb needs raw USB device access):

    ! sudo .venv/bin/python tools/pyftdi_probe.py

While this runs, capture usbmon in another terminal:

    sudo modprobe usbmon
    sudo cat /sys/kernel/debug/usb/usbmon/1u > /tmp/usbmon-our.txt &
    # run the probe
    kill %1

Then compare /tmp/usbmon-our.txt against the Zerene capture to spot
any remaining difference.
"""

from __future__ import annotations

import sys
import time

import usb.core
import usb.util

VID = 0x0403
PID = 0x6001
EP_IN = 0x81
EP_OUT = 0x02

BMREQ_VENDOR_OUT = 0x40
BMREQ_VENDOR_IN = 0xC0

SIO_RESET = 0x00
SIO_SET_MODEM_CTRL = 0x01     # we deliberately never call this
SIO_SET_FLOW_CTRL = 0x02
SIO_SET_BAUDRATE = 0x03
SIO_SET_LINE_PROPERTY = 0x04
SIO_POLL_MODEM_STATUS = 0x05
SIO_GET_LATENCY_TIMER = 0x0A
SIO_SET_BIT_MODE = 0x0B
"""FTDI vendor request 0x0B (SET_BITMODE).

For the StackShot, the Cognisys reference C++ app calls
``FT_SetBitMode(handle, 0xFF, 0x20)`` immediately after open, then
sleeps 500 ms (their inline comment: "Sleep required here for slave
device to debounce pins and enter USB control"). 0x20 = CBUS bit bang
mode; 0xFF = all CBUS pins as outputs. The StackShot MCU sees this as
the "host wants to drive me via USB" signal and enables its serial
command listener. Without this call, the MCU stays in front-panel-only
mode and ignores everything we send.

Encoded on the wire as wValue = (mode << 8) | mask = 0x20FF.
"""

RESET_SIO = 0x0000
RESET_PURGE_RX = 0x0001
RESET_PURGE_TX = 0x0002


def vendor_out(dev, request: int, value: int, index: int = 0) -> None:
    dev.ctrl_transfer(BMREQ_VENDOR_OUT, request, value, index, 0)


def vendor_in(dev, request: int, length: int) -> bytes:
    return bytes(dev.ctrl_transfer(BMREQ_VENDOR_IN, request, 0, 0, length))


def reset_cycle(dev) -> None:
    """Zerene's reset cycle: 1× RESET_SIO, 6× PURGE_RX, 1× PURGE_TX."""
    vendor_out(dev, SIO_RESET, RESET_SIO)
    for _ in range(6):
        vendor_out(dev, SIO_RESET, RESET_PURGE_RX)
    vendor_out(dev, SIO_RESET, RESET_PURGE_TX)


def configure_uart(dev) -> None:
    """Zerene's order: BAUDRATE, then LINE_PROPERTY, then FLOW_CTRL.

    Note: never SIO_SET_MODEM_CTRL. Linux's ftdi_sio asserts DTR by
    default on tty open, which wedges the StackShot's MCU.
    """
    vendor_out(dev, SIO_SET_BAUDRATE, 0xC04E)         # 38400
    vendor_out(dev, SIO_SET_LINE_PROPERTY, 0x0008)    # 8N1
    vendor_out(dev, SIO_SET_FLOW_CTRL, 0x0000)        # NONE


def drain_bulk_in(dev, *, max_iters: int = 20, timeout_ms: int = 50) -> None:
    """Read until the bulk-IN endpoint stops returning data. The first
    2 bytes of every FT232 packet are the modem/line status header,
    so a "drained" packet has length 2."""
    for _ in range(max_iters):
        try:
            chunk = dev.read(EP_IN, 64, timeout=timeout_ms)
        except usb.core.USBTimeoutError:
            return
        if not chunk or len(chunk) <= 2:
            return


def main() -> int:
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        print(f"no FT232R found at {VID:04x}:{PID:04x}", file=sys.stderr)
        return 1

    if dev.is_kernel_driver_active(0):
        dev.detach_kernel_driver(0)
        print("detached kernel driver from interface 0")

    cfg = dev.get_active_configuration()
    if cfg is None or cfg.bConfigurationValue != 1:
        dev.set_configuration()
        print("set_configuration(1)")
    else:
        print("already in config 1; not calling set_configuration()")

    usb.util.claim_interface(dev, 0)
    print("interface 0 claimed")

    try:
        # Phase 0: SET_BITMODE — the missing call we found in the
        # Cognisys reference C++ source. Without this, the StackShot
        # MCU stays in front-panel mode and ignores every command.
        print("\nphase 0: SET_BITMODE = CBUS bit bang (0x20FF) + 500ms settle")
        vendor_out(dev, SIO_SET_BIT_MODE, 0x20FF)
        time.sleep(0.5)

        # Phase 1 (mirrors Zerene's first reset block).
        print("\nphase 1: reset cycle")
        reset_cycle(dev)

        # Read modem status + latency (Zerene does this).
        ms = vendor_in(dev, SIO_POLL_MODEM_STATUS, 2)
        lt = vendor_in(dev, SIO_GET_LATENCY_TIMER, 1)
        print(f"  modem_status={ms.hex()}  latency_timer={lt.hex()}")

        # Phase 2 (Zerene does a second reset cycle ~500ms later).
        time.sleep(0.5)
        print("\nphase 2: reset cycle (mirroring Zerene's second pass)")
        reset_cycle(dev)

        # Phase 3: configure UART.
        time.sleep(0.5)
        print("\nphase 3: configure UART (BAUDRATE, LINE_PROPERTY, FLOW_CTRL)")
        configure_uart(dev)

        # Phase 4: another reset cycle (Zerene does this).
        print("\nphase 4: reset cycle")
        reset_cycle(dev)

        # Drain whatever is in the bulk-IN buffer.
        print("\ndraining bulk-IN")
        drain_bulk_in(dev)

        # Wait a bit (Zerene has a ~500ms gap before sending).
        time.sleep(0.5)

        # Send Get Software String — same command Zerene used first.
        # Use Zerene's literal sentinel checksum byte (0x69), not the
        # sum-of-bytes value (which would be 0x68). The PDF says the
        # checksum is "sent but not currently verified" and Zerene
        # confirms that by sending a constant 0x69 regardless of frame
        # contents.
        frame = bytes([0x55, 0x11, 0x02, 0x00, 0x00, 0x69])
        print(f"\nSEND: {frame.hex()}")
        n = dev.write(EP_OUT, frame, timeout=2000)
        print(f"  wrote {n} bytes")

        deadline = time.time() + 5.0
        payload = b""
        raw_packets: list[bytes] = []
        while time.time() < deadline:
            try:
                chunk = dev.read(EP_IN, 64, timeout=300)
            except usb.core.USBTimeoutError:
                continue
            if not chunk:
                continue
            data = bytes(chunk)
            raw_packets.append(data)
            if len(data) > 2:
                payload += data[2:]
            if len(payload) >= 6:
                length = payload[4]
                if len(payload) >= 5 + length + 1:
                    break

        print(f"\nraw bulk-IN packets seen: {len(raw_packets)}")
        for p in raw_packets[:8]:
            print(f"  {p.hex()}")
        print(f"\nRECV payload: {len(payload)} bytes hex={payload.hex()}")
        if len(payload) >= 6 and payload[0] == 0xAA:
            cmd = (payload[1] << 8) | payload[2]
            action = payload[3]
            length = payload[4]
            data = payload[5:5 + length]
            print(
                f"  parsed: sync=0xAA  cmd=0x{cmd:04x}  "
                f"action=0x{action:02x}  len={length}"
            )
            print(f"  data hex={data.hex()}")
            try:
                ascii_str = data.rstrip(b"\x00").decode("ascii")
                print(f"  data ascii={ascii_str!r}")
            except UnicodeDecodeError:
                pass
            return 0

        return 2
    finally:
        usb.util.release_interface(dev, 0)
        usb.util.dispose_resources(dev)


if __name__ == "__main__":
    sys.exit(main())
