"""Probe the StackShot via pyftdi / libusb, bypassing ftdi_sio entirely.

Run with sudo (libusb needs raw USB device access):

    ! sudo .venv/bin/python tools/pyftdi_probe.py

Some FT232 quirks (DTR pulse on open, latency timer init, hardware
flow-control negotiation) are handled differently in the kernel
ftdi_sio path vs libusb. If the StackShot responds via libusb but
not via /dev/ttyUSB0, we know ftdi_sio is the issue and we can either
work around it or switch the production transport to pyftdi.
"""

from __future__ import annotations

import time

from pyftdi.ftdi import Ftdi


def main() -> None:
    url = "ftdi://0x0403:0x6001/1"
    print(f"opening {url}")
    ftdi = Ftdi.create_from_url(url)

    print("set 38400 8N1")
    ftdi.set_baudrate(38400)
    ftdi.set_line_property(8, 1, "N")
    ftdi.purge_buffers()

    # Settle so the device is happy after open.
    time.sleep(1.0)

    # 1. Listen for unsolicited data first.
    end = time.time() + 1.0
    buf = b""
    while time.time() < end:
        chunk = ftdi.read_data(64)
        if chunk:
            buf += bytes(chunk)
    print(f"unsolicited 1s: {len(buf)} bytes hex={buf.hex()}")

    # 2. Send Get Software ID.
    body = bytes([0x55, 0x11, 0x03, 0x00, 0x00])
    frame = body + bytes([sum(body) & 0xFF])
    print(f"\nSEND (Get Software ID): {frame.hex()}")
    n = ftdi.write_data(frame)
    print(f"  wrote {n} bytes")

    end = time.time() + 3.0
    buf = b""
    while time.time() < end:
        chunk = ftdi.read_data(64)
        if chunk:
            buf += bytes(chunk)
    print(f"RECV: {len(buf)} bytes hex={buf.hex()}")

    # 3. Send Get Status.
    time.sleep(0.5)
    ftdi.purge_buffers()
    body = bytes([0x55, 0x10, 0x05, 0x00, 0x00])
    frame = body + bytes([sum(body) & 0xFF])
    print(f"\nSEND (Get Status): {frame.hex()}")
    ftdi.write_data(frame)

    end = time.time() + 3.0
    buf = b""
    while time.time() < end:
        chunk = ftdi.read_data(64)
        if chunk:
            buf += bytes(chunk)
    print(f"RECV: {len(buf)} bytes hex={buf.hex()}")

    ftdi.close()
    print("\nclosed")


if __name__ == "__main__":
    main()
