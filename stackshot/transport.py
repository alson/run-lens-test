"""Transport layer for the StackShot controller.

The framing logic lives entirely in :mod:`stackshot.protocol`. This
module wraps the wire transport (USB) and applies the mandatory 2 ms
inter-frame delay around every request/response exchange.

Two implementations:

* :class:`SerialTransport` — pyserial via ``/dev/ttyUSB0``. Kept
  for posterity and tests; **not** the production path. The Linux
  kernel's ``ftdi_sio`` driver pulses DTR by default on tty open,
  which the StackShot's MCU treats as a signal that wedges it. The
  serial transport works on machines / configurations where DTR
  isn't asserted, but on this rig it is the wedge cause.

* :class:`FtdiTransport` — libusb-direct via :mod:`pyusb`. The
  production path. Mirrors the init recipe in Cognisys's reference
  C++ source (``StackshotUSB_app.zip``):

  1. ``SET_BITMODE wValue=0x20FF`` — CBUS bit-bang mode, all CBUS
     pins as outputs. The StackShot's MCU watches a CBUS pin to
     decide whether to listen on its serial port; without this,
     the MCU stays in standalone / front-panel mode and silently
     drops every byte we send.
  2. 500 ms settle (per Cognisys's inline comment: "Sleep required
     here for slave device to debounce pins and enter USB control").
  3. Standard FT232 reset + UART config (BAUDRATE / LINE_PROPERTY /
     FLOW_CTRL).

  Notably, **never** issues ``SIO_SET_MODEM_CTRL``. DTR/RTS stay in
  whatever state the chip powers up in.

Both classes implement :class:`Transport`; the controller takes any
implementation via dependency injection so tests use a
``FakeTransport`` from the test module.
"""

from __future__ import annotations

import time
from typing import Protocol

import serial

from errors import StackShotProtocolError
from stackshot.protocol import Cmd, Frame, check_response

try:
    import usb.core
    import usb.util
    _PYUSB_AVAILABLE = True
except ImportError:                                  # pragma: no cover
    _PYUSB_AVAILABLE = False


class Transport(Protocol):
    """Interface used by :class:`StackShotController`.

    Implementations must serialise the frame on the wire, observe the
    inter-frame delay, and deliver back the next slave response.
    """

    def request(self, frame: Frame, expected_cmd: Cmd) -> Frame: ...

    def close(self) -> None: ...


class SerialTransport:
    """Production transport over an FTDI serial bridge."""

    def __init__(
        self,
        port: str,
        *,
        baud: int = 38400,
        timeout_s: float = 2.0,
        interframe_s: float = 0.002,
        sleep=time.sleep,
    ) -> None:
        self._port = port
        self._baud = baud
        self._timeout_s = timeout_s
        self._interframe_s = interframe_s
        self._sleep = sleep
        self._ser: serial.Serial | None = None

    def open(self) -> None:
        if self._ser is not None:
            return
        self._ser = serial.Serial(
            port=self._port,
            baudrate=self._baud,
            timeout=self._timeout_s,
            write_timeout=self._timeout_s,
        )
        # Make sure we don't carry any half-finished frame from before.
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def request(self, frame: Frame, expected_cmd: Cmd) -> Frame:
        if self._ser is None:
            raise RuntimeError("SerialTransport.request called before open()")
        # Inter-frame delay so the slave can resync if a previous
        # exchange aborted partway through.
        self._sleep(self._interframe_s)
        self._ser.write(frame.to_bytes())
        self._ser.flush()
        return check_response(self._read_one_frame(), expected_cmd)

    def _read_one_frame(self) -> Frame:
        assert self._ser is not None
        header = self._read_exact(5)
        length = header[4]
        tail = self._read_exact(length + 1)  # data + checksum
        return Frame.from_bytes(header + tail)

    def _read_exact(self, n: int) -> bytes:
        assert self._ser is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                raise StackShotProtocolError(
                    f"timed out waiting for {n} bytes (got {len(buf)})"
                )
            buf.extend(chunk)
        return bytes(buf)


# ---- Production transport: libusb-direct via pyusb -----------------------

# FTDI vendor request codes (host-to-device, recipient = device,
# bmRequestType = 0x40).
_BMREQ_VENDOR_OUT = 0x40
_SIO_RESET = 0x00
_SIO_SET_FLOW_CTRL = 0x02
_SIO_SET_BAUDRATE = 0x03
_SIO_SET_LINE_PROPERTY = 0x04
_SIO_SET_BIT_MODE = 0x0B

# wValue for SIO_RESET.
_RESET_SIO = 0x0000
_RESET_PURGE_RX = 0x0001
_RESET_PURGE_TX = 0x0002

# wValue for SIO_SET_BIT_MODE: top byte = mode, bottom byte = mask.
# mode 0x20 = CBUS bit bang; mask 0xFF = all CBUS pins as outputs.
_BITMODE_CBUS_BANG_ALL = 0x20FF

# wValue for SIO_SET_BAUDRATE at 38400. FTDI-encoded divisor for the
# FT232R; copied from the Zerene Stacker usbmon capture.
_BAUDRATE_38400 = 0xC04E

# wValue for SIO_SET_LINE_PROPERTY: 8 data bits, 1 stop, no parity.
_LINE_8N1 = 0x0008

# bulk endpoint addresses for FT232R (single channel).
_EP_IN = 0x81
_EP_OUT = 0x02

# FT232 prepends every bulk-IN packet with two status bytes (modem
# status + line status). They aren't part of the StackShot protocol
# and must be stripped before reassembling frames.
_FT232_STATUS_HEADER_LEN = 2

# Default StackShot USB IDs.
_DEFAULT_VID = 0x0403
_DEFAULT_PID = 0x6001


class FtdiTransport:
    """Production transport that drives the FT232R via libusb / pyusb.

    Performs the initialisation recipe found in Cognisys's reference
    C++ source (``StackshotUSB_app.zip``, ``Form1.h:2335-2342``) so
    the StackShot's MCU enters USB-control mode. The undocumented
    critical step is :data:`_SIO_SET_BIT_MODE` with wValue=0x20FF
    (CBUS bit bang, all pins as outputs) followed by a 500 ms settle.

    Caller is responsible for libusb permissions: either run as root
    (``sudo``) or install a udev rule giving the user access to
    ``0403:6001``.
    """

    def __init__(
        self,
        *,
        vid: int = _DEFAULT_VID,
        pid: int = _DEFAULT_PID,
        timeout_s: float = 2.0,
        interframe_s: float = 0.002,
        bitmode_settle_s: float = 0.5,
        sleep=time.sleep,
        clock=time.monotonic,
        usb_module=None,
    ) -> None:
        if not _PYUSB_AVAILABLE and usb_module is None:
            raise RuntimeError(
                "pyusb not installed; install with `uv add pyftdi` "
                "(brings pyusb as a transitive dep) or pin pyusb directly"
            )
        self._vid = vid
        self._pid = pid
        self._timeout_s = timeout_s
        self._interframe_s = interframe_s
        self._bitmode_settle_s = bitmode_settle_s
        self._sleep = sleep
        self._clock = clock
        self._usb = usb_module if usb_module is not None else usb
        self._dev = None
        self._buffer = b""

    # ---- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """Find the device, take it from the kernel driver if needed,
        run the FTDI init recipe."""
        if self._dev is not None:
            return
        dev = self._usb.core.find(idVendor=self._vid, idProduct=self._pid)
        if dev is None:
            raise StackShotProtocolError(
                f"no FT232R found at {self._vid:04x}:{self._pid:04x}"
            )
        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)
        cfg = dev.get_active_configuration()
        if cfg is None or cfg.bConfigurationValue != 1:
            dev.set_configuration()
        self._usb.util.claim_interface(dev, 0)
        self._init_ftdi(dev)
        self._dev = dev

    def close(self) -> None:
        if self._dev is None:
            return
        try:
            self._usb.util.release_interface(self._dev, 0)
            self._usb.util.dispose_resources(self._dev)
        finally:
            self._dev = None
            self._buffer = b""

    # ---- init recipe ---------------------------------------------------

    def _init_ftdi(self, dev) -> None:
        """Replicates Cognisys's reference C++ init.

        The SET_BITMODE step is the critical, undocumented one. Without
        it, the StackShot MCU silently drops every byte we send."""
        # Phase 0: SET_BITMODE = CBUS bit-bang + 500 ms settle.
        self._ctrl_out(dev, _SIO_SET_BIT_MODE, _BITMODE_CBUS_BANG_ALL)
        self._sleep(self._bitmode_settle_s)

        # Phase 1: standard FT232 reset cycle.
        self._reset_cycle(dev)

        # Phase 2: configure UART.
        self._ctrl_out(dev, _SIO_SET_BAUDRATE, _BAUDRATE_38400)
        self._ctrl_out(dev, _SIO_SET_LINE_PROPERTY, _LINE_8N1)
        self._ctrl_out(dev, _SIO_SET_FLOW_CTRL, 0x0000)

        # Phase 3: another reset (Zerene's capture does this too;
        # belt-and-braces against any garbage queued before the new
        # baud took effect).
        self._reset_cycle(dev)

    def _reset_cycle(self, dev) -> None:
        self._ctrl_out(dev, _SIO_RESET, _RESET_SIO)
        for _ in range(6):
            self._ctrl_out(dev, _SIO_RESET, _RESET_PURGE_RX)
        self._ctrl_out(dev, _SIO_RESET, _RESET_PURGE_TX)

    @staticmethod
    def _ctrl_out(dev, request: int, value: int, index: int = 0) -> None:
        dev.ctrl_transfer(_BMREQ_VENDOR_OUT, request, value, index, 0)

    # ---- request/response ----------------------------------------------

    def request(self, frame: Frame, expected_cmd: Cmd) -> Frame:
        if self._dev is None:
            raise RuntimeError("FtdiTransport.request called before open()")
        # Mandatory 2 ms inter-frame delay (PDF §2).
        self._sleep(self._interframe_s)
        self._dev.write(
            _EP_OUT, frame.to_bytes(),
            timeout=int(self._timeout_s * 1000),
        )
        return check_response(self._read_one_frame(), expected_cmd)

    def _read_one_frame(self) -> Frame:
        deadline = self._clock() + self._timeout_s
        while True:
            # Try to parse a complete frame from whatever is in
            # ``self._buffer``. A frame is sync(1) + cmd(2) + action(1)
            # + length(1) + data(length) + checksum(1).
            if len(self._buffer) >= 6:
                length = self._buffer[4]
                total = 5 + length + 1
                if len(self._buffer) >= total:
                    raw = bytes(self._buffer[:total])
                    self._buffer = self._buffer[total:]
                    return Frame.from_bytes(raw)

            if self._clock() >= deadline:
                raise StackShotProtocolError(
                    f"timed out waiting for response after "
                    f"{self._timeout_s:.1f}s; buffer={self._buffer.hex()}"
                )
            try:
                chunk = self._dev.read(_EP_IN, 64, timeout=200)
            except self._usb.core.USBError:
                # In production, dev.read(timeout=200) blocks for the
                # timeout before raising — natural pacing. Sleep a
                # short tick to enforce the same pacing under tests
                # where the fake device raises instantly.
                self._sleep(0.05)
                continue
            if chunk and len(chunk) > _FT232_STATUS_HEADER_LEN:
                self._buffer += bytes(chunk[_FT232_STATUS_HEADER_LEN:])
