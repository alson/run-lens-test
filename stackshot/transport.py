"""Serial transport for the StackShot controller.

The framing logic lives entirely in :mod:`stackshot.protocol`. This
module is just a thin wrapper around :class:`serial.Serial` that
applies the mandatory 2 ms inter-frame delay and reads the variable-
length response in one place.

Kept small and dependency-injectable so that
:class:`stackshot.controller.StackShotController` is testable without
a real serial port.
"""

from __future__ import annotations

import time
from typing import Protocol

import serial

from errors import StackShotProtocolError
from stackshot.protocol import Cmd, Frame, check_response


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
