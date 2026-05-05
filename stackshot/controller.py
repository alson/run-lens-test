"""High-level StackShot controller.

Wraps a :class:`stackshot.transport.Transport` with semantically named
operations: move-and-wait, get position, zero, stop, identification.

The class deliberately accepts a transport, a clock callable, and a
sleep callable so the unit tests can run without serial hardware and
without real time passing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from errors import StackShotTimeout, StackShotUserAborted
from stackshot.protocol import (
    Action,
    Cmd,
    Direction,
    Frame,
    RailStatus,
    Units,
    build_request,
    decode_id_response,
    decode_position_response,
    decode_status_response,
    encode_move,
    encode_units_only,
)
from stackshot.transport import FtdiTransport, SerialTransport, Transport

DEFAULT_BAUD = 38400


@dataclass(frozen=True)
class StackShotInfo:
    """Identification fields read at session startup."""

    software_id: int
    hardware_id: int


class StackShotController:
    """Direct serial control of the original (1-axis) StackShot rail.

    Parameters
    ----------
    transport :
        Anything implementing :class:`stackshot.transport.Transport`.
    poll_interval_s :
        Status-poll cadence used by :meth:`wait_until_idle`.
    move_timeout_s :
        Hard deadline for any single move; if exceeded we issue a STOP
        and raise :class:`StackShotTimeout`.
    clock :
        Monotonic time source (defaults to :func:`time.monotonic`).
    sleep :
        Sleep callable (defaults to :func:`time.sleep`).
    """

    def __init__(
        self,
        transport: Transport,
        *,
        poll_interval_s: float = 0.10,
        move_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tx = transport
        self._poll_interval_s = poll_interval_s
        self._move_timeout_s = move_timeout_s
        self._clock = clock
        self._sleep = sleep

    # ---- Identification ---------------------------------------------------

    def info(self) -> StackShotInfo:
        return StackShotInfo(
            software_id=self.software_id(),
            hardware_id=self.hardware_id(),
        )

    def software_id(self) -> int:
        rsp = self._read(Cmd.SOFTWARE_ID)
        return decode_id_response(rsp.data)

    def hardware_id(self) -> int:
        rsp = self._read(Cmd.HARDWARE_ID)
        return decode_id_response(rsp.data)

    # ---- Position ---------------------------------------------------------

    def get_position(self, units: Units = Units.METRIC) -> float:
        rsp = self._read(Cmd.RAIL_POSITION_CURRENT, encode_units_only(units))
        rsp_units, position = decode_position_response(rsp.data)
        if rsp_units is not units:
            # Normally the slave echoes our requested units. If it ever
            # didn't, the position would be in a different unit than
            # the caller asked for — better to fail loudly.
            raise ValueError(
                f"position response in {rsp_units!r} but {units!r} was requested"
            )
        return position

    def zero_position(self) -> None:
        self._write(Cmd.RAIL_POSITION_ZERO)

    # ---- Status / motion --------------------------------------------------

    def get_status(self) -> RailStatus:
        # CC_RAIL_STATUS is documented as Read-Only but the cmd-data is
        # empty — i.e. we send a READ with length 0.
        rsp = self._read(Cmd.RAIL_STATUS)
        return decode_status_response(rsp.data)

    def stop(self) -> None:
        self._write(Cmd.RAIL_STOP)

    def move(
        self,
        direction: Direction,
        distance_mm: float,
        *,
        backlash: bool = True,
    ) -> None:
        """Issue a relative move; does not wait for completion.

        ``CC_RAIL_MOVE`` is relative to the *target* position, not the
        current position — so back-to-back moves stack. Callers should
        :meth:`wait_until_idle` between moves to avoid surprises.
        """
        data = encode_move(direction, Units.METRIC, backlash, distance_mm)
        self._write(Cmd.RAIL_MOVE, data)

    def wait_until_idle(self, *, timeout_s: float | None = None) -> None:
        """Poll status until MOVING clears or timeout / abort fires.

        Raises
        ------
        StackShotUserAborted
            If the controller observed a front-panel abort.
        StackShotTimeout
            If the move did not complete within ``timeout_s`` (defaults
            to the controller's ``move_timeout_s``).
        """
        deadline = self._clock() + (timeout_s or self._move_timeout_s)
        while True:
            status = self.get_status()
            if RailStatus.USER_ABORTED in status:
                # The bit is sticky and cleared on read; raise so the
                # caller treats the move as failed.
                raise StackShotUserAborted("front-panel abort during move")
            if RailStatus.MOVING not in status:
                return
            if self._clock() >= deadline:
                # Best-effort: try to stop the rail before giving up.
                try:
                    self.stop()
                except Exception:  # noqa: BLE001 — diagnostic only
                    pass
                raise StackShotTimeout(
                    f"move did not complete within {timeout_s or self._move_timeout_s}s"
                )
            self._sleep(self._poll_interval_s)

    def move_and_wait(
        self,
        direction: Direction,
        distance_mm: float,
        *,
        backlash: bool = True,
        timeout_s: float | None = None,
    ) -> None:
        """Convenience: :meth:`move` followed by :meth:`wait_until_idle`."""
        self.move(direction, distance_mm, backlash=backlash)
        self.wait_until_idle(timeout_s=timeout_s)

    # ---- Internals --------------------------------------------------------

    def _read(self, cmd: Cmd, data: bytes = b"") -> Frame:
        return self._tx.request(build_request(cmd, Action.READ, data), cmd)

    def _write(self, cmd: Cmd, data: bytes = b"") -> Frame:
        return self._tx.request(build_request(cmd, Action.WRITE, data), cmd)

    # ---- Lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._tx.close()


# ---- Convenience constructor for the production case ---------------------

def open_serial(
    port: str,
    *,
    baud: int = DEFAULT_BAUD,
    poll_interval_s: float = 0.10,
    move_timeout_s: float = 30.0,
) -> StackShotController:
    """Open over pyserial / /dev/ttyUSB0.

    NOTE: not the production path. The Linux ``ftdi_sio`` kernel
    driver pulses DTR by default on tty open, which on this rig
    wedges the StackShot's MCU until power-cycle. Kept for tests and
    for use on rigs / firmware versions where DTR isn't asserted.
    Use :func:`open_ftdi` in production.
    """
    transport = SerialTransport(port, baud=baud)
    transport.open()
    return StackShotController(
        transport,
        poll_interval_s=poll_interval_s,
        move_timeout_s=move_timeout_s,
    )


def open_ftdi(
    *,
    poll_interval_s: float = 0.10,
    move_timeout_s: float = 30.0,
) -> StackShotController:
    """Open via libusb-direct (pyusb), running the Cognisys-reference
    init recipe (SET_BITMODE 0x20FF + 500 ms settle, then RESET /
    BAUDRATE / LINE_PROPERTY / FLOW_CTRL). This is the production
    path. See :class:`stackshot.transport.FtdiTransport` for details.

    Requires libusb permissions on VID 0x0403, PID 0x6001 — either
    ``sudo`` or a udev rule (see ``INTEGRATION_TESTING.md``).
    """
    transport = FtdiTransport()
    transport.open()
    return StackShotController(
        transport,
        poll_interval_s=poll_interval_s,
        move_timeout_s=move_timeout_s,
    )


# ---- CLI for rig smoke-tests ---------------------------------------------

def _cli() -> None:
    """Tiny CLI: ``python -m stackshot.controller --info`` etc.

    Default transport is libusb-direct via :func:`open_ftdi`. Pass
    ``--serial /dev/ttyUSB0`` to fall back to the legacy pyserial
    path on rigs where it works.
    """
    import argparse

    parser = argparse.ArgumentParser(description="StackShot controller CLI")
    parser.add_argument("--serial", default=None, metavar="PORT",
                        help="legacy: open via pyserial at this device path")
    parser.add_argument("--info", action="store_true",
                        help="print software/hardware ID + current position")
    parser.add_argument("--move", type=float, default=None,
                        help="relative forward move in mm (use --back for backward)")
    parser.add_argument("--back", action="store_true",
                        help="reverse direction for --move")
    parser.add_argument("--zero", action="store_true",
                        help="zero the current position")
    args = parser.parse_args()

    if args.serial:
        controller = open_serial(args.serial)
    else:
        controller = open_ftdi()
    try:
        if args.info or args.move is None and not args.zero:
            info = controller.info()
            position = controller.get_position()
            print(f"software_id = {info.software_id:#010x}")
            print(f"hardware_id = {info.hardware_id:#010x}")
            print(f"position    = {position:+.4f} mm")
        if args.zero:
            controller.zero_position()
            print("position zeroed")
        if args.move is not None:
            direction = Direction.BACKWARD if args.back else Direction.FORWARD
            controller.move_and_wait(direction, abs(args.move))
            position = controller.get_position()
            print(f"after move: {position:+.4f} mm")
    finally:
        controller.close()


if __name__ == "__main__":
    _cli()
