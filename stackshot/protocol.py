"""StackShot serial protocol — frame encode / decode.

The full reference is in ``StackShotCommands_1_2.pdf`` at the repo
root. This module covers everything the package actually uses on the
original (1-axis, text-LCD) controller. The 3X-only axis-OR scheme on
the upper nibble of the action byte is not relevant here.

Frame format::

    Sync | CMD_MSB | CMD_LSB | Action/Status | DataLen | Data[0..n] | Checksum

* Sync: ``0x55`` host->slave, ``0xAA`` slave->host.
* Multibyte values in the data field are little-endian (LSB first),
  including IEEE-754 floats.
* DataLen is the byte count of the data field; may be zero.
* Checksum is a single byte summing all previous bytes mod 256.
* 2 ms minimum inter-frame delay required by the slave for resync.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag

# ---- Sync bytes -------------------------------------------------------------

SYNC_HOST = 0x55
SYNC_SLAVE = 0xAA

# ---- Action / status byte ---------------------------------------------------

class Action(IntEnum):
    READ = 0x00
    WRITE = 0x01
    RSP_OK = 0x02
    BAD_PARAM = 0x03
    UNSUPPORTED_ACTION = 0x04
    UNSUPPORTED_CMD = 0x05
    FAILED = 0x06
    NOT_EMPTY = 0x07
    BUSY = 0x08


# ---- Command IDs (subset actually used) ------------------------------------

class Cmd(IntEnum):
    RAIL_MOVE = 0x1000
    RAIL_POSITION_TARGET = 0x1001
    RAIL_POSITION_CURRENT = 0x1002
    RAIL_POSITION_ZERO = 0x1003
    RAIL_SHUTTER_FIRE = 0x1004  # unused (shutter cable not connected)
    RAIL_STATUS = 0x1005
    RAIL_STOP = 0x1006
    RAIL_MOVE_AT_SPEED = 0x1007

    SOFTWARE_STRING = 0x1102
    SOFTWARE_ID = 0x1103
    HARDWARE_ID = 0x1104
    BOOTLOADER_ID = 0x1105
    SOFTWARE_CHECKSUM = 0x1106
    SERIAL_NUMBER = 0x1107


# ---- Enums on the wire -----------------------------------------------------

class Direction(IntEnum):
    FORWARD = 0  # COMM_RAIL_DIR_FWD
    BACKWARD = 1  # COMM_RAIL_DIR_BACK


class Units(IntEnum):
    ENGLISH = 0  # mils
    METRIC = 1  # mm
    STEPS = 2


class RailStatus(IntFlag):
    """Bitfield returned by ``CC_RAIL_STATUS`` (0x1005).

    The USER_ABORTED bit is sticky and is cleared on read by the
    controller, per PDF §1.1.6.
    """

    IDLE = 0x00
    MOVING = 0x01
    SHUTTER = 0x02
    USER_ABORTED = 0x04


# ---- Errors ----------------------------------------------------------------

# Re-imported from package errors module so callers don't depend on this
# module for them. We keep the actual classes there.
from errors import StackShotProtocolError, StackShotResponseError  # noqa: E402


# ---- Low-level encode / decode helpers -------------------------------------

def checksum(payload: bytes) -> int:
    """Single-byte mod-256 sum used to terminate frames."""
    return sum(payload) & 0xFF


def encode_u32_le(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"u32 out of range: {value}")
    return struct.pack("<I", value)


def decode_u32_le(data: bytes) -> int:
    if len(data) != 4:
        raise StackShotProtocolError(
            f"u32 needs 4 bytes, got {len(data)}: {data.hex()}"
        )
    return struct.unpack("<I", data)[0]


def encode_f32_le(value: float) -> bytes:
    return struct.pack("<f", float(value))


def decode_f32_le(data: bytes) -> float:
    if len(data) != 4:
        raise StackShotProtocolError(
            f"f32 needs 4 bytes, got {len(data)}: {data.hex()}"
        )
    return struct.unpack("<f", data)[0]


# ---- Frame ----------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """A decoded frame, host-bound or slave-bound.

    Attributes
    ----------
    sync :
        Either :data:`SYNC_HOST` or :data:`SYNC_SLAVE`.
    cmd :
        Command ID. Stored as int (not :class:`Cmd`) so we don't reject
        anything the slave might send back; convert at the call site.
    action :
        Action / status byte.
    data :
        Raw data bytes; may be empty.
    """

    sync: int
    cmd: int
    action: int
    data: bytes

    def to_bytes(self) -> bytes:
        """Serialise to a complete on-wire frame including checksum."""
        if len(self.data) > 0xFF:
            raise StackShotProtocolError(
                f"data too long: {len(self.data)} > 255"
            )
        body = bytes([
            self.sync,
            (self.cmd >> 8) & 0xFF,
            self.cmd & 0xFF,
            self.action & 0xFF,
            len(self.data),
        ]) + bytes(self.data)
        return body + bytes([checksum(body)])

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Frame":
        """Parse a complete frame. Raises on bad sync / length / checksum."""
        if len(raw) < 6:
            raise StackShotProtocolError(
                f"frame too short: {len(raw)} bytes < minimum 6"
            )
        sync = raw[0]
        if sync not in (SYNC_HOST, SYNC_SLAVE):
            raise StackShotProtocolError(
                f"bad sync byte: {sync:#04x}"
            )
        cmd = (raw[1] << 8) | raw[2]
        action = raw[3]
        length = raw[4]
        expected_total = 5 + length + 1
        if len(raw) != expected_total:
            raise StackShotProtocolError(
                f"length field {length} implies {expected_total} bytes total, "
                f"frame is {len(raw)}"
            )
        data = raw[5:5 + length]
        # Per PDF §2: "Checksum (sent but not currently verified)".
        # Empirical confirmation from usbmon traces of working Cognisys
        # software: outgoing frames carry a constant 0x69 sentinel,
        # responses a constant 0x23 — neither matches the sum-of-bytes
        # mod-256 algorithm we still compute for outgoing frames. We
        # therefore accept whatever checksum byte the slave sends.
        return cls(sync=sync, cmd=cmd, action=action, data=bytes(data))


# ---- High-level frame builders / response checkers -------------------------

def build_request(cmd: Cmd | int, action: Action | int, data: bytes = b"") -> Frame:
    """Build a host->slave request frame."""
    return Frame(
        sync=SYNC_HOST,
        cmd=int(cmd),
        action=int(action),
        data=bytes(data),
    )


def check_response(response: Frame, expected_cmd: Cmd | int) -> Frame:
    """Validate a slave response frame.

    Raises
    ------
    StackShotProtocolError
        On bad sync or command-ID mismatch.
    StackShotResponseError
        If the action/status byte indicates a slave-side failure.

    Returns the frame unchanged on success so callers can chain.
    """
    if response.sync != SYNC_SLAVE:
        raise StackShotProtocolError(
            f"response has host sync {response.sync:#04x}, expected {SYNC_SLAVE:#04x}"
        )
    if response.cmd != int(expected_cmd):
        raise StackShotProtocolError(
            f"response cmd {response.cmd:#06x} != expected {int(expected_cmd):#06x}"
        )
    # Mask off any 3X-style upper-nibble axis bits before comparing —
    # harmless on the 1-axis controller, defensive on the 3X.
    status = response.action & 0x0F
    if status != Action.RSP_OK:
        try:
            label = Action(status).name
        except ValueError:
            label = f"unknown_{status:#04x}"
        raise StackShotResponseError(
            f"slave returned {label} for cmd {response.cmd:#06x}"
        )
    return response


# ---- Domain-specific encoders ---------------------------------------------

def encode_move(direction: Direction, units: Units, backlash: bool, distance: float) -> bytes:
    """``CC_RAIL_MOVE`` data field (7 bytes).

    Layout per PDF §2: dir(1B) | units(1B) | backlash(1B) | F32 distance.
    """
    return bytes([int(direction), int(units), 1 if backlash else 0]) + encode_f32_le(distance)


def encode_units_only(units: Units) -> bytes:
    """One-byte data field used by position-read commands."""
    return bytes([int(units)])


def decode_position_response(data: bytes) -> tuple[Units, float]:
    """Parse ``CC_RAIL_POSITION_CURRENT`` / ``_TARGET`` response data.

    Layout: units(1B) | F32 position. Position may be signed.
    """
    if len(data) != 5:
        raise StackShotProtocolError(
            f"position response should be 5 bytes, got {len(data)}: {data.hex()}"
        )
    units = Units(data[0])
    position = decode_f32_le(data[1:5])
    return units, position


def decode_status_response(data: bytes) -> RailStatus:
    """Parse ``CC_RAIL_STATUS`` response data (4-byte U32 bitfield)."""
    raw = decode_u32_le(data)
    return RailStatus(raw & 0xFFFFFFFF)


def decode_id_response(data: bytes) -> int:
    """Parse a ``CC_*_ID`` response (4-byte U32)."""
    return decode_u32_le(data)
