"""Tests for stackshot.protocol — frame codec and helpers.

The hand-computed examples come straight from the protocol PDF (§2):

* ``CC_RAIL_POSITION_CURRENT`` read frame for the X axis is documented
  as ``0x55 10 02 00 00 CK`` — i.e. cmd 0x1002, action READ, length 0.
  PDF leaves the checksum as ``CK`` placeholder; we recompute it.
* The float ``12345678.0`` is documented as encoding to LSB-first bytes
  ``0x4e 0x61 0x3c 0x4b``.
* The U32 example ``0x12345678`` encodes to ``0x78 0x56 0x34 0x12``.
"""

from __future__ import annotations

import pytest

from errors import StackShotProtocolError, StackShotResponseError
from stackshot.protocol import (
    Action,
    Cmd,
    Direction,
    Frame,
    RailStatus,
    SYNC_HOST,
    SYNC_SLAVE,
    Units,
    build_request,
    check_response,
    checksum,
    decode_f32_le,
    decode_id_response,
    decode_position_response,
    decode_status_response,
    decode_u32_le,
    encode_f32_le,
    encode_move,
    encode_u32_le,
    encode_units_only,
)


# ---- Low-level helpers -----------------------------------------------------


class TestU32Codec:
    def test_pdf_example_encodes_lsb_first(self) -> None:
        # PDF: "To send the value 0x12345678 the byte order would be
        # 0x78 0x56 0x34 0x12"
        assert encode_u32_le(0x12345678) == bytes([0x78, 0x56, 0x34, 0x12])

    def test_round_trip(self) -> None:
        for value in (0, 1, 0xFF, 0x100, 0xFFFFFFFF):
            assert decode_u32_le(encode_u32_le(value)) == value

    def test_decode_rejects_wrong_length(self) -> None:
        with pytest.raises(StackShotProtocolError):
            decode_u32_le(b"\x00\x00\x00")

    def test_encode_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            encode_u32_le(-1)
        with pytest.raises(ValueError):
            encode_u32_le(0x1_0000_0000)


class TestF32Codec:
    def test_pdf_example(self) -> None:
        # PDF: 12345678.0 -> 0x4b3c614e on the wire as 0x4e 0x61 0x3c 0x4b.
        assert encode_f32_le(12345678.0) == bytes([0x4E, 0x61, 0x3C, 0x4B])

    def test_round_trip(self) -> None:
        for value in (-12.5, 0.0, 1.0, 100.0, -3.5):
            decoded = decode_f32_le(encode_f32_le(value))
            assert decoded == pytest.approx(value)

    def test_decode_rejects_wrong_length(self) -> None:
        with pytest.raises(StackShotProtocolError):
            decode_f32_le(b"\x00\x00\x00")


class TestChecksum:
    def test_sum_mod_256(self) -> None:
        assert checksum(b"\x01\x02\x03") == 6
        assert checksum(b"\xff\x01") == 0  # wraps cleanly

    def test_empty_is_zero(self) -> None:
        assert checksum(b"") == 0


# ---- Frame round-trip ------------------------------------------------------


class TestFrameToBytes:
    def test_pdf_read_position_x(self) -> None:
        # PDF example: read X-axis position is 0x55 10 02 00 00 CK.
        # cmd=0x1002, action=READ, no data.
        frame = build_request(Cmd.RAIL_POSITION_CURRENT, Action.READ)
        raw = frame.to_bytes()
        # Body before checksum.
        assert raw[:5] == bytes([0x55, 0x10, 0x02, 0x00, 0x00])
        # Checksum: 0x55 + 0x10 + 0x02 = 0x67.
        assert raw[5] == 0x67
        assert len(raw) == 6

    def test_with_data_field(self) -> None:
        # READ position with metric units in cmd-data (1 byte payload).
        frame = build_request(
            Cmd.RAIL_POSITION_CURRENT,
            Action.READ,
            encode_units_only(Units.METRIC),
        )
        raw = frame.to_bytes()
        assert raw[:5] == bytes([0x55, 0x10, 0x02, 0x00, 0x01])
        assert raw[5] == 0x01  # data
        # checksum = 0x55 + 0x10 + 0x02 + 0x00 + 0x01 + 0x01 = 0x69.
        assert raw[-1] == 0x69

    def test_data_too_long(self) -> None:
        with pytest.raises(StackShotProtocolError):
            Frame(SYNC_HOST, 0x1000, 0x01, b"\x00" * 256).to_bytes()


class TestFrameFromBytes:
    def test_round_trip(self) -> None:
        frame = build_request(
            Cmd.RAIL_MOVE,
            Action.WRITE,
            encode_move(Direction.FORWARD, Units.METRIC, True, 1.0),
        )
        round_tripped = Frame.from_bytes(frame.to_bytes())
        assert round_tripped == frame

    def test_rejects_bad_sync(self) -> None:
        # A frame body that is otherwise valid, but with sync = 0x42.
        body = bytes([0x42, 0x10, 0x02, 0x00, 0x00])
        raw = body + bytes([checksum(body)])
        with pytest.raises(StackShotProtocolError, match="sync"):
            Frame.from_bytes(raw)

    def test_rejects_bad_checksum(self) -> None:
        body = bytes([0x55, 0x10, 0x02, 0x00, 0x00])
        raw = body + bytes([(checksum(body) + 1) & 0xFF])
        with pytest.raises(StackShotProtocolError, match="checksum"):
            Frame.from_bytes(raw)

    def test_rejects_length_mismatch(self) -> None:
        # Length field claims 4 bytes of data but only 2 follow.
        body = bytes([0x55, 0x10, 0x02, 0x00, 0x04, 0x00, 0x00])
        raw = body + bytes([checksum(body)])
        with pytest.raises(StackShotProtocolError, match="length"):
            Frame.from_bytes(raw)

    def test_rejects_too_short(self) -> None:
        with pytest.raises(StackShotProtocolError, match="too short"):
            Frame.from_bytes(b"\x55\x10")


# ---- Response checking -----------------------------------------------------


class TestCheckResponse:
    def _response(self, cmd: int, action: int, data: bytes = b"") -> Frame:
        return Frame(sync=SYNC_SLAVE, cmd=cmd, action=action, data=data)

    def test_ok_response_passes(self) -> None:
        rsp = self._response(int(Cmd.RAIL_STATUS), Action.RSP_OK, b"\x00\x00\x00\x00")
        result = check_response(rsp, Cmd.RAIL_STATUS)
        assert result is rsp

    def test_host_sync_rejected(self) -> None:
        rsp = Frame(sync=SYNC_HOST, cmd=int(Cmd.RAIL_STATUS), action=Action.RSP_OK, data=b"")
        with pytest.raises(StackShotProtocolError, match="sync"):
            check_response(rsp, Cmd.RAIL_STATUS)

    def test_cmd_mismatch_rejected(self) -> None:
        rsp = self._response(int(Cmd.RAIL_STOP), Action.RSP_OK)
        with pytest.raises(StackShotProtocolError, match="cmd"):
            check_response(rsp, Cmd.RAIL_STATUS)

    @pytest.mark.parametrize(
        "action,label",
        [
            (Action.BAD_PARAM, "BAD_PARAM"),
            (Action.UNSUPPORTED_CMD, "UNSUPPORTED_CMD"),
            (Action.BUSY, "BUSY"),
            (Action.FAILED, "FAILED"),
        ],
    )
    def test_failure_codes_raise_response_error(self, action: int, label: str) -> None:
        rsp = self._response(int(Cmd.RAIL_STATUS), action)
        with pytest.raises(StackShotResponseError, match=label):
            check_response(rsp, Cmd.RAIL_STATUS)

    def test_3x_axis_or_in_upper_nibble_is_masked(self) -> None:
        # PDF §2 says 3X controllers may OR axis bits into bits 4-5 of
        # the action byte. The 1-axis classic doesn't do this, but if a
        # firmware ever did, RSP_OK must still be recognised after we
        # mask.
        rsp = self._response(int(Cmd.RAIL_STATUS), 0x10 | Action.RSP_OK)
        result = check_response(rsp, Cmd.RAIL_STATUS)
        assert result is rsp


# ---- Domain encoders / decoders --------------------------------------------


class TestMoveEncoding:
    def test_forward_metric_with_backlash(self) -> None:
        data = encode_move(Direction.FORWARD, Units.METRIC, True, 1.0)
        # 7 bytes total: dir, units, backlash, then 4-byte F32.
        assert len(data) == 7
        assert data[0] == 0  # FWD
        assert data[1] == 1  # METRIC
        assert data[2] == 1  # backlash true
        assert decode_f32_le(data[3:]) == pytest.approx(1.0)

    def test_backward_no_backlash(self) -> None:
        data = encode_move(Direction.BACKWARD, Units.METRIC, False, 5.5)
        assert data[0] == 1
        assert data[2] == 0
        assert decode_f32_le(data[3:]) == pytest.approx(5.5)


class TestPositionResponse:
    def test_decode(self) -> None:
        data = bytes([Units.METRIC]) + encode_f32_le(12.5)
        units, position = decode_position_response(data)
        assert units is Units.METRIC
        assert position == pytest.approx(12.5)

    def test_signed_position(self) -> None:
        data = bytes([Units.METRIC]) + encode_f32_le(-3.25)
        _, position = decode_position_response(data)
        assert position == pytest.approx(-3.25)

    def test_wrong_length_rejected(self) -> None:
        with pytest.raises(StackShotProtocolError):
            decode_position_response(b"\x01\x00\x00")


class TestStatusResponse:
    def test_idle(self) -> None:
        assert decode_status_response(b"\x00\x00\x00\x00") == RailStatus.IDLE

    def test_moving(self) -> None:
        result = decode_status_response(b"\x01\x00\x00\x00")
        assert RailStatus.MOVING in result
        assert RailStatus.USER_ABORTED not in result

    def test_user_aborted_set(self) -> None:
        result = decode_status_response(b"\x04\x00\x00\x00")
        assert RailStatus.USER_ABORTED in result

    def test_combined_flags(self) -> None:
        # Moving + shutter active + user aborted = 0x07.
        result = decode_status_response(b"\x07\x00\x00\x00")
        assert RailStatus.MOVING in result
        assert RailStatus.SHUTTER in result
        assert RailStatus.USER_ABORTED in result


class TestIdResponse:
    def test_decode(self) -> None:
        # Some software ID like 0x00010002.
        assert decode_id_response(b"\x02\x00\x01\x00") == 0x00010002
