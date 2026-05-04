"""Tests for stackshot.transport.SerialTransport.

The framing logic (encoding / decoding individual frames) lives in
:mod:`stackshot.protocol` and is covered by ``test_protocol.py``.
This file only tests how the transport drives a (mocked) serial port:
how it splits the read into header + body, how it surfaces timeouts,
and that the inter-frame delay is observed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from errors import StackShotProtocolError
from stackshot.protocol import (
    Action,
    Cmd,
    Frame,
    SYNC_HOST,
    SYNC_SLAVE,
    build_request,
    encode_u32_le,
)
from stackshot.transport import SerialTransport


def _ok_frame_bytes(cmd: Cmd, data: bytes = b"") -> bytes:
    return Frame(sync=SYNC_SLAVE, cmd=int(cmd), action=Action.RSP_OK, data=data).to_bytes()


@pytest.fixture
def fake_serial():
    """A MagicMock that mimics serial.Serial. The script() helper sets
    up the next read sequence."""

    class FakeSerial:
        def __init__(self):
            self._reads: list[bytes] = []
            self.written: bytes = b""
            self.flushed = False

        def script(self, *chunks: bytes) -> None:
            self._reads.extend(chunks)

        def read(self, n: int) -> bytes:
            if not self._reads:
                return b""
            head = self._reads[0]
            if len(head) <= n:
                self._reads.pop(0)
                return head
            self._reads[0] = head[n:]
            return head[:n]

        def write(self, data: bytes) -> int:
            self.written += data
            return len(data)

        def flush(self) -> None:
            self.flushed = True

        def reset_input_buffer(self) -> None:
            self._reads.clear()

        def reset_output_buffer(self) -> None:
            self.written = b""

        def close(self) -> None:
            pass

    return FakeSerial()


@pytest.fixture
def transport(fake_serial):
    sleeps: list[float] = []

    def _sleep(s: float) -> None:
        sleeps.append(s)

    tx = SerialTransport(
        port="/dev/null",
        timeout_s=1.0,
        interframe_s=0.002,
        sleep=_sleep,
    )
    tx._ser = fake_serial  # bypass open() — already injected
    tx.sleeps = sleeps  # type: ignore[attr-defined]
    return tx


class TestRequestRoundTrip:
    def test_writes_frame_and_reads_response(self, transport, fake_serial) -> None:
        request = build_request(Cmd.RAIL_STATUS, Action.READ)
        response = _ok_frame_bytes(Cmd.RAIL_STATUS, encode_u32_le(0))

        # Serve the response in the natural pattern: header first, then
        # body+checksum. The transport must handle either chunking.
        fake_serial.script(response[:5], response[5:])

        result = transport.request(request, Cmd.RAIL_STATUS)
        assert result.cmd == int(Cmd.RAIL_STATUS)
        assert result.data == encode_u32_le(0)

        assert fake_serial.written == request.to_bytes()
        assert fake_serial.flushed is True
        # Inter-frame delay observed exactly once.
        assert transport.sleeps == [0.002]  # type: ignore[attr-defined]

    def test_handles_response_arriving_one_byte_at_a_time(
        self, transport, fake_serial
    ) -> None:
        request = build_request(Cmd.RAIL_STATUS, Action.READ)
        response = _ok_frame_bytes(Cmd.RAIL_STATUS, encode_u32_le(7))
        # Stream every byte separately.
        fake_serial.script(*[bytes([b]) for b in response])

        result = transport.request(request, Cmd.RAIL_STATUS)
        assert result.data == encode_u32_le(7)


class TestTimeout:
    def test_timeout_during_header_raises(self, transport, fake_serial) -> None:
        # No bytes scripted — read() returns b'' which our fake treats
        # as a timeout, exactly as serial.Serial does.
        request = build_request(Cmd.RAIL_STATUS, Action.READ)
        with pytest.raises(StackShotProtocolError, match="timed out"):
            transport.request(request, Cmd.RAIL_STATUS)

    def test_timeout_during_body_raises(self, transport, fake_serial) -> None:
        # Header arrives fine, body never does.
        response = _ok_frame_bytes(Cmd.RAIL_STATUS, encode_u32_le(0))
        fake_serial.script(response[:5])
        request = build_request(Cmd.RAIL_STATUS, Action.READ)
        with pytest.raises(StackShotProtocolError, match="timed out"):
            transport.request(request, Cmd.RAIL_STATUS)


class TestRequestWithoutOpen:
    def test_raises_runtime_error(self) -> None:
        tx = SerialTransport(port="/dev/null")
        # _ser is None until open() is called.
        with pytest.raises(RuntimeError, match="open"):
            tx.request(build_request(Cmd.RAIL_STATUS, Action.READ), Cmd.RAIL_STATUS)


class TestOpenIsIdempotent:
    @patch("stackshot.transport.serial.Serial")
    def test_double_open_doesnt_reopen(self, serial_cls: MagicMock) -> None:
        instance = MagicMock()
        serial_cls.return_value = instance
        tx = SerialTransport(port="/dev/null")
        tx.open()
        tx.open()
        assert serial_cls.call_count == 1
        instance.reset_input_buffer.assert_called_once()
