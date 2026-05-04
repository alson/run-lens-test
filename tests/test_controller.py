"""Tests for stackshot.controller — exercises move-and-wait, ID
readers, abort handling, and timeout — all without a real serial port.

The fake transport scripts a queue of responses and records every
outgoing frame, so each test asserts both:
  * what bytes the controller sent
  * how it interpreted what came back
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from errors import (
    StackShotProtocolError,
    StackShotResponseError,
    StackShotTimeout,
    StackShotUserAborted,
)
from stackshot.controller import StackShotController, StackShotInfo
from stackshot.protocol import (
    Action,
    Cmd,
    Direction,
    Frame,
    RailStatus,
    SYNC_HOST,
    SYNC_SLAVE,
    Units,
    decode_position_response,
    encode_f32_le,
    encode_u32_le,
)


# ---- Fake transport --------------------------------------------------------


@dataclass
class _Sent:
    frame: Frame
    expected_cmd: Cmd


class FakeTransport:
    """Records sent frames and pops scripted responses.

    Each call to :meth:`request` pulls the next entry from
    ``responses``. If a response is an exception class or instance, it
    is raised instead of returned — this is used to simulate transport-
    layer failures.
    """

    def __init__(self) -> None:
        self.sent: list[_Sent] = []
        self.responses: deque[Frame | Exception | type[Exception]] = deque()
        self.closed = False

    def script(self, *frames: Frame | Exception | type[Exception]) -> None:
        self.responses.extend(frames)

    def request(self, frame: Frame, expected_cmd: Cmd) -> Frame:
        self.sent.append(_Sent(frame=frame, expected_cmd=expected_cmd))
        if not self.responses:
            raise AssertionError(
                f"FakeTransport got cmd {expected_cmd!r} but no scripted response"
            )
        nxt = self.responses.popleft()
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, type) and issubclass(nxt, Exception):
            raise nxt()
        return nxt

    def close(self) -> None:
        self.closed = True


def _ok_response(cmd: Cmd, data: bytes = b"") -> Frame:
    return Frame(sync=SYNC_SLAVE, cmd=int(cmd), action=Action.RSP_OK, data=data)


def _status_response(status: RailStatus) -> Frame:
    return _ok_response(Cmd.RAIL_STATUS, encode_u32_le(int(status)))


def _position_response(position_mm: float) -> Frame:
    return _ok_response(
        Cmd.RAIL_POSITION_CURRENT,
        bytes([Units.METRIC]) + encode_f32_le(position_mm),
    )


# ---- Fixtures --------------------------------------------------------------


class FakeClock:
    """Monotonic clock + sleep that the controller can drive forwards
    without real time passing."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def controller(transport: FakeTransport, clock: FakeClock) -> StackShotController:
    return StackShotController(
        transport,
        poll_interval_s=0.1,
        move_timeout_s=5.0,
        clock=clock.time,
        sleep=clock.sleep,
    )


# ---- Identification --------------------------------------------------------


class TestInfo:
    def test_reads_software_and_hardware_ids(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(
            _ok_response(Cmd.SOFTWARE_ID, encode_u32_le(0x00010002)),
            _ok_response(Cmd.HARDWARE_ID, encode_u32_le(0x00020003)),
        )
        info = controller.info()
        assert info == StackShotInfo(
            software_id=0x00010002, hardware_id=0x00020003
        )
        assert [s.expected_cmd for s in transport.sent] == [
            Cmd.SOFTWARE_ID,
            Cmd.HARDWARE_ID,
        ]
        # Both sent frames are READs with empty data.
        for s in transport.sent:
            assert s.frame.action == Action.READ
            assert s.frame.data == b""
            assert s.frame.sync == SYNC_HOST


# ---- Position reads --------------------------------------------------------


class TestGetPosition:
    def test_decodes_position(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_position_response(12.5))
        assert controller.get_position() == pytest.approx(12.5)

        sent = transport.sent[0]
        assert sent.expected_cmd is Cmd.RAIL_POSITION_CURRENT
        # cmd-data is one byte: the units enum.
        assert sent.frame.data == bytes([Units.METRIC])

    def test_signed_position(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_position_response(-3.25))
        assert controller.get_position() == pytest.approx(-3.25)


class TestZeroPosition:
    def test_sends_write_with_no_data(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_ok_response(Cmd.RAIL_POSITION_ZERO))
        controller.zero_position()
        sent = transport.sent[0]
        assert sent.frame.action == Action.WRITE
        assert sent.frame.cmd == int(Cmd.RAIL_POSITION_ZERO)
        assert sent.frame.data == b""

    def test_busy_response_propagates(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        # Per PDF §1.1.9, zero is rejected with BUSY while moving.
        transport.script(StackShotResponseError("BUSY"))
        with pytest.raises(StackShotResponseError):
            controller.zero_position()


# ---- Move + wait_until_idle ------------------------------------------------


class TestMove:
    def test_forward_metric_with_backlash(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_ok_response(Cmd.RAIL_MOVE))
        controller.move(Direction.FORWARD, 1.0, backlash=True)

        sent = transport.sent[0]
        assert sent.frame.cmd == int(Cmd.RAIL_MOVE)
        assert sent.frame.action == Action.WRITE
        assert len(sent.frame.data) == 7
        assert sent.frame.data[0] == Direction.FORWARD
        assert sent.frame.data[1] == Units.METRIC
        assert sent.frame.data[2] == 1  # backlash
        # F32 distance round-trip through the wire format.
        from stackshot.protocol import decode_f32_le
        assert decode_f32_le(sent.frame.data[3:]) == pytest.approx(1.0)

    def test_backward_no_backlash(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_ok_response(Cmd.RAIL_MOVE))
        controller.move(Direction.BACKWARD, 5.0, backlash=False)

        sent = transport.sent[0]
        assert sent.frame.data[0] == Direction.BACKWARD
        assert sent.frame.data[2] == 0


class TestWaitUntilIdle:
    def test_returns_immediately_when_already_idle(
        self,
        controller: StackShotController,
        transport: FakeTransport,
        clock: FakeClock,
    ) -> None:
        transport.script(_status_response(RailStatus.IDLE))
        controller.wait_until_idle()
        assert clock.sleep_calls == []  # never had to poll

    def test_polls_until_idle(
        self,
        controller: StackShotController,
        transport: FakeTransport,
        clock: FakeClock,
    ) -> None:
        transport.script(
            _status_response(RailStatus.MOVING),
            _status_response(RailStatus.MOVING),
            _status_response(RailStatus.IDLE),
        )
        controller.wait_until_idle()
        # Slept twice between the three status polls.
        assert clock.sleep_calls == [0.1, 0.1]

    def test_user_aborted_raises(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(
            _status_response(RailStatus.MOVING | RailStatus.USER_ABORTED),
        )
        with pytest.raises(StackShotUserAborted):
            controller.wait_until_idle()

    def test_timeout_issues_stop_and_raises(
        self,
        controller: StackShotController,
        transport: FakeTransport,
        clock: FakeClock,
    ) -> None:
        # Always-MOVING. Each poll sleeps 0.1s; timeout is 5s; so we'll
        # take 50 polls to exceed the deadline. Keep it simpler: make
        # timeout 0.05s so the second status poll is past the deadline.
        controller._move_timeout_s = 0.05
        # First status: MOVING. Second status: MOVING (but now past deadline).
        # Then the controller's STOP write needs a response.
        transport.script(
            _status_response(RailStatus.MOVING),
            _status_response(RailStatus.MOVING),
            _ok_response(Cmd.RAIL_STOP),
        )
        with pytest.raises(StackShotTimeout):
            controller.wait_until_idle()
        assert any(
            s.frame.cmd == int(Cmd.RAIL_STOP) for s in transport.sent
        )


class TestMoveAndWait:
    def test_send_then_poll_until_idle(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(
            _ok_response(Cmd.RAIL_MOVE),
            _status_response(RailStatus.MOVING),
            _status_response(RailStatus.IDLE),
        )
        controller.move_and_wait(Direction.FORWARD, 0.5)
        cmds = [s.expected_cmd for s in transport.sent]
        assert cmds == [Cmd.RAIL_MOVE, Cmd.RAIL_STATUS, Cmd.RAIL_STATUS]


# ---- Stop ------------------------------------------------------------------


class TestStop:
    def test_sends_stop_write(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        transport.script(_ok_response(Cmd.RAIL_STOP))
        controller.stop()
        sent = transport.sent[0]
        assert sent.frame.cmd == int(Cmd.RAIL_STOP)
        assert sent.frame.action == Action.WRITE
        assert sent.frame.data == b""


# ---- close passthrough -----------------------------------------------------


class TestClose:
    def test_close_propagates_to_transport(
        self, controller: StackShotController, transport: FakeTransport
    ) -> None:
        controller.close()
        assert transport.closed is True
