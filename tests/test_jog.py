"""Tests for stackshot.jog.jog_loop.

The loop calls a controller and a key reader; both are mocks so the
test asserts the exact sequence of moves and writes.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, call

import pytest

from errors import SessionAborted
from stackshot.jog import JogKeyMap, _step_for_key, jog_loop
from stackshot.protocol import Direction


@pytest.fixture
def keys() -> JogKeyMap:
    # Tighter step sizes than the default so test math is obvious.
    return JogKeyMap(fine_mm=0.1, coarse_mm=1.0)


@pytest.fixture
def controller():
    """A mock controller whose get_position increments per move call.

    move_and_wait records the args and bumps a position counter so
    successive get_position() calls show the rail "moving".
    """
    c = MagicMock()
    c._pos = 0.0

    def _move(direction, distance_mm, backlash=True):
        c._pos += distance_mm if direction == Direction.FORWARD else -distance_mm

    def _zero():
        c._pos = 0.0

    def _get_position():
        return c._pos

    c.move_and_wait.side_effect = _move
    c.zero_position.side_effect = _zero
    c.get_position.side_effect = _get_position
    return c


def _scripted_reader(*chars: str):
    queue = deque(chars)
    return lambda: queue.popleft()


# ---- _step_for_key --------------------------------------------------------


class TestStepForKey:
    def test_a_is_back_fine(self, keys: JogKeyMap) -> None:
        assert _step_for_key("a", keys) == pytest.approx(-0.1)

    def test_d_is_forward_fine(self, keys: JogKeyMap) -> None:
        assert _step_for_key("d", keys) == pytest.approx(+0.1)

    def test_uppercase_is_coarse(self, keys: JogKeyMap) -> None:
        assert _step_for_key("A", keys) == pytest.approx(-1.0)
        assert _step_for_key("D", keys) == pytest.approx(+1.0)

    def test_unknown_key_is_none(self, keys: JogKeyMap) -> None:
        assert _step_for_key("x", keys) is None
        assert _step_for_key("\x1b", keys) is None  # ESC


# ---- jog_loop -------------------------------------------------------------


class TestJogLoopMovement:
    def test_set_start_zeros_and_returns_zero(
        self, controller, keys: JogKeyMap
    ) -> None:
        controller._pos = 4.2
        write_status = MagicMock()
        # The user just hits 's' immediately.
        position = jog_loop(
            controller,
            read_key=_scripted_reader("s"),
            write_status=write_status,
            keys=keys,
            set_key="s",
            label="START",
        )
        assert position == 0.0
        controller.zero_position.assert_called_once()
        # Two status writes: initial 4.2, then 0 after zero.
        assert write_status.call_count == 2
        assert write_status.call_args_list[-1].args[0] == 0.0

    def test_set_end_returns_current_position_without_zero(
        self, controller, keys: JogKeyMap
    ) -> None:
        controller._pos = 12.0
        position = jog_loop(
            controller,
            read_key=_scripted_reader("e"),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
            label="END",
        )
        assert position == pytest.approx(12.0)
        controller.zero_position.assert_not_called()

    def test_movement_keys_call_move_and_wait_with_correct_direction(
        self, controller, keys: JogKeyMap
    ) -> None:
        # d (forward fine) → A (back coarse) → e (end)
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "A", "e"),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        assert controller.move_and_wait.call_args_list == [
            call(Direction.FORWARD, pytest.approx(0.1), backlash=True),
            call(Direction.BACKWARD, pytest.approx(1.0), backlash=True),
        ]
        # Net position: +0.1 - 1.0 = -0.9 mm.
        assert controller._pos == pytest.approx(-0.9)

    def test_unknown_keys_are_ignored(
        self, controller, keys: JogKeyMap
    ) -> None:
        jog_loop(
            controller,
            read_key=_scripted_reader("x", "?", "\n", "s"),
            write_status=MagicMock(),
            keys=keys,
            set_key="s",
        )
        # No moves issued.
        controller.move_and_wait.assert_not_called()

    def test_status_written_after_every_move(
        self, controller, keys: JogKeyMap
    ) -> None:
        write_status = MagicMock()
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "d", "s"),
            write_status=write_status,
            keys=keys,
            set_key="s",
        )
        # Initial + 2 moves + final-after-zero = 4.
        assert write_status.call_count == 4


class TestJogLoopAbort:
    def test_q_raises_session_aborted(self, controller, keys: JogKeyMap) -> None:
        with pytest.raises(SessionAborted, match="START"):
            jog_loop(
                controller,
                read_key=_scripted_reader("d", "d", "q"),
                write_status=MagicMock(),
                keys=keys,
                set_key="s",
                label="START",
            )

    def test_custom_abort_key(self, controller) -> None:
        keys = JogKeyMap(abort_key="x")
        with pytest.raises(SessionAborted):
            jog_loop(
                controller,
                read_key=_scripted_reader("x"),
                write_status=MagicMock(),
                keys=keys,
                set_key="s",
            )


class TestJogLoopInitialStatus:
    def test_writes_initial_position_before_first_keypress(
        self, controller, keys: JogKeyMap
    ) -> None:
        controller._pos = 7.0
        write_status = MagicMock()
        jog_loop(
            controller,
            read_key=_scripted_reader("e"),
            write_status=write_status,
            keys=keys,
            set_key="e",
            label="END",
        )
        # First call must be the initial position before any user input.
        first = write_status.call_args_list[0]
        assert first.args[0] == pytest.approx(7.0)
        assert first.args[1] == "END"
