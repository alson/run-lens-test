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
    """Build a read_key callable for tests. Compatible with both the
    blocking signature (``read_key()``) and the non-blocking variant
    (``read_key(0.0)``) the jog loop uses to drain queued keys."""
    queue = deque(chars)

    def _read(timeout=None):
        if queue:
            return queue.popleft()
        # Empty queue: blocking would hang in production; tests should
        # never reach this. Non-blocking returns None (drained).
        if timeout is None:
            raise IndexError("scripted reader exhausted")
        return None

    return _read


# ---- _step_for_key --------------------------------------------------------


class TestStepForKey:
    """Lowercase = coarse (1 mm), uppercase = fine (0.05 mm)."""

    def test_a_is_back_coarse(self, keys: JogKeyMap) -> None:
        assert _step_for_key("a", keys) == pytest.approx(-1.0)

    def test_d_is_forward_coarse(self, keys: JogKeyMap) -> None:
        assert _step_for_key("d", keys) == pytest.approx(+1.0)

    def test_uppercase_is_fine(self, keys: JogKeyMap) -> None:
        assert _step_for_key("A", keys) == pytest.approx(-0.1)
        assert _step_for_key("D", keys) == pytest.approx(+0.1)

    def test_unknown_key_is_none(self, keys: JogKeyMap) -> None:
        assert _step_for_key("x", keys) is None
        assert _step_for_key("\x1b", keys) is None  # ESC
        # SPACE is a known key (panel-release toggle), not a movement.
        assert _step_for_key(" ", keys) is None


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
        # d (forward coarse) → A (back fine) → e (end)
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "A", "e"),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        assert controller.move_and_wait.call_args_list == [
            call(Direction.FORWARD, pytest.approx(1.0), backlash=True),
            call(Direction.BACKWARD, pytest.approx(0.1), backlash=True),
        ]
        # Net position: +1.0 - 0.1 = +0.9 mm.
        assert controller._pos == pytest.approx(+0.9)

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
        # Use distinct keys so the drain doesn't consolidate
        # consecutive same-direction presses into one move.
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "A", "s"),
            write_status=write_status,
            keys=keys,
            set_key="s",
        )
        # Initial + 2 moves + final-after-zero = 4.
        assert write_status.call_count == 4


class TestJogLoopDrainAfterMove:
    """A held key generates ~30 Hz auto-repeat keystrokes that queue
    in stdin while the rail is busy with a previous move. Without
    drain, those queued moves fire after the user releases — running
    into an end-stop. The loop drains them after each move."""

    def test_consecutive_same_direction_keys_collapse_to_one_move(
        self, controller, keys: JogKeyMap
    ) -> None:
        # Scripted as if the user briefly held 'd' (4 auto-repeats)
        # then pressed 's' to set START.
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "d", "d", "d", "s"),
            write_status=MagicMock(),
            keys=keys,
            set_key="s",
        )
        # Only ONE move — the first 'd'. The other three were drained
        # and discarded.
        assert controller.move_and_wait.call_count == 1
        # Final position is 0.0 (after zero on 's'); rail moved +1 mm
        # before zero.

    def test_drain_keeps_first_different_key_as_pending(
        self, controller, keys: JogKeyMap
    ) -> None:
        # Hold d (3 keys), then press 'A' (a fine backward step).
        # Expect: one coarse forward move, then one fine backward move.
        jog_loop(
            controller,
            read_key=_scripted_reader("d", "d", "d", "A", "e"),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        assert controller.move_and_wait.call_args_list == [
            call(Direction.FORWARD, pytest.approx(1.0), backlash=True),
            call(Direction.BACKWARD, pytest.approx(0.1), backlash=True),
        ]
        # Net: +1.0 - 0.1 = +0.9 mm.
        assert controller._pos == pytest.approx(+0.9)

    def test_drain_works_with_legacy_no_timeout_reader(
        self, controller, keys: JogKeyMap
    ) -> None:
        """If a test passes a read_key callable that doesn't accept
        ``timeout``, drain silently no-ops — the loop falls back to
        the pre-drain behaviour."""
        # Build a reader that ONLY accepts no-arg calls (TypeError on
        # any positional arg). This mimics older test fixtures.
        queue = deque(["d", "d", "s"])

        def _legacy_read():
            return queue.popleft()

        jog_loop(
            controller,
            read_key=_legacy_read,
            write_status=MagicMock(),
            keys=keys,
            set_key="s",
        )
        # Drain TypeErrored, so both 'd's were processed individually.
        assert controller.move_and_wait.call_count == 2


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


class TestJogLoopPanelRelease:
    """SPACE drops control to the StackShot's front panel; ENTER
    returns. Position read after re-acquire reflects whatever the
    user did with the rail's physical buttons in between."""

    def test_space_releases_then_enter_reacquires(
        self, controller, keys: JogKeyMap
    ) -> None:
        # Simulate the user moving the rail forward 5 mm via the
        # physical buttons while we were released.
        def _take_from_panel():
            controller._pos = 5.0

        controller.take_from_panel.side_effect = _take_from_panel
        write_status = MagicMock()
        jog_loop(
            controller,
            read_key=_scripted_reader(" ", "\r", "e"),
            write_status=write_status,
            keys=keys,
            set_key="e",
            label="END",
        )
        controller.release_to_panel.assert_called_once()
        controller.take_from_panel.assert_called_once()
        # After re-acquire, position is the value the rail was left
        # at (5.0 mm). The "END set" status row reflects that.
        last = write_status.call_args_list[-1]
        assert last.args[0] == pytest.approx(5.0)

    def test_panel_release_falls_back_when_unsupported(
        self, controller, keys: JogKeyMap
    ) -> None:
        # SerialTransport raises NotImplementedError; the jog loop
        # must surface that gracefully and not crash.
        controller.release_to_panel.side_effect = NotImplementedError("no")
        write_status = MagicMock()
        jog_loop(
            controller,
            read_key=_scripted_reader(" ", "e"),
            write_status=write_status,
            keys=keys,
            set_key="e",
        )
        # take_from_panel never gets called (we bailed early).
        controller.take_from_panel.assert_not_called()

    def test_lf_also_resumes_keyboard(
        self, controller, keys: JogKeyMap
    ) -> None:
        """`\\n` should be accepted as ENTER, not just `\\r`."""
        jog_loop(
            controller,
            read_key=_scripted_reader(" ", "\n", "e"),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.take_from_panel.assert_called_once()


class TestJogLoopNumericEntry:
    """``:`` opens a numeric-target entry mode; the user types an
    absolute mm position and ENTER moves the rail there."""

    def test_colon_then_typed_target_moves_rail(
        self, controller, keys: JogKeyMap,
    ) -> None:
        controller._pos = -1.0
        # Type ":-3.10<ENTER>e" → move to -3.10, then 'e' sets END.
        chars = list(":-3.10\re")
        write_status = MagicMock()
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=write_status,
            keys=keys,
            set_key="e",
        )
        # One absolute move from -1.0 to -3.10 → backward 2.10 mm.
        controller.move_and_wait.assert_called_once_with(
            Direction.BACKWARD, pytest.approx(2.10), backlash=True
        )
        assert result == pytest.approx(-3.10)

    def test_forward_target_uses_forward_direction(
        self, controller, keys: JogKeyMap,
    ) -> None:
        controller._pos = -2.0
        chars = list(":+0.5\re")
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.move_and_wait.assert_called_once_with(
            Direction.FORWARD, pytest.approx(2.5), backlash=True
        )
        assert result == pytest.approx(0.5)

    def test_empty_input_cancels(
        self, controller, keys: JogKeyMap,
    ) -> None:
        controller._pos = -2.0
        # ":<ENTER>e" — empty buffer → no move, then 'e' sets END.
        chars = list(":\re")
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.move_and_wait.assert_not_called()
        assert result == pytest.approx(-2.0)

    def test_escape_cancels(
        self, controller, keys: JogKeyMap,
    ) -> None:
        controller._pos = -2.0
        chars = list(":1.5\x1be")
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.move_and_wait.assert_not_called()
        assert result == pytest.approx(-2.0)

    def test_backspace_edits_buffer(
        self, controller, keys: JogKeyMap,
    ) -> None:
        controller._pos = 0.0
        # Type a typo digit then backspace it before continuing.
        # ":-31<bksp>.10<ENTER>e" → buffer becomes "-3.10".
        chars = [":", "-", "3", "1", "\x7f", ".", "1", "0", "\r", "e"]
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.move_and_wait.assert_called_once_with(
            Direction.BACKWARD, pytest.approx(3.10), backlash=True
        )
        assert result == pytest.approx(-3.10)

    def test_garbage_input_keeps_entry_open(
        self, controller, keys: JogKeyMap,
    ) -> None:
        # Typing letters then ENTER → "not a number"; entry stays
        # open. User then types a real number and submits.
        controller._pos = 0.0
        # ":abc<ENTER>" gets ignored chars (only NUMERIC_CHARS accepted),
        # so "abc" never makes it into the buffer; ENTER on empty buffer
        # cancels. Then "e" sets END at 0.0 unchanged.
        chars = [":", "a", "b", "c", "\r", "e"]
        result = jog_loop(
            controller,
            read_key=_scripted_reader(*chars),
            write_status=MagicMock(),
            keys=keys,
            set_key="e",
        )
        controller.move_and_wait.assert_not_called()
        assert result == pytest.approx(0.0)

    def test_status_line_echoes_buffer(
        self, controller, keys: JogKeyMap,
    ) -> None:
        # The status callback should see the partial buffer printed
        # while the user types so they can read what they're entering.
        controller._pos = 0.0
        write_status = MagicMock()
        jog_loop(
            controller,
            read_key=_scripted_reader(*list(":1.5\re")),
            write_status=write_status,
            keys=keys,
            set_key="e",
        )
        # Find any call where the message contains the buffer prefix.
        msgs = [c.args[1] for c in write_status.call_args_list]
        assert any("enter target mm:" in m for m in msgs)
        assert any("1.5" in m for m in msgs)
