"""Interactive rail-jog UI.

Used at session start to walk the rail to the start and end of the
range a given lens configuration should cover.

The actual termios cbreak handling and the print-to-stdout status
line are isolated behind two injectable callables — :data:`read_key`
and :data:`write_status` — so the jog *loop* is pure logic and tests
drive it without a TTY.
"""

from __future__ import annotations

import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from errors import SessionAborted
from stackshot.controller import StackShotController
from stackshot.protocol import Direction


# ---- Key map ---------------------------------------------------------------

# Distances are mm; direction is encoded in the sign of the step in the
# loop (negative = BACK, positive = FORWARD). Tests rely on these
# constants matching :mod:`config`.
@dataclass(frozen=True)
class JogKeyMap:
    fine_mm: float = 0.05
    coarse_mm: float = 1.0
    set_key: str = "s"
    abort_key: str = "q"


def _step_for_key(key: str, keys: JogKeyMap) -> float | None:
    """Return the relative move (mm, signed) for a movement key, or
    ``None`` if the key isn't a movement key.

    Lowercase = coarse (1 mm), uppercase = fine (0.05 mm).
    ``a`` = back, ``d`` = forward.
    """
    if key == "a":
        return -keys.coarse_mm
    if key == "d":
        return +keys.coarse_mm
    if key == "A":
        return -keys.fine_mm
    if key == "D":
        return +keys.fine_mm
    return None


# Hot-key for "release to front panel" inside :func:`jog_loop`.
PANEL_RELEASE_KEY = " "


# ---- Pure jog loop ---------------------------------------------------------


def jog_loop(
    controller: StackShotController,
    *,
    read_key: Callable[[], str],
    write_status: Callable[[float, str], None],
    keys: JogKeyMap | None = None,
    set_key: str = "s",
    label: str = "START",
) -> float:
    """Run a single jog session and return the rail's final position.

    Parameters
    ----------
    controller :
        Object with :meth:`get_position`, :meth:`move_and_wait`. Movement
        is always ``backlash=True`` since the user reverses freely.
    read_key :
        Returns a single character on each call. Blocks if no key.
    write_status :
        Receives ``(position_mm, label)`` after every position change so
        the UI can redraw.
    keys :
        Movement step sizes and the abort key. Defaults to
        :class:`JogKeyMap()`.
    set_key :
        Key that ends the jog and accepts the current position. ``"s"``
        for set-START (which also zeros the rail), ``"e"`` for set-END.
    label :
        Free-text label passed through to ``write_status`` so the user
        sees ``START`` vs ``END`` in the prompt.

    Returns
    -------
    float
        The position at the moment ``set_key`` was pressed. Always in
        millimetres. After a START set, this is always 0.0 because the
        rail was just zeroed.

    Raises
    ------
    SessionAborted
        If the user pressed the abort key.
    """
    keys = keys or JogKeyMap()

    position = controller.get_position()
    write_status(position, label)
    while True:
        key = read_key()
        if key == keys.abort_key:
            raise SessionAborted(f"jog aborted at {label}")
        if key == set_key:
            if set_key == "s":
                controller.zero_position()
                position = controller.get_position()
                write_status(position, f"{label} set (zeroed)")
            else:
                write_status(position, f"{label} set")
            return position
        if key == PANEL_RELEASE_KEY:
            position = _release_to_panel(
                controller=controller,
                read_key=read_key,
                write_status=write_status,
                label=label,
            )
            continue
        step = _step_for_key(key, keys)
        if step is None:
            # Unknown key — ignore. Tests can assert nothing happened.
            continue
        direction = Direction.FORWARD if step > 0 else Direction.BACKWARD
        controller.move_and_wait(direction, abs(step), backlash=True)
        position = controller.get_position()
        write_status(position, label)


def _release_to_panel(
    *,
    controller: StackShotController,
    read_key: Callable[[], str],
    write_status: Callable[[float, str], None],
    label: str,
) -> float:
    """Drop USB control so the user can drive the rail with the
    StackShot's physical buttons. Returns the position the user
    left the rail at after pressing ENTER (returns to keyboard
    control).

    Falls back to a clear "not supported" message when the underlying
    transport can't toggle bit-mode (e.g.
    :class:`stackshot.transport.SerialTransport`).
    """
    try:
        controller.release_to_panel()
    except NotImplementedError as exc:
        write_status(controller.get_position(), f"{label} (panel release not supported: {exc})")
        return controller.get_position()
    write_status(
        controller.get_position(),
        f"{label} | PANEL — use rail buttons; ENTER to resume keyboard",
    )
    while True:
        k = read_key()
        if k in ("\r", "\n"):
            break
    controller.take_from_panel()
    position = controller.get_position()
    write_status(position, label)
    return position


# ---- Production wiring: termios + stdout -----------------------------------


@contextmanager
def cbreak_stdin():
    """Put stdin in cbreak mode for the duration of the with-block.

    Restores the original termios attributes on exit even if the body
    raises. Safe to call when stdin is not a TTY (no-op).
    """
    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _stdin_read_key() -> str:
    """Block until one character is available on stdin and return it."""
    return sys.stdin.read(1)


def _stdout_write_status(position_mm: float, label: str) -> None:
    sys.stdout.write(f"\r[{label}] position = {position_mm:+8.3f} mm   ")
    sys.stdout.flush()


def jog_interactive(
    controller: StackShotController,
    *,
    label: str,
    set_key: str,
    keys: JogKeyMap | None = None,
) -> float:
    """Production wrapper: cbreak-mode stdin + stdout status line."""
    with cbreak_stdin():
        try:
            position = jog_loop(
                controller,
                read_key=_stdin_read_key,
                write_status=_stdout_write_status,
                keys=keys,
                set_key=set_key,
                label=label,
            )
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()
    return position


@dataclass(frozen=True)
class JogResult:
    end_mm: float


def jog_start_and_end(
    controller: StackShotController,
    *,
    keys: JogKeyMap | None = None,
) -> JogResult:
    """Convenience: walk the user through the START then END jog.

    On START the rail is zeroed (so further positions are relative to
    the chosen start). The END value returned is in mm relative to
    that origin.
    """
    prompt = (
        "  a/d=coarse(1mm)  A/D=fine(0.05mm)  SPACE=front-panel  "
        "{set}=set {label}  q=abort"
    )
    print("Jog rail to START.")
    print(prompt.format(set="s", label="START"))
    jog_interactive(controller, label="START", set_key="s", keys=keys)
    print("\nJog rail to END.")
    print(prompt.format(set="e", label="END"))
    end_mm = jog_interactive(controller, label="END", set_key="e", keys=keys)
    return JogResult(end_mm=end_mm)


# ---- CLI for rig smoke-test -----------------------------------------------

def _cli() -> None:
    import argparse

    from stackshot.controller import open_ftdi, open_serial

    parser = argparse.ArgumentParser(description="StackShot jog session")
    parser.add_argument("--serial", default=None, metavar="PORT",
                        help="legacy: open via pyserial at this device path")
    args = parser.parse_args()

    if args.serial:
        controller = open_serial(args.serial)
    else:
        controller = open_ftdi()
    try:
        result = jog_start_and_end(controller)
        print(f"START = 0.000 mm")
        print(f"END   = {result.end_mm:+.3f} mm")
    finally:
        controller.close()


if __name__ == "__main__":
    _cli()
