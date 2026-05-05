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
    read_key: Callable[..., str | None],
    write_status: Callable[[float, str], None],
    keys: JogKeyMap | None = None,
    set_key: str = "s",
    label: str = "START",
    on_release: Callable[[], None] | None = None,
    on_reacquire: Callable[[], None] | None = None,
) -> float:
    """Run a single jog session and return the rail's final position.

    Parameters
    ----------
    controller :
        Object with :meth:`get_position`, :meth:`move_and_wait`. Movement
        is always ``backlash=True`` since the user reverses freely.
    read_key :
        Reads a single keystroke. Called as ``read_key()`` (or
        equivalently ``read_key(None)``) to block until input is
        available. Called as ``read_key(0.0)`` to return immediately
        with whatever is queued, or ``None`` if nothing is. The
        non-blocking form is used to drain queued same-direction
        keystrokes after each movement so a held key doesn't queue
        up post-release moves into an end-stop.
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
    on_release / on_reacquire :
        Optional callbacks invoked when the user toggles into / out
        of front-panel mode (SPACE / ENTER). Used by the production
        runner to also release the camera (close gphoto2) so the
        camera's LCD shows live view while the user is positioning
        the rail manually, then reacquire it after.

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
    pending_key: str | None = None
    while True:
        if pending_key is not None:
            key = pending_key
            pending_key = None
        else:
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
            position, panel_pending = _release_to_panel(
                controller=controller,
                read_key=read_key,
                write_status=write_status,
                label=label,
                on_release=on_release,
                on_reacquire=on_reacquire,
            )
            if panel_pending is not None:
                pending_key = panel_pending
            continue
        step = _step_for_key(key, keys)
        if step is None:
            # Unknown key — ignore. Tests can assert nothing happened.
            continue
        direction = Direction.FORWARD if step > 0 else Direction.BACKWARD
        controller.move_and_wait(direction, abs(step), backlash=True)
        position = controller.get_position()
        write_status(position, label)
        # Drain any queued same-direction keystrokes that piled up
        # while the move ran (held key + ~30 Hz auto-repeat). Discard
        # them so we don't run into an end-stop after release. Keep
        # the first different keystroke for the next iteration.
        pending_key = _drain_after_move(read_key, key)


def _drain_after_move(
    read_key: Callable[..., str | None],
    moved_key: str,
) -> str | None:
    """Pop everything currently queued. Discard same-direction
    keystrokes; return the first different keystroke (if any) so the
    main loop processes it next."""
    while True:
        try:
            extra = read_key(0.0)
        except TypeError:
            # read_key callable doesn't accept a timeout (legacy
            # signature in some tests). Skip drain silently.
            return None
        if extra is None:
            return None
        if extra == moved_key:
            continue
        return extra


def _release_to_panel(
    *,
    controller: StackShotController,
    read_key: Callable[[], str],
    write_status: Callable[[float, str], None],
    label: str,
    on_release: Callable[[], None] | None = None,
    on_reacquire: Callable[[], None] | None = None,
) -> tuple[float, str | None]:
    """Drop USB control so the user can drive the rail with the
    StackShot's physical buttons (and, via ``on_release``, regain
    direct LCD/AF control of the camera). Returns the position the
    user left the rail at after pressing ENTER.

    ``on_release`` runs after the rail is released (typically closes
    the gphoto2 session so the camera's LCD shows live view).
    ``on_reacquire`` runs before the rail is taken back from the
    panel (typically re-opens the gphoto2 session).

    Falls back to a clear "not supported" message when the underlying
    transport can't toggle bit-mode (e.g.
    :class:`stackshot.transport.SerialTransport`).
    """
    # Cache the position before releasing — once the rail's MCU is
    # in front-panel mode (after SET_BITMODE 0x20F0) it doesn't
    # reliably answer USB protocol queries, so any get_position()
    # call here would time out. Status writes during panel mode
    # therefore reuse this cached value (the user is moving the rail
    # via the StackShot's own LCD, so they don't need a live read
    # from us).
    cached_position = controller.get_position()
    try:
        controller.release_to_panel()
    except NotImplementedError as exc:
        write_status(cached_position, f"{label} (panel release not supported: {exc})")
        return cached_position, None
    if on_release is not None:
        try:
            on_release()
        except Exception as exc:                     # noqa: BLE001
            write_status(
                cached_position,
                f"{label} | PANEL (on_release callback failed: {exc})",
            )
    write_status(
        cached_position,
        f"{label} | PANEL — rail buttons + camera live view; ENTER to resume",
    )
    while True:
        k = read_key()
        if k in ("\r", "\n"):
            break
    # Immediate visual feedback BEFORE the slow reacquire (camera
    # re-init can take ~2 s). Without this the user thinks ENTER
    # didn't register and queues more keystrokes that leak into the
    # next prompt.
    write_status(
        cached_position,
        f"{label} | reacquiring camera + rail…",
    )
    if on_reacquire is not None:
        try:
            on_reacquire()
        except Exception as exc:                     # noqa: BLE001
            write_status(
                cached_position,
                f"{label} (on_reacquire callback failed: {exc})",
            )
    controller.take_from_panel()
    # Drain extra ENTERs the user queued during the slow reacquire,
    # but if any non-ENTER key shows up forward it to the caller as
    # a pending key (so e.g. `s`/`e`/`q` typed in eagerness still
    # get processed on the next iteration). Best-effort: skip if
    # read_key doesn't accept a timeout (legacy test fixtures).
    pending: str | None = None
    try:
        while True:
            k = read_key(0.0)
            if k is None:
                break
            if k in ("\r", "\n"):
                continue
            pending = k
            break
    except TypeError:
        pass
    position = controller.get_position()
    write_status(position, label)
    return position, pending


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


def _stdin_read_key(timeout: float | None = None) -> str | None:
    """Read one character from stdin via the raw fd.

    We deliberately bypass ``sys.stdin`` (a buffered ``TextIOWrapper``)
    because :func:`select.select` only sees the kernel's input
    buffer — anything Python has already pulled into its own buffer
    is invisible to ``select``. With ``sys.stdin.read(1)`` the drain
    in :func:`jog_loop` would return early while queued keystrokes
    sat in the Python buffer, then the next blocking read would
    find them and execute them as moves *after* the user released
    the key. Using :func:`os.read` on fd 0 directly avoids the
    Python buffer entirely; ``select`` and reads see the same data.

    ``timeout=None`` blocks until a character is available.
    ``timeout=0.0`` returns immediately with whatever is queued or
    ``None`` if nothing is. ``timeout>0`` waits up to that many
    seconds.
    """
    import os
    import select

    fd = sys.stdin.fileno()
    if timeout is not None:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
    try:
        b = os.read(fd, 1)
    except OSError:
        return None
    if not b:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _stdout_write_status(position_mm: float, label: str) -> None:
    # ``\x1b[K`` is ANSI "clear from cursor to end of line", so a
    # shorter status doesn't leave trailing chars from a longer
    # previous one (the panel-release status line is much longer
    # than the keyboard status line).
    sys.stdout.write(f"\r[{label}] position = {position_mm:+8.3f} mm\x1b[K")
    sys.stdout.flush()


def jog_interactive(
    controller: StackShotController,
    *,
    label: str,
    set_key: str,
    keys: JogKeyMap | None = None,
    on_release: Callable[[], None] | None = None,
    on_reacquire: Callable[[], None] | None = None,
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
                on_release=on_release,
                on_reacquire=on_reacquire,
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
    on_release: Callable[[], None] | None = None,
    on_reacquire: Callable[[], None] | None = None,
) -> JogResult:
    """Convenience: walk the user through the START then END jog.

    On START the rail is zeroed (so further positions are relative to
    the chosen start). The END value returned is in mm relative to
    that origin.

    Optional ``on_release`` / ``on_reacquire`` callbacks are forwarded
    into the SPACE / ENTER panel-release toggle. The production runner
    uses them to also close / reopen the gphoto2 camera session so
    the camera's LCD shows live view while the rail is in front-panel
    mode.
    """
    prompt = (
        "  a/d=coarse(1mm)  A/D=fine(0.05mm)  SPACE=front-panel+camera  "
        "{set}=set {label}  q=abort"
    )
    print("Jog rail to START.")
    print(prompt.format(set="s", label="START"))
    jog_interactive(
        controller, label="START", set_key="s", keys=keys,
        on_release=on_release, on_reacquire=on_reacquire,
    )
    print("\nJog rail to END.")
    print(prompt.format(set="e", label="END"))
    end_mm = jog_interactive(
        controller, label="END", set_key="e", keys=keys,
        on_release=on_release, on_reacquire=on_reacquire,
    )
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
