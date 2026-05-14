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
# Hot-key for "type a target rail position in mm" — colon makes it
# unmistakable vs ``a/d/A/D``, and matches the vim/less convention.
NUMERIC_ENTRY_KEY = ":"
# Characters allowed inside a numeric-entry buffer.
_NUMERIC_CHARS: frozenset[str] = frozenset("0123456789.+-")


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
    start_in_panel: bool = True,
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
        the rail manually, then reacquire it after. When the user
        ends the jog via ``set_key`` while in panel mode,
        ``on_reacquire`` is NOT called — the camera is left released
        so the next jog can stay in live view without paying the
        ~2 s reacquire cost.
    start_in_panel :
        When True (default), the loop immediately enters front-panel
        mode at startup; pressing ENTER drops to the supervised
        keyboard sub-mode. Tests pass ``start_in_panel=False`` to
        skip the auto-release and start in keyboard mode directly.

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
    if start_in_panel:
        # Drop straight into panel mode. The user can press s/e to
        # set position without leaving live view (camera stays
        # released across both jogs for "s"; "e" reacquires since
        # capture begins immediately after), or press ENTER to drop
        # to keyboard mode (which reacquires).
        position, panel_pending, panel_set = _release_to_panel(
            controller=controller,
            read_key=read_key,
            write_status=write_status,
            label=label,
            on_release=on_release,
            on_reacquire=on_reacquire,
            set_keys=(set_key, keys.abort_key),
        )
        if panel_set == keys.abort_key:
            raise SessionAborted(f"jog aborted at {label}")
        if panel_set == set_key:
            # s/e was pressed in panel mode. _release_to_panel already
            # took the rail back from the front panel (so we can read
            # / zero) and skipped on_reacquire. Do the set logic, then
            # only reacquire the camera for "e" (capturing starts).
            if set_key == "s":
                controller.zero_position()
                position = controller.get_position()
                write_status(position, f"{label} set (zeroed)")
            else:
                write_status(position, f"{label} set")
                if on_reacquire is not None:
                    if sys.stdout.isatty():
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    try:
                        on_reacquire()
                    except Exception as exc:          # noqa: BLE001
                        write_status(
                            position,
                            f"{label} (on_reacquire callback failed: {exc})",
                        )
            return position
        if panel_pending is not None:
            pending_key = panel_pending
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
            position, panel_pending, panel_set = _release_to_panel(
                controller=controller,
                read_key=read_key,
                write_status=write_status,
                label=label,
                on_release=on_release,
                on_reacquire=on_reacquire,
                set_keys=(set_key, keys.abort_key),
            )
            if panel_set == keys.abort_key:
                raise SessionAborted(f"jog aborted at {label}")
            if panel_set == set_key:
                if set_key == "s":
                    controller.zero_position()
                    position = controller.get_position()
                    write_status(position, f"{label} set (zeroed)")
                else:
                    write_status(position, f"{label} set")
                    if on_reacquire is not None:
                        if sys.stdout.isatty():
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        try:
                            on_reacquire()
                        except Exception as exc:      # noqa: BLE001
                            write_status(
                                position,
                                f"{label} (on_reacquire callback failed: {exc})",
                            )
                return position
            if panel_pending is not None:
                pending_key = panel_pending
            continue
        if key == NUMERIC_ENTRY_KEY:
            position = _numeric_entry(
                controller=controller,
                read_key=read_key,
                write_status=write_status,
                label=label,
                start_position=position,
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
        # Drain any queued same-direction keystrokes that piled up
        # while the move ran (held key + ~30 Hz auto-repeat). Discard
        # them so we don't run into an end-stop after release. Keep
        # the first different keystroke for the next iteration.
        pending_key = _drain_after_move(read_key, key)


def _numeric_entry(
    *,
    controller: StackShotController,
    read_key: Callable[..., str | None],
    write_status: Callable[[float, str], None],
    label: str,
    start_position: float,
) -> float:
    """Read a typed absolute target position, then move there.

    Activated inside :func:`jog_loop` when the user presses ``:``.
    Echoes typed characters into the status line so the user sees
    what they're entering. Returns the rail's position after the
    move (or ``start_position`` unchanged if the user cancelled).

    Accepted keystrokes:
    * ``0``-``9``, ``.``, ``+``, ``-``: appended to the buffer.
    * Backspace (``\\x7f`` or ``\\b``): pops the last character.
    * Enter (``\\r`` or ``\\n``): parse + move. Empty input cancels.
    * ``:`` (entry key) again, or ESC (``\\x1b``), or ``q``: cancel.

    On parse errors the buffer is cleared and the user is shown an
    error hint; entry stays open so they can retry without leaving
    the jog loop.
    """
    buffer = ""
    error: str | None = None
    while True:
        prefix = f"{label} | enter target mm:"
        if error:
            write_status(start_position, f"{prefix} {buffer}_  [{error}]")
        else:
            write_status(start_position, f"{prefix} {buffer}_")
        key = read_key()
        if key is None:
            # Stream closed; treat as cancel.
            return start_position
        if key in ("\r", "\n"):
            text = buffer.strip()
            if not text:
                return start_position
            try:
                target = float(text)
            except ValueError:
                error = "not a number"
                buffer = ""
                continue
            # Compute delta and execute the move.
            delta = target - start_position
            if delta == 0:
                return start_position
            direction = Direction.FORWARD if delta > 0 else Direction.BACKWARD
            controller.move_and_wait(direction, abs(delta), backlash=True)
            new_position = controller.get_position()
            write_status(new_position, label)
            return new_position
        if key in (NUMERIC_ENTRY_KEY, "\x1b", "q"):
            # Cancel — restore the regular status line.
            write_status(start_position, label)
            return start_position
        if key in ("\x7f", "\b"):
            buffer = buffer[:-1]
            error = None
            continue
        if key in _NUMERIC_CHARS:
            buffer += key
            error = None
            continue
        # Anything else (cursor keys, etc.): silently ignore.


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
    set_keys: tuple[str, ...] = (),
) -> tuple[float, str | None, str | None]:
    """Drop USB control so the user can drive the rail with the
    StackShot's physical buttons (and, via ``on_release``, regain
    direct LCD/AF control of the camera). Returns
    ``(position, pending_key, set_key_pressed)``: the position the
    user left the rail at, an optional key queued during reacquire
    to forward to the caller's main loop, and which set key (if any)
    was pressed inside panel mode to exit early.

    ``on_release`` runs after the rail is released (typically closes
    the gphoto2 session so the camera's LCD shows live view).
    ``on_reacquire`` runs before the rail is taken back from the
    panel (typically re-opens the gphoto2 session). When the user
    exits panel mode by pressing one of ``set_keys`` (s/e/q),
    ``on_reacquire`` is skipped — the camera is left released so the
    next jog can stay in live view without paying the reacquire cost.

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
        return cached_position, None, None
    if on_release is not None:
        # Break the in-place \r status line *before* the on_release
        # callback runs — the callback typically emits log records
        # via the logging handler, which has no idea our cursor is
        # parked at column 0 of the status line. Without this
        # newline, the log message gets glued to the end of the
        # status text:
        #   [END] position = +0.000 mm2026-05-05 22:59:12 INFO ...
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()
        try:
            on_release()
        except Exception as exc:                     # noqa: BLE001
            write_status(
                cached_position,
                f"{label} | PANEL (on_release callback failed: {exc})",
            )
    set_hint = "/".join(set_keys) if set_keys else ""
    write_status(
        cached_position,
        f"{label} | PANEL — rail buttons + camera live view; "
        f"ENTER to resume" + (f", {set_hint} to set" if set_hint else ""),
    )
    exit_set_key: str | None = None
    while True:
        k = read_key()
        if k in ("\r", "\n"):
            break
        if k in set_keys:
            exit_set_key = k
            break
    if exit_set_key is not None:
        # Set was pressed in panel — take rail back briefly so we can
        # read its current position (and zero it for "s"). Skip
        # on_reacquire: the caller wants the camera left in live view
        # so the next jog (END) can stay in panel without paying the
        # reacquire cost.
        controller.take_from_panel()
        position = controller.get_position()
        write_status(position, label)
        return position, None, exit_set_key
    # Immediate visual feedback BEFORE the slow reacquire (camera
    # re-init can take ~2 s). Without this the user thinks ENTER
    # didn't register and queues more keystrokes that leak into the
    # next prompt.
    write_status(
        cached_position,
        f"{label} | reacquiring camera + rail…",
    )
    if on_reacquire is not None:
        # Same newline trick as on_release above — the reacquire
        # callback's log records would otherwise glue onto the
        # "reacquiring camera + rail…" status line.
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()
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
    return position, pending, None


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

    Both jogs start in front-panel mode by default (rail under its
    own buttons + camera in live view); pressing ENTER drops to the
    supervised keyboard sub-mode. Pressing ``s`` in panel ends the
    START jog without reacquiring the camera, so the END jog stays
    in live view too. ``e`` in panel ends the END jog and reacquires
    the camera (capture begins immediately after).

    Optional ``on_release`` / ``on_reacquire`` callbacks are wrapped
    to be idempotent across the two-jog sequence so a no-op second
    release / first reacquire never reaches the camera. The
    production runner passes camera close / open functions; this
    wrapper guarantees they're called the correct number of times
    even when the user toggles between panel and keyboard modes.
    """
    # State machine — gates on_release/on_reacquire so the camera is
    # only closed when open and reopened when closed.
    camera_released = [False]

    def _wrap_release() -> None:
        if camera_released[0]:
            return
        if on_release is not None:
            on_release()
        camera_released[0] = True

    def _wrap_reacquire() -> None:
        if not camera_released[0]:
            return
        if on_reacquire is not None:
            on_reacquire()
        camera_released[0] = False

    prompt = (
        "  PANEL: rail buttons + camera live view  ENTER=keyboard  "
        "{set}=set {label}  q=abort"
    )
    keyboard_prompt = (
        "  a/d=coarse(1mm)  A/D=fine(0.05mm)  :=type-target-mm  "
        "SPACE=front-panel+camera  {set}=set {label}  q=abort"
    )
    print("Jog rail to START.")
    print(prompt.format(set="s", label="START"))
    print(keyboard_prompt.format(set="s", label="START"))
    jog_interactive(
        controller, label="START", set_key="s", keys=keys,
        on_release=_wrap_release, on_reacquire=_wrap_reacquire,
    )
    print("\nJog rail to END.")
    print(prompt.format(set="e", label="END"))
    print(keyboard_prompt.format(set="e", label="END"))
    end_mm = jog_interactive(
        controller, label="END", set_key="e", keys=keys,
        on_release=_wrap_release, on_reacquire=_wrap_reacquire,
    )
    # Final safety: if the user managed to end the END jog with the
    # camera still released (shouldn't happen — set_key="e" in panel
    # reacquires, and ENTER → keyboard also reacquires), reacquire
    # now so capture can proceed.
    _wrap_reacquire()
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
