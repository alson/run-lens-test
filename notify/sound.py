"""Local audio alert via paplay (PulseAudio / PipeWire).

Used in addition to remote ntfy so the user is alerted whether
they're at the rig or at the Mac. Failures here must never abort the
session.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


SubprocessRunner = Callable[..., subprocess.CompletedProcess]


def play_alert(
    wav_path: Path | str,
    *,
    runner: SubprocessRunner = subprocess.run,
    paplay: str | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """Play the given WAV file once. Returns True on success.

    Parameters
    ----------
    wav_path : Path to a .wav file readable by paplay.
    runner : Replaceable for testing (defaults to :func:`subprocess.run`).
    paplay : Path to the binary; auto-discovered with shutil.which()
        if None.
    timeout_s : Hard cap on how long we'll wait.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        log.warning("alert wav missing: %s", wav_path)
        return False
    binary = paplay or shutil.which("paplay")
    if binary is None:
        log.info("paplay not found on PATH; skipping local alert sound")
        return False
    try:
        result = runner(
            [binary, str(wav_path)],
            timeout=timeout_s,
            check=False,
            capture_output=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log.warning("paplay failed: %s", exc)
        return False
