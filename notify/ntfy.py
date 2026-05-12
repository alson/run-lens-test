"""Push notifications via ntfy.sh.

The Linux laptop POSTs to ``https://ntfy.sh/<topic>`` and the user's
MacBook (subscribed to the same topic in the ntfy app) receives the
alert with a sound. Failures here must never abort the session — we
log and swallow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notification:
    title: str
    body: str
    priority: str = "default"   # "min" | "low" | "default" | "high" | "max"
    tags: tuple[str, ...] = ()


# Common alert types — the runner picks one of these so titles /
# priorities are consistent across the codebase.
LENS_CHANGE_REQUIRED = Notification(
    title="Change lens config",
    body="",
    priority="default",
    tags=("camera_with_flash",),
)
BATTERY_DROOP = Notification(
    title="Battery droop detected",
    body="",
    priority="high",
    tags=("warning", "battery"),
)
FLASH_POWER_LIMIT = Notification(
    title="Flash power limit reached",
    body="",
    priority="default",
    tags=("warning",),
)
MANUAL_APERTURE_READY = Notification(
    title="Next aperture ready",
    body="",
    priority="default",
    tags=("camera_with_flash",),
)
"""Sent at the start of each per-aperture cascade iteration in
manual-aperture mode. The rail has just finished sweeping the
previous aperture and the rig is about to prompt for the next
ring turn + flash power. Useful for stepping away during the
multi-minute sweep and getting pinged when attention is needed."""

MANUAL_CASCADE_ABORTED = Notification(
    title="Manual cascade aborted",
    body="",
    priority="high",
    tags=("warning", "rotating_light"),
)
"""Sent when the first-aperture coarse search exhausts the SB-5000
power range (either still over-exposed at 1/128 or still
under-exposed at 1/1) — the rig can't calibrate without operator
intervention (reduce ambient / add ND / change flash position)."""

CALIBRATION_NOT_CONVERGED = Notification(
    title="Calibration did not converge",
    body="",
    priority="high",
    tags=("warning",),
)
"""Sent when a manual-aperture cascade completes with
hit_tolerance=False — the final EV delta from baseline is outside
the ⅓-stop tolerance. The session continues; this is a
quality-flag, not a hard abort."""


HttpPost = Callable[[str, dict, bytes], object]


class NtfyClient:
    """POST helper. Injected ``post`` makes the class trivially mockable.

    Parameters
    ----------
    base_url : e.g. ``"https://ntfy.sh"``.
    topic : the user's private topic name (set in :mod:`config`).
    post : a callable matching ``requests.post`` (defaults to it).
    timeout_s : socket timeout for the POST call.
    """

    def __init__(
        self,
        base_url: str,
        topic: str,
        *,
        post=requests.post,
        timeout_s: float = 5.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/{topic}"
        self._post = post
        self._timeout_s = timeout_s

    def send(self, message: Notification) -> bool:
        """Returns ``True`` if delivered, ``False`` if anything failed."""
        headers = {
            "Title": message.title,
            "Priority": message.priority,
        }
        if message.tags:
            headers["Tags"] = ",".join(message.tags)
        try:
            resp = self._post(
                self._url,
                headers=headers,
                data=message.body.encode("utf-8"),
                timeout=self._timeout_s,
            )
            ok = 200 <= getattr(resp, "status_code", 599) < 300
            if not ok:
                log.warning("ntfy POST returned %s", getattr(resp, "status_code", "?"))
            return ok
        except Exception as exc:                     # noqa: BLE001
            log.warning("ntfy POST failed: %s", exc)
            return False

    def with_body(self, message: Notification, body: str) -> bool:
        """Convenience: send a copy of ``message`` with a new body."""
        return self.send(
            Notification(
                title=message.title,
                body=body,
                priority=message.priority,
                tags=message.tags,
            )
        )


# ---- CLI smoke-test -------------------------------------------------------

def _cli() -> None:
    import argparse

    import config

    parser = argparse.ArgumentParser(description="ntfy smoke-test")
    parser.add_argument("--test", action="store_true",
                        help="post a test message to NTFY_TOPIC")
    args = parser.parse_args()

    client = NtfyClient(config.NTFY_BASE_URL, config.NTFY_TOPIC)
    if args.test:
        ok = client.with_body(
            Notification(title="ntfy test", body=""),
            "Hello from macro-lens-test.",
        )
        print("delivered" if ok else "failed")


if __name__ == "__main__":
    _cli()
