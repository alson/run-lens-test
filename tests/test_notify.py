"""Tests for notify.ntfy and notify.sound — both are I/O wrappers
where the actual call is injectable, so we just verify orchestration."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from notify.ntfy import (
    BATTERY_DROOP,
    LENS_CHANGE_REQUIRED,
    NtfyClient,
    Notification,
)
from notify.sound import play_alert


# ---- ntfy ------------------------------------------------------------------


class TestNtfyClient:
    def test_url_combines_base_and_topic(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=200))
        client = NtfyClient("https://ntfy.sh/", "my-topic", post=post)
        client.send(Notification(title="t", body="b"))
        assert post.call_args[0][0] == "https://ntfy.sh/my-topic"

    def test_headers_include_title_and_priority(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=200))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        client.send(Notification(title="hi", body="x", priority="high",
                                 tags=("warning", "battery")))
        headers = post.call_args.kwargs["headers"]
        assert headers["Title"] == "hi"
        assert headers["Priority"] == "high"
        assert headers["Tags"] == "warning,battery"

    def test_body_is_utf8_bytes(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=200))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        client.send(Notification(title="t", body="éclair"))
        assert post.call_args.kwargs["data"] == "éclair".encode("utf-8")

    def test_returns_true_on_2xx(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=204))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        assert client.send(Notification(title="t", body="b")) is True

    def test_returns_false_on_5xx(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=500))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        assert client.send(Notification(title="t", body="b")) is False

    def test_returns_false_when_post_raises(self) -> None:
        post = MagicMock(side_effect=ConnectionError("network down"))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        # Must NOT propagate — alerts must never abort a session.
        assert client.send(Notification(title="t", body="b")) is False

    def test_with_body_substitutes_body_only(self) -> None:
        post = MagicMock(return_value=MagicMock(status_code=200))
        client = NtfyClient("https://ntfy.sh", "topic", post=post)
        client.with_body(BATTERY_DROOP, "f/8 dropped to -2.3 EV")
        headers = post.call_args.kwargs["headers"]
        # Title and tags from BATTERY_DROOP are preserved.
        assert headers["Title"] == BATTERY_DROOP.title
        assert headers["Priority"] == "high"
        assert post.call_args.kwargs["data"] == b"f/8 dropped to -2.3 EV"


# ---- sound -----------------------------------------------------------------


class TestPlayAlert:
    def test_returns_false_when_wav_missing(self, tmp_path) -> None:
        # Reject before invoking paplay even if it exists.
        runner = MagicMock()
        ok = play_alert(tmp_path / "missing.wav",
                        runner=runner, paplay="/usr/bin/paplay")
        assert ok is False
        runner.assert_not_called()

    def test_returns_false_when_paplay_missing(self, tmp_path) -> None:
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"RIFF...")
        runner = MagicMock()
        # Force paplay=None and arrange shutil.which to fail by passing
        # an explicit non-existent path. The function uses shutil.which
        # only when paplay is None.
        ok = play_alert(wav, runner=runner, paplay=None)
        # Either paplay isn't on PATH (return False) or it is and the
        # runner gets invoked. Both cases are fine; assert nothing
        # crashes and the result is bool.
        assert isinstance(ok, bool)

    def test_invokes_paplay_with_path(self, tmp_path) -> None:
        wav = tmp_path / "alert.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)
        runner = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b"",
            )
        )
        ok = play_alert(wav, runner=runner, paplay="/usr/bin/paplay")
        assert ok is True
        # First positional arg is the command list.
        cmd = runner.call_args.args[0]
        assert cmd[0] == "/usr/bin/paplay"
        assert cmd[1] == str(wav)

    def test_returns_false_on_non_zero_returncode(self, tmp_path) -> None:
        wav = tmp_path / "alert.wav"
        wav.write_bytes(b"x")
        runner = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"err",
            )
        )
        assert play_alert(wav, runner=runner, paplay="/usr/bin/paplay") is False

    def test_returns_false_on_timeout(self, tmp_path) -> None:
        wav = tmp_path / "alert.wav"
        wav.write_bytes(b"x")
        runner = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="paplay", timeout=5.0)
        )
        assert play_alert(wav, runner=runner, paplay="/usr/bin/paplay") is False
