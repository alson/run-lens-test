"""Tests for camera.nikon_z7.NikonZ7.

The whole gphoto2 layer is mocked. We verify the orchestration:
- verify_healthy raises with the right message
- apply_session_config writes to every present widget
- drain_events deletes file-added events
- capture_to retries on GPhoto2Error and gives up after the budget
- fec_sanity_check runs both shots and compares medians
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from camera.nikon_z7 import NikonZ7
from errors import CameraCaptureError, CameraStateError, FECSanityError
from tests.test_widgets import make_widget


class FakeGPhotoError(Exception):
    """Stand-in for gphoto2.GPhoto2Error."""

    code: int = -1


def make_gp_module():
    """A minimal gphoto2 stand-in with the constants the wrapper uses."""
    gp = MagicMock()
    gp.GPhoto2Error = FakeGPhotoError
    gp.GP_EVENT_TIMEOUT = 1
    gp.GP_EVENT_FILE_ADDED = 2
    gp.GP_CAPTURE_IMAGE = 0
    gp.GP_FILE_TYPE_NORMAL = 0
    return gp


def make_camera():
    """A MagicMock standing in for a gp.Camera().

    NOTE: ``get_config`` is wired to return a real widget tree by
    default. If we left it as a default MagicMock,
    :func:`camera.widgets.find` / :func:`count_widgets` would recurse
    forever — MagicMock's ``__index__`` returns 1, so
    ``range(mock.count_children())`` is non-empty and the recursion
    keeps allocating fresh MagicMocks per level. That was a real OOM
    earlier (37 GiB resident before the kernel killed pytest).
    """
    cam = MagicMock()
    cam.wait_for_event.return_value = (1, None)  # GP_EVENT_TIMEOUT
    # Empty-tree root with count_children() == 0 — terminates recursion.
    cam.get_config.return_value = make_widget("main")
    return cam


@pytest.fixture
def gp():
    return make_gp_module()


@pytest.fixture
def camera():
    return make_camera()


@pytest.fixture
def nikon(gp, camera) -> NikonZ7:
    n = NikonZ7(gp_module=gp, sleep=lambda s: None)
    # Use the constructed Camera mock; bypass open()/init().
    n._camera = camera
    # Make gp.Camera() return our pre-built mock too (in case open() runs).
    gp.Camera.return_value = camera
    return n


# ---- verify_healthy --------------------------------------------------------


def _config_tree(*, n_widgets: int, expprogram: str = "M"):
    # Build a tree with the given total widget count.
    leaves = [make_widget(f"w{i}") for i in range(max(0, n_widgets - 2))]
    ep = make_widget("expprogram", expprogram)
    root = make_widget("main", children=[ep, *leaves])
    return root


class TestVerifyHealthy:
    def test_passes_with_full_tree_in_M_mode(self, nikon, camera) -> None:
        camera.get_config.return_value = _config_tree(n_widgets=120)
        nikon.verify_healthy(min_widgets=100)

    def test_raises_on_minimal_mtp_tree(self, nikon, camera) -> None:
        camera.get_config.return_value = _config_tree(n_widgets=23)
        with pytest.raises(CameraStateError, match="minimal-MTP"):
            nikon.verify_healthy(min_widgets=100)

    def test_raises_when_not_in_M(self, nikon, camera) -> None:
        camera.get_config.return_value = _config_tree(
            n_widgets=120, expprogram="A"
        )
        with pytest.raises(CameraStateError, match="expprogram"):
            nikon.verify_healthy(min_widgets=100)

    def test_tolerates_transient_question_mark(self, nikon, camera) -> None:
        # First two get_config calls return ?, third returns M.
        trees = [
            _config_tree(n_widgets=120, expprogram="?"),
            _config_tree(n_widgets=120, expprogram="?"),
            _config_tree(n_widgets=120, expprogram="M"),
        ]
        # First call is the size check, after that the value retries
        # use fresh get_config. Easiest: return a tree where total
        # count is fine and walk the qm retries.
        camera.get_config.side_effect = trees + trees + trees
        nikon.verify_healthy(min_widgets=100)

    def test_gives_up_on_persistent_question_mark(self, nikon, camera) -> None:
        camera.get_config.return_value = _config_tree(
            n_widgets=120, expprogram="?"
        )
        with pytest.raises(CameraStateError, match="expprogram"):
            nikon.verify_healthy(min_widgets=100)


# ---- apply_session_config --------------------------------------------------


class TestApplySessionConfig:
    def test_sets_all_present_widgets(self, nikon, camera) -> None:
        # Build one big tree with all the widgets the wrapper touches.
        # Order matches the wrapper's get_config call sequence — VR
        # scan happens between autofocus and flashexposurecompensation.
        widgets = {
            "capturetarget": make_widget(
                "capturetarget", "Memory card", choices=["Internal RAM", "Memory card"]
            ),
            "imagequality": make_widget(
                "imagequality", "JPEG Basic", choices=["JPEG Fine", "JPEG Basic"]
            ),
            "iso": make_widget("iso", "100", choices=["100", "200", "400"]),
            "shutterspeed": make_widget(
                "shutterspeed", "1/250", choices=["0.0050s", "1/250"]
            ),
            "autofocus": make_widget("autofocus", "On", choices=["On", "Off"]),
            "vr": make_widget("vr", "On", choices=["On", "Off"]),
            "flashexposurecompensation": make_widget(
                "flashexposurecompensation", 1.0
            ),
        }
        # Each get_config call must return a fresh tree because the
        # wrapper does get_config -> mutate -> set_config per setter.
        camera.get_config.side_effect = [
            make_widget("main", children=[w]) for w in widgets.values()
        ]
        nikon.apply_session_config()
        # set_config called once per widget present (including VR).
        assert camera.set_config.call_count == 7
        widgets["capturetarget"].set_value.assert_called_with("Internal RAM")
        widgets["imagequality"].set_value.assert_called_with("JPEG Fine")
        widgets["iso"].set_value.assert_called_with("100")
        widgets["shutterspeed"].set_value.assert_called_with("0.0050s")
        widgets["autofocus"].set_value.assert_called_with("Off")
        widgets["vr"].set_value.assert_called_with("Off")
        widgets["flashexposurecompensation"].set_value.assert_called_with(0.0)

    def test_skips_missing_widgets(self, nikon, camera) -> None:
        # Empty tree: every widget lookup returns None.
        camera.get_config.return_value = make_widget("main")
        nikon.apply_session_config()
        # No exception raised, no set_config calls.
        assert camera.set_config.call_count == 0

    def test_raises_when_no_candidate_matches(self, nikon, camera) -> None:
        widget = make_widget(
            "shutterspeed", "1.0s", choices=["1.0s", "2.0s"]
        )
        camera.get_config.return_value = make_widget("main", children=[widget])
        with pytest.raises(CameraStateError, match="shutterspeed"):
            nikon.apply_session_config()

    def test_disables_vr_under_alt_widget_name(self, nikon, camera) -> None:
        # Z7 firmware that exposes VR as `vibrationreduction` instead
        # of `vr` is still handled.
        vr = make_widget(
            "vibrationreduction", "On", choices=["On", "Off"]
        )
        camera.get_config.return_value = make_widget("main", children=[vr])
        nikon.apply_session_config()
        vr.set_value.assert_called_with("Off")

    def test_skips_vr_when_no_off_choice(self, nikon, camera) -> None:
        # No Off-like label among choices: silently skip rather than
        # picking the wrong setting.
        vr = make_widget("vr", "Normal", choices=["Normal", "Sport"])
        camera.get_config.return_value = make_widget("main", children=[vr])
        nikon.apply_session_config()
        vr.set_value.assert_not_called()

    def test_skips_vr_when_widget_absent(self, nikon, camera) -> None:
        # No VR widget present (lens has no IS, or firmware variant
        # without the widget): no error, no set_value.
        camera.get_config.return_value = make_widget("main")
        nikon.apply_session_config()
        # Empty tree -> no widgets touched at all.
        assert camera.set_config.call_count == 0


# ---- drain_events ----------------------------------------------------------


class TestDrainEvents:
    def test_returns_zero_when_no_events(self, nikon, camera, gp) -> None:
        camera.wait_for_event.return_value = (gp.GP_EVENT_TIMEOUT, None)
        assert nikon.drain_events() == 0

    def test_deletes_file_added_events(self, nikon, camera, gp) -> None:
        # Use a real object — MagicMock(name=...) is reserved and does
        # not set the .name attribute the way the kwargs syntax suggests.
        class _Evt:
            folder = "/store_00010001"
            name = "capt0000.jpg"

        camera.wait_for_event.side_effect = [
            (gp.GP_EVENT_FILE_ADDED, _Evt()),
            (gp.GP_EVENT_TIMEOUT, None),
        ]
        drained = nikon.drain_events()
        assert drained == 1
        camera.file_delete.assert_called_once_with(
            "/store_00010001", "capt0000.jpg"
        )

    def test_caps_at_20_events(self, nikon, camera, gp) -> None:
        # Other event types (not TIMEOUT, not FILE_ADDED): wrapper
        # increments drained but doesn't touch the camera. Cap fires at
        # drained > 20 so we should never consume more than 21 entries.
        camera.wait_for_event.side_effect = [(3, None)] * 30
        drained = nikon.drain_events()
        assert drained <= 21


# ---- capture_to ------------------------------------------------------------


class _FilePath:
    """Stand-in for gphoto2.CameraFilePath. ``MagicMock(name=...)`` is
    reserved and would not actually set ``.name`` on the mock."""

    def __init__(self, folder: str, name: str) -> None:
        self.folder = folder
        self.name = name


class TestCaptureTo:
    def test_happy_path(self, nikon, camera, gp, tmp_path) -> None:
        camera.wait_for_event.return_value = (gp.GP_EVENT_TIMEOUT, None)
        fp = _FilePath("/x", "capt0000.jpg")
        camera.capture.return_value = fp

        dest = tmp_path / "out.jpg"

        def _save(path: str) -> None:
            Path(path).write_bytes(b"x" * 12345)

        cf = MagicMock()
        cf.save.side_effect = _save
        camera.file_get.return_value = cf

        size = nikon.capture_to(dest)
        assert size == 12345
        camera.capture.assert_called_once_with(gp.GP_CAPTURE_IMAGE)
        camera.file_get.assert_called_once_with(
            "/x", "capt0000.jpg", gp.GP_FILE_TYPE_NORMAL
        )
        camera.file_delete.assert_called_once_with("/x", "capt0000.jpg")

    def test_retries_then_succeeds(self, nikon, camera, gp, tmp_path) -> None:
        camera.wait_for_event.return_value = (gp.GP_EVENT_TIMEOUT, None)
        fp = _FilePath("/x", "ok.jpg")
        cf = MagicMock()
        cf.save.side_effect = lambda p: Path(p).write_bytes(b"data")
        camera.capture.side_effect = [
            FakeGPhotoError("transient"),
            fp,
        ]
        camera.file_get.return_value = cf

        # Provide an autofocus widget for the recovery branch.
        af = make_widget("autofocus", "On", choices=["On", "Off"])
        camera.get_config.return_value = make_widget("main", children=[af])

        dest = tmp_path / "retry.jpg"
        size = nikon.capture_to(dest, max_retries=3)
        assert size == 4
        assert camera.capture.call_count == 2

    def test_gives_up_after_max_retries(self, nikon, camera, gp, tmp_path) -> None:
        camera.wait_for_event.return_value = (gp.GP_EVENT_TIMEOUT, None)
        camera.capture.side_effect = FakeGPhotoError("always fails")
        # Autofocus widget for recovery on attempt 2.
        af = make_widget("autofocus", "On", choices=["On", "Off"])
        camera.get_config.return_value = make_widget("main", children=[af])

        with pytest.raises(CameraCaptureError, match="3 attempts"):
            nikon.capture_to(tmp_path / "fail.jpg", max_retries=3)


# ---- FEC sanity check ------------------------------------------------------


class TestFECSanityCheck:
    def _wire_capture_to(self, nikon, camera, gp, tmp_path) -> None:
        # Each capture is treated as a successful capture-to-file.
        camera.wait_for_event.return_value = (gp.GP_EVENT_TIMEOUT, None)
        camera.capture.return_value = _FilePath("/x", "x.jpg")
        cf = MagicMock()
        cf.save.side_effect = lambda p: Path(p).write_bytes(b"x")
        camera.file_get.return_value = cf

    def test_passes_when_low_is_dim(self, nikon, camera, gp, tmp_path) -> None:
        self._wire_capture_to(nikon, camera, gp, tmp_path)
        # FEC=0 → 200, FEC=-2 → 50; ratio 0.25, well below 0.5.
        medians = iter([200.0, 50.0])
        nikon.fec_sanity_check(
            capture_dir=tmp_path,
            measure=lambda _p: next(medians),
        )

    def test_fails_when_low_is_too_bright(self, nikon, camera, gp, tmp_path) -> None:
        self._wire_capture_to(nikon, camera, gp, tmp_path)
        # FEC=-2 only barely darker — ratio 0.95 > 0.5.
        medians = iter([200.0, 190.0])
        with pytest.raises(FECSanityError, match="SU-800"):
            nikon.fec_sanity_check(
                capture_dir=tmp_path,
                measure=lambda _p: next(medians),
            )

    def test_zero_baseline_is_a_failure(self, nikon, camera, gp, tmp_path) -> None:
        self._wire_capture_to(nikon, camera, gp, tmp_path)
        medians = iter([0.0, 0.0])
        with pytest.raises(FECSanityError, match="zero median"):
            nikon.fec_sanity_check(
                capture_dir=tmp_path,
                measure=lambda _p: next(medians),
            )

    def test_restores_fec_to_baseline(self, nikon, camera, gp, tmp_path) -> None:
        self._wire_capture_to(nikon, camera, gp, tmp_path)
        # Track set_value calls on the FEC widget.
        fec_widget = make_widget("flashexposurecompensation", 0.0)
        camera.get_config.return_value = make_widget(
            "main", children=[fec_widget]
        )
        medians = iter([200.0, 50.0])
        nikon.fec_sanity_check(
            capture_dir=tmp_path,
            measure=lambda _p: next(medians),
        )
        # 0, then -2, then 0 again.
        assert fec_widget.set_value.call_args_list == [
            call(0.0), call(-2.0), call(0.0),
        ]


# ---- list_apertures --------------------------------------------------------


class TestListApertures:
    def test_returns_choices(self, nikon, camera) -> None:
        f_number = make_widget(
            "f-number", "f/4", choices=["f/2.8", "f/4", "f/5.6", "f/8"]
        )
        camera.get_config.return_value = make_widget(
            "main", children=[f_number]
        )
        assert nikon.list_apertures() == ["f/2.8", "f/4", "f/5.6", "f/8"]

    def test_missing_returns_empty(self, nikon, camera) -> None:
        camera.get_config.return_value = make_widget("main")
        assert nikon.list_apertures() == []
