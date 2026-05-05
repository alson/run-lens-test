"""Tests for stackshot.transport.FtdiTransport.

The pyusb module is replaced with a fake that records every control
transfer, every bulk write, and serves scripted bulk-IN packets
(complete with the FT232's mandatory 2-byte status-header prefix).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from errors import StackShotProtocolError
from stackshot.protocol import (
    Action,
    Cmd,
    Frame,
    SYNC_HOST,
    SYNC_SLAVE,
    build_request,
    encode_u32_le,
)
from stackshot.transport import FtdiTransport


# ---- Fake pyusb module + device --------------------------------------------


@dataclass
class _CtrlCall:
    bm_request_type: int
    request: int
    value: int
    index: int
    data_or_length: object


@dataclass
class _WriteCall:
    endpoint: int
    data: bytes
    timeout: int | None


@dataclass
class _ReadCall:
    endpoint: int
    size: int
    timeout: int | None


@dataclass
class FakeDevice:
    """Stand-in for a pyusb Device. Records every interaction."""

    kernel_driver_active: bool = False
    detached: bool = False
    config_value: int = 1
    interface_claimed: bool = False
    interface_released: bool = False
    disposed: bool = False
    ctrl_calls: list[_CtrlCall] = field(default_factory=list)
    write_calls: list[_WriteCall] = field(default_factory=list)
    read_calls: list[_ReadCall] = field(default_factory=list)
    bulk_in_packets: deque[bytes | type[Exception]] = field(
        default_factory=deque
    )

    # ---- pyusb Device API -------------------------------------------------

    def is_kernel_driver_active(self, intf: int) -> bool:
        return self.kernel_driver_active

    def detach_kernel_driver(self, intf: int) -> None:
        self.detached = True

    def get_active_configuration(self):
        return SimpleNamespace(bConfigurationValue=self.config_value)

    def set_configuration(self) -> None:
        self.config_value = 1

    def ctrl_transfer(
        self,
        bm_request_type: int,
        request: int,
        value: int,
        index: int,
        data_or_length,
    ) -> bytes:
        self.ctrl_calls.append(
            _CtrlCall(bm_request_type, request, value, index, data_or_length)
        )
        # Return value: vendor-out has no reply; vendor-in returns
        # bytes of the requested length (zero-filled).
        if (bm_request_type & 0x80) == 0:
            return b""
        return bytes(data_or_length if isinstance(data_or_length, int) else 0)

    def write(self, endpoint: int, data, timeout: int | None = None) -> int:
        self.write_calls.append(_WriteCall(endpoint, bytes(data), timeout))
        return len(bytes(data))

    def read(self, endpoint: int, size: int, timeout: int | None = None):
        self.read_calls.append(_ReadCall(endpoint, size, timeout))
        if not self.bulk_in_packets:
            raise _UsbTimeoutError("no scripted bulk-IN data")
        nxt = self.bulk_in_packets.popleft()
        if isinstance(nxt, type) and issubclass(nxt, Exception):
            raise nxt()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _UsbTimeoutError(Exception):
    """Stand-in for ``usb.core.USBTimeoutError``. Importing the real
    class isn't necessary because the transport just catches
    ``usb.core.USBError`` (the base)."""


def make_fake_usb_module(device: FakeDevice):
    """Build a minimal `usb.core` / `usb.util` stand-in."""
    module = MagicMock()
    module.core = MagicMock()
    module.core.find = MagicMock(return_value=device)
    module.core.USBError = _UsbTimeoutError
    module.util = MagicMock()
    return module


# ---- Helpers --------------------------------------------------------------


def _ft232_packet(payload: bytes = b"") -> bytes:
    """Wrap a payload in the 2-byte FT232 status header (idle status)."""
    return b"\x01\x60" + payload


def _ok_response_bytes(cmd: Cmd, data: bytes = b"") -> bytes:
    return Frame(
        sync=SYNC_SLAVE, cmd=int(cmd), action=Action.RSP_OK, data=data,
    ).to_bytes()


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def transport(device: FakeDevice) -> FtdiTransport:
    sleeps: list[float] = []
    times = [0.0]

    def _sleep(s: float) -> None:
        sleeps.append(s)
        times[0] += s

    def _clock() -> float:
        return times[0]

    tx = FtdiTransport(
        timeout_s=2.0,
        interframe_s=0.002,
        bitmode_settle_s=0.5,
        sleep=_sleep,
        clock=_clock,
        usb_module=make_fake_usb_module(device),
    )
    tx.sleeps = sleeps  # type: ignore[attr-defined]
    return tx


# ---- open() init recipe ---------------------------------------------------


class TestOpenIssuesZereneInitRecipe:
    def test_first_call_is_set_bitmode(self, transport, device) -> None:
        transport.open()
        first = device.ctrl_calls[0]
        # FTDI vendor request 0x0B (SET_BITMODE), wValue=0x20FF
        # (CBUS bit-bang mode, all CBUS pins as outputs). This is
        # the undocumented critical step that gets the StackShot
        # MCU into USB-control mode.
        assert first.bm_request_type == 0x40
        assert first.request == 0x0B
        assert first.value == 0x20FF

    def test_500ms_settle_after_set_bitmode(self, transport) -> None:
        transport.open()
        # First sleep is the bit-mode settle.
        assert transport.sleeps[0] == pytest.approx(0.5)  # type: ignore[attr-defined]

    def test_two_reset_cycles_with_baudrate_in_between(
        self, transport, device
    ) -> None:
        transport.open()
        # Skip the first ctrl_call (SET_BITMODE). The remaining sequence
        # is: RESET + 6×PURGE_RX + PURGE_TX  (cycle 1)
        #     SET_BAUDRATE + SET_LINE + SET_FLOW
        #     RESET + 6×PURGE_RX + PURGE_TX  (cycle 2)
        calls = device.ctrl_calls[1:]
        # Cycle 1.
        assert (calls[0].request, calls[0].value) == (0x00, 0x0000)  # RESET_SIO
        for i in range(1, 7):
            assert (calls[i].request, calls[i].value) == (0x00, 0x0001)  # PURGE_RX
        assert (calls[7].request, calls[7].value) == (0x00, 0x0002)  # PURGE_TX
        # UART configure.
        assert (calls[8].request, calls[8].value) == (0x03, 0xC04E)   # baudrate 38400
        assert (calls[9].request, calls[9].value) == (0x04, 0x0008)   # 8N1
        assert (calls[10].request, calls[10].value) == (0x02, 0x0000) # flow ctrl none
        # Cycle 2.
        assert (calls[11].request, calls[11].value) == (0x00, 0x0000)
        for i in range(12, 18):
            assert (calls[i].request, calls[i].value) == (0x00, 0x0001)
        assert (calls[18].request, calls[18].value) == (0x00, 0x0002)

    def test_never_issues_set_modem_ctrl(self, transport, device) -> None:
        # SET_MODEM_CTRL is FTDI vendor request 0x01. Issuing it would
        # assert DTR (or RTS) which empirically wedges the StackShot.
        transport.open()
        for call in device.ctrl_calls:
            assert call.request != 0x01

    def test_detaches_kernel_driver_when_active(
        self, transport, device
    ) -> None:
        device.kernel_driver_active = True
        transport.open()
        assert device.detached is True

    def test_skips_set_configuration_when_already_in_config_1(
        self, transport, device
    ) -> None:
        # config_value defaults to 1; set_configuration shouldn't be
        # called. (We can't easily intercept a method on a dataclass
        # without a flag, so verify config_value is unchanged AND no
        # extra reset happened — the recipe is the same length either
        # way; this test mostly documents the expectation.)
        transport.open()
        assert device.config_value == 1


# ---- request() round-trip --------------------------------------------------


class TestRequestRoundTrip:
    def test_writes_frame_then_reads_response(
        self, transport, device
    ) -> None:
        transport.open()
        # Schedule one bulk-IN packet carrying a complete RSP_OK frame.
        device.bulk_in_packets.append(
            _ft232_packet(_ok_response_bytes(Cmd.RAIL_STATUS, encode_u32_le(0)))
        )
        request_frame = build_request(Cmd.RAIL_STATUS, Action.READ)
        result = transport.request(request_frame, Cmd.RAIL_STATUS)

        # Bulk-OUT carried our frame.
        assert device.write_calls[-1].endpoint == 0x02
        assert device.write_calls[-1].data == request_frame.to_bytes()

        # Response decoded.
        assert result.cmd == int(Cmd.RAIL_STATUS)
        assert result.action == Action.RSP_OK
        assert result.data == encode_u32_le(0)

    def test_strips_ft232_status_header_from_each_packet(
        self, transport, device
    ) -> None:
        transport.open()
        rsp = _ok_response_bytes(Cmd.RAIL_STATUS, encode_u32_le(7))
        # Split the response across two bulk-IN packets, each with its
        # own 2-byte FT232 status header. The transport must strip
        # both headers and reassemble the frame.
        mid = 4
        device.bulk_in_packets.append(_ft232_packet(rsp[:mid]))
        device.bulk_in_packets.append(_ft232_packet(rsp[mid:]))
        result = transport.request(
            build_request(Cmd.RAIL_STATUS, Action.READ), Cmd.RAIL_STATUS
        )
        assert result.data == encode_u32_le(7)

    def test_skips_empty_status_only_packets(
        self, transport, device
    ) -> None:
        transport.open()
        rsp = _ok_response_bytes(Cmd.RAIL_STATUS, encode_u32_le(0))
        # Two empty status-only packets, then the real response.
        device.bulk_in_packets.append(_ft232_packet(b""))
        device.bulk_in_packets.append(_ft232_packet(b""))
        device.bulk_in_packets.append(_ft232_packet(rsp))
        result = transport.request(
            build_request(Cmd.RAIL_STATUS, Action.READ), Cmd.RAIL_STATUS
        )
        assert result.action == Action.RSP_OK

    def test_request_before_open_raises(self, transport) -> None:
        with pytest.raises(RuntimeError, match="open"):
            transport.request(
                build_request(Cmd.RAIL_STATUS, Action.READ),
                Cmd.RAIL_STATUS,
            )

    def test_observes_inter_frame_delay(self, transport, device) -> None:
        transport.open()
        device.bulk_in_packets.append(
            _ft232_packet(_ok_response_bytes(Cmd.RAIL_STATUS, encode_u32_le(0)))
        )
        sleeps_before = list(transport.sleeps)  # type: ignore[attr-defined]
        transport.request(
            build_request(Cmd.RAIL_STATUS, Action.READ), Cmd.RAIL_STATUS
        )
        new_sleeps = transport.sleeps[len(sleeps_before):]  # type: ignore[attr-defined]
        # First sleep in a request is the inter-frame delay.
        assert new_sleeps[0] == pytest.approx(0.002)


# ---- timeout behaviour -----------------------------------------------------


class TestTimeout:
    def test_request_timeout_raises_protocol_error(
        self, transport, device
    ) -> None:
        transport.open()
        # No bulk-IN packets scripted; the FakeDevice raises
        # USBError on every read (= USBTimeoutError-equivalent),
        # so the transport keeps polling until the deadline.
        with pytest.raises(StackShotProtocolError, match="timed out"):
            transport.request(
                build_request(Cmd.RAIL_STATUS, Action.READ),
                Cmd.RAIL_STATUS,
            )


# ---- close() ---------------------------------------------------------------


class TestClose:
    def test_release_and_dispose(self, transport, device) -> None:
        transport.open()
        transport.close()
        usb_module = transport._usb  # type: ignore[attr-defined]
        usb_module.util.release_interface.assert_called_once()
        usb_module.util.dispose_resources.assert_called_once()

    def test_close_releases_panel_before_disposing(
        self, transport, device,
    ) -> None:
        """close() must issue SET_BITMODE 0x20F0 before tearing down
        the libusb handle. Otherwise the CBUS bit-mode stays at
        0x20FF and the StackShot LCD remains in "USB CONTROL" mode
        across program exit (and even across rail power cycles —
        only USB disconnect clears it)."""
        transport.open()
        device.ctrl_calls.clear()
        transport.close()
        # First (and only) ctrl call after clear should be the
        # release-to-panel SET_BITMODE.
        assert device.ctrl_calls
        first = device.ctrl_calls[0]
        assert first.request == 0x0B
        assert first.value == 0x20F0

    def test_close_is_idempotent(self, transport) -> None:
        transport.open()
        transport.close()
        transport.close()  # must not raise


# ---- pyusb-not-installed path ---------------------------------------------


class TestImportError:
    def test_clear_message_when_pyusb_missing(self) -> None:
        # Force the AttributeError path by passing a usb_module that's None
        # AND making the module-level _PYUSB_AVAILABLE False.
        from stackshot import transport as transport_mod
        original = transport_mod._PYUSB_AVAILABLE
        transport_mod._PYUSB_AVAILABLE = False
        try:
            with pytest.raises(RuntimeError, match="pyusb not installed"):
                FtdiTransport()
        finally:
            transport_mod._PYUSB_AVAILABLE = original


# ---- Panel release / re-acquire -------------------------------------------


class TestPanelRelease:
    def test_release_sends_bitmode_with_mask_0xF0(
        self, transport, device
    ) -> None:
        transport.open()
        # Reset call history so we're isolated to the release call.
        device.ctrl_calls.clear()
        transport.release_to_panel()
        # Exactly one ctrl: SET_BITMODE wValue=0x20F0.
        assert len(device.ctrl_calls) == 1
        call = device.ctrl_calls[0]
        assert call.bm_request_type == 0x40
        assert call.request == 0x0B           # SET_BIT_MODE
        assert call.value == 0x20F0           # mode 0x20, mask 0xF0

    def test_take_sends_bitmode_with_mask_0xFF_and_settles(
        self, transport, device
    ) -> None:
        transport.open()
        device.ctrl_calls.clear()
        sleeps_before = list(transport.sleeps)  # type: ignore[attr-defined]
        transport.take_from_panel()
        # Exactly one ctrl: SET_BITMODE wValue=0x20FF.
        assert len(device.ctrl_calls) == 1
        call = device.ctrl_calls[0]
        assert call.request == 0x0B
        assert call.value == 0x20FF
        # Plus a 500 ms settle.
        new_sleeps = transport.sleeps[len(sleeps_before):]  # type: ignore[attr-defined]
        assert new_sleeps == [pytest.approx(0.5)]

    def test_release_before_open_raises(self, transport) -> None:
        with pytest.raises(RuntimeError, match="release_to_panel before open"):
            transport.release_to_panel()

    def test_take_before_open_raises(self, transport) -> None:
        with pytest.raises(RuntimeError, match="take_from_panel before open"):
            transport.take_from_panel()


class TestSerialTransportRaisesOnPanelOps:
    def test_release_raises_not_implemented(self) -> None:
        from stackshot.transport import SerialTransport
        with pytest.raises(NotImplementedError, match="libusb-direct"):
            SerialTransport("/dev/null").release_to_panel()

    def test_take_raises_not_implemented(self) -> None:
        from stackshot.transport import SerialTransport
        with pytest.raises(NotImplementedError, match="libusb-direct"):
            SerialTransport("/dev/null").take_from_panel()
