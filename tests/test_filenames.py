"""Tests for session.filenames step-number helpers."""

from __future__ import annotations

import pytest

from session.filenames import parse_step_filename, step_filename


class TestStepFilename:
    def test_main_sweep_starts_at_offset(self) -> None:
        assert step_filename(0, offset=100) == "step_0100.jpg"
        assert step_filename(29, offset=100) == "step_0129.jpg"

    def test_far_extension_continues_upward(self) -> None:
        # Past the original sweep length; numbers continue.
        assert step_filename(30, offset=100) == "step_0130.jpg"
        assert step_filename(31, offset=100) == "step_0131.jpg"

    def test_near_extension_dips_below_offset(self) -> None:
        # Near side: signed step index is negative, filename stays
        # positive thanks to the offset.
        assert step_filename(-1, offset=100) == "step_0099.jpg"
        assert step_filename(-2, offset=100) == "step_0098.jpg"
        assert step_filename(-99, offset=100) == "step_0001.jpg"
        assert step_filename(-100, offset=100) == "step_0000.jpg"

    def test_filenames_sort_lexicographically_in_rail_order(self) -> None:
        # If we shoot near[-2], near[-1], main[0..2], far[3], the
        # sorted filename order should match the rail-position order.
        names = [
            step_filename(-2, offset=100),
            step_filename(-1, offset=100),
            step_filename(0, offset=100),
            step_filename(1, offset=100),
            step_filename(2, offset=100),
            step_filename(3, offset=100),
        ]
        assert names == sorted(names)

    def test_negative_overall_raises(self) -> None:
        # Step too far below the offset: filename can't go negative.
        with pytest.raises(ValueError, match="< 0"):
            step_filename(-101, offset=100)

    def test_overflow_raises(self) -> None:
        # 4-digit width caps at 9999; step 9900 + offset 100 = 10000.
        with pytest.raises(ValueError, match="exceeds width"):
            step_filename(9900, offset=100, width=4)

    def test_custom_extension(self) -> None:
        assert step_filename(0, offset=100, extension=".tiff") == "step_0100.tiff"


class TestParseStepFilename:
    def test_round_trip_main_sweep(self) -> None:
        assert parse_step_filename("step_0100.jpg", offset=100) == 0
        assert parse_step_filename("step_0129.jpg", offset=100) == 29

    def test_round_trip_near_extension(self) -> None:
        assert parse_step_filename("step_0099.jpg", offset=100) == -1
        assert parse_step_filename("step_0001.jpg", offset=100) == -99

    def test_round_trip_far_extension(self) -> None:
        assert parse_step_filename("step_0150.jpg", offset=100) == 50

    def test_rejects_unknown_prefix(self) -> None:
        with pytest.raises(ValueError, match="step filename"):
            parse_step_filename("frame_0100.jpg", offset=100)

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError, match="non-integer"):
            parse_step_filename("step_abcd.jpg", offset=100)
