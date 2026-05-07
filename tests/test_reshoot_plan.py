"""Tests for ui.reshoot_plan — the curated lens-config menu source."""

from __future__ import annotations

from pathlib import Path

import pytest

from ui.reshoot_plan import PlanEntry, load_plan


# Real fixture: the actual reshoot plan that the runner consumes in
# production. Driving tests off a synthetic markdown would let the
# fixture and the real file drift apart silently.
PLAN_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "analyse-lens-test"
    / "reshoot-plan-2026-05.md"
)


@pytest.fixture
def real_plan() -> list[PlanEntry]:
    if not PLAN_PATH.exists():
        pytest.skip(f"reshoot plan missing at {PLAN_PATH}")
    return load_plan(PLAN_PATH)


class TestRealPlan:
    def test_first_entry_is_group_1_row_1(self, real_plan: list[PlanEntry]) -> None:
        e = real_plan[0]
        assert e.number == 1
        assert e.group.startswith("Group 1")
        assert e.lens_config.startswith("NIKKOR Z MC 105mm")
        assert e.estimated is True
        # Z MC 105 + 61 mm tube: F_eff at 1:1 ≈ 49 mm, so the
        # F_eff-corrected estimate is ≈ 1 + 61/49 ≈ 2.25.
        assert e.magnification == pytest.approx(2.25)
        assert e.magnification_text.startswith("(est") is False  # has digits before
        assert "est" in e.magnification_text.lower()

    def test_groups_are_both_present(self, real_plan: list[PlanEntry]) -> None:
        groups = {e.group for e in real_plan}
        assert any(g.startswith("Group 1") for g in groups)
        assert any(g.startswith("Group 2") for g in groups)

    def test_observed_entry_not_estimated(self, real_plan: list[PlanEntry]) -> None:
        observed = next(
            e for e in real_plan
            if e.lens_config == "NIKKOR Z MC 105mm f/2.8 VR S + 6T + 5T"
        )
        assert observed.estimated is False
        assert observed.magnification == pytest.approx(1.6)
        assert observed.status == "reshoot"

    def test_inverse_ratio_form(self, real_plan: list[PlanEntry]) -> None:
        # Viltrox 85 + 61 mm tube: corrected estimate "1:1.18" using
        # m_native (= 0.13 at MFD) + L_tube/F_eff (= 61/85). The
        # observable here is just that the inverse-ratio parser
        # handles "1:N" forms — exact value tracks the reshoot plan.
        e = next(e for e in real_plan if "Viltrox 85" in e.lens_config and "61mm" in e.lens_config)
        assert e.magnification == pytest.approx(1 / 1.18, rel=1e-3)

    def test_numbers_are_strictly_increasing(self, real_plan: list[PlanEntry]) -> None:
        nums = [e.number for e in real_plan]
        assert nums == sorted(nums)
        assert len(set(nums)) == len(nums)

    def test_dropped_section_not_picked_up(self, real_plan: list[PlanEntry]) -> None:
        # "What was dropped" has rows like `| MC 105 + 51mm extension | … |`
        # — those have only 2 cells, not 4, and `int(cells[0])` fails. The
        # parser must skip them silently.
        for e in real_plan:
            assert "Sits between" not in e.lens_config


class TestParser:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_plan(tmp_path / "nope.md")

    def test_synthetic_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "plan.md"
        p.write_text(
            "## Group 1 — bench\n"
            "\n"
            "| # | Lens + attachment | Mag | Status |\n"
            "|---|---|---|---|\n"
            "| 1 | Lens A | 1:1 | reshoot |\n"
            "| 2 | Lens B | 2:1 (est.) | NEW |\n"
            "\n"
            "## Group 2 — field\n"
            "\n"
            "| # | Lens + attachment | Mag | Status |\n"
            "|---|---|---|---|\n"
            "| 3 | Lens C | 1:2 | reshoot |\n",
            encoding="utf-8",
        )
        entries = load_plan(p)
        assert [e.number for e in entries] == [1, 2, 3]
        assert entries[0].group.startswith("Group 1")
        assert entries[1].estimated is True
        assert entries[1].magnification == pytest.approx(2.0)
        assert entries[2].group.startswith("Group 2")
        assert entries[2].magnification == pytest.approx(0.5)

    def test_unparseable_mag_is_skipped(self, tmp_path: Path, capsys) -> None:
        p = tmp_path / "plan.md"
        p.write_text(
            "## Group 1 — bench\n"
            "| # | Lens | Mag | Status |\n"
            "|---|---|---|---|\n"
            "| 1 | Lens A | abc | reshoot |\n"
            "| 2 | Lens B | 1:1 | reshoot |\n",
            encoding="utf-8",
        )
        entries = load_plan(p)
        assert [e.number for e in entries] == [2]
        # Warning printed to stderr, not raised.
        err = capsys.readouterr().err
        assert "skipping row 1" in err
