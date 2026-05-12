"""Tests for ui.prompts — slug + step-count + lens-config input."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ui.prompts import (
    OTHER,
    _format_choice_title,
    confirm,
    prompt_lens_config,
    prompt_lens_config_from_plan,
    prompt_step_count,
    slugify_lens_config,
)
from ui.reshoot_plan import PlanEntry


class TestSlug:
    def test_replaces_colon_and_slash(self) -> None:
        assert (
            slugify_lens_config("Sigma 105/2.8 EX DG: macro test 1")
            == "Sigma 105_2.8 EX DG_ macro test 1"
        )

    def test_preserves_internal_spaces_and_strips_edges(self) -> None:
        assert slugify_lens_config("  abc def  ") == "abc def"

    def test_no_changes_on_clean_text(self) -> None:
        assert slugify_lens_config("Voigtlander 65mm APO") == "Voigtlander 65mm APO"


class TestPromptLensConfig:
    def test_returns_none_on_empty(self) -> None:
        assert prompt_lens_config(input_func=lambda _: "") is None

    def test_returns_none_on_whitespace_only(self) -> None:
        assert prompt_lens_config(input_func=lambda _: "   ") is None

    def test_returns_none_on_eof(self) -> None:
        def _eof(_p: str) -> str:
            raise EOFError
        assert prompt_lens_config(input_func=_eof) is None

    def test_returns_trimmed_text(self) -> None:
        assert prompt_lens_config(input_func=lambda _: "  Sigma 105  ") == "Sigma 105"

    def test_short_input_quits(self) -> None:
        # Inputs shorter than min_length (default 5) act as quit so a
        # stray ``q`` / ``quit`` / typo doesn't slugify to a junk
        # lens-config dir before aborting at the jog.
        for s in ("q", "quit", "abc", "abcd"):
            assert prompt_lens_config(input_func=lambda _, s=s: s) is None

    def test_five_char_input_accepted(self) -> None:
        # Boundary: exactly min_length characters is valid.
        assert prompt_lens_config(input_func=lambda _: "abcde") == "abcde"

    def test_min_length_override(self) -> None:
        # Caller can lower / raise the gate as needed.
        assert prompt_lens_config(
            input_func=lambda _: "abc", min_length=2
        ) == "abc"


class TestPromptMagnification:
    def test_accepts_ratio(self) -> None:
        from ui.prompts import prompt_magnification
        assert prompt_magnification(input_func=lambda _: "1:1") == 1.0
        assert prompt_magnification(input_func=lambda _: "1.5:1") == 1.5

    def test_accepts_bare_float(self) -> None:
        from ui.prompts import prompt_magnification
        assert prompt_magnification(input_func=lambda _: "0.5") == 0.5

    def test_reprompts_on_bad_input(self, capsys) -> None:
        from ui.prompts import prompt_magnification
        # Yields two bad inputs then a good one.
        seq = iter(["abc", "0", "1:1"])
        result = prompt_magnification(input_func=lambda _: next(seq))
        assert result == 1.0
        # The bad-input feedback was printed at least twice.
        captured = capsys.readouterr()
        assert captured.out.count("try e.g.") >= 2

    def test_eof_reraises(self) -> None:
        from ui.prompts import prompt_magnification
        def _eof(_p: str) -> str:
            raise EOFError
        with pytest.raises(EOFError):
            prompt_magnification(input_func=_eof)


class TestPromptStepCount:
    def test_accepts_valid_int(self) -> None:
        assert prompt_step_count(input_func=lambda _: "30") == 30

    def test_re_prompts_on_non_int(self, capsys) -> None:
        inputs = iter(["abc", "5"])
        result = prompt_step_count(input_func=lambda _: next(inputs))
        assert result == 5
        out = capsys.readouterr().out
        assert "not an integer" in out

    def test_re_prompts_below_minimum(self, capsys) -> None:
        inputs = iter(["1", "2"])
        result = prompt_step_count(input_func=lambda _: next(inputs), minimum=2)
        assert result == 2
        out = capsys.readouterr().out
        assert "must be >= 2" in out

    def test_eof_raises(self) -> None:
        def _eof(_p: str) -> str:
            raise EOFError
        with pytest.raises(ValueError, match="EOF"):
            prompt_step_count(input_func=_eof)


class TestPromptLensConfigFromPlan:
    @staticmethod
    def _entries() -> list[PlanEntry]:
        return [
            PlanEntry(
                group="Group 1 — bench",
                number=1,
                lens_config="Lens A",
                magnification=1.0,
                magnification_text="1:1",
                status="reshoot",
                estimated=False,
            ),
            PlanEntry(
                group="Group 1 — bench",
                number=2,
                lens_config="Lens B",
                magnification=2.0,
                magnification_text="2:1 (est.)",
                status="NEW",
                estimated=True,
            ),
            PlanEntry(
                group="Group 2 — field",
                number=3,
                lens_config="Lens C",
                magnification=0.5,
                magnification_text="1:2",
                status="reshoot",
                estimated=False,
            ),
        ]

    def test_pick_preset_returns_entry(self) -> None:
        entries = self._entries()
        # Fake `ask`: pick the second entry.
        result = prompt_lens_config_from_plan(
            entries, completed=set(),
            ask=lambda choices: entries[1],
        )
        assert isinstance(result, PlanEntry)
        assert result.lens_config == "Lens B"
        assert result.magnification == 2.0

    def test_other_sentinel_returned(self) -> None:
        entries = self._entries()
        result = prompt_lens_config_from_plan(
            entries, completed=set(),
            ask=lambda choices: OTHER,
        )
        assert result == OTHER

    def test_quit_returns_none(self) -> None:
        entries = self._entries()
        result = prompt_lens_config_from_plan(
            entries, completed=set(),
            ask=lambda choices: None,
        )
        assert result is None

    def test_keyboard_interrupt_returns_none(self) -> None:
        entries = self._entries()

        def _raise(_choices):
            raise KeyboardInterrupt
        assert prompt_lens_config_from_plan(
            entries, completed=set(), ask=_raise,
        ) is None

    def test_eof_returns_none(self) -> None:
        entries = self._entries()

        def _raise(_choices):
            raise EOFError
        assert prompt_lens_config_from_plan(
            entries, completed=set(), ask=_raise,
        ) is None

    def test_done_marker_present_for_completed_slug(self) -> None:
        entries = self._entries()
        completed = {slugify_lens_config("Lens A")}
        captured: list = []
        prompt_lens_config_from_plan(
            entries, completed=completed,
            ask=lambda choices: captured.extend(choices) or None,
        )
        # Choices list contains separators + Choice objects + tail
        # Choice/Separator entries; titles for entries are formatted by
        # _format_choice_title. We assert the marker shows up exactly
        # for the completed entry.
        titles = [getattr(c, "title", "") for c in captured]
        marked = [t for t in titles if "[done]" in str(t)]
        unmarked = [
            t for t in titles
            if isinstance(t, str)
            and ("Lens" in t)
            and "[done]" not in t
        ]
        assert len(marked) == 1
        assert "Lens A" in str(marked[0])
        assert any("Lens B" in u for u in unmarked)
        assert any("Lens C" in u for u in unmarked)

    def test_groups_render_separators(self) -> None:
        import questionary
        entries = self._entries()
        captured: list = []
        prompt_lens_config_from_plan(
            entries, completed=set(),
            ask=lambda choices: captured.extend(choices) or None,
        )
        sep_titles = [
            c.title for c in captured if isinstance(c, questionary.Separator)
        ]
        assert any("Group 1" in str(t) for t in sep_titles)
        assert any("Group 2" in str(t) for t in sep_titles)

    def test_format_choice_title_marker(self) -> None:
        e = self._entries()[0]
        marked = _format_choice_title(e, {"Lens A"}, slugify_lens_config)
        unmarked = _format_choice_title(e, set(), slugify_lens_config)
        assert "[done]" in marked
        assert "[done]" not in unmarked
        assert "Lens A" in marked
        assert "1:1" in marked

    def test_f16_only_marker_present(self) -> None:
        e = PlanEntry(
            group="Group 1 — bench",
            number=99,
            lens_config="Lens F16-only",
            magnification=1.0,
            magnification_text="1:1",
            status="reshoot (F16-only)",
            estimated=False,
        )
        title = _format_choice_title(e, set(), slugify_lens_config)
        assert "[F16-opt]" in title
        assert "Lens F16-only" in title

    def test_done_takes_precedence_over_f16_only(self) -> None:
        # If a lens is both completed and F16-only, the done marker wins.
        e = PlanEntry(
            group="Group 1 — bench",
            number=99,
            lens_config="Lens F16-only",
            magnification=1.0,
            magnification_text="1:1",
            status="reshoot (F16-only)",
            estimated=False,
        )
        title = _format_choice_title(
            e, {slugify_lens_config("Lens F16-only")}, slugify_lens_config
        )
        assert "[done]" in title
        assert "[F16-opt]" not in title

    def test_status_without_f16_only_gets_no_marker(self) -> None:
        # Status containing "F16" but not "F16-only" must not match
        # (e.g. a hypothetical "reshoot (F16+truncated)").
        e = PlanEntry(
            group="Group 1 — bench",
            number=1,
            lens_config="Lens X",
            magnification=1.0,
            magnification_text="1:1",
            status="reshoot",  # plain status
            estimated=False,
        )
        title = _format_choice_title(e, set(), slugify_lens_config)
        assert "[F16-opt]" not in title
        assert "[done]" not in title

    def test_f16_only_entries_sink_to_bottom_of_group(self) -> None:
        # Group with normal entries, an F16-only entry interleaved, and
        # entries from a second group. F16-only must appear after the
        # normal ones in its own group; group order is preserved.
        entries = [
            PlanEntry(
                group="Group 1", number=1, lens_config="L1",
                magnification=1.0, magnification_text="1:1",
                status="reshoot", estimated=False,
            ),
            PlanEntry(
                group="Group 1", number=2, lens_config="L2-F16",
                magnification=1.0, magnification_text="1:1",
                status="reshoot (F16-only)", estimated=False,
            ),
            PlanEntry(
                group="Group 1", number=3, lens_config="L3",
                magnification=1.0, magnification_text="1:1",
                status="NEW", estimated=False,
            ),
            PlanEntry(
                group="Group 2", number=4, lens_config="L4-F16",
                magnification=1.0, magnification_text="1:1",
                status="reshoot (F16-only)", estimated=False,
            ),
            PlanEntry(
                group="Group 2", number=5, lens_config="L5",
                magnification=1.0, magnification_text="1:1",
                status="reshoot", estimated=False,
            ),
        ]
        captured: list = []
        prompt_lens_config_from_plan(
            entries, completed=set(),
            ask=lambda choices: captured.extend(choices) or None,
        )
        # Extract just the entry-titled choices in shown order.
        titles_in_order = [
            getattr(c, "title", None)
            for c in captured
            if hasattr(c, "value") and isinstance(getattr(c, "value", None), PlanEntry)
        ]
        # Within Group 1: L1, L3, L2-F16 (F16-only sinks)
        # Within Group 2: L5, L4-F16
        idx = {name: i for i, t in enumerate(titles_in_order) for name in [
            "L1", "L2-F16", "L3", "L4-F16", "L5"
        ] if name in str(t)}
        assert idx["L1"] < idx["L3"] < idx["L2-F16"]
        assert idx["L5"] < idx["L4-F16"]
        # Group order preserved (G1 before G2)
        assert idx["L2-F16"] < idx["L5"]


class TestConfirm:
    def test_enter_confirms(self) -> None:
        assert confirm("Press ENTER: ", input_func=lambda _: "") is True

    def test_q_aborts(self) -> None:
        assert confirm("Press ENTER: ", input_func=lambda _: "q") is False

    def test_quit_aborts(self) -> None:
        assert confirm("Press ENTER: ", input_func=lambda _: "QUIT") is False

    def test_eof_aborts(self) -> None:
        def _eof(_p: str) -> str:
            raise EOFError
        assert confirm("Press ENTER: ", input_func=_eof) is False


class TestPromptMagnificationIndicated:
    def test_accepts_ratio(self) -> None:
        from ui.prompts import prompt_magnification_indicated
        result = prompt_magnification_indicated(
            default=1.0, input_func=lambda _: "1:2",
        )
        assert result == 0.5

    def test_accepts_bare_float(self) -> None:
        from ui.prompts import prompt_magnification_indicated
        result = prompt_magnification_indicated(
            default=1.0, input_func=lambda _: "0.5",
        )
        assert result == 0.5

    def test_empty_returns_default(self) -> None:
        from ui.prompts import prompt_magnification_indicated
        result = prompt_magnification_indicated(
            default=0.7, input_func=lambda _: "",
        )
        assert result == 0.7

    def test_whitespace_returns_default(self) -> None:
        """Trimmed empty input is treated as empty → default."""
        from ui.prompts import prompt_magnification_indicated
        result = prompt_magnification_indicated(
            default=0.3, input_func=lambda _: "   ",
        )
        assert result == 0.3

    def test_reprompts_on_bad_input(self, capsys) -> None:
        from ui.prompts import prompt_magnification_indicated
        seq = iter(["abc", "0", "1:2"])
        result = prompt_magnification_indicated(
            default=1.0, input_func=lambda _: next(seq),
        )
        assert result == 0.5
        captured = capsys.readouterr()
        assert captured.out.count("try e.g.") >= 2

    def test_eof_reraises(self) -> None:
        from ui.prompts import prompt_magnification_indicated
        def _eof(_p: str) -> str:
            raise EOFError
        with pytest.raises(EOFError):
            prompt_magnification_indicated(default=1.0, input_func=_eof)


class TestPromptFocalLength:
    def test_accepts_integer(self) -> None:
        from ui.prompts import prompt_focal_length
        assert prompt_focal_length(input_func=lambda _: "55") == 55.0

    def test_accepts_decimal(self) -> None:
        from ui.prompts import prompt_focal_length
        assert prompt_focal_length(input_func=lambda _: "55.5") == 55.5

    def test_reprompts_on_non_number(self, capsys) -> None:
        from ui.prompts import prompt_focal_length
        seq = iter(["abc", "55"])
        result = prompt_focal_length(input_func=lambda _: next(seq))
        assert result == 55.0
        out = capsys.readouterr().out
        assert "not a number" in out

    def test_reprompts_on_nonpositive(self, capsys) -> None:
        from ui.prompts import prompt_focal_length
        seq = iter(["0", "-5", "55"])
        result = prompt_focal_length(input_func=lambda _: next(seq))
        assert result == 55.0
        out = capsys.readouterr().out
        assert "must be > 0" in out

    def test_eof_reraises(self) -> None:
        from ui.prompts import prompt_focal_length
        def _eof(_p: str) -> str:
            raise EOFError
        with pytest.raises(EOFError):
            prompt_focal_length(input_func=_eof)


class TestPromptMaxAperture:
    def test_accepts_decimal(self) -> None:
        from ui.prompts import prompt_max_aperture
        assert prompt_max_aperture(input_func=lambda _: "2.8") == 2.8

    def test_accepts_f_prefix(self) -> None:
        from ui.prompts import prompt_max_aperture
        assert prompt_max_aperture(input_func=lambda _: "f/2.8") == 2.8
        assert prompt_max_aperture(input_func=lambda _: "F2.8") == 2.8

    def test_reprompts_on_non_number(self, capsys) -> None:
        from ui.prompts import prompt_max_aperture
        seq = iter(["abc", "2.8"])
        result = prompt_max_aperture(input_func=lambda _: next(seq))
        assert result == 2.8
        out = capsys.readouterr().out
        assert "not a number" in out


class TestPromptTurnRing:
    def test_enter_confirms(self) -> None:
        from ui.prompts import prompt_turn_ring
        result = prompt_turn_ring(
            "4",
            ring_stop=2.8,
            m_indicated=0.5,
            input_func=lambda _: "",
        )
        assert result is True

    def test_q_aborts(self) -> None:
        from ui.prompts import prompt_turn_ring
        result = prompt_turn_ring(
            "4",
            ring_stop=2.8,
            m_indicated=0.5,
            input_func=lambda _: "q",
        )
        assert result is False

    def test_message_shows_only_ring_stop(self) -> None:
        """The prompt names only the ring f-number the user dials.
        The effective label is metadata for the file path / analysis
        and is intentionally omitted from the capture-time prompt
        to keep the operator focused on the action."""
        from ui.prompts import prompt_turn_ring
        seen_prompts: list[str] = []
        prompt_turn_ring(
            "4",
            ring_stop=2.8,
            m_indicated=0.5,
            input_func=lambda p: seen_prompts.append(p) or "",
        )
        assert seen_prompts, "prompt was never invoked"
        msg = seen_prompts[0]
        assert "f/2.8" in msg
        assert "effective" not in msg
        assert "m_indicated" not in msg

    def test_whole_stop_ring_renders_without_decimal(self) -> None:
        from ui.prompts import prompt_turn_ring
        seen_prompts: list[str] = []
        prompt_turn_ring(
            "11",
            ring_stop=8.0,
            m_indicated=0.5,
            input_func=lambda p: seen_prompts.append(p) or "",
        )
        msg = seen_prompts[0]
        assert "f/8" in msg
        assert "f/8.0" not in msg
        assert "effective" not in msg
