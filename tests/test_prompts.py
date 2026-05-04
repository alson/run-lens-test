"""Tests for ui.prompts — slug + step-count + lens-config input."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ui.prompts import (
    confirm,
    prompt_lens_config,
    prompt_step_count,
    slugify_lens_config,
)


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
