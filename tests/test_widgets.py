"""Tests for camera.widgets (gphoto2-free)."""

from __future__ import annotations

from unittest.mock import MagicMock

from camera.widgets import (
    count_widgets,
    find,
    first_matching_choice,
    widget_choices,
)


def make_widget(
    name: str,
    value=None,
    *,
    choices: list[str] | None = None,
    children: list = (),
):
    """Build a duck-typed widget that supports the gphoto2 API used by
    :mod:`camera.widgets`."""
    w = MagicMock()
    w.get_name.return_value = name
    w.get_value.return_value = value
    w.count_children.return_value = len(children)
    w.get_child.side_effect = lambda i: children[i]
    if choices is not None:
        w.count_choices.return_value = len(choices)
        w.get_choice.side_effect = lambda i: choices[i]
    else:
        w.count_choices.return_value = 0
    return w


class TestFind:
    def test_root_match(self) -> None:
        root = make_widget("main")
        assert find(root, "main") is root

    def test_nested_match(self) -> None:
        leaf = make_widget("iso", "100")
        section = make_widget("settings", children=[leaf])
        root = make_widget("main", children=[section])
        assert find(root, "iso") is leaf

    def test_missing_returns_none(self) -> None:
        root = make_widget("main", children=[make_widget("aperture")])
        assert find(root, "iso") is None


class TestCountWidgets:
    def test_single_widget(self) -> None:
        assert count_widgets(make_widget("only")) == 1

    def test_deep_tree(self) -> None:
        leaf_a = make_widget("a")
        leaf_b = make_widget("b")
        section = make_widget("section", children=[leaf_a, leaf_b])
        root = make_widget("main", children=[section])
        # main + section + a + b = 4
        assert count_widgets(root) == 4


class TestWidgetChoices:
    def test_returns_all_choices(self) -> None:
        w = make_widget("iso", choices=["100", "200", "400"])
        assert widget_choices(w) == ["100", "200", "400"]

    def test_no_choices(self) -> None:
        w = make_widget("range", choices=[])
        assert widget_choices(w) == []


class TestFirstMatchingChoice:
    def test_returns_first_present_candidate(self) -> None:
        w = make_widget("shutterspeed", choices=["0.0050s", "0.01s"])
        # Z7 firmware variant: try "1/200" first, fall back to "0.0050s".
        assert first_matching_choice(w, ["1/200", "0.0050s"]) == "0.0050s"

    def test_returns_none_when_no_match(self) -> None:
        w = make_widget("shutterspeed", choices=["1.0s", "0.5s"])
        assert first_matching_choice(w, ["1/200", "0.0050s"]) is None

    def test_first_match_wins(self) -> None:
        w = make_widget("x", choices=["a", "b"])
        assert first_matching_choice(w, ["a", "b"]) == "a"
