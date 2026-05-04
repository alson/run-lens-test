"""Pure widget-tree helpers — no gphoto2 import required at the call
site.

The gphoto2 widget objects expose ``get_name``, ``get_value``,
``set_value``, ``get_type``, ``get_readonly``, ``count_children``,
``get_child(i)``, ``count_choices``, ``get_choice(i)``. These
functions just walk that interface, so tests can drive them with any
duck-typed object.
"""

from __future__ import annotations

from typing import Iterable


def find(root, name: str):
    """Depth-first search for a widget by name. Returns ``None`` if not found."""
    if root.get_name() == name:
        return root
    for i in range(root.count_children()):
        result = find(root.get_child(i), name)
        if result is not None:
            return result
    return None


def count_widgets(root) -> int:
    """Total widget count in the subtree (including ``root``)."""
    n = 1
    for i in range(root.count_children()):
        n += count_widgets(root.get_child(i))
    return n


def widget_choices(widget) -> list[str]:
    return [widget.get_choice(i) for i in range(widget.count_choices())]


def first_matching_choice(widget, candidates: Iterable[str]) -> str | None:
    """Return the first ``candidate`` that appears in the widget's
    choices list, or ``None`` if none match.

    Used to handle the Z7's quirky shutterspeed labelling (``"1/200"``
    vs ``"0.0050s"`` depending on firmware).
    """
    choices = widget_choices(widget)
    for cand in candidates:
        if cand in choices:
            return cand
    return None
