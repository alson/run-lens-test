"""Filename helpers used by the runner, calibrator, and adapter.

Centralised here so the same naming convention is applied across
sweep / probe / extension shots and the cropped output. The single
source of truth is the source JPEG's basename — the cropper writes
crops under ``<aperture>/cropped/<crop-name>/<source-basename>``.
"""

from __future__ import annotations


def step_filename(
    step_index: int,
    *,
    offset: int,
    width: int = 4,
    extension: str = ".jpg",
    replica: int = 0,
) -> str:
    """Build a ``step_NNNN[<suffix>].jpg`` filename for ``step_index``.

    The signed ``step_index`` is shifted by ``offset`` so the *filename*
    is always non-negative. With the default offset of 100, the main
    sweep starts at ``step_0100.jpg``, far-side extensions continue
    upward (``step_0130.jpg`` …), and near-side extensions go below
    the offset (``step_0099.jpg`` …).

    ``replica`` distinguishes multiple shots taken at the same
    (step, aperture) for noise averaging. Replica 0 has no suffix
    (``step_0100.jpg``) so existing offline-analysis tooling that
    expects single-shot filenames keeps working. Replicas 1, 2, …
    get suffixes ``_b``, ``_c``, … (``step_0100_b.jpg``,
    ``step_0100_c.jpg``).

    Raises ``ValueError`` if the result would be negative, wouldn't
    fit in the requested width, or ``replica`` is out of range
    (0..25; suffixes are single lowercase letters).
    """
    n = offset + step_index
    if n < 0:
        raise ValueError(
            f"step_index {step_index} + offset {offset} = {n} < 0; "
            "increase STEP_NUMBER_OFFSET in config.py"
        )
    formatted = f"{n:0{width}d}"
    if len(formatted) > width:
        raise ValueError(
            f"step_index {step_index} + offset {offset} = {n} exceeds "
            f"width {width}; increase STEP_NUMBER_WIDTH"
        )
    if replica < 0 or replica > 25:
        raise ValueError(
            f"replica {replica} out of range (0..25 supported)"
        )
    suffix = "" if replica == 0 else f"_{chr(ord('a') + replica)}"
    return f"step_{formatted}{suffix}{extension}"


def parse_step_filename(filename: str, *, offset: int) -> int:
    """Extract the signed ``step_index`` from a filename produced by
    :func:`step_filename`.

    Returns the ``step_index`` (signed). Raises ``ValueError`` if the
    filename doesn't match the convention.
    """
    if not filename.startswith("step_"):
        raise ValueError(f"not a step filename: {filename!r}")
    stem = filename[len("step_"):]
    # Strip extension.
    if "." in stem:
        stem = stem.split(".", 1)[0]
    try:
        return int(stem) - offset
    except ValueError as exc:
        raise ValueError(f"non-integer step in {filename!r}") from exc
