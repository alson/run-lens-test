"""Live cropping pipeline.

The session main loop drives shooting — the sharpness package gets
notified after each capture. ``LiveCollector`` schedules the cropping
on a thread pool so it runs concurrently with the next flash recycle
(typically 3–7 s of dead time per shot, plenty to crop one JPEG).

Threading model:

- One ``concurrent.futures.ThreadPoolExecutor`` owned by the collector.
- ``record_shot`` returns immediately after submission.
- A small thread-safe ``deque`` of futures keeps track of pending work;
  ``shutdown`` blocks until all crops are done.
- Per-shot results are merged into a guarded dict under a single
  ``threading.Lock``.

The collector deliberately knows nothing about the camera or the rail —
it accepts ``(rail_step, aperture, jpg_path)`` tuples and produces a
neutral data structure (``CollectedDataset``) that the detector and
advisor work on. This keeps it trivial to test and to drive from a
non-rig context (e.g. post-processing a directory of JPEGs).
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from sharpness.cropper import CropResult, crop_image_file


@dataclass(frozen=True)
class ShotSample:
    """One frame's worth of crop sizes, keyed by rail step + aperture."""

    rail_step: int
    aperture: float  # f-number, e.g. 8.0
    sizes: dict[str, int]  # crop name -> JPEG size in bytes


@dataclass
class CollectedDataset:
    """Aggregated samples across the session.

    ``samples`` is ordered by capture sequence (insertion order) but
    consumers should not rely on rail-step monotonicity — the camera
    main loop is the source of truth on ordering.
    """

    samples: list[ShotSample] = field(default_factory=list)

    def by_aperture(self) -> dict[float, list[ShotSample]]:
        out: dict[float, list[ShotSample]] = {}
        for s in self.samples:
            out.setdefault(s.aperture, []).append(s)
        for v in out.values():
            v.sort(key=lambda s: s.rail_step)
        return out

    def crop_series_for(
        self, aperture: float, crop_name: str
    ) -> list[tuple[int, int]]:
        """Sorted ``[(rail_step, size), ...]`` for one aperture / crop."""
        rows = [
            (s.rail_step, s.sizes[crop_name])
            for s in self.samples
            if s.aperture == aperture and crop_name in s.sizes
        ]
        rows.sort()
        return rows


# Type alias: a function with the same signature as ``crop_image_file``,
# returning a ``CropResult``. Tests can substitute a fake.
CropFn = Callable[[Path, Path], CropResult]


class LiveCollector:
    """Thread-pool-backed wrapper around the cropper.

    Use as a context manager to guarantee shutdown::

        with LiveCollector(out_dir) as c:
            for step, ap, jpg in shooting_loop():
                c.record_shot(step, ap, jpg)
        dataset = c.dataset

    Or call ``shutdown()`` explicitly when context-manager use is
    inconvenient.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        max_workers: int = 4,
        crop_fn: CropFn | None = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: deque[Future[None]] = deque()
        self._lock = threading.Lock()
        self._dataset = CollectedDataset()
        self._errors: list[BaseException] = []
        self._crop_fn = crop_fn or crop_image_file
        self._shutdown_called = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "LiveCollector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # -- public ------------------------------------------------------------

    def record_shot(
        self,
        rail_step: int,
        aperture: float,
        jpg_path: Path,
    ) -> Future[None]:
        """Schedule cropping for one shot. Returns immediately."""
        if self._shutdown_called:
            raise RuntimeError("record_shot called after shutdown")
        fut = self._executor.submit(
            self._process, rail_step, aperture, Path(jpg_path)
        )
        self._futures.append(fut)
        return fut

    def wait_for_pending(self) -> None:
        """Block until every ``record_shot()`` submitted so far has
        finished cropping. Subsequent ``record_shot()`` calls continue
        to use the same pool — this is a non-terminating barrier, used
        by the session runner between phases (sweep / probe / extension)
        so each phase's evaluator sees a complete dataset without
        requiring a fresh thread pool per phase.

        Errors from any of the awaited crops are surfaced the same way
        :meth:`shutdown` surfaces them — :class:`RuntimeError` with a
        chained cause from the first failure.
        """
        # Snapshot+clear under the lock so concurrent record_shot calls
        # don't race the Future drain. Errors are surfaced after release
        # of the lock to avoid holding it across Future.result().
        with self._lock:
            pending = list(self._futures)
            self._futures.clear()
        for fut in pending:
            # Future.result() re-raises any exception from the worker.
            # _process already records exceptions in self._errors before
            # re-raising, so we surface them via the standard message.
            try:
                fut.result()
            except BaseException:                    # noqa: BLE001
                # Suppress here — the canonical surface is the
                # self._errors list, which we re-raise below in shape
                # consistent with shutdown().
                pass
        with self._lock:
            errors = list(self._errors)
        if errors:
            # Clear errors so subsequent successful work after a
            # wait_for_pending() doesn't keep re-raising the same one.
            with self._lock:
                self._errors.clear()
            raise RuntimeError(
                f"{len(errors)} cropping task(s) failed; "
                f"first: {errors[0]!r}"
            ) from errors[0]

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work and (optionally) wait for in-flight crops."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._executor.shutdown(wait=wait)
        if self._errors:
            raise RuntimeError(
                f"{len(self._errors)} cropping task(s) failed; first: {self._errors[0]!r}"
            ) from self._errors[0]

    @property
    def dataset(self) -> CollectedDataset:
        """Thread-safe snapshot of the accumulated samples."""
        with self._lock:
            return CollectedDataset(samples=list(self._dataset.samples))

    def snapshot(self) -> CollectedDataset:
        """Alias of ``dataset`` — clearer at call sites."""
        return self.dataset

    # -- worker ------------------------------------------------------------

    def _process(self, rail_step: int, aperture: float, jpg_path: Path) -> None:
        try:
            result = self._crop_fn(jpg_path, self._out_dir)
        except BaseException as exc:  # noqa: BLE001 — re-raise via shutdown
            with self._lock:
                self._errors.append(exc)
            raise
        sample = ShotSample(rail_step=rail_step, aperture=aperture, sizes=result.sizes)
        with self._lock:
            self._dataset.samples.append(sample)


def collect_from_dir(
    paths: Iterable[tuple[int, float, Path]],
    out_dir: Path,
    *,
    max_workers: int = 4,
    crop_fn: CropFn | None = None,
) -> CollectedDataset:
    """Convenience entry: drive the collector synchronously from an iterable.

    Useful for post-processing a directory of pre-captured JPEGs without
    spinning up the full session machinery.
    """
    with LiveCollector(out_dir, max_workers=max_workers, crop_fn=crop_fn) as c:
        for step, ap, jpg in paths:
            c.record_shot(step, ap, jpg)
    return c.dataset
