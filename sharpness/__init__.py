"""Real-time sharpness analysis for the lens-test rig.

This package mirrors the offline ``analyse-lens-test/rail-coverage`` tool
but runs *as the session is shooting*: each captured JPEG is cropped into
17 fixed regions on a thread pool, file sizes are recorded as a proxy
for sharpness, and at the end of the rail sweep the per-aperture series
are checked for bell-curve truncation and off-target crops. Because the
user is still at the rig when this runs, the recommendation includes
both a direction (extend rail forward / backward) and an estimated
extension distance — unlike the offline tool, where subject-distance
ranges can't be reproduced and direction information is useless.

Public API::

    from sharpness import LiveCollector, advise

    with LiveCollector(out_dir) as c:
        for step, ap, jpg in shooting_loop():
            c.record_shot(step, ap, jpg)
    for ap, rec in sorted(advise(c.dataset).items()):
        print(rec.summary())
"""

from sharpness.advisor import Direction, Recommendation, Verdict, advise
from sharpness.collector import CollectedDataset, LiveCollector, ShotSample
from sharpness.crops import CROP_NAMES, CropBox, all_crop_boxes
from sharpness.detect import CropAnalysis, CropStatus

__all__ = [
    "CROP_NAMES",
    "CollectedDataset",
    "CropAnalysis",
    "CropBox",
    "CropStatus",
    "Direction",
    "LiveCollector",
    "Recommendation",
    "ShotSample",
    "Verdict",
    "advise",
    "all_crop_boxes",
]
