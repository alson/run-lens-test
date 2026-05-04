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
"""
