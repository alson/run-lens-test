"""Top-level exception types for the macro lens test package.

Kept in one module so test code and runner code can match on them
without importing from package internals.
"""


class MacroLensTestError(Exception):
    """Base for all package-raised errors."""


# ---- Camera ---------------------------------------------------------------

class CameraError(MacroLensTestError):
    pass


class CameraStateError(CameraError):
    """Camera came up wrong: minimal-MTP tree, mode dial off M, etc.

    The user message attached to this exception should be specific
    enough to drive the operator's next action (battery-pull, dial fix,
    …).
    """


class CameraCaptureError(CameraError):
    """Capture failed after the configured retry budget."""


class FECSanityError(CameraError):
    """FEC=0 vs FEC=-2 brightness ratio insufficient — most likely the
    SU-800 was mounted instead of the SB-5000 commander."""


# ---- StackShot ------------------------------------------------------------

class StackShotError(MacroLensTestError):
    pass


class StackShotProtocolError(StackShotError):
    """Frame parse failed: bad sync, bad checksum, length mismatch."""


class StackShotResponseError(StackShotError):
    """Slave returned a non-OK action/status code."""


class StackShotTimeout(StackShotError):
    """Move did not reach IDLE within the configured timeout."""


class StackShotUserAborted(StackShotError):
    """RAIL_STATUS_USER_ABORTED bit observed during a move."""


# ---- Session --------------------------------------------------------------

class SessionAborted(MacroLensTestError):
    """User aborted (Ctrl-C, jog 'q', empty lens-config)."""
