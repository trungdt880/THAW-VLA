"""Training-only distillation modules (FastWAM teacher -> Qwen student)."""

from .fastwam_repa import (  # noqa: F401
    FastWAMRepaHead,
    RepaProjector,
    find_image_token_runs,
    pool_student_image_tokens,
)
from .ladder_readout import LadderReadout  # noqa: F401

# FastWAMFutureHead imports the action-head DiT (heavier deps); import lazily on use.
# VJepaOnlineTeacher imports app.vjepa_droid (needs vjepa2 on path); import lazily on use.
__all__ = [
    "FastWAMRepaHead",
    "RepaProjector",
    "find_image_token_runs",
    "pool_student_image_tokens",
    "LadderReadout",
    "FastWAMFutureHead",
    "VJepaOnlineTeacher",
]


def __getattr__(name):  # PEP 562 lazy import
    if name == "FastWAMFutureHead":
        from .fastwam_future import FastWAMFutureHead

        return FastWAMFutureHead
    if name == "VJepaOnlineTeacher":
        from .vjepa_online_teacher import VJepaOnlineTeacher

        return VJepaOnlineTeacher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
