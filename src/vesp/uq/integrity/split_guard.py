"""G2 -- split-leakage & oracle-isolation guard.

Two failure modes silently inflate UQ results: (a) selecting or tuning on the *test* split, and
(b) letting the true-error *oracle* leak into a score it is supposed to be judged against. This
module makes both trip a loud error rather than pass unnoticed, using a deliberately lightweight,
opt-in-at-the-seams design -- there is **no** pervasive tensor wrapping. The pattern is:

1. :func:`tag` a boundary object (the held/test error, the oracle true-error) with its
   :class:`Split` and a ``role`` string, producing a thin :class:`Tagged` wrapper.
2. Wrap the region that must not see the test labels in :func:`assert_no_test_access`. Inside it,
   :func:`reveal` of a ``TEST``-tagged object raises :class:`SplitLeakageError`. Outside it,
   ``reveal`` returns the underlying data (legitimate post-selection metric evaluation).
3. Call :func:`forbid_oracle` at the top of any score builder; it raises
   :class:`OracleLeakageError` if handed an ``oracle``-tagged array.

The leakage flag is thread-local, so concurrent runs (or nested guards) do not interfere. Untagged
objects always pass through every helper untouched, so the guard can be wired in at the seams
without disturbing the rest of the pipeline.
"""

from __future__ import annotations

import enum
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class Split(enum.Enum):
    """Which data split a tagged object belongs to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Tagged:
    """A boundary object carrying split / role provenance without copying its payload.

    ``data`` is the wrapped payload (typically a tensor); ``split`` records the split it came from;
    ``role`` is a short tag (``"error"``, ``"oracle"``, ``"positions"``, ...). The guard helpers
    inspect these fields -- the payload itself is never mutated.
    """

    data: Any
    split: Split
    role: str = ""


class SplitLeakageError(RuntimeError):
    """Raised when a ``TEST``-tagged object is read inside an :func:`assert_no_test_access` region."""


class OracleLeakageError(RuntimeError):
    """Raised when an ``oracle``-tagged array reaches a score builder via :func:`forbid_oracle`."""


_state = threading.local()


def _forbid_test_active() -> bool:
    return bool(getattr(_state, "forbid_test", False))


def tag(data: Any, *, split: Split, role: str = "") -> Tagged:
    """Wrap ``data`` with split / role provenance (idempotent re-tag if already :class:`Tagged`)."""

    if isinstance(data, Tagged):
        return Tagged(data.data, split, role or data.role)
    return Tagged(data, split, role)


def reveal(obj: Any, *, allow_test: bool = False) -> Any:
    """Return the underlying payload of ``obj`` (or ``obj`` itself if it is not :class:`Tagged`).

    If ``obj`` is ``TEST``-tagged and an :func:`assert_no_test_access` region is active, this raises
    :class:`SplitLeakageError` -- unless ``allow_test=True`` explicitly authorizes the read (used for
    the legitimate, post-selection metric evaluation that *must* touch the test labels).
    """

    if not isinstance(obj, Tagged):
        return obj
    if obj.split is Split.TEST and _forbid_test_active() and not allow_test:
        role = f" (role={obj.role!r})" if obj.role else ""
        raise SplitLeakageError(
            f"read of TEST-tagged data{role} inside a no-test-access region -- "
            "selection / tuning must not see the test split"
        )
    return obj.data


def forbid_oracle(*arrays: Any) -> None:
    """Raise :class:`OracleLeakageError` if any argument is an ``oracle``-tagged :class:`Tagged`.

    Call at the top of a score builder to assert the true-error oracle never reaches it. Untagged
    arguments (and non-oracle tags) are ignored, so this is safe to sprinkle at score-function
    entry points without changing their behavior.
    """

    for arr in arrays:
        if isinstance(arr, Tagged) and arr.role == "oracle":
            raise OracleLeakageError(
                "an oracle-tagged array reached a score builder -- the true-error oracle must "
                "never enter a risk score"
            )


@contextmanager
def assert_no_test_access():
    """Context manager: inside the block, revealing ``TEST``-tagged data raises (G2 selection guard).

    Nesting is safe -- the previous flag is restored on exit, so an inner guard does not clear an
    outer one. Thread-local, so it does not leak across threads.
    """

    prev = getattr(_state, "forbid_test", False)
    _state.forbid_test = True
    try:
        yield
    finally:
        _state.forbid_test = prev
