"""Tests for G2 -- split-leakage & oracle-isolation guard."""

from __future__ import annotations

import threading

import pytest
import torch

from vesp.uq.integrity.split_guard import (
    OracleLeakageError,
    Split,
    SplitLeakageError,
    Tagged,
    assert_no_test_access,
    forbid_oracle,
    reveal,
    tag,
)


def test_tag_wraps_and_reveal_unwraps_outside_guard():
    x = torch.arange(5)
    t = tag(x, split=Split.TEST, role="oracle")
    assert isinstance(t, Tagged) and t.split is Split.TEST and t.role == "oracle"
    # outside any guard, revealing test data is legitimate (post-selection metric evaluation)
    assert torch.equal(reveal(t), x)


def test_untagged_objects_pass_through_all_helpers():
    x = torch.zeros(3)
    assert reveal(x) is x          # not Tagged -> returned as-is
    forbid_oracle(x, None, "str")  # no Tagged-oracle present -> no raise
    with assert_no_test_access():
        assert reveal(x) is x      # untagged is unaffected by the guard


def test_reading_test_label_inside_guard_raises():
    secret = tag(torch.ones(4), split=Split.TEST, role="oracle")
    with assert_no_test_access():
        with pytest.raises(SplitLeakageError):
            reveal(secret)


def test_allow_test_authorizes_a_deliberate_read_inside_guard():
    secret = tag(torch.ones(4), split=Split.TEST, role="oracle")
    with assert_no_test_access():
        revealed = reveal(secret, allow_test=True)  # explicit, authorized
    assert torch.equal(revealed, torch.ones(4))


def test_non_test_splits_are_not_blocked():
    val = tag(torch.ones(2), split=Split.VAL, role="error")
    with assert_no_test_access():
        assert torch.equal(reveal(val), torch.ones(2))  # VAL is allowed during selection


def test_forbid_oracle_trips_on_oracle_tag():
    oracle = tag(torch.ones(3), split=Split.TEST, role="oracle")
    with pytest.raises(OracleLeakageError):
        forbid_oracle(torch.zeros(3), oracle)


def test_forbid_oracle_ignores_non_oracle_tags():
    positions = tag(torch.zeros(3), split=Split.TRAIN, role="positions")
    forbid_oracle(positions)  # role != "oracle" -> no raise


def test_guard_is_reentrant_and_restores_state():
    secret = tag(torch.ones(1), split=Split.TEST, role="oracle")
    with assert_no_test_access():
        with assert_no_test_access():
            with pytest.raises(SplitLeakageError):
                reveal(secret)
        # inner guard exited but outer is still active
        with pytest.raises(SplitLeakageError):
            reveal(secret)
    # both guards exited -> reads are legal again
    assert torch.equal(reveal(secret), torch.ones(1))


def test_guard_flag_is_thread_local():
    secret = tag(torch.ones(1), split=Split.TEST, role="oracle")
    other_thread_ok = {}

    def worker():
        # no guard active on this thread, even though the main thread holds one
        other_thread_ok["value"] = torch.equal(reveal(secret), torch.ones(1))

    with assert_no_test_access():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert other_thread_ok["value"] is True
