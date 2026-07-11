"""Fail-closed config validation for VESPUQPlugin.from_config (R2WP-3).

A typo'd scientific setting must abort the run, not silently change the numerical experiment:
pre-fix, an unknown dtype fell back to float32, and an unparseable ``lambda_l2`` silently became
30.0 *and* mutated ``reg_method`` from ``fixed`` to ``lcurve``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

from vesp.uq import VESPUQPlugin
from vesp.uq.plugin import _validate_uq_config_keys

_BASE = {"model": {"n_source": 16}}


def _cfg(**overrides) -> dict:
    cfg = dict(_BASE)
    cfg.update(overrides)
    return cfg


def test_valid_config_still_builds():
    plugin = VESPUQPlugin.from_config(
        _cfg(dtype="float64", uq={"regularization": {"method": "fixed", "lambda_l2": 1.0e-6}})
    )
    assert plugin.reg_method == "fixed"
    assert plugin.lambda_l2 == pytest.approx(1.0e-6)


@pytest.mark.parametrize("bad", ["floatl64", "fp64", "float16", "quad"])
def test_unknown_dtype_raises(bad):
    with pytest.raises(ValueError, match="dtype"):
        VESPUQPlugin.from_config(_cfg(dtype=bad))


def test_unparseable_lambda_raises_and_never_mutates_reg_method():
    with pytest.raises(ValueError, match="lambda_l2"):
        VESPUQPlugin.from_config(
            _cfg(uq={"regularization": {"method": "fixed", "lambda_l2": "3O.0"}})
        )


def test_unknown_uq_key_raises():
    with pytest.raises(ValueError, match="unknown uq key"):
        VESPUQPlugin.from_config(_cfg(uq={"regularizatoin": {"method": "lcurve"}}))


def test_unknown_uq_subblock_key_raises():
    with pytest.raises(ValueError, match="unknown uq.risk key"):
        VESPUQPlugin.from_config(_cfg(uq={"risk": {"scorring": "max"}}))
    with pytest.raises(ValueError, match="unknown uq.conformal key"):
        VESPUQPlugin.from_config(_cfg(uq={"conformal": {"aplly": True}}))


def test_unknown_scoring_raises_at_construction():
    with pytest.raises(ValueError, match="unknown scoring"):
        VESPUQPlugin.from_config(_cfg(uq={"risk": {"scoring": "expected_err"}}))


def test_shipped_configs_pass_uq_key_validation():
    """No shipped config may depend on the old fail-open behavior."""

    root = Path(__file__).resolve().parents[1]
    paths = sorted(glob.glob(str(root / "configs" / "**" / "*.yaml"), recursive=True))
    assert paths, "expected shipped configs under configs/"
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            continue
        uq = cfg.get("uq", cfg.get("uncertainty"))
        if isinstance(uq, dict):
            _validate_uq_config_keys(uq)  # must not raise
