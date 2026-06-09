"""Post-hoc conformal calibration for VESP-UQ predictive force-error uncertainty.

VESP-UQ is itself a fitted uncertainty model: its nominal predictive intervals are only as
trustworthy as the linear-Gaussian assumptions behind them, and they can under- or over-cover the
true force-model error on held-out samples. This module adds a *post-hoc* split-conformal wrapper
that does not assume Gaussian correctness. Given paired held-out residual samples it learns a
single multiplicative scale ``c`` so that the calibrated predictive band ``c * predicted_error``
empirically covers the true force error at the requested ``1 - alpha`` level (with the standard
finite-sample correction).

The nonconformity score is the normalized residual ``s_i = true_error_i / predicted_error_i``, and
the conformal scale is its conservative ``(1 - alpha)`` quantile. Larger ``predicted_error`` (and a
larger learned scale) therefore means a *more conservative* interval. Because it is a wrapper, this
never replaces :class:`~vesp.uq.plugin.VESPUQPlugin`; it sits on top of the plugin's predictions.

Everything here concerns **force-model error** (``a_reference - a_surrogate``). It is not a
position-accuracy calibrator and makes no statement about trajectory position accuracy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

# Reduction modes shared by fit/apply/coverage. They control how (possibly vector) predicted and
# true errors are collapsed into the per-sample scalars the conformal scale operates on.
CONFORMAL_MODES = ("norm", "component_max", "mahalanobis")

# Trajectory-level calibration accepts the same reductions plus ``p95`` (the default), which treats
# the already-aggregated per-trajectory scalars as-is.
TRAJECTORY_MODES = ("p95", *CONFORMAL_MODES)

_TINY = 1.0e-30


@dataclass
class ConformalCalibrator:
    """Learned post-hoc conformal scale for force-error predictive uncertainty.

    ``scale`` multiplies a predicted error/uncertainty so that ``scale * predicted_error``
    empirically covers the true force error at the ``1 - alpha`` level on exchangeable held-out
    samples. ``scale >= 1`` indicates the raw predictions were under-covering (and were inflated);
    ``scale < 1`` indicates over-covering (the band was tightened). The remaining fields record how
    the scale was derived so a report can be fully reproduced.
    """

    scale: float
    alpha: float
    mode: str
    n_calibration: int
    quantile_level: float
    coverage_before: float
    coverage_after: float

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def apply(self, predicted_error):
        """Return the calibrated (conservatively scaled) predicted error."""

        return apply_conformal_scale(predicted_error, self.scale)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_coverage"] = self.target_coverage
        return d


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return a


def _as_tensor(values, name: str) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float64)
    if t.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return t


def _reduce(values, mode: str, name: str) -> torch.Tensor:
    """Collapse predicted/true errors into a per-sample nonconformity scalar (nonnegative).

    - ``norm``: vector error magnitude ``||e||`` (per-row L2 norm for ``(N, k)`` input; ``|e|`` for
      ``(N,)`` input).
    - ``component_max``: maximum absolute component ``max_j |e_j|``.
    - ``mahalanobis``: the input is already a 1-D normalized residual score (e.g. a per-sample
      Mahalanobis distance and its predicted radius); no further reduction is applied. Pass
      pre-reduced 1-D arrays -- this keeps the module dependency-light and free of covariance
      inversion.
    """

    if mode not in CONFORMAL_MODES:
        raise ValueError(f"mode must be one of {CONFORMAL_MODES}, got {mode!r}")
    t = _as_tensor(values, name)
    if mode == "mahalanobis":
        if t.ndim != 1:
            raise ValueError(
                "mahalanobis mode expects 1-D normalized residual scores; precompute the "
                "per-sample Mahalanobis distance / predicted radius and pass 1-D arrays"
            )
        return t.abs()
    if t.ndim == 1:
        return t.abs()
    flat = t.reshape(t.shape[0], -1)
    if mode == "norm":
        return torch.linalg.norm(flat, dim=-1)
    return flat.abs().max(dim=-1).values  # component_max


def _conformal_quantile_level(n: int, alpha: float) -> float:
    """Finite-sample-corrected ``(1 - alpha)`` conformal level: ``ceil((n+1)(1-alpha)) / n``.

    Clamped to ``1.0`` (use the sample max) when the correction would exceed 1 for small ``n``.
    """

    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    return min(1.0, level)


def _coverage(true_reduced: torch.Tensor, band_reduced: torch.Tensor) -> float:
    """Empirical coverage: fraction of samples whose true error is within the predictive band."""

    return float((true_reduced <= band_reduced).to(torch.float64).mean())


def fit_conformal_scale(
    predicted_error,
    true_error,
    alpha: float = 0.10,
    mode: str = "norm",
) -> ConformalCalibrator:
    """Fit a split-conformal multiplicative scale for force-error predictive uncertainty.

    ``predicted_error`` is the model's per-sample predicted uncertainty (e.g. the VESP-UQ predictive
    ``sigma`` or per-component std) and ``true_error`` is the observed held-out force error (residual
    vector or magnitude). Both are reduced to per-sample scalars by ``mode`` and the conformal scale
    is the conservative ``(1 - alpha)`` quantile of the normalized residuals ``true / predicted``.

    Returns a :class:`ConformalCalibrator`; ``calibrator.scale`` is ``>= 1`` exactly when the raw
    predictions under-cover (so the band is inflated to reach nominal coverage). Makes no Gaussian
    assumption -- coverage is purely empirical on the supplied samples.
    """

    alpha = _validate_alpha(alpha)
    pred = _reduce(predicted_error, mode, "predicted_error")
    true = _reduce(true_error, mode, "true_error")
    if pred.shape != true.shape:
        raise ValueError(
            f"predicted_error and true_error must reduce to the same shape, got {tuple(pred.shape)} "
            f"vs {tuple(true.shape)}"
        )
    n = int(pred.numel())

    scores = true / pred.clamp_min(_TINY)
    level = _conformal_quantile_level(n, alpha)
    scale = float(torch.quantile(scores, level)) if level < 1.0 else float(scores.max())
    scale = max(scale, 0.0)

    coverage_before = _coverage(true, pred)
    coverage_after = _coverage(true, pred * scale)
    return ConformalCalibrator(
        scale=scale,
        alpha=alpha,
        mode=mode,
        n_calibration=n,
        quantile_level=level,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
    )


def apply_conformal_scale(predicted_error, scale: float) -> torch.Tensor:
    """Apply a learned conformal scale to a predicted error, returning ``scale * predicted_error``.

    Shape-preserving (the scale is a single conservative multiplier), so it works on per-sample
    scalars, per-component arrays, or whole predictive-uncertainty tensors alike.
    """

    s = float(scale)
    if s < 0.0:
        raise ValueError(f"scale must be nonnegative, got {scale!r}")
    return _as_tensor(predicted_error, "predicted_error") * s


def coverage_before_after(
    predicted_error,
    true_error,
    calibrated_error,
    alpha: float = 0.10,
    mode: str = "norm",
) -> dict:
    """Report empirical force-error coverage before and after conformal scaling.

    All three arrays are reduced to per-sample scalars by ``mode``. ``coverage_before`` uses the raw
    ``predicted_error`` band, ``coverage_after`` uses the ``calibrated_error`` band, and both are
    compared against the ``1 - alpha`` target. ``covers_target_after`` flags whether the calibrated
    band reaches nominal coverage on these samples (an empirical statement, not a guarantee).
    """

    alpha = _validate_alpha(alpha)
    pred = _reduce(predicted_error, mode, "predicted_error")
    true = _reduce(true_error, mode, "true_error")
    cal = _reduce(calibrated_error, mode, "calibrated_error")
    if not (pred.shape == true.shape == cal.shape):
        raise ValueError(
            "predicted_error, true_error and calibrated_error must reduce to the same shape, got "
            f"{tuple(pred.shape)}, {tuple(true.shape)}, {tuple(cal.shape)}"
        )
    before = _coverage(true, pred)
    after = _coverage(true, cal)
    target = 1.0 - alpha
    return {
        "alpha": alpha,
        "target_coverage": target,
        "coverage_before": before,
        "coverage_after": after,
        "coverage_improvement": after - before,
        "covers_target_after": bool(after >= target),
        "mode": mode,
        "n": int(true.numel()),
    }


def calibrate_trajectory_risk(
    scores,
    true_errors,
    alpha: float = 0.10,
    mode: str = "p95",
    *,
    per_point: bool = False,
) -> dict:
    """Learn a conservative conformal correction from held-out trajectory-level force errors.

    Maps per-trajectory risk ``scores`` onto the held-out true *force* error so the calibrated score
    ``scale * score`` empirically upper-bounds the true force error at the ``1 - alpha`` level. The
    correction is a single multiplicative conformal scale (the ``(1 - alpha)`` quantile of
    ``true_error / score``); ``mode`` records the aggregation that produced the per-trajectory
    scalars (``p95`` by default, matching :func:`vesp.uq.scoring.aggregate_trajectory_error`).

    Two calibration granularities are supported:

    - **trajectory-level** (``per_point=False``, default): ``scores`` and ``true_errors`` are 1-D,
      one scalar per trajectory.
    - **per-point** (``per_point=True``): ``scores`` and ``true_errors`` are sequences of per-point
      1-D arrays (one per trajectory); they are flattened and calibrated pointwise.

    Returns a dict reporting the learned scale and empirical coverage before/after. This concerns
    force-model error only and never mixes in position-accuracy diagnostics.
    """

    alpha = _validate_alpha(alpha)
    if mode not in TRAJECTORY_MODES:
        raise ValueError(f"mode must be one of {TRAJECTORY_MODES}, got {mode!r}")

    if per_point:
        s = torch.cat([_as_tensor(x, "scores").reshape(-1) for x in scores])
        t = torch.cat([_as_tensor(x, "true_errors").reshape(-1) for x in true_errors])
        n_trajectories = len(list(scores))
    else:
        s = _as_tensor(scores, "scores").reshape(-1)
        t = _as_tensor(true_errors, "true_errors").reshape(-1)
        n_trajectories = int(s.numel())

    if s.numel() != t.numel():
        raise ValueError(
            f"scores and true_errors must have the same number of samples, got {s.numel()} vs "
            f"{t.numel()}"
        )
    n = int(s.numel())

    nonconformity = t / s.clamp_min(_TINY)
    level = _conformal_quantile_level(n, alpha)
    scale = float(torch.quantile(nonconformity, level)) if level < 1.0 else float(nonconformity.max())
    scale = max(scale, 0.0)

    calibrated = s * scale
    return {
        "mode": mode,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "quantile_level": level,
        "scale": scale,
        "per_point": bool(per_point),
        "n_samples": n,
        "n_trajectories": n_trajectories,
        "coverage_before": _coverage(t, s),
        "coverage_after": _coverage(t, calibrated),
        "calibrated_scores": calibrated.tolist(),
    }
