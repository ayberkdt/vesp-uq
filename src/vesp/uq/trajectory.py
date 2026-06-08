"""Trajectory-level risk scoring and selective high-fidelity rerun logic (Phases 3-4).

VESP-UQ scores a *whole trajectory* by aggregating the per-position predictive uncertainty
that :class:`~vesp.uq.plugin.VESPUQPlugin` produces along it, then flags the riskiest
trajectories for recomputation with a higher-fidelity force model. The point is operational:
run a cheap surrogate Monte Carlo, score every trajectory here, and rerun only the small
flagged subset -- preserving most of the speed advantage while removing blind trust in the
surrogate where it is least reliable (low altitude / ill-conditioned regimes).

Nothing here evaluates a gravity model; it consumes ``sigma`` (predictive std) and ``radius``
arrays, so it is fully surrogate-agnostic and cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

# Scoring functions that turn a per-position sigma/radius profile into one trajectory number.
SCORING_FUNCTIONS = ("max", "mean", "low_alt_integral", "time_above", "combined")


@dataclass
class TrajectoryScore:
    """Aggregated risk summary for a single trajectory's output points."""

    n_points: int
    max_sigma: float
    mean_sigma: float
    low_altitude_sigma_integral: float
    time_above_threshold: float
    combined_altitude_risk: float
    risk_score: float
    scoring: str
    min_radius: float
    mean_radius: float
    mean_epistemic_sigma: float

    def to_dict(self) -> dict:
        return asdict(self)


def _altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Weight that grows toward the surface: ``1 / max(r - 1, h_floor)``.

    Concentrates risk where the residual-gravity surrogate is known to be least reliable, so
    a trajectory that is uncertain *and* low scores higher than one uncertain only up high.
    """

    return 1.0 / (radius - 1.0).clamp_min(h_floor)


def score_sigma_profile(
    sigma: torch.Tensor,
    radius: torch.Tensor,
    *,
    scoring: str = "max",
    sigma_threshold: float | None = None,
    low_altitude_radius: float = 1.15,
    h_floor: float = 1.0e-3,
    epistemic_sigma: torch.Tensor | None = None,
) -> TrajectoryScore:
    """Aggregate a per-output-point uncertainty profile into a :class:`TrajectoryScore`.

    ``sigma`` and ``radius`` are 1-D tensors over the trajectory's output points (assumed
    roughly uniform sampling in time, so a discrete sum approximates a time integral).

    - ``max`` / ``mean``: extreme / average uncertainty along the trajectory.
    - ``low_alt_integral``: summed uncertainty over points below ``low_altitude_radius``
      (low-altitude uncertainty exposure).
    - ``time_above``: fraction of points whose sigma exceeds ``sigma_threshold``.
    - ``combined``: mean of ``sigma`` weighted by an altitude weight (uncertain-and-low).

    ``risk_score`` is whichever of the above ``scoring`` selects.
    """

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"scoring must be one of {SCORING_FUNCTIONS}, got {scoring!r}")
    sigma = torch.as_tensor(sigma).reshape(-1).to(torch.float64)
    radius = torch.as_tensor(radius).reshape(-1).to(torch.float64)
    if sigma.shape != radius.shape:
        raise ValueError("sigma and radius must have the same length")
    if sigma.numel() == 0:
        raise ValueError("cannot score an empty trajectory")

    low_mask = radius <= float(low_altitude_radius)
    max_sigma = float(sigma.max())
    mean_sigma = float(sigma.mean())
    low_alt_integral = float(sigma[low_mask].sum()) if bool(low_mask.any()) else 0.0
    if sigma_threshold is not None:
        time_above = float((sigma > float(sigma_threshold)).to(torch.float64).mean())
    else:
        time_above = float("nan")
    weight = _altitude_weight(radius, h_floor=h_floor)
    combined = float((sigma * weight).mean())

    table = {
        "max": max_sigma,
        "mean": mean_sigma,
        "low_alt_integral": low_alt_integral,
        "time_above": time_above,
        "combined": combined,
    }
    if epistemic_sigma is not None:
        mean_epi = float(torch.as_tensor(epistemic_sigma).reshape(-1).to(torch.float64).mean())
    else:
        mean_epi = float("nan")

    return TrajectoryScore(
        n_points=int(sigma.numel()),
        max_sigma=max_sigma,
        mean_sigma=mean_sigma,
        low_altitude_sigma_integral=low_alt_integral,
        time_above_threshold=time_above,
        combined_altitude_risk=combined,
        risk_score=table[scoring],
        scoring=scoring,
        min_radius=float(radius.min()),
        mean_radius=float(radius.mean()),
        mean_epistemic_sigma=mean_epi,
    )


@dataclass
class RiskScreeningReport:
    """Outcome of selecting which trajectories to rerun at high fidelity."""

    n_trajectories: int
    threshold: float
    rerun_fraction: float
    n_flagged: int
    flagged_indices: list[int]
    mean_risk_flagged: float | None = None
    mean_risk_accepted: float | None = None
    # Validation against a ground-truth error metric (only when ``true_error`` is supplied):
    capture_rate: float | None = None  # share of truly-high-error trajectories that got flagged
    precision: float | None = None  # share of flagged trajectories that were truly high-error
    spearman_risk_vs_error: float | None = None
    mean_error_flagged: float | None = None
    mean_error_accepted: float | None = None
    error_ratio_flagged_to_accepted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (no scipy); ties broken by argsort order."""

    if a.numel() < 2:
        return float("nan")

    def _rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(x.numel(), dtype=torch.float64)
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if float(denom) <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def select_reruns(
    risk_scores,
    *,
    rerun_fraction: float | None = None,
    threshold: float | None = None,
    true_error=None,
    true_error_quantile: float = 0.90,
) -> RiskScreeningReport:
    """Flag the riskiest trajectories for high-fidelity rerun.

    Provide exactly one budget: ``rerun_fraction`` (rerun the top fraction; the threshold is
    the matching risk quantile) or an absolute ``threshold`` on the risk score.

    When ``true_error`` (one scalar per trajectory, e.g. the actual reference-model error along
    it) is supplied, the report also validates the screen: ``capture_rate`` is the share of the
    truly-high-error trajectories (top ``1 - true_error_quantile``) that were flagged, and
    ``spearman_risk_vs_error`` measures monotonic agreement between risk and real error.
    """

    risk = torch.as_tensor(risk_scores).reshape(-1).to(torch.float64)
    n = int(risk.numel())
    if n == 0:
        raise ValueError("risk_scores is empty")
    if (rerun_fraction is None) == (threshold is None):
        raise ValueError("provide exactly one of rerun_fraction or threshold")

    if rerun_fraction is not None:
        if not 0.0 < float(rerun_fraction) <= 1.0:
            raise ValueError("rerun_fraction must be in (0, 1]")
        # threshold = quantile so that ~rerun_fraction of trajectories land above it
        thr = float(torch.quantile(risk, 1.0 - float(rerun_fraction)))
    else:
        thr = float(threshold)

    flagged_mask = risk >= thr
    accepted_mask = ~flagged_mask
    flagged_indices = [int(i) for i in torch.nonzero(flagged_mask, as_tuple=False).reshape(-1).tolist()]
    n_flagged = len(flagged_indices)

    report = RiskScreeningReport(
        n_trajectories=n,
        threshold=thr,
        rerun_fraction=n_flagged / n,
        n_flagged=n_flagged,
        flagged_indices=flagged_indices,
        mean_risk_flagged=float(risk[flagged_mask].mean()) if n_flagged > 0 else float("nan"),
        mean_risk_accepted=float(risk[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan"),
    )

    if true_error is not None:
        err = torch.as_tensor(true_error).reshape(-1).to(torch.float64)
        if err.numel() != n:
            raise ValueError("true_error must have one value per trajectory")
        high_thr = float(torch.quantile(err, float(true_error_quantile)))
        truly_high = err >= high_thr
        n_high = int(truly_high.sum())
        true_positive = int((flagged_mask & truly_high).sum())
        report.capture_rate = (true_positive / n_high) if n_high > 0 else float("nan")
        report.precision = (true_positive / n_flagged) if n_flagged > 0 else float("nan")
        report.spearman_risk_vs_error = _spearman(risk, err)
        report.mean_error_flagged = float(err[flagged_mask].mean()) if n_flagged > 0 else float("nan")
        accepted = ~flagged_mask
        report.mean_error_accepted = float(err[accepted].mean()) if bool(accepted.any()) else float("nan")
        if report.mean_error_accepted and report.mean_error_accepted > 0.0:
            report.error_ratio_flagged_to_accepted = report.mean_error_flagged / report.mean_error_accepted

    return report


def run_risk_screening(
    plugin,
    trajectories,
    *,
    true_error=None,
    rerun_fraction: float | None = 0.20,
    threshold: float | None = None,
    scoring: str = "max",
) -> dict:
    """Score a trajectory ensemble with ``plugin`` and select the high-fidelity rerun subset.

    ``plugin`` is any object exposing ``score_ensemble(trajectories, scoring=...)`` (the
    :class:`~vesp.uq.plugin.VESPUQPlugin`). Returns a dict with:
      - ``trajectory_scores``: list of :class:`TrajectoryScore` (one per trajectory),
      - ``selected_reruns``: indices flagged for high-fidelity rerun,
      - ``risk_screening_report``: the :class:`RiskScreeningReport` (validated when
        ``true_error`` -- one scalar per trajectory -- is supplied).
    """

    scores = plugin.score_ensemble(trajectories, scoring=scoring)
    risk = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
    report = select_reruns(
        risk,
        rerun_fraction=None if threshold is not None else rerun_fraction,
        threshold=threshold,
        true_error=true_error,
    )
    return {
        "trajectory_scores": scores,
        "selected_reruns": report.flagged_indices,
        "risk_screening_report": report,
    }
