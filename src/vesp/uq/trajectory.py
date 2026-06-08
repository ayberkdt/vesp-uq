"""Trajectory-level risk scoring and selective high-fidelity rerun logic (Phases 3-4).

VESP-UQ scores a *whole trajectory* by aggregating the per-position predictive uncertainty
that :class:`~vesp.uq.plugin.VESPUQPlugin` produces along it, then flags the riskiest
trajectories for recomputation with a higher-fidelity force model. The point is operational:
run a cheap surrogate Monte Carlo, score every trajectory here, and rerun only the small
flagged subset -- preserving most of the speed advantage while removing blind trust in the
surrogate where it is least reliable (low altitude / ill-conditioned / out-of-support regimes).

Two families of per-point risk are supported:

- the original ``sigma`` (predictive std) modes, kept verbatim for backward compatibility;
- the stronger *supervisor* modes built on ``expected_error = sqrt(bias^2 + sigma^2)``, an
  altitude weight, and (optionally) a domain-support penalty -- so a trajectory is flagged for
  having large *expected error where it matters*, not merely large uncertainty.

Nothing here evaluates a gravity model; it consumes per-point arrays, so it is fully
surrogate-agnostic and cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

# Scoring functions that turn a per-position profile into one trajectory number.
# The first five are sigma-only (legacy); the last five use expected_error / domain support.
SCORING_FUNCTIONS = (
    "max",
    "mean",
    "low_alt_integral",
    "time_above",
    "combined",
    "expected",
    "expected_p95",
    "expected_low_alt",
    "supervisor",
    "supervisor_p95",
)

# Modes that need a per-point ``expected_error`` profile (and so cannot run on sigma alone).
_EXPECTED_MODES = frozenset(
    {"expected", "expected_p95", "expected_low_alt", "supervisor", "supervisor_p95"}
)

# Aggregators for collapsing a per-point true-error profile into one trajectory scalar.
TRUE_ERROR_AGGREGATORS = ("max", "mean", "p95")


@dataclass
class TrajectoryScore:
    """Aggregated risk summary for a single trajectory's output points.

    The ``*_sigma`` / ``*_altitude_risk`` fields are the legacy sigma-based aggregations. The
    ``*_expected_error`` / ``*_point_risk`` / ``*_domain_risk`` fields are the stronger
    supervisor metrics; they are ``nan`` when the relevant per-point profile was not supplied
    (e.g. calling :func:`score_sigma_profile` with sigma only, or with domain support disabled).
    """

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
    # --- supervisor metrics (expected error = sqrt(bias^2 + sigma^2)) ---
    max_expected_error: float = float("nan")
    mean_expected_error: float = float("nan")
    p95_expected_error: float = float("nan")
    low_altitude_expected_error_integral: float = float("nan")
    max_mean_error_magnitude: float = float("nan")
    mean_mean_error_magnitude: float = float("nan")
    mean_point_risk: float = float("nan")
    p95_point_risk: float = float("nan")
    # --- domain-support metrics (only when domain support is supplied) ---
    max_domain_risk: float = float("nan")
    time_outside_support: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _as_1d(x, n: int | None = None, name: str = "array") -> torch.Tensor:
    t = torch.as_tensor(x).reshape(-1).to(torch.float64)
    if n is not None and t.numel() != n:
        raise ValueError(f"{name} must have length {n}, got {t.numel()}")
    return t


def _normalize_weights(weights, n: int) -> torch.Tensor | None:
    """Return weights normalized to sum 1, or ``None`` for the uniform/legacy path."""

    if weights is None:
        return None
    w = _as_1d(weights, n, "weights")
    if bool((w < 0).any()):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _wmean(x: torch.Tensor, w: torch.Tensor | None) -> float:
    """Weighted mean (``w`` already normalized to sum 1); uniform when ``w is None``."""

    return float((x * w).sum()) if w is not None else float(x.mean())


def _weighted_quantile(x: torch.Tensor, q: float, w: torch.Tensor | None) -> float:
    """``q``-quantile of ``x``; weighted (``w`` sums to 1) or plain ``torch.quantile``.

    The weighted branch uses the standard cumulative-weight definition (lower value at the
    first point whose cumulative weight reaches ``q``), which is robust and dependency-free.
    """

    if x.numel() == 0:
        return float("nan")
    if w is None:
        return float(torch.quantile(x, q))
    order = torch.argsort(x)
    xs, ws = x[order], w[order]
    cum = torch.cumsum(ws, dim=0)
    cum = cum / cum[-1]
    idx = int(torch.searchsorted(cum, torch.tensor(float(q), dtype=x.dtype)))
    idx = min(idx, xs.numel() - 1)
    return float(xs[idx])


def aggregate_trajectory_error(values, mode: str = "p95") -> float:
    """Collapse a per-point error profile into one trajectory scalar (``max`` / ``mean`` / ``p95``).

    Shared by the nearest-neighbour oracle and the report so that risk and true error are
    aggregated consistently. ``p95`` (the default) is robust to a single nearest-neighbour
    spike while still rewarding a sustained high-error pass, unlike ``max`` (spike-dominated)
    or ``mean`` (washes the pass out).
    """

    if mode not in TRUE_ERROR_AGGREGATORS:
        raise ValueError(f"mode must be one of {TRUE_ERROR_AGGREGATORS}, got {mode!r}")
    v = _as_1d(values)
    if v.numel() == 0:
        return float("nan")
    if mode == "max":
        return float(v.max())
    if mode == "mean":
        return float(v.mean())
    return float(torch.quantile(v, 0.95))


def _altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Weight that grows toward the surface: ``1 / max(r - 1, h_floor)``.

    Concentrates risk where the residual-gravity surrogate is known to be least reliable, so
    a trajectory that is uncertain *and* low scores higher than one uncertain only up high.
    """

    return 1.0 / (radius - 1.0).clamp_min(h_floor)


def _normalized_altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Altitude weight rescaled by its own median so a typical-altitude point weighs ~1.

    Keeps the supervisor point risk on roughly the same scale as ``expected_error`` (instead of
    the raw ``1/(r-1)`` blow-up), so trajectories stay comparable across altitude profiles.
    """

    w = _altitude_weight(radius, h_floor=h_floor)
    med = torch.median(w)
    return w / med.clamp_min(torch.finfo(w.dtype).tiny)


def score_sigma_profile(
    sigma: torch.Tensor,
    radius: torch.Tensor,
    *,
    scoring: str = "max",
    sigma_threshold: float | None = None,
    low_altitude_radius: float = 1.15,
    h_floor: float = 1.0e-3,
    epistemic_sigma: torch.Tensor | None = None,
    expected_error: torch.Tensor | None = None,
    mean_error_magnitude: torch.Tensor | None = None,
    domain_risk: torch.Tensor | None = None,
    domain_weight: float = 1.0,
    weights: torch.Tensor | None = None,
) -> TrajectoryScore:
    """Aggregate a per-output-point profile into a :class:`TrajectoryScore`.

    ``sigma`` and ``radius`` are 1-D tensors over the trajectory's output points. By default the
    points are assumed roughly uniform in time (a discrete sum approximates a time integral);
    pass ``weights`` (one per point, e.g. proportional to local dt) to correct for non-uniform
    sampling -- ``None`` preserves the legacy uniform behavior exactly.

    Legacy sigma modes:
      - ``max`` / ``mean``: extreme / average uncertainty along the trajectory.
      - ``low_alt_integral``: summed uncertainty over points below ``low_altitude_radius``.
      - ``time_above``: (weighted) fraction of points whose sigma exceeds ``sigma_threshold``.
      - ``combined``: mean of ``sigma`` times an altitude weight (uncertain-and-low).

    Supervisor modes (require ``expected_error``):
      - ``expected``: mean expected error.
      - ``expected_p95``: 95th-percentile expected error.
      - ``expected_low_alt``: summed expected error below ``low_altitude_radius``.
      - ``supervisor``: mean of ``point_risk = expected_error * norm_altitude_weight *
        (1 + domain_weight * domain_risk)``.
      - ``supervisor_p95``: 95th percentile of that same ``point_risk``.

    ``risk_score`` is whichever of the above ``scoring`` selects.
    """

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"scoring must be one of {SCORING_FUNCTIONS}, got {scoring!r}")
    sigma = _as_1d(sigma)
    radius = _as_1d(radius)
    if sigma.shape != radius.shape:
        raise ValueError("sigma and radius must have the same length")
    n = int(sigma.numel())
    if n == 0:
        raise ValueError("cannot score an empty trajectory")
    if scoring in _EXPECTED_MODES and expected_error is None:
        raise ValueError(
            f"scoring={scoring!r} requires an expected_error profile; score via "
            "VESPUQPlugin.score_trajectory or pass expected_error explicitly"
        )

    w = _normalize_weights(weights, n)
    low_mask = radius <= float(low_altitude_radius)

    # ---- legacy sigma aggregations (unchanged when weights is None) ----
    max_sigma = float(sigma.max())
    mean_sigma = _wmean(sigma, w)
    if bool(low_mask.any()):
        low_alt_integral = (
            float(sigma[low_mask].sum()) if w is None else float((sigma * w)[low_mask].sum())
        )
    else:
        low_alt_integral = 0.0
    if sigma_threshold is not None:
        above = (sigma > float(sigma_threshold)).to(torch.float64)
        time_above = float(above.mean()) if w is None else float((above * w).sum())
    else:
        time_above = float("nan")
    alt_weight = _altitude_weight(radius, h_floor=h_floor)
    combined = _wmean(sigma * alt_weight, w)

    mean_epi = (
        _wmean(_as_1d(epistemic_sigma, n, "epistemic_sigma"), w)
        if epistemic_sigma is not None
        else float("nan")
    )

    # ---- supervisor metrics (expected error / mean-error magnitude / domain) ----
    max_ee = mean_ee = p95_ee = low_alt_ee = float("nan")
    mean_point_risk = p95_point_risk = float("nan")
    if expected_error is not None:
        ee = _as_1d(expected_error, n, "expected_error")
        max_ee = float(ee.max())
        mean_ee = _wmean(ee, w)
        p95_ee = _weighted_quantile(ee, 0.95, w)
        if bool(low_mask.any()):
            low_alt_ee = float(ee[low_mask].sum()) if w is None else float((ee * w)[low_mask].sum())
        else:
            low_alt_ee = 0.0

        norm_alt = _normalized_altitude_weight(radius, h_floor=h_floor)
        if domain_risk is not None:
            dr = _as_1d(domain_risk, n, "domain_risk")
            domain_factor = 1.0 + float(domain_weight) * dr
        else:
            domain_factor = torch.ones_like(ee)
        point_risk = ee * norm_alt * domain_factor
        mean_point_risk = _wmean(point_risk, w)
        p95_point_risk = _weighted_quantile(point_risk, 0.95, w)

    max_mem = mean_mem = float("nan")
    if mean_error_magnitude is not None:
        mem = _as_1d(mean_error_magnitude, n, "mean_error_magnitude")
        max_mem = float(mem.max())
        mean_mem = _wmean(mem, w)

    max_domain_risk = time_outside_support = float("nan")
    if domain_risk is not None:
        dr = _as_1d(domain_risk, n, "domain_risk")
        max_domain_risk = float(dr.max())
        outside = (dr > 1.0).to(torch.float64)
        time_outside_support = float(outside.mean()) if w is None else float((outside * w).sum())

    table = {
        "max": max_sigma,
        "mean": mean_sigma,
        "low_alt_integral": low_alt_integral,
        "time_above": time_above,
        "combined": combined,
        "expected": mean_ee,
        "expected_p95": p95_ee,
        "expected_low_alt": low_alt_ee,
        "supervisor": mean_point_risk,
        "supervisor_p95": p95_point_risk,
    }

    return TrajectoryScore(
        n_points=n,
        max_sigma=max_sigma,
        mean_sigma=mean_sigma,
        low_altitude_sigma_integral=low_alt_integral,
        time_above_threshold=time_above,
        combined_altitude_risk=combined,
        risk_score=table[scoring],
        scoring=scoring,
        min_radius=float(radius.min()),
        mean_radius=_wmean(radius, w),
        mean_epistemic_sigma=mean_epi,
        max_expected_error=max_ee,
        mean_expected_error=mean_ee,
        p95_expected_error=p95_ee,
        low_altitude_expected_error_integral=low_alt_ee,
        max_mean_error_magnitude=max_mem,
        mean_mean_error_magnitude=mean_mem,
        mean_point_risk=mean_point_risk,
        p95_point_risk=p95_point_risk,
        max_domain_risk=max_domain_risk,
        time_outside_support=time_outside_support,
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
    # Selection-policy bookkeeping (so the report says *why* a given count was flagged):
    selection_mode: str = "fraction"  # fraction | threshold | threshold+max_fraction
    max_rerun_fraction: float | None = None
    n_above_threshold: int | None = None
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
    max_rerun_fraction: float | None = None,
    true_error=None,
    true_error_quantile: float = 0.90,
) -> RiskScreeningReport:
    """Flag the riskiest trajectories for high-fidelity rerun.

    Three selection policies:

    - ``rerun_fraction`` only: rerun the top fraction (threshold = matching risk quantile).
    - ``threshold`` only: rerun every trajectory at or above an absolute risk threshold. If
      nothing exceeds it, *zero* trajectories are flagged -- a safe in-distribution benchmark is
      allowed to raise no alarms.
    - ``threshold`` + ``max_rerun_fraction``: take everything above the absolute threshold, but
      if that exceeds the budget, keep only the top ``max_rerun_fraction`` by risk.

    When ``true_error`` (one scalar per trajectory) is supplied, the report also validates the
    screen: ``capture_rate`` is the share of the truly-high-error trajectories (top
    ``1 - true_error_quantile``) that were flagged, and ``spearman_risk_vs_error`` measures
    monotonic agreement between risk and real error.
    """

    risk = _as_1d(risk_scores)
    n = int(risk.numel())
    if n == 0:
        raise ValueError("risk_scores is empty")

    has_frac = rerun_fraction is not None
    has_thr = threshold is not None
    has_max = max_rerun_fraction is not None

    if has_max:
        if not has_thr:
            raise ValueError("max_rerun_fraction requires an absolute threshold")
        if has_frac:
            raise ValueError("combine max_rerun_fraction with threshold, not rerun_fraction")
        selection_mode = "threshold+max_fraction"
    elif has_frac and has_thr:
        raise ValueError(
            "provide exactly one of rerun_fraction or threshold (or threshold + max_rerun_fraction)"
        )
    elif has_frac:
        selection_mode = "fraction"
    elif has_thr:
        selection_mode = "threshold"
    else:
        raise ValueError("provide rerun_fraction, threshold, or threshold + max_rerun_fraction")

    n_above_threshold: int | None = None
    if selection_mode == "fraction":
        if not 0.0 < float(rerun_fraction) <= 1.0:
            raise ValueError("rerun_fraction must be in (0, 1]")
        thr = float(torch.quantile(risk, 1.0 - float(rerun_fraction)))
        flagged_mask = risk >= thr
    else:
        thr = float(threshold)
        above_mask = risk >= thr
        n_above_threshold = int(above_mask.sum())
        if selection_mode == "threshold":
            flagged_mask = above_mask
        else:  # threshold + max_fraction
            if not 0.0 < float(max_rerun_fraction) <= 1.0:
                raise ValueError("max_rerun_fraction must be in (0, 1]")
            cap = int(float(max_rerun_fraction) * n)
            if n_above_threshold <= cap:
                flagged_mask = above_mask
            else:
                order = torch.argsort(risk, descending=True)
                flagged_mask = torch.zeros(n, dtype=torch.bool)
                flagged_mask[order[:cap]] = True

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
        selection_mode=selection_mode,
        max_rerun_fraction=float(max_rerun_fraction) if has_max else None,
        n_above_threshold=n_above_threshold,
    )

    if true_error is not None:
        err = _as_1d(true_error)
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
        report.mean_error_accepted = float(err[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan")
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
    max_rerun_fraction: float | None = None,
    scoring: str = "max",
    weights=None,
) -> dict:
    """Score a trajectory ensemble with ``plugin`` and select the high-fidelity rerun subset.

    ``plugin`` is any object exposing ``score_ensemble(trajectories, scoring=..., weights=...)``
    (the :class:`~vesp.uq.plugin.VESPUQPlugin`). Returns a dict with:
      - ``trajectory_scores``: list of :class:`TrajectoryScore` (one per trajectory),
      - ``selected_reruns``: indices flagged for high-fidelity rerun,
      - ``risk_screening_report``: the :class:`RiskScreeningReport` (validated when
        ``true_error`` -- one scalar per trajectory -- is supplied).

    Selection follows :func:`select_reruns`: ``threshold`` (optionally with
    ``max_rerun_fraction``) takes precedence over ``rerun_fraction`` when supplied.
    """

    scores = plugin.score_ensemble(trajectories, scoring=scoring, weights=weights)
    risk = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
    if threshold is not None:
        report = select_reruns(
            risk,
            threshold=threshold,
            max_rerun_fraction=max_rerun_fraction,
            true_error=true_error,
        )
    else:
        report = select_reruns(risk, rerun_fraction=rerun_fraction, true_error=true_error)
    return {
        "trajectory_scores": scores,
        "selected_reruns": report.flagged_indices,
        "risk_screening_report": report,
    }
