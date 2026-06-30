"""Per-trajectory risk scoring for VESP-UQ (force-risk / OOD, not position error).

This module turns a per-output-point profile (predictive ``sigma``, ``expected_error``, optional
domain-support risk) into a single :class:`TrajectoryScore`. It is fully surrogate-agnostic: it
consumes arrays, never a gravity model, and never claims to predict trajectory *position* error
-- it summarizes the surrogate's expected *force-model* error and out-of-support risk.

Three scoring families:

- legacy ``sigma`` modes (predictive-std magnitude), kept verbatim for backward compatibility;
- *relative* expected-error / supervisor modes, normalized per trajectory -- good for RANKING one
  ensemble (which orbits to prioritize), not comparable across trajectories;
- *absolute* expected-error / supervisor modes, normalized by a fixed altitude reference -- so a
  single physical force-risk budget means the same thing for every trajectory (zero-alarm screen).

``expected_error = sqrt(bias^2 + sigma^2)`` underpins both supervisor families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vesp.uq.metrics import local_radial_frame

# Scoring functions that turn a per-position profile into one trajectory number.
# Legacy sigma-only modes:
_SIGMA_MODES = ("max", "mean", "low_alt_integral", "time_above", "combined")
# Expected-error modes. `expected`/`expected_p95` are aliases of the absolute variants (no
# altitude weighting); `supervisor`/`supervisor_p95` are aliases of the RELATIVE variants
# (per-trajectory altitude normalization, for ranking).
_EXPECTED_ONLY_MODES = (
    "expected",
    "expected_abs",
    "expected_p95",
    "expected_abs_p95",
    "expected_low_alt",
)
_CALIBRATED_SUPERVISOR_MODES = (
    "calibrated_supervisor",
    "calibrated_supervisor_p95",
)
_SUPERVISOR_MODES = (
    "supervisor",
    "supervisor_rel",
    "supervisor_p95",
    "supervisor_rel_p95",
    "supervisor_abs",
    "supervisor_abs_p95",
) + _CALIBRATED_SUPERVISOR_MODES
# Directional risk modes (M1): exploit the per-point 3x3 covariance geometry / bias direction to
# rank force error *within* an altitude band -- value beyond the scalar altitude heuristic. All are
# p95-aggregated per trajectory.
#   radial_expected   -- |mean_error . r_hat| + sigma_radial  (bias projected onto the radial axis
#                        plus the radial predictive spread; the radial component dominates the
#                        altitude-dependent force-model error).
#   anisotropy_gated  -- expected_error * (1 + kappa*(lambda_max/lambda_min - 1)); anisotropy as a
#                        multiplier on the expected error (isotropic covariance -> factor 1).
#   largest_eigenvalue-- sqrt(lambda_max(covariance)); the worst-direction predictive std.
_DIRECTIONAL_MODES = ("radial_expected", "anisotropy_gated", "largest_eigenvalue")
# Epistemic-targeted screening (M2): up-weight points whose uncertainty is reducible (epistemic)
# rather than aleatoric -- where a high-fidelity rerun actually helps.
#   expected_epistemic -- expected_error * epistemic_fraction**gamma, p95-aggregated.
_EPISTEMIC_MODES = ("expected_epistemic",)
# Subset of the directional modes that need the FULL 3x3 covariance (not just std_components).
_COVARIANCE_MODES = frozenset({"anisotropy_gated", "largest_eigenvalue"})
# Default anisotropy gate strength for ``anisotropy_gated``.
ANISOTROPY_KAPPA_DEFAULT = 0.5

SCORING_FUNCTIONS = (
    _SIGMA_MODES + _EXPECTED_ONLY_MODES + _SUPERVISOR_MODES + _DIRECTIONAL_MODES + _EPISTEMIC_MODES
)

# Modes that need a per-point ``expected_error`` profile (and so cannot run on sigma alone).
_EXPECTED_MODES = frozenset(_EXPECTED_ONLY_MODES + _SUPERVISOR_MODES)
# Directional / epistemic modes also produce a stand-alone p95 risk profile (handled separately).
_GEOMETRIC_MODES = frozenset(_DIRECTIONAL_MODES + _EPISTEMIC_MODES)

# Aggregators for collapsing a per-point true-error profile into one trajectory scalar.
TRUE_ERROR_AGGREGATORS = ("max", "mean", "p95")

# ---- scoring-mode classification (relative ranking vs absolute physical-budget scale) ----
# RELATIVE modes normalize altitude per trajectory (good for ranking one ensemble, NOT for
# cross-trajectory absolute thresholds). ABSOLUTE / absolute-like modes are on a fixed
# expected-force-error scale, so a single physical budget means the same for every trajectory.
_RELATIVE_SCORINGS = frozenset(
    {
        "supervisor",
        "supervisor_rel",
        "supervisor_p95",
        "supervisor_rel_p95",
        "calibrated_supervisor",
        "calibrated_supervisor_p95",
    }
)
_ABSOLUTE_SCORINGS = frozenset(
    {
        "expected",
        "expected_abs",
        "expected_p95",
        "expected_abs_p95",
        "expected_low_alt",
        "supervisor_abs",
        "supervisor_abs_p95",
    }
    | set(_DIRECTIONAL_MODES)
    | set(_EPISTEMIC_MODES)
)
_EXPECTED_ONLY_SCORINGS = frozenset(_EXPECTED_ONLY_MODES)
# Canonical names for the backward-compatible aliases.
_CANONICAL_ALIASES = {
    "expected": "expected_abs",
    "expected_p95": "expected_abs_p95",
    "supervisor": "supervisor_rel",
    "supervisor_p95": "supervisor_rel_p95",
}


def _validate_scoring(scoring: str) -> str:
    """Return ``scoring`` if it is a known mode, else raise a clear ``ValueError``."""

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"unknown scoring {scoring!r}; must be one of {SCORING_FUNCTIONS}")
    return scoring


def canonical_scoring_name(scoring: str) -> str:
    """Map a (possibly aliased) scoring name to its canonical name.

    ``expected -> expected_abs``, ``expected_p95 -> expected_abs_p95``,
    ``supervisor -> supervisor_rel``, ``supervisor_p95 -> supervisor_rel_p95``; all other known
    names map to themselves. Unknown names raise ``ValueError``.
    """

    return _CANONICAL_ALIASES.get(_validate_scoring(scoring), scoring)


def is_relative_scoring(scoring: str) -> bool:
    """True for per-trajectory-normalized supervisor modes (ranking only, not absolute budgets)."""

    return _validate_scoring(scoring) in _RELATIVE_SCORINGS


def is_absolute_scoring(scoring: str) -> bool:
    """True for absolute / absolute-like expected-force-error modes (safe for physical budgets)."""

    return _validate_scoring(scoring) in _ABSOLUTE_SCORINGS


def is_expected_only_scoring(scoring: str) -> bool:
    """True for the pure expected-error modes (no altitude weighting at all)."""

    return _validate_scoring(scoring) in _EXPECTED_ONLY_SCORINGS


def needs_covariance(scoring: str) -> bool:
    """True if ``scoring`` needs the full per-point 3x3 covariance (anisotropy / largest eigenvalue).

    The plugin uses this to gate the extra :meth:`predict_covariance_3x3` build so default scoring
    pays nothing -- ``radial_expected`` (diagonal projection) and the epistemic mode do not need it.
    """

    return _validate_scoring(scoring) in _COVARIANCE_MODES


# --------------------------------------------------------------------------------------------- #
# M1 -- directional / covariance-geometry per-point profile builders (pure functions on arrays)
# --------------------------------------------------------------------------------------------- #
def _as_matrix(x, n: int, cols: int, name: str) -> torch.Tensor:
    t = torch.as_tensor(x).to(torch.float64)
    if t.ndim != 2 or t.shape != (n, cols):
        raise ValueError(f"{name} must have shape ({n}, {cols}), got {tuple(t.shape)}")
    return t


def radial_profile(
    mean_error: torch.Tensor, std_components: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Per-point radial force-risk ``(N,)``: ``|mean_error . r_hat| + sigma_radial``.

    ``mean_error`` ``(N, 3)`` is the predicted bias vector, ``std_components`` ``(N, 3)`` the per-axis
    predictive std, ``positions`` ``(N, 3)`` the query points. The radial axis ``r_hat`` is taken
    from :func:`vesp.uq.metrics.local_radial_frame`; ``sigma_radial`` is the predictive std along it
    under the diagonal-covariance approximation (``sqrt(sum_k (r_hat_k * std_k)^2)``). The radial
    component carries the altitude-dependent force-model error, so this isolates the part of the risk
    that altitude alone cannot rank.
    """

    me = torch.as_tensor(mean_error).to(torch.float64)
    if me.ndim != 2 or me.shape[-1] != 3:
        raise ValueError("mean_error must have shape (N, 3)")
    n = me.shape[0]
    std = _as_matrix(std_components, n, 3, "std_components")
    pos = _as_matrix(positions, n, 3, "positions")
    r_hat = local_radial_frame(pos)[:, 0, :]  # (N, 3) radial axis
    bias_radial = (me * r_hat).sum(dim=-1).abs()
    var_radial = (std.clamp_min(0.0) ** 2 * r_hat**2).sum(dim=-1)
    return bias_radial + var_radial.clamp_min(0.0).sqrt()


def anisotropy_multiplier(covariance: torch.Tensor, kappa: float = ANISOTROPY_KAPPA_DEFAULT) -> torch.Tensor:
    """Per-point anisotropy gate ``(N,) >= 1``: ``1 + kappa*(lambda_max/lambda_min - 1)``.

    ``covariance`` is the ``(N, 3, 3)`` predictive covariance; eigenvalues are taken on the
    symmetrized matrix and the smallest is floored to keep the ratio finite. An isotropic covariance
    yields exactly ``1.0`` (no gating); a highly anisotropic one amplifies the score.
    """

    cov = torch.as_tensor(covariance).to(torch.float64)
    if cov.ndim != 3 or cov.shape[-2:] != (3, 3):
        raise ValueError("covariance must have shape (N, 3, 3)")
    sym = 0.5 * (cov + cov.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(sym)  # ascending (N, 3)
    tiny = torch.finfo(torch.float64).tiny
    lam_min = eig[:, 0].clamp_min(tiny)
    lam_max = eig[:, -1].clamp_min(tiny)
    return 1.0 + float(kappa) * (lam_max / lam_min - 1.0)


def largest_eigenvalue_profile(covariance: torch.Tensor) -> torch.Tensor:
    """Per-point worst-direction predictive std ``(N,)``: ``sqrt(lambda_max(covariance))``."""

    cov = torch.as_tensor(covariance).to(torch.float64)
    if cov.ndim != 3 or cov.shape[-2:] != (3, 3):
        raise ValueError("covariance must have shape (N, 3, 3)")
    sym = 0.5 * (cov + cov.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(sym)
    return eig[:, -1].clamp_min(0.0).sqrt()


@dataclass
class TrajectoryScore:
    """Aggregated risk summary for a single trajectory's output points.

    The ``*_sigma`` / ``*_altitude_risk`` fields are the legacy sigma-based aggregations. The
    ``*_expected_error`` / ``*_point_risk*`` / ``*_domain_risk`` fields are the supervisor
    metrics; they are ``nan`` when the relevant per-point profile was not supplied (e.g. calling
    :func:`score_sigma_profile` with sigma only, or with domain support disabled).

    ``mean_point_risk`` / ``p95_point_risk`` are the RELATIVE supervisor risk (per-trajectory
    altitude normalization -- for ranking). ``mean_point_risk_abs`` / ``p95_point_risk_abs`` are
    the ABSOLUTE supervisor risk (fixed altitude reference -- for cross-trajectory thresholds).
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
    # relative supervisor point risk (per-trajectory altitude normalization)
    mean_point_risk: float = float("nan")
    p95_point_risk: float = float("nan")
    # absolute supervisor point risk (fixed altitude reference)
    mean_point_risk_abs: float = float("nan")
    p95_point_risk_abs: float = float("nan")
    # validation-calibrated supervisor point risk
    mean_calibrated_point_risk: float = float("nan")
    p95_calibrated_point_risk: float = float("nan")
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


def aggregate_trajectory_error(values, mode: str = "p95", weights=None) -> float:
    """Collapse a per-point error profile into one trajectory scalar (``max`` / ``mean`` / ``p95``).

    Shared by the nearest-neighbour oracle and the report so that risk and true error are
    aggregated consistently. ``p95`` (the default) is robust to a single nearest-neighbour
    spike while still rewarding a sustained high-error pass, unlike ``max`` (spike-dominated)
    or ``mean`` (washes the pass out). Optional ``weights`` apply the same time-weighting
    convention as trajectory risk scoring; ``max`` is unchanged by weights.
    """

    if mode not in TRUE_ERROR_AGGREGATORS:
        raise ValueError(f"mode must be one of {TRUE_ERROR_AGGREGATORS}, got {mode!r}")
    v = _as_1d(values)
    if v.numel() == 0:
        return float("nan")
    w = _normalize_weights(weights, int(v.numel()))
    if mode == "max":
        return float(v.max())
    if mode == "mean":
        return _wmean(v, w)
    return _weighted_quantile(v, 0.95, w)


def calibrate_risk_threshold(values, quantile: float = 0.95, multiplier: float = 1.0) -> float:
    """Absolute risk threshold from a held-out risk distribution: ``quantile(values) * multiplier``.

    ``values`` is a 1-D array of in-distribution risk samples -- either per-point
    ``expected_error`` or per-trajectory risk scores from a calibration set. The returned
    threshold is meant for the absolute ``select_reruns(threshold=...)`` path so a downstream
    test set with lower risk can legitimately raise zero alarms.
    """

    v = _as_1d(values)
    if v.numel() == 0:
        return float("nan")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(torch.quantile(v, float(quantile))) * float(multiplier)


def _altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Weight that grows toward the surface: ``1 / max(r - 1, h_floor)``.

    Concentrates risk where the residual-gravity surrogate is known to be least reliable, so
    a trajectory that is uncertain *and* low scores higher than one uncertain only up high.
    """

    return 1.0 / (radius - 1.0).clamp_min(h_floor)


def _relative_altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Altitude weight rescaled by its OWN median so a typical point on THIS trajectory weighs ~1.

    Good for ranking within an ensemble (keeps the supervisor point risk on the scale of
    ``expected_error``), but NOT comparable across trajectories -- a constant-altitude orbit
    always normalizes to 1 regardless of how low it is. Use the absolute weight for thresholds.
    """

    w = _altitude_weight(radius, h_floor=h_floor)
    med = torch.median(w)
    return w / med.clamp_min(torch.finfo(w.dtype).tiny)


def _absolute_altitude_weight(
    radius: torch.Tensor, *, h_floor: float, reference_h: float
) -> torch.Tensor:
    """Altitude weight rescaled by a FIXED reference altitude so it means the same everywhere.

    ``abs_weight = raw_weight / reference_weight`` with ``reference_weight = 1 / reference_h``;
    i.e. a point at altitude ``h = reference_h`` weighs 1, lower points weigh >1, higher <1 --
    the same mapping for every trajectory, so absolute thresholds are consistent.
    """

    raw = _altitude_weight(radius, h_floor=h_floor)
    reference_weight = 1.0 / max(float(reference_h), h_floor)
    return raw / reference_weight


def score_sigma_profile(
    sigma: torch.Tensor,
    radius: torch.Tensor,
    *,
    scoring: str = "max",
    sigma_threshold: float | None = None,
    low_altitude_radius: float = 1.15,
    h_floor: float = 1.0e-3,
    altitude_reference_h: float | None = None,
    epistemic_sigma: torch.Tensor | None = None,
    expected_error: torch.Tensor | None = None,
    mean_error_magnitude: torch.Tensor | None = None,
    domain_risk: torch.Tensor | None = None,
    domain_weight: float = 1.0,
    calibrated_point_risk: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    # M1 (directional) / M2 (epistemic) inputs -- only consulted by the geometric/epistemic modes.
    mean_error_vector: torch.Tensor | None = None,
    std_components: torch.Tensor | None = None,
    covariance: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    epistemic_fraction: torch.Tensor | None = None,
    epistemic_gamma: float = 1.0,
    anisotropy_kappa: float = ANISOTROPY_KAPPA_DEFAULT,
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

    Expected-error modes (require ``expected_error``):
      - ``expected`` / ``expected_abs``: mean expected error.
      - ``expected_p95`` / ``expected_abs_p95``: 95th-percentile expected error.
      - ``expected_low_alt``: summed expected error below ``low_altitude_radius``.

    Supervisor modes (``point_risk = expected_error * altitude_weight * (1 + domain_weight *
    domain_risk)``):
      - ``supervisor`` / ``supervisor_rel`` (+ ``_p95``): RELATIVE altitude weight (per-trajectory
        median) -- for ranking an ensemble.
      - ``supervisor_abs`` (+ ``_p95``): ABSOLUTE altitude weight (fixed reference
        ``altitude_reference_h``, default ``low_altitude_radius - 1``) -- for absolute thresholds.

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
    if scoring in _EXPECTED_MODES and scoring not in _CALIBRATED_SUPERVISOR_MODES and expected_error is None:
        raise ValueError(
            f"scoring={scoring!r} requires an expected_error profile; score via "
            "VESPUQPlugin.score_trajectory or pass expected_error explicitly"
        )
    if scoring in _CALIBRATED_SUPERVISOR_MODES and calibrated_point_risk is None:
        raise ValueError(
            f"scoring={scoring!r} requires a calibrated_point_risk profile; fit VESPUQPlugin "
            "with risk.calibrated_supervisor.enabled=true or pass calibrated_point_risk explicitly"
        )
    if scoring == "radial_expected" and (
        mean_error_vector is None or std_components is None or positions is None
    ):
        raise ValueError(
            "scoring='radial_expected' requires mean_error_vector, std_components and positions"
        )
    if scoring in _COVARIANCE_MODES and covariance is None:
        raise ValueError(f"scoring={scoring!r} requires a per-point 3x3 covariance profile")
    if scoring == "anisotropy_gated" and expected_error is None:
        raise ValueError("scoring='anisotropy_gated' requires an expected_error profile")
    if scoring == "expected_epistemic" and (expected_error is None or epistemic_fraction is None):
        raise ValueError("scoring='expected_epistemic' requires expected_error and epistemic_fraction")

    reference_h = (
        float(altitude_reference_h)
        if altitude_reference_h is not None
        else max(float(low_altitude_radius) - 1.0, h_floor)
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

    # ---- expected-error + supervisor metrics ----
    max_ee = mean_ee = p95_ee = low_alt_ee = float("nan")
    mean_pr_rel = p95_pr_rel = float("nan")
    mean_pr_abs = p95_pr_abs = float("nan")
    mean_cal_pr = p95_cal_pr = float("nan")
    ee = None
    if expected_error is not None:
        ee = _as_1d(expected_error, n, "expected_error")
        max_ee = float(ee.max())
        mean_ee = _wmean(ee, w)
        p95_ee = _weighted_quantile(ee, 0.95, w)
        if bool(low_mask.any()):
            low_alt_ee = float(ee[low_mask].sum()) if w is None else float((ee * w)[low_mask].sum())
        else:
            low_alt_ee = 0.0

        if domain_risk is not None:
            dr = _as_1d(domain_risk, n, "domain_risk")
            domain_factor = 1.0 + float(domain_weight) * dr
        else:
            domain_factor = torch.ones_like(ee)

        rel_alt = _relative_altitude_weight(radius, h_floor=h_floor)
        point_risk_rel = ee * rel_alt * domain_factor
        mean_pr_rel = _wmean(point_risk_rel, w)
        p95_pr_rel = _weighted_quantile(point_risk_rel, 0.95, w)

        abs_alt = _absolute_altitude_weight(radius, h_floor=h_floor, reference_h=reference_h)
        point_risk_abs = ee * abs_alt * domain_factor
        mean_pr_abs = _wmean(point_risk_abs, w)
        p95_pr_abs = _weighted_quantile(point_risk_abs, 0.95, w)

    if calibrated_point_risk is not None:
        cal_pr = _as_1d(calibrated_point_risk, n, "calibrated_point_risk")
        mean_cal_pr = _wmean(cal_pr, w)
        p95_cal_pr = _weighted_quantile(cal_pr, 0.95, w)

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

    # ---- M1 directional / M2 epistemic risk (per-point profile -> p95) ----
    geometric_risk = float("nan")
    if scoring in _GEOMETRIC_MODES:
        # inputs are validated non-None above; assert to narrow the types for the builders.
        if scoring == "radial_expected":
            assert mean_error_vector is not None and std_components is not None and positions is not None
            profile = radial_profile(mean_error_vector, std_components, positions)
        elif scoring == "anisotropy_gated":
            assert ee is not None and covariance is not None
            profile = ee * anisotropy_multiplier(covariance, kappa=anisotropy_kappa)
        elif scoring == "largest_eigenvalue":
            assert covariance is not None
            profile = largest_eigenvalue_profile(covariance)
        else:  # expected_epistemic
            assert ee is not None
            ef = _as_1d(epistemic_fraction, n, "epistemic_fraction").clamp(0.0, 1.0)
            gamma = float(epistemic_gamma)
            profile = ee if gamma == 0.0 else ee * ef.pow(gamma)
        geometric_risk = _weighted_quantile(_as_1d(profile, n, scoring), 0.95, w)

    table = {
        "max": max_sigma,
        "mean": mean_sigma,
        "low_alt_integral": low_alt_integral,
        "time_above": time_above,
        "combined": combined,
        # expected-error (absolute scale; `expected`/`expected_p95` are backward-compat aliases)
        "expected": mean_ee,
        "expected_abs": mean_ee,
        "expected_p95": p95_ee,
        "expected_abs_p95": p95_ee,
        "expected_low_alt": low_alt_ee,
        # relative supervisor (ranking) -- `supervisor`/`supervisor_p95` are aliases
        "supervisor": mean_pr_rel,
        "supervisor_rel": mean_pr_rel,
        "supervisor_p95": p95_pr_rel,
        "supervisor_rel_p95": p95_pr_rel,
        # absolute supervisor (thresholds)
        "supervisor_abs": mean_pr_abs,
        "supervisor_abs_p95": p95_pr_abs,
        # validation-calibrated supervisor (ranking)
        "calibrated_supervisor": mean_cal_pr,
        "calibrated_supervisor_p95": p95_cal_pr,
        # M1 directional + M2 epistemic (p95 of the per-point geometric/epistemic risk profile)
        "radial_expected": geometric_risk,
        "anisotropy_gated": geometric_risk,
        "largest_eigenvalue": geometric_risk,
        "expected_epistemic": geometric_risk,
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
        mean_point_risk=mean_pr_rel,
        p95_point_risk=p95_pr_rel,
        mean_point_risk_abs=mean_pr_abs,
        p95_point_risk_abs=p95_pr_abs,
        mean_calibrated_point_risk=mean_cal_pr,
        p95_calibrated_point_risk=p95_cal_pr,
        max_domain_risk=max_domain_risk,
        time_outside_support=time_outside_support,
    )
