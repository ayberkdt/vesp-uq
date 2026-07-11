"""VESP-UQ trajectory score variants for the score-ablation study (WP6).

The default supervisor is one of many ways to collapse a per-point predictive distribution into a
single trajectory force-risk scalar. This module builds a family of alternative scores from the
quantities VESP-UQ already exposes per point -- the expected error, the total / radial / tangential
predictive spread, and the full 3x3 covariance geometry -- combined with different
trajectory-level aggregators (mean, p95, max, top-k, periapsis window, low-altitude integral).

Each variant maps to an entry in the journal-review list; the Mahalanobis-residual variant needs
true per-point residual *vectors*, which a generated Keplerian ensemble does not carry, so it is
reported as unavailable rather than fabricated.

Everything targets trajectory-level true FORCE-model error (``a_reference - a_surrogate``), never
position error. Scores are deterministic given a fitted plugin.
"""

from __future__ import annotations

import math

import torch

from vesp.uq.scoring import SCORING_FUNCTIONS, score_sigma_profile

__all__ = ["PRODUCTION_SCORE_VARIANTS", "SCORE_VARIANTS", "compute_score_variants"]

# Variant name -> short description (also the stable column order for tables).
SCORE_VARIANTS = (
    "expected_error_mean",
    "expected_error_p95",
    "expected_error_max",
    "expected_error_topk",
    "expected_error_periapsis_window",
    "expected_error_low_alt_integral",
    "expected_error_plus_noise_floor",
    "altitude_weighted_expected_error",
    "expected_error_x_ood",
    "expected_error_x_alt_x_ood",
    "trace_covariance",
    "largest_eigenvalue_covariance",
    "radial_component_uncertainty",
    "tangential_component_uncertainty",
    "covariance_anisotropy",
    # M1 directional + M2 epistemic first-class composites (production scoring modes; p95-aggregated)
    "radial_expected_p95",
    # diagonal-covariance ablation of radial_expected_p95 (R2WP-4: measures the ranking impact of
    # dropping the cross-covariance terms from the radial projection)
    "radial_expected_diag_p95",
    "anisotropy_gated_p95",
    "expected_epistemic_p95",
)

# Ablation variants that must stay tied to first-class production scoring modes.
# Keys are report/table names; values are entries from ``scoring.SCORING_FUNCTIONS``.
PRODUCTION_SCORE_VARIANTS = {
    "radial_expected_p95": "radial_expected",
    "radial_expected_diag_p95": "radial_expected_diag",
    "anisotropy_gated_p95": "anisotropy_gated",
    "expected_epistemic_p95": "expected_epistemic",
}

_missing_scoring = sorted(set(PRODUCTION_SCORE_VARIANTS.values()) - set(SCORING_FUNCTIONS))
if _missing_scoring:
    raise RuntimeError(f"score variant registry references unknown scoring modes: {_missing_scoring}")


def _aggregate(values: torch.Tensor, mode: str, weights: torch.Tensor | None = None) -> float:
    v = values.reshape(-1).to(torch.float64)
    if v.numel() == 0:
        return float("nan")
    if mode == "mean":
        if weights is None:
            return float(v.mean())
        w = weights.reshape(-1).to(torch.float64).clamp_min(0.0)
        wsum = float(w.sum())
        return float((v * w).sum() / wsum) if wsum > 0 else float(v.mean())
    if mode == "max":
        return float(v.max())
    if mode == "p95":
        return float(torch.quantile(v, 0.95))
    if mode == "topk":
        k = max(1, int(math.ceil(0.10 * v.numel())))
        return float(torch.topk(v, k).values.mean())
    raise ValueError(f"unknown aggregate mode {mode!r}")


def _periapsis_window_mean(values: torch.Tensor, radius: torch.Tensor, *, frac: float = 0.25) -> float:
    r = radius.reshape(-1).to(torch.float64)
    v = values.reshape(-1).to(torch.float64)
    if r.numel() == 0:
        return float("nan")
    lo, hi = float(r.min()), float(r.max())
    cutoff = lo + frac * (hi - lo) if hi > lo else hi
    mask = r <= cutoff
    sel = v[mask]
    return float(sel.mean()) if sel.numel() else float(v.min())


def _low_alt_integral(values: torch.Tensor, radius: torch.Tensor, low_alt_radius: float) -> float:
    r = radius.reshape(-1).to(torch.float64)
    v = values.reshape(-1).to(torch.float64)
    mask = r <= float(low_alt_radius)
    return float(v[mask].sum()) if mask.any() else 0.0


def compute_score_variants(
    plugin,
    trajectories,
    *,
    low_altitude_radius: float = 1.15,
    weights=None,
) -> dict:
    """Per-trajectory scores for every variant in :data:`SCORE_VARIANTS`.

    Returns ``{"scores": {variant: tensor(n_traj)}, "unavailable": {variant: reason}}``. All
    trajectory points are scored in one batched pass (``predict_uncertainty`` +
    ``predict_covariance_3x3``); per-trajectory aggregation then matches the rest of the layer.
    """

    traj_list = [plugin._prep_positions(t) for t in trajectories]
    if not traj_list:
        return {"scores": {name: torch.empty(0) for name in SCORE_VARIANTS}, "unavailable": {}}
    lengths = [int(t.shape[0]) for t in traj_list]
    pts = torch.cat(traj_list, dim=0)

    pred = plugin.predict_uncertainty(pts)
    cov = plugin.predict_covariance_3x3(pts)
    domain = plugin.domain_support_score(pred.positions) if getattr(plugin, "domain_support", False) else None

    radius = pred.radius.to(torch.float64)
    expected = pred.expected_error.to(torch.float64)
    sigma = pred.sigma.to(torch.float64)
    noise_floor = float(torch.median(sigma)) if sigma.numel() else 0.0
    ee_floor = torch.sqrt(expected * expected + noise_floor * noise_floor)
    # altitude emphasis toward periapsis (bounded, ~1 at the low-altitude radius)
    alt_weight = (float(low_altitude_radius) / radius.clamp_min(1e-9)).clamp(0.0, 4.0) ** 4
    ood = domain.to(torch.float64) if domain is not None else torch.ones_like(expected)

    C = cov.covariance.to(torch.float64)  # (N, 3, 3)
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)
    eig = torch.linalg.eigvalsh(C)  # ascending
    lam_max = eig[..., -1].clamp_min(0.0)
    r_hat = pred.positions.to(torch.float64) / radius.clamp_min(1e-9).unsqueeze(-1)
    radial_var = torch.einsum("ni,nij,nj->n", r_hat, C, r_hat).clamp_min(0.0)
    tangential_var = (trace - radial_var).clamp_min(0.0)
    anisotropy = lam_max / trace.clamp_min(1e-30)

    # per-point value tensors that feed each variant
    point_values = {
        "expected_error_mean": expected,
        "expected_error_p95": expected,
        "expected_error_max": expected,
        "expected_error_topk": expected,
        "expected_error_plus_noise_floor": ee_floor,
        "altitude_weighted_expected_error": expected * alt_weight,
        "expected_error_x_ood": expected * ood,
        "expected_error_x_alt_x_ood": expected * alt_weight * ood,
        "trace_covariance": trace,
        "largest_eigenvalue_covariance": lam_max,
        "radial_component_uncertainty": torch.sqrt(radial_var),
        "tangential_component_uncertainty": torch.sqrt(tangential_var),
        "covariance_anisotropy": anisotropy,
    }
    agg_mode = {
        "expected_error_p95": "p95",
        "expected_error_max": "max",
        "expected_error_topk": "topk",
    }

    weight_list = [None] * len(traj_list) if weights is None else list(weights)
    scores = {name: torch.empty(len(traj_list), dtype=torch.float64) for name in SCORE_VARIANTS}
    offset = 0
    for i, n in enumerate(lengths):
        sl = slice(offset, offset + n)
        offset += n
        w = weight_list[i]
        w_t = None if w is None else torch.as_tensor(w, dtype=torch.float64)
        r_i = radius[sl]
        for name, vals in point_values.items():
            mode = agg_mode.get(name, "mean")
            scores[name][i] = _aggregate(vals[sl], mode, weights=w_t if mode == "mean" else None)
        for name, scoring in PRODUCTION_SCORE_VARIANTS.items():
            scores[name][i] = score_sigma_profile(
                sigma[sl],
                radius[sl],
                scoring=scoring,
                low_altitude_radius=low_altitude_radius,
                expected_error=expected[sl],
                mean_error_vector=pred.mean_error[sl],
                std_components=pred.std_components[sl],
                covariance=C[sl],
                positions=pred.positions[sl],
                epistemic_fraction=pred.epistemic_fraction[sl],
                weights=w_t,
            ).risk_score
        scores["expected_error_periapsis_window"][i] = _periapsis_window_mean(expected[sl], r_i)
        scores["expected_error_low_alt_integral"][i] = _low_alt_integral(
            expected[sl], r_i, low_altitude_radius
        )

    unavailable = {
        "mahalanobis_residual": "requires true per-point residual vectors; unavailable for a "
        "generated Keplerian ensemble (no per-point a_reference - a_surrogate pairs)."
    }
    return {"scores": scores, "unavailable": unavailable}
