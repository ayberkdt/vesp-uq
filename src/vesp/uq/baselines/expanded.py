"""Expanded force-risk baselines for the journal comparison (WP5).

Beyond the trivial single-feature baselines, a fair comparison needs to ask whether VESP-UQ beats
*combinations* of cheap signals. This module provides:

* altitude+uncertainty and altitude+OOD **hybrids** (z-scored, with a tunable mixing weight),
* a **learned linear (ridge) supervisor** over the available per-trajectory features, and
* an **altitude-bin empirical residual-RMSE lookup** (a cheap data-driven baseline that needs no
  uncertainty model at all).

All hybrid/ranker hyper-parameters are chosen on a validation split and reported on a disjoint test
split by the driver in :mod:`vesp.uq.ablation`; the functions here are pure and label-free except
for :func:`fit_ridge_ranker`, which fits weights on an explicitly supplied training target.

Everything targets trajectory-level true FORCE-model error, never position error.
"""

from __future__ import annotations

import torch

__all__ = [
    "zscore",
    "altitude_uncertainty_hybrid",
    "altitude_ood_hybrid",
    "fit_ridge_ranker",
    "apply_ridge_ranker",
    "altitude_bin_rmse_lookup",
    "HYBRID_WEIGHTS",
]

HYBRID_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def zscore(x: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
    """Standardize ``x`` using the mean/std of ``ref`` (or of ``x`` itself when ``ref`` is None)."""

    x = x.reshape(-1).to(torch.float64)
    base = x if ref is None else ref.reshape(-1).to(torch.float64)
    mean = base.mean()
    std = base.std()
    if not torch.isfinite(std) or float(std) <= 0.0:
        return x - mean
    return (x - mean) / std


def altitude_uncertainty_hybrid(altitude_risk, uncertainty, a: float, ref=None) -> torch.Tensor:
    """``z(altitude_risk) + a * z(uncertainty)`` -- altitude-anchored, uncertainty-augmented."""

    alt = altitude_risk if ref is None else (altitude_risk, ref[0])
    unc = uncertainty if ref is None else (uncertainty, ref[1])
    za = zscore(*alt) if isinstance(alt, tuple) else zscore(alt)
    zu = zscore(*unc) if isinstance(unc, tuple) else zscore(unc)
    return za + float(a) * zu


def altitude_ood_hybrid(altitude_risk, domain_support, b: float, ref=None) -> torch.Tensor:
    """``z(altitude_risk) + b * z(domain_support)`` -- altitude-anchored, OOD-augmented."""

    return altitude_uncertainty_hybrid(altitude_risk, domain_support, b, ref=ref)


def fit_ridge_ranker(features: torch.Tensor, target: torch.Tensor, lam: float = 1.0) -> dict:
    """Closed-form ridge fit of ``target`` on z-scored ``features`` (plus bias).

    Returns the standardization stats + weights so :func:`apply_ridge_ranker` can score any split
    consistently. ``features`` is ``(n, d)`` and ``target`` is ``(n,)`` true force error.
    """

    X = features.to(torch.float64)
    y = target.reshape(-1).to(torch.float64)
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    Xz = (X - mean) / std
    n = Xz.shape[0]
    Xb = torch.cat([Xz, torch.ones(n, 1, dtype=torch.float64)], dim=1)
    d = Xb.shape[1]
    reg = float(lam) * torch.eye(d, dtype=torch.float64)
    reg[-1, -1] = 0.0  # do not penalize the bias term
    w = torch.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return {"mean": mean, "std": std, "weights": w, "lam": float(lam)}


def apply_ridge_ranker(model: dict, features: torch.Tensor) -> torch.Tensor:
    """Apply a :func:`fit_ridge_ranker` model to new features -> per-trajectory risk score."""

    X = features.to(torch.float64)
    Xz = (X - model["mean"]) / model["std"]
    Xb = torch.cat([Xz, torch.ones(Xz.shape[0], 1, dtype=torch.float64)], dim=1)
    return Xb @ model["weights"]


def altitude_bin_rmse_lookup(
    held_positions: torch.Tensor,
    held_error: torch.Tensor,
    trajectories,
    *,
    n_bins: int = 12,
) -> torch.Tensor:
    """Empirical baseline: per-trajectory mean of the held-out force-error RMS in its altitude bins.

    Held-out residual points are binned by radius; each bin's RMS force-error magnitude becomes a
    lookup table. A trajectory's score is the mean looked-up RMS over its points' radii -- a purely
    data-driven altitude risk with no uncertainty model. Higher = riskier.
    """

    pos = held_positions.to(torch.float64)
    err = held_error.to(torch.float64)
    r = torch.linalg.norm(pos, dim=-1)
    mag = torch.linalg.norm(err, dim=-1)
    edges = torch.linspace(float(r.min()), float(r.max()) + 1e-9, int(n_bins) + 1, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_rms = torch.empty(int(n_bins), dtype=torch.float64)
    for k in range(int(n_bins)):
        mask = (r >= edges[k]) & (r < edges[k + 1])
        bin_rms[k] = torch.sqrt(torch.mean(mag[mask] ** 2)) if bool(mask.any()) else torch.tensor(float("nan"))
    # fill empty bins by nearest finite neighbour so the lookup is always defined
    finite = torch.isfinite(bin_rms)
    if not bool(finite.all()) and bool(finite.any()):
        idx_finite = torch.nonzero(finite, as_tuple=False).reshape(-1)
        for k in range(int(n_bins)):
            if not bool(finite[k]):
                nearest = idx_finite[torch.argmin(torch.abs(idx_finite - k))]
                bin_rms[k] = bin_rms[nearest]

    out = torch.empty(len(trajectories), dtype=torch.float64)
    for i, traj in enumerate(trajectories):
        tr = torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1)
        b = torch.bucketize(tr, edges[1:-1])
        b = b.clamp(0, int(n_bins) - 1)
        out[i] = bin_rms[b].mean()
    _ = centers  # centers retained for potential reporting; lookup uses bin index
    return out
