"""Vector-level (3D) calibration metrics for the acceleration-error covariance.

Component-wise PICP/z_std (in :mod:`vesp.extensions.probabilistic`) treat the three
acceleration-error components independently. When VESP-UQ produces a full ``3x3`` predictive
covariance, the natural calibration check is the *ellipsoid* one: the squared Mahalanobis
distance ``d^2 = e^T Cov^{-1} e`` of a calibrated 3D Gaussian follows a chi-square distribution
with 3 degrees of freedom. Coverage of the ``p``-level chi-square ellipsoid should match ``p``,
and the mean ``d^2`` should be ~3.
"""

from __future__ import annotations

import math

import torch


def chi2_3_cdf(x: torch.Tensor | float) -> torch.Tensor:
    """CDF of the chi-square(3) distribution (closed form via the regularized lower gamma)."""

    x = torch.as_tensor(x, dtype=torch.float64)
    x = x.clamp_min(0.0)
    s = x / 2.0
    # P(3/2, s) = erf(sqrt(s)) - (2/sqrt(pi)) sqrt(s) e^{-s}
    return torch.erf(torch.sqrt(s)) - (2.0 / math.sqrt(math.pi)) * torch.sqrt(s) * torch.exp(-s)


def chi2_3_ppf(p: float, *, hi: float = 60.0, iters: int = 80) -> float:
    """Inverse CDF (quantile) of chi-square(3) via bisection on the monotone CDF."""

    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, high = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + high)
        if float(chi2_3_cdf(mid)) < p:
            lo = mid
        else:
            high = mid
    return 0.5 * (lo + high)


# Cached thresholds for the standard levels.
_CHI2_3_LEVELS = (0.50, 0.68, 0.90, 0.95)
_CHI2_3_THRESHOLDS = {p: chi2_3_ppf(p) for p in _CHI2_3_LEVELS}


def mahalanobis_squared(
    error_vectors: torch.Tensor, covariances: torch.Tensor, *, jitter: float = 1.0e-12
) -> torch.Tensor:
    """``d^2_i = e_i^T Cov_i^{-1} e_i`` for ``error_vectors`` (N,3) and ``covariances`` (N,3,3)."""

    if error_vectors.ndim != 2 or error_vectors.shape[-1] != 3:
        raise ValueError("error_vectors must have shape (N, 3)")
    if covariances.shape != (error_vectors.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3) matching error_vectors")
    eye = torch.eye(3, dtype=covariances.dtype, device=covariances.device)
    scale = torch.diagonal(covariances, dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
    reg = covariances + jitter * scale * eye
    sol = torch.linalg.solve(reg, error_vectors.unsqueeze(-1)).squeeze(-1)
    return torch.sum(error_vectors * sol, dim=-1).clamp_min(0.0)


def diagonal_covariances(std_components: torch.Tensor) -> torch.Tensor:
    """Build ``(N, 3, 3)`` diagonal covariances from per-component std ``(N, 3)``."""

    return torch.diag_embed(std_components.clamp_min(0.0) ** 2)


def vector_calibration_metrics(error_vectors: torch.Tensor, covariances: torch.Tensor) -> dict:
    """Ellipsoid (Mahalanobis / chi-square-3) calibration metrics for a 3D error covariance.

    Returns ``ellipsoid_picp_{50,68,90,95}`` (empirical coverage of each chi-square(3) ellipsoid;
    should match the nominal level for a calibrated model), plus ``mean_mahalanobis_d2`` (~3 when
    calibrated) and ``median_mahalanobis_d2``. Pass diagonal covariances (see
    :func:`diagonal_covariances`) for the cheaper diagonal approximation.
    """

    d2 = mahalanobis_squared(error_vectors, covariances)
    out: dict[str, float] = {
        "n": int(d2.numel()),
        "mean_mahalanobis_d2": float(d2.mean().detach().cpu()),
        "median_mahalanobis_d2": float(d2.median().detach().cpu()),
    }
    for p in _CHI2_3_LEVELS:
        thr = _CHI2_3_THRESHOLDS[p]
        out[f"ellipsoid_picp_{int(round(p * 100))}"] = float((d2 <= thr).to(torch.float64).mean().detach().cpu())
    return out
