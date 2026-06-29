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


# Standard-normal two-sided interval half-widths (z_alpha/2), reused for component PICP / Winkler.
_NORMAL_HALF_WIDTH = {
    0.50: 0.6744897501960817,
    0.68: 0.9944578832097531,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def local_radial_frame(positions: torch.Tensor) -> torch.Tensor:
    """Per-point orthonormal frame ``(N, 3, 3)`` with rows ``[radial, tangential_1, tangential_2]``.

    The radial axis is ``position / ||position||``; the two tangential axes complete a right-handed
    orthonormal basis (built from a reference direction not parallel to the radial). Multiplying a
    world-frame vector ``v`` by this matrix (``R @ v``) yields its ``[radial, t1, t2]`` components.
    Physically the radial component dominates the altitude-dependent force-model error, so splitting
    calibration into radial vs tangential exposes *where* a band is mis-calibrated.
    """

    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    pos = positions.to(torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    radial = pos / torch.linalg.norm(pos, dim=-1, keepdim=True).clamp_min(tiny)
    # reference axis: e_z, except where radial is nearly parallel to e_z (then use e_x)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(radial)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).expand_as(radial)
    near_pole = (radial[:, 2].abs() > 0.9).unsqueeze(-1)
    ref = torch.where(near_pole, ex, ez)
    t1 = torch.linalg.cross(radial, ref)
    t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True).clamp_min(tiny)
    t2 = torch.linalg.cross(radial, t1)
    t2 = t2 / torch.linalg.norm(t2, dim=-1, keepdim=True).clamp_min(tiny)
    return torch.stack([radial, t1, t2], dim=1)  # (N, 3, 3), rows are the frame axes


def component_calibration_metrics(
    error_vectors: torch.Tensor,
    covariances: torch.Tensor,
    positions: torch.Tensor,
    *,
    levels: tuple[float, ...] = (0.68, 0.90, 0.95),
    winkler_level: float = 0.90,
) -> dict:
    """Radial vs tangential calibration of the predictive error covariance in the local frame.

    Rotates the predictive residual and its ``3x3`` covariance into the per-point
    ``[radial, tangential]`` frame (:func:`local_radial_frame`), then reports, separately for the
    radial component and the pooled tangential components: ``z_std`` (~1 calibrated), interval
    coverage ``picp_XX`` (should match the nominal level), and the mean **Winkler interval score**
    at ``winkler_level`` (sharpness + coverage in one number, lower is better). The scalar
    ``calibration_error_90`` is the mean ``|picp_90 - 0.90|`` over the two component groups -- one
    comparable number to put beside the PICP table. The predictive residual is assumed mean-centered
    (the posterior mean is subtracted upstream), so intervals are centered at 0.
    """

    if error_vectors.shape != positions.shape:
        raise ValueError("error_vectors and positions must both have shape (N, 3)")
    if covariances.shape != (error_vectors.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3) matching error_vectors")
    e = error_vectors.to(torch.float64)
    cov = covariances.to(torch.float64)
    rot = local_radial_frame(positions)
    e_local = torch.einsum("nij,nj->ni", rot, e)               # (N, 3) [rad, t1, t2]
    cov_local = torch.einsum("nij,njk,nlk->nil", rot, cov, rot)  # R cov R^T
    var_local = torch.diagonal(cov_local, dim1=-2, dim2=-1).clamp_min(0.0)  # (N, 3)
    std_local = torch.sqrt(var_local).clamp_min(torch.finfo(torch.float64).tiny)

    z_radial = (e_local[:, 0] / std_local[:, 0]).reshape(-1)
    z_tangential = (e_local[:, 1:] / std_local[:, 1:]).reshape(-1)  # pool both tangential axes

    def _group(z: torch.Tensor, std: torch.Tensor, err: torch.Tensor, prefix: str) -> dict:
        out = {f"{prefix}_z_std": float(torch.std(z).detach().cpu()),
               f"{prefix}_z_mean": float(torch.mean(z).detach().cpu())}
        abs_z = z.abs()
        for level in levels:
            half = _NORMAL_HALF_WIDTH.get(round(float(level), 2))
            if half is None:
                continue
            out[f"{prefix}_picp_{int(round(level * 100))}"] = float(
                torch.mean((abs_z <= half).to(torch.float64)).detach().cpu())
        # mean Winkler interval score at winkler_level (interval centered at 0: [-h*std, h*std])
        h = _NORMAL_HALF_WIDTH.get(round(float(winkler_level), 2), 1.6448536269514722)
        alpha = 1.0 - float(winkler_level)
        width = 2.0 * h * std
        below = (-h * std - err).clamp_min(0.0)
        above = (err - h * std).clamp_min(0.0)
        winkler = width + (2.0 / alpha) * (below + above)
        out[f"{prefix}_winkler_{int(round(winkler_level * 100))}"] = float(
            torch.mean(winkler).detach().cpu())
        return out

    metrics: dict[str, float] = {"n": int(e.shape[0])}
    metrics.update(_group(z_radial, std_local[:, 0], e_local[:, 0], "radial"))
    metrics.update(_group(z_tangential, std_local[:, 1:].reshape(-1),
                          e_local[:, 1:].reshape(-1), "tangential"))
    rad90 = metrics.get("radial_picp_90")
    tan90 = metrics.get("tangential_picp_90")
    devs = [abs(v - 0.90) for v in (rad90, tan90) if v is not None]
    if devs:
        metrics["calibration_error_90"] = sum(devs) / len(devs)
    return metrics


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
