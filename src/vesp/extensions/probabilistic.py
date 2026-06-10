"""Stage 3C: exact linear-Gaussian posterior over equivalent-source strengths.

This is the *honest* MaxEnt move. The earlier point-estimate entropy experiments showed
that maximizing entropy over the sources does not improve accuracy and is strictly
dominated by ridge on out-of-distribution generalization. But MaxEnt's classical value
was never point-estimate RMSE — it is *calibrated uncertainty*.

Because the model is linear in ``sigma`` (``b = A sigma + noise``), the maximum-entropy
posterior consistent with a Gaussian likelihood and a Gaussian (Tikhonov / L2) prior is
itself Gaussian (a Gaussian is the maximum-entropy distribution under fixed mean and
covariance). That posterior is available in closed form:

    prior:      sigma ~ N(0, (noise_var / lambda) I)
    likelihood: b | sigma ~ N(A sigma, noise_var I)
    posterior:  sigma ~ N(mu, Sigma)
        Sigma = noise_var * (A^T A + lambda I)^{-1}
        mu    = (A^T A + lambda I)^{-1} A^T b     # == the ridge / Tikhonov solution

So the posterior MEAN is exactly the ridge point estimate (the accuracy story is
unchanged), and ``Sigma`` adds a calibrated covariance that propagates to predictive
error bars on the exterior field. Whether those error bars are actually calibrated (and
whether they correctly grow where the model is extrapolating, e.g. low altitude) is the
falsifiable question this module exists to answer.

This is NOT a full nonlinear/variational Bayesian framework and does not learn the noise
model; it is the exact conjugate posterior for the (already linear) equivalent-source
problem, plus calibration diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Standard-normal half-widths z such that P(|Z| <= z) = level. Hardcoded to avoid a
# scipy dependency in the hot path (values are exact to the digits shown).
_NORMAL_HALF_WIDTH = {
    0.50: 0.6744897501960817,
    0.68: 0.9944578832097531,
    0.80: 1.2815515594457414,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


@dataclass
class LinearGaussianPosterior:
    """Exact conjugate Gaussian posterior over source strengths (Bayesian ridge)."""

    mean: torch.Tensor  # (n_source,)
    cov: torch.Tensor  # (n_source, n_source)
    noise_var: float
    lambda_l2: float | None = None  # the Tikhonov weight used (set by fit / fit_evidence)

    @classmethod
    def fit(
        cls,
        operator: torch.Tensor,
        target: torch.Tensor,
        *,
        lambda_l2: float,
        noise_var: float | None = None,
        jitter: float = 1.0e-10,
    ) -> LinearGaussianPosterior:
        """Fit the conjugate posterior for ``b = A sigma + noise`` with an L2 prior.

        ``noise_var`` defaults to the empirical residual variance of the ridge mean
        (``||A mu - b||^2 / n_rows``), i.e. the data are assumed homoscedastic with the
        noise level implied by the fit. ``lambda_l2`` is the Tikhonov weight (prior
        precision up to the noise scale).
        """

        A = operator
        b = target
        n_rows, n_source = A.shape
        gram = A.transpose(-1, -2) @ A
        eye = torch.eye(n_source, dtype=A.dtype, device=A.device)
        regularized = gram + float(lambda_l2) * eye

        chol = _safe_cholesky(regularized, jitter=jitter)
        rhs = (A.transpose(-1, -2) @ b).unsqueeze(-1)
        mean = torch.cholesky_solve(rhs, chol).squeeze(-1)

        if noise_var is None:
            residual = A @ mean - b
            ssr = float(torch.sum(residual * residual).detach().cpu())
            noise_var = ssr / max(1.0, float(n_rows))
        noise_var = max(float(noise_var), torch.finfo(A.dtype).tiny)

        precision_inv = torch.cholesky_inverse(chol)
        cov = noise_var * precision_inv
        return cls(mean=mean, cov=cov, noise_var=noise_var, lambda_l2=float(lambda_l2))

    @classmethod
    def fit_evidence(
        cls,
        operator: torch.Tensor,
        target: torch.Tensor,
        *,
        iters: int = 100,
        tol: float = 1.0e-6,
        jitter: float = 1.0e-10,
    ) -> LinearGaussianPosterior:
        """Empirical-Bayes (MacKay evidence) selection of noise variance and prior precision.

        Instead of a fixed ``lambda_l2`` and a naive residual-variance noise estimate, this
        iterates the evidence-maximization updates for the linear-Gaussian model:

            prior precision  alpha,  noise precision  beta
            gamma = sum_i (beta d_i) / (beta d_i + alpha)        # effective # parameters
            alpha <- gamma / (m^T m)
            beta  <- (N - gamma) / ||A m - b||^2

        where ``d_i`` are the eigenvalues of ``A^T A`` and ``m`` the posterior mean. The
        global noise/prior scale is then optimal for the *training* data, which improves
        in-distribution calibration. It does NOT model heteroscedasticity, so it cannot by
        itself fix extreme-OOD overconfidence (uncertainty that should grow with altitude).
        """

        A = operator
        b = target
        n_rows, n_source = A.shape
        tiny = torch.finfo(A.dtype).tiny
        gram = A.transpose(-1, -2) @ A
        atb = A.transpose(-1, -2) @ b
        eye = torch.eye(n_source, dtype=A.dtype, device=A.device)
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)

        beta = 1.0 / max(float(torch.var(b).detach().cpu()), tiny)
        alpha = 1.0
        for _ in range(int(iters)):
            chol = _safe_cholesky(beta * gram + alpha * eye, jitter=jitter)
            mean = torch.cholesky_solve((beta * atb).unsqueeze(-1), chol).squeeze(-1)
            gamma = float(torch.sum((beta * eigvals) / (beta * eigvals + alpha)).detach().cpu())
            mtm = max(float((mean @ mean).detach().cpu()), tiny)
            residual = A @ mean - b
            sse = max(float((residual @ residual).detach().cpu()), tiny)
            new_alpha = gamma / mtm
            new_beta = max(float(n_rows) - gamma, 1.0e-6) / sse
            converged = abs(new_alpha - alpha) <= tol * max(alpha, tiny) and abs(new_beta - beta) <= tol * max(beta, tiny)
            alpha, beta = new_alpha, new_beta
            if converged:
                break

        lambda_l2 = alpha / max(beta, tiny)
        noise_var = 1.0 / max(beta, tiny)
        return cls.fit(A, b, lambda_l2=lambda_l2, noise_var=noise_var, jitter=jitter)

    def predict(
        self,
        query_operator: torch.Tensor,
        *,
        include_noise: bool = True,
        noise_variance: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predictive mean and variance for rows of ``query_operator`` (each row -> scalar).

        ``noise_variance`` overrides the aleatoric term: pass a per-row tensor (shape ``(m,)``)
        for a heteroscedastic noise model, or a scalar. ``None`` uses the fitted global
        ``self.noise_var`` (homoscedastic, unchanged behavior).

        Returns ``mean`` (m,), ``variance`` (m,, total = epistemic + aleatoric),
        ``epistemic_variance`` (m,), and ``std`` (m,).
        """

        mean = query_operator @ self.mean
        cov_q = query_operator @ self.cov
        epistemic = torch.sum(cov_q * query_operator, dim=-1).clamp_min(0.0)
        if not include_noise:
            noise: torch.Tensor | float = 0.0
        elif noise_variance is None:
            noise = self.noise_var
        else:
            noise = noise_variance
        variance = epistemic + noise
        return {
            "mean": mean,
            "variance": variance,
            "epistemic_variance": epistemic,
            "std": torch.sqrt(variance.clamp_min(torch.finfo(mean.dtype).tiny)),
        }


@dataclass
class AltitudeNoiseModel:
    """Heteroscedastic predictive noise as a power law in altitude ``h = r - 1``.

        sigma^2(h) = exp(log_a) * h^(-b),   b >= 0   (grows toward the surface)

    The "noise" here is dominantly altitude-dependent MODEL MISFIT: the equivalent-source
    model represents the high-frequency near-surface residual worst, so the predictive
    intervals must widen at low altitude. A single global noise term cannot do this; this
    monotone, physically-motivated law can (and it extrapolates sensibly). Fit by maximizing
    the Gaussian predictive log-likelihood of training residuals given ``epistemic_var +
    sigma^2(h)``.
    """

    log_a: float
    b: float
    h_floor: float = 1.0e-3

    def variance(self, radii: torch.Tensor) -> torch.Tensor:
        h = (radii - 1.0).clamp_min(self.h_floor)
        a = torch.as_tensor(self.log_a, dtype=radii.dtype, device=radii.device).exp()
        return a * h.pow(-float(self.b))

    @classmethod
    def fit(
        cls,
        radii: torch.Tensor,
        residuals: torch.Tensor,
        epistemic_var: torch.Tensor,
        *,
        iters: int = 400,
        lr: float = 0.05,
        h_floor: float = 1.0e-3,
    ) -> AltitudeNoiseModel:
        radii = radii.detach()
        res2 = (residuals.detach()) ** 2
        epistemic_var = epistemic_var.detach()
        h = (radii - 1.0).clamp_min(h_floor)

        init_var = float(torch.clamp(torch.mean(res2) - torch.mean(epistemic_var), min=1.0e-12).detach().cpu())
        log_a = torch.tensor(math.log(init_var), dtype=radii.dtype, requires_grad=True)
        raw_b = torch.zeros((), dtype=radii.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([log_a, raw_b], lr=lr)
        for _ in range(int(iters)):
            b = torch.nn.functional.softplus(raw_b)
            sigma2 = torch.exp(log_a) * h.pow(-b)
            total = epistemic_var + sigma2
            nll = 0.5 * torch.log(total) + 0.5 * res2 / total
            optimizer.zero_grad(set_to_none=True)
            nll.mean().backward()
            optimizer.step()
        b_final = float(torch.nn.functional.softplus(raw_b).detach().cpu())
        return cls(log_a=float(log_a.detach().cpu()), b=b_final, h_floor=h_floor)

    @property
    def a(self) -> float:
        return math.exp(self.log_a)


def _safe_cholesky(matrix: torch.Tensor, *, jitter: float) -> torch.Tensor:
    """Cholesky with escalating diagonal jitter for near-singular Gram matrices."""

    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    scale = float(torch.linalg.diagonal(matrix).abs().mean().clamp_min(1.0).detach().cpu())
    current = float(jitter)
    last_error: RuntimeError | None = None
    for _ in range(12):
        try:
            return torch.linalg.cholesky(matrix + current * scale * eye)
        except RuntimeError as exc:  # not positive-definite yet
            last_error = exc
            current *= 10.0
    raise RuntimeError(f"Cholesky failed even with jitter; matrix is too ill-conditioned ({last_error})")


def calibration_metrics(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: tuple[float, ...] = (0.5, 0.68, 0.9, 0.95),
) -> dict[str, float]:
    """Calibration diagnostics for Gaussian predictive intervals.

    ``picp_XX`` is the empirical coverage of the nominal ``XX``% interval — for a
    well-calibrated model it should match the nominal level. ``z_std`` should be ~1
    (overconfident if > 1, underconfident if < 1). ``nll`` is the Gaussian negative log
    predictive density (lower is better).
    """

    eps = torch.finfo(pred_mean.dtype).tiny
    std = pred_std.clamp_min(eps)
    z = (target - pred_mean) / std
    abs_z = torch.abs(z)
    out: dict[str, float] = {
        "n": int(target.numel()),
        "rmse": float(torch.sqrt(torch.mean((target - pred_mean) ** 2)).detach().cpu()),
        "mean_pred_std": float(torch.mean(std).detach().cpu()),
        "z_mean": float(torch.mean(z).detach().cpu()),
        "z_std": float(torch.std(z).detach().cpu()),
    }
    for level in levels:
        half = _NORMAL_HALF_WIDTH.get(round(level, 2))
        if half is None:
            continue
        coverage = float(torch.mean((abs_z <= half).to(pred_mean.dtype)).detach().cpu())
        out[f"picp_{int(round(level * 100))}"] = coverage
    variance = (std * std)
    nll = 0.5 * torch.log(2.0 * torch.pi * variance) + 0.5 * z * z
    out["nll"] = float(torch.mean(nll).detach().cpu())
    # Closed-form Gaussian CRPS (lower is better; ~0.234*sigma per point is NOT the target —
    # a perfectly-calibrated unit forecast averages to ~0.5642*sigma).
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    out["crps"] = float(torch.mean(crps).detach().cpu())
    return out


def empirical_acceleration_covariance(samples: torch.Tensor) -> torch.Tensor:
    """Compute covariance from acceleration samples ``[S, B, 3]`` (utility kept from scaffold)."""

    centered = samples - samples.mean(dim=0, keepdim=True)
    return torch.einsum("sbi,sbj->bij", centered, centered) / max(1, samples.shape[0] - 1)
