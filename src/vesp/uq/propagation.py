"""Phase 4 (EXPLORATORY): Monte Carlo orbit-dispersion sampling from the VESP-UQ posterior.

Propagates a batch of trajectories under a central gravity field (or a supplied base
acceleration) plus Monte Carlo samples drawn from the VESP-UQ force-error posterior. Each sample
draws a source-strength vector ``sigma ~ N(mean, cov)`` and adds the resulting equivalent-source
acceleration-error field to the base field, so the spread across samples reflects the *force-error
posterior* uncertainty.

**Scope / honesty caveat.** This is an exploratory diagnostic, NOT a validated operational
orbit-determination or state-covariance product:

- It samples the local *force-model* error posterior; it does not model measurement processing,
  dynamic mismodelling beyond the fitted residual, or realistic process noise.
- The empirical force-risk finding is that the VESP-UQ score does **not** rank a surrogate's
  long-horizon *position* error on the in-distribution lunar set; do not read this dispersion as a
  calibrated position-error or covariance-realism claim. See ``docs/VESP_UQ_LIMITATIONS.md``.

The sampled error field is kept exactly consistent with the fitted posterior: it honors the
plugin's ``acceleration_sign`` and softening ``eps`` and the source quadrature weights, so the
per-point sample mean/covariance match :meth:`VESPUQPlugin.predict_uncertainty` /
:meth:`predict_covariance_3x3`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from vesp.core.kernels import acceleration_kernel
from vesp.uq.plugin import VESPUQPlugin


def _stable_cholesky(cov: torch.Tensor, *, base_rel_jitter: float = 1.0e-12, max_tries: int = 7) -> torch.Tensor:
    """Cholesky factor of ``cov`` with a *scale-relative* jitter, grown only if not PD.

    The jitter starts at ``base_rel_jitter * mean|diag(cov)|`` so it is negligible relative to the
    posterior covariance (unlike a fixed absolute floor, which can inflate the sampled covariance
    when the diagonal is itself tiny).
    """

    n = cov.shape[0]
    eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
    scale = torch.diagonal(cov).abs().mean().clamp_min(torch.finfo(cov.dtype).tiny)
    jitter = float(base_rel_jitter) * float(scale)
    last_err: RuntimeError | None = None
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(cov + jitter * eye)
        except RuntimeError as err:  # not positive-definite -> grow the jitter and retry
            last_err = err
            jitter = jitter * 10.0 if jitter > 0.0 else float(scale) * 1.0e-12
    raise RuntimeError(f"posterior covariance is not positive-definite even with jitter: {last_err}")


def draw_posterior_samples(
    plugin: VESPUQPlugin, n_samples: int, seed: int = 42
) -> torch.Tensor:
    """Draw `n_samples` of source strengths (sigma) from the VESP-UQ posterior.
    
    Returns:
        sigma_samples: Tensor of shape [n_samples, n_sources]
    """
    if plugin.posterior is None:
        raise RuntimeError("Plugin must be fitted before drawing samples.")
        
    mean = plugin.posterior.mean
    cov = plugin.posterior.cov

    generator = torch.Generator(device=mean.device).manual_seed(seed)

    # Cholesky sampling. The jitter is scaled to the covariance magnitude (a fixed absolute floor
    # like 1e-8 can be the same order as this posterior's diagonal and would materially inflate the
    # sampled covariance); it is grown only if the factorization is not PD.
    L = _stable_cholesky(cov)

    # Z ~ N(0, I)
    Z = torch.randn(n_samples, mean.shape[0], device=mean.device, dtype=mean.dtype, generator=generator)
    
    # samples = mean + L @ Z
    # Z is [N_samples, n_sources], L is [n_sources, n_sources]
    samples = mean.unsqueeze(0) + (L @ Z.unsqueeze(-1)).squeeze(-1)
    
    return samples


class VESPMonteCarloPropagator:
    """
    Batched RK4 Monte Carlo propagator for VESP-UQ error fields.
    
    Integrates N parallel trajectories where each trajectory is perturbed
    by a different sample from the VESP-UQ posterior.
    """

    def __init__(
        self,
        plugin: VESPUQPlugin,
        n_samples: int = 100,
        dt_s: float = 60.0,
        mu: float = 1.0,  # Normalized central gravity parameter
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        base_accel_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.plugin = plugin
        self.n_samples = n_samples
        self.dt = float(dt_s)
        self.mu = float(mu)
        self.device = torch.device(device)
        self.dtype = dtype
        self.base_accel_fn = base_accel_fn
        
        # 1. Draw MC samples of the force error
        self.sigma_samples = draw_posterior_samples(plugin, n_samples, seed=seed).to(dtype=self.dtype, device=self.device)
        
        # 2. Extract source configuration
        self.source_positions = plugin.sources.positions.to(dtype=self.dtype, device=self.device)
        self.source_weights = plugin.sources.weights.to(dtype=self.dtype, device=self.device)
        # Honor the fitted operator's acceleration sign + softening so the sampled error field is
        # consistent with the posterior the samples were drawn from (otherwise the MC field can be
        # sign-flipped or unsoftened relative to the fit).
        self.eps = float(plugin.eps)
        self.accel_sign = float(plugin.acceleration_sign)

        # Pre-multiply samples by weights for fast kernel evaluation
        self.weighted_samples = self.sigma_samples * self.source_weights.unsqueeze(0)

    def _rhs(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate [v; a] for a state batch [N, 6]."""
        r = s[:, :3]  # [N, 3]
        v = s[:, 3:]  # [N, 3]
        
        # Base acceleration (either ST-LRPS or point mass)
        if self.base_accel_fn is not None:
            a_central = self.base_accel_fn(r)
        else:
            r2 = torch.sum(r * r, dim=1, keepdim=True)
            r_mag3 = r2 * torch.sqrt(r2)
            a_central = -self.mu * r / r_mag3
        
        # VESP-UQ Error Evaluation
        # Kernel: [N, n_sources, 3] -- sign/eps match the fitted operator (weights applied below).
        ker = acceleration_kernel(r, self.source_positions, eps=self.eps, sign=self.accel_sign)

        # Multiply by weighted sigma samples [N, n_sources] and sum over sources
        a_error = (ker * self.weighted_samples.unsqueeze(-1)).sum(dim=1)
        
        # Total acceleration
        a_total = a_central + a_error
        
        return torch.cat([v, a_total], dim=1)

    def _rk4_step(self, s: torch.Tensor, h: float) -> torch.Tensor:
        k1 = self._rhs(s)
        k2 = self._rhs(s + (h * 0.5) * k1)
        k3 = self._rhs(s + (h * 0.5) * k2)
        k4 = self._rhs(s + h * k3)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(
        self,
        y0: np.ndarray,  # Single initial state [6,] (position, velocity)
        duration_s: float,
        output_dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate the initial state subject to N different error realizations.
        
        Returns:
            t_out: (T,) array of time points
            Y_out: (T, N_samples, 6) array of propagated states
        """
        if y0.shape != (6,):
            raise ValueError("y0 must be a 1D array of shape (6,)")
            
        N = self.n_samples
        snap_interval = float(output_dt_s)
        total_time = float(duration_s)

        steps_per_snap = max(1, round(snap_interval / self.dt))
        dt_eff = snap_interval / steps_per_snap
        n_snaps = max(1, round(total_time / snap_interval))

        t_out = np.linspace(0.0, n_snaps * snap_interval, n_snaps + 1, dtype=np.float64)
        Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)

        # Broadcast initial state to all N samples
        y0_batch = np.tile(y0, (N, 1))
        state = torch.tensor(y0_batch, dtype=self.dtype, device=self.device)

        Y_out[0] = state.cpu().numpy()

        t_prop_start = time.perf_counter()

        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                state = self._rk4_step(state, dt_eff)

            Y_out[snap_idx + 1] = state.cpu().numpy()

        t_prop = time.perf_counter() - t_prop_start
        total_steps = n_snaps * steps_per_snap
        traj_steps_per_s = (N * total_steps) / max(t_prop, 1e-9)
        print(f"[VESP-UQ MC] Propagation complete: {t_prop:.2f}s ({traj_steps_per_s:,.0f} steps/s)")

        return t_out, Y_out
