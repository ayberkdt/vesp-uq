"""Phase 4 (EXPLORATORY): linearized (STM) force-error covariance propagation.

A deterministic, sampling-free alternative to the Monte Carlo orbit-dispersion sampler in
:mod:`vesp.uq.propagation`. Instead of drawing posterior samples and propagating a batch, it
propagates the *variational* sensitivity of the trajectory to the equivalent-source strengths and
maps the source posterior covariance into a state covariance:

    J(t) = d[x(t)] / d[sigma]    (6 x n_sources),      P(t) = J(t) Sigma_sigma J(t)^T   (6 x 6)

with the variational equation (around the nominal trajectory)

    Jr' = Jv,    Jv' = G(r) Jr + K(r),

where ``G(r) = d a_base / d r`` is the nominal-dynamics gravity gradient and ``K(r) = d a_error / d
sigma`` is the equivalent-source acceleration operator block (the same sign/eps/weight convention as
the fitted posterior). This is exactly the linearization of the MC sampler's *static force-error
field* model, so in the small-perturbation regime ``P(t)`` matches the MC sample covariance -- but
without sampling noise.

**Scope / honesty caveat.** This is an exploratory diagnostic, NOT validated operational orbit
determination or state-covariance realism:

- it maps the *local force-model* error posterior into a linearized state covariance; it does not
  model measurement processing, realistic process noise, or dynamic mismodelling beyond the fitted
  residual, and it does not claim the force-risk score predicts long-horizon position error;
- the linearization uses the central (point-mass ``mu``) gravity gradient by default; a custom base
  field uses a finite-difference Jacobian of that field. Large uncertainty or long horizons break
  the linear assumption -- cross-check against the MC sampler.

See ``docs/VESP_UQ_LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vesp.core.kernels import acceleration_kernel
from vesp.uq.plugin import VESPUQPlugin


@dataclass
class LinearCovarianceResult:
    """Output of :meth:`LinearForceErrorCovariancePropagator.propagate`.

    ``states`` is the nominal trajectory (base field + posterior-mean force-error correction);
    ``covariances`` is the linearized ``6x6`` state covariance per output time; ``position_sigma``
    is ``sqrt(trace(P[:3, :3]))`` (the 1-sigma 3D position dispersion implied by the force-error
    posterior).
    """

    times: np.ndarray  # (T,)
    states: np.ndarray  # (T, 6) nominal [r, v]
    covariances: np.ndarray  # (T, 6, 6)
    position_sigma: np.ndarray  # (T,)
    velocity_sigma: np.ndarray  # (T,)


class LinearForceErrorCovariancePropagator:
    """Linearized state-covariance propagation of the VESP-UQ force-error posterior."""

    def __init__(
        self,
        plugin: VESPUQPlugin,
        *,
        dt_s: float = 60.0,
        mu: float = 1.0,
        base_accel_fn=None,
        fd_eps: float = 1.0e-6,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if plugin.posterior is None:
            raise RuntimeError("Plugin must be fitted before propagating covariance.")
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt_s)
        self.mu = float(mu)
        self.base_accel_fn = base_accel_fn
        self.fd_eps = float(fd_eps)

        self.src_positions = plugin.sources.positions.to(dtype=self.dtype, device=self.device)
        self.src_weights = plugin.sources.weights.to(dtype=self.dtype, device=self.device)
        # Honor the fitted operator's sign/softening so K(r) matches the posterior (see propagation).
        self.eps = float(plugin.eps)
        self.accel_sign = float(plugin.acceleration_sign)
        self.cov_sigma = plugin.posterior.cov.to(dtype=self.dtype, device=self.device)
        self.mean_sigma = plugin.posterior.mean.to(dtype=self.dtype, device=self.device)
        self.n_sources = int(plugin.sources.n_sources)
        self._eye3 = torch.eye(3, dtype=self.dtype, device=self.device)

    def _operator_block(self, r: torch.Tensor) -> torch.Tensor:
        """``K(r)`` = the (3, n_sources) acceleration operator block at position ``r`` (3,)."""

        ker = acceleration_kernel(r.unsqueeze(0), self.src_positions, eps=self.eps, sign=self.accel_sign)[0]
        return (ker * self.src_weights.unsqueeze(-1)).transpose(0, 1)  # (3, n_sources)

    def _base_accel(self, r: torch.Tensor) -> torch.Tensor:
        if self.base_accel_fn is not None:
            return self.base_accel_fn(r.unsqueeze(0)).reshape(3).to(dtype=self.dtype, device=self.device)
        r2 = torch.dot(r, r)
        return -self.mu * r / (r2 * torch.sqrt(r2))

    def _gravity_gradient(self, r: torch.Tensor) -> torch.Tensor:
        """``G(r) = d a_base / d r`` (3, 3): analytic point-mass tide, else central difference."""

        if self.base_accel_fn is None:
            r2 = torch.dot(r, r)
            rn = torch.sqrt(r2)
            rhat = r / rn
            return -self.mu / (rn ** 3) * (self._eye3 - 3.0 * torch.outer(rhat, rhat))
        h = self.fd_eps
        cols = []
        for j in range(3):
            e = torch.zeros(3, dtype=self.dtype, device=self.device)
            e[j] = h
            cols.append((self._base_accel(r + e) - self._base_accel(r - e)) / (2.0 * h))
        return torch.stack(cols, dim=1)  # (3, 3)

    def _rhs(self, state: torch.Tensor, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r, v = state[:3], state[3:]
        K = self._operator_block(r)  # (3, m)
        a = self._base_accel(r) + K @ self.mean_sigma  # nominal = base + posterior-mean correction
        dstate = torch.cat([v, a])
        G = self._gravity_gradient(r)  # (3, 3)
        Jr, Jv = J[:3], J[3:]  # (3, m) each
        dJ = torch.cat([Jv, G @ Jr + K], dim=0)  # (6, m)
        return dstate, dJ

    def _rk4_step(self, state, J, h):
        k1s, k1J = self._rhs(state, J)
        k2s, k2J = self._rhs(state + 0.5 * h * k1s, J + 0.5 * h * k1J)
        k3s, k3J = self._rhs(state + 0.5 * h * k2s, J + 0.5 * h * k2J)
        k4s, k4J = self._rhs(state + h * k3s, J + h * k3J)
        state_next = state + (h / 6.0) * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
        J_next = J + (h / 6.0) * (k1J + 2.0 * k2J + 2.0 * k3J + k4J)
        return state_next, J_next

    def _covariance(self, J: torch.Tensor) -> torch.Tensor:
        """``P = J Sigma_sigma J^T`` (6, 6), symmetrized for numerical cleanliness."""

        P = J @ self.cov_sigma @ J.transpose(0, 1)
        return 0.5 * (P + P.transpose(0, 1))

    def propagate(self, y0, duration_s: float, output_dt_s: float) -> LinearCovarianceResult:
        """Propagate the nominal state and linearized ``6x6`` covariance.

        ``y0`` is a single initial state ``[r, v]`` (shape ``(6,)``). The covariance starts at zero
        (``J(0) = 0``) and grows as the force-error posterior is integrated along the trajectory.
        """

        y0 = np.asarray(y0, dtype=np.float64)
        if y0.shape != (6,):
            raise ValueError("y0 must be a 1D array of shape (6,)")

        snap_interval = float(output_dt_s)
        total_time = float(duration_s)
        steps_per_snap = max(1, round(snap_interval / self.dt))
        dt_eff = snap_interval / steps_per_snap
        n_snaps = max(1, round(total_time / snap_interval))

        times = np.linspace(0.0, n_snaps * snap_interval, n_snaps + 1, dtype=np.float64)
        states = np.empty((n_snaps + 1, 6), dtype=np.float64)
        covs = np.empty((n_snaps + 1, 6, 6), dtype=np.float64)

        state = torch.tensor(y0, dtype=self.dtype, device=self.device)
        J = torch.zeros(6, self.n_sources, dtype=self.dtype, device=self.device)
        states[0] = state.cpu().numpy()
        covs[0] = self._covariance(J).cpu().numpy()

        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                state, J = self._rk4_step(state, J, dt_eff)
            states[snap_idx + 1] = state.cpu().numpy()
            covs[snap_idx + 1] = self._covariance(J).cpu().numpy()

        pos_var = np.clip(np.einsum("tii->t", covs[:, :3, :3]), 0.0, None)
        vel_var = np.clip(np.einsum("tii->t", covs[:, 3:, 3:]), 0.0, None)
        return LinearCovarianceResult(
            times=times,
            states=states,
            covariances=covs,
            position_sigma=np.sqrt(pos_var),
            velocity_sigma=np.sqrt(vel_var),
        )
