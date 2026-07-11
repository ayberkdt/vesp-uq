"""A respected probabilistic UQ baseline for VESP-UQ: an exact Gaussian-process residual model.

The benchmark suite compares VESP-UQ against cheap *heuristic* selectors (altitude, kNN). A journal
reviewer will also ask the harder question: *why the equivalent-source posterior rather than a
standard spatial UQ method?* This module answers it with numbers, by fitting an independent-output
Gaussian process (RBF kernel, marginal-likelihood-selected noise) to the residual force-error vector
field and evaluating it on the **same** held-out calibration and trajectory-screening metrics as the
plugin (so the comparison is apples-to-apples).

The GP is deliberately exact (Cholesky), with a median-heuristic lengthscale and a per-component
noise chosen by evidence over a small grid -- a strong, well-understood baseline, not a tuned
competitor. The vanilla :class:`GPResidualUQ` carries no physics structure and no altitude
information; :class:`AltitudeAwareGPResidualUQ` (R2WP-6) closes the information gap -- altitude
input feature + the same post-hoc altitude noise law the plugin gets -- so that any claimed
VESP-UQ superiority is measured against an altitude-fair control, not an altitude-blind one.

No new dependency: implemented in torch (the repo ships neither scikit-learn nor gpytorch). Targets
force-model error only; never position error; no density claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vesp.extensions.probabilistic import AltitudeNoiseModel, calibration_metrics
from vesp.uq.metrics import component_calibration_metrics, vector_calibration_metrics

_NOISE_FRACTION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass
class GPPrediction:
    """Per-position GP output mirroring the fields the suite needs from the plugin."""

    positions: torch.Tensor       # (N, 3)
    radius: torch.Tensor          # (N,)
    mean_error: torch.Tensor      # (N, 3)
    std_components: torch.Tensor  # (N, 3) per-axis predictive std (latent + noise)
    sigma: torch.Tensor           # (N,) sqrt(sum of component variances)
    expected_error: torch.Tensor  # (N,) sqrt(||mean||^2 + sigma^2) -- RMS magnitude, not E||e||

    @property
    def rms_predictive_error(self) -> torch.Tensor:
        """Canonical name for :attr:`expected_error` (RMS predictive error magnitude)."""

        return self.expected_error


class GPResidualUQ:
    """Independent-output exact GP on the residual force-error vector field.

    Each acceleration-error component is modeled by its own zero-mean GP with a shared RBF
    lengthscale (median heuristic) and a per-component signal/noise variance. The predictive variance
    of an *observed* residual includes the noise term, so it is directly comparable to the plugin's
    predictive std on held-out errors.
    """

    def __init__(self, *, jitter: float = 1.0e-8, max_train: int = 1500,
                 dtype: torch.dtype = torch.float64, seed: int = 0) -> None:
        self.jitter = float(jitter)
        self.max_train = int(max_train)
        self.dtype = dtype
        self.seed = int(seed)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def _rbf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b) ** 2
        return torch.exp(-0.5 * d2 / (self.lengthscale_ ** 2))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel input features for positions ``x`` (identity here; subclass hook)."""

        return x

    def _predictive_variance(
        self, epistemic: torch.Tensor, noise: float, radius: torch.Tensor
    ) -> torch.Tensor:
        """Per-point predictive variance from the epistemic (latent) part (subclass hook)."""

        return epistemic + noise

    def fit(self, positions, error) -> GPResidualUQ:
        x = torch.as_tensor(positions, dtype=self.dtype)
        y = torch.as_tensor(error, dtype=self.dtype)
        if x.ndim != 2 or x.shape[-1] != 3 or y.shape != x.shape:
            raise ValueError("positions and error must both have shape (N, 3)")
        n = x.shape[0]
        if n > self.max_train:  # deterministic subsample for tractable O(M^3) Cholesky
            g = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)[: self.max_train]
            x, y = x[idx], y[idx]
        self.train_x_ = x
        self.train_f_ = self._features(x)
        m = x.shape[0]

        # median-heuristic lengthscale over pairwise feature distances (exclude the zero diagonal)
        with torch.no_grad():
            dists = torch.cdist(self.train_f_, self.train_f_)
            triu = dists[torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)]
            self.lengthscale_ = float(torch.median(triu).clamp_min(1.0e-6)) if triu.numel() else 1.0

        base = self._rbf(self.train_f_, self.train_f_)
        eye = torch.eye(m, dtype=self.dtype)
        self.signal_var_ = []
        self.noise_var_ = []
        self.alpha_ = []
        self.chol_ = []
        for c in range(3):
            yc = y[:, c]
            signal = float(yc.var().clamp_min(1.0e-30))
            best = None
            for frac in _NOISE_FRACTION_GRID:
                noise = max(signal * frac, 1.0e-30)
                k = signal * base + (noise + self.jitter) * eye
                try:
                    chol = torch.linalg.cholesky(k)
                except RuntimeError:
                    continue
                alpha = torch.cholesky_solve(yc.unsqueeze(-1), chol).squeeze(-1)
                # log marginal likelihood (evidence)
                logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
                lml = -0.5 * float(yc @ alpha) - 0.5 * float(logdet) - 0.5 * m * math.log(2.0 * math.pi)
                if best is None or lml > best[0]:
                    best = (lml, signal, noise, alpha, chol)
            if best is None:  # numerically hopeless -> isotropic fallback
                noise = signal
                chol = torch.linalg.cholesky(signal * base + (noise + 1.0e-6) * eye)
                alpha = torch.cholesky_solve(yc.unsqueeze(-1), chol).squeeze(-1)
                best = (0.0, signal, noise, alpha, chol)
            _, signal, noise, alpha, chol = best
            self.signal_var_.append(signal)
            self.noise_var_.append(noise)
            self.alpha_.append(alpha)
            self.chol_.append(chol)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, positions, *, chunk: int = 4096) -> GPPrediction:
        if not self._fitted:
            raise RuntimeError("GPResidualUQ.predict called before fit")
        x = torch.as_tensor(positions, dtype=self.dtype)
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("positions must have shape (Q, 3)")
        q = x.shape[0]
        radius = torch.linalg.norm(x, dim=-1)
        means = torch.empty(q, 3, dtype=self.dtype)
        stds = torch.empty(q, 3, dtype=self.dtype)
        for a in range(0, q, chunk):
            b = min(q, a + chunk)
            ks = self._rbf(self._features(x[a:b]), self.train_f_)  # (nb, m)
            for c in range(3):
                signal, noise = self.signal_var_[c], self.noise_var_[c]
                ks_c = signal * ks
                means[a:b, c] = ks_c @ self.alpha_[c]
                v = torch.linalg.solve_triangular(self.chol_[c], ks_c.transpose(0, 1), upper=False)
                epistemic = (signal - (v ** 2).sum(dim=0)).clamp_min(0.0)  # latent prior - reduction
                var = self._predictive_variance(epistemic, noise, radius[a:b])
                stds[a:b, c] = var.clamp_min(1.0e-30).sqrt()
        sigma = torch.sqrt((stds ** 2).sum(dim=-1))
        mean_mag = torch.linalg.norm(means, dim=-1)
        expected = torch.sqrt(mean_mag ** 2 + sigma ** 2)
        return GPPrediction(positions=x, radius=radius, mean_error=means, std_components=stds,
                            sigma=sigma, expected_error=expected)

    # ------------------------------------------------------------------ calibration (same metrics)
    def evaluate_calibration(self, positions, error, *, altitude_bands: dict | None = None) -> dict:
        """Per-band calibration on the SAME metric functions the plugin uses (apples-to-apples)."""

        x = torch.as_tensor(positions, dtype=self.dtype)
        e = torch.as_tensor(error, dtype=self.dtype)
        pred = self.predict(x)
        residual_vec = e - pred.mean_error
        cov = torch.diag_embed(pred.std_components ** 2)  # independent components -> diagonal cov
        radius = pred.radius
        row_radii = radius.repeat(3)
        mean_flat = torch.cat([pred.mean_error[:, 0], pred.mean_error[:, 1], pred.mean_error[:, 2]])
        std_flat = torch.cat([pred.std_components[:, 0], pred.std_components[:, 1], pred.std_components[:, 2]])
        target = torch.cat([e[:, 0], e[:, 1], e[:, 2]])
        bands = altitude_bands or {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}

        def _band(row_mask: torch.Tensor, point_mask: torch.Tensor) -> dict:
            m = calibration_metrics(mean_flat[row_mask], std_flat[row_mask], target[row_mask])
            m["mean_radius"] = float(torch.mean(row_radii[row_mask]).detach().cpu())
            if int(point_mask.sum()) >= 10:
                m.update(vector_calibration_metrics(residual_vec[point_mask], cov[point_mask]))
                m.update(component_calibration_metrics(
                    residual_vec[point_mask], cov[point_mask], x[point_mask]))
            return m

        report: dict = {"all": _band(torch.ones_like(row_radii, dtype=torch.bool),
                                     torch.ones_like(radius, dtype=torch.bool))}
        for name, rng in bands.items():
            if rng is None:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            row_mask = (row_radii >= lo) & (row_radii <= hi)
            point_mask = (radius >= lo) & (radius <= hi)
            if int(row_mask.sum()) >= 30:
                report[name] = _band(row_mask, point_mask)
        return report

    # ------------------------------------------------------------------ trajectory scoring
    def score_trajectories(self, trajectories, *, aggregator: str = "p95") -> torch.Tensor:
        """One expected-error risk score per trajectory (mirrors the plugin's expected-error mode)."""

        from vesp.uq.scoring import aggregate_trajectory_error

        trajs = [torch.as_tensor(t, dtype=self.dtype) for t in trajectories]
        if not trajs:
            return torch.empty(0, dtype=torch.float64)
        lengths = [t.shape[0] for t in trajs]
        allpts = torch.cat(trajs, dim=0)
        pred = self.predict(allpts)
        profile = pred.expected_error.to(torch.float64)
        out = torch.empty(len(trajs), dtype=torch.float64)
        off = 0
        for i, n in enumerate(lengths):
            out[i] = aggregate_trajectory_error(profile[off:off + n], aggregator)
            off += n
        return out


class AltitudeAwareGPResidualUQ(GPResidualUQ):
    """GP baseline given the SAME altitude information diet as VESP-UQ (R2WP-6).

    The vanilla :class:`GPResidualUQ` is deliberately altitude-blind (stationary RBF on Cartesian
    positions, homoscedastic per-component noise), while the plugin gets a heteroscedastic
    altitude noise law plus optional conformal scaling -- so "better calibration than the GP"
    partly measures information access, not method quality. This control closes that gap two ways:

    - **altitude input feature**: the kernel operates on standardized ``(x, y, z, log h)`` with
      ``h = r - 1``, so the stationary RBF can express altitude dependence;
    - **heteroscedastic altitude noise**: the same 2-parameter power law
      (:class:`~vesp.extensions.probabilistic.AltitudeNoiseModel`) the plugin uses, fit post-hoc
      on a held-in validation split's residuals -- the identical recalibration budget. At predict
      time it replaces the homoscedastic noise term: ``var = epistemic + sigma^2(h)``.

    Whatever calibration/decision superiority VESP-UQ claims over a GP must be measured against
    this variant, not (only) the altitude-blind one.
    """

    def __init__(self, *, val_fraction: float = 0.25, h_floor: float = 1.0e-3, **kw) -> None:
        super().__init__(**kw)
        self.val_fraction = float(val_fraction)
        self.h_floor = float(h_floor)
        self.altitude_noise_: AltitudeNoiseModel | None = None
        self._feat_mean_: torch.Tensor | None = None
        self._feat_scale_: torch.Tensor | None = None

    def _raw_features(self, x: torch.Tensor) -> torch.Tensor:
        h = (torch.linalg.norm(x, dim=-1) - 1.0).clamp_min(self.h_floor)
        return torch.cat([x, torch.log(h).unsqueeze(-1)], dim=-1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        f = self._raw_features(x)
        if self._feat_mean_ is None:  # standardization params come from the training positions
            return f
        return (f - self._feat_mean_) / self._feat_scale_

    def _predictive_variance(
        self, epistemic: torch.Tensor, noise: float, radius: torch.Tensor
    ) -> torch.Tensor:
        if self.altitude_noise_ is None:
            return epistemic + noise
        return epistemic + self.altitude_noise_.variance(radius)

    def fit(self, positions, error) -> AltitudeAwareGPResidualUQ:
        x = torch.as_tensor(positions, dtype=self.dtype)
        y = torch.as_tensor(error, dtype=self.dtype)
        if x.ndim != 2 or x.shape[-1] != 3 or y.shape != x.shape:
            raise ValueError("positions and error must both have shape (N, 3)")
        n = x.shape[0]

        # held-in validation split for the post-hoc altitude noise law (same budget as the plugin)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        n_val = int(round(self.val_fraction * n))
        use_val = n_val >= 30 and (n - n_val) >= 30
        val_idx, tr_idx = (perm[:n_val], perm[n_val:]) if use_val else (perm[:0], perm)

        raw = self._raw_features(x[tr_idx])
        self._feat_mean_ = raw.mean(dim=0)
        self._feat_scale_ = raw.std(dim=0).clamp_min(1.0e-12)

        self.altitude_noise_ = None  # base fit + val prediction run with homoscedastic noise
        super().fit(x[tr_idx], y[tr_idx])

        if use_val:
            pred = self.predict(x[val_idx])
            residual = y[val_idx] - pred.mean_error  # (n_val, 3)
            noise = torch.tensor(self.noise_var_, dtype=self.dtype)  # (3,)
            epistemic = (pred.std_components**2 - noise).clamp_min(0.0)  # (n_val, 3)
            radii3 = pred.radius.repeat(3)
            self.altitude_noise_ = AltitudeNoiseModel.fit(
                radii3,
                residual.transpose(0, 1).reshape(-1),
                epistemic.transpose(0, 1).reshape(-1),
                h_floor=self.h_floor,
            )
        return self


