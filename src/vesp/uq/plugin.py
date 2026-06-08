"""VESP-UQ: a surrogate-agnostic equivalent-source uncertainty calibration layer (Phase 2).

VESP-UQ is *not* a residual-gravity surrogate. It is an uncertainty layer that wraps any
model with the interface ``x -> residual acceleration`` and answers a different question:
*where should that surrogate be trusted?* It samples the surrogate's error against a
higher-fidelity reference,

    e_a(x) = a_reference(x) - a_surrogate(x),

and fits a physics-consistent equivalent-source error model ``e_a(x) ~ A(x) sigma`` whose
sources live strictly inside the Moon. Because the model is linear in ``sigma``, the
Tikhonov/ridge solution has an exact linear-Gaussian posterior (see
:class:`~vesp.extensions.probabilistic.LinearGaussianPosterior`), turning the deterministic
error fit into calibrated, altitude-aware predictive uncertainty over the force-error field.

Pipeline (matching the VESP-UQ plan):

    fit(positions, surrogate_acc, reference_acc)   # Steps 1-2, 4-5
    predict_uncertainty(positions) -> mean error, std, per-point risk    # Step 6
    score_trajectory(positions_over_time) -> TrajectoryScore             # Steps 6-7

The posterior MEAN equals the ridge point estimate, so this never claims to improve
deterministic accuracy (the entropy/point-estimate story is kept only as an ablation). Its
value is the *error bars* and the trajectory risk screen they enable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vesp.core.diagnostics import source_diagnostics
from vesp.core.operators import build_acceleration_operator
from vesp.core.regularization import lcurve_lambda
from vesp.core.sources import SourceSet, make_shell_sources
from vesp.extensions.probabilistic import (
    AltitudeNoiseModel,
    LinearGaussianPosterior,
    calibration_metrics,
)
from vesp.uq.metrics import vector_calibration_metrics
from vesp.uq.trajectory import TrajectoryScore, score_sigma_profile

COVARIANCE_MODES = ("exact", "diagonal", "lowrank")


@dataclass
class UncertaintyPrediction:
    """Per-position output of :meth:`VESPUQPlugin.predict_uncertainty`.

    All tensors are indexed by query position. ``sigma`` is the scalar predictive std of the
    error-vector magnitude scale (``sqrt`` of the summed per-component variance) and is the
    default per-position ``risk_score``.

    ``mean_error_magnitude`` is the norm of the posterior-MEAN error vector (the surrogate's
    expected bias at that point) and ``expected_error`` combines it with the predictive spread,
    ``expected_error = sqrt(mean_error_magnitude^2 + sigma^2)`` -- a single point estimate of
    "how wrong is the surrogate expected to be here", used by the stronger trajectory-scoring
    modes. ``epistemic_fraction`` is ``epistemic_sigma / sigma`` (share of the spread that is
    reducible source-posterior uncertainty rather than the aleatoric floor).
    """

    positions: torch.Tensor  # (N, 3)
    radius: torch.Tensor  # (N,)
    mean_error: torch.Tensor  # (N, 3) predicted mean residual-force error vector
    std_components: torch.Tensor  # (N, 3) per-component predictive std
    sigma: torch.Tensor  # (N,) total predictive std
    epistemic_sigma: torch.Tensor  # (N,) epistemic-only (source-posterior) std
    mean_error_magnitude: torch.Tensor  # (N,) ||mean_error|| (posterior-mean residual magnitude)
    expected_error: torch.Tensor  # (N,) sqrt(mean_error_magnitude^2 + sigma^2)
    epistemic_fraction: torch.Tensor  # (N,) epistemic_sigma / sigma in [0, 1]
    risk_score: torch.Tensor  # (N,)

    def to_numpy(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in asdict(self).items()}


@dataclass
class CovariancePrediction:
    """Per-position 3x3 predictive covariance output of :meth:`VESPUQPlugin.predict_covariance_3x3`."""

    positions: torch.Tensor  # (N, 3)
    mean_error: torch.Tensor  # (N, 3)
    covariance: torch.Tensor  # (N, 3, 3) symmetric PSD predictive covariance
    std_components: torch.Tensor  # (N, 3)
    sigma: torch.Tensor  # (N,)

    def to_numpy(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in asdict(self).items()}


def _flatten_acc(acc: torch.Tensor) -> torch.Tensor:
    """(N, 3) acceleration -> (3N,) in the [x-block, y-block, z-block] row order of the operator."""

    return torch.cat([acc[:, 0], acc[:, 1], acc[:, 2]])


class VESPUQPlugin:
    """Equivalent-source uncertainty calibration layer for residual-gravity surrogates."""

    def __init__(
        self,
        sources: SourceSet,
        *,
        eps: float = 0.0,
        acceleration_sign: float = 1.0,
        source_chunk_size: int | None = 1024,
        reg_method: str = "lcurve",
        lambda_l2: float = 30.0,
        noise_model: str = "heteroscedastic",
        covariance_mode: str = "exact",
        lowrank_rank: int = 64,
        val_fraction: float = 0.25,
        low_altitude_radius: float = 1.15,
        risk_scoring: str = "max",
        sigma_threshold: float | None = None,
        domain_support: bool = False,
        domain_k: int = 8,
        domain_weight: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if reg_method not in {"lcurve", "evidence", "fixed"}:
            raise ValueError("reg_method must be 'lcurve', 'evidence', or 'fixed'")
        if noise_model not in {"homoscedastic", "heteroscedastic"}:
            raise ValueError("noise_model must be 'homoscedastic' or 'heteroscedastic'")
        if covariance_mode not in COVARIANCE_MODES:
            raise ValueError(f"covariance_mode must be one of {COVARIANCE_MODES}")
        self.covariance_mode = covariance_mode
        self.lowrank_rank = int(lowrank_rank)
        self._cov_eig: tuple[torch.Tensor, torch.Tensor] | None = None
        self.dtype = dtype
        self.device = torch.device(device)
        self.sources = sources.to(self.device)
        self.eps = float(eps)
        self.acceleration_sign = float(acceleration_sign)
        self.source_chunk_size = source_chunk_size
        self.reg_method = reg_method
        self.lambda_l2 = float(lambda_l2)
        self.noise_model = noise_model
        self.val_fraction = float(val_fraction)
        self.low_altitude_radius = float(low_altitude_radius)
        self.risk_scoring = risk_scoring
        self.sigma_threshold = sigma_threshold
        self.domain_support = bool(domain_support)
        self.domain_k = int(domain_k)
        self.domain_weight = float(domain_weight)
        self.seed = int(seed)

        self.posterior: LinearGaussianPosterior | None = None
        self.altitude_noise: AltitudeNoiseModel | None = None
        self.fit_info: dict = {}

        # domain-support state (populated by fit_error; used by domain_support_score)
        self.train_positions: torch.Tensor | None = None
        self.train_radii: torch.Tensor | None = None
        self.val_positions: torch.Tensor | None = None
        self.val_radii: torch.Tensor | None = None
        self._domain_scale: float | None = None
        self._domain_scale_k: int | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: dict) -> "VESPUQPlugin":
        """Build a plugin from a config dict (reuses the ``model``/``kernel`` conventions)."""

        dtype = torch.float64 if str(config.get("dtype", "float64")).lower() in {"float64", "double"} else torch.float32
        device = torch.device(config.get("device", "cpu"))
        model = config.get("model", {})
        if model.get("type") == "multishell":
            alphas = [float(a) for a in model["shell_alphas"]]
            counts = model["n_sources_per_shell"]
        else:
            alphas = [float(model.get("shell_alpha", 0.86))]
            counts = int(model.get("n_source", 512))
        sources = make_shell_sources(
            alphas,
            counts,
            weight_mode=str(model.get("weight_mode", "surface_area")),
            dtype=dtype,
            device=device,
        )
        kernel = config.get("kernel", {})
        uq = config.get("uq", config.get("uncertainty", {}))
        reg = uq.get("regularization", {})
        reg_method = str(reg.get("method", uq.get("reg_method", "lcurve"))).lower()
        # accept a numeric lambda either as the fixed value or as the seed for other methods
        lam_raw = reg.get("lambda_l2", config.get("solver", {}).get("lambda_l2", 30.0))
        try:
            lambda_l2 = float(lam_raw)
        except (TypeError, ValueError):
            lambda_l2 = 30.0
            if reg_method == "fixed":
                reg_method = "lcurve"
        risk = uq.get("risk", {})
        bands = config.get("evaluation", {}).get("altitude_bands", {}) or {}
        low_band = bands.get("low") or [1.03, 1.15]
        return cls(
            sources,
            eps=float(kernel.get("eps", kernel.get("softening", 0.0))),
            acceleration_sign=float(kernel.get("acceleration_sign", 1.0)),
            source_chunk_size=kernel.get("source_chunk_size", 1024),
            reg_method=reg_method,
            lambda_l2=lambda_l2,
            noise_model=str(uq.get("noise_model", "heteroscedastic")).lower(),
            covariance_mode=str(uq.get("covariance_mode", "exact")).lower(),
            lowrank_rank=int(uq.get("lowrank_rank", 64)),
            val_fraction=float(uq.get("val_fraction", 0.25)),
            low_altitude_radius=float(risk.get("low_altitude_radius", low_band[1])),
            risk_scoring=str(risk.get("scoring", "max")).lower(),
            sigma_threshold=risk.get("sigma_threshold"),
            domain_support=bool(risk.get("domain_support", False)),
            domain_k=int(risk.get("domain_k", 8)),
            domain_weight=float(risk.get("domain_weight", 1.0)),
            dtype=dtype,
            device=device,
            seed=int(config.get("seed", 0)),
        )

    # ------------------------------------------------------------------ internals
    def _prep_positions(self, positions) -> torch.Tensor:
        x = torch.as_tensor(positions, dtype=self.dtype, device=self.device)
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        return x

    def _operator(self, positions: torch.Tensor) -> torch.Tensor:
        return build_acceleration_operator(
            positions,
            self.sources,
            eps=self.eps,
            sign=self.acceleration_sign,
            source_chunk_size=self.source_chunk_size,
        )

    def _require_fitted(self) -> None:
        if self.posterior is None:
            raise RuntimeError("VESPUQPlugin is not fitted; call fit(...) first")

    def _point_noise(self, radii: torch.Tensor) -> torch.Tensor | float:
        """Aleatoric noise variance per row/point: global floor + altitude excess if het."""

        if self.altitude_noise is None:
            return self.posterior.noise_var
        return self.posterior.noise_var + self.altitude_noise.variance(radii)

    def _cov_eigpairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-``lowrank_rank`` eigenpairs of the posterior covariance (cached after fit)."""

        if self._cov_eig is None:
            vals, vecs = torch.linalg.eigh(self.posterior.cov)  # ascending
            k = min(self.lowrank_rank, int(vals.numel()))
            self._cov_eig = (vals[-k:].clamp_min(0.0), vecs[:, -k:])
        return self._cov_eig

    def _epistemic_variance(self, operator: torch.Tensor) -> torch.Tensor:
        """Per-row epistemic (source-posterior) variance, honoring ``covariance_mode``.

        ``exact`` uses the full covariance; ``diagonal`` keeps only its diagonal (drops source
        correlations -> O(m*n) instead of O(m*n^2)); ``lowrank`` uses the top-k eigenpairs.
        """

        if self.covariance_mode == "diagonal":
            diag = torch.diagonal(self.posterior.cov)
            return ((operator * operator) @ diag).clamp_min(0.0)
        if self.covariance_mode == "lowrank":
            vals, vecs = self._cov_eigpairs()
            proj = operator @ vecs
            return ((proj * proj) @ vals).clamp_min(0.0)
        cov_q = operator @ self.posterior.cov
        return torch.sum(cov_q * operator, dim=-1).clamp_min(0.0)

    def _predict_rows(self, operator: torch.Tensor, radii: torch.Tensor) -> dict[str, torch.Tensor]:
        """Row-level (3N) predictive mean/variance honoring noise model + covariance mode."""

        mean = operator @ self.posterior.mean
        epistemic = self._epistemic_variance(operator)
        variance = epistemic + self._point_noise(radii)
        return {
            "mean": mean,
            "epistemic_variance": epistemic,
            "variance": variance,
            "std": torch.sqrt(variance.clamp_min(torch.finfo(mean.dtype).tiny)),
        }

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        positions,
        surrogate_acceleration,
        reference_acceleration,
        *,
        val_positions=None,
        val_surrogate_acceleration=None,
        val_reference_acceleration=None,
    ) -> "VESPUQPlugin":
        """Fit the equivalent-source error posterior from surrogate/reference acceleration samples.

        Computes ``error = reference - surrogate`` and delegates to :meth:`fit_error`.
        """

        positions = self._prep_positions(positions)
        error = self._prep_positions(reference_acceleration) - self._prep_positions(surrogate_acceleration)
        val_error = None
        if val_positions is not None:
            if val_reference_acceleration is None or val_surrogate_acceleration is None:
                raise ValueError(
                    "val_positions requires both val_reference_acceleration and "
                    "val_surrogate_acceleration (or use fit_error with an explicit val_error)"
                )
            val_positions = self._prep_positions(val_positions)
            val_error = self._prep_positions(val_reference_acceleration) - self._prep_positions(
                val_surrogate_acceleration
            )
        return self.fit_error(positions, error, val_positions=val_positions, val_error=val_error)

    def fit_error(self, positions, error, *, val_positions=None, val_error=None) -> "VESPUQPlugin":
        """Fit directly from sampled force-error vectors ``error = a_reference - a_surrogate``.

        The Tikhonov weight is selected automatically (L-curve corner by default). The posterior
        mean is the ridge solution; the global noise floor and the altitude-dependent excess
        noise are calibrated on a HELD-OUT validation split (an internal random split unless an
        explicit ``val_positions``/``val_error`` is supplied), because training residuals are
        optimistic and underestimate the altitude-dependent generalization error.
        """

        positions = self._prep_positions(positions)
        error = self._prep_positions(error)

        if val_positions is None:
            generator = torch.Generator().manual_seed(self.seed)
            n = positions.shape[0]
            perm = torch.randperm(n, generator=generator)
            n_val = max(1, int(round(self.val_fraction * n))) if n > 1 else 0
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            train_pos, train_err = positions[train_idx], error[train_idx]
            val_pos = positions[val_idx] if n_val > 0 else None
            val_err = error[val_idx] if n_val > 0 else None
        else:
            train_pos, train_err = positions, error
            val_pos = self._prep_positions(val_positions)
            val_err = self._prep_positions(val_error)

        # Stash the calibration-support geometry for domain_support_score(). These are cheap
        # references; the (potentially expensive) nearest-neighbour scale is computed lazily.
        self.train_positions = train_pos.detach()
        self.train_radii = torch.linalg.norm(train_pos, dim=-1).detach()
        if val_pos is not None:
            self.val_positions = val_pos.detach()
            self.val_radii = torch.linalg.norm(val_pos, dim=-1).detach()
        else:
            self.val_positions = None
            self.val_radii = None
        self._domain_scale = None  # invalidate any cached nearest-neighbour scale
        self._domain_scale_k = None

        operator = self._operator(train_pos)
        target = _flatten_acc(train_err)

        # --- Step 2: equivalent-source ridge fit with automatic regularization ---
        lcurve_points: list[dict] | None = None
        if self.reg_method == "evidence":
            posterior = LinearGaussianPosterior.fit_evidence(operator, target)
            lambda_used = posterior.lambda_l2
        else:
            if self.reg_method == "lcurve":
                lambda_used, lcurve_points = lcurve_lambda(operator, target)
            else:  # fixed
                lambda_used = self.lambda_l2
            # noise floor from HELD-OUT residuals (honest), falling back to the training fit
            noise_var = None
            if val_pos is not None:
                tmp = LinearGaussianPosterior.fit(operator, target, lambda_l2=lambda_used)
                val_resid = tmp.predict(self._operator(val_pos), include_noise=False)["mean"] - _flatten_acc(val_err)
                noise_var = float(torch.mean(val_resid * val_resid).detach().cpu())
            posterior = LinearGaussianPosterior.fit(
                operator, target, lambda_l2=lambda_used, noise_var=noise_var
            )
        self.posterior = posterior
        self._cov_eig = None  # invalidate the cached low-rank eigendecomposition

        # --- Step 5: altitude-dependent heteroscedastic recalibration on held-out residuals ---
        self.altitude_noise = None
        if self.noise_model == "heteroscedastic" and val_pos is not None:
            val_op = self._operator(val_pos)
            val_pred = posterior.predict(val_op, include_noise=False)
            val_resid = val_pred["mean"] - _flatten_acc(val_err)
            val_row_radii = torch.linalg.norm(val_pos, dim=-1).repeat(3)
            self.altitude_noise = AltitudeNoiseModel.fit(
                val_row_radii, val_resid, val_pred["epistemic_variance"] + posterior.noise_var
            )

        self.fit_info = {
            "n_train": int(train_pos.shape[0]),
            "n_val": int(val_pos.shape[0]) if val_pos is not None else 0,
            "reg_method": self.reg_method,
            "lambda_l2": float(lambda_used) if lambda_used is not None else None,
            "noise_var": posterior.noise_var,
            "noise_std": float(posterior.noise_var ** 0.5),
            "noise_model": self.noise_model,
            "covariance_mode": self.covariance_mode,
            "n_sources": int(self.sources.n_sources),
        }
        if self.altitude_noise is not None:
            self.fit_info["altitude_noise_a"] = self.altitude_noise.a
            self.fit_info["altitude_noise_b"] = self.altitude_noise.b
        if lcurve_points is not None:
            self.fit_info["lcurve"] = lcurve_points
        return self

    # ------------------------------------------------------------------ prediction
    def predict_uncertainty(self, positions) -> UncertaintyPrediction:
        """Predict the mean force-error and calibrated per-position predictive uncertainty."""

        self._require_fitted()
        positions = self._prep_positions(positions)
        n = positions.shape[0]
        op = self._operator(positions)
        radius = torch.linalg.norm(positions, dim=-1)
        pred = self._predict_rows(op, radius.repeat(3))

        # operator rows are [x-block, y-block, z-block]; reshape(3, N).T -> (N, 3)
        mean3 = pred["mean"].reshape(3, n).transpose(0, 1)
        var3 = pred["variance"].reshape(3, n).transpose(0, 1)
        epi3 = pred["epistemic_variance"].reshape(3, n).transpose(0, 1)
        std3 = torch.sqrt(var3.clamp_min(0.0))
        sigma = torch.sqrt(var3.sum(dim=1).clamp_min(0.0))
        epistemic_sigma = torch.sqrt(epi3.sum(dim=1).clamp_min(0.0))

        # Posterior-mean residual magnitude (expected surrogate bias) and the combined
        # expected-error point estimate sqrt(bias^2 + spread^2). These feed the stronger
        # trajectory-scoring modes; the posterior mean is still the ridge point estimate, so
        # this never claims to improve deterministic accuracy -- it only summarizes it.
        mean_error_magnitude = torch.sqrt((mean3 * mean3).sum(dim=1).clamp_min(0.0))
        expected_error = torch.sqrt((mean_error_magnitude * mean_error_magnitude + sigma * sigma).clamp_min(0.0))
        epistemic_fraction = epistemic_sigma / sigma.clamp_min(torch.finfo(sigma.dtype).tiny)
        return UncertaintyPrediction(
            positions=positions,
            radius=radius,
            mean_error=mean3,
            std_components=std3,
            sigma=sigma,
            epistemic_sigma=epistemic_sigma,
            mean_error_magnitude=mean_error_magnitude,
            expected_error=expected_error,
            epistemic_fraction=epistemic_fraction,
            risk_score=sigma,
        )

    def predict_covariance_3x3(self, positions) -> CovariancePrediction:
        """Full ``3x3`` predictive covariance of the acceleration-error vector at each position.

        For a query point with operator rows ``Q_i`` (3, n_sources),
        ``Cov_a(x_i) = Q_i Sigma_sigma Q_i^T + noise_i I_3`` -- a symmetric PSD matrix combining
        the source-posterior (epistemic) covariance and the aleatoric noise floor. ``diagonal``
        mode returns diagonal covariances (off-diagonal source correlations dropped); ``exact``
        and ``lowrank`` return the full (or low-rank-approximated) ``3x3``.
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        n = positions.shape[0]
        op = self._operator(positions)
        opx, opy, opz = op[:n], op[n : 2 * n], op[2 * n :]
        radius = torch.linalg.norm(positions, dim=-1)

        zeros = torch.zeros(n, dtype=self.dtype, device=self.device)
        if self.covariance_mode == "diagonal":
            diag = torch.diagonal(self.posterior.cov)
            cxx = ((opx * opx) @ diag).clamp_min(0.0)
            cyy = ((opy * opy) @ diag).clamp_min(0.0)
            czz = ((opz * opz) @ diag).clamp_min(0.0)
            cxy = cxz = cyz = zeros
        else:
            if self.covariance_mode == "lowrank":
                vals, vecs = self._cov_eigpairs()
                tx, ty, tz = opx @ vecs, opy @ vecs, opz @ vecs  # transformed blocks (N, k)

                def _dot(a, b):
                    return (a * b) @ vals

            else:  # exact
                tx, ty, tz = opx @ self.posterior.cov, opy @ self.posterior.cov, opz @ self.posterior.cov

                def _dot(a, b):
                    # a is (N,n) already multiplied by cov; b is the raw operator block (N,n)
                    return torch.sum(a * b, dim=-1)

            if self.covariance_mode == "lowrank":
                cxx = _dot(tx, tx).clamp_min(0.0)
                cyy = _dot(ty, ty).clamp_min(0.0)
                czz = _dot(tz, tz).clamp_min(0.0)
                cxy, cxz, cyz = _dot(tx, ty), _dot(tx, tz), _dot(ty, tz)
            else:
                cxx = _dot(tx, opx).clamp_min(0.0)
                cyy = _dot(ty, opy).clamp_min(0.0)
                czz = _dot(tz, opz).clamp_min(0.0)
                cxy, cxz, cyz = _dot(tx, opy), _dot(tx, opz), _dot(ty, opz)

        noise = self._point_noise(radius)
        if not torch.is_tensor(noise):
            noise = torch.full((n,), float(noise), dtype=self.dtype, device=self.device)
        cov = torch.zeros(n, 3, 3, dtype=self.dtype, device=self.device)
        cov[:, 0, 0] = cxx + noise
        cov[:, 1, 1] = cyy + noise
        cov[:, 2, 2] = czz + noise
        cov[:, 0, 1] = cov[:, 1, 0] = cxy
        cov[:, 0, 2] = cov[:, 2, 0] = cxz
        cov[:, 1, 2] = cov[:, 2, 1] = cyz

        mean3 = (op @ self.posterior.mean).reshape(3, n).transpose(0, 1)
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (N, 3)
        std_components = torch.sqrt(diag.clamp_min(0.0))
        sigma = torch.sqrt(diag.sum(dim=1).clamp_min(0.0))
        return CovariancePrediction(
            positions=positions,
            mean_error=mean3,
            covariance=cov,
            std_components=std_components,
            sigma=sigma,
        )

    # ------------------------------------------------------------------ domain support
    def _domain_nn_scale(self, k: int, *, subset: int = 512) -> float:
        """Robust training-set length scale: median ``k``-th nearest-neighbour distance.

        Computed over a random subset of training points (excluding self), cached per ``k``.
        Used to normalize :meth:`domain_support_score` so that a query sitting at the typical
        training spacing scores ~0 and points farther out score progressively higher.
        """

        if self.train_positions is None:
            raise RuntimeError("domain support needs a fit; train positions are not stored")
        if self._domain_scale is not None and self._domain_scale_k == k:
            return self._domain_scale
        train = self.train_positions
        n = int(train.shape[0])
        if n < 2:
            scale = 1.0
        else:
            g = torch.Generator().manual_seed(self.seed)
            sub_idx = torch.randperm(n, generator=g)[: min(subset, n)]
            sub = train[sub_idx]
            kk = min(k + 1, n)  # +1 to drop the self-distance (0)
            knn = torch.empty(sub.shape[0], dtype=self.dtype, device=self.device)
            for start in range(0, sub.shape[0], 1024):
                d = torch.cdist(sub[start : start + 1024], train)
                knn[start : start + 1024] = torch.topk(d, kk, largest=False).values[:, -1]
            scale = float(torch.median(knn))
        scale = max(scale, 1.0e-12)
        self._domain_scale, self._domain_scale_k = scale, k
        return scale

    def domain_support_score(self, positions, k: int | None = None, *, chunk: int = 1024) -> torch.Tensor:
        """Per-position domain-support (out-of-support) score for query ``positions`` ``(N, 3)``.

        Returns a nonnegative tensor where ``0`` means well inside the calibration support and
        values ``> 1`` mean increasingly extrapolated. It combines two robust signals:

        - a **distance** term ``(d_k / scale - 1)+``, where ``d_k`` is the query's ``k``-th
          nearest training-point distance and ``scale`` is the median training ``k``-NN spacing;
        - a **radius extrapolation** term that grows once the query radius falls below the
          minimum or rises above the maximum training radius (normalized by the same scale).

        Chunked over queries so it stays cheap even for large ensembles.
        """

        if self.train_positions is None:
            raise RuntimeError("domain support needs a fit; call fit(...)/fit_error(...) first")
        k = int(self.domain_k if k is None else k)
        pos = self._prep_positions(positions)
        train = self.train_positions
        n_train = int(train.shape[0])
        k_eff = max(1, min(k, n_train))
        scale = self._domain_nn_scale(k)

        knn = torch.empty(pos.shape[0], dtype=self.dtype, device=self.device)
        for start in range(0, pos.shape[0], chunk):
            d = torch.cdist(pos[start : start + chunk], train)
            knn[start : start + chunk] = torch.topk(d, k_eff, largest=False).values[:, -1]
        dist_score = (knn / scale - 1.0).clamp_min(0.0)

        radius = torch.linalg.norm(pos, dim=-1)
        r_min = float(self.train_radii.min())
        r_max = float(self.train_radii.max())
        below = (r_min - radius).clamp_min(0.0)
        above = (radius - r_max).clamp_min(0.0)
        radius_penalty = (below + above) / scale
        return (dist_score + radius_penalty).clamp_min(0.0)

    # ------------------------------------------------------------------ trajectory scoring
    def score_trajectory(
        self, positions_over_time, *, scoring: str | None = None, weights=None
    ) -> TrajectoryScore:
        """Score one trajectory (``(T, 3)`` output positions) into a :class:`TrajectoryScore`.

        ``weights`` (optional, one per output point) lets callers down-weight oversampled
        regions (e.g. periapsis for true-anomaly-uniform orbits); ``None`` keeps the uniform
        time assumption. Domain-support point risk is included only when ``domain_support`` was
        enabled on the plugin.
        """

        pred = self.predict_uncertainty(positions_over_time)
        domain_risk = self.domain_support_score(pred.positions) if self.domain_support else None
        return score_sigma_profile(
            pred.sigma,
            pred.radius,
            scoring=scoring or self.risk_scoring,
            sigma_threshold=self.sigma_threshold,
            low_altitude_radius=self.low_altitude_radius,
            epistemic_sigma=pred.epistemic_sigma,
            expected_error=pred.expected_error,
            mean_error_magnitude=pred.mean_error_magnitude,
            domain_risk=domain_risk,
            domain_weight=self.domain_weight,
            weights=weights,
        )

    def score_ensemble(
        self, trajectories, *, scoring: str | None = None, weights=None
    ) -> list[TrajectoryScore]:
        """Score an iterable of trajectories (each ``(T_i, 3)``).

        ``weights`` is either ``None`` (uniform time weighting for every trajectory) or an
        iterable of per-trajectory weight vectors aligned with ``trajectories`` (entries may be
        ``None`` to keep a given trajectory uniform).
        """

        traj_list = list(trajectories)
        if weights is None:
            return [self.score_trajectory(t, scoring=scoring) for t in traj_list]
        weight_list = list(weights)
        if len(weight_list) != len(traj_list):
            raise ValueError("weights must be None or one weight vector per trajectory")
        return [
            self.score_trajectory(t, scoring=scoring, weights=w)
            for t, w in zip(traj_list, weight_list)
        ]

    # ------------------------------------------------------------------ calibration report
    def evaluate_calibration(self, positions, error, *, altitude_bands: dict | None = None) -> dict:
        """Per-band calibration metrics (PICP, z_std, NLL, CRPS) for held-out error samples.

        This is Experiment 1: does the layer's nominal interval cover the held-out residuals,
        and does its uncertainty grow toward low altitude where the surrogate is overconfident?
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        op = self._operator(positions)
        radius = torch.linalg.norm(positions, dim=-1)
        row_radii = radius.repeat(3)
        pred = self._predict_rows(op, row_radii)
        mean, std = pred["mean"], pred["std"]
        epistemic_std = torch.sqrt(pred["epistemic_variance"].clamp_min(0.0))
        target = _flatten_acc(error)

        # vector (ellipsoid) calibration uses the full 3x3 predictive covariance per point and
        # the predictive RESIDUAL (observed error minus the posterior-mean error prediction).
        cov_pred = self.predict_covariance_3x3(positions)
        residual_vec = error - cov_pred.mean_error
        point_radius = radius
        point_mask_all = torch.ones_like(point_radius, dtype=torch.bool)

        bands = altitude_bands or {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}

        def _band(row_mask: torch.Tensor, point_mask: torch.Tensor) -> dict:
            m = calibration_metrics(mean[row_mask], std[row_mask], target[row_mask])
            m["mean_epistemic_std"] = float(torch.mean(epistemic_std[row_mask]).detach().cpu())
            m["mean_pred_sigma"] = float(
                torch.mean(std[row_mask]).detach().cpu()
            )
            m["mean_radius"] = float(torch.mean(row_radii[row_mask]).detach().cpu())
            if int(point_mask.sum()) >= 10:
                m.update(
                    vector_calibration_metrics(residual_vec[point_mask], cov_pred.covariance[point_mask])
                )
            return m

        report: dict = {"all": _band(torch.ones_like(row_radii, dtype=torch.bool), point_mask_all)}
        for name, rng in bands.items():
            if rng is None:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            row_mask = (row_radii >= lo) & (row_radii <= hi)
            point_mask = (point_radius >= lo) & (point_radius <= hi)
            if int(row_mask.sum()) >= 30:
                report[name] = _band(row_mask, point_mask)
        low, high = report.get("low"), report.get("high")
        if low and high and high.get("mean_epistemic_std"):
            report["low_high_epistemic_std_ratio"] = low["mean_epistemic_std"] / max(
                high["mean_epistemic_std"], 1.0e-30
            )
            report["low_high_pred_sigma_ratio"] = low["mean_pred_sigma"] / max(
                high["mean_pred_sigma"], 1.0e-30
            )
        return report

    # ------------------------------------------------------------------ diagnostics
    def source_health(self) -> dict:
        """Step 3 source-health diagnostics on the fitted posterior mean (sigma)."""

        self._require_fitted()
        return source_diagnostics(
            source_positions=self.sources.positions,
            source_weights=self.sources.weights,
            shell_ids=self.sources.shell_ids,
            sigma=self.posterior.mean,
        )
