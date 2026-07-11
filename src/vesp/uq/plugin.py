"""VESP-UQ: a surrogate-interface-agnostic equivalent-source uncertainty calibration layer.

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
from pathlib import Path

import torch

from vesp.common.artifacts import atomic_torch_save, json_safe
from vesp.core.diagnostics import source_diagnostics
from vesp.core.operators import build_acceleration_operator
from vesp.core.regularization import lcurve_lambda
from vesp.core.sources import SourceSet, make_shell_sources
from vesp.extensions.probabilistic import (
    AltitudeNoiseModel,
    BinnedAltitudeNoiseModel,
    LinearGaussianPosterior,
    _safe_cholesky,
    calibration_metrics,
)
from vesp.uq.conformal import fit_conformal_scale
from vesp.uq.domain_support import (
    domain_support_components as _domain_support_components,
)
from vesp.uq.domain_support import (
    make_domain_backend,
    median_angular_scale,
    median_knn_scale,
)
from vesp.uq.metrics import (
    component_calibration_metrics,
    mahalanobis_squared,
    vector_calibration_metrics,
)
from vesp.uq.ranking import average_ranks
from vesp.uq.scoring import (
    TrajectoryScore,
    needs_covariance,
    score_sigma_profile,
)
from vesp.uq.scoring import (
    calibrate_risk_threshold as _calibrate_risk_threshold,
)

COVARIANCE_MODES = ("exact", "diagonal", "lowrank")
NOISE_MODELS = ("homoscedastic", "heteroscedastic", "altitude_binned")
PREDICTIVE_CONFORMAL_MODES = ("norm", "component_max", "mahalanobis")

# Versioned on-disk format for a fitted plugin (see VESPUQPlugin.save / .load).
PLUGIN_STATE_FORMAT = "vesp.uq.plugin"
PLUGIN_STATE_VERSION = 3


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
        query_chunk_size: int | None = 8192,
        reg_method: str = "lcurve",
        lambda_l2: float = 30.0,
        noise_model: str = "heteroscedastic",
        altitude_noise_bins: int = 8,
        covariance_mode: str = "exact",
        lowrank_rank: int = 64,
        val_fraction: float = 0.25,
        conformal_apply: bool = False,
        conformal_alpha: float = 0.10,
        conformal_mode: str = "norm",
        conformal_by_band: bool = False,
        conformal_bands: dict | None = None,
        conformal_min_band_n: int = 30,
        conformal_by_region: bool = False,
        conformal_min_region_n: int = 30,
        low_altitude_radius: float = 1.15,
        risk_scoring: str = "max",
        sigma_threshold: float | None = None,
        altitude_reference_h: float | None = None,
        calibrate_supervisor: bool = False,
        calibrated_supervisor_grid: dict | None = None,
        domain_support: bool = False,
        domain_k: int = 8,
        domain_weight: float = 1.0,
        domain_distance_weight: float = 1.0,
        domain_radial_weight: float = 1.0,
        domain_angular_weight: float = 0.0,
        domain_backend: str = "torch",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if reg_method not in {"lcurve", "evidence", "fixed"}:
            raise ValueError("reg_method must be 'lcurve', 'evidence', or 'fixed'")
        noise_model = str(noise_model).lower()
        if noise_model in {"binned", "altitude_bins", "binned_heteroscedastic"}:
            noise_model = "altitude_binned"
        if noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
        if int(altitude_noise_bins) <= 0:
            raise ValueError("altitude_noise_bins must be positive")
        if covariance_mode not in COVARIANCE_MODES:
            raise ValueError(f"covariance_mode must be one of {COVARIANCE_MODES}")
        if query_chunk_size is not None and int(query_chunk_size) <= 0:
            raise ValueError("query_chunk_size must be a positive int or None")
        conformal_mode = str(conformal_mode).lower()
        if conformal_mode not in PREDICTIVE_CONFORMAL_MODES:
            raise ValueError(f"conformal_mode must be one of {PREDICTIVE_CONFORMAL_MODES}")
        if not 0.0 < float(conformal_alpha) < 1.0:
            raise ValueError("conformal_alpha must be in (0, 1)")
        if int(conformal_min_band_n) <= 0:
            raise ValueError("conformal_min_band_n must be positive")
        if int(conformal_min_region_n) <= 0:
            raise ValueError("conformal_min_region_n must be positive")
        self.covariance_mode = covariance_mode
        self.lowrank_rank = int(lowrank_rank)
        self._cov_eig: tuple[torch.Tensor, torch.Tensor] | None = None
        self.dtype = dtype
        self.device = torch.device(device)
        self.sources = sources.to(self.device)
        self.eps = float(eps)
        self.acceleration_sign = float(acceleration_sign)
        self.source_chunk_size = source_chunk_size
        self.query_chunk_size = int(query_chunk_size) if query_chunk_size is not None else None
        self.reg_method = reg_method
        self.lambda_l2 = float(lambda_l2)
        self.noise_model = noise_model
        self.altitude_noise_bins = int(altitude_noise_bins)
        self.val_fraction = float(val_fraction)
        self.conformal_apply = bool(conformal_apply)
        self.conformal_alpha = float(conformal_alpha)
        self.conformal_mode = conformal_mode
        self.conformal_by_band = bool(conformal_by_band)
        self.conformal_bands = {
            str(name): [float(rng[0]), float(rng[1])]
            for name, rng in (conformal_bands or {}).items()
            if rng is not None
        }
        self.conformal_min_band_n = int(conformal_min_band_n)
        self.conformal_by_region = bool(conformal_by_region)
        self.conformal_min_region_n = int(conformal_min_region_n)
        self.conformal_calibration: dict | None = None
        self.low_altitude_radius = float(low_altitude_radius)
        self.risk_scoring = risk_scoring
        self.sigma_threshold = sigma_threshold
        self.altitude_reference_h = (
            float(altitude_reference_h) if altitude_reference_h is not None else None
        )
        self.calibrate_supervisor = bool(calibrate_supervisor)
        self.calibrated_supervisor_grid = dict(calibrated_supervisor_grid or {})
        self.calibrated_supervisor: dict | None = None
        self.domain_support = bool(domain_support)
        self.domain_k = int(domain_k)
        self.domain_weight = float(domain_weight)
        self.domain_distance_weight = float(domain_distance_weight)
        self.domain_radial_weight = float(domain_radial_weight)
        self.domain_angular_weight = float(domain_angular_weight)
        self.domain_backend = str(domain_backend).lower()
        self.seed = int(seed)

        self.posterior: LinearGaussianPosterior | None = None
        self.altitude_noise: AltitudeNoiseModel | BinnedAltitudeNoiseModel | None = None
        self.fit_info: dict = {}
        # Free-form, JSON-safe metadata persisted with the model (e.g. the training run's
        # decision policy / provenance). Round-trips through save()/load(); never interpreted
        # by the plugin itself.
        self.user_metadata: dict = {}

        # domain-support state (populated by fit_error; used by domain_support_score)
        self.train_positions: torch.Tensor | None = None
        self.train_radii: torch.Tensor | None = None
        self.val_positions: torch.Tensor | None = None
        self.val_radii: torch.Tensor | None = None
        self._domain_scale: float | None = None
        self._domain_scale_k: int | None = None
        self._domain_angular_scale: float | None = None
        self._domain_backend_impl = None  # lazily constructed (honors scipy fallback)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: dict) -> VESPUQPlugin:
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
        conformal = uq.get("conformal", {}) or {}
        conformal_bands = conformal.get("bands") or bands
        noise_cfg = uq.get("noise", {}) or {}
        supervisor_cfg = risk.get("calibrated_supervisor", risk.get("calibrate_supervisor", False))
        if isinstance(supervisor_cfg, dict):
            calibrate_supervisor = bool(supervisor_cfg.get("enabled", False))
            calibrated_supervisor_grid = dict(supervisor_cfg)
        else:
            calibrate_supervisor = bool(supervisor_cfg)
            calibrated_supervisor_grid = {}
        return cls(
            sources,
            eps=float(kernel.get("eps", kernel.get("softening", 0.0))),
            acceleration_sign=float(kernel.get("acceleration_sign", 1.0)),
            source_chunk_size=kernel.get("source_chunk_size", 1024),
            query_chunk_size=uq.get("query_chunk_size", 8192),
            reg_method=reg_method,
            lambda_l2=lambda_l2,
            noise_model=str(uq.get("noise_model", "heteroscedastic")).lower(),
            altitude_noise_bins=int(noise_cfg.get("altitude_bins", uq.get("altitude_noise_bins", 8))),
            covariance_mode=str(uq.get("covariance_mode", "exact")).lower(),
            lowrank_rank=int(uq.get("lowrank_rank", 64)),
            val_fraction=float(uq.get("val_fraction", 0.25)),
            conformal_apply=bool(conformal.get("apply", False)),
            conformal_alpha=float(conformal.get("alpha", 0.10)),
            conformal_mode=str(conformal.get("prediction_mode", conformal.get("mode", "norm"))).lower(),
            conformal_by_band=bool(conformal.get("by_band", conformal.get("per_band", False))),
            conformal_bands=conformal_bands,
            conformal_min_band_n=int(conformal.get("min_band_n", 30)),
            conformal_by_region=bool(conformal.get("by_region", conformal.get("per_region", False))),
            conformal_min_region_n=int(conformal.get("min_region_n", 30)),
            low_altitude_radius=float(risk.get("low_altitude_radius", low_band[1])),
            risk_scoring=str(risk.get("scoring", "max")).lower(),
            sigma_threshold=risk.get("sigma_threshold"),
            altitude_reference_h=(
                float(risk["altitude_reference_h"]) if risk.get("altitude_reference_h") is not None else None
            ),
            calibrate_supervisor=calibrate_supervisor,
            calibrated_supervisor_grid=calibrated_supervisor_grid,
            domain_support=bool(risk.get("domain_support", False)),
            domain_k=int(risk.get("domain_k", 8)),
            domain_weight=float(risk.get("domain_weight", 1.0)),
            domain_distance_weight=float(risk.get("domain_distance_weight", 1.0)),
            domain_radial_weight=float(risk.get("domain_radial_weight", 1.0)),
            domain_angular_weight=float(risk.get("domain_angular_weight", 0.0)),
            domain_backend=str(risk.get("domain_backend", "torch")).lower(),
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

    def _require_fitted(self) -> LinearGaussianPosterior:
        posterior = self.posterior
        if posterior is None:
            raise RuntimeError("VESPUQPlugin is not fitted; call fit(...) first")
        return posterior

    def _point_noise(self, radii: torch.Tensor) -> torch.Tensor | float:
        """Aleatoric noise variance per row/point: global floor + altitude excess if het."""

        posterior = self._require_fitted()
        if self.altitude_noise is None:
            return posterior.noise_var
        return posterior.noise_var + self.altitude_noise.variance(radii)

    def _cov_eigpairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-``lowrank_rank`` eigenpairs of the posterior covariance (cached after fit)."""

        posterior = self._require_fitted()
        if self._cov_eig is None:
            vals, vecs = torch.linalg.eigh(posterior.cov)  # ascending
            k = min(self.lowrank_rank, int(vals.numel()))
            self._cov_eig = (vals[-k:].clamp_min(0.0), vecs[:, -k:])
        return self._cov_eig

    def _epistemic_variance(self, operator: torch.Tensor) -> torch.Tensor:
        """Per-row epistemic (source-posterior) variance, honoring ``covariance_mode``.

        ``exact`` uses the full covariance; ``diagonal`` keeps only its diagonal (drops source
        correlations -> O(m*n) instead of O(m*n^2)); ``lowrank`` uses the top-k eigenpairs.
        """

        posterior = self._require_fitted()
        if self.covariance_mode == "diagonal":
            diag = torch.diagonal(posterior.cov)
            return ((operator * operator) @ diag).clamp_min(0.0)
        if self.covariance_mode == "lowrank":
            vals, vecs = self._cov_eigpairs()
            proj = operator @ vecs
            return ((proj * proj) @ vals).clamp_min(0.0)
        cov_q = operator @ posterior.cov
        return torch.sum(cov_q * operator, dim=-1).clamp_min(0.0)

    def _predict_rows(self, operator: torch.Tensor, radii: torch.Tensor) -> dict[str, torch.Tensor]:
        """Row-level (3N) predictive mean/variance honoring noise model + covariance mode."""

        posterior = self._require_fitted()
        mean = operator @ posterior.mean
        epistemic = self._epistemic_variance(operator)
        variance = epistemic + self._point_noise(radii)
        return {
            "mean": mean,
            "epistemic_variance": epistemic,
            "variance": variance,
            "std": torch.sqrt(variance.clamp_min(torch.finfo(mean.dtype).tiny)),
        }

    def _query_chunks(self, n: int) -> list[tuple[int, int]]:
        """Half-open ``(start, end)`` query slices honoring ``query_chunk_size`` (one slice if off)."""

        size = self.query_chunk_size
        if size is None or n <= size:
            return [(0, n)]
        return [(start, min(start + size, n)) for start in range(0, n, size)]

    def _fit_altitude_noise_model(
        self,
        radii: torch.Tensor,
        residuals: torch.Tensor,
        epistemic_var: torch.Tensor,
    ) -> AltitudeNoiseModel | BinnedAltitudeNoiseModel | None:
        """Fit the configured held-out altitude-noise law."""

        if self.noise_model == "heteroscedastic":
            return AltitudeNoiseModel.fit(radii, residuals, epistemic_var)
        if self.noise_model == "altitude_binned":
            return BinnedAltitudeNoiseModel.fit(
                radii,
                residuals,
                epistemic_var,
                n_bins=self.altitude_noise_bins,
            )
        return None

    def _record_altitude_noise_fit_info(self) -> None:
        """Attach altitude-noise provenance to ``fit_info`` without assuming a model type."""

        noise = self.altitude_noise
        if noise is None:
            return
        if isinstance(noise, BinnedAltitudeNoiseModel):
            self.fit_info["altitude_noise_type"] = "altitude_binned"
            self.fit_info["altitude_noise_bins"] = int(noise.n_bins)
            self.fit_info["altitude_noise_radius_min"] = float(noise.radii.min().detach().cpu())
            self.fit_info["altitude_noise_radius_max"] = float(noise.radii.max().detach().cpu())
        else:
            self.fit_info["altitude_noise_type"] = "power_law"
            self.fit_info["altitude_noise_a"] = noise.a
            self.fit_info["altitude_noise_b"] = noise.b

    def _altitude_noise_state(self, cpu_fn) -> dict | None:
        noise = self.altitude_noise
        if noise is None:
            return None
        if isinstance(noise, BinnedAltitudeNoiseModel):
            return {
                "type": "altitude_binned",
                "radii": cpu_fn(noise.radii),
                "log_variance": cpu_fn(noise.log_variance),
                "h_floor": float(noise.h_floor),
            }
        return {
            "type": "power_law",
            "log_a": noise.log_a,
            "b": noise.b,
            "h_floor": noise.h_floor,
        }

    @staticmethod
    def _spearman_rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        """Spearman rank correlation with average ranks for ties."""

        x = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
        y = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
        if x.numel() != y.numel():
            raise ValueError("Spearman inputs must have the same length")
        if x.numel() < 2 or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            return float("nan")

        # base is irrelevant here: ranks only feed the shift-invariant correlation below.
        rx = average_ranks(x, base=0.0)
        ry = average_ranks(y, base=0.0)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = torch.sqrt((rx * rx).sum() * (ry * ry).sum())
        if float(denom) <= 0.0:
            return float("nan")
        return float((rx * ry).sum() / denom)

    @staticmethod
    def _fit_altitude_expected_curve_from_points(
        radius: torch.Tensor,
        expected_error: torch.Tensor,
        *,
        n_bins: int = 8,
    ) -> dict:
        """Fit a small monotone-free altitude curve for calibrated supervisor residuals."""

        r = radius.detach().reshape(-1).to(torch.float64)
        ee = expected_error.detach().reshape(-1).to(torch.float64)
        if r.numel() != ee.numel():
            raise ValueError("radius and expected_error must have the same length")
        if r.numel() == 0:
            raise ValueError("cannot fit an altitude curve with no samples")
        tiny = torch.finfo(ee.dtype).tiny
        order = torch.argsort(r)
        chunks = torch.chunk(order, max(1, min(int(n_bins), int(order.numel()))))
        centers: list[float] = []
        values: list[float] = []
        for idx in chunks:
            if idx.numel() == 0:
                continue
            centers.append(float(torch.median(r[idx]).detach().cpu()))
            values.append(float(torch.median(ee[idx].clamp_min(tiny)).detach().cpu()))
        return {"radii": centers, "expected_error": values}

    @staticmethod
    def _predict_altitude_expected_from_curve(radius: torch.Tensor, curve: dict) -> torch.Tensor:
        if not curve or "radii" not in curve or "expected_error" not in curve:
            return torch.ones_like(radius)
        centers = torch.as_tensor(curve["radii"], dtype=radius.dtype, device=radius.device).reshape(-1)
        values = torch.as_tensor(curve["expected_error"], dtype=radius.dtype, device=radius.device).reshape(-1)
        if centers.numel() == 0:
            return torch.ones_like(radius)
        if centers.numel() == 1:
            return values[0].expand_as(radius)
        lo = float(centers[0].detach().cpu())
        hi = float(centers[-1].detach().cpu())
        x = radius.reshape(-1).clamp(min=lo, max=hi)
        idx_hi = torch.searchsorted(centers, x, right=False).clamp(1, centers.numel() - 1)
        idx_lo = idx_hi - 1
        x0, x1 = centers[idx_lo], centers[idx_hi]
        y0 = torch.log(values[idx_lo].clamp_min(torch.finfo(radius.dtype).tiny))
        y1 = torch.log(values[idx_hi].clamp_min(torch.finfo(radius.dtype).tiny))
        denom = (x1 - x0).clamp_min(torch.finfo(radius.dtype).eps)
        t = (x - x0) / denom
        return torch.exp(y0 + t * (y1 - y0)).reshape_as(radius)

    def _calibrated_supervisor_point_risk(
        self,
        pred: UncertaintyPrediction,
        domain_risk: torch.Tensor | None,
        *,
        state: dict | None = None,
    ) -> torch.Tensor | None:
        """Learned point-risk formula used by ``calibrated_supervisor`` scoring modes."""

        cal = self.calibrated_supervisor if state is None else state
        if not cal or not cal.get("enabled", False):
            return None
        weights = cal["weights"]
        tiny = torch.finfo(pred.expected_error.dtype).tiny
        expected = pred.expected_error.clamp_min(tiny)
        altitude_curve = cal.get("altitude_expected_curve") or {}
        altitude_expected = self._predict_altitude_expected_from_curve(pred.radius, altitude_curve).clamp_min(tiny)
        altitude_residual = (expected / altitude_expected).clamp_min(tiny)
        epi_factor = 1.0 + float(weights.get("epistemic_weight", 0.0)) * pred.epistemic_fraction.clamp_min(0.0)
        if domain_risk is not None and float(weights.get("domain_weight", 0.0)) > 0.0:
            domain_factor = 1.0 + float(weights["domain_weight"]) * domain_risk.clamp_min(0.0)
        else:
            domain_factor = torch.ones_like(expected)
        return (
            expected.pow(float(weights.get("expected_power", 1.0)))
            * altitude_residual.pow(float(weights.get("altitude_residual_power", 0.0)))
            * epi_factor
            * domain_factor
        )

    @staticmethod
    def _grid_values(config: dict, key: str, default: list[float]) -> list[float]:
        raw = config.get(key, default)
        if raw is None:
            raw = default
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(v) for v in raw]

    def fit_calibrated_supervisor(self, positions, error) -> dict:
        """Calibrate a learned supervisor point-risk formula on held-out samples."""

        self._require_fitted()
        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        if positions.shape != error.shape:
            raise ValueError("positions and error must have the same (N, 3) shape")
        if positions.shape[0] < 3:
            self.calibrated_supervisor = {
                "enabled": False,
                "reason": "n_calibration < 3",
                "n_calibration": int(positions.shape[0]),
            }
            self.fit_info["calibrated_supervisor_enabled"] = False
            return self.calibrated_supervisor

        pred = self.predict_uncertainty(positions)
        domain_risk = self.domain_support_score(positions) if self.domain_support else None
        target = torch.linalg.norm(error, dim=-1).to(dtype=torch.float64)
        curve_bins = int(self.calibrated_supervisor_grid.get("altitude_bins", 8))
        curve = self._fit_altitude_expected_curve_from_points(
            pred.radius,
            pred.expected_error,
            n_bins=curve_bins,
        )
        base_state = {"enabled": True, "altitude_expected_curve": curve, "weights": {}}

        expected_grid = self._grid_values(
            self.calibrated_supervisor_grid,
            "expected_power",
            [0.75, 1.0, 1.25],
        )
        altitude_grid = self._grid_values(
            self.calibrated_supervisor_grid,
            "altitude_residual_power",
            [0.0, 0.5, 1.0],
        )
        epistemic_grid = self._grid_values(
            self.calibrated_supervisor_grid,
            "epistemic_weight",
            [0.0, 0.5, 1.0],
        )
        domain_grid = (
            self._grid_values(self.calibrated_supervisor_grid, "domain_weight", [0.0, 0.5, 1.0])
            if domain_risk is not None
            else [0.0]
        )

        best_score = float("-inf")
        best_weights: dict[str, float] | None = None
        best_risk: torch.Tensor | None = None
        for expected_power in expected_grid:
            for altitude_power in altitude_grid:
                for epistemic_weight in epistemic_grid:
                    for domain_weight in domain_grid:
                        weights = {
                            "expected_power": float(expected_power),
                            "altitude_residual_power": float(altitude_power),
                            "epistemic_weight": float(epistemic_weight),
                            "domain_weight": float(domain_weight),
                        }
                        state = {**base_state, "weights": weights}
                        risk = self._calibrated_supervisor_point_risk(
                            pred,
                            domain_risk,
                            state=state,
                        )
                        if risk is None:
                            continue
                        rho = self._spearman_rank_corr(risk.detach().cpu(), target.detach().cpu())
                        if not bool(torch.isfinite(torch.tensor(rho))):
                            continue
                        tie_breaks_domain_off = (
                            best_weights is not None
                            and abs(rho - best_score) <= 1.0e-12
                            and float(domain_weight) < float(best_weights.get("domain_weight", 0.0))
                        )
                        if rho > best_score + 1.0e-12 or tie_breaks_domain_off:
                            best_score = float(rho)
                            best_weights = weights
                            best_risk = risk

        if best_weights is None or best_risk is None:
            self.calibrated_supervisor = {
                "enabled": False,
                "reason": "no finite calibration objective",
                "n_calibration": int(positions.shape[0]),
            }
            self.fit_info["calibrated_supervisor_enabled"] = False
            return self.calibrated_supervisor

        self.calibrated_supervisor = {
            "enabled": True,
            "objective": "spearman_pointwise_true_error_magnitude",
            "n_calibration": int(positions.shape[0]),
            "spearman": best_score,
            "weights": best_weights,
            "altitude_expected_curve": curve,
            "domain_disabled": float(best_weights.get("domain_weight", 0.0)) <= 0.0,
            "grid": {
                "expected_power": expected_grid,
                "altitude_residual_power": altitude_grid,
                "epistemic_weight": epistemic_grid,
                "domain_weight": domain_grid,
            },
            "mean_point_risk": float(best_risk.mean().detach().cpu()),
        }
        self.fit_info.update(
            {
                "calibrated_supervisor_enabled": True,
                "calibrated_supervisor_spearman": best_score,
                "calibrated_supervisor_expected_power": best_weights["expected_power"],
                "calibrated_supervisor_altitude_residual_power": best_weights["altitude_residual_power"],
                "calibrated_supervisor_epistemic_weight": best_weights["epistemic_weight"],
                "calibrated_supervisor_domain_weight": best_weights["domain_weight"],
                "calibrated_supervisor_domain_disabled": self.calibrated_supervisor["domain_disabled"],
                "calibrated_supervisor_stale_after_update": False,
            }
        )
        return self.calibrated_supervisor

    @staticmethod
    def _conformal_region_ids(positions: torch.Tensor) -> torch.Tensor:
        """Eight angular octants encoded as 0..7 from the signs of x/y/z."""

        signs = (positions >= 0.0).to(torch.int64)
        return signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]

    @staticmethod
    def _conformal_region_signs(region_id: int) -> list[int]:
        return [
            1 if (int(region_id) & 4) else -1,
            1 if (int(region_id) & 2) else -1,
            1 if (int(region_id) & 1) else -1,
        ]

    def _conformal_scale_for_query(
        self,
        radius: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Per-query predictive-std scale, optionally refined by angular region."""

        cal = self.conformal_calibration
        if not cal or not cal.get("enabled", False):
            return None
        scale = torch.full_like(radius, float(cal["global"]["scale"]))
        for band in cal.get("bands", []):
            if not band.get("used", True) or "scale" not in band:
                continue
            lo, hi = float(band["radius_range"][0]), float(band["radius_range"][1])
            mask = (radius >= lo) & (radius <= hi)
            if bool(mask.any()):
                scale[mask] = float(band["scale"])
        if positions is not None:
            region_ids = self._conformal_region_ids(positions)
            for region in cal.get("regions", []):
                if not region.get("used", True) or "scale" not in region:
                    continue
                mask = region_ids == int(region["region_id"])
                if bool(mask.any()):
                    scale[mask] = float(region["scale"])
        return scale

    def _conformal_scale_for_radius(self, radius: torch.Tensor) -> torch.Tensor | None:
        """Per-query predictive-std scale, or ``None`` when operational conformal is off."""

        return self._conformal_scale_for_query(radius)

    def _apply_conformal_to_std(
        self,
        std_components: torch.Tensor,
        sigma: torch.Tensor,
        radius: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale predictive standard deviations when operational conformal is enabled."""

        scale = self._conformal_scale_for_query(radius, positions)
        if scale is None:
            return std_components, sigma
        return std_components * scale.unsqueeze(-1), sigma * scale

    def _apply_conformal_to_covariance(
        self,
        covariance: torch.Tensor,
        radius: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scale predictive covariance by ``scale^2`` when operational conformal is enabled."""

        scale = self._conformal_scale_for_query(radius, positions)
        if scale is None:
            return covariance
        return covariance * scale.square().reshape(-1, 1, 1)

    def fit_conformal_calibration(self, positions, error, *, altitude_bands: dict | None = None) -> dict:
        """Fit the opt-in operational conformal scale on held-out force-error samples.

        The learned scale multiplies predictive standard deviations. If ``conformal_by_band`` is
        enabled, in-band queries use their band scale and all extrapolated / under-populated regions
        fall back to the global scale.
        """

        self._require_fitted()
        if not self.conformal_apply:
            self.conformal_calibration = None
            return {"enabled": False, "apply": False}

        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        if positions.shape != error.shape:
            raise ValueError("positions and error must have the same (N, 3) shape")
        if positions.shape[0] == 0:
            raise ValueError("conformal calibration requires at least one held-out sample")

        previous = self.conformal_calibration
        self.conformal_calibration = None
        try:
            cov = self.predict_covariance_3x3(positions)
        except Exception:
            self.conformal_calibration = previous
            raise

        residual = error - cov.mean_error
        if self.conformal_mode == "mahalanobis":
            true_for_conformal = torch.sqrt(mahalanobis_squared(residual, cov.covariance).clamp_min(0.0))
            predicted = torch.ones_like(true_for_conformal)
        else:
            true_for_conformal = residual
            predicted = cov.std_components if self.conformal_mode == "component_max" else cov.sigma
        global_cal = fit_conformal_scale(
            predicted,
            true_for_conformal,
            alpha=self.conformal_alpha,
            mode=self.conformal_mode,
        )
        radius = torch.linalg.norm(positions, dim=-1)
        bands_out: list[dict] = []
        if self.conformal_by_band:
            for name, rng in (altitude_bands or self.conformal_bands).items():
                if rng is None:
                    continue
                lo, hi = float(rng[0]), float(rng[1])
                mask = (radius >= lo) & (radius <= hi)
                n_band = int(mask.sum().detach().cpu())
                if n_band < self.conformal_min_band_n:
                    bands_out.append(
                        {
                            "name": str(name),
                            "radius_range": [lo, hi],
                            "n_calibration": n_band,
                            "used": False,
                            "reason": f"n < min_band_n ({self.conformal_min_band_n})",
                        }
                    )
                    continue
                band_predicted = predicted[mask]
                band_true = true_for_conformal[mask]
                band_cal = fit_conformal_scale(
                    band_predicted,
                    band_true,
                    alpha=self.conformal_alpha,
                    mode=self.conformal_mode,
                )
                band_dict = band_cal.to_dict()
                band_dict.update(
                    {
                        "name": str(name),
                        "radius_range": [lo, hi],
                        "used": True,
                    }
                )
                bands_out.append(band_dict)

        regions_out: list[dict] = []
        if self.conformal_by_region:
            region_ids = self._conformal_region_ids(positions)
            for region_id in range(8):
                mask = region_ids == region_id
                n_region = int(mask.sum().detach().cpu())
                if n_region < self.conformal_min_region_n:
                    regions_out.append(
                        {
                            "region_id": int(region_id),
                            "signs": self._conformal_region_signs(region_id),
                            "n_calibration": n_region,
                            "used": False,
                            "reason": f"n < min_region_n ({self.conformal_min_region_n})",
                        }
                    )
                    continue
                region_cal = fit_conformal_scale(
                    predicted[mask],
                    true_for_conformal[mask],
                    alpha=self.conformal_alpha,
                    mode=self.conformal_mode,
                )
                region_dict = region_cal.to_dict()
                region_dict.update(
                    {
                        "region_id": int(region_id),
                        "signs": self._conformal_region_signs(region_id),
                        "used": True,
                    }
                )
                regions_out.append(region_dict)

        scope_parts = []
        if self.conformal_by_band:
            scope_parts.append("per_band")
        if self.conformal_by_region:
            scope_parts.append("per_region")
        scope = "+".join(scope_parts) if scope_parts else "global"
        self.conformal_calibration = {
            "enabled": True,
            "apply": True,
            "alpha": self.conformal_alpha,
            "mode": self.conformal_mode,
            "scope": scope,
            "min_band_n": self.conformal_min_band_n,
            "min_region_n": self.conformal_min_region_n,
            "global": global_cal.to_dict(),
            "bands": bands_out,
            "regions": regions_out,
            "extrapolation_rule": (
                "queries outside a fitted band/region use the global conformal scale; "
                "region scales override band scales when both apply"
            ),
        }
        self.fit_info.update(
            {
                "conformal_apply": True,
                "conformal_mode": self.conformal_mode,
                "conformal_alpha": self.conformal_alpha,
                "conformal_scope": self.conformal_calibration["scope"],
                "conformal_scale": float(global_cal.scale),
                "conformal_coverage_before": float(global_cal.coverage_before),
                "conformal_coverage_after": float(global_cal.coverage_after),
                "conformal_stale_after_update": False,
            }
        )
        if bands_out:
            self.fit_info["conformal_band_scales"] = {
                b["name"]: b.get("scale") for b in bands_out if b.get("used")
            }
        if regions_out:
            self.fit_info["conformal_region_scales"] = {
                str(b["region_id"]): b.get("scale") for b in regions_out if b.get("used")
            }
        return self.conformal_calibration

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
    ) -> VESPUQPlugin:
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

    def fit_error(self, positions, error, *, val_positions=None, val_error=None) -> VESPUQPlugin:
        """Fit directly from sampled force-error vectors ``error = a_reference - a_surrogate``.

        The Tikhonov weight is selected automatically (L-curve corner by default). The posterior
        mean is the ridge solution; the global noise floor and the altitude-dependent excess
        noise are calibrated on a HELD-OUT validation split (an internal random split unless an
        explicit ``val_positions``/``val_error`` is supplied), because training residuals are
        optimistic and underestimate the altitude-dependent generalization error.
        """

        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        self.conformal_calibration = None
        self.calibrated_supervisor = None

        if (val_positions is None) != (val_error is None):
            raise ValueError("supply both val_positions and val_error (or neither)")
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
            assert val_error is not None
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
        self._domain_angular_scale = None

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
                assert val_err is not None
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
        if self.noise_model in {"heteroscedastic", "altitude_binned"} and val_pos is not None:
            assert val_err is not None
            val_op = self._operator(val_pos)
            val_pred = posterior.predict(val_op, include_noise=False)
            val_resid = val_pred["mean"] - _flatten_acc(val_err)
            val_row_radii = torch.linalg.norm(val_pos, dim=-1).repeat(3)
            self.altitude_noise = self._fit_altitude_noise_model(
                val_row_radii,
                val_resid,
                val_pred["epistemic_variance"] + posterior.noise_var,
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
            "domain_support_enabled": self.domain_support,
            "domain_backend": self.domain_backend,
            "domain_k": self.domain_k,
            "domain_weight": float(self.domain_weight),
            "domain_distance_weight": float(self.domain_distance_weight),
            "domain_radial_weight": float(self.domain_radial_weight),
            "domain_angular_weight": float(self.domain_angular_weight),
            "domain_support_components": ["distance_score", "radius_penalty", "angular_score", "total_score"],
            "train_radius_min": float(self.train_radii.min()),
            "train_radius_max": float(self.train_radii.max()),
        }
        if self.domain_support:
            # the nearest-neighbour scale is otherwise lazy; materialize it for the report
            self.fit_info["domain_scale"] = float(self._domain_nn_scale(self.domain_k))
            if self.domain_angular_weight > 0.0:
                self.fit_info["domain_angular_scale"] = float(self._domain_angular_nn_scale(self.domain_k))
        self._record_altitude_noise_fit_info()
        if lcurve_points is not None:
            self.fit_info["lcurve"] = lcurve_points
        if self.conformal_apply:
            if val_pos is None or val_err is None:
                raise ValueError("conformal_apply=True requires held-out validation samples")
            self.fit_conformal_calibration(val_pos, val_err)
        if self.calibrate_supervisor:
            if val_pos is None or val_err is None:
                raise ValueError("calibrate_supervisor=True requires held-out validation samples")
            self.fit_calibrated_supervisor(val_pos, val_err)
        return self

    # ------------------------------------------------------------------ sequential update
    def update_error(self, positions, error, *, val_positions=None, val_error=None) -> VESPUQPlugin:
        """EXACT sequential Bayesian update of the fitted posterior with new error samples.

        Because the model is linear-Gaussian, conditioning the current posterior on additional
        rows ``(A2, b2)`` with the SAME Tikhonov weight and noise floor is exact -- it equals the
        batch refit on the concatenated data:

            M1    = noise_var * Sigma1^{-1}          (= A1^T A1 + lambda I, recovered from cov)
            M2    = M1 + A2^T A2
            mean2 = M2^{-1} (M1 mean1 + A2^T b2)     (M1 mean1 = A1^T b1 exactly)
            Sigma2 = noise_var * M2^{-1}

        Deliberately FIXED across an update (exactness over re-selection):

        - the Tikhonov weight ``lambda`` -- the L-curve is NOT re-run (re-selecting lambda would
          change the prior and break the exact-update contract);
        - the global noise floor and the altitude noise law -- UNLESS a fresh held-out
          ``val_positions``/``val_error`` pair is supplied, in which case both are recalibrated
          on it (same procedure as :meth:`fit_error`). Without fresh validation data the
          calibration is the one from the original fit: re-validate before relying on per-band
          coverage after large updates.

        The domain-support calibration geometry is extended with the new positions (cached
        nearest-neighbour scales are invalidated and recomputed lazily).
        """

        posterior = self._require_fitted()
        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        if positions.shape[0] == 0:
            raise ValueError("update_error requires at least one new sample")
        if positions.shape != error.shape:
            raise ValueError("positions and error must have the same (N, 3) shape")

        noise_var = posterior.noise_var
        lambda_used = posterior.lambda_l2

        # Recover the posterior precision (up to the noise scale) from the stored covariance.
        chol_cov = torch.linalg.cholesky(posterior.cov)
        m1 = noise_var * torch.cholesky_inverse(chol_cov)
        a2 = self._operator(positions)
        b2 = _flatten_acc(error)
        m2 = m1 + a2.transpose(-1, -2) @ a2
        rhs = m1 @ posterior.mean + a2.transpose(-1, -2) @ b2
        chol2 = _safe_cholesky(m2, jitter=1.0e-10)
        mean2 = torch.cholesky_solve(rhs.unsqueeze(-1), chol2).squeeze(-1)

        # Optional honest recalibration on FRESH held-out data (mirrors fit_error: the noise
        # floor comes from held-out residuals of the updated mean, which is noise-scale-free).
        val_pos = val_err = None
        if val_positions is not None or val_error is not None:
            if val_positions is None or val_error is None:
                raise ValueError("supply both val_positions and val_error (or neither)")
            val_pos = self._prep_positions(val_positions)
            val_err = self._prep_positions(val_error)
            val_op = self._operator(val_pos)
            val_resid = val_op @ mean2 - _flatten_acc(val_err)
            noise_var = max(float(torch.mean(val_resid * val_resid).detach().cpu()),
                            float(torch.finfo(mean2.dtype).tiny))

        self.posterior = LinearGaussianPosterior(
            mean=mean2,
            cov=noise_var * torch.cholesky_inverse(chol2),
            noise_var=noise_var,
            lambda_l2=lambda_used,
        )
        self._cov_eig = None

        # Extend the domain-support geometry; the new samples are calibration support now.
        if self.train_positions is None:
            raise RuntimeError("fitted plugin is missing stored training positions")
        self.train_positions = torch.cat([self.train_positions, positions.detach()], dim=0)
        self.train_radii = torch.linalg.norm(self.train_positions, dim=-1).detach()
        self._domain_scale = None
        self._domain_scale_k = None
        self._domain_angular_scale = None

        if val_pos is not None:
            assert val_err is not None
            self.val_positions = val_pos.detach()
            self.val_radii = torch.linalg.norm(val_pos, dim=-1).detach()
            if self.noise_model in {"heteroscedastic", "altitude_binned"}:
                val_op = self._operator(val_pos)
                val_pred = self.posterior.predict(val_op, include_noise=False)
                val_resid = val_pred["mean"] - _flatten_acc(val_err)
                self.altitude_noise = self._fit_altitude_noise_model(
                    torch.linalg.norm(val_pos, dim=-1).repeat(3),
                    val_resid,
                    val_pred["epistemic_variance"] + noise_var,
                )

        # Bookkeeping: the fit provenance now reflects the updated state.
        self.fit_info["n_train"] = int(self.train_positions.shape[0])
        self.fit_info["n_updates"] = int(self.fit_info.get("n_updates", 0)) + 1
        self.fit_info["noise_var"] = self.posterior.noise_var
        self.fit_info["noise_std"] = float(self.posterior.noise_var ** 0.5)
        self.fit_info["train_radius_min"] = float(self.train_radii.min())
        self.fit_info["train_radius_max"] = float(self.train_radii.max())
        if val_pos is not None:
            self.fit_info["n_val"] = int(val_pos.shape[0])
            self._record_altitude_noise_fit_info()
            if self.conformal_apply:
                self.fit_conformal_calibration(val_pos, val_err)
            if self.calibrate_supervisor:
                self.fit_calibrated_supervisor(val_pos, val_err)
        elif self.conformal_apply and self.conformal_calibration:
            self.fit_info["conformal_stale_after_update"] = True
            self.fit_info["conformal_stale_reason"] = "update_error called without fresh validation data"
        if val_pos is None and self.calibrated_supervisor:
            self.fit_info["calibrated_supervisor_stale_after_update"] = True
            self.fit_info["calibrated_supervisor_stale_reason"] = (
                "update_error called without fresh validation data"
            )
        return self

    # ------------------------------------------------------------------ prediction
    def predict_uncertainty(self, positions) -> UncertaintyPrediction:
        """Predict the mean force-error and calibrated per-position predictive uncertainty.

        Queries are processed in ``query_chunk_size`` blocks so the dense ``(3N, n_sources)``
        operator never has to be materialized for a large position set at once.
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        chunks = self._query_chunks(positions.shape[0])
        if len(chunks) == 1:
            return self._predict_uncertainty_block(positions)
        parts = [self._predict_uncertainty_block(positions[a:b]) for a, b in chunks]
        fields = {
            name: torch.cat([getattr(p, name) for p in parts], dim=0)
            for name in (
                "positions",
                "radius",
                "mean_error",
                "std_components",
                "sigma",
                "epistemic_sigma",
                "mean_error_magnitude",
                "expected_error",
                "epistemic_fraction",
                "risk_score",
            )
        }
        return UncertaintyPrediction(**fields)

    def _predict_uncertainty_block(self, positions: torch.Tensor) -> UncertaintyPrediction:
        """Single-block uncertainty prediction (``positions`` already validated/prepped)."""

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
        std3, sigma = self._apply_conformal_to_std(std3, sigma, radius, positions)

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

        Queries are processed in ``query_chunk_size`` blocks (same policy as
        :meth:`predict_uncertainty`).
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        chunks = self._query_chunks(positions.shape[0])
        if len(chunks) == 1:
            return self._predict_covariance_block(positions)
        parts = [self._predict_covariance_block(positions[a:b]) for a, b in chunks]
        fields = {
            name: torch.cat([getattr(p, name) for p in parts], dim=0)
            for name in ("positions", "mean_error", "covariance", "std_components", "sigma")
        }
        return CovariancePrediction(**fields)

    def _predict_covariance_block(self, positions: torch.Tensor, *, operator: torch.Tensor | None = None) -> CovariancePrediction:
        """Single-block ``3x3`` covariance prediction (``positions`` already validated/prepped)."""

        posterior = self._require_fitted()
        n = positions.shape[0]
        op = operator if operator is not None else self._operator(positions)
        opx, opy, opz = op[:n], op[n : 2 * n], op[2 * n :]
        radius = torch.linalg.norm(positions, dim=-1)

        zeros = torch.zeros(n, dtype=self.dtype, device=self.device)
        if self.covariance_mode == "diagonal":
            diag = torch.diagonal(posterior.cov)
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
                tx, ty, tz = opx @ posterior.cov, opy @ posterior.cov, opz @ posterior.cov

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
        cov = self._apply_conformal_to_covariance(cov, radius, positions)

        mean3 = (op @ posterior.mean).reshape(3, n).transpose(0, 1)
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
    def _domain_backend(self):
        """Lazily construct the domain-support nearest-neighbour backend (scipy-safe fallback)."""

        if self._domain_backend_impl is None:
            self._domain_backend_impl = make_domain_backend(self.domain_backend)
        return self._domain_backend_impl

    def _domain_nn_scale(self, k: int, *, subset: int = 512) -> float:
        """Robust training-set length scale: median ``k``-th nearest-neighbour distance (cached)."""

        if self.train_positions is None:
            raise RuntimeError("domain support needs a fit; train positions are not stored")
        if self._domain_scale is not None and self._domain_scale_k == k:
            return self._domain_scale
        scale = median_knn_scale(
            self.train_positions, k, backend=self._domain_backend(), seed=self.seed, subset=subset
        )
        self._domain_scale, self._domain_scale_k = scale, k
        return scale

    def _domain_angular_nn_scale(self, k: int, *, subset: int = 512) -> float:
        """Robust angular spacing: median ``k``-th nearest training-direction angle (radians)."""

        train_positions = self.train_positions
        if train_positions is None:
            raise RuntimeError("domain support needs a fit; train positions are not stored")
        if self._domain_angular_scale is None:
            self._domain_angular_scale = median_angular_scale(
                train_positions, k, seed=self.seed + 1, subset=subset
            )
        return self._domain_angular_scale

    def domain_support_components(
        self, positions, k: int | None = None, *, chunk: int = 1024
    ) -> dict[str, torch.Tensor]:
        """Decomposed per-position domain-support score for query ``positions`` ``(N, 3)``.

        Returns the (already weight-applied, nonnegative) contributions plus their sum:

        - ``distance_score``: ``domain_distance_weight * (d_k / scale - 1)+`` -- the query's
          ``k``-th nearest training-point Euclidean distance vs the median training ``k``-NN
          spacing. This already captures *angular* out-of-support (a same-radius query far from
          the training directions has a large Euclidean nearest distance).
        - ``radius_penalty``: ``domain_radial_weight * (radius extrapolation below r_min / above
          r_max) / scale`` -- the purely radial out-of-support term.
        - ``angular_score``: ``domain_angular_weight * (theta_k / angular_scale - 1)+`` -- an
          OPTIONAL nearest-direction angular term (off by default; the distance term usually
          suffices). Zeros unless ``domain_angular_weight > 0``.
        - ``total_score``: the sum of the three (so the components always sum to the total).

        ``0`` means well inside the calibration support; ``> 1`` means increasingly extrapolated.
        Chunked over queries so it stays cheap even for large ensembles.
        """

        train_positions = self.train_positions
        train_radii = self.train_radii
        if train_positions is None or train_radii is None:
            raise RuntimeError("domain support needs a fit; call fit(...)/fit_error(...) first")
        k = int(self.domain_k if k is None else k)
        pos = self._prep_positions(positions)
        want_angular = self.domain_angular_weight > 0.0
        return _domain_support_components(
            pos,
            train_positions,
            train_radii,
            k=k,
            distance_scale=self._domain_nn_scale(k),
            distance_weight=self.domain_distance_weight,
            radial_weight=self.domain_radial_weight,
            angular_weight=self.domain_angular_weight,
            angular_scale=self._domain_angular_nn_scale(k) if want_angular else None,
            backend=self._domain_backend(),
            chunk=chunk,
        )

    def domain_support_score(self, positions, k: int | None = None, *, chunk: int = 1024) -> torch.Tensor:
        """Total per-position domain-support score (sum of :meth:`domain_support_components`)."""

        return self.domain_support_components(positions, k=k, chunk=chunk)["total_score"]

    # ------------------------------------------------------------------ trajectory scoring
    def _score_profile(
        self,
        pred: UncertaintyPrediction,
        *,
        domain_risk: torch.Tensor | None,
        scoring: str | None,
        weights,
        covariance: torch.Tensor | None = None,
    ) -> TrajectoryScore:
        """Aggregate one trajectory's per-point profile with the plugin's scoring settings.

        ``covariance`` (the per-point ``3x3``) is supplied only for the directional modes that need
        it (gated upstream via :func:`vesp.uq.scoring.needs_covariance`); it is ``None`` otherwise.
        """

        calibrated_point_risk = self._calibrated_supervisor_point_risk(pred, domain_risk)
        return score_sigma_profile(
            pred.sigma,
            pred.radius,
            scoring=scoring or self.risk_scoring,
            sigma_threshold=self.sigma_threshold,
            low_altitude_radius=self.low_altitude_radius,
            altitude_reference_h=self.altitude_reference_h,
            epistemic_sigma=pred.epistemic_sigma,
            expected_error=pred.expected_error,
            mean_error_magnitude=pred.mean_error_magnitude,
            domain_risk=domain_risk,
            domain_weight=self.domain_weight,
            calibrated_point_risk=calibrated_point_risk,
            weights=weights,
            # M1 / M2 directional + epistemic inputs (consulted only by those modes)
            mean_error_vector=pred.mean_error,
            std_components=pred.std_components,
            covariance=covariance,
            positions=pred.positions,
            epistemic_fraction=pred.epistemic_fraction,
        )

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
        covariance = (
            self.predict_covariance_3x3(pred.positions).covariance
            if needs_covariance(scoring or self.risk_scoring) else None
        )
        return self._score_profile(pred, domain_risk=domain_risk, scoring=scoring, weights=weights,
                                   covariance=covariance)

    def score_ensemble(
        self, trajectories, *, scoring: str | None = None, weights=None
    ) -> list[TrajectoryScore]:
        """Score an iterable of trajectories (each ``(T_i, 3)``).

        ``weights`` is either ``None`` (uniform time weighting for every trajectory) or an
        iterable of per-trajectory weight vectors aligned with ``trajectories`` (entries may be
        ``None`` to keep a given trajectory uniform).

        The ensemble is scored in one batched pass: all trajectory points are concatenated and
        pushed through :meth:`predict_uncertainty` (which is query-chunked, so memory stays
        bounded) and one domain-support call, then the per-point profile is split back per
        trajectory. Per-trajectory numbers are identical to calling :meth:`score_trajectory` in
        a loop -- this is purely an amortization of operator construction and matmul dispatch.
        """

        traj_list = list(trajectories)
        if weights is None:
            weight_list = [None] * len(traj_list)
        else:
            weight_list = list(weights)
            if len(weight_list) != len(traj_list):
                raise ValueError("weights must be None or one weight vector per trajectory")
        if not traj_list:
            return []

        prepped = [self._prep_positions(t) for t in traj_list]
        lengths = [int(t.shape[0]) for t in prepped]
        all_positions = torch.cat(prepped, dim=0)
        pred = self.predict_uncertainty(all_positions)
        domain_risk = self.domain_support_score(pred.positions) if self.domain_support else None
        # M1: build the full 3x3 covariance once (gated) for the directional modes that need it,
        # so default scoring pays nothing for the extra operator pass.
        covariance = (
            self.predict_covariance_3x3(all_positions).covariance
            if needs_covariance(scoring or self.risk_scoring) else None
        )

        scores: list[TrajectoryScore] = []
        offset = 0
        for n, w in zip(lengths, weight_list, strict=True):
            sl = slice(offset, offset + n)
            offset += n
            block = UncertaintyPrediction(
                positions=pred.positions[sl],
                radius=pred.radius[sl],
                mean_error=pred.mean_error[sl],
                std_components=pred.std_components[sl],
                sigma=pred.sigma[sl],
                epistemic_sigma=pred.epistemic_sigma[sl],
                mean_error_magnitude=pred.mean_error_magnitude[sl],
                expected_error=pred.expected_error[sl],
                epistemic_fraction=pred.epistemic_fraction[sl],
                risk_score=pred.risk_score[sl],
            )
            scores.append(
                self._score_profile(
                    block,
                    domain_risk=domain_risk[sl] if domain_risk is not None else None,
                    scoring=scoring,
                    weights=w,
                    covariance=covariance[sl] if covariance is not None else None,
                )
            )
        return scores

    # ------------------------------------------------------------------ threshold calibration
    def calibrate_pointwise_expected_error_threshold(
        self, positions, *, quantile: float = 0.95, multiplier: float = 1.0
    ) -> float:
        """Absolute pointwise force-error budget: ``quantile`` of held-out ``expected_error``.

        The result is on the **absolute expected-force-error scale**, so it is only meaningful
        as a ``select_reruns(threshold=...)`` budget for absolute-scale trajectory scores
        (``expected_abs*`` / ``supervisor_abs*``). Do NOT compare it against a relative
        (``supervisor_rel*``) trajectory score -- the scales differ.
        """

        self._require_fitted()
        values = self.predict_uncertainty(positions).expected_error
        return _calibrate_risk_threshold(values, quantile=quantile, multiplier=multiplier)

    def calibrate_trajectory_risk_threshold(
        self,
        trajectories,
        *,
        scoring: str,
        quantile: float = 0.95,
        multiplier: float = 1.0,
        weights=None,
    ) -> float:
        """Trajectory-level risk budget: ``quantile`` of ``scoring`` over calibration orbits.

        Because it scores the calibration trajectories with the **same** ``scoring`` that will be
        screened, the budget is on the screened score's own scale -- so this is safe for BOTH
        relative and absolute scoring modes (unlike the pointwise budget).
        """

        self._require_fitted()
        scores = self.score_ensemble(trajectories, scoring=scoring, weights=weights)
        values = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
        return _calibrate_risk_threshold(values, quantile=quantile, multiplier=multiplier)

    def calibrate_risk_threshold(
        self,
        positions=None,
        *,
        scoring: str = "expected_abs_p95",
        quantile: float = 0.95,
        multiplier: float = 1.0,
        trajectories=None,
    ) -> float:
        """Backward-compatible wrapper around the two explicit calibration methods.

        Prefer :meth:`calibrate_pointwise_expected_error_threshold` (held-out positions) or
        :meth:`calibrate_trajectory_risk_threshold` (calibration trajectories) directly -- they
        make the threshold *scale* (pointwise expected-error vs trajectory score) explicit.
        """

        if (positions is None) == (trajectories is None):
            raise ValueError("provide exactly one of positions or trajectories")
        if trajectories is not None:
            return self.calibrate_trajectory_risk_threshold(
                trajectories, scoring=scoring, quantile=quantile, multiplier=multiplier
            )
        return self.calibrate_pointwise_expected_error_threshold(
            positions, quantile=quantile, multiplier=multiplier
        )

    # ------------------------------------------------------------------ calibration report
    def _calibration_arrays(self, positions, error) -> dict[str, torch.Tensor]:
        """Predictive arrays behind :meth:`evaluate_calibration`, exactly as served.

        ``_predict_covariance_block`` already applies the operational conformal scale (once);
        scaling again here would report ``c^2 * sigma`` / ``c^4 * cov`` instead of the served
        ``c * sigma`` / ``c^2 * cov``. The regression tests assert these arrays match
        :meth:`predict_uncertainty` / :meth:`predict_covariance_3x3` outputs, so the report can
        never drift from the serving path.
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        n = positions.shape[0]
        radius = torch.linalg.norm(positions, dim=-1)

        # One operator build per query chunk feeds BOTH the row-level prediction and the 3x3
        # covariance (the operator is the expensive part). Chunk row outputs come back in
        # [x_blk, y_blk, z_blk] order and are reassembled into the full-set
        # [x-all, y-all, z-all] row order via the (3, n_blk) reshape.
        mean_parts, std_parts, epi_parts, cov_parts, mean3_parts = [], [], [], [], []
        for a, b in self._query_chunks(n):
            pos_blk = positions[a:b]
            op_blk = self._operator(pos_blk)
            rows = self._predict_rows(op_blk, radius[a:b].repeat(3))
            nb = pos_blk.shape[0]
            mean_parts.append(rows["mean"].reshape(3, nb))
            epi_parts.append(rows["epistemic_variance"].reshape(3, nb))
            cov_blk = self._predict_covariance_block(pos_blk, operator=op_blk)
            std_parts.append(cov_blk.std_components.transpose(0, 1))
            cov_parts.append(cov_blk.covariance)
            mean3_parts.append(cov_blk.mean_error)
        return {
            "positions": positions,
            "radius": radius,
            "row_radii": radius.repeat(3),
            "mean": torch.cat(mean_parts, dim=1).reshape(-1),
            "std": torch.cat(std_parts, dim=1).reshape(-1),
            "epistemic_std": torch.sqrt(torch.cat(epi_parts, dim=1).reshape(-1).clamp_min(0.0)),
            "covariance": torch.cat(cov_parts, dim=0),
            "target": _flatten_acc(error),
            # vector (ellipsoid) calibration uses the full 3x3 predictive covariance per point and
            # the predictive RESIDUAL (observed error minus the posterior-mean error prediction).
            "residual_vec": error - torch.cat(mean3_parts, dim=0),
        }

    def evaluate_calibration(self, positions, error, *, altitude_bands: dict | None = None) -> dict:
        """Per-band calibration metrics (PICP, z_std, NLL, CRPS) for held-out error samples.

        This is Experiment 1: does the layer's nominal interval cover the held-out residuals,
        and does its uncertainty grow toward low altitude where the surrogate is overconfident?
        """

        arrays = self._calibration_arrays(positions, error)
        positions = arrays["positions"]
        radius = arrays["radius"]
        row_radii = arrays["row_radii"]
        mean = arrays["mean"]
        std = arrays["std"]
        epistemic_std = arrays["epistemic_std"]
        covariance = arrays["covariance"]
        target = arrays["target"]
        residual_vec = arrays["residual_vec"]
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
                    vector_calibration_metrics(residual_vec[point_mask], covariance[point_mask])
                )
                m.update(
                    component_calibration_metrics(
                        residual_vec[point_mask], covariance[point_mask], positions[point_mask]
                    )
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

        posterior = self._require_fitted()
        return source_diagnostics(
            source_positions=self.sources.positions,
            source_weights=self.sources.weights,
            shell_ids=self.sources.shell_ids,
            sigma=posterior.mean,
        )

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        """Serializable snapshot of the FITTED plugin.

        The payload is ``torch.load(..., weights_only=True)``-safe: nested dicts/lists of CPU
        tensors and Python primitives only. It captures everything prediction-relevant --
        source geometry, kernel/risk/domain options, the linear-Gaussian posterior, the
        altitude noise law, the domain-support calibration geometry, and ``fit_info`` -- so a
        loaded plugin predicts/scores identically without refitting. Use :meth:`save` /
        :meth:`load` for files.
        """

        posterior = self._require_fitted()

        def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.detach().cpu() if t is not None else None

        return {
            "format": PLUGIN_STATE_FORMAT,
            "version": PLUGIN_STATE_VERSION,
            "sources": {
                "positions": _cpu(self.sources.positions),
                "weights": _cpu(self.sources.weights),
                "shell_ids": _cpu(self.sources.shell_ids),
                "shell_radii": [float(r) for r in self.sources.shell_radii],
            },
            "options": {
                "eps": self.eps,
                "acceleration_sign": self.acceleration_sign,
                "source_chunk_size": self.source_chunk_size,
                "query_chunk_size": self.query_chunk_size,
                "reg_method": self.reg_method,
                "lambda_l2": self.lambda_l2,
                "noise_model": self.noise_model,
                "altitude_noise_bins": self.altitude_noise_bins,
                "covariance_mode": self.covariance_mode,
                "lowrank_rank": self.lowrank_rank,
                "val_fraction": self.val_fraction,
                "conformal_apply": self.conformal_apply,
                "conformal_alpha": self.conformal_alpha,
                "conformal_mode": self.conformal_mode,
                "conformal_by_band": self.conformal_by_band,
                "conformal_bands": dict(self.conformal_bands),
                "conformal_min_band_n": self.conformal_min_band_n,
                "conformal_by_region": self.conformal_by_region,
                "conformal_min_region_n": self.conformal_min_region_n,
                "low_altitude_radius": self.low_altitude_radius,
                "risk_scoring": self.risk_scoring,
                "sigma_threshold": self.sigma_threshold,
                "altitude_reference_h": self.altitude_reference_h,
                "calibrate_supervisor": self.calibrate_supervisor,
                "calibrated_supervisor_grid": json_safe(dict(self.calibrated_supervisor_grid)),
                "domain_support": self.domain_support,
                "domain_k": self.domain_k,
                "domain_weight": self.domain_weight,
                "domain_distance_weight": self.domain_distance_weight,
                "domain_radial_weight": self.domain_radial_weight,
                "domain_angular_weight": self.domain_angular_weight,
                "domain_backend": self.domain_backend,
                "seed": self.seed,
                "dtype": "float64" if self.dtype == torch.float64 else "float32",
            },
            "posterior": {
                "mean": _cpu(posterior.mean),
                "cov": _cpu(posterior.cov),
                "noise_var": float(posterior.noise_var),
                "lambda_l2": posterior.lambda_l2,
            },
            "altitude_noise": self._altitude_noise_state(_cpu),
            "domain_state": {
                "train_positions": _cpu(self.train_positions),
                "train_radii": _cpu(self.train_radii),
                "val_positions": _cpu(self.val_positions),
                "val_radii": _cpu(self.val_radii),
                "domain_scale": self._domain_scale,
                "domain_scale_k": self._domain_scale_k,
                "domain_angular_scale": self._domain_angular_scale,
            },
            "fit_info": dict(self.fit_info),
            "conformal_calibration": json_safe(self.conformal_calibration),
            "calibrated_supervisor": json_safe(self.calibrated_supervisor),
            # JSON-safe passthrough block (decision policy / provenance from the training run).
            "user_metadata": json_safe(dict(self.user_metadata)),
        }

    def save(self, path: str | Path, *, extra_metadata: dict | None = None) -> None:
        """Persist the fitted plugin to ``path`` (atomic write; conventional suffix ``.pt``).

        ``extra_metadata`` (JSON-safe dict) is merged into :attr:`user_metadata` before saving --
        the conventional place for the training run's decision policy (scoring mode, resolved
        threshold + provenance) and dataset provenance, so the model artifact is self-describing.
        """

        if extra_metadata:
            self.user_metadata = {**self.user_metadata, **json_safe(dict(extra_metadata))}
        atomic_torch_save(path, self.state_dict())

    @classmethod
    def from_state_dict(cls, state: dict, *, device: torch.device | str = "cpu") -> VESPUQPlugin:
        """Rebuild a fitted plugin from :meth:`state_dict` output (version-checked)."""

        if not isinstance(state, dict) or state.get("format") != PLUGIN_STATE_FORMAT:
            raise ValueError("not a serialized VESPUQPlugin state (missing format tag)")
        version = int(state.get("version", -1))
        if version < 1 or version > PLUGIN_STATE_VERSION:
            raise ValueError(
                f"unsupported plugin state version {version} (supported: 1..{PLUGIN_STATE_VERSION})"
            )

        options = dict(state["options"])
        dtype = torch.float64 if str(options.pop("dtype", "float64")) == "float64" else torch.float32
        src = state["sources"]
        sources = SourceSet(
            positions=src["positions"].to(dtype),
            weights=src["weights"].to(dtype),
            shell_ids=src["shell_ids"],
            shell_radii=tuple(float(r) for r in src["shell_radii"]),
        )
        plugin = cls(sources, dtype=dtype, device=device, **options)

        def _dev(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device=plugin.device, dtype=dtype) if t is not None else None

        def _required_tensor(t: torch.Tensor | None, name: str) -> torch.Tensor:
            value = _dev(t)
            if value is None:
                raise ValueError(f"serialized plugin state is missing posterior {name}")
            return value

        post = state["posterior"]
        plugin.posterior = LinearGaussianPosterior(
            mean=_required_tensor(post["mean"], "mean"),
            cov=_required_tensor(post["cov"], "cov"),
            noise_var=float(post["noise_var"]),
            lambda_l2=post["lambda_l2"],
        )
        noise = state.get("altitude_noise")
        if noise is None:
            plugin.altitude_noise = None
        else:
            noise_type = str(noise.get("type", "power_law")).lower()
            if noise_type in {"altitude_binned", "binned"}:
                plugin.altitude_noise = BinnedAltitudeNoiseModel(
                    radii=_required_tensor(noise.get("radii"), "altitude_noise.radii"),
                    log_variance=_required_tensor(noise.get("log_variance"), "altitude_noise.log_variance"),
                    h_floor=float(noise.get("h_floor", 1.0e-3)),
                )
            else:
                plugin.altitude_noise = AltitudeNoiseModel(
                    log_a=float(noise["log_a"]),
                    b=float(noise["b"]),
                    h_floor=float(noise["h_floor"]),
                )
        domain = state.get("domain_state", {})
        plugin.train_positions = _dev(domain.get("train_positions"))
        plugin.train_radii = _dev(domain.get("train_radii"))
        plugin.val_positions = _dev(domain.get("val_positions"))
        plugin.val_radii = _dev(domain.get("val_radii"))
        plugin._domain_scale = domain.get("domain_scale")
        plugin._domain_scale_k = domain.get("domain_scale_k")
        plugin._domain_angular_scale = domain.get("domain_angular_scale")
        plugin.fit_info = dict(state.get("fit_info", {}))
        plugin.conformal_calibration = state.get("conformal_calibration")
        plugin.calibrated_supervisor = state.get("calibrated_supervisor")
        plugin.user_metadata = dict(state.get("user_metadata", {}))
        return plugin

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> VESPUQPlugin:
        """Load a plugin saved by :meth:`save` (safe ``weights_only`` load, version-checked)."""

        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls.from_state_dict(state, device=device)
