"""Simple baseline trajectory-risk selectors for force-error screening comparison.

Each function returns ONE scalar score per trajectory, where a HIGHER score always means HIGHER
risk. The target these scores are compared against is always trajectory-level true *force-model*
error (never position error) -- see :mod:`vesp.uq.benchmarking` and
``scripts/compare_risk_baselines.py``.

The baselines are intentionally cheap so a VESP-UQ supervisor score can be judged against trivial
heuristics on the same target. Most are torch-only; the optional kNN helper uses scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_H_FLOOR = 1.0e-3


def _as_traj_list(trajectories) -> list[torch.Tensor]:
    trajs = list(trajectories)
    if not trajs:
        raise ValueError("trajectories is empty")
    for t in trajs:
        tt = torch.as_tensor(t)
        if tt.ndim != 2 or tt.shape[-1] != 3:
            raise ValueError("each trajectory must have shape (T, 3)")
    return trajs


def random_scores(n: int, seed: int = 0) -> torch.Tensor:
    """Deterministic uniform random risk scores in ``[0, 1)`` -- the chance-level reference."""

    if int(n) <= 0:
        raise ValueError("n must be positive")
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(int(n), generator=g, dtype=torch.float64)


# A negative-control random score must be INDEPENDENT of the trajectory ensemble. Ensembles are
# generated from the same integer ``seed``; calling ``random_scores(seed)`` would replay the exact
# torch RNG stream the ensemble drew its periapsides from, making `random` a perfect (anti-)altitude
# ranker instead of a chance control. Offsetting the seed by a large prime decouples the streams.
PLACEBO_SEED_OFFSET = 1_000_003


def independent_random_scores(n: int, seed: int = 0) -> torch.Tensor:
    """Chance-level random scores whose seed is decoupled from the trajectory-ensemble seed.

    Use this (not :func:`random_scores` with the run seed) for the `random` negative control in any
    study that also generates a seeded trajectory ensemble, so the placebo is genuinely at chance and
    not silently aligned with the ensemble geometry (see :data:`PLACEBO_SEED_OFFSET`).
    """

    return random_scores(int(n), seed=int(seed) + PLACEBO_SEED_OFFSET)


def label_shuffled_scores(true_error, seed: int = 0) -> torch.Tensor:
    """Negative control (G4): the true-error values permuted by a seeded RNG.

    This placebo has the *right marginal distribution* of scores but no alignment to the
    trajectories, so it must score at chance (capture ~ rerun fraction, Spearman ~ 0). Unlike
    :func:`random_scores` it shares the true error's heavy-tailed shape, making it a tighter
    upper-bound-at-chance control: any continuous leakage or fabrication that aligns scores with the
    oracle would push this placebo above chance and trip the suite's placebo assertion.

    It deliberately consumes the oracle ``true_error`` -- it is a control, not a competing method --
    so it must only ever be built *outside* a selection / scoring region (the suite constructs it
    after the guarded score-assembly block, never inside it).
    """

    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    if e.numel() == 0:
        raise ValueError("true_error is empty")
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(e.numel(), generator=g)
    return e[perm].clone()


def min_altitude_scores(trajectories) -> torch.Tensor:
    """Minimum-altitude heuristic: lower periapsis -> higher risk.

    ``score = 1 / max(min_radius - 1, h_floor)`` (monotonically larger as the lowest point of the
    trajectory approaches the surface). A strong, trivial baseline because force-model error
    typically grows toward low altitude.
    """

    trajs = _as_traj_list(trajectories)
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        r = torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1)
        out[i] = 1.0 / max(float(r.min()) - 1.0, _H_FLOOR)
    return out


def low_altitude_exposure_scores(
    trajectories, low_altitude_radius: float = 1.15, weights=None
) -> torch.Tensor:
    """Low-altitude exposure heuristic: the (weighted) fraction of points below ``low_altitude_radius``.

    ``0`` means the trajectory never dips below the threshold, ``1`` means it is entirely below it.
    ``weights`` (optional) is an iterable of per-trajectory weight vectors (e.g. ~dt) aligned with
    ``trajectories``; ``None`` weights all points uniformly.
    """

    trajs = _as_traj_list(trajectories)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(trajs):
            raise ValueError("weights must be None or one weight vector per trajectory")
    thr = float(low_altitude_radius)
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        r = torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1)
        below = (r <= thr).to(torch.float64)
        if weights is None or weights[i] is None:
            out[i] = float(below.mean())
        else:
            w = torch.as_tensor(weights[i], dtype=torch.float64).reshape(-1)
            if w.numel() != below.numel():
                raise ValueError("weight vector length must match the trajectory length")
            total = float(w.sum())
            if total <= 0.0:
                raise ValueError("weights must sum to a positive value")
            out[i] = float((below * (w / total)).sum())
    return out


def _weighted_quantile(x: torch.Tensor, q: float, weights: torch.Tensor | None = None) -> float:
    if x.numel() == 0:
        return float("nan")
    if weights is None:
        return float(torch.quantile(x, float(q)))
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if w.numel() != x.numel():
        raise ValueError("weight vector length must match the trajectory length")
    if bool((w < 0).any()):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    order = torch.argsort(x)
    xs = x[order]
    ws = (w / total)[order]
    cum = torch.cumsum(ws, dim=0)
    idx = int(torch.searchsorted(cum, torch.tensor(float(q), dtype=torch.float64)))
    return float(xs[min(idx, xs.numel() - 1)])


def _aggregate_profile(values: torch.Tensor, aggregator: str, weights=None) -> float:
    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if aggregator == "max":
        return float(v.max())
    if aggregator == "mean":
        if weights is None:
            return float(v.mean())
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if w.numel() != v.numel():
            raise ValueError("weight vector length must match the trajectory length")
        if bool((w < 0).any()):
            raise ValueError("weights must be nonnegative")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return float((v * (w / total)).sum())
    if aggregator == "p95":
        return _weighted_quantile(v, 0.95, weights)
    raise ValueError("aggregator must be 'mean', 'p95', or 'max'")


def domain_support_scores(plugin, trajectories, weights=None) -> torch.Tensor:
    """Domain-support-only risk: (weighted) mean per-point out-of-support score along each trajectory.

    Requires a fitted ``plugin`` exposing ``domain_support_score(positions)`` (raises a clear error
    otherwise). Higher = more of the trajectory lies outside the calibration support. ``weights``
    follows the same per-trajectory convention as ``score_ensemble``.
    """

    if not hasattr(plugin, "domain_support_score"):
        raise ValueError("plugin does not expose domain_support_score(...)")
    trajs = _as_traj_list(trajectories)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(trajs):
            raise ValueError("weights must be None or one weight vector per trajectory")
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        ds = plugin.domain_support_score(traj)  # per-point; raises if the plugin is not fitted
        out[i] = _aggregate_profile(
            torch.as_tensor(ds, dtype=torch.float64),
            "mean",
            None if weights is None else weights[i],
        )
    return out


def vespuq_scores(plugin, trajectories, scoring: str, weights=None) -> torch.Tensor:
    """VESP-UQ trajectory risk scores for a given ``scoring`` mode (one scalar per trajectory).

    ``scoring`` must be a supported VESP-UQ mode (e.g. ``mean`` for an uncertainty-only baseline,
    ``supervisor_rel_p95`` for the full supervisor); unsupported modes raise a clear ``ValueError``.
    """

    from vesp.uq.integrity.split_guard import forbid_oracle
    from vesp.uq.scoring import SCORING_FUNCTIONS

    forbid_oracle(trajectories, weights)  # G2: a risk score must never consume the true-error oracle
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"unsupported scoring {scoring!r}; must be one of {SCORING_FUNCTIONS}")
    if not hasattr(plugin, "score_ensemble"):
        raise ValueError("plugin does not expose score_ensemble(...)")
    scores = plugin.score_ensemble(list(trajectories), scoring=scoring, weights=weights)
    return torch.tensor([s.risk_score for s in scores], dtype=torch.float64)


@dataclass(frozen=True)
class AltitudeExpectedCurve:
    """Piecewise-linear altitude-only expected-error baseline.

    The curve is fit from VESP-UQ's own expected-error profile on calibration positions, then used
    to ask a narrower question: does a trajectory have more expected force error than altitude
    alone would predict?
    """

    radii: torch.Tensor
    expected_error: torch.Tensor

    def predict(self, radius: torch.Tensor) -> torch.Tensor:
        r = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
        x = self.radii.to(dtype=torch.float64)
        y = self.expected_error.to(dtype=torch.float64)
        if x.numel() == 1:
            return torch.full_like(r, float(y[0]))

        idx = torch.searchsorted(x, r)
        right = idx.clamp(1, x.numel() - 1)
        left = right - 1
        denom = (x[right] - x[left]).clamp_min(torch.finfo(torch.float64).tiny)
        t = (r - x[left]) / denom
        out = y[left] + t * (y[right] - y[left])
        out = torch.where(r <= x[0], y[0], out)
        out = torch.where(r >= x[-1], y[-1], out)
        return out.clamp_min(torch.finfo(torch.float64).tiny)


def fit_altitude_expected_curve(
    plugin,
    calibration_positions: torch.Tensor,
    *,
    n_bins: int = 8,
    statistic: str = "median",
) -> AltitudeExpectedCurve:
    """Fit an altitude-only expected-error curve from calibration positions.

    ``statistic`` is ``median`` by default so isolated noisy calibration points do not define the
    whole altitude trend. The fitted curve is intentionally simple and monotonicity-free: it is a
    diagnostic baseline, not a new learned uncertainty model.
    """

    if int(n_bins) <= 0:
        raise ValueError("n_bins must be positive")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    pos = torch.as_tensor(calibration_positions, dtype=torch.float64).detach().cpu()
    if pos.ndim != 2 or pos.shape[-1] != 3 or pos.shape[0] == 0:
        raise ValueError("calibration_positions must have shape (N, 3) with N > 0")
    pred = plugin.predict_uncertainty(pos)
    radius = torch.linalg.norm(pos, dim=-1)
    expected = torch.as_tensor(pred.expected_error.detach().cpu(), dtype=torch.float64)
    order = torch.argsort(radius)
    chunks = torch.chunk(order, min(int(n_bins), int(order.numel())))
    centers: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for idx in chunks:
        if idx.numel() == 0:
            continue
        r_bin = radius[idx]
        e_bin = expected[idx]
        centers.append(torch.median(r_bin))
        values.append(torch.median(e_bin) if statistic == "median" else e_bin.mean())
    if not centers:
        raise ValueError("could not fit an altitude expected-error curve")
    x = torch.stack(centers).to(dtype=torch.float64)
    y = torch.stack(values).to(dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    order2 = torch.argsort(x)
    return AltitudeExpectedCurve(radii=x[order2], expected_error=y[order2])


def altitude_residual_expected_scores(
    plugin,
    trajectories,
    *,
    calibration_positions: torch.Tensor | None = None,
    curve: AltitudeExpectedCurve | None = None,
    mode: str = "ratio",
    aggregator: str = "p95",
    n_bins: int = 8,
    weights=None,
) -> torch.Tensor:
    """Expected-error score after removing the altitude-only expected-error trend.

    ``mode="ratio"`` scores ``expected_error / f_alt(radius)``; ``mode="delta"`` scores
    ``expected_error - f_alt(radius)``. In both cases larger means the trajectory is riskier than
    altitude alone would suggest. ``aggregator`` is the trajectory reduction (``mean``/``p95``/``max``);
    optional ``weights`` are per-trajectory time weights.
    """

    if mode not in {"ratio", "delta"}:
        raise ValueError("mode must be 'ratio' or 'delta'")
    if curve is None:
        if calibration_positions is None:
            calibration_positions = getattr(plugin, "val_positions", None)
        if calibration_positions is None:
            calibration_positions = getattr(plugin, "train_positions", None)
        if calibration_positions is None:
            raise ValueError("calibration_positions are required when the plugin has no stored fit geometry")
        curve = fit_altitude_expected_curve(plugin, calibration_positions, n_bins=n_bins)

    trajs = _as_traj_list(trajectories)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(trajs):
            raise ValueError("weights must be None or one weight vector per trajectory")
    lengths = [int(torch.as_tensor(t).shape[0]) for t in trajs]
    all_positions = torch.cat([torch.as_tensor(t, dtype=torch.float64) for t in trajs], dim=0)
    pred = plugin.predict_uncertainty(all_positions)
    base = curve.predict(pred.radius.detach().cpu())
    expected = torch.as_tensor(pred.expected_error.detach().cpu(), dtype=torch.float64)
    profile = expected / base if mode == "ratio" else expected - base

    out = torch.empty(len(trajs), dtype=torch.float64)
    offset = 0
    for i, n in enumerate(lengths):
        out[i] = _aggregate_profile(
            profile[offset : offset + n],
            aggregator,
            None if weights is None else weights[i],
        )
        offset += n
    return out


def knn_p95_scores(train_positions: torch.Tensor, trajectories, k: int = 8) -> torch.Tensor:
    """kNN/OOD-distance-only heuristic: 95th percentile of k-NN distance.

    Fits a nearest-neighbor model on `train_positions` (calibration set). For each trajectory,
    computes the Euclidean distance to the k-nearest calibration points for every point,
    averages over the k neighbors, and then takes the 95th percentile over the trajectory.
    """

    trajs = _as_traj_list(trajectories)
    out = torch.empty(len(trajs), dtype=torch.float64)
    train_pos = torch.as_tensor(train_positions, dtype=torch.float64).numpy()

    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(train_pos)

    for i, traj in enumerate(trajs):
        t = torch.as_tensor(traj, dtype=torch.float64).numpy()
        dist, _ = tree.query(t, k=k)
        mean_knn = dist.mean(axis=-1)
        p95 = np.percentile(mean_knn, 95)
        out[i] = float(p95)

    return out
