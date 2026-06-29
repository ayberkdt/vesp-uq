"""Statistical significance for VESP-UQ benchmark comparisons (WP-A).

Every headline comparison in the benchmark suite (e.g. ``vespuq_supervisor`` vs ``min_altitude``)
ships with an uncertainty interval and a significance verdict, so "matches or modestly improves on
altitude" is a *tested* statement rather than an eyeball of overlapping mean +/- std.

Two complementary tests, neither inventing a number:

* :func:`paired_bootstrap_ci` resamples *trajectories* (with replacement) within one fitted layer
  and bootstraps the **difference** in a ranking/decision metric between two scores. Trajectory-level
  resampling gives power the 5-seed std cannot (the seeds share a single calibration fit).
* :func:`seed_paired_test` runs the Wilcoxon signed-rank test on the per-seed metric differences
  (exact for the small seed counts used here), so the across-seed agreement is also tested.

The metrics being compared are the same ones the suite already reports: Spearman (rank agreement),
capture rate at a budget, and AUROC for high-error detection. All target trajectory-level true
FORCE-model error -- never position error.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from vesp.uq.altitude_controlled import spearman
from vesp.uq.benchmarking import detection_metrics, evaluate_score_against_true_error

# A metric callable maps (scores, true_error) -> float (nan-safe).
Metric = Callable[[torch.Tensor, torch.Tensor], float]


def metric_spearman(scores, true_error) -> float:
    """Spearman rank correlation of a risk score with true force error."""

    return spearman(scores, true_error)


def metric_capture(fraction: float = 0.20) -> Metric:
    """Factory: capture rate (top-decile-error captured) at a fixed rerun ``fraction``."""

    def _metric(scores, true_error) -> float:
        return float(evaluate_score_against_true_error(
            scores, true_error, rerun_fraction=float(fraction))["capture_rate"])

    return _metric


def metric_auroc(high_quantile: float = 0.90) -> Metric:
    """Factory: AUROC for detecting the high-error trajectory class."""

    def _metric(scores, true_error) -> float:
        return float(detection_metrics(scores, true_error, high_quantile=high_quantile)["auroc"])

    return _metric


def resolve_metric(metric) -> Metric:
    """Resolve a metric name (``"spearman"`` / ``"capture"`` / ``"auroc"``) or callable to a callable."""

    if callable(metric):
        return metric
    if metric == "spearman":
        return metric_spearman
    if metric == "capture":
        return metric_capture()
    if metric == "auroc":
        return metric_auroc()
    raise ValueError(f"unknown metric {metric!r}; pass a callable or one of spearman/capture/auroc")


def paired_bootstrap_ci(
    scores_a, scores_b, true_error, metric="spearman", *,
    n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """Bootstrap CI + p-value for ``metric(a) - metric(b)`` resampling trajectories with replacement.

    Returns the full-sample ``delta``, the ``(1 - alpha)`` percentile confidence interval
    ``[ci_low, ci_high]``, a two-sided bootstrap ``p_value`` (twice the smaller tail mass crossing
    zero), and the verdict ``significant`` (CI excludes 0). ``delta > 0`` means score ``a`` beats
    score ``b`` on the metric. Determinstic given ``seed``.
    """

    a = torch.as_tensor(scores_a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(scores_b, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    n = int(a.numel())
    if not (b.numel() == n == e.numel()):
        raise ValueError("scores_a, scores_b and true_error must have equal length")
    if n < 2:
        raise ValueError("need at least 2 trajectories to bootstrap")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    fn = resolve_metric(metric)

    delta = fn(a, e) - fn(b, e)

    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, n, (int(n_boot), n), generator=gen)
    deltas: list[float] = []
    for r in range(int(n_boot)):
        sel = idx[r]
        d = fn(a[sel], e[sel]) - fn(b[sel], e[sel])
        if d == d and math.isfinite(d):  # skip nan/inf draws (degenerate resamples)
            deltas.append(float(d))

    if len(deltas) < 2:
        return {"metric": metric if isinstance(metric, str) else "custom", "delta": delta,
                "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"),
                "n_boot_effective": len(deltas), "significant": False}

    boot = torch.tensor(sorted(deltas), dtype=torch.float64)
    ci_low = float(torch.quantile(boot, alpha / 2.0))
    ci_high = float(torch.quantile(boot, 1.0 - alpha / 2.0))
    frac_le0 = float((boot <= 0.0).to(torch.float64).mean())
    frac_ge0 = float((boot >= 0.0).to(torch.float64).mean())
    p_value = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return {
        "metric": metric if isinstance(metric, str) else "custom",
        "delta": float(delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_boot_effective": len(deltas),
        "significant": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def _wilcoxon_signed_rank_exact(diffs: list[float]) -> dict:
    """Exact two-sided Wilcoxon signed-rank test on nonzero differences (enumerates 2**n signs).

    For the small seed counts here (n <= ~20) the exact null is cheap and avoids a scipy dependency
    and any normal-approximation error. Returns the signed-rank statistic ``W+`` and the exact
    two-sided p-value. Zeros are dropped (Wilcoxon convention).
    """

    nz = [d for d in diffs if d != 0.0 and math.isfinite(d)]
    n = len(nz)
    if n == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": 0, "method": "exact"}
    mags = sorted((abs(d), i) for i, d in enumerate(nz))
    # average ranks for tied magnitudes
    ranks = [0.0] * n
    j = 0
    while j < n:
        k = j + 1
        while k < n and mags[k][0] == mags[j][0]:
            k += 1
        avg = 0.5 * (j + k - 1) + 1.0
        for t in range(j, k):
            ranks[mags[t][1]] = avg
        j = k
    signs_pos = [r for r, d in zip(ranks, nz, strict=True) if d > 0.0]
    w_plus = sum(signs_pos)

    # exact null: each rank's sign is +/- with equal probability -> enumerate all 2**n assignments.
    rank_list = ranks
    achievable: dict[float, int] = {0.0: 1}
    for r in rank_list:
        nxt: dict[float, int] = {}
        for val, cnt in achievable.items():
            nxt[val] = nxt.get(val, 0) + cnt              # this rank negative
            nxt[val + r] = nxt.get(val + r, 0) + cnt      # this rank positive
        achievable = nxt
    total = 2 ** n
    total_rank = sum(rank_list)
    # two-sided: mass at statistics at least as extreme as W+ about the mean total_rank/2
    mean_w = total_rank / 2.0
    obs_dev = abs(w_plus - mean_w)
    extreme = sum(cnt for val, cnt in achievable.items() if abs(val - mean_w) >= obs_dev - 1e-9)
    p_value = min(1.0, extreme / total)
    return {"statistic": float(w_plus), "p_value": float(p_value), "n": n, "method": "exact"}


def seed_paired_test(per_seed_a, per_seed_b) -> dict:
    """Wilcoxon signed-rank test across seeds on the per-seed metric differences ``a - b``.

    ``per_seed_a`` / ``per_seed_b`` are the same metric (e.g. Spearman) for two scores, one value per
    seed. Non-finite pairs are dropped. Returns the mean difference, the exact two-sided p-value, the
    number of usable seeds, and ``a_wins`` (mean difference > 0). Complements the within-fit bootstrap.
    """

    a = list(per_seed_a)
    b = list(per_seed_b)
    if len(a) != len(b):
        raise ValueError("per_seed_a and per_seed_b must have equal length")
    diffs = []
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            continue
        try:
            dx, dy = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(dx) and math.isfinite(dy):
            diffs.append(dx - dy)
    if not diffs:
        return {"mean_delta": float("nan"), "p_value": float("nan"), "n_seeds": 0, "a_wins": False}
    test = _wilcoxon_signed_rank_exact(diffs)
    mean_delta = sum(diffs) / len(diffs)
    return {
        "mean_delta": float(mean_delta),
        "p_value": test["p_value"],
        "statistic": test["statistic"],
        "n_seeds": len(diffs),
        "a_wins": bool(mean_delta > 0.0),
    }
