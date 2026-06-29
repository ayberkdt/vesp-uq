"""Altitude-controlled incremental-value diagnostics for trajectory force-risk scores.

The trivial ``min_altitude`` baseline is strong because force-model error usually grows toward
low altitude. These diagnostics separate "the score is just altitude" from "the score adds
information beyond altitude" by holding altitude (min-radius) fixed three different ways:

* :func:`within_altitude_bin_ranking` -- Spearman(score, true force error) *inside* min-radius
  quantile bins, so cross-altitude trend cannot inflate the correlation.
* :func:`partial_correlations` -- linear partial correlation of each score with true force error
  after regressing out min-radius.
* :func:`matched_altitude_pairs` -- a paired sign test: among trajectories matched to within a
  small min-radius caliper, does the higher-scoring one carry the higher true force error?

Everything here is measured against trajectory-level true FORCE-MODEL error
(``a_reference - a_surrogate``), never position error. The functions are dependency-light (torch
only) and deterministic.
"""

from __future__ import annotations

import math

import torch

from vesp.uq.benchmarking import evaluate_score_against_true_error

__all__ = [
    "min_radius_scores",
    "spearman",
    "pearson",
    "partial_pearson_given_altitude",
    "within_altitude_bin_ranking",
    "partial_correlations",
    "matched_altitude_pairs",
]


def _as_float_tensor(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties shared); deterministic and dependency-free."""

    v = _as_float_tensor(values)
    order = torch.argsort(v, stable=True)
    sorted_v = v[order]
    ranks_sorted = torch.empty_like(sorted_v)
    n = int(v.numel())
    start = 0
    while start < n:
        end = start + 1
        while end < n and bool(sorted_v[end] == sorted_v[start]):
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def pearson(a, b) -> float:
    x = _as_float_tensor(a)
    y = _as_float_tensor(b)
    if x.numel() != y.numel():
        raise ValueError("correlation inputs must have the same length")
    if x.numel() < 2 or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.linalg.norm(xc) * torch.linalg.norm(yc)
    if float(denom) <= 0.0:
        return float("nan")
    return float((xc @ yc) / denom)


def spearman(a, b) -> float:
    return pearson(_rankdata(a), _rankdata(b))


def _residualize(y, x) -> torch.Tensor:
    """Residual of ``y`` after an OLS fit on ``x`` (used to partial out altitude)."""

    yy = _as_float_tensor(y)
    xx = _as_float_tensor(x)
    xc = xx - xx.mean()
    denom = float(xc @ xc)
    if denom <= 0.0:
        return yy - yy.mean()
    beta = float((xc @ (yy - yy.mean())) / denom)
    alpha = float(yy.mean()) - beta * float(xx.mean())
    return yy - (alpha + beta * xx)


def partial_pearson_given_altitude(score, true_error, min_radius) -> float:
    """Pearson(score, true_error | min_radius) -- linear partial correlation given altitude."""

    return pearson(_residualize(score, min_radius), _residualize(true_error, min_radius))


def min_radius_scores(trajectories) -> torch.Tensor:
    """Per-trajectory minimum radius (the altitude control variable)."""

    out = torch.empty(len(trajectories), dtype=torch.float64)
    for i, traj in enumerate(trajectories):
        out[i] = float(torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1).min())
    return out


def within_altitude_bin_ranking(
    scores: dict,
    true_error,
    min_radius,
    *,
    n_bins: int = 5,
    capture_fraction: float = 0.10,
) -> dict:
    """Per-score Spearman with true force error inside min-radius quantile bins.

    ``scores`` maps a method name -> per-trajectory score tensor. Bins are equal-count chunks of
    the min-radius order, so altitude is held roughly fixed within each bin and a positive
    within-bin Spearman means the score ranks force error *beyond* the altitude trend. ``capture``
    is the top-``capture_fraction`` capture rate of the high-error trajectories within the bin
    (computed only when the bin holds enough trajectories).
    """

    e = _as_float_tensor(true_error)
    r = _as_float_tensor(min_radius)
    order = torch.argsort(r)
    n_bins = min(max(1, int(n_bins)), int(order.numel()))
    chunks = torch.chunk(order, n_bins)

    bins = []
    per_method_weighted: dict[str, list[tuple[float, int]]] = {name: [] for name in scores}
    for b_idx, idx in enumerate(chunks):
        if idx.numel() < 3:
            continue
        bin_entry = {
            "bin": int(b_idx),
            "n": int(idx.numel()),
            "min_radius_low": float(r[idx].min()),
            "min_radius_high": float(r[idx].max()),
            "true_error_var": float(torch.var(e[idx])) if idx.numel() > 1 else float("nan"),
            "methods": {},
        }
        for name, score in scores.items():
            s = _as_float_tensor(score)
            rho = spearman(s[idx], e[idx])
            cap = float("nan")
            if int(idx.numel()) >= max(10, int(math.ceil(1.0 / capture_fraction))):
                cap = float(
                    evaluate_score_against_true_error(
                        s[idx], e[idx], rerun_fraction=capture_fraction
                    )["capture_rate"]
                )
            bin_entry["methods"][name] = {"spearman": rho, "capture": cap}
            if math.isfinite(rho):
                per_method_weighted[name].append((rho, int(idx.numel())))
        bins.append(bin_entry)

    weighted = {}
    for name, pairs in per_method_weighted.items():
        total = sum(w for _, w in pairs)
        weighted[name] = (sum(v * w for v, w in pairs) / total) if total else float("nan")

    return {
        "n_bins": n_bins,
        "capture_fraction": float(capture_fraction),
        "weighted_within_bin_spearman": weighted,
        "bins": bins,
    }


def partial_correlations(scores: dict, true_error, min_radius) -> dict:
    """Raw vs altitude-controlled correlation of each score with true force error.

    Returns, per method: ``pearson`` / ``spearman`` (raw) and ``partial_pearson_given_min_radius``
    (the linear correlation after regressing min-radius out of both the score and the true error).
    """

    e = _as_float_tensor(true_error)
    r = _as_float_tensor(min_radius)
    out = {}
    for name, score in scores.items():
        s = _as_float_tensor(score)
        out[name] = {
            "pearson": pearson(s, e),
            "spearman": spearman(s, e),
            "partial_pearson_given_min_radius": partial_pearson_given_altitude(s, e, r),
        }
    return out


def _normal_sf(z: float) -> float:
    """One-sided survival function of the standard normal (for the sign-test p-value)."""

    return 0.5 * math.erfc(z / math.sqrt(2.0))


def matched_altitude_pairs(
    score,
    true_error,
    min_radius,
    *,
    caliper: float | None = None,
    max_pairs: int | None = None,
) -> dict:
    """Matched-altitude paired sign test for incremental value beyond altitude.

    Trajectories are sorted by min-radius and greedily paired with their nearest unused neighbour
    whose min-radius gap is within ``caliper`` (default: 0.25 * std(min_radius)). For each pair the
    higher-scoring trajectory ("A") is compared to the lower-scoring one ("B"); the pair is
    *concordant* when ``true_error[A] > true_error[B]``. A concordance rate above 0.5 means that,
    among trajectories at essentially the same altitude, higher score tracks higher true force
    error -- i.e. the score carries altitude-independent ranking signal.

    Returns the concordance rate, the signed mean true-error gap (A minus B), a one-sided binomial
    sign-test p-value (normal approximation), and the per-pair rows.
    """

    s = _as_float_tensor(score)
    e = _as_float_tensor(true_error)
    r = _as_float_tensor(min_radius)
    n = int(s.numel())
    if not (e.numel() == n and r.numel() == n):
        raise ValueError("score, true_error, min_radius must have equal length")

    if caliper is None:
        std_r = float(torch.std(r)) if n > 1 else 0.0
        caliper = 0.25 * std_r if std_r > 0 else float("inf")
    caliper = float(caliper)

    order = torch.argsort(r).tolist()
    used = [False] * n
    rows = []
    concordant = 0
    tie = 0
    delta_sum = 0.0
    gap_sum = 0.0
    for pos, i in enumerate(order):
        if used[i]:
            continue
        # nearest unused successor in radius order (radius-sorted, so the next unused is nearest).
        j = None
        for k in range(pos + 1, len(order)):
            cand = order[k]
            if not used[cand]:
                j = cand
                break
        if j is None:
            break
        gap = abs(float(r[i] - r[j]))
        if gap > caliper:
            used[i] = True  # no admissible partner within the caliper; drop it
            continue
        used[i] = True
        used[j] = True
        a, b = (i, j) if float(s[i]) >= float(s[j]) else (j, i)  # A = higher score
        d_err = float(e[a] - e[b])
        if float(s[a]) == float(s[b]):
            tie += 1
        elif d_err > 0:
            concordant += 1
        delta_sum += d_err
        gap_sum += gap
        rows.append(
            {
                "traj_high_score": int(a),
                "traj_low_score": int(b),
                "min_radius_high_score": float(r[a]),
                "min_radius_low_score": float(r[b]),
                "min_radius_gap": gap,
                "score_high": float(s[a]),
                "score_low": float(s[b]),
                "true_error_high_score": float(e[a]),
                "true_error_low_score": float(e[b]),
                "delta_true_error": d_err,
                "concordant": int(d_err > 0),
            }
        )
        if max_pairs is not None and len(rows) >= int(max_pairs):
            break

    n_pairs = len(rows)
    n_decisive = n_pairs - tie
    concordance_rate = (concordant / n_decisive) if n_decisive > 0 else float("nan")
    mean_delta = (delta_sum / n_pairs) if n_pairs else float("nan")
    mean_gap = (gap_sum / n_pairs) if n_pairs else float("nan")
    # One-sided sign-test p-value: P(X >= concordant) under Binom(n_decisive, 0.5), normal approx.
    if n_decisive > 0:
        mu = 0.5 * n_decisive
        sigma = math.sqrt(0.25 * n_decisive)
        z = (concordant - 0.5 - mu) / sigma if sigma > 0 else 0.0  # continuity correction
        p_value = _normal_sf(z)
    else:
        p_value = float("nan")

    return {
        "caliper": caliper,
        "n_pairs": n_pairs,
        "n_decisive_pairs": n_decisive,
        "n_tied_score_pairs": tie,
        "n_concordant": concordant,
        "concordance_rate": concordance_rate,
        "mean_delta_true_error": mean_delta,
        "mean_min_radius_gap": mean_gap,
        "sign_test_p_value_one_sided": p_value,
        "rows": rows,
    }
