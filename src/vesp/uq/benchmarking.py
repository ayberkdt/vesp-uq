"""Compare trajectory-risk scores against trajectory-level true FORCE-MODEL error.

These utilities evaluate any per-trajectory risk score (a VESP-UQ supervisor score or a trivial
baseline from :mod:`vesp.uq.baselines`) against the same target: trajectory-level true
*force-model* error. The target is never position error.

Selection reuses :func:`vesp.uq.selection.select_reruns` (top-k fraction + capture/precision/
Spearman validation) so there is a single source of truth for the selection logic.
"""

from __future__ import annotations

import math

import torch

from vesp.uq.selection import select_reruns

# Per-baseline metric keys (stable column order for CSV / report tables).
METRIC_KEYS = (
    "n_trajectories",
    "rerun_fraction",
    "n_flagged",
    "spearman",
    "capture_rate",
    "precision",
    "lift_over_random",
    "mean_true_error_flagged",
    "mean_true_error_accepted",
    "force_error_ratio_flagged_to_accepted",
)


def evaluate_score_against_true_error(scores, true_error, rerun_fraction: float = 0.10,
                                      *, high_quantile: float = 0.90) -> dict:
    """Evaluate one risk score against trajectory-level true force error.

    Flags the top ``rerun_fraction`` of trajectories by ``scores`` (exact top-k, tie-robust) and
    reports the Spearman correlation, capture rate / precision against the truly-high-error
    trajectories (the top ``1 - high_quantile`` by true error), lift over a random screen, and the
    flagged-vs-accepted true-force-error split. Safe for small arrays and ties (constant scores yield
    a ``nan`` Spearman, never a crash).
    """

    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    if s.numel() == 0:
        raise ValueError("scores is empty")
    if s.numel() != e.numel():
        raise ValueError(f"scores ({s.numel()}) and true_error ({e.numel()}) must have equal length")

    report = select_reruns(s, rerun_fraction=float(rerun_fraction), true_error=e,
                           true_error_quantile=float(high_quantile))
    cap = report.capture_rate
    rf = report.rerun_fraction
    lift = (cap / rf) if (cap is not None and rf and not math.isnan(float(cap))) else float("nan")
    return {
        "n_trajectories": int(s.numel()),
        "rerun_fraction": rf,
        "n_flagged": report.n_flagged,
        "spearman": report.spearman_risk_vs_error,
        "capture_rate": report.capture_rate,
        "precision": report.precision,
        "lift_over_random": lift,
        "mean_true_error_flagged": report.mean_error_flagged,
        "mean_true_error_accepted": report.mean_error_accepted,
        "force_error_ratio_flagged_to_accepted": report.error_ratio_flagged_to_accepted,
    }


def compare_baselines(baseline_scores: dict, true_error, rerun_fraction: float = 0.10) -> dict:
    """Evaluate every named baseline score against the same true-force-error target.

    ``baseline_scores`` maps a baseline name to its per-trajectory score tensor. Returns a dict
    mapping each name to its metrics from :func:`evaluate_score_against_true_error`.
    """

    if not baseline_scores:
        raise ValueError("baseline_scores is empty")
    return {
        name: evaluate_score_against_true_error(scores, true_error, rerun_fraction=rerun_fraction)
        for name, scores in baseline_scores.items()
    }


def _best_by(results: dict, key: str) -> str | None:
    """Name of the baseline with the largest finite value of ``key`` (``None`` if all nan)."""

    best, best_val = None, -math.inf
    for name, m in results.items():
        v = m.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        if v > best_val:
            best, best_val = name, v
    return best


# --------------------------------------------------------------------------------------------- #
# Decision-quality metrics (WP-B): evaluate a risk score as the screening tool it is, rather than
# as a single correlation. Threshold-free detection (AUROC / AUPRC), a budget-integrated capture
# (capture-AUC), and the operationally meaningful loss vs the oracle ranking (normalized regret).
# All are computed against trajectory-level true FORCE-model error; none invents physical units.
# --------------------------------------------------------------------------------------------- #
DECISION_METRIC_KEYS = (
    "auroc",
    "auprc",
    "capture_auc",
    "capture_auc_normalized",
    "oracle_regret",
)


def average_ranks(values, *, base: float = 1.0) -> torch.Tensor:
    """Ascending average ranks (ties share their mean rank); dependency-free, deterministic.

    Fully vectorized: no per-tie-group Python loop, so it stays fast even on the 50k-trajectory
    resamples the significance bootstrap draws. ``base`` is the rank assigned to the smallest
    element -- ``1.0`` gives 1-based ranks (required by the tie-aware Mann-Whitney AUROC identity),
    ``0.0`` gives 0-based ranks (shift-invariant, harmless for Spearman/Pearson).
    """

    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    n = int(v.numel())
    ranks = torch.empty(n, dtype=torch.float64)
    if n == 0:
        return ranks
    order = torch.argsort(v, stable=True)
    sorted_v = v[order]
    # Collapse equal-value runs; a run spanning 0-based [start, end) gets mean position
    # 0.5*(start + end - 1), then shifted by ``base``.
    counts = torch.unique_consecutive(sorted_v, return_counts=True)[1]
    ends = torch.cumsum(counts, dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64)
    group_rank = 0.5 * (starts + ends - 1.0) + base
    ranks[order] = group_rank.repeat_interleave(counts)
    return ranks


def _avg_ranks(values: torch.Tensor) -> torch.Tensor:
    """1-based average ranks (kept as the internal name used by :func:`detection_metrics`)."""

    return average_ranks(values, base=1.0)


def _high_error_labels(true_error: torch.Tensor, high_quantile: float) -> torch.Tensor:
    """Boolean positive class: trajectories at or above the ``high_quantile`` of true error."""

    thr = float(torch.quantile(true_error, float(high_quantile)))
    return true_error >= thr


def detection_metrics(scores, true_error, high_quantile: float = 0.90) -> dict:
    """Threshold-free detection of high-error trajectories: AUROC + AUPRC (average precision).

    The positive class is the top ``1 - high_quantile`` of trajectories by true force error. AUROC
    is the tie-aware Mann-Whitney statistic (0.5 = chance, 1.0 = perfect); AUPRC is the average
    precision (its chance level is the positive-class prevalence). Both return ``nan`` if a class is
    empty (degenerate split), never a crash.
    """

    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    if s.numel() != e.numel():
        raise ValueError(f"scores ({s.numel()}) and true_error ({e.numel()}) must have equal length")
    if s.numel() == 0:
        raise ValueError("scores is empty")
    if not bool(torch.isfinite(s).all()) or not bool(torch.isfinite(e).all()):
        return {"auroc": float("nan"), "auprc": float("nan"), "high_quantile": float(high_quantile)}

    y = _high_error_labels(e, high_quantile)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "high_quantile": float(high_quantile)}

    ranks = _avg_ranks(s)
    auroc = (float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    order = torch.argsort(s, descending=True, stable=True)
    y_sorted = y[order].to(torch.float64)
    cum_tp = torch.cumsum(y_sorted, dim=0)
    precision_at_k = cum_tp / torch.arange(1, y_sorted.numel() + 1, dtype=torch.float64)
    auprc = float((precision_at_k * y_sorted).sum() / n_pos)
    return {"auroc": float(auroc), "auprc": auprc, "high_quantile": float(high_quantile)}


def _captured_error_mass(scores: torch.Tensor, true_error: torch.Tensor, fraction: float) -> float:
    """Sum of true error over the top ``fraction`` of trajectories ranked by ``scores``."""

    n = int(scores.numel())
    k = min(n, max(1, int(math.ceil(float(fraction) * n))))
    top = torch.argsort(scores, descending=True, stable=True)[:k]
    return float(true_error[top].sum())


def capture_auc(scores, true_error, fractions=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
                high_quantile: float = 0.90) -> dict:
    """Trapezoidal area under the capture-rate-vs-rerun-fraction curve (one number for all budgets).

    ``capture_auc`` integrates capture rate (share of top-decile-error trajectories flagged) over the
    rerun-fraction grid; ``capture_auc_normalized`` divides by the oracle's capture-AUC on the same
    grid, giving a budget-integrated efficiency in ``[0, 1]`` (1.0 = matches the oracle everywhere).
    """

    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    fracs = sorted({float(f) for f in fractions})
    if not fracs:
        raise ValueError("fractions is empty")

    def _auc(score_vec: torch.Tensor) -> float:
        caps = [evaluate_score_against_true_error(
            score_vec, e, rerun_fraction=f, high_quantile=high_quantile)["capture_rate"] for f in fracs]
        caps = [c if (c is not None and not math.isnan(float(c))) else float("nan") for c in caps]
        if any(math.isnan(c) for c in caps):
            return float("nan")
        area = 0.0
        for i in range(1, len(fracs)):
            area += 0.5 * (fracs[i] - fracs[i - 1]) * (caps[i] + caps[i - 1])
        return area

    raw = _auc(s)
    oracle = _auc(e)  # ranking by the true error itself is the optimal capture curve
    normalized = (raw / oracle) if (oracle and not math.isnan(oracle) and oracle > 0) else float("nan")
    return {"capture_auc": raw, "capture_auc_normalized": normalized,
            "fractions": fracs, "high_quantile": float(high_quantile)}


def oracle_regret(scores, true_error, fraction: float = 0.20) -> dict:
    """Normalized regret of the score's flagged set vs the oracle at a fixed rerun budget.

    At budget ``fraction`` the oracle flags the trajectories with the largest true error; a random
    screen captures ``fraction`` of the total error mass in expectation. ``oracle_regret`` is the
    captured-error gap to the oracle, normalized so 0.0 = oracle-optimal and 1.0 = no better than
    random (``nan`` when the oracle and random captures coincide). The raw captured masses are also
    returned for traceability.
    """

    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    if s.numel() != e.numel():
        raise ValueError("scores and true_error must have equal length")
    n = int(e.numel())
    k = min(n, max(1, int(math.ceil(float(fraction) * n)))) if n else 0
    score_mass = _captured_error_mass(s, e, fraction)
    oracle_mass = _captured_error_mass(e, e, fraction)
    # Expected mass a random screen captures over the SAME k flagged trajectories: (k / n), not the
    # requested fraction. They differ when fraction * n is non-integer (k = ceil rounds up), and using
    # the requested fraction there makes the normalization baseline inconsistent with the rounded
    # top-k that score/oracle actually flag -- which can push regret outside [0, 1] before clamping.
    random_mass = (k / n) * float(e.sum()) if n else 0.0
    denom = oracle_mass - random_mass
    regret = ((oracle_mass - score_mass) / denom) if abs(denom) > 0 else float("nan")
    if regret == regret:  # finite: clamp tiny numerical excursions outside [0, 1]
        regret = min(1.0, max(0.0, regret))
    return {"oracle_regret": regret, "fraction": float(fraction),
            "captured_mass_score": score_mass, "captured_mass_oracle": oracle_mass,
            "captured_mass_random": random_mass}


def decision_quality_metrics(scores, true_error, *, fractions, high_quantile: float = 0.90,
                             reference_fraction: float = 0.20) -> dict:
    """All WP-B decision-quality scalars for one risk score against true force error.

    Combines :func:`detection_metrics` (AUROC/AUPRC), :func:`capture_auc` (budget-integrated capture,
    raw + oracle-normalized), and :func:`oracle_regret` at ``reference_fraction`` (snapped to the
    nearest available budget in ``fractions``). The keys in :data:`DECISION_METRIC_KEYS` are always
    present so the suite can aggregate them mean +/- std across seeds.
    """

    fracs = sorted({float(f) for f in fractions})
    ref = min(fracs, key=lambda f: abs(f - float(reference_fraction))) if fracs else float(reference_fraction)
    det = detection_metrics(scores, true_error, high_quantile=high_quantile)
    cap = capture_auc(scores, true_error, fractions=fracs, high_quantile=high_quantile)
    reg = oracle_regret(scores, true_error, fraction=ref)
    return {
        "auroc": det["auroc"],
        "auprc": det["auprc"],
        "capture_auc": cap["capture_auc"],
        "capture_auc_normalized": cap["capture_auc_normalized"],
        "oracle_regret": reg["oracle_regret"],
        "high_quantile": float(high_quantile),
        "reference_fraction": ref,
    }
