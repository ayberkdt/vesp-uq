"""Force-risk follow-up selection: turn per-trajectory risk scores into a flagged subset.

Given one risk score per trajectory (from :mod:`vesp.uq.scoring`), :func:`select_reruns` flags
which trajectories are candidates for reference-data or higher-fidelity follow-up. Three policies:

- top-fraction (``rerun_fraction``, exact ``ceil(frac*n)`` top-k by default);
- absolute threshold (``threshold``, may legitimately flag *zero* in a safe regime);
- threshold + ``max_rerun_fraction`` cap.

When a ground-truth per-trajectory error is supplied, the report also validates the screen
(capture rate / precision / Spearman) -- but note this is a *diagnostic* comparison, not a claim
that the force-risk score predicts that error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from vesp.uq.ranking import average_ranks

from vesp.uq.scoring import _as_1d

# Fraction-mode selection policies for select_reruns.
FRACTION_POLICIES = ("topk", "quantile")


@dataclass
class RiskScreeningReport:
    """Outcome of selecting which trajectories are flagged for force-risk follow-up."""

    n_trajectories: int
    threshold: float
    rerun_fraction: float
    n_flagged: int
    flagged_indices: list[int]
    mean_risk_flagged: float | None = None
    mean_risk_accepted: float | None = None
    # Selection-policy bookkeeping (so the report says *why* a given count was flagged):
    selection_mode: str = "fraction"  # fraction | threshold | threshold+max_fraction
    fraction_policy: str | None = None  # topk | quantile (fraction mode only)
    requested_rerun_fraction: float | None = None
    n_requested: int | None = None
    n_ties_at_cutoff: int | None = None
    max_rerun_fraction: float | None = None
    n_above_threshold: int | None = None
    threshold_source: str | None = None  # manual | calibration_quantile
    threshold_quantile: float | None = None
    # Validation against a ground-truth error metric (only when ``true_error`` is supplied):
    capture_rate: float | None = None  # share of truly-high-error trajectories that got flagged
    precision: float | None = None  # share of flagged trajectories that were truly high-error
    spearman_risk_vs_error: float | None = None
    mean_error_flagged: float | None = None
    mean_error_accepted: float | None = None
    error_ratio_flagged_to_accepted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (no scipy), using average ranks for ties."""

    if a.numel() != b.numel():
        raise ValueError("Spearman inputs must have the same length")
    if b.device != a.device:
        b = b.to(device=a.device)
    if a.numel() < 2 or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        return float("nan")

    # base is irrelevant here: ranks only feed the shift-invariant Pearson correlation below.
    ra, rb = average_ranks(a, base=0.0), average_ranks(b, base=0.0)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if float(denom) <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def select_reruns(
    risk_scores,
    *,
    rerun_fraction: float | None = None,
    threshold: float | None = None,
    max_rerun_fraction: float | None = None,
    fraction_policy: str = "topk",
    true_error=None,
    true_error_quantile: float = 0.90,
    threshold_source: str | None = None,
    threshold_quantile: float | None = None,
) -> RiskScreeningReport:
    """Flag the riskiest trajectories for force-risk follow-up.

    Three selection policies:

    - ``rerun_fraction`` only: rerun the top fraction. ``fraction_policy="topk"`` (default) flags
      EXACTLY ``ceil(rerun_fraction * n)`` trajectories by stable top-k (robust to ties);
      ``"quantile"`` keeps the legacy quantile-threshold behavior (which can over-flag on ties).
    - ``threshold`` only: rerun every trajectory at or above an absolute risk threshold. If
      nothing exceeds it, *zero* trajectories are flagged -- a safe set may raise no alarms.
    - ``threshold`` + ``max_rerun_fraction``: take everything above the absolute threshold, but
      if that exceeds the budget ``ceil(max_rerun_fraction * n)``, keep only the top of them
      (at least 1 whenever anything is above the threshold).

    When ``true_error`` (one scalar per trajectory) is supplied, the report also validates the
    screen: ``capture_rate`` is the share of the truly-high-error trajectories (top
    ``1 - true_error_quantile``) that were flagged, and ``spearman_risk_vs_error`` measures
    monotonic agreement between risk and real error.
    """

    risk = _as_1d(risk_scores)
    n = int(risk.numel())
    if n == 0:
        raise ValueError("risk_scores is empty")
    if not bool(torch.isfinite(risk).all()):
        raise ValueError("risk_scores must contain only finite values")
    if fraction_policy not in FRACTION_POLICIES:
        raise ValueError(f"fraction_policy must be one of {FRACTION_POLICIES}, got {fraction_policy!r}")

    has_frac = rerun_fraction is not None
    has_thr = threshold is not None
    has_max = max_rerun_fraction is not None

    if has_max:
        if not has_thr:
            raise ValueError("max_rerun_fraction requires an absolute threshold")
        if has_frac:
            raise ValueError("combine max_rerun_fraction with threshold, not rerun_fraction")
        selection_mode = "threshold+max_fraction"
    elif has_frac and has_thr:
        raise ValueError(
            "provide exactly one of rerun_fraction or threshold (or threshold + max_rerun_fraction)"
        )
    elif has_frac:
        selection_mode = "fraction"
    elif has_thr:
        selection_mode = "threshold"
    else:
        raise ValueError("provide rerun_fraction, threshold, or threshold + max_rerun_fraction")

    order = torch.argsort(risk, descending=True, stable=True)  # stable top-k by risk
    n_above_threshold: int | None = None
    n_requested: int | None = None
    n_ties_at_cutoff: int | None = None
    applied_fraction_policy: str | None = None

    if selection_mode == "fraction":
        if rerun_fraction is None:
            raise AssertionError("fraction mode requires rerun_fraction")
        f = float(rerun_fraction)
        if not math.isfinite(f) or not 0.0 < f <= 1.0:
            raise ValueError("rerun_fraction must be in (0, 1]")
        applied_fraction_policy = fraction_policy
        k = n if f >= 1.0 else min(n, int(math.ceil(f * n)))
        n_requested = k
        if fraction_policy == "topk":
            flagged_mask = torch.zeros(n, dtype=torch.bool, device=risk.device)
            flagged_mask[order[:k]] = True
            cutoff = float(risk[order[k - 1]]) if k > 0 else float("inf")
            thr = cutoff
            n_ties_at_cutoff = int((risk == risk[order[k - 1]]).sum()) if k > 0 else 0
        else:  # quantile (legacy)
            thr = float(torch.quantile(risk, 1.0 - f))
            flagged_mask = risk >= thr
            n_ties_at_cutoff = int((risk == thr).sum())
    else:
        if threshold is None:
            raise AssertionError("threshold mode requires threshold")
        thr = float(threshold)
        if not math.isfinite(thr):
            raise ValueError("threshold must be finite")
        above_mask = risk >= thr
        n_above_threshold = int(above_mask.sum())
        if threshold_source is None:
            threshold_source = "manual"
        if selection_mode == "threshold":
            flagged_mask = above_mask
        else:  # threshold + max_fraction
            if max_rerun_fraction is None:
                raise AssertionError("threshold+max_fraction mode requires max_rerun_fraction")
            mf = float(max_rerun_fraction)
            if not math.isfinite(mf) or not 0.0 < mf <= 1.0:
                raise ValueError("max_rerun_fraction must be in (0, 1]")
            cap = int(math.ceil(mf * n))
            cap = max(1, cap) if n_above_threshold > 0 else 0
            n_requested = cap
            if n_above_threshold <= cap:
                flagged_mask = above_mask
            else:
                flagged_mask = torch.zeros(n, dtype=torch.bool, device=risk.device)
                flagged_mask[order[:cap]] = True

    accepted_mask = ~flagged_mask
    flagged_indices = [int(i) for i in torch.nonzero(flagged_mask, as_tuple=False).reshape(-1).tolist()]
    n_flagged = len(flagged_indices)

    report = RiskScreeningReport(
        n_trajectories=n,
        threshold=thr,
        rerun_fraction=n_flagged / n,
        n_flagged=n_flagged,
        flagged_indices=flagged_indices,
        mean_risk_flagged=float(risk[flagged_mask].mean()) if n_flagged > 0 else float("nan"),
        mean_risk_accepted=float(risk[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan"),
        selection_mode=selection_mode,
        fraction_policy=applied_fraction_policy,
        requested_rerun_fraction=float(rerun_fraction) if rerun_fraction is not None else None,
        n_requested=n_requested,
        n_ties_at_cutoff=n_ties_at_cutoff,
        max_rerun_fraction=float(max_rerun_fraction) if max_rerun_fraction is not None else None,
        n_above_threshold=n_above_threshold,
        threshold_source=threshold_source,
        threshold_quantile=threshold_quantile,
    )

    if true_error is not None:
        err = _as_1d(true_error).to(device=risk.device)
        if err.numel() != n:
            raise ValueError("true_error must have one value per trajectory")
        if not bool(torch.isfinite(err).all()):
            raise ValueError("true_error must contain only finite values")
        error_quantile = float(true_error_quantile)
        if not math.isfinite(error_quantile) or not 0.0 <= error_quantile <= 1.0:
            raise ValueError("true_error_quantile must be in [0, 1]")
        high_thr = float(torch.quantile(err, error_quantile))
        truly_high = err >= high_thr
        n_high = int(truly_high.sum())
        true_positive = int((flagged_mask & truly_high).sum())
        report.capture_rate = (true_positive / n_high) if n_high > 0 else float("nan")
        report.precision = (true_positive / n_flagged) if n_flagged > 0 else float("nan")
        report.spearman_risk_vs_error = _spearman(risk, err)
        report.mean_error_flagged = float(err[flagged_mask].mean()) if n_flagged > 0 else float("nan")
        report.mean_error_accepted = float(err[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan")
        if report.mean_error_accepted and report.mean_error_accepted > 0.0:
            report.error_ratio_flagged_to_accepted = report.mean_error_flagged / report.mean_error_accepted

    return report


def run_risk_screening(
    plugin,
    trajectories,
    *,
    true_error=None,
    rerun_fraction: float | None = 0.20,
    threshold: float | None = None,
    max_rerun_fraction: float | None = None,
    fraction_policy: str = "topk",
    scoring: str = "max",
    weights=None,
    threshold_source: str | None = None,
    threshold_quantile: float | None = None,
) -> dict:
    """Score a trajectory ensemble with ``plugin`` and select the force-risk follow-up subset.

    ``plugin`` is any object exposing ``score_ensemble(trajectories, scoring=..., weights=...)``
    (the :class:`~vesp.uq.plugin.VESPUQPlugin`). Returns a dict with:
      - ``trajectory_scores``: list of :class:`~vesp.uq.scoring.TrajectoryScore` (one per orbit),
      - ``selected_reruns``: indices flagged for force-risk follow-up,
      - ``risk_screening_report``: the :class:`RiskScreeningReport` (validated when
        ``true_error`` -- one scalar per trajectory -- is supplied).

    Selection follows :func:`select_reruns`: ``threshold`` (optionally with
    ``max_rerun_fraction``) takes precedence over ``rerun_fraction`` when supplied.
    """

    scores = plugin.score_ensemble(trajectories, scoring=scoring, weights=weights)
    risk = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
    if threshold is not None:
        report = select_reruns(
            risk,
            threshold=threshold,
            max_rerun_fraction=max_rerun_fraction,
            true_error=true_error,
            threshold_source=threshold_source,
            threshold_quantile=threshold_quantile,
        )
    else:
        report = select_reruns(
            risk,
            rerun_fraction=rerun_fraction,
            fraction_policy=fraction_policy,
            true_error=true_error,
        )
    return {
        "trajectory_scores": scores,
        "selected_reruns": report.flagged_indices,
        "risk_screening_report": report,
    }
