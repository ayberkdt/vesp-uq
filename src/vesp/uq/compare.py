"""VESP-UQ Model Comparison and Drift reporting.

Compares two fitted VESPUQPlugin instances side-by-side to assess drift
and promotion readiness.
"""

from __future__ import annotations

import torch

from vesp.uq.plugin import VESPUQPlugin


def compare_models(
    plugin_a: VESPUQPlugin,
    plugin_b: VESPUQPlugin,
    held_out_positions: torch.Tensor,
    held_out_error: torch.Tensor,
    trajectory_ensemble: list[torch.Tensor] | None = None,
    altitude_bands: dict[str, list[float]] | None = None,
    scoring_mode: str = "supervisor_rel",
    threshold_policy: dict | None = None,
) -> dict:
    """Compare two fitted models on posterior distance, calibration, and screening agreement.

    ``trajectory_ensemble`` is a list of ``(T_i, 3)`` position tensors (the same shape
    :meth:`VESPUQPlugin.score_ensemble` consumes).
    """

    plugin_a._require_fitted()
    plugin_b._require_fitted()

    # Posterior distance
    mean_a = plugin_a.posterior.mean
    mean_b = plugin_b.posterior.mean
    mean_l2_diff = float(torch.linalg.norm(mean_a - mean_b))

    cov_a = plugin_a.posterior.cov
    cov_b = plugin_b.posterior.cov
    # Tr(Cov_a + Cov_b - 2(Cov_a Cov_b)^0.5) is Frechet distance, but a simple trace diff
    # or frobenius norm of difference is sufficient for a drift diagnostic summary.
    cov_frob_diff = float(torch.linalg.norm(cov_a - cov_b))

    noise_var_a = plugin_a.posterior.noise_var
    noise_var_b = plugin_b.posterior.noise_var
    noise_var_delta = float(noise_var_b - noise_var_a)

    # Domain support drift
    # Evaluate B's domain support on A's training points
    domain_shift_mean = 0.0
    domain_shift_max = 0.0
    if plugin_b.domain_support and plugin_a.train_positions is not None:
        b_support = plugin_b.domain_support_components(plugin_a.train_positions)
        domain_shift_mean = float(b_support["total_score"].mean())
        domain_shift_max = float(b_support["total_score"].max())

    # Calibration side-by-side
    cal_a = plugin_a.evaluate_calibration(held_out_positions, held_out_error, altitude_bands=altitude_bands)
    cal_b = plugin_b.evaluate_calibration(held_out_positions, held_out_error, altitude_bands=altitude_bands)

    # Restructure calibration as a combined dictionary for easy markdown rendering. The
    # calibration report also carries scalar summary keys (e.g. low_high_epistemic_std_ratio);
    # only the per-band dict entries are band rows.
    calibration_comparison = {}
    for band, metrics_a in cal_a.items():
        metrics_b = cal_b.get(band)
        if not isinstance(metrics_a, dict) or not isinstance(metrics_b, dict):
            continue
        calibration_comparison[band] = {
            "rmse": {"A": metrics_a["rmse"], "B": metrics_b["rmse"]},
            "mean_pred_std": {"A": metrics_a["mean_pred_std"], "B": metrics_b["mean_pred_std"]},
            "picp_90": {"A": metrics_a["picp_90"], "B": metrics_b["picp_90"]},
        }

    # Screening agreement
    agreement = {}
    if trajectory_ensemble:
        from vesp.uq.selection import _spearman, select_reruns

        # Score ensemble with both models
        res_a = plugin_a.score_ensemble(
            trajectory_ensemble, scoring=scoring_mode
        )
        res_b = plugin_b.score_ensemble(
            trajectory_ensemble, scoring=scoring_mode
        )

        scores_a = [r.risk_score for r in res_a]
        scores_b = [r.risk_score for r in res_b]

        # Risk Spearman (the same dependency-free rank correlation the selection layer reports)
        spearman = _spearman(
            torch.tensor(scores_a, dtype=torch.float64), torch.tensor(scores_b, dtype=torch.float64)
        )

        # Flag overlap
        policy_kwargs = threshold_policy or {"rerun_fraction": 0.2}
        if "type" in policy_kwargs:
            policy_kwargs = {"rerun_fraction": policy_kwargs.get("fraction", 0.2)}

        report_a = select_reruns(scores_a, **policy_kwargs)
        report_b = select_reruns(scores_b, **policy_kwargs)

        set_a = set(report_a.flagged_indices)
        set_b = set(report_b.flagged_indices)

        if len(set_a) == 0 and len(set_b) == 0:
            overlap = 1.0
        elif len(set_a) == 0 or len(set_b) == 0:
            overlap = 0.0
        else:
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            overlap = intersection / union

        agreement = {
            "risk_spearman": spearman,
            "flag_overlap": overlap,
            "n_flagged_A": len(set_a),
            "n_flagged_B": len(set_b),
        }

    return {
        "posterior_distance": {
            "mean_l2_diff": mean_l2_diff,
            "cov_frob_diff": cov_frob_diff,
            "noise_var_delta": noise_var_delta,
        },
        "domain_shift": {
            "mean_score_on_A": domain_shift_mean,
            "max_score_on_A": domain_shift_max,
        },
        "calibration": calibration_comparison,
        "screening_agreement": agreement,
    }
