"""Validation-tuned supervisor exponents for VESP-UQ (Design A).

The hand-set supervisor combines three physical per-point components with FIXED unit weights:

    point_risk = expected_error * rel_altitude_weight * (1 + domain_weight * domain_risk)
    supervisor_rel_p95 = p95( point_risk )            # over a trajectory's output points

The expanded-baseline and score-ablation studies showed those fixed weights leave ranking
performance on the table. This module keeps the *physical multiplicative form* but learns three
non-negative exponents on a held-out split:

    point_risk(beta) = expected_error^b1 * rel_altitude_weight^b2 * (1 + b3 * domain_risk)

``beta = (1, 1, 1)`` reproduces the current supervisor exactly (with the default ``domain_weight``),
so this is a strict generalization, not a different score. Tuning is done on a validation split and
reported on a disjoint test split; nothing is fit on the test labels. It improves ranking only where
the residual carries altitude-independent signal -- it cannot beat altitude where altitude dominates.

Implemented as an external re-scorer over the plugin's per-point outputs (like the other study
modules), so the core scoring path and the default supervisor are untouched.
"""

from __future__ import annotations

import torch

from vesp.uq.altitude_controlled import spearman

H_FLOOR = 1.0e-3  # matches scoring._altitude_weight's default floor
_B12_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
_B3_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)

__all__ = [
    "supervisor_components",
    "apply_learned_supervisor",
    "fit_learned_supervisor",
    "DEFAULT_BETAS",
]

DEFAULT_BETAS = (1.0, 1.0, 1.0)  # reproduces the hand-set supervisor


def supervisor_components(plugin, trajectories) -> dict:
    """Per-point supervisor components stacked as ``(n_traj, n_points)`` tensors.

    Returns ``expected_error``, the per-trajectory *relative* altitude weight (rescaled by each
    trajectory's own median, exactly as the supervisor), and the domain-support (OOD) risk. Requires
    a uniform output-point count across trajectories (true for generated / screening ensembles).
    """

    prepped = [plugin._prep_positions(t) for t in trajectories]
    if not prepped:
        empty = torch.empty(0, 0, dtype=torch.float64)
        return {"expected_error": empty, "rel_alt": empty, "domain_risk": empty, "n_points": 0}
    lengths = {int(t.shape[0]) for t in prepped}
    if len(lengths) != 1:
        raise ValueError("learned supervisor requires a uniform output-point count per trajectory")
    n_points = lengths.pop()
    n_traj = len(prepped)

    pts = torch.cat(prepped, dim=0)
    pred = plugin.predict_uncertainty(pts)
    dr_all = (plugin.domain_support_score(pred.positions) if getattr(plugin, "domain_support", False)
              else torch.zeros(pts.shape[0], dtype=torch.float64))

    ee = pred.expected_error.to(torch.float64).reshape(n_traj, n_points)
    radius = pred.radius.to(torch.float64).reshape(n_traj, n_points)
    dr = dr_all.to(torch.float64).reshape(n_traj, n_points)

    alt = 1.0 / (radius - 1.0).clamp_min(H_FLOOR)               # _altitude_weight
    med = torch.median(alt, dim=1, keepdim=True).values.clamp_min(torch.finfo(alt.dtype).tiny)
    rel_alt = alt / med                                         # _relative_altitude_weight (per row)
    return {"expected_error": ee, "rel_alt": rel_alt, "domain_risk": dr, "n_points": n_points}


def apply_learned_supervisor(components: dict, betas) -> torch.Tensor:
    """Per-trajectory ``supervisor_rel_p95`` score under exponents ``betas = (b1, b2, b3)``."""

    b1, b2, b3 = (float(b) for b in betas)
    ee = components["expected_error"].clamp_min(0.0)
    rel = components["rel_alt"].clamp_min(0.0)
    dr = components["domain_risk"]
    point_risk = (ee ** b1) * (rel ** b2) * (1.0 + b3 * dr)
    if point_risk.numel() == 0:
        return torch.empty(0, dtype=torch.float64)
    return torch.quantile(point_risk, 0.95, dim=1)


def fit_learned_supervisor(components: dict, true_error, *, b12_grid=_B12_GRID, b3_grid=_B3_GRID) -> dict:
    """Grid-fit non-negative exponents maximizing Spearman(score, true force error).

    Spearman is rank-based (non-differentiable), so a deterministic grid search is used. The grid
    includes ``(1, 1, 1)`` so the learned supervisor never scores worse than the hand-set one on the
    fit split. Returns the chosen ``betas``, the fit-split Spearman, and the baseline (1,1,1) value.
    """

    te = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    best_betas = DEFAULT_BETAS
    best_rho = spearman(apply_learned_supervisor(components, DEFAULT_BETAS), te)
    baseline_rho = best_rho
    for b1 in b12_grid:
        for b2 in b12_grid:
            for b3 in b3_grid:
                rho = spearman(apply_learned_supervisor(components, (b1, b2, b3)), te)
                if rho == rho and rho > best_rho:  # skip NaN
                    best_rho, best_betas = rho, (float(b1), float(b2), float(b3))
    return {
        "betas": list(best_betas),
        "fit_spearman": best_rho,
        "baseline_spearman": baseline_rho,
        "improves_on_fit_split": bool(best_rho > baseline_rho + 1e-9),
    }
