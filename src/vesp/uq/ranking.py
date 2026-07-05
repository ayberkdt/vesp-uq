"""Shared average-rank helper: the single source of truth for tie-aware ranking.

Ranking underpins several unrelated statistics across the UQ package -- the tie-aware
Mann-Whitney AUROC identity (:func:`vesp.uq.benchmarking.detection_metrics`), every Spearman
correlation (:mod:`vesp.uq.selection`, :mod:`vesp.uq.plugin`, :mod:`vesp.uq.gate_diagnostics`,
:mod:`vesp.uq.altitude_controlled`), and the altitude-controlled rank diagnostics. Keeping the
algorithm here -- a leaf module that depends only on ``torch`` -- lets those higher-level modules
reuse it without importing each other (which would form cycles, e.g. ``benchmarking`` already
imports ``selection``).
"""

from __future__ import annotations

import torch


def average_ranks(values, *, base: float = 1.0) -> torch.Tensor:
    """Ascending average ranks (ties share their mean rank); dependency-free, deterministic.

    Fully vectorized: no per-tie-group Python loop, so it stays fast even on the 50k-trajectory
    resamples the significance bootstrap draws. ``base`` is the rank assigned to the smallest
    element -- ``1.0`` gives 1-based ranks (required by the tie-aware Mann-Whitney AUROC identity),
    ``0.0`` gives 0-based ranks (shift-invariant, harmless for Spearman/Pearson).
    """

    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    n = int(v.numel())
    # Allocate on ``v``'s device: every downstream index/scatter (``order``, ``counts``) lives there,
    # so a CPU-only scatter target would break for CUDA inputs (e.g. select_reruns on a cuda score).
    ranks = torch.empty(n, dtype=torch.float64, device=v.device)
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


__all__ = ["average_ranks"]
