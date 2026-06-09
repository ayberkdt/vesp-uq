"""Sentinel auditing of accepted low-risk trajectories for force-error false negatives.

When a fixed rerun budget reruns only the high-risk trajectories at higher fidelity, the *accepted*
(low-risk) set is taken on trust. But VESP-UQ is a fitted risk model, so some genuinely high
force-error trajectories may slip into the accepted set -- false negatives. This module draws a
small, deterministic random *sentinel* sample from the accepted set so those false negatives can be
estimated empirically rather than assumed away.

Everything here is measured against **true force-model error** (``a_reference - a_surrogate``),
never trajectory position accuracy. A "false negative" is an accepted trajectory whose true force
error is in the high-error tail.
"""

from __future__ import annotations

import math

import torch


def _validate_fraction(value: float, name: str) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return v


def select_sentinel_audit(
    flagged_indices,
    n_total: int,
    audit_fraction: float = 0.02,
    min_audit: int = 5,
    seed: int = 0,
) -> list[int]:
    """Draw a deterministic random sentinel sample from the accepted (non-flagged) trajectories.

    The accepted set is ``range(n_total)`` minus ``flagged_indices``. The sentinel size is
    ``max(min_audit, ceil(audit_fraction * n_accepted))``, capped at the number of accepted
    trajectories -- if there are too few accepted, as many as possible are returned. Selection is
    reproducible for a given ``seed`` and the returned (sorted) indices never overlap ``flagged``.
    """

    n_total = int(n_total)
    if n_total < 0:
        raise ValueError(f"n_total must be nonnegative, got {n_total}")
    audit_fraction = _validate_fraction(audit_fraction, "audit_fraction")
    min_audit = int(min_audit)
    if min_audit < 0:
        raise ValueError(f"min_audit must be nonnegative, got {min_audit}")

    flagged = {int(i) for i in flagged_indices}
    accepted = [i for i in range(n_total) if i not in flagged]
    n_accepted = len(accepted)
    if n_accepted == 0:
        return []

    n_audit = max(min_audit, int(math.ceil(audit_fraction * n_accepted)))
    n_audit = min(n_audit, n_accepted)

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_accepted, generator=generator)[:n_audit]
    sentinel = sorted(accepted[int(i)] for i in perm)
    return sentinel


def evaluate_false_negatives(
    flagged_indices,
    sentinel_indices,
    true_error,
    high_error_quantile: float = 0.90,
) -> dict:
    """Estimate force-error false negatives among accepted trajectories using the sentinel sample.

    ``true_error`` is the per-trajectory true *force* error (one scalar per trajectory). A trajectory
    is "high-error" when its true force error is at or above the ``high_error_quantile`` of the whole
    ensemble. A *false negative* is a high-error trajectory that was accepted (not flagged).

    Reports both the full-ensemble false-negative accounting (available here because the offline true
    error is known for every trajectory) and the sentinel estimate -- the high-error rate within the
    audited sentinel sample, which is what an operational audit on a held-back rerun budget would
    observe. All quantities are force-error based; none refer to position accuracy.
    """

    err = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    n_total = int(err.numel())
    if n_total == 0:
        raise ValueError("true_error is empty")
    if not bool(torch.isfinite(err).all()):
        raise ValueError("true_error contains NaN or infinite values")
    q = float(high_error_quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"high_error_quantile must be in [0, 1], got {high_error_quantile!r}")

    flagged = sorted({int(i) for i in flagged_indices})
    sentinel = sorted({int(i) for i in sentinel_indices})
    for idx in flagged + sentinel:
        if not 0 <= idx < n_total:
            raise ValueError(f"index {idx} out of range for {n_total} trajectories")
    if set(flagged) & set(sentinel):
        raise ValueError("sentinel_indices must not overlap flagged_indices")

    high_threshold = float(torch.quantile(err, q))
    is_high = err >= high_threshold
    flagged_mask = torch.zeros(n_total, dtype=torch.bool)
    if flagged:
        flagged_mask[torch.tensor(flagged)] = True
    accepted_mask = ~flagged_mask

    n_high = int(is_high.sum())
    n_high_flagged = int((is_high & flagged_mask).sum())  # captured by the rerun budget
    n_high_accepted = int((is_high & accepted_mask).sum())  # missed -> false negatives
    false_negative_rate = (n_high_accepted / n_high) if n_high > 0 else float("nan")

    n_accepted = int(accepted_mask.sum())
    accepted_fn_rate = (n_high_accepted / n_accepted) if n_accepted > 0 else float("nan")

    # Sentinel estimate: high-error rate among the audited accepted sample.
    n_sentinel = len(sentinel)
    if n_sentinel > 0:
        sentinel_high = is_high[torch.tensor(sentinel)]
        n_sentinel_high = int(sentinel_high.sum())
        sentinel_fn_rate = n_sentinel_high / n_sentinel
        sentinel_high_indices = [sentinel[i] for i in range(n_sentinel) if bool(sentinel_high[i])]
    else:
        n_sentinel_high = 0
        sentinel_fn_rate = float("nan")
        sentinel_high_indices = []

    return {
        "high_error_quantile": q,
        "high_error_threshold": high_threshold,
        "n_total": n_total,
        "n_high_error": n_high,
        "n_high_error_flagged": n_high_flagged,
        "n_false_negatives": n_high_accepted,
        "false_negative_rate": false_negative_rate,
        "n_accepted": n_accepted,
        "accepted_false_negative_rate": accepted_fn_rate,
        "n_sentinel": n_sentinel,
        "n_sentinel_high_error": n_sentinel_high,
        "sentinel_false_negative_rate": sentinel_fn_rate,
        "sentinel_high_error_indices": sentinel_high_indices,
        "error_basis": "true_force_model_error",
        "note": (
            "False negatives are accepted trajectories whose true FORCE-MODEL error is in the "
            "high-error tail; this is not a position-accuracy diagnostic."
        ),
    }


def audit_summary_dict(
    n_total: int,
    flagged_indices,
    sentinel_indices,
    false_negatives: dict,
    *,
    audit_fraction: float,
    min_audit: int,
    high_error_quantile: float,
    seed: int,
) -> dict:
    """Assemble a JSON-serializable summary of the sentinel audit configuration and outcome.

    Bundles the selection parameters, the flagged/sentinel/accepted counts, and the
    :func:`evaluate_false_negatives` result into one dict for the audit report.
    """

    flagged = sorted({int(i) for i in flagged_indices})
    sentinel = sorted({int(i) for i in sentinel_indices})
    n_total = int(n_total)
    n_accepted = n_total - len(flagged)
    return {
        "n_total": n_total,
        "n_flagged": len(flagged),
        "n_accepted": n_accepted,
        "n_sentinel": len(sentinel),
        "audit_fraction": float(audit_fraction),
        "min_audit": int(min_audit),
        "high_error_quantile": float(high_error_quantile),
        "seed": int(seed),
        "flagged_indices": flagged,
        "sentinel_indices": sentinel,
        "false_negatives": false_negatives,
        "error_basis": "true_force_model_error",
    }
