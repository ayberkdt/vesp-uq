"""Physical acceleration-budget *sweep* for VESP-UQ.

The single-budget screener (:mod:`scripts.run_physical_budget_screening`) answers "at budget B, which
trajectories are flagged?". The journal question is the *curve*: "as the physical acceleration-error
budget tightens, how many trajectories need a high-fidelity rerun, and what error slips through?".

This sweeps a grid of physical budgets against a single fitted layer. The VESP-UQ pipeline is run
**once** to obtain every trajectory's force-risk score and held-out true force-model error; each
budget is then a threshold applied analytically to those arrays (no refit per budget). For each
budget it reports, in the requested physical units:

* ``flagged_fraction``            -- fraction of trajectories whose risk score meets the budget,
* ``budget_exceedance_rate``      -- fraction whose *true* force error exceeds the budget (the ground truth),
* ``capture_rate``                -- recall: of the true exceedances, the fraction flagged,
* ``false_negative_rate``         -- 1 - capture: true exceedances that slip through unflagged,
* ``mean_true_error_accepted``    -- mean true force error among the *accepted* (unflagged) trajectories,
* ``worst_accepted_true_error``   -- the largest true force error that was accepted.

Requires an explicit physical acceleration scale in the config (``body.acceleration_scale_m_s2`` or a
physical ``body.acceleration_units``) and an absolute scoring mode -- relative ranking scores cannot
be compared to a physical budget. Everything targets force-model error, never position error; flagged
counts are screening guidance, not a safety guarantee.

    python scripts/run_physical_budget_sweep.py --config configs/vespuq/vespuq_real_lunar.yaml \
        --budgets 1e-9 3e-9 1e-8 3e-8 1e-7 --units m/s^2 --scoring expected_abs_p95
"""

from __future__ import annotations

import argparse
import math

# scripts/run_physical_budget_screening.py is a sibling module (not importable as a package), so pull
# its single-budget machinery in via the path.
import sys
from pathlib import Path

import torch

from vesp.common.config import load_config
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.physical_units import resolve_acceleration_scale
from vesp.uq.scoring import is_absolute_scoring, is_relative_scoring
from vesp.uq.thresholds import PHYSICAL_BUDGET_SCORINGS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_physical_budget_screening import (  # noqa: E402
    _configure_physical_budget,
    run_physical_budget_screening,
)

DEFAULT_BUDGETS = (1e-9, 3e-9, 1e-8, 3e-8, 1e-7)


def _budget_metrics(risk_phys, true_phys, budget: float) -> dict:
    """Threshold the physical risk / true-error arrays at ``budget`` and compute the sweep metrics."""

    flagged = risk_phys >= budget
    exceed = true_phys >= budget  # ground-truth budget violation
    n = int(true_phys.numel())
    n_exceed = int(exceed.sum())
    captured = int((flagged & exceed).sum())
    accepted = ~flagged
    n_accepted = int(accepted.sum())

    capture_rate = (captured / n_exceed) if n_exceed else float("nan")
    accepted_true = true_phys[accepted]
    return {
        "budget": float(budget),
        "n_trajectories": n,
        "n_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.float().mean()),
        "n_budget_exceedances": n_exceed,
        "budget_exceedance_rate": float(exceed.float().mean()),
        "capture_rate": capture_rate,
        "false_negative_rate": (1.0 - capture_rate) if n_exceed else float("nan"),
        "n_accepted": n_accepted,
        "mean_true_error_accepted": float(accepted_true.mean()) if n_accepted else float("nan"),
        "worst_accepted_true_error": float(accepted_true.max()) if n_accepted else float("nan"),
    }


def run_budget_sweep(config: dict, *, budgets, units: str, scoring: str) -> dict:
    """Run the pipeline once and sweep ``budgets`` (physical units) as thresholds on the fitted layer."""

    if is_relative_scoring(scoring) or not is_absolute_scoring(scoring):
        raise SystemExit(
            f"physical-budget sweep requires an absolute scoring mode {PHYSICAL_BUDGET_SCORINGS}; "
            f"got --scoring/{scoring!r} (relative supervisor scores cannot be compared to a budget)."
        )
    scale = resolve_acceleration_scale(config)
    if not scale.physical:
        raise SystemExit(
            "physical-budget sweep requires an explicit acceleration scale: set "
            "body.acceleration_scale_m_s2 or a physical body.acceleration_units in the config."
        )

    # Configure the budget block once (value is irrelevant to the per-trajectory arrays) and run the
    # pipeline a single time to obtain risk + true-error for every trajectory.
    ref_args = argparse.Namespace(
        budget=float(budgets[0]), units=units, scoring=scoring,
        max_rerun_fraction=None, conformal=False,
    )
    _configure_physical_budget(config, ref_args)
    screened = run_physical_budget_screening(config)
    rows = screened["_rows"]
    risk_phys = torch.tensor([r["risk_score_physical"] for r in rows], dtype=torch.float64)
    true_phys = torch.tensor([r["true_force_error_physical"] for r in rows], dtype=torch.float64)

    sweep = [_budget_metrics(risk_phys, true_phys, float(b)) for b in sorted(budgets)]
    return {
        "config_path": config.get("_config_path"),
        "error_basis": "true_force_model_error",
        "scope_note": (
            "Sweeps a grid of physical acceleration-error budgets against one fitted VESP-UQ layer. "
            "Flagged counts are screening guidance against a force-model-error tolerance; this is not "
            "a position-accuracy or orbit-covariance diagnostic and does not guarantee safety."
        ),
        "units": units,
        "scoring": scoring,
        "acceleration_scale_m_s2": scale.scale_m_s2,
        "n_trajectories": len(rows),
        "sweep": sweep,
    }


_COLS = (
    "budget", "n_trajectories", "n_flagged", "flagged_fraction",
    "n_budget_exceedances", "budget_exceedance_rate", "capture_rate", "false_negative_rate",
    "n_accepted", "mean_true_error_accepted", "worst_accepted_true_error",
)


def _sweep_csv(result: dict) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return repr(v) if isinstance(v, float) else str(v)

    lines = [",".join(_COLS)]
    lines += [",".join(cell(row[c]) for c in _COLS) for row in result["sweep"]]
    return "\n".join(lines) + "\n"


def _sweep_md(result: dict) -> str:
    def f(x, s=".3e"):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(float(x), s)

    header = [
        "# VESP-UQ Physical Acceleration-Budget Sweep",
        "",
        "Each row tightens the physical acceleration-error budget against a single fitted layer. "
        "`flagged_fraction` is the rerun workload; `capture_rate` is the recall on true budget "
        "exceedances; `worst_accepted_true_error` is the largest force error that slipped through "
        "unflagged. Target: trajectory true FORCE-model error. Screening guidance, not a safety "
        "guarantee.",
        "",
        f"- config: `{result.get('config_path')}`  |  scoring: `{result['scoring']}`  |  "
        f"units: `{result['units']}`",
        f"- acceleration scale: 1 model unit = {f(result['acceleration_scale_m_s2'])} m/s^2  |  "
        f"trajectories: {result['n_trajectories']}",
        "",
        f"| budget ({result['units']}) | flagged frac | exceedance rate | capture | FN rate | "
        "mean accepted err | worst accepted err |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["sweep"]:
        header.append(
            f"| {f(row['budget'])} | {f(row['flagged_fraction'], '.3f')} | "
            f"{f(row['budget_exceedance_rate'], '.3f')} | {f(row['capture_rate'], '.3f')} | "
            f"{f(row['false_negative_rate'], '.3f')} | {f(row['mean_true_error_accepted'])} | "
            f"{f(row['worst_accepted_true_error'])} |"
        )
    header += [
        "",
        "Interpretation: as the budget tightens (top -> bottom), more trajectories are flagged and the "
        "worst accepted error falls. A budget where capture is high while flagged fraction stays within "
        "the rerun budget is the operating point; a high worst-accepted error at a loose budget is the "
        "force error a screening miss would let through.",
        "",
    ]
    return "\n".join(header) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ physical acceleration-budget sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS),
                        help="physical acceleration-error budget grid (in --units)")
    parser.add_argument("--units", default="m/s^2", help="m/s^2 | km/s^2 | mm/s^2 | um/s^2")
    parser.add_argument("--scoring", default="expected_abs_p95", help="absolute scoring mode")
    parser.add_argument("--out-dir", default="outputs/physical_budget_sweep")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    result = run_budget_sweep(config, budgets=args.budgets, units=args.units, scoring=args.scoring)

    markdown = _sweep_md(result)
    write_run_artifacts(
        Path(args.out_dir),
        tool="run_physical_budget_sweep",
        config=config,
        json_files={"physical_budget_sweep.json": result},
        text_files={"physical_budget_sweep.md": markdown, "physical_budget_sweep.csv": _sweep_csv(result)},
    )
    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_physical_budget_sweep: {Path(args.out_dir) / 'physical_budget_sweep.md'}")


if __name__ == "__main__":
    main()
