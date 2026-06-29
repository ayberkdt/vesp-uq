"""Physical acceleration-budget screening status for VESP-UQ (WP11).

An operational "rerun any trajectory whose force-model error may exceed B m/s^2" screen requires a
verified mapping from model score units to m/s^2. This module reports, per config, whether that
scaling metadata is present and -- when it is -- runs the absolute-budget screen and reports the
alarm count / fraction (and false positives/negatives against the held-out true force error). When
the implied alarm fraction is degenerate (0% or 100%), the physical tolerance lies outside the
data's physical range, so the screen is flagged as implemented-but-not-operationally-activated:
the scaling metadata must be verified before the absolute budget is meaningful.

No physical scaling is invented here -- the conversion is read from the config
(``body.acceleration_units`` or ``body.acceleration_scale_m_s2``); absent it, the screen is
reported as not activated.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import _build_trajectories
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.physical_units import (
    acceleration_to_model_units,
    acceleration_to_physical,
    has_physical_acceleration_scale,
    resolve_acceleration_scale,
)
from vesp.uq.risk_baselines import prepare
from vesp.uq.scoring import aggregate_trajectory_error
from vesp.uq.suite import _csv, _fmt, band_label, git_commit_hash

DEFAULT_TOLERANCE_M_S2 = 1.0e-8
_MAX_ORBITS = 2000


def _to_model_threshold(tolerance_m_s2: float, scale) -> float:
    val = acceleration_to_model_units(torch.tensor([float(tolerance_m_s2)], dtype=torch.float64),
                                      scale, source_units="m/s^2")
    return float(val.reshape(-1)[0])


def physical_budget_status_run(config: dict, *, seed: int, tolerance_m_s2: float) -> dict:
    """Per-config physical-budget screen status (one seed)."""

    band = band_label(config)
    scale = resolve_acceleration_scale(config)
    if not has_physical_acceleration_scale(config):
        return {
            "band": band, "activated": False,
            "reason": "no physical scaling metadata (set body.acceleration_units to a physical unit "
                      "or body.acceleration_scale_m_s2); absolute budget screening not activated.",
            "tolerance_m_s2": float(tolerance_m_s2),
        }

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    screen_cfg = cfg.setdefault("uq", {}).setdefault("screening", {})
    screen_cfg["n_orbits"] = min(int(screen_cfg.get("n_orbits", _MAX_ORBITS)), _MAX_ORBITS)

    plugin, _samples, _train, held, dtype, _ = prepare(cfg)
    traj_info = _build_trajectories(screen_cfg, seed=int(seed), dtype=dtype, config=cfg)
    trajectories = traj_info["trajectories"]
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()

    scored = plugin.score_ensemble(trajectories, scoring="expected_abs_p95")
    risk = torch.tensor([s.risk_score for s in scored], dtype=torch.float64)  # absolute, model units
    true_error = torch.empty(len(trajectories), dtype=torch.float64)
    for i, traj in enumerate(trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
        true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)

    thr_model = _to_model_threshold(tolerance_m_s2, scale)
    flagged = risk > thr_model
    truth_exceeds = true_error > thr_model
    n = len(trajectories)
    n_flagged = int(flagged.sum())
    tp = int((flagged & truth_exceeds).sum())
    fp = int((flagged & ~truth_exceeds).sum())
    fn = int((~flagged & truth_exceeds).sum())
    fraction = n_flagged / n if n else float("nan")
    degenerate = n_flagged == 0 or n_flagged == n

    # physical magnitude context: median predicted absolute risk and true error in m/s^2
    med_risk_phys = float(acceleration_to_physical(
        torch.median(risk).reshape(1), scale, target_units="m/s^2").reshape(-1)[0])
    med_true_phys = float(acceleration_to_physical(
        torch.median(true_error).reshape(1), scale, target_units="m/s^2").reshape(-1)[0])

    return {
        "band": band, "activated": True, "seed": int(seed),
        "tolerance_m_s2": float(tolerance_m_s2),
        "scale_m_s2_per_model_unit": float(scale.scale_m_s2),
        "threshold_model_units": thr_model,
        "n_trajectories": n, "n_flagged": n_flagged, "flagged_fraction": fraction,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "median_pred_risk_m_s2": med_risk_phys, "median_true_error_m_s2": med_true_phys,
        "operationally_activated": not degenerate,
        "reason": ("alarm fraction is degenerate (0% or 100%): the physical tolerance lies outside "
                   "the data's physical range; verify scaling metadata before operational use."
                   if degenerate else "physical budget screen produced a non-degenerate alarm set."),
    }


def _status_md(rows: list[dict], tolerance_m_s2: float) -> str:
    lines = [
        "# VESP-UQ Physical Acceleration-Budget Screening Status (WP11)",
        "",
        f"Absolute force-error budget: **{tolerance_m_s2:g} m/s^2**. The screen flags trajectories "
        "whose estimated absolute force error (p95 expected error) exceeds the budget. Activation "
        "requires verified model->m/s^2 scaling metadata; no scaling is invented.",
        "",
        "| band | activated | scale (m/s^2 / unit) | threshold (model) | flagged | TP/FP/FN | "
        "median pred / true (m/s^2) | operational |",
        "| --- | :---: | ---: | ---: | ---: | --- | --- | :---: |",
    ]
    for r in rows:
        if not r["activated"]:
            lines.append(f"| {r['band']} | no | n/a | n/a | n/a | n/a | n/a | not activated |")
            continue
        lines.append(
            f"| {r['band']} | yes | {_fmt(r['scale_m_s2_per_model_unit'], '.3e')} | "
            f"{_fmt(r['threshold_model_units'], '.3e')} | "
            f"{r['n_flagged']}/{r['n_trajectories']} ({_fmt(r['flagged_fraction'], '.3f')}) | "
            f"{r['true_positives']}/{r['false_positives']}/{r['false_negatives']} | "
            f"{_fmt(r['median_pred_risk_m_s2'], '.2e')} / {_fmt(r['median_true_error_m_s2'], '.2e')} | "
            f"{'yes' if r['operationally_activated'] else 'NOT (degenerate)'} |"
        )
    lines += ["", "## Per-config notes", ""]
    for r in rows:
        lines.append(f"- **{r['band']}**: {r['reason']}")
    lines += [
        "",
        "The screening mechanism is implemented and unit-correct. Where the alarm set is degenerate, "
        "the configured physical tolerance is far outside the residual dataset's physical magnitude, "
        "so the absolute screen is reported as implemented-but-not-operationally-activated for that "
        "band until the scaling metadata is independently verified. This prevents overclaiming a "
        "physical operating point that the current normalization does not support.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _status_csv(rows: list[dict]) -> str:
    cols = ["band", "activated", "tolerance_m_s2", "scale_m_s2_per_model_unit", "threshold_model_units",
            "n_trajectories", "n_flagged", "flagged_fraction", "true_positives", "false_positives",
            "false_negatives", "median_pred_risk_m_s2", "median_true_error_m_s2",
            "operationally_activated"]
    out = [cols]
    for r in rows:
        out.append([r.get(c) for c in cols])
    return _csv(out)


def run_physical_budget_status(
    configs, *, seed=0, tolerance_m_s2=DEFAULT_TOLERANCE_M_S2, out_dir="outputs/physical_budget_status/",
) -> dict:
    """Run the physical-budget screening status over configs; write status table + manifest."""

    out_dir = Path(out_dir)
    rows = [physical_budget_status_run(cfg, seed=seed, tolerance_m_s2=tolerance_m_s2) for cfg in configs]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_artifacts(
        out_dir,
        tool="run_physical_budget_status",
        config=configs[0],
        json_files={"physical_budget_status.json": {
            "git_commit": git_commit_hash(), "tolerance_m_s2": float(tolerance_m_s2), "rows": rows,
        }},
        text_files={
            "physical_budget_status.csv": _status_csv(rows),
            "physical_budget_status.md": _status_md(rows, tolerance_m_s2),
        },
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "rows": rows}
