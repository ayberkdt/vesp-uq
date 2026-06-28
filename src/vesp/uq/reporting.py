"""Report construction for VESP-UQ runs: summary rollups, Markdown report, CSV tables.

Pure formatting over the report dict assembled by :mod:`vesp.uq.experiment`. The report keeps
*force-risk / OOD* screening visually distinct from the *supplied true-error metric* (a
diagnostic oracle, e.g. a position-error read), and never claims the force-risk score predicts
that metric.
"""

from __future__ import annotations

import csv as _csv
import io
import math


def csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _stat(values) -> dict:
    """mean / max / p95 of a list of scalars, dropping NaNs (empty -> {})."""

    import torch

    t = torch.tensor([float(v) for v in values], dtype=torch.float64)
    t = t[~torch.isnan(t)]
    if t.numel() == 0:
        return {}
    return {
        "mean": float(t.mean()),
        "max": float(t.max()),
        "p95": float(torch.quantile(t, 0.95)),
    }


def expected_error_summary(scores, domain_support: bool) -> dict:
    """Ensemble-level rollup of the supervisor per-trajectory metrics for the report body."""

    out = {
        "mean_expected_error": _stat([s.mean_expected_error for s in scores]),
        "max_expected_error": _stat([s.max_expected_error for s in scores]),
        "p95_expected_error": _stat([s.p95_expected_error for s in scores]),
        "mean_point_risk": _stat([s.mean_point_risk for s in scores]),
    }
    if domain_support:
        out["max_domain_risk"] = _stat([s.max_domain_risk for s in scores])
        out["time_outside_support"] = _stat([s.time_outside_support for s in scores])
    return out


def build_summary(report: dict) -> dict:
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]["screen"]
    out: dict = {}
    if "low_high_epistemic_std_ratio" in cal:
        out["low_high_epistemic_std_ratio"] = cal["low_high_epistemic_std_ratio"]
        out["epistemic_grows_at_low_altitude"] = cal["low_high_epistemic_std_ratio"] > 1.0
    if "low_high_pred_sigma_ratio" in cal:
        out["low_high_pred_sigma_ratio"] = cal["low_high_pred_sigma_ratio"]
    low = cal.get("low", {})
    if "picp_90" in low:
        out["low_band_picp_90"] = low["picp_90"]
        out["low_band_calibrated_90"] = abs(low["picp_90"] - 0.90) <= 0.1
    if "ellipsoid_picp_90" in low:
        out["low_band_ellipsoid_picp_90"] = low["ellipsoid_picp_90"]
    out["selection_mode"] = screen.get("selection_mode")
    out["rerun_fraction"] = screen["rerun_fraction"]
    out["n_flagged"] = screen.get("n_flagged")
    out["zero_alarms"] = screen.get("n_flagged") == 0
    out["capture_rate"] = screen.get("capture_rate")
    out["spearman_risk_vs_error"] = screen.get("spearman_risk_vs_error")
    out["error_ratio_flagged_to_accepted"] = screen.get("error_ratio_flagged_to_accepted")
    if screen.get("error_ratio_flagged_to_accepted"):
        out["screen_concentrates_error"] = screen["error_ratio_flagged_to_accepted"] > 1.0
    # Lift over a random screen: a random top-fraction selection captures ~rerun_fraction of the
    # high-error set, so lift = capture_rate / rerun_fraction (>1 means better than chance).
    cap = screen.get("capture_rate")
    rf = screen.get("rerun_fraction")
    if cap is not None and rf and float(rf) > 0.0 and not math.isnan(float(cap)):
        out["lift_over_random"] = float(cap) / float(rf)
    return out


def build_tables(scores, screening, true_error, flagged_set) -> dict:
    # Legacy sigma columns are kept verbatim and in order; the supervisor / domain columns are
    # appended before the two trailing bookkeeping columns so old readers stay valid.
    traj_header = [
        "trajectory_id", "risk_score", "max_sigma", "mean_sigma", "low_altitude_sigma_integral",
        "time_above_threshold", "combined_altitude_risk", "min_radius", "mean_radius",
        "mean_epistemic_sigma",
        "max_expected_error", "mean_expected_error", "p95_expected_error",
        "low_altitude_expected_error_integral", "max_mean_error_magnitude",
        "mean_mean_error_magnitude", "mean_point_risk", "p95_point_risk",
        "mean_point_risk_abs", "p95_point_risk_abs",
        "max_domain_risk", "time_outside_support",
        "flagged_for_rerun", "true_error",
    ]
    traj_rows = []
    for i, s in enumerate(scores):
        traj_rows.append([
            i, s.risk_score, s.max_sigma, s.mean_sigma, s.low_altitude_sigma_integral,
            s.time_above_threshold, s.combined_altitude_risk, s.min_radius, s.mean_radius,
            s.mean_epistemic_sigma,
            s.max_expected_error, s.mean_expected_error, s.p95_expected_error,
            s.low_altitude_expected_error_integral, s.max_mean_error_magnitude,
            s.mean_mean_error_magnitude, s.mean_point_risk, s.p95_point_risk,
            s.mean_point_risk_abs, s.p95_point_risk_abs,
            s.max_domain_risk, s.time_outside_support,
            int(i in flagged_set), float(true_error[i]),
        ])
    flag_col = traj_header.index("flagged_for_rerun")
    flagged_rows = [r for r in traj_rows if r[flag_col] == 1]
    return {"trajectory_header": traj_header, "trajectory_rows": traj_rows, "flagged_rows": flagged_rows}


def calibration_table(calibration: dict) -> tuple[list[str], list[list]]:
    metric_keys = [
        "n", "mean_radius", "rmse", "mean_pred_std", "mean_epistemic_std", "z_std",
        "picp_50", "picp_68", "picp_90", "picp_95", "nll", "crps",
        "ellipsoid_picp_50", "ellipsoid_picp_68", "ellipsoid_picp_90", "ellipsoid_picp_95",
        "mean_mahalanobis_d2", "median_mahalanobis_d2",
    ]
    header = ["band"] + metric_keys
    rows = []
    for name in ("all", "low", "mid", "high"):
        m = calibration.get(name)
        if not m:
            continue
        rows.append([name] + [m.get(k, "") for k in metric_keys])
    return header, rows


def fmt(x, spec: str = ".3g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _conformal_prediction_summary(conformal: dict) -> str:
    """Compact human-readable summary for the operational conformal prediction layer."""

    if not conformal.get("enabled"):
        return "off"
    global_cal = conformal.get("global") or {}
    return (
        f"{fmt(global_cal.get('scale'), '.3g')} "
        f"(mode `{conformal.get('mode')}`, scope `{conformal.get('scope')}`, "
        f"target coverage {fmt(global_cal.get('target_coverage'), '.2f')})"
    )


def build_report_md(report: dict) -> str:
    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]
    sc = screen["screen"]
    rt = report["runtime"]
    s = report["summary"]
    units = report.get("units", {})
    lines = [
        "# VESP-UQ Report - Equivalent-Source Force-Risk / OOD Calibration Layer",
        "",
        f"dataset: `{report['dataset']}`",
        f"sources: {fit['n_sources']}  |  reg: {fit['reg_method']} (lambda_l2={fmt(fit.get('lambda_l2'))})  "
        f"|  noise_model: {fit['noise_model']}  |  covariance_mode: {fit.get('covariance_mode', 'exact')}  "
        f"|  global noise_std={fmt(fit.get('noise_std'))}",
    ]
    if units:
        lines.append(
            f"units: risk_score=`{units.get('risk_score_units')}`, "
            f"acceleration=`{units.get('acceleration_metric_units')}`, position=`{units.get('position_units')}`"
            + (f"  ({units['force_error_scale_note']})" if units.get("force_error_scale_note") else "")
        )
        if units.get("physical_conversion_available"):
            lines.append(
                f"physical acceleration conversion: available "
                f"(1 model unit = {fmt(units.get('acceleration_scale_m_s2'), '.3e')} m/s^2, "
                f"source `{units.get('acceleration_scale_source')}`); model-normalized values are "
                "also retained."
            )
        else:
            lines.append(
                "physical acceleration conversion unavailable; values are reported in "
                "model-normalized acceleration units."
            )
    if "altitude_noise_b" in fit:
        lines.append(
            f"altitude noise sigma^2(h)=a*h^(-b): a={fmt(fit['altitude_noise_a'], '.3e')}, "
            f"b={fmt(fit['altitude_noise_b'], '.3f')} (h=r-1; larger b = faster growth toward surface)"
        )
    conformal = report.get("conformal_calibration") or {}
    if conformal.get("enabled"):
        lines.append(f"operational conformal prediction scale: {_conformal_prediction_summary(conformal)}")
    lines += [
        "",
        "## Experiment 1 - Standalone residual-error calibration",
        "",
        "| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "low", "mid", "high"):
        m = cal.get(name)
        if not m:
            continue
        lines.append(
            f"| {name} | {fmt(m.get('mean_radius'), '.3f')} | {fmt(m.get('rmse'), '.3e')} | "
            f"{fmt(m.get('mean_pred_std'), '.3e')} | {fmt(m.get('mean_epistemic_std'), '.3e')} | "
            f"{fmt(m.get('z_std'), '.2f')} | {fmt(m.get('picp_90'), '.2f')} | "
            f"{fmt(m.get('ellipsoid_picp_90'), '.2f')} | {fmt(m.get('mean_mahalanobis_d2'), '.2f')} | "
            f"{fmt(m.get('nll'), '.3f')} |"
        )
    if "low_high_epistemic_std_ratio" in cal:
        grows = s.get("epistemic_grows_at_low_altitude")
        lines += [
            "",
            f"- Epistemic uncertainty grows toward low altitude: **{'YES' if grows else 'NO'}** "
            f"(low/high epistemic std ratio = {fmt(cal['low_high_epistemic_std_ratio'], '.2f')}, "
            f"predictive sigma ratio = {fmt(cal.get('low_high_pred_sigma_ratio'), '.2f')}).",
        ]
    ee = screen.get("expected_error", {})
    sel_mode = sc.get("selection_mode", "fraction")
    zero_alarms = sc.get("n_flagged") == 0

    scoring_scale = screen.get("scoring_scale", "sigma")
    canon = screen.get("scoring_canonical", screen["scoring"])
    if scoring_scale == "relative":
        scale_line = (
            f"- **Relative scoring mode** (`{screen['scoring']}` = `{canon}`): for "
            "prioritization/ranking only, **not** absolute physical thresholding."
        )
    elif scoring_scale == "absolute":
        scale_line = (
            f"- **Absolute scoring mode** (`{screen['scoring']}` = `{canon}`): on a fixed "
            "force-risk scale, suitable for physical-budget / zero-alarm thresholding."
        )
    else:
        scale_line = (
            f"- Legacy **sigma** scoring mode (`{screen['scoring']}`): predictive-uncertainty "
            "magnitude, not an expected-force-error budget."
        )

    selection_line = f"- selection: `{sel_mode}`"
    if sel_mode == "fraction":
        selection_line += (
            f" (policy `{sc.get('fraction_policy', 'topk')}`, requested "
            f"{fmt(100 * (sc.get('requested_rerun_fraction') or 0.0), '.1f')}%)"
        )
    else:
        selection_line += f" -> absolute force-risk budget (risk score) {fmt(sc['threshold'], '.3e')}"
        if screen.get("threshold_physical_value") is not None:
            selection_line += (
                f" = physical budget {fmt(screen['threshold_physical_value'], '.3e')} "
                f"{screen.get('threshold_physical_units')} "
                f"(model-unit threshold {fmt(screen.get('threshold_model_units'), '.3e')})"
            )
        tsrc = screen.get("threshold_source")
        if tsrc:
            extra = f"source: {tsrc}"
            if screen.get("threshold_quantile") is not None:
                extra += f", q={fmt(screen['threshold_quantile'], '.3g')}"
            if screen.get("threshold_multiplier") not in (None, 1.0):
                extra += f", x{fmt(screen['threshold_multiplier'], '.3g')}"
            if screen.get("threshold_calibration_scoring"):
                extra += f", calib scoring={screen['threshold_calibration_scoring']}"
            if screen.get("threshold_calibration_n") is not None:
                extra += f", calib n={screen['threshold_calibration_n']}"
            selection_line += f" [{extra}]"
        if sc.get("max_rerun_fraction") is not None:
            selection_line += f", max rerun fraction {fmt(sc.get('max_rerun_fraction'), '.2f')}"
        if sc.get("n_above_threshold") is not None:
            selection_line += f", {sc.get('n_above_threshold')} above threshold"
    compat_note = screen.get("threshold_compatibility_note")

    sp = sc.get("spearman_risk_vs_error")
    sp_note = ""
    try:
        if sp is not None and not math.isnan(float(sp)) and abs(float(sp)) < 0.1:
            sp_note = " -- near zero: risk did not rank the supplied true-error metric on this set"
    except (TypeError, ValueError):
        pass

    if zero_alarms:
        validation_lines = [
            "- validation: 0 flagged -> capture rate / precision are not meaningful (no positives)",
            f"- Spearman(force-risk, supplied true-error metric): {fmt(sp, '.2f')}{sp_note}",
        ]
    else:
        validation_lines = [
            f"- capture rate (top-decile true-error orbits flagged): **{fmt(sc.get('capture_rate'), '.2f')}**  "
            f"| precision: {fmt(sc.get('precision'), '.2f')}  | lift over random: {fmt(s.get('lift_over_random'), '.2f')}x",
            f"- Spearman(force-risk, supplied true-error metric): {fmt(sp, '.2f')}{sp_note}",
            f"- mean true error  flagged: {fmt(sc.get('mean_error_flagged'), '.3e')}  vs  "
            f"accepted: {fmt(sc.get('mean_error_accepted'), '.3e')}  "
            f"(ratio {fmt(sc.get('error_ratio_flagged_to_accepted'), '.2f')}x)",
        ]

    src_label = screen.get("trajectory_source", "generated")
    lines += [
        "",
        "## Experiment 3 - Trajectory risk screening (force-risk vs supplied true-error metric)",
        "",
        f"- ensemble: {screen['n_trajectories']} trajectories ({src_label}), "
        f"{screen['n_output_points_total']} output points "
        f"(scoring = `{screen['scoring']}`, oracle = `{screen['oracle_source']}`, "
        f"true-error aggregator = `{screen.get('true_error_aggregator', 'p95')}`, "
        f"time-weighting = `{screen.get('time_weighting', 'none')}`"
        f"{', domain-support on' if screen.get('domain_support') else ''})",
        scale_line,
        selection_line,
    ]
    if compat_note:
        lines.append(f"- note: {compat_note}")
    lines += [
        f"- flagged {sc['n_flagged']}/{sc['n_trajectories']} ({fmt(100 * sc['rerun_fraction'], '.1f')}%)"
        + ("  -- **no trajectory exceeded the absolute force-risk budget (zero alarms)**" if zero_alarms and sel_mode != "fraction" else ""),
        f"- expected force-error per orbit (ensemble mean | max): mean "
        f"{fmt((ee.get('mean_expected_error') or {}).get('mean'), '.3e')} | "
        f"max {fmt((ee.get('max_expected_error') or {}).get('max'), '.3e')} "
        f"({units.get('risk_score_units', 'normalized accel units') if units else 'normalized accel units'})",
        *validation_lines,
        "",
        "### What these metrics mean",
        "",
        "- **force-risk score** = the VESP-UQ trajectory risk (expected force-model error / OOD). "
        "The **supplied true-error metric** is an external diagnostic oracle (e.g. a position-error "
        "read) used only to *validate* ranking; VESP-UQ does not predict it by construction.",
        "- **force-risk ranking** (Spearman, lift): does the force-risk score order orbits the way "
        "the supplied true-error metric does?",
        "- **trajectory-error ranking** (capture rate, error ratio): do flagged orbits carry larger "
        "*true trajectory* error -- a different question from force-risk calibration.",
        "- **false-alarm behavior**: under an absolute force-risk budget a safe set may flag zero; a "
        "fixed top-fraction always flags ~`rerun_fraction` by construction.",
        "- **rerun prioritization**: relative supervisor modes *rank* which orbits to rerun first; "
        "absolute modes decide whether *any* orbit exceeds a physical budget.",
        "",
        "## Runtime",
        "",
        f"- fit: {fmt(rt['fit_seconds'], '.3f')} s  |  calibration eval: {fmt(rt['calibration_eval_seconds'], '.3f')} s",
        f"- scoring: {fmt(rt['score_ms_per_trajectory'], '.3f')} ms/trajectory "
        f"({fmt(rt['score_us_per_output_point'], '.2f')} us/output point, {screen['n_output_points_total']} points total)",
        f"- _{rt['note']}_",
    ]
    lines += _iac_summary_md(report)
    return "\n".join(lines) + "\n"


def _iac_summary_md(report: dict) -> list[str]:
    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    sc = report["experiment_3_screening"]["screen"]
    s = report["summary"]
    rt = report["runtime"]
    low, mid, high = cal.get("low", {}), cal.get("mid", {}), cal.get("high", {})
    sel_mode = sc.get("selection_mode", "fraction")
    zero_alarms = sc.get("n_flagged") == 0
    if zero_alarms and sel_mode != "fraction":
        flagged_line = (
            "- **Fraction of trajectories flagged:** 0.0% -- no trajectory exceeded the absolute "
            "force-risk budget, so this safe in-distribution regime correctly raised zero alarms."
        )
        concentrate_line = (
            "- **Did flagged trajectories carry larger true error?** N/A -- nothing was flagged."
        )
    else:
        flagged_line = (
            f"- **Fraction of trajectories flagged:** {fmt(100 * sc['rerun_fraction'], '.1f')}% "
            f"(selection `{sel_mode}`, capture rate {fmt(sc.get('capture_rate'), '.2f')}, "
            f"lift over random {fmt(s.get('lift_over_random'), '.2f')}x)."
        )
        concentrate_line = (
            f"- **Did flagged trajectories carry larger true error?** "
            f"{'Yes' if s.get('screen_concentrates_error') else 'No'} "
            f"({fmt(sc.get('error_ratio_flagged_to_accepted'), '.2f')}x the accepted-set error)."
        )
    noise_kind = fit.get("altitude_noise_type") or fit.get("noise_model", "heteroscedastic")
    return [
        "",
        "## IAC claim summary",
        "",
        f"- **What was fitted?** An interior equivalent-source posterior over the residual-force "
        f"error `e_a = a_reference - a_surrogate` ({fit['n_sources']} sources, "
        f"{fit['reg_method']} regularization).",
        f"- **What was calibrated?** Altitude-dependent predictive uncertainty (post-hoc "
        f"{noise_kind} recalibration) on held-out validation residuals; the posterior mean equals "
        "the ridge point estimate.",
        f"- **Did low-altitude uncertainty increase?** "
        f"{'Yes' if s.get('epistemic_grows_at_low_altitude') else 'No'} "
        f"(low/high epistemic std ratio = {fmt(cal.get('low_high_epistemic_std_ratio'), '.2f')}).",
        f"- **PICP90 by band (low/mid/high):** {fmt(low.get('picp_90'), '.2f')} / "
        f"{fmt(mid.get('picp_90'), '.2f')} / {fmt(high.get('picp_90'), '.2f')}.",
        flagged_line,
        concentrate_line,
        f"- **Runtime overhead:** {fmt(rt['score_ms_per_trajectory'], '.3f')} ms/trajectory, "
        f"{fmt(rt['score_us_per_output_point'], '.2f')} us/output point (post-processing only).",
        "- **What should NOT be claimed:** not a better deterministic surrogate; not a "
        "position-error predictor; not true lunar density recovery; not operational orbit "
        "covariance propagation; not integrated with ST-LRPS. VESP-UQ is a force-risk / OOD "
        "uncertainty-calibration layer at the acceleration interface.",
        "",
    ]


def build_model_card(report: dict, *, model_filename: str, metadata: dict) -> str:
    """Model card for a persisted fitted layer (written next to ``vespuq_plugin.pt``).

    Industrial-registry practice: the saved model ships with a human-readable card stating what
    it is, what data and decision policy produced it, its held-out calibration, and -- in this
    project's claims-policy spirit -- what it must NOT be used to claim. All facts come from the
    training run's ``report`` and the metadata embedded in the artifact, so card and model cannot
    drift apart.
    """

    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]
    units = report.get("units", {})
    policy = metadata.get("decision_policy", {})
    provenance = metadata.get("provenance", {})
    conformal = metadata.get("conformal_prediction") or report.get("conformal_calibration") or {}
    altitude_noise_type = fit.get("altitude_noise_type")
    if altitude_noise_type == "altitude_binned":
        altitude_noise_summary = (
            f"  |  altitude-binned noise: {fit.get('altitude_noise_bins', 'n/a')} bins "
            f"(r={fmt(fit.get('altitude_noise_radius_min'), '.3f')}.."
            f"{fmt(fit.get('altitude_noise_radius_max'), '.3f')})"
        )
    elif fit.get("altitude_noise_b") is not None:
        altitude_noise_summary = (
            f"  |  altitude noise sigma^2(h) = a*h^(-b): a = {fmt(fit.get('altitude_noise_a'), '.3e')}, "
            f"b = {fmt(fit.get('altitude_noise_b'), '.3f')}"
        )
    else:
        altitude_noise_summary = ""

    lines = [
        f"# Model card - `{model_filename}`",
        "",
        "**VESP-UQ fitted equivalent-source force-error uncertainty layer.** Surrogate-agnostic: "
        "wraps any residual-gravity model sampled as `e_a(x) = a_reference(x) - a_surrogate(x)`.",
        "",
        "## Intended use",
        "",
        "- Calibrated, altitude-aware predictive uncertainty over a surrogate's force-model error.",
        "- Trajectory force-risk / OOD screening (rerun prioritization or absolute budgets) via "
        "`python -m vesp.uq.screen --model <this file>`.",
        "- NOT a deterministic accuracy improver, position-error predictor, density model, or "
        "operational orbit-covariance product (see `docs/VESP_UQ_LIMITATIONS.md`).",
        "",
        "## Provenance",
        "",
        f"- created (UTC): {provenance.get('created_at_utc', 'n/a')}",
        f"- training dataset: `{provenance.get('dataset', report.get('dataset', 'n/a'))}`"
        + (f" (sha256 `{provenance['dataset_sha256']}`)" if provenance.get("dataset_sha256") else ""),
        f"- samples: {fit.get('n_train', 'n/a')} train / {fit.get('n_val', 'n/a')} held-out val",
        f"- state format: `vesp.uq.plugin` v{metadata.get('state_version', 1)}",
        "",
        "## Fit",
        "",
        f"- sources: {fit.get('n_sources')}  |  regularization: {fit.get('reg_method')} "
        f"(lambda_l2 = {fmt(fit.get('lambda_l2'))})  |  noise model: {fit.get('noise_model')}",
        f"- global noise std: {fmt(fit.get('noise_std'), '.3e')}" + altitude_noise_summary,
        f"- domain support: {'on' if fit.get('domain_support_enabled') else 'off'} "
        f"(k = {fit.get('domain_k')}, backend = {fit.get('domain_backend')})",
        f"- operational conformal prediction scale: {_conformal_prediction_summary(conformal)}",
        "",
        "## Held-out calibration (training run)",
        "",
        "| band | n | z_std | picp_90 | ellipsoid_picp_90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "low", "mid", "high"):
        m = cal.get(name)
        if not m:
            continue
        lines.append(
            f"| {name} | {m.get('n', '')} | {fmt(m.get('z_std'), '.2f')} | "
            f"{fmt(m.get('picp_90'), '.2f')} | {fmt(m.get('ellipsoid_picp_90'), '.2f')} |"
        )
    lines += [
        "",
        "## Decision policy (packaged with the model)",
        "",
        f"- scoring: `{policy.get('scoring', screen.get('scoring'))}` "
        f"(canonical `{policy.get('scoring_canonical', screen.get('scoring_canonical'))}`, "
        f"scale `{policy.get('scoring_scale', screen.get('scoring_scale'))}`)",
        f"- threshold: {fmt(policy.get('threshold'), '.4e') if policy.get('threshold') is not None else 'none (fraction mode)'}"
        + (f"  |  source: {policy.get('threshold_source')}" if policy.get("threshold_source") else ""),
        f"- fallback rerun fraction: {fmt(policy.get('rerun_fraction'), '.2f')} "
        f"(policy `{policy.get('fraction_policy', 'topk')}`)",
        f"- time weighting: `{policy.get('time_weighting', 'none')}`",
        f"- risk-score units: `{units.get('risk_score_units', 'model_normalized_accel')}`"
        + (
            f"  |  physical scale: 1 model unit = {fmt(units.get('acceleration_scale_m_s2'), '.3e')} m/s^2"
            if units.get("physical_conversion_available")
            else "  |  physical conversion: unavailable (model-normalized only)"
        ),
        "",
        "## Scope boundaries",
        "",
        "Binding claims policy: `docs/SCIENTIFIC_CLAIMS.md`; scope limits: "
        "`docs/VESP_UQ_LIMITATIONS.md`. The posterior mean equals the ridge point estimate -- "
        "this layer never improves deterministic accuracy and never predicts position error.",
        "",
    ]
    return "\n".join(lines)
