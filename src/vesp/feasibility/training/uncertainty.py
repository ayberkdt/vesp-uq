"""Stage 3C uncertainty evaluation: calibrated posterior over equivalent sources.

Fits the exact linear-Gaussian posterior (``LinearGaussianPosterior``) on the training
acceleration data, then asks the questions that actually matter for MaxEnt-as-uncertainty:

1. Are the predictive error bars *calibrated*? (Does the nominal 90% interval contain ~90%
   of held-out residuals?)
2. Does the *epistemic* (source) uncertainty correctly grow where the model extrapolates
   — i.e. is the low-altitude predictive std larger than high-altitude? A model that is
   inaccurate at low altitude (the known bottleneck) but *knows* it is uncertain there is
   exactly the value uncertainty quantification can add over a bare ridge point estimate.

The posterior mean equals the (acceleration-channel) ridge solution, so the point-estimate
accuracy story is unchanged; this only adds — and validates — the error bars.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch

from vesp.common.artifacts import atomic_write_json, atomic_write_text, ensure_run_layout
from vesp.common.config import get_device, get_dtype, load_config, merge_defaults, validate_config
from vesp.core.operators import build_joint_operator
from vesp.core.sources import SourceSet
from vesp.extensions.probabilistic import AltitudeNoiseModel, LinearGaussianPosterior, calibration_metrics
from vesp.feasibility.training.train_discrete import make_data_splits, make_model


def _acceleration_system(positions, acceleration, model, kernel_cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (unweighted) acceleration operator + flattened target for a set of queries."""

    sources = SourceSet(
        positions=model.source_positions,
        weights=model.source_weights,
        shell_ids=model.shell_ids,
        shell_radii=tuple(float(v) for v in model.shell_radii),
    )
    bundle = build_joint_operator(
        positions,
        sources,
        potential=None,
        acceleration=acceleration,
        use_potential=False,
        use_acceleration=True,
        eps=float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0))),
        sign=float(kernel_cfg.get("acceleration_sign", 1.0)),
        source_chunk_size=kernel_cfg.get("source_chunk_size"),
        column_normalize=False,
    )
    return bundle.operator, bundle.target


def _radius_band(positions: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(positions, dim=-1)


def run_uncertainty_eval(config: dict) -> dict:
    config = merge_defaults(config)
    validate_config(config)
    dtype = get_dtype(config)
    device = get_device(config)
    kernel_cfg = config.get("kernel", {})
    loss_cfg = config.get("loss", {})
    solver_cfg = config.get("solver", {}) if isinstance(config.get("solver"), dict) else {}
    lambda_raw = loss_cfg.get("lambda_l2", solver_cfg.get("lambda_l2", 1.0e-6))
    try:
        lambda_l2 = float(lambda_raw)
    except (TypeError, ValueError):
        # 'auto' (L-curve) or other non-numeric: fall back; the evidence path ignores it anyway.
        lambda_l2 = 1.0e-6

    splits = make_data_splits(config, dtype=dtype)
    model = make_model(config, dtype=dtype, model_cls=_model_cls(config)).to(device)

    train = splits.train.to(device)
    operator, target = _acceleration_system(train.positions, train.acceleration, model, kernel_cfg)

    hyperparams = str(config.get("uncertainty", {}).get("hyperparams", "fixed")).lower()
    if hyperparams == "evidence":
        # empirical-Bayes: pick noise variance + prior precision by evidence maximization
        posterior = LinearGaussianPosterior.fit_evidence(operator, target)
    else:
        posterior = LinearGaussianPosterior.fit(operator, target, lambda_l2=lambda_l2)

    # Optional heteroscedastic (altitude-dependent) predictive noise. It is fit on the
    # VALIDATION residuals (held out from the ridge mean): training residuals are optimistic
    # (ridge fits the training points), so they badly underestimate the altitude-dependent
    # generalization error. This is post-hoc variance recalibration on a held-out set.
    noise_model_kind = str(config.get("uncertainty", {}).get("noise_model", "homoscedastic")).lower()
    altitude_noise: AltitudeNoiseModel | None = None
    if noise_model_kind == "heteroscedastic":
        cal = splits.val.to(device)
        cal_op, cal_tgt = _acceleration_system(cal.positions, cal.acceleration, model, kernel_cfg)
        cal_pred = posterior.predict(cal_op, include_noise=False)
        cal_residual = cal_pred["mean"] - cal_tgt
        cal_row_radii = _radius_band(cal.positions).repeat(3)  # acc rows are [x,y,z] blocks
        # fit the power-law as EXCESS over the global evidence noise floor, so high-altitude
        # extrapolation keeps the floor (the bare power law -> 0 as h grows and over-shrinks there).
        altitude_noise = AltitudeNoiseModel.fit(
            cal_row_radii, cal_residual, cal_pred["epistemic_variance"] + posterior.noise_var
        )

    report: dict = {
        "hyperparams": hyperparams,
        "noise_model": noise_model_kind,
        "lambda_l2": posterior.lambda_l2 if posterior.lambda_l2 is not None else lambda_l2,
        "config_lambda_l2": lambda_l2,
        "noise_var": posterior.noise_var,
        "noise_std": float(posterior.noise_var ** 0.5),
        "n_sources": int(model.n_sources),
        "bands": {},
    }
    if altitude_noise is not None:
        report["altitude_noise_a"] = altitude_noise.a
        report["altitude_noise_b"] = altitude_noise.b

    eval_sets = {"val": splits.val}
    if splits.test_high is not None and splits.test_high.positions.shape[0] > 0:
        eval_sets["test_high"] = splits.test_high
    if splits.test_low is not None and splits.test_low.positions.shape[0] > 0:
        eval_sets["test_low"] = splits.test_low

    altitude_bands = config.get("evaluation", {}).get("altitude_bands") or {
        "low": [1.03, 1.15],
        "mid": [1.15, 1.35],
        "high": [1.35, 1.60],
    }

    for name, data in eval_sets.items():
        data = data.to(device)
        op, tgt = _acceleration_system(data.positions, data.acceleration, model, kernel_cfg)
        row_radii = _radius_band(data.positions).repeat(3)  # acc rows are [x,y,z] blocks
        if altitude_noise is not None:
            # total noise = global floor + altitude-dependent excess
            het_noise = posterior.noise_var + altitude_noise.variance(row_radii)
            pred = posterior.predict(op, noise_variance=het_noise)
            homo = posterior.predict(op, include_noise=True)
        else:
            pred = posterior.predict(op, include_noise=True)
            homo = None
        mean = pred["mean"]
        std = pred["std"]
        epistemic_std = torch.sqrt(pred["epistemic_variance"].clamp_min(0.0))

        def _masked_metrics(mask: torch.Tensor) -> dict:
            metrics = calibration_metrics(mean[mask], std[mask], tgt[mask])
            metrics["mean_epistemic_std"] = float(torch.mean(epistemic_std[mask]).detach().cpu())
            metrics["mean_radius"] = float(torch.mean(row_radii[mask]).detach().cpu())
            if homo is not None:
                hm = calibration_metrics(homo["mean"][mask], homo["std"][mask], tgt[mask])
                metrics["homo_picp_90"] = hm.get("picp_90")
                metrics["homo_z_std"] = hm.get("z_std")
                metrics["homo_nll"] = hm.get("nll")
            return metrics

        report["bands"][name] = _masked_metrics(torch.ones_like(row_radii, dtype=torch.bool))
        # altitude sub-bands within this split (so a random-split run still shows per-altitude
        # calibration, which is the whole point of the heteroscedastic model)
        for band_name, band_range in altitude_bands.items():
            if band_range is None:
                continue
            lo, hi = float(band_range[0]), float(band_range[1])
            mask = (row_radii >= lo) & (row_radii <= hi)
            if int(mask.sum().detach().cpu()) >= 30:
                report["bands"][f"{name}@{band_name}"] = _masked_metrics(mask)

    report["summary"] = _summary(report)
    return report


def _model_cls(config: dict):
    from vesp.core.models import DiscreteVESP, MultiShellDiscreteVESP

    return MultiShellDiscreteVESP if config.get("model", {}).get("type") == "multishell" else DiscreteVESP


def _summary(report: dict) -> dict:
    bands = report["bands"]
    out: dict = {}
    # prefer the OOD splits; fall back to in-split altitude sub-bands (random-split runs)
    low = bands.get("test_low") or bands.get("val@low") or {}
    high = bands.get("test_high") or bands.get("val@high") or {}
    if low and high and high.get("mean_epistemic_std"):
        out["low_high_epistemic_std_ratio"] = low["mean_epistemic_std"] / max(
            high["mean_epistemic_std"], 1.0e-30
        )
        out["epistemic_grows_at_low_altitude"] = out["low_high_epistemic_std_ratio"] > 1.0
    # rough calibration verdict on the validation band: 90% interval within +/-10 points
    val = bands.get("val", {})
    if "picp_90" in val:
        out["val_picp_90"] = val["picp_90"]
        out["val_calibrated_90"] = abs(val["picp_90"] - 0.90) <= 0.1
    # worst-band calibration gap (how far any band's PICP90 is from nominal)
    gaps = [abs(m["picp_90"] - 0.90) for m in bands.values() if "picp_90" in m]
    if gaps:
        out["max_picp90_gap"] = max(gaps)
        out["all_bands_calibrated_90"] = max(gaps) <= 0.1
    return out


def _build_report_md(report: dict) -> str:
    noise_model = report.get("noise_model", "homoscedastic")
    header = [
        "# Stage 3C Uncertainty / Calibration Report",
        "",
        f"hyperparams: {report.get('hyperparams', 'fixed')} "
        f"(lambda_l2={report['lambda_l2']:.4g}, config_lambda_l2={report.get('config_lambda_l2')})",
        f"noise_model: {noise_model}",
        f"estimated (global) noise_std: {report['noise_std']:.4g}",
        f"n_sources: {report['n_sources']}",
    ]
    if "altitude_noise_b" in report:
        header.append(
            f"altitude noise sigma^2(h) = a * h^(-b): a={report['altitude_noise_a']:.3e}, "
            f"b={report['altitude_noise_b']:.3f}  (h = r - 1; larger b = faster growth toward the surface)"
        )
    lines = header + [
        "",
        "Posterior mean == ridge (acceleration channel); this report only validates the error bars.",
        "",
        "| band | mean_radius | rmse | mean_pred_std | mean_epistemic_std | z_std | picp_50 | picp_68 | picp_90 | picp_95 | nll | crps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in report["bands"].items():
        lines.append(
            f"| {name} | {m.get('mean_radius', float('nan')):.3f} | {m['rmse']:.3e} | "
            f"{m['mean_pred_std']:.3e} | {m['mean_epistemic_std']:.3e} | {m['z_std']:.2f} | "
            f"{m.get('picp_50', float('nan')):.2f} | {m.get('picp_68', float('nan')):.2f} | "
            f"{m.get('picp_90', float('nan')):.2f} | {m.get('picp_95', float('nan')):.2f} | "
            f"{m['nll']:.3f} | {m.get('crps', float('nan')):.3e} |"
        )
    if noise_model == "heteroscedastic" and any("homo_picp_90" in m for m in report["bands"].values()):
        lines += [
            "",
            "Before/after (homoscedastic -> heteroscedastic) per band:",
            "",
            "| band | picp_90 homo -> hetero | z_std homo -> hetero |",
            "| --- | ---: | ---: |",
        ]
        for name, m in report["bands"].items():
            lines.append(
                f"| {name} | {m.get('homo_picp_90', float('nan')):.2f} -> {m.get('picp_90', float('nan')):.2f} | "
                f"{m.get('homo_z_std', float('nan')):.2f} -> {m.get('z_std', float('nan')):.2f} |"
            )
    summary = report.get("summary", {})
    lines += ["", "## Verdict", ""]
    if "low_high_epistemic_std_ratio" in summary:
        ratio = summary["low_high_epistemic_std_ratio"]
        verdict = "YES" if summary.get("epistemic_grows_at_low_altitude") else "NO"
        lines.append(
            f"- Epistemic uncertainty grows toward the low-altitude OOD band: **{verdict}** "
            f"(low/high epistemic std ratio = {ratio:.2f}). This is the value uncertainty adds: "
            f"the model flags where it is extrapolating."
        )
    if "val_picp_90" in summary:
        cal = "calibrated" if summary.get("val_calibrated_90") else "MIScalibrated"
        lines.append(
            f"- Validation 90% interval coverage = {summary['val_picp_90']:.2f} (nominal 0.90): **{cal}**. "
            f"z_std > 1 means overconfident, < 1 underconfident."
        )
    if noise_model == "heteroscedastic":
        lines.append(
            "\n_Honest caveat: exact conjugate posterior (mean == ridge) with a parametric "
            "altitude-dependent noise sigma^2(h)=a*h^(-b) fit on HELD-OUT validation residuals "
            "(post-hoc variance recalibration). It is a simple 2-parameter power-law misfit model, "
            "not a learned/full noise model; in-distribution it calibrates per band, but at extreme "
            "OOD it must EXTRAPOLATE the law._"
        )
    else:
        lines.append(
            "\n_Honest caveat: exact conjugate posterior (mean == ridge) with a single global "
            "(homoscedastic) noise estimate. Set uncertainty.noise_model: heteroscedastic for the "
            "altitude-dependent predictive noise._"
        )
    return "\n".join(lines) + "\n"


def run(config: dict) -> dict:
    report = run_uncertainty_eval(config)
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    run_name = str(output_cfg.get("run_name", "uncertainty"))
    layout = ensure_run_layout(output_dir / run_name)
    atomic_write_json(layout.run_dir / "uncertainty_report.json", report)
    markdown = _build_report_md(report)
    atomic_write_text(layout.run_dir / "uncertainty_report.md", markdown)
    # encoding-safe console output (Windows consoles may not be UTF-8)
    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_uncertainty_report: {layout.run_dir / 'uncertainty_report.md'}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 3C: calibrated linear-Gaussian posterior over sources.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
