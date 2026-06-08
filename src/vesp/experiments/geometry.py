"""E8 source-geometry shootout analysis: is the low-altitude bottleneck geometry?

E7 showed the low-altitude problem is not a regularization-flavor problem. The forward
hypothesis is that it is a *representation* problem: the equivalent-source model cannot
capture the near-surface high-frequency residual (model misfit). E8 compares source
GEOMETRIES (shell radii / counts) at the same auto-selected regularization and asks which
geometry minimizes the held-out low-altitude error, and at what conditioning cost.

This module just ranks the standard summary rows; it adds no physics. ``lambda_l2: auto``
(L-curve) is expected upstream so each geometry is fairly regularized and regularization is
not a confound.
"""

from __future__ import annotations

from vesp.experiments.summarize import _to_float

# Primary ranking key: held-out OOD low-altitude error if present, else the val low band.
PRIMARY_KEY = "test_low_acceleration_rmse"
FALLBACK_KEY = "low_altitude_acceleration_rmse"

REPORT_COLUMNS = [
    "run_name",
    "test_low_acceleration_rmse",
    "low_altitude_acceleration_rmse",
    "low_to_high_error_ratio",
    "relative_acceleration_rmse",
    "shell_cancellation_ratio",
    "sigma_l2",
    "effective_source_count",
    "selected_lambda_l2",
]
BASELINE_NAME_HINT = "baseline"


def _low_altitude(row: dict) -> float | None:
    value = _to_float(row.get(PRIMARY_KEY))
    return value if value is not None else _to_float(row.get(FALLBACK_KEY))


def _fmt(value) -> str:
    v = _to_float(value)
    return "-" if v is None else f"{v:.4g}"


def geometry_report(rows: list[dict]) -> tuple[str, list[dict]]:
    """Return ``(markdown, ranking)`` ranking geometries by held-out low-altitude error."""

    ranked = sorted(
        (r for r in rows if _low_altitude(r) is not None),
        key=lambda r: _low_altitude(r),
    )
    lines = [
        "# E8 Source Geometry Shootout: is the low-altitude bottleneck geometry?",
        "",
        "Each geometry is fairly (L-curve auto-λ) regularized; ranked by held-out low-altitude error.",
        "",
        "| geometry | test_low | low_alt | low/high | rel_acc | cancel | sigma_l2 | eff_N | sel_lambda |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ranked:
        lines.append(
            f"| {r.get('run_name', '')} | {_fmt(r.get('test_low_acceleration_rmse'))} | "
            f"{_fmt(r.get('low_altitude_acceleration_rmse'))} | {_fmt(r.get('low_to_high_error_ratio'))} | "
            f"{_fmt(r.get('relative_acceleration_rmse'))} | {_fmt(r.get('shell_cancellation_ratio'))} | "
            f"{_fmt(r.get('sigma_l2'))} | {_fmt(r.get('effective_source_count'))} | "
            f"{_fmt(r.get('selected_lambda_l2'))} |"
        )

    lines += ["", "## Verdict", ""]
    if not ranked:
        lines.append("- No rows with a low-altitude metric; nothing to rank.")
        return "\n".join(lines) + "\n", ranked

    best = ranked[0]
    best_low = _low_altitude(best)
    baseline = _find_baseline(rows)
    lines.append(
        f"- **Best low-altitude geometry: `{best.get('run_name')}`** "
        f"(low-altitude error = {_fmt(best_low)}, cancel = {_fmt(best.get('shell_cancellation_ratio'))}, "
        f"sigma_l2 = {_fmt(best.get('sigma_l2'))})."
    )
    if baseline is not None and baseline.get("run_name") != best.get("run_name"):
        base_low = _low_altitude(baseline)
        if base_low and best_low is not None and base_low > 0:
            change = (best_low - base_low) / base_low * 100.0
            direction = "reduces" if change < 0 else "does NOT reduce (increases)"
            lines.append(
                f"- Versus the baseline (`{baseline.get('run_name')}`, low-altitude error "
                f"{_fmt(base_low)}), the best geometry {direction} low-altitude error by "
                f"{abs(change):.0f}%."
            )
            # conditioning cost note
            best_cancel = _to_float(best.get("shell_cancellation_ratio"))
            base_cancel = _to_float(baseline.get("shell_cancellation_ratio"))
            worsened_cond = best_cancel is not None and base_cancel is not None and best_cancel > base_cancel * 1.1
            if best_cancel is not None and base_cancel is not None:
                cond = "without worsening" if not worsened_cond else "at the cost of worsened"
                lines.append(f"- It achieves this {cond} conditioning (cancellation {_fmt(base_cancel)} -> {_fmt(best_cancel)}).")
            if change < -20.0 and not worsened_cond:
                lines.append(
                    "- Interpretation: geometry is a **strong** lever for the low-altitude bottleneck "
                    "(large gain, no conditioning cost); the E7 forward hypothesis is supported."
                )
            elif change < -5.0:
                lines.append(
                    "- Interpretation: geometry is only a **weak / modest** lever; it helps somewhat "
                    "(typically just via more sources at the same radii), but surface-near shells tend to "
                    "**destabilize** (low/high ratio and cancellation blow up). This points to a band-limit / "
                    "conditioning ceiling; the realistic next lever is the heteroscedastic posterior (Stage 3C+)."
                )
            else:
                lines.append(
                    "- Interpretation: geometry does **not** materially help low-altitude error — consistent "
                    "with a degree-band-limit / fundamental representation ceiling. Redirect to the "
                    "heteroscedastic posterior (Stage 3C+)."
                )
    lines.append("")
    lines.append("_Lower test_low / low_alt / cancel / sigma_l2 is better; higher effective_source_count is better._")
    return "\n".join(lines) + "\n", ranked


def _find_baseline(rows: list[dict]) -> dict | None:
    for r in rows:
        if BASELINE_NAME_HINT in str(r.get("run_name", "")).lower():
            return r
    return None
