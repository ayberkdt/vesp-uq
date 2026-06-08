"""E7 regularizer shootout analysis: L2 vs entropy at MATCHED data error.

Ridge is the data-accuracy ceiling, so comparing MaxEnt to ridge on RMSE is the wrong
test. The right question (the user's framing) is: *for a given, tolerable amount of extra
data error, which regularizer buys more source-distribution "health"* — lower brittle
cancellation, lower source norm, lower concentration, higher effective source count?

This module aligns the two regularizer families by data error: for each constrained-MaxEnt
trial it interpolates the ridge "health frontier" at the SAME ``relative_acceleration_rmse``
and reports the per-metric winner. The verdict is computed from the data, not hardcoded.
"""

from __future__ import annotations

from vesp.experiments.summarize import _to_float

ERROR_KEY = "relative_acceleration_rmse"

# (metric, "lower" | "higher")  -> which direction is healthier
HEALTH_METRICS: list[tuple[str, str]] = [
    ("shell_cancellation_ratio", "lower"),
    ("sigma_l2", "lower"),
    ("top_5pct_source_contribution", "lower"),
    ("effective_source_count", "higher"),
]

_TIE_REL_TOL = 0.02


def _family(rows: list[dict], solver: str) -> list[dict]:
    out = []
    for row in rows:
        if str(row.get("solver", "")).lower() != solver:
            continue
        if _to_float(row.get(ERROR_KEY)) is None:
            continue
        out.append(row)
    return out


def _frontier(ridge_rows: list[dict], metric: str) -> list[tuple[float, float]]:
    pts = []
    for row in ridge_rows:
        x = _to_float(row.get(ERROR_KEY))
        y = _to_float(row.get(metric))
        if x is not None and y is not None:
            pts.append((x, y))
    pts.sort(key=lambda p: p[0])
    return pts


def _interp(points: list[tuple[float, float]], xq: float) -> float | None:
    """Linear interpolation of the ridge frontier at error ``xq`` (clamped to endpoints)."""

    if not points:
        return None
    if xq <= points[0][0]:
        return points[0][1]
    if xq >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if xq <= x1:
            if x1 == x0:
                return y0
            t = (xq - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def shootout_report(rows: list[dict]) -> tuple[str, list[dict], dict]:
    """Return ``(markdown, matched_rows, tally)`` comparing entropy vs L2 at matched error."""

    ridge = _family(rows, "ridge")
    maxent = _family(rows, "maxent")
    tally = {metric: {"maxent": 0, "ridge": 0, "tie": 0} for metric, _ in HEALTH_METRICS}
    matched: list[dict] = []

    frontiers = {metric: _frontier(ridge, metric) for metric, _ in HEALTH_METRICS}

    for mr in maxent:
        err = _to_float(mr.get(ERROR_KEY))
        entry: dict = {
            "run_name": mr.get("run_name", ""),
            "entropy_mode": mr.get("entropy_mode", ""),
            "entropy_weight": _to_float(mr.get("entropy_weight")),
            ERROR_KEY: err,
        }
        for metric, direction in HEALTH_METRICS:
            mv = _to_float(mr.get(metric))
            rv = _interp(frontiers[metric], err) if err is not None else None
            entry[f"maxent_{metric}"] = mv
            entry[f"ridge_at_err_{metric}"] = rv
            winner = ""
            if mv is not None and rv is not None:
                denom = max(abs(rv), 1.0e-12)
                rel = (mv - rv) / denom
                if abs(rel) < _TIE_REL_TOL:
                    winner = "tie"
                elif (direction == "lower" and mv < rv) or (direction == "higher" and mv > rv):
                    winner = "maxent"
                else:
                    winner = "ridge"
                tally[metric][winner] += 1
            entry[f"winner_{metric}"] = winner
        matched.append(entry)

    return _build_markdown(ridge, maxent, matched, tally), matched, tally


def _fmt(value) -> str:
    v = _to_float(value)
    return "-" if v is None else f"{v:.4g}"


def _build_markdown(ridge: list[dict], maxent: list[dict], matched: list[dict], tally: dict) -> str:
    lines = [
        "# E7 Regularizer Shootout: L2 vs Entropy at matched data error",
        "",
        "Ridge is the accuracy ceiling; the question is which regularizer buys more source",
        "health per unit of (tolerable) extra data error. For each MaxEnt trial the ridge",
        "health is interpolated at the SAME relative_acceleration_rmse.",
        "",
        f"Ridge trials: {len(ridge)}    MaxEnt trials: {len(maxent)}",
        "",
        "## Ridge health frontier (the L2 knob)",
        "",
        "| relative_acceleration_rmse | shell_cancellation_ratio | sigma_l2 | top_5pct | effective_source_count |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(ridge, key=lambda r: _to_float(r.get(ERROR_KEY)) or 0.0):
        lines.append(
            f"| {_fmt(row.get(ERROR_KEY))} | {_fmt(row.get('shell_cancellation_ratio'))} | "
            f"{_fmt(row.get('sigma_l2'))} | {_fmt(row.get('top_5pct_source_contribution'))} | "
            f"{_fmt(row.get('effective_source_count'))} |"
        )

    lines += [
        "",
        "## MaxEnt vs ridge-at-matched-error",
        "",
        "| maxent run | err | cancel (me/ridge) | sigma_l2 (me/ridge) | top5 (me/ridge) | eff_N (me/ridge) |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for e in matched:
        def cell(metric: str) -> str:
            w = e.get(f"winner_{metric}", "")
            mark = {"maxent": " ✅", "ridge": " ❌", "tie": " ≈"}.get(w, "")
            return f"{_fmt(e.get('maxent_' + metric))} / {_fmt(e.get('ridge_at_err_' + metric))}{mark}"

        lines.append(
            f"| {e['run_name']} | {_fmt(e.get(ERROR_KEY))} | {cell('shell_cancellation_ratio')} | "
            f"{cell('sigma_l2')} | {cell('top_5pct_source_contribution')} | {cell('effective_source_count')} |"
        )

    lines += ["", "## Tally (who is healthier at matched error)", "", "| metric | maxent wins | ridge wins | ties |", "| --- | ---: | ---: | ---: |"]
    total_me = total_ridge = 0
    for metric, _ in HEALTH_METRICS:
        t = tally[metric]
        total_me += t["maxent"]
        total_ridge += t["ridge"]
        lines.append(f"| {metric} | {t['maxent']} | {t['ridge']} | {t['tie']} |")

    lines += ["", "## Verdict", ""]
    if matched and total_me == 0 and total_ridge == 0:
        lines.append(
            "- **Entropy made no meaningful difference at matched error** (every health metric ties "
            "with the ridge frontier). On this setup the constrained-MaxEnt budget permits only a "
            "negligible entropy perturbation of the ill-conditioned ridge solution — the brittle "
            "near-cancellation makes the residual hypersensitive to any entropy spreading — so it "
            "effectively reduces to ridge and changes no health metric."
        )
    elif total_ridge > total_me:
        lines.append(
            f"- **L2 (Tikhonov) buys more source-health per unit data-error than entropy** "
            f"({total_ridge} vs {total_me} matched-error health wins). On the collapse/norm setup "
            f"this is expected: the cancellation pathology is a source-norm problem that L2 penalizes "
            f"directly and entropy does not."
        )
    elif total_me > total_ridge:
        lines.append(
            f"- **Entropy buys more source-health per unit data-error than L2** "
            f"({total_me} vs {total_ridge} matched-error health wins) — entropy earns a niche here "
            f"(typically the de-concentration axis L2 does not control directly)."
        )
    else:
        lines.append(f"- L2 and entropy are comparable on source-health at matched error ({total_me} vs {total_ridge}).")
    lines.append("")
    lines.append("_Health directions: lower cancellation / sigma_l2 / top5 is healthier; higher effective_source_count is healthier._")
    return "\n".join(lines) + "\n"
