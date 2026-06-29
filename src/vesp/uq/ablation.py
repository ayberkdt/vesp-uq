"""Drivers for the VESP-UQ score-variant ablation (WP6) and expanded baselines (WP5).

These reuse the fit / trajectory / true-force-error core (:mod:`vesp.uq.risk_baselines`) and the
suite helpers (:mod:`vesp.uq.suite`) so the ablation studies share one source of truth for
splitting, scoring, and aggregation. Variant / hybrid *selection* is done on a held-out validation
split of the trajectory ensemble and the chosen candidate is reported on a disjoint test split, so
no candidate is selected on the numbers it is judged by.

Everything targets trajectory-level true FORCE-model error, never position error.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from vesp.uq.altitude_controlled import spearman
from vesp.uq.baselines import random_scores
from vesp.uq.benchmarking import evaluate_score_against_true_error
from vesp.uq.expanded_baselines import (
    HYBRID_WEIGHTS,
    altitude_bin_rmse_lookup,
    altitude_ood_hybrid,
    altitude_uncertainty_hybrid,
    apply_ridge_ranker,
    fit_ridge_ranker,
)
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.learned_supervisor import (
    DEFAULT_BETAS,
    apply_learned_supervisor,
    fit_learned_supervisor,
    supervisor_components,
)
from vesp.uq.risk_baselines import assemble_baseline_scores, prepare, true_force_error
from vesp.uq.score_variants import SCORE_VARIANTS, compute_score_variants
from vesp.uq.suite import (
    _csv,
    _fmt,
    _pm,
    band_label,
    git_commit_hash,
    mean_std,
)

PRIMARY_FRACTION = 0.20
_EVAL_METRICS = ("spearman", "capture_rate", "precision", "lift_over_random",
                 "force_error_ratio_flagged_to_accepted")


def _val_test_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic 50/50 validation / test split of trajectory indices."""

    g = torch.Generator().manual_seed(int(seed) + 7919)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    return perm[:half], perm[half:]


def _evaluate(score: torch.Tensor, true_error: torch.Tensor, idx: torch.Tensor, frac: float) -> dict:
    m = evaluate_score_against_true_error(score[idx], true_error[idx], rerun_fraction=float(frac))
    return {k: m.get(k) for k in _EVAL_METRICS}


def _trajectory_setup(config: dict, seed: int):
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    plugin, _samples, train, held, dtype, _ = prepare(cfg)
    screen_cfg = cfg.get("uq", {}).get("screening", {})
    traj_info = _build_trajectories(screen_cfg, seed=int(seed), dtype=dtype, config=cfg)
    trajectories = traj_info["trajectories"]
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()
    time_weighting = _resolve_time_weighting(screen_cfg)
    weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None
    te, te_source = true_force_error(
        trajectories, residuals=traj_info["residuals"], held=held,
        aggregator=aggregator, dtype=dtype, weights=weights,
    )
    low_alt = float(cfg.get("uq", {}).get("risk", {}).get("low_altitude_radius", 1.15))
    return {
        "cfg": cfg, "plugin": plugin, "train": train, "held": held, "trajectories": trajectories,
        "true_error": te, "te_source": te_source, "weights": weights, "low_alt": low_alt,
        "band": band_label(cfg),
    }


def _three_way_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic 40/30/30 train / validation / test split of trajectory indices."""

    g = torch.Generator().manual_seed(int(seed) + 104729)
    perm = torch.randperm(n, generator=g)
    a, b = int(0.4 * n), int(0.7 * n)
    return perm[:a], perm[a:b], perm[b:]


# --------------------------------------------------------------------------------------------- #
# WP6 -- score-variant ablation
# --------------------------------------------------------------------------------------------- #
def score_variant_run(config: dict, *, seed: int, primary_fraction: float = PRIMARY_FRACTION) -> dict:
    """Evaluate every score variant at one seed on a held-out test split (selection uses val)."""

    setup = _trajectory_setup(config, seed)
    t0 = time.perf_counter()
    variants = compute_score_variants(
        setup["plugin"], setup["trajectories"], low_altitude_radius=setup["low_alt"], weights=setup["weights"]
    )
    compute_seconds = time.perf_counter() - t0
    scores = variants["scores"]
    te = setup["true_error"]
    n = len(setup["trajectories"])
    val_idx, test_idx = _val_test_split(n, seed)

    rows = []
    for name in SCORE_VARIANTS:
        s = scores[name]
        val = _evaluate(s, te, val_idx, primary_fraction)
        test = _evaluate(s, te, test_idx, primary_fraction)
        rows.append({
            "band": setup["band"], "seed": int(seed), "variant": name,
            "val_spearman": val["spearman"],
            "runtime_ms_per_traj": 1000.0 * compute_seconds / max(1, n) / len(SCORE_VARIANTS),
            **{f"test_{k}": test[k] for k in _EVAL_METRICS},
        })
    return {
        "band": setup["band"], "seed": int(seed), "rows": rows,
        "unavailable": variants["unavailable"], "te_source": setup["te_source"],
        "n_trajectories": n,
    }


def _aggregate_variant(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["variant"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        agg = {"val_spearman": mean_std([r["val_spearman"] for r in rs])}
        for k in _EVAL_METRICS:
            agg[f"test_{k}"] = mean_std([r[f"test_{k}"] for r in rs])
        agg["n_seeds"] = len(rs)
        out[key] = agg
    return out


def _variant_csv(agg: dict) -> str:
    cols = ["band", "variant", "n_seeds", "val_spearman_mean"]
    for k in _EVAL_METRICS:
        cols += [f"test_{k}_mean", f"test_{k}_std"]
    rows = [cols]
    for (band, variant), a in sorted(agg.items()):
        row = [band, variant, a["n_seeds"], a["val_spearman"]["mean"]]
        for k in _EVAL_METRICS:
            row += [a[f"test_{k}"]["mean"], a[f"test_{k}"]["std"]]
        rows.append(row)
    return _csv(rows)


def _selected_by_band(agg: dict) -> dict:
    bands = sorted({b for (b, _) in agg})
    selected = {}
    for band in bands:
        cands = [(v, a) for (b, v), a in agg.items() if b == band]
        best = max(cands, key=lambda t: (t[1]["val_spearman"]["mean"] if t[1]["val_spearman"]["mean"] is not None else -1))
        selected[band] = best[0]
    return selected


def _variant_md(agg: dict, selected: dict, unavailable: dict, primary_fraction: float) -> str:
    lines = [
        "# VESP-UQ Score-Variant Ablation (WP6)",
        "",
        "Each variant collapses the per-point predictive distribution into one trajectory force-risk "
        "scalar. Variants are *selected* by validation-split Spearman and reported on a disjoint test "
        f"split (capture/precision/lift at the {primary_fraction:.0%} budget). Mean +/- std across seeds.",
        "",
        "| band | variant | selected | val spearman | test spearman | test capture | test lift |",
        "| --- | --- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, variant), a in sorted(agg.items()):
        mark = "*" if selected.get(band) == variant else ""
        lines.append(
            f"| {band} | {variant} | {mark} | {_fmt(a['val_spearman']['mean'], '.3f')} | "
            f"{_pm(a['test_spearman'], '.3f')} | {_pm(a['test_capture_rate'], '.3f')} | "
            f"{_pm(a['test_lift_over_random'], '.2f')} |"
        )
    lines += ["", "Validation-selected variant per band: "
              + ", ".join(f"{b} -> `{v}`" for b, v in sorted(selected.items())), ""]
    if unavailable:
        lines += ["## Unavailable variants", ""]
        for name, reason in unavailable.items():
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _plot_variant(agg: dict, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _) in agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(8 * max(1, len(bands)), 6), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        items = sorted(
            [(v, a["test_spearman"]["mean"], a["test_spearman"]["std"]) for (b, v), a in agg.items() if b == band],
            key=lambda t: (t[1] if t[1] is not None else -1),
        )
        labels = [i[0] for i in items]
        vals = [i[1] if i[1] is not None else 0.0 for i in items]
        errs = [i[2] if i[2] is not None else 0.0 for i in items]
        ax.barh(labels, vals, xerr=errs, capsize=2, color="#5aa469")
        ax.set_title(f"{band}: test Spearman by variant")
        ax.set_xlabel("Spearman(score, true force error)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_score_variant_ablation(
    configs, *, seeds=(0, 1, 2, 3, 4), out_dir="outputs/score_ablation/",
    primary_fraction: float = PRIMARY_FRACTION, make_plots: bool = True,
) -> dict:
    """Run the score-variant ablation over configs x seeds; write tables + figure + manifest."""

    out_dir = Path(out_dir)
    runs = [score_variant_run(cfg, seed=s, primary_fraction=primary_fraction)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in run["rows"]]
    agg = _aggregate_variant(rows)
    selected = _selected_by_band(agg)
    unavailable = {}
    for run in runs:
        unavailable.update(run["unavailable"])

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot_variant(agg, out_dir / "score_variant_ablation.png"):
        artifact_files["score_variant_ablation.png"] = out_dir / "score_variant_ablation.png"

    runs_cols = ["band", "seed", "variant", "val_spearman", "runtime_ms_per_traj",
                 *[f"test_{k}" for k in _EVAL_METRICS]]
    write_run_artifacts(
        out_dir,
        tool="run_score_ablation",
        config=configs[0],
        json_files={"score_ablation_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "selected_variant_by_band": selected, "primary_fraction": primary_fraction,
            "unavailable": unavailable,
        }},
        text_files={
            "score_variant_ablation.csv": _variant_csv(agg),
            "score_variant_ablation_runs.csv": _csv([runs_cols] + [[r.get(c) for c in runs_cols] for r in rows]),
            "score_variant_ablation.md": _variant_md(agg, selected, unavailable, primary_fraction),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "selected": selected, "agg": agg}


# --------------------------------------------------------------------------------------------- #
# WP5 -- expanded baselines (hybrids, learned ridge supervisor, empirical residual lookup)
# --------------------------------------------------------------------------------------------- #
_RIDGE_LAMBDAS = (0.1, 1.0, 10.0)
_FEATURE_ORDER = (
    "min_altitude",
    "low_altitude_exposure",
    "uncertainty_only",
    "knn_p95",
    "supervisor",
    "domain_support",
    "altitude_residual_expected_ratio",
)


def expanded_baselines_run(config: dict, *, seed: int, primary_fraction: float = PRIMARY_FRACTION) -> dict:
    """Evaluate trivial + hybrid + learned + empirical baselines at one seed on a held-out test split."""

    setup = _trajectory_setup(config, seed)
    plugin, trajectories, te = setup["plugin"], setup["trajectories"], setup["true_error"]
    base = assemble_baseline_scores(setup["cfg"], plugin, trajectories, setup["train"].positions,
                                    weights=setup["weights"])
    n = len(trajectories)
    base["random"] = random_scores(n, seed=int(seed))
    base["altitude_bin_rmse_lookup"] = altitude_bin_rmse_lookup(
        setup["held"].positions, setup["held"].error, trajectories
    )
    train_idx, val_idx, test_idx = _three_way_split(n, seed)
    fit_ref = torch.cat([train_idx, val_idx])  # standardization reference (no test stats)

    candidates: dict[str, torch.Tensor] = {}
    selection: dict[str, float] = {}

    trivial = ["random", "min_altitude", "low_altitude_exposure", "uncertainty_only",
               "knn_p95", "supervisor", "altitude_bin_rmse_lookup"]
    for name in trivial:
        if name in base:
            candidates[name] = base[name]
    if "domain_support" in base:
        candidates["domain_support"] = base["domain_support"]

    # 5.1 altitude + uncertainty hybrid -- pick mixing weight a on the validation split
    alt = base["min_altitude"]
    unc = base["uncertainty_only"]
    best_a, best_val = HYBRID_WEIGHTS[0], -2.0
    for a in HYBRID_WEIGHTS:
        s = altitude_uncertainty_hybrid(alt, unc, a, ref=(alt[fit_ref], unc[fit_ref]))
        rho = spearman(s[val_idx], te[val_idx])
        if rho is not None and rho == rho and rho > best_val:
            best_val, best_a = rho, a
    candidates["altitude+uncertainty_hybrid"] = altitude_uncertainty_hybrid(
        alt, unc, best_a, ref=(alt[fit_ref], unc[fit_ref])
    )
    selection["altitude+uncertainty_hybrid_a"] = best_a

    # 5.2 altitude + OOD hybrid (only when domain support is available)
    if "domain_support" in base:
        dom = base["domain_support"]
        best_b, best_val = HYBRID_WEIGHTS[0], -2.0
        for b in HYBRID_WEIGHTS:
            s = altitude_ood_hybrid(alt, dom, b, ref=(alt[fit_ref], dom[fit_ref]))
            rho = spearman(s[val_idx], te[val_idx])
            if rho is not None and rho == rho and rho > best_val:
                best_val, best_b = rho, b
        candidates["altitude+ood_hybrid"] = altitude_ood_hybrid(
            alt, dom, best_b, ref=(alt[fit_ref], dom[fit_ref])
        )
        selection["altitude+ood_hybrid_b"] = best_b

    # 5.3 learned linear (ridge) supervisor -- fit on train, pick lambda on val, report on test
    feat_names = [f for f in _FEATURE_ORDER if f in base]
    feats = torch.stack([base[f].to(torch.float64).reshape(-1) for f in feat_names], dim=1)
    best_lam, best_val, best_model = _RIDGE_LAMBDAS[0], -2.0, None
    for lam in _RIDGE_LAMBDAS:
        model = fit_ridge_ranker(feats[train_idx], te[train_idx], lam=lam)
        rho = spearman(apply_ridge_ranker(model, feats[val_idx]), te[val_idx])
        if rho is not None and rho == rho and rho > best_val:
            best_val, best_lam, best_model = rho, lam, model
    if best_model is None:
        best_model = fit_ridge_ranker(feats[train_idx], te[train_idx], lam=_RIDGE_LAMBDAS[0])
    candidates["learned_ridge_supervisor"] = apply_ridge_ranker(best_model, feats)
    selection["learned_ridge_lambda"] = best_lam
    selection["learned_ridge_features"] = feat_names

    rows = []
    for name, score in candidates.items():
        test = _evaluate(score, te, test_idx, primary_fraction)
        rows.append({"band": setup["band"], "seed": int(seed), "baseline": name,
                     **{f"test_{k}": test[k] for k in _EVAL_METRICS}})
    return {"band": setup["band"], "seed": int(seed), "rows": rows, "selection": selection,
            "te_source": setup["te_source"], "n_trajectories": n}


def _aggregate_expanded(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["baseline"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        agg = {f"test_{k}": mean_std([r[f"test_{k}"] for r in rs]) for k in _EVAL_METRICS}
        agg["n_seeds"] = len(rs)
        out[key] = agg
    return out


def _expanded_csv(agg: dict) -> str:
    cols = ["band", "baseline", "n_seeds"]
    for k in _EVAL_METRICS:
        cols += [f"test_{k}_mean", f"test_{k}_std"]
    rows = [cols]
    for (band, baseline), a in sorted(agg.items()):
        row = [band, baseline, a["n_seeds"]]
        for k in _EVAL_METRICS:
            row += [a[f"test_{k}"]["mean"], a[f"test_{k}"]["std"]]
        rows.append(row)
    return _csv(rows)


def _expanded_md(agg: dict, selections: dict, primary_fraction: float) -> str:
    lines = [
        "# VESP-UQ Expanded Baseline Comparison (WP5)",
        "",
        "Trivial single-feature baselines vs altitude+uncertainty / altitude+OOD hybrids, a learned "
        "ridge supervisor, and an altitude-bin empirical residual-RMSE lookup. Hybrid mixing weights "
        "and the ridge lambda are chosen on a validation split; all numbers below are on a disjoint "
        f"test split at the {primary_fraction:.0%} budget. Mean +/- std across seeds.",
        "",
        "| band | baseline | test spearman | test capture | test precision | test lift | test err_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, baseline), a in sorted(agg.items()):
        lines.append(
            f"| {band} | {baseline} | {_pm(a['test_spearman'], '.3f')} | "
            f"{_pm(a['test_capture_rate'], '.3f')} | {_pm(a['test_precision'], '.3f')} | "
            f"{_pm(a['test_lift_over_random'], '.2f')} | "
            f"{_pm(a['test_force_error_ratio_flagged_to_accepted'], '.2f')} |"
        )
    lines += ["", "## Validation-selected hyper-parameters (per band, first seed shown)", ""]
    for band, sel in sorted(selections.items()):
        lines.append(f"- {band}: " + ", ".join(f"{k}={v}" for k, v in sel.items()))
    lines += [
        "",
        "Interpretation: if a hybrid or the learned ridge supervisor beats the VESP-UQ supervisor on "
        "test Spearman, the supervisor's hand-set weights can be tuned in future work; if the "
        "supervisor matches them, its added value is the calibrated local covariance, not a superior "
        "scalar ranking. Force-risk diagnostic only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_expanded(agg: dict, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _) in agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(8 * max(1, len(bands)), 6), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        items = sorted(
            [(v, a["test_spearman"]["mean"], a["test_spearman"]["std"]) for (b, v), a in agg.items() if b == band],
            key=lambda t: (t[1] if t[1] is not None else -1),
        )
        labels = [i[0] for i in items]
        vals = [i[1] if i[1] is not None else 0.0 for i in items]
        errs = [i[2] if i[2] is not None else 0.0 for i in items]
        colors = ["#d2691e" if ("hybrid" in lbl or "learned" in lbl or "lookup" in lbl) else "#4a6fa5"
                  for lbl in labels]
        ax.barh(labels, vals, xerr=errs, capsize=2, color=colors)
        ax.set_title(f"{band}: test Spearman by baseline")
        ax.set_xlabel("Spearman(score, true force error)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_expanded_baselines(
    configs, *, seeds=(0, 1, 2, 3, 4), out_dir="outputs/expanded_baselines/",
    primary_fraction: float = PRIMARY_FRACTION, make_plots: bool = True,
) -> dict:
    """Run the expanded baseline comparison over configs x seeds; write tables + figure + manifest."""

    out_dir = Path(out_dir)
    runs = [expanded_baselines_run(cfg, seed=s, primary_fraction=primary_fraction)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in run["rows"]]
    agg = _aggregate_expanded(rows)
    selections = {run["band"]: run["selection"] for run in runs}  # last seed per band

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot_expanded(agg, out_dir / "expanded_baselines.png"):
        artifact_files["expanded_baselines.png"] = out_dir / "expanded_baselines.png"

    runs_cols = ["band", "seed", "baseline", *[f"test_{k}" for k in _EVAL_METRICS]]
    write_run_artifacts(
        out_dir,
        tool="run_expanded_baselines",
        config=configs[0],
        json_files={"expanded_baselines_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "selections_by_band": selections, "primary_fraction": primary_fraction,
        }},
        text_files={
            "expanded_baselines.csv": _expanded_csv(agg),
            "expanded_baselines_runs.csv": _csv([runs_cols] + [[r.get(c) for c in runs_cols] for r in rows]),
            "expanded_baselines.md": _expanded_md(agg, selections, primary_fraction),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg, "selections": selections}


# --------------------------------------------------------------------------------------------- #
# Learned supervisor (Design A): validation-tuned exponents vs the hand-set supervisor
# --------------------------------------------------------------------------------------------- #
def _subset_components(c: dict, idx) -> dict:
    return {"expected_error": c["expected_error"][idx], "rel_alt": c["rel_alt"][idx],
            "domain_risk": c["domain_risk"][idx], "n_points": c["n_points"]}


def learned_supervisor_run(config: dict, *, seed: int, primary_fraction: float = PRIMARY_FRACTION) -> dict:
    """Fit supervisor exponents on a validation split; compare hand-set vs learned on a test split."""

    setup = _trajectory_setup(config, seed)
    plugin, trajectories, te = setup["plugin"], setup["trajectories"], setup["true_error"]
    comps = supervisor_components(plugin, trajectories)
    n = len(trajectories)
    _train_idx, val_idx, test_idx = _three_way_split(n, seed)

    fit = fit_learned_supervisor(_subset_components(comps, val_idx), te[val_idx])
    betas = fit["betas"]
    scores = {
        "supervisor_handtuned": apply_learned_supervisor(comps, DEFAULT_BETAS),
        "supervisor_learned": apply_learned_supervisor(comps, betas),
    }
    rows = []
    for name, score in scores.items():
        test = _evaluate(score, te, test_idx, primary_fraction)
        rows.append({"band": setup["band"], "seed": int(seed), "method": name,
                     "beta_ee": betas[0], "beta_alt": betas[1], "beta_ood": betas[2],
                     "val_fit_spearman": fit["fit_spearman"],
                     **{f"test_{k}": test[k] for k in _EVAL_METRICS}})
    return {"band": setup["band"], "seed": int(seed), "rows": rows, "betas": betas, "fit": fit}


def _aggregate_learned(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["method"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        agg = {f"test_{k}": mean_std([r[f"test_{k}"] for r in rs]) for k in _EVAL_METRICS}
        agg["beta_ee"] = mean_std([r["beta_ee"] for r in rs])
        agg["beta_alt"] = mean_std([r["beta_alt"] for r in rs])
        agg["beta_ood"] = mean_std([r["beta_ood"] for r in rs])
        agg["n_seeds"] = len(rs)
        out[key] = agg
    return out


def _learned_csv(agg: dict) -> str:
    cols = ["band", "method", "n_seeds", "beta_ee_mean", "beta_alt_mean", "beta_ood_mean"]
    for k in _EVAL_METRICS:
        cols += [f"test_{k}_mean", f"test_{k}_std"]
    rows = [cols]
    for (band, method), a in sorted(agg.items()):
        row = [band, method, a["n_seeds"], a["beta_ee"]["mean"], a["beta_alt"]["mean"], a["beta_ood"]["mean"]]
        for k in _EVAL_METRICS:
            row += [a[f"test_{k}"]["mean"], a[f"test_{k}"]["std"]]
        rows.append(row)
    return _csv(rows)


def _learned_md(agg: dict, primary_fraction: float) -> str:
    lines = [
        "# VESP-UQ Learned Supervisor (Design A)",
        "",
        "Validation-tuned exponents on the supervisor's physical components "
        "(`point_risk = expected_error^b1 * rel_alt^b2 * (1 + b3 * domain_risk)`), reported on a "
        f"disjoint test split at the {primary_fraction:.0%} budget. `beta=(1,1,1)` is the hand-set "
        "supervisor. Mean +/- std across seeds.",
        "",
        "| band | method | betas (ee, alt, ood) | test spearman | test capture | test lift |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (band, method), a in sorted(agg.items()):
        betas = (f"({_fmt(a['beta_ee']['mean'], '.2f')}, {_fmt(a['beta_alt']['mean'], '.2f')}, "
                 f"{_fmt(a['beta_ood']['mean'], '.2f')})") if method == "supervisor_learned" else "(1, 1, 1)"
        lines.append(
            f"| {band} | {method} | {betas} | {_pm(a['test_spearman'], '.3f')} | "
            f"{_pm(a['test_capture_rate'], '.3f')} | {_pm(a['test_lift_over_random'], '.2f')} |"
        )
    lines += [
        "",
        "Interpretation: where the learned supervisor beats the hand-set one on test Spearman/capture, "
        "the supervisor's fixed component weights were sub-optimal and can be validation-tuned (the "
        "physical multiplicative form is preserved; only the exponents change). Where altitude "
        "dominates, even tuned exponents cannot beat altitude -- the value there remains the "
        "calibrated covariance. Default behavior is unchanged: `beta=(1,1,1)` reproduces the current "
        "supervisor exactly.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_learned_supervisor(
    configs, *, seeds=(0, 1, 2, 3, 4), out_dir="outputs/learned_supervisor/",
    primary_fraction: float = PRIMARY_FRACTION,
) -> dict:
    """Run the learned-supervisor comparison over configs x seeds; write table + manifest."""

    out_dir = Path(out_dir)
    runs = [learned_supervisor_run(cfg, seed=s, primary_fraction=primary_fraction)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in run["rows"]]
    agg = _aggregate_learned(rows)
    betas_by_band = {}
    for run in runs:
        betas_by_band.setdefault(run["band"], []).append(run["betas"])

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_cols = ["band", "seed", "method", "beta_ee", "beta_alt", "beta_ood", "val_fit_spearman",
                 *[f"test_{k}" for k in _EVAL_METRICS]]
    write_run_artifacts(
        out_dir,
        tool="run_learned_supervisor",
        config=configs[0],
        json_files={"learned_supervisor_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "primary_fraction": primary_fraction, "betas_by_band": betas_by_band,
            "parameterization": "point_risk = expected_error^b1 * rel_alt^b2 * (1 + b3 * domain_risk)",
            "default_betas_reproduce_handset_supervisor": list(DEFAULT_BETAS),
        }},
        text_files={
            "learned_supervisor.csv": _learned_csv(agg),
            "learned_supervisor_runs.csv": _csv([runs_cols] + [[r.get(c) for c in runs_cols] for r in rows]),
            "learned_supervisor.md": _learned_md(agg, primary_fraction),
        },
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg, "betas_by_band": betas_by_band}
