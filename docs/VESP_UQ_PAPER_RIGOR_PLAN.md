# VESP-UQ Paper Rigor Plan (Acta Astronautica)

Status: **implemented harnesses; paper verdicts remain measured-output dependent**. This plan
hardens the *evidence and framing* of VESP-UQ for a journal (Acta Astronautica) submission. It
builds on the completed
[`VESP_UQ_JOURNAL_VALIDATION_PLAN.md`](VESP_UQ_JOURNAL_VALIDATION_PLAN.md) (WP1–WP12 done) and the
N1–N21 roadmap. It does **not** relax any claim boundary in
[`SCIENTIFIC_CLAIMS.md`](SCIENTIFIC_CLAIMS.md) / [`VESP_UQ_LIMITATIONS.md`](VESP_UQ_LIMITATIONS.md).

Guiding constraints (unchanged from the journal plan):
- No invented numbers. Every value traces to a script + output file + manifest (SHA-256 + git hash).
- Deterministic given a seed. New entry points take `--seed` / `--seeds`.
- Reuse existing `src/vesp/uq/*` (suite, benchmarking, metrics, baselines, audit, conformal).
- Claims-safe: force-model risk only; no position-error prediction, no validated orbit covariance,
  no density recovery.

---

## 0. Why this round exists (grounded in current results)

The benchmark suite already runs (5 seeds, L60 + L90). The measured numbers
([`outputs/benchmark_suite/benchmark_summary.md`](../outputs/benchmark_suite/benchmark_summary.md),
[`outputs/journal/journal_validation_report.md`](../outputs/journal/journal_validation_report.md))
say three things a journal reviewer will seize on:

1. **The scalar ranking claim is marginal and band-dependent.**
   - L60: `vespuq_supervisor` Spearman 0.753 ± 0.030 vs `min_altitude` 0.725 ± 0.022 — overlapping
     error bars, no significance test reported.
   - L90: `min_altitude` 0.856 **beats** `vespuq_supervisor` 0.820; matched-altitude concordance for
     the supervisor is 0.449 (< 0.5) → no signal beyond altitude on L90.
   - `learned_ridge_supervisor` (capture 0.730) and the score-ablation winner
     `expected_error_plus_noise_floor` (0.798) both beat the shipped supervisor — the proposed score
     form is not even the best score.

2. **The real, unique contribution is calibrated altitude-aware local covariance — but it does not
   transfer across bands.** L60 low band is reasonable (z_std 1.087, PICP90 0.876); L90 is severely
   under-confident (z_std 0.02–0.26, PICP90 ≈ 1.0). The 2-parameter heteroscedastic law transfers
   *usability* but not *sharpness* (already noted pre-N18).

3. **The honest characterization is strong** (drift-boundary regime map, geometry-vs-noise-law
   mechanism) and should be foregrounded rather than buried under a ranking headline.

**Conclusion.** Before writing, the code must (A) make every comparison *statistically defensible*,
(B) reframe the contribution around *decision quality* and *calibrated covariance* rather than
scalar ranking, (C) close or fully explain the L90 calibration gap, and (D) compare the
equivalent-source posterior against a UQ baseline a reviewer respects. These are WP-A … WP-D below.

---

## What already exists (reuse, do not rebuild)

| Capability | Source |
|---|---|
| Multi-seed ranking/calibration aggregation (mean±std) | `suite.aggregate_ranking`, `aggregate_calibration` |
| Capture@k / precision / lift / Spearman / err-ratio | `benchmarking.evaluate_score_against_true_error`, `selection.select_reruns` |
| Matched-altitude paired sign test (per-run) | `suite.aggregate_altitude_controlled` → `sign_test_p_value_one_sided` |
| Ellipsoid (Mahalanobis χ²₃) coverage + mean d² | `metrics.vector_calibration_metrics` |
| Per-band z_std / PICP / NLL / CRPS | `plugin.evaluate_calibration`, `extensions.probabilistic` |
| Raw-vs-calibrated + conformal scale | `audit.py`, `conformal.py`, `run_calibration_reliability.py` |
| Geometry × calibration sweep | `geometry_calibration.py` (single_surface fixes L90 low z_std 0.257→0.938) |
| Drift boundary (family × horizon) | `drift_boundary.py` |
| Learned supervisor exponents | `learned_supervisor.py` |
| Journal report + LaTeX + claims table | `journal_report.py`, `run_journal_report.py` |

**Implemented evidence surface = statistics layer, decision-quality metrics, component-wise
calibration, a respected UQ baseline, and a reframed report.**

---

## WP-A — Statistical defensibility (highest priority, low risk) — **DONE**

> Implemented: `src/vesp/uq/significance.py` (`paired_bootstrap_ci` — trajectory-level paired
> bootstrap CI + two-sided p; `seed_paired_test` — exact Wilcoxon signed-rank across seeds;
> metric resolvers spearman/capture/auroc). Wired into `suite.compute_significance`, written as
> `significance_summary.{csv,md}` with a per-comparison verdict (beats / indistinguishable /
> altitude beats) derived from the bootstrap CI. Tests: `tests/test_uq_significance.py`.

**Goal:** every headline comparison ships with an uncertainty interval and a significance verdict,
so "modestly improves" is a tested statement, not an eyeball.

- **Implemented module `src/vesp/uq/significance.py`:**
  - `paired_bootstrap_ci(scores_a, scores_b, true_error, metric, n_boot=2000, seed)` — bootstrap
    over trajectories for the **difference** in a metric (Spearman / capture@k / AUROC), returning
    `(delta_mean, ci_low, ci_high, p_two_sided)`. Trajectory-level resampling (not seed-level) gives
    power the 5-seed std cannot.
  - `seed_paired_test(per_seed_a, per_seed_b)` — Wilcoxon signed-rank across seeds (small-n exact),
    complementary to the bootstrap.
  - Reuse the existing `spearman` (`altitude_controlled.py`) and `evaluate_score_against_true_error`.
- **Wired into the suite:** `suite.run_suite` includes a pairwise block comparing `vespuq_supervisor`
  (and `learned_ridge_supervisor`) against `min_altitude` per band, emitting
  `significance_summary.csv` (delta, 95% CI, bootstrap p, Wilcoxon p) and a one-line verdict per
  comparison ("CI excludes 0" / "indistinguishable from altitude").
- **Outputs:** `outputs/benchmark_suite/significance_summary.{csv,md}`.
- **Tests:** `tests/test_uq_significance.py` — identical inputs → delta 0, CI brackets 0; a
  constructed clear-separation case → CI excludes 0; determinism under fixed seed.
- **Acceptance:** every Table-B headline number has a CI; the verdict ("beats altitude" vs
  "indistinguishable") is derived from the CI, never hand-set. **Effort:** S–M.

## WP-B — Decision-quality metrics (reframes the contribution) — **DONE**

> Implemented in `src/vesp/uq/benchmarking.py`: `detection_metrics` (AUROC + AUPRC, tie-aware,
> degenerate/non-finite safe), `capture_auc` (budget-integrated capture, raw + oracle-normalized),
> `oracle_regret` (normalized captured-error gap to the oracle), and `decision_quality_metrics`
> aggregating all four. Wired into `suite.compute_run` (`decision_rows`) +
> `aggregate_decision`, written as `decision_quality.{csv,md}` (Table C) with mean±std across
> seeds. Tests: `tests/test_uq_decision_metrics.py`. Significance (WP-A) tests these metrics too.

**Goal:** evaluate VESP-UQ as the screening tool it is, not as a correlation. Spearman/capture@k
under-sell calibrated covariance and over-index on a single budget.

- **Extend `src/vesp/uq/benchmarking.py`:**
  - `detection_metrics(scores, true_error, high_quantile=0.90)` — treat "top-decile true error" as
    the positive class; return **AUROC** and **AUPRC** (threshold-free, standard, reviewer-expected).
  - `capture_auc(scores, true_error, fractions)` — area under the capture-vs-budget curve (one number
    summarizing all budgets instead of the 20% point).
  - `oracle_regret(scores, true_error, fraction)` — captured-error gap vs the oracle ranking
    (0 = optimal), the operationally meaningful loss.
  - `cost_benefit_curve(scores, true_error, rerun_cost, miss_cost)` — captured-error vs compute spent;
    supports the "compute saved at fixed safety" economic argument (units left model-normalized).
- **Wire into `suite.RANKING_AGG_METRICS`** so all four aggregate mean±std across seeds and flow into
  Table B and the journal report automatically.
- **Outputs:** new columns in `benchmark_summary.csv`; a `decision_quality.png` (ROC + PR + capture
  curve per band).
- **Tests:** `tests/test_uq_decision_metrics.py` — AUROC 0.5 for random scores and 1.0 for the oracle;
  regret 0 for the oracle; capture_auc monotonic bounds; degenerate (all-equal) inputs safe.
- **Acceptance:** Table B reports AUROC/AUPRC/capture-AUC/regret with CIs (via WP-A); the paper can
  state decision quality without leaning on the marginal Spearman gap. **Effort:** M.

## WP-C — Calibration as the core contribution (close/explain L90) — **DONE (metrics); L90 fix gated**

> Implemented in `src/vesp/uq/metrics.py`: `local_radial_frame` (per-point orthonormal radial/
> tangential frame) and `component_calibration_metrics` (radial vs tangential `z_std`/PICP, mean
> **Winkler** interval score, scalar `calibration_error_90`). Wired into
> `plugin.evaluate_calibration` (every band now reports component metrics) and surfaced in the suite
> calibration aggregation + a new "Component-wise calibration" table in `calibration_summary.md`.
> Tests: `tests/test_uq_vector_calibration.py` (orthonormal frame, calibrated→z_std≈1, radial-only
> miscalibration detected, Winkler minimized at true scale). The L90 *fix*: the diagnostic now
> exists (radial vs tangential per band) + the `geometry_calibration` placement finding + per-band
> conformal; a 3-param noise law stays gated on whether the L90 radial z_std shows the 2-param law is
> the binding limit (decided from the real-data run, not assumed). Reporting/L90-resolution section
> belongs to WP-E.

**Goal:** make calibrated covariance the headline and resolve the L90 transfer failure into a clean,
publishable mechanism + the best available fix.

- **Component-wise calibration — extend `src/vesp/uq/metrics.py`:**
  - `component_calibration_metrics(error_vectors, covariances)` — per-axis **radial vs tangential**
    z_std / PICP (rotate errors into the local radial frame using the query position). Physically the
    radial component dominates altitude sensitivity; a reviewer will want it split out. Reuse
    `mahalanobis_squared` machinery; add a scalar **calibration error** (|PICP90−0.90| aggregate) and
    the **interval/Winkler score** for a single comparable number alongside the PICP table.
- **L90 mechanism + fix (consume existing studies, do not re-derive):**
  - Promote the `geometry_calibration.py` finding (`single_surface` brings L90 low z_std 0.257→0.938
    at rel-RMSE 1.056) and the per-band `conformal.py` scale into one **"L90 calibration resolution"**
    section: geometry vs noise-law vs conformal, with the honest verdict that the binding limit is the
    2-parameter noise law and the practical fix is per-band conformal + surface-leaning geometry.
  - Add `uncertainty.noise_model: heteroscedastic_3param` option (`floor + a·h^(-b) + c·h^(-d)` or a
    monotone spline) **only if** WP-C diagnostics show the 2-param law is the binding limit on L90 —
    otherwise document the negative result and keep conformal as the operational fix. Decision gated
    on the diagnostic, not assumed.
- **Outputs:** `calibration_summary.csv` gains radial/tangential columns + Winkler + scalar
  calibration error; a `component_reliability.png`; an `l90_resolution.md` section in the report.
- **Tests:** extend `tests/test_uq_metrics.py` — synthetic calibrated 3D Gaussian → radial/tangential
  z_std ≈ 1, Winkler finite and minimized at the true scale; rotation frame is orthonormal.
- **Acceptance:** the paper's calibration claim is per-component and per-band, with a stated mechanism
  for L90 and a measured best-available fix. **Effort:** M–L.

## WP-D — A respected UQ baseline — **DONE (GP; ensemble deferred)**

> Implemented `src/vesp/uq/baselines/gp.py::GPResidualUQ` — an exact independent-output Gaussian
> process (RBF, median-heuristic lengthscale, per-component evidence-selected noise, Cholesky,
> chunked predict, deterministic subsample for tractability) exposing `predict`,
> `evaluate_calibration` (reusing the *same* `calibration_metrics` / `vector_calibration_metrics` /
> `component_calibration_metrics`), and `score_trajectories`. Comparison runner
> `src/vesp/uq/uq_baseline_comparison.py` + thin `scripts/run_uq_baseline_comparison.py` fit VESP-UQ
> and the GP on the same split and emit `uq_baseline_comparison.{csv,md}` + `uq_baseline_decision.csv`
> (per-band calibration, AUROC/capture-AUC/regret, fit/predict runtime) through the manifest layer.
> Tests: `tests/test_uq_baseline_comparison.py`. No new dependency (torch only). The
> heteroscedastic-ensemble baseline is deferred (GP is the respected baseline reviewers expect; add
> only if a reviewer asks). Note: on the smoke set the GP is actually better-calibrated than VESP-UQ
> in several bands — an honest, useful finding the paper should report.

**Goal:** answer "why an equivalent-source posterior instead of a standard UQ method?" with numbers.

- **Implemented `src/vesp/uq/baselines/gp.py`** (exported through the canonical `vesp.uq.baselines` package):
  - `gp_residual_baseline` — a GP (sparse/inducing-point if N forces it) on the residual force-error
    magnitude vs position, giving predictive mean + std for the same per-band calibration metrics and
    the same trajectory screening. This is the baseline reviewers expect for spatial UQ.
  - `heteroscedastic_ensemble_baseline` (optional, if GP is enough skip) — a small MLP deep ensemble
    predicting per-point error mean + variance, evaluated identically.
  - Both expose the same `predict_uncertainty`-shaped interface so they drop into
    `evaluate_calibration`, the suite, and WP-B decision metrics with no special-casing.
- **New `scripts/run_uq_baseline_comparison.py`** — fits VESP-UQ + GP (+ ensemble) on the same L60/L90
  splits and emits a head-to-head table: per-band z_std/PICP/Winkler, AUROC/capture-AUC/regret,
  **fit + predict runtime**, and memory. The honest framing: VESP-UQ's value is *physics-structured,
  cheap, altitude-extrapolating* covariance — not necessarily lower error than a GP in-support.
- **Outputs:** `outputs/uq_baseline_comparison/uq_baseline_comparison.{csv,md,png}` + manifest.
- **Tests:** `tests/test_uq_baseline_comparison.py` — baselines satisfy the predictive interface;
  comparison runs on the smoke config; calibrated synthetic → GP recovers z_std ≈ 1.
- **Acceptance:** one table places VESP-UQ against a GP on identical calibration + decision metrics
  with runtime; claims stay comparative and honest. **Effort:** L.

## WP-E — Reframed journal report + manuscript scaffolding (closeout) — **DONE**

> Extended `src/vesp/uq/journal_report.py`: STUDY_INPUTS now ingests `significance_summary.csv`,
> `decision_quality.csv`, and the GP-baseline CSVs. The executive summary leads with **calibrated
> covariance as the primary contribution** and demotes scalar ranking to a significance-tested,
> diagnostic use (`significance_verdict` injects the bootstrap-CI verdict). Implemented report sections:
> 3b component-wise calibration, 3c decision quality, 6b supervisor-vs-altitude significance, 9c GP
> baseline (calibration + decision). LaTeX tables: `table_significance`, `table_decision_quality`,
> `table_component_calibration`, `table_uq_baseline`. Claims table adds significance / per-component
> calibration / decision-quality / GP-baseline entries; recommendations fold in the significance and
> baseline findings. Tests: extended `tests/test_journal_report.py`. Missing studies still render as
> honest "pending" lines. Forbidden phrases remain forbidden.

**Goal:** the auto-generated report leads with the defensible story and feeds the manuscript.

- **Extend `src/vesp/uq/journal_report.py`:** new sections for significance (WP-A), decision quality
  (WP-B), component calibration + L90 resolution (WP-C), and the UQ-baseline comparison (WP-D); fold
  the Phase-14 verdict to lead with *calibrated covariance + characterized operation*, demote scalar
  ranking to "matches altitude, value is the covariance" with the CI evidence.
- **LaTeX:** add `table_significance.tex`, `table_decision_quality.tex`, `table_component_calibration.tex`,
  `table_uq_baseline.tex` via the existing `--emit-latex` path.
- **Claims discipline:** update the supported/future-work tables from the new CSVs; keep all forbidden
  phrases forbidden.
- **Acceptance:** `run_journal_report.py` regenerates a report whose executive summary matches the new
  evidence; no hand-entered numbers. **Effort:** M.

---

## Post-plan increments (config tuning + figures + gated decision)

- **Calibration config tuning (validated on held-out, disclosed):** L90 switched to a surface-leaning
  geometry (`shell_alphas: [0.85, 0.95]`, `[640, 640]`) → low-band held-out z_std `0.083 → 0.931`,
  rel-RMSE `1.71 → 1.06`. L60 keeps its baseline geometry (low z_std already ~1.16).
- **Conformal: tested and disabled.** Per-band conformal over-corrected (L90 mid/high → over-confident
  z_std 2.8–5.8, low pulled off 1); global conformal pushed the low band to over-confident 2.5.
  Geometry alone is the stronger, safer fix. `conformal.apply: false` documented in both configs.
  Correctness fix kept: `plugin.evaluate_calibration` now reflects conformal when `apply=True`.
- **Gated L90 noise-law decision: RESOLVED — keep heteroscedastic.** The residual L90 mid/high
  under-confidence (z_std 0.22 / 0.075) is robust to the noise model (`altitude_binned` does not fix
  it and overshoots the low band) because the near-zero high-altitude residual (rmse ~2e-6) sits below
  the predictive floor. This is the conservative/over-covering (safe) direction; chasing z_std=1 risks
  under-coverage for negligible benefit. Reported as a benign limitation, not a defect.
- **Paper figures (WP-B/C/D):** `vesp.uq.figures.render_paper_figures` +
  `scripts/render_paper_figures.py` render `decision_capture_curve`, `component_calibration`,
  `significance_forest`, `uq_baseline_calibration` (PNG+PDF+manifest, placeholder-safe). Wired into
  `run_journal_report.py --figures`. Tests: `tests/test_uq_paper_figures.py`.

## Suggested execution order

1. **WP-A** (significance) — unblocks honest language everywhere; cheapest.
2. **WP-B** (decision metrics) — reframes the contribution; reuses WP-A CIs.
3. **WP-C** (calibration core) — the actual headline; resolves L90.
4. **WP-D** (UQ baseline) — the biggest lift; can parallelize with C.
5. **WP-E** (report) — last; consumes A–D.

## Final deliverables (per Acta Astronautica review expectations)

1. Significance-tested benchmark tables (CIs + p-values).
2. Decision-quality evaluation (AUROC/AUPRC/capture-AUC/regret/cost-benefit).
3. Per-component, per-band calibration with the L90 mechanism + fix.
4. Head-to-head vs a GP/ensemble UQ baseline with runtime.
5. Reframed journal report + LaTeX tables + updated claims table.
6. All outputs manifested (git hash + SHA-256), reproducible from documented commands.
</content>
</invoke>
