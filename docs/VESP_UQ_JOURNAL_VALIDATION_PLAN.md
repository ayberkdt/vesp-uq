# VESP-UQ Journal Validation Plan (Acta Astronautica readiness)

Status: **implemented harnesses; full real-run verdicts pending**. This document records the
implemented entry points and artifact contracts. It does not change any manuscript claim by itself;
claims are revised only from measured suite outputs (see "Claim discipline").

Guiding constraints (apply to every work package):
- No invented numbers. Every value in a table/figure traces to a script + output file + manifest.
- Deterministic given a seed. All new entry points take `--seed` / `--seeds`.
- Reuse `scripts/compare_risk_baselines.py` and `src/vesp/uq/*` rather than re-implementing
  scoring, metrics, partial correlation, or fraction sweeps.
- Every emitted artifact gets a manifest with git commit hash + SHA-256 checksums.

---

## 0. Repository audit (grounded in current files)

| Claim / Experiment | Exists? | Source | Reproducible? | Needs work |
|---|---|---|---|---|
| L60 / L90 calibration | Yes | `outputs/vespuq_real_lunar*`, `scripts/run_calibration_audit.py`, `src/vesp/uq/audit.py` | Yes | Single seed only |
| Baseline comparison | Yes (strong) | `scripts/compare_risk_baselines.py` (673 L), `src/vesp/uq/baselines/`, `outputs/baselines_L60/L90` | Yes | Single seed; no aggregate tables |
| Partial corr / within-bin Spearman | Yes | `compare_risk_baselines.py` for single-run diagnostics; `scripts/run_altitude_controlled.py` for first-class multi-seed outputs | Yes | Use the multi-seed harness for paper claims |
| Rerun-fraction sweep | Yes | `compare_risk_baselines.py` for single-run diagnostics; `scripts/run_vespuq_benchmark_suite.py` for curves and seed aggregation | Yes | Use the suite outputs for paper claims |
| Source-stability / shell-cancellation | Yes | `scripts/geometry_shootout.py`, `tests/test_shell_cancellation.py`, `outputs/ablation_real_lunar_shells` | Yes | Expand to full sensitivity grid + metrics |
| Regularization sensitivity | Partial | `scripts/regularizer_shootout.py`, `outputs/ablation_real_lunar_regularization` | Yes | Metric set + lambda grid |
| OOD altitude-sweep | Yes | `outputs/altitude_ood`, `outputs/real_lunar_gl0420a_ood` | Yes | — |
| Conformal calibration | Yes | `scripts/run_conformal_validation.py`, `src/vesp/uq/conformal.py`, `conformal_validation_runs/` | Yes | No raw-vs-calibrated table |
| Physical-budget screening | Yes | `scripts/run_physical_budget_screening.py`, `src/vesp/uq/physical_units.py`, `outputs/physical_budget` | Yes | Activation/metadata status report |
| ST-LRPS position-error diagnostic | Yes | `scripts/run_stlrps_propagation.py`, `src/vesp/uq/linear_propagation.py`, `outputs/linear_propagation` | Yes | Single horizon; no multi-horizon |
| MC/STM covariance propagation | Yes | `scripts/benchmark_stm_dispersion.py`, `src/vesp/uq/propagation.py` | Yes | — |
| Online correction | Yes | `scripts/run_force_correction_benchmark.py`, `src/vesp/uq/correction.py`, `outputs/correction` | Yes | — |
| Reproducibility manifest + git hash | Yes | `write_run_artifacts(...)`, `run_manifest.json` everywhere | Yes | SHA-256 checksums are part of the manifest contract |
| **Multi-seed robustness** | Yes (harness) | `scripts/run_vespuq_benchmark_suite.py`, `src/vesp/uq/suite.py` | Yes | Needs full L60/L90 run outputs for paper verdict |
| **Rerun-budget curves (plots)** | Yes (harness) | `src/vesp/uq/suite.py`, `src/vesp/uq/figures.py`, `scripts/run_vespuq_benchmark_suite.py` | Yes | Needs full L60/L90 run outputs |
| **Matched-altitude pair test** | Yes | `scripts/run_altitude_controlled.py`, `src/vesp/uq/altitude_controlled.py` | Yes | Interpret per measured band/seed |
| **Score-variant ablation (15+ variants)** | Yes | `scripts/run_score_ablation.py`, `src/vesp/uq/score_variants.py`, `src/vesp/uq/scoring.py` | Yes | Real-run winner selection remains validation-only |
| **Learned linear/ridge supervisor baseline** | Yes | `src/vesp/uq/learned_supervisor.py`, `src/vesp/uq/baselines/expanded.py`, `scripts/run_expanded_baselines.py` | Yes | Report honestly if learned selector does not win |
| **Raw-vs-calibrated reliability table** | Yes | `scripts/run_calibration_reliability.py`, `src/vesp/uq/calibration_reliability.py` | Yes | Needs current L60/L90 artifacts for final tables |
| **Unified benchmark suite runner** | Yes | `scripts/run_vespuq_benchmark_suite.py`, `src/vesp/uq/suite.py` | Yes | Use `--quick` for smoke, full seeds for claims |
| **Journal report + LaTeX tables/figs** | Yes | `scripts/run_journal_report.py`, `src/vesp/uq/journal_report.py`, `src/vesp/uq/figures.py` | Yes | Missing study CSVs render as pending |
| Makefile target | No (no Makefile) | — | — | Use `pyproject` console-script alias instead |

**Conclusion:** the scientific core and evidence harnesses are now in place. The remaining gap is
not code structure; it is executing the full L60/L90 multi-seed runs and letting the measured
outputs decide the claim wording.

Relevant `src/vesp/uq/` modules to reuse: `baselines/`, `scoring.py`, `metrics.py`,
`selection.py`, `screen.py`, `audit.py`, `conformal.py`, `linear_propagation.py`,
`physical_units.py`, `reporting.py`, `figures.py`, `io/` (`write_run_artifacts`).

---

## Milestones overview

| Milestone | Work packages | Review phases | Answers |
|---|---|---|---|
| **M1 — Incremental value** | WP1–WP4 | 1,2,3,4,13 | "Does VESP-UQ add value beyond altitude, and when?" |
| **M2 — Strengthen + honesty** | WP5–WP8 | 5,6,7,8 | Stronger baselines, score ablation, calibration, source sensitivity |
| **M3 — Scope breadth** | WP9–WP11 | 9,10,11 | Trajectory families, drift horizons, physical units |
| **M4 — Packaging + claims** | WP12 | 12,13,14 | Journal report, LaTeX, claims table, tests, repro |

---

## Milestone 1 — Incremental value (Priority 1)

### WP1 — Unified benchmark runner skeleton — **IMPLEMENTED**
- **Goal:** one entry point that fans out over configs × seeds × selectors × rerun-fractions and
  emits manifested artifacts.
- **Reuse:** `run_baseline_comparison(...)` from `compare_risk_baselines.py`, `write_run_artifacts`
  from `src/vesp/uq/io`, `git_commit_hash()`.
- **Implemented files:**
  - `scripts/run_vespuq_benchmark_suite.py` — CLI:
    `--configs ... --seeds 0 1 2 3 4 --rerun-fractions 0.05 0.1 0.15 0.2 0.3 --selectors ... --out outputs/benchmark_suite/ [--quick] [--dry-run]`
  - `src/vesp/uq/suite.py` — orchestration: per-(config,seed,fraction,selector) run, row schema,
    aggregation (mean±std), manifest assembly, SHA-256 over emitted files, config snapshot, env info.
- **Outputs:** `outputs/benchmark_suite/{benchmark_runs.csv, benchmark_summary.csv,
  benchmark_summary.md, manifest.json, config_snapshot/, env.json}`.
- **Acceptance:** `--quick --dry-run` lists the run matrix without compute; a real `--quick` run on
  `vespuq_smoke.yaml` produces non-empty CSVs + manifest with checksums; rerunning with the same
  seed reproduces `benchmark_runs.csv` byte-for-byte.
- **Depends on:** —

### WP2 — Multi-seed robustness (L60 + L90) — **IMPLEMENTED HARNESS; FULL RUN PENDING**
- **Goal:** dispel the "single split/seed" criticism.
- **Reuse:** WP1 runner; calibration metrics from `src/vesp/uq/audit.py` + `metrics.py`; ranking
  metrics already returned by `compare_baselines(...)`.
- **Implemented:** aggregation helpers in `src/vesp/uq/suite.py` compute mean±std across seeds for:
  - Calibration: RMSE, mean predicted std, z-std, PICP50/68/90, ellipsoid PICP90, low/mid/high band
    metrics, low/high epistemic-std ratio.
  - Ranking: Spearman, (Kendall tau if cheap), capture@{5,10,20,30}, precision@{5,10,20,30},
    lift-over-random, flagged/accepted true-error ratio, mean flagged/accepted true error,
    runtime per trajectory + per point.
- **Outputs:** `benchmark_runs.csv` (one row per seed), `benchmark_summary.{csv,md}` with
  **Table A (calibration robustness)** and **Table B (ranking robustness)** as mean±std.
- **Acceptance:** ≥5 seeds for L60 and L90; std columns populated; summary md renders both tables.
- **Depends on:** WP1.

### WP3 — Rerun-budget curves — **IMPLEMENTED HARNESS; FULL RUN PENDING**
- **Goal:** curve instead of single 10%/20% point.
- **Reuse:** existing `altitude_incremental_value.fraction_sweep`; extend the fraction grid.
- **Implemented:** `src/vesp/uq/figures.py` / `suite.py` plot capture / lift / error-ratio
  vs fraction, per band, per selector, with mean±std bands across seeds.
- **Fractions:** 0.01 (if n permits), 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40.
- **Outputs:** `rerun_budget_curves.csv`, `rerun_budget_curves.png` (+ per-band `_L60`/`_L90`).
- **Acceptance:** monotonic-where-expected curves; honest note if VESP-UQ only wins at some budgets.
- **Depends on:** WP1 (WP2 for seed bands).

### WP4 — Altitude-controlled incremental-value diagnostics — **IMPLEMENTED**
- **Goal:** separate "VESP-UQ = altitude" from "VESP-UQ adds info beyond altitude."
- **Reuse:** `partial_pearson_given_min_radius` + `within_altitude_bin_spearman` already in the
  baseline payload — promote them to first-class multi-seed outputs.
- **Implemented files:** `scripts/run_altitude_controlled.py` + `src/vesp/uq/altitude_controlled.py`:
  - 4.1 within-altitude-bin ranking (5/10 bins by min-altitude/min-radius/low-alt-exposure):
    Spearman of {vespuq, uncertainty_only, domain_support} vs true error, capture@10 per bin,
    true-error variance per bin → `altitude_bin_ranking.{csv,png}`.
  - 4.2 partial correlation (vespuq / uncertainty_only / domain_support given min_altitude)
    → `partial_correlation_summary.csv`.
  - 4.3 matched-altitude pair/group test (nearest-neighbour pairing on min-altitude, compare higher
    vs lower VESP-UQ score → does higher score ⇒ higher true error?)
    → `matched_altitude_pairs.csv`, `matched_altitude_summary.md`.
- **Acceptance:** all three sub-diagnostics emit files for L60 and L90 across seeds; summary states
  whether incremental value survives altitude control.
- **Depends on:** WP2.

---

## Milestone 2 — Strengthen + honesty (Priority 2)

### WP5 — Expanded baseline set — **IMPLEMENTED**
- **Reuse:** `src/vesp/uq/baselines/core.py` (`min_altitude_scores`, `low_altitude_exposure_scores`,
  `knn_p95_scores`, `domain_support_scores`, `altitude_residual_expected_scores`).
- **Implemented (in the `baselines/` package + suite):**
  - 5.1 altitude+uncertainty hybrid `S = z(min_alt_risk) + a·z(uncertainty)`, sweep a∈{0.25,0.5,1,2,4},
    pick `a` on **validation split only**, evaluate on test.
  - 5.2 altitude+OOD hybrid `S = z(min_alt_risk) + b·z(domain_support)`.
  - 5.3 learned linear/ridge/logistic supervisor over features {min_altitude, low_alt_exposure,
    uncertainty_only, domain_support, expected_error, score percentiles}; target = true-error score
    or top-decile label; strict train/val/test separation via `src/vesp/uq/learned_supervisor.py`.
  - 5.4 empirical residual lookup baselines (altitude-bin RMSE, altitude+spatial-bin, kNN residual
    magnitude) instead of an expensive GP.
- **Outputs:** `selector_ablation.{csv,png}` (single table with all baselines).
- **Acceptance:** no test-label leakage (val-only tuning asserted in code/test); learned vs
  hand-designed supervisor reported honestly.
- **Depends on:** WP1; WP4 for fair altitude framing.

### WP6 — Score-variant ablation (VESP-UQ score itself) — **IMPLEMENTED**
- **Reuse:** `src/vesp/uq/scoring.py` already exposes scoring modes (expected, supervisor_rel/abs,
  p95, low-alt integral, etc.).
- **Implemented:** `scripts/run_score_ablation.py` evaluates the review variants (expected_error_only,
  trace_cov, largest_eigenvalue, radial/tangential component, altitude-weighted, ×OOD,
  +noise_floor, percentile/max-over-traj, periapsis-window, top-k, low-alt-segment integral,
  anisotropy, Mahalanobis-if-residuals-available). Metrics: Spearman, capture@{5,10,20},
  precision@{5,10,20}, error ratio, runtime. **Variant selection on validation split only.**
- **Outputs:** `score_variant_ablation.{csv,md,png}`.
- **Depends on:** WP1.

### WP7 — Calibration: raw vs calibrated + reliability — **IMPLEMENTED**
- **Reuse:** `src/vesp/uq/audit.py`, `conformal.py`, `run_calibration_audit.py`.
- **Implemented:** before/after table (raw vs calibrated PICP90, z-std per altitude bin), reliability
  diagrams at nominal {0.50,0.68,0.80,0.90,0.95} for L60/L90 × all/low/mid/high, sharpness metrics
  (mean predicted vs empirical std, |PICP90−0.90|, mean interval width), split-conformal audit
  noting which bins reach target / become conservative.
- **Outputs:** `calibration_summary.{csv,md}`, reliability PNGs.
- **Acceptance:** raw outputs sourced from an actual uncalibrated run, not back-computed.
- **Depends on:** —.

### WP8 — Source geometry & regularization sensitivity — **IMPLEMENTED HARNESS**
- **Reuse:** `scripts/geometry_shootout.py`, `regularizer_shootout.py`,
  `outputs/ablation_real_lunar_{shells,regularization}`.
- **Implemented/extended:** grid over n_sources {320,640,1280,2560}, shell radius/depth, #shells, lambda,
  regularizer type {Ridge/Tikhonov, MaxEnt, truncated-SVD if feasible}. Metrics: relative accel
  RMSE, shell-cancellation ratio, condition number / effective rank, mean predicted std, PICP90,
  runtime, memory.
- **Outputs:** `source_geometry_sensitivity.{csv,md}`, `regularization_sensitivity.{csv,png}`.
- **Acceptance:** cautious interpretation; **no** density-recovery claim.
- **Depends on:** —.

---

## Milestone 3 — Scope breadth (Priority 3)

### WP9 — Trajectory-family diversity — **IMPLEMENTED**
- **Reuse:** `src/vesp/uq/trajectory_families.py` Keplerian generation; `screening.*` config block.
- **Implemented:** family generators (low-alt near-circular, eccentric perilune, polar, equatorial,
  inclined, descent arcs, high-alt transfer, OOD low-alt). Per family: counts, altitude/inclination/
  eccentricity ranges, band, ranking metrics, best baseline, incremental-value verdict.
- **Outputs:** `trajectory_family_summary.{csv,md}`, `trajectory_family_budget_curves.png`.
- **Depends on:** WP3 (budget curves), WP4 (incremental-value verdict).

### WP10 — Force-risk → trajectory-drift multi-horizon diagnostic — **IMPLEMENTED**
- **Reuse:** `scripts/run_stlrps_propagation.py`, `src/vesp/uq/linear_propagation.py`,
  `benchmark_stm_dispersion.py`.
- **Implemented:** split into (10.1) force-error ranking diagnostic [main claim] and (10.2) drift diagnostic
  [future-work], plus short-horizon controlled test at {1 orbit, 12 h, 1 day, 5 days} to show whether
  decorrelation is a horizon/dynamics effect. Null result is acceptable and reported as scope.
- **Outputs:** horizon-sweep CSV + figure; report section.
- **Depends on:** WP4.

### WP11 — Physical-unit budget screening status — **IMPLEMENTED**
- **Reuse:** `scripts/run_physical_budget_screening.py`, `src/vesp/uq/physical_units.py`.
- **Implemented:** with L60/L90 metadata present (`acceleration_units: km/s^2`), run an absolute-mode budget
  screen at e.g. 1e-8 m/s²; emit alarms, flagged fraction, FP/FN if truth available, accepted/rejected
  examples. If a config lacks explicit scaling, emit an explicit "implemented but not activated" note
  to prevent overclaiming.
- **Outputs:** physical-budget report + CSV.
- **Depends on:** —.

---

## Milestone 4 — Packaging + claim discipline (Priority 4 / closeout)

### WP12 — Journal report, LaTeX tables, figures, tests, reproducibility — **IMPLEMENTED GENERATOR**
- **Implemented report:** `outputs/journal_validation_report.md` with the 14 review sections (exec summary →
  supported/unsupported claims).
- **LaTeX tables:** `table_calibration_robustness.tex`, `table_ranking_robustness.tex`,
  `table_expanded_baselines.tex`, `table_score_ablation.tex`, `table_altitude_controlled.tex`,
  `table_source_sensitivity.tex` (emitted by a `--emit-latex` flag on the suite, from the same CSVs).
- **Figures (png + pdf):** `fig_rerun_budget_curve_L60/L90`, `fig_reliability_raw_vs_calibrated_L60/L90`,
  `fig_altitude_controlled_spearman`, `fig_score_ablation`, `fig_source_sensitivity`.
- **System hardening (Phase 13):** config schema validation (extend `tests/test_config_validation.py`);
  unit tests for posterior shapes, covariance PSD, score aggregation, altitude binning, calibration
  metrics, benchmark determinism; smoke tests for L60/L90 (`--quick`); loud failure on missing data;
  `--dry-run`/`--quick`; manifest tracking + SHA-256; README repro section; `pyproject` console-script
  alias `vespuq-benchmark-journal`; artifact-pack command (extend `scripts/build_iac_pack.py`).
- **Consistent labels:** VESP-UQ Supervisor, Min-altitude, Low-altitude exposure, Uncertainty-only,
  Domain support, kNN p95, Random.
- **Depends on:** all prior WPs (consumes their CSVs).

### Claim discipline (Phase 14) — applied only after M1 outputs exist
Decision rule encoded in the report generator from `benchmark_summary.csv`:
- VESP-UQ beats altitude across seeds → "consistent incremental value beyond altitude-only in the
  tested setting."
- Mixed → "matches or modestly improves on altitude heuristics depending on band/metric/budget."
- Altitude wins → "low-altitude exposure dominates the ranking signal; VESP-UQ's value is calibrated
  local covariance rather than superior scalar ranking."
- Force-risk ⊥ position error → "VESP-UQ ranks force-model risk, not long-horizon position error."

Forbidden phrases unless newly supported: "validated operational covariance", "guaranteed risk
bound", "trajectory correction", "physical density recovery", "outperforms all baselines",
"end-to-end ST-LRPS validation".

---

## Suggested execution order
1. WP1 → WP2 → WP3 → WP4 (Milestone 1) — answers the core reviewer question.
2. Apply claim-discipline rule from M1 results before touching the manuscript.
3. M2 (WP5–WP8) and M3 (WP9–WP11) can parallelize.
4. WP12 last (consumes everything).

## Final deliverables (per review)
1. Code-change summary. 2. Benchmark audit report (this doc + `journal_validation_report.md`).
3. Reproducible commands. 4. CSV/MD/LaTeX/figure outputs. 5. Claims supported/unsupported table.
6. Manuscript-update recommendations.
