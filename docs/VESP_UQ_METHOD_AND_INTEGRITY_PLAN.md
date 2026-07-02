# VESP-UQ Method-Strengthening & Integrity Plan (implementation log)

Status: **implemented core integrity + method hooks; measured verdicts pending for the real runs**.
Two intertwined pillars:

> *Stronger method and better results — and, while doing so, build features that guarantee we never
> produce a fabricated metric.*

- **Pillar I — Stronger method (M1–M4):** each justified by a *measured* weakness.
- **Pillar II — Integrity features (G1–G7):** system features that make a fabricated/leaked/invalid
  metric structurally impossible or loudly detectable.

Hard constraints: no invented numbers; deterministic given a seed; selection on validation only;
force-model error only; every artifact manifested (git hash + SHA-256). All new public functions take
explicit args (no global state); all randomness takes a `seed`.

**Environment note for next session:** the C: drive may be full — export
`TMPDIR/TEMP/TMP/MPLCONFIGDIR` to a D: path (e.g. project-local `.tmp_run`) before running anything;
`outputs/` on D: is fine. Run the suite and the GP comparison **one at a time** (concurrent
multi-threaded torch oversubscribes cores and thrashes).

---

## 0. Grounding: the measured weaknesses (the "why")

Tuned-config, 3-seed real GRAIL results:
- Ranking ≈ altitude (L90 `min_altitude` Spearman 0.856 > supervisor 0.820; matched concordance ≈0.5).
- Low band calibrated after the geometry fix (L60 z_std 1.09, L90 0.84); mid/high over-cover
  (L90 high z_std ~0.07) — robust to noise-model choice (conformal & `altitude_binned` over-corrected).
- A GP baseline can match VESP-UQ in-support calibration.
- Integrity is convention, not enforcement.

### Key code anchors (verified)
- Scoring: `src/vesp/uq/scoring.py` — `SCORING_FUNCTIONS`, `score_sigma_profile(...)`,
  `TrajectoryScore`, `_relative_altitude_weight`, `_EXPECTED_MODES`. The plugin builds per-point
  `expected_error`, `sigma`, `epistemic_sigma`, `mean_error`, `std_components` in
  `UncertaintyPrediction` (`plugin.py`), and `score_ensemble(trajectories, scoring=...)` returns
  `list[TrajectoryScore]`.
- Suite: `src/vesp/uq/suite.py` — `compute_run`, `RANKING_AGG_METRICS`, `DECISION_AGG_METRICS`,
  `aggregate_*`, `run_suite`, the `_*_csv` / `_*_md` writers, `compute_significance`.
- Metrics: `src/vesp/uq/benchmarking.py` (`detection_metrics`, `capture_auc`, `oracle_regret`,
  `decision_quality_metrics`), `src/vesp/uq/metrics.py` (`component_calibration_metrics`,
  `vector_calibration_metrics`), `src/vesp/uq/significance.py`.
- Noise models: `src/vesp/extensions/probabilistic.py` (`AltitudeNoiseModel`,
  `BinnedAltitudeNoiseModel`); fit dispatch in `plugin._fit_altitude_noise_model`,
  applied in `plugin.fit` step 5.
- Artifacts/manifest: `src/vesp/uq/io/run_artifacts.py` `write_run_artifacts(...)` →
  `run_manifest.json` with `artifacts[name] = {sha256, bytes, origin}`, `inputs`, `_provenance`.
- GP baseline: `src/vesp/uq/baselines/gp.py` (`GPResidualUQ`, also exported from
  `vesp.uq.baselines`), comparison in
  `src/vesp/uq/uq_baseline_comparison.py`.

**Convention for new integrity code:** put it under a new package `src/vesp/uq/integrity/`
(`__init__.py` re-exporting the public funcs). Tests under `tests/test_integrity_*.py`.

---

## Pillar I — Stronger method

### M1 — Anisotropy / directional risk score (attack the ranking gap)

**Why.** Altitude is a scalar; VESP-UQ uniquely knows the per-point `3x3` covariance (anisotropy,
radial vs tangential) and the bias direction. If any of these rank force error *within* altitude
bins, that is value beyond altitude — the central reviewer question.

**Files.**
- Updated `src/vesp/uq/scoring.py` with per-point *profile builders* (pure functions on the predicted
  arrays) and registered scoring names.
- Updated `src/vesp/uq/plugin.py`: `score_ensemble` / `score_trajectory` pass the extra per-point
  arrays (`std_components`, `covariance` if available, `mean_error`) into the profile builder.
- Updated `scripts/run_score_ablation.py` / `src/vesp/uq/score_variants.py` so ablation variants map
  back to production scoring modes on the validation-to-test path.

**Implemented scoring names (`_DIRECTIONAL_MODES`, registered in `SCORING_FUNCTIONS`):**
- `radial_expected` — expected error projected onto the radial axis:
  `point_risk = |mean_error · r_hat| + sigma_radial`, aggregated p95.
- `anisotropy_gated` — `expected_error * (1 + kappa * (lambda_max/lambda_min - 1))` where
  `lambda_*` are eigenvalues of the `3x3` covariance (anisotropy as a multiplier; `kappa` a fixed
  small constant, e.g. 0.5).
- `largest_eigenvalue` — `sqrt(lambda_max(covariance))` per point, now a first-class scoring name.

**Signatures (scoring.py).**
```python
def radial_profile(mean_error: Tensor, std_components: Tensor, positions: Tensor) -> Tensor: ...
    # returns per-point radial risk (N,); uses local_radial_frame from vesp.uq.metrics
def anisotropy_multiplier(covariance: Tensor, kappa: float = 0.5) -> Tensor: ...
    # (N,) >= 1; eigvalsh on the symmetrized 3x3; clamp_min on lambda_min
```
Route them through `score_sigma_profile` by adding the new branches (they need the new optional
kwargs `covariance: Tensor | None`, `positions: Tensor | None`). `score_sigma_profile` already takes
`mean_error_magnitude`, `expected_error`; extend its signature with `covariance=None, positions=None`
and raise a clear error if a directional mode is requested without them.

**Integration.** `plugin.score_ensemble` currently builds the per-point profile from
`predict_uncertainty`. For directional modes it must also build the `3x3` covariance — reuse
`predict_covariance_3x3` (already query-chunked). Gate the extra covariance build on
`scoring in _DIRECTIONAL_MODES` so default scoring pays nothing.

**Evaluation (no new harness).** Add the names to `run_score_ablation.py`'s variant list and to the
suite's `DEFAULT_SELECTORS` only after they earn it. The decisive metrics already exist:
`within_altitude_bin_ranking`, `matched_altitude_pairs` (suite), `decision_quality_metrics`.

**Integrity guardrail.** Variant selection on the **validation** split (the ablation already does
val→test); headline = test. Wrapped by G2 (no test access during selection) and G4 (placebos still at
chance). If nothing beats altitude *within bins*, report exactly that.

**Tests (`tests/test_uq_scoring.py`).**
- `test_radial_profile_matches_manual_projection` — hand-computed 3-point case.
- `test_anisotropy_multiplier_ge_one_and_isotropic_is_one` — isotropic cov → 1.0.
- `test_directional_scoring_requires_covariance` — raises without covariance/positions.
- `test_directional_scores_finite_on_smoke_ensemble` — via `score_ensemble` on the smoke plugin.

**Acceptance.** Each variant emits val + test Spearman, within-bin Spearman, matched concordance,
capture-AUC, runtime; the "beats altitude within bins" verdict is rule-derived (reuse the suite's
significance machinery for the supervisor-vs-altitude delta on the winning variant).

---

### M2 — Epistemic-targeted screening (rerun where a high-fidelity run helps)

**Why.** Rerun value = where the surrogate is wrong **and** the uncertainty is reducible (epistemic),
not aleatoric. `UncertaintyPrediction.epistemic_fraction` exists but is unused as a screening signal.

**Files.** `src/vesp/uq/scoring.py` (new mode), `scripts/run_score_ablation.py`.

**New mode `expected_epistemic` (add to `_EXPECTED_ONLY_MODES`-like family, relative scale):**
`point_risk = expected_error * (epistemic_fraction ** gamma)`, p95 aggregated. `epistemic_fraction`
= `epistemic_sigma / sigma` (already computed). `gamma` default 1.0, a tunable.

**Signature.** Extend `score_sigma_profile` to accept `epistemic_fraction: Tensor | None` and
`epistemic_gamma: float = 1.0`; new branch. `plugin.score_ensemble` already has `epistemic_sigma` and
`sigma` per point → pass the ratio.

**Integrity guardrail.** `gamma` selected on validation (small grid {0.5,1,2}); oracle never enters
the score (G2 oracle-isolation). Reported honestly if it loses to plain expected-error.

**Tests.** `test_expected_epistemic_reduces_to_expected_when_gamma_zero` (gamma=0 → identical to
expected p95); `test_epistemic_fraction_in_unit_interval`.

**Acceptance.** capture-AUC / oracle-regret on test vs `expected_abs_p95` and `min_altitude`.

---

### M3 — Smooth monotone variance recalibration (fix mid/high over-coverage)

**Why.** Low band calibrated; mid/high over-cover. Conformal and `altitude_binned` both
over-corrected (documented negatives). A **monotone, low-DOF** variance scale `s(r)` fit on
validation residuals can sharpen high altitude toward z_std=1 without swinging a band into
over-confidence.

**Files.** `src/vesp/extensions/probabilistic.py` (new `MonotoneVarianceRecalibrator`),
`src/vesp/uq/plugin.py` (new `noise_model: "monotone_spline"` branch in
`_fit_altitude_noise_model`, applied in fit step 5 and persisted in `_altitude_noise_state` /
`from_state_dict`).

**Algorithm.** On held-out val: compute per-point standardized residual `z_i = ||e_i|| / sigma_pred_i`
and radius `r_i`. Fit a **monotone (isotonic) regression** of a target scale `s(r)` such that the
local empirical z-std → 1, i.e. `s(r_i) ≈ local_std(z | r)`. Use few knots (e.g. 5) + isotonic
projection (PAVA) so `s(r)` is smooth/monotone and cannot overfit per point. Predictive std scaled by
`s(r)` (and covariance by `s(r)^2`), exactly like the conformal hooks
(`_apply_conformal_to_std/_covariance`) — reuse that application path.

**Signature.**
```python
@dataclass
class MonotoneVarianceRecalibrator:
    radii: Tensor       # knot radii (sorted)
    scale: Tensor       # monotone scale at knots, > 0
    @classmethod
    def fit(cls, radii, std_pred, residual_norm, *, n_knots=5) -> "MonotoneVarianceRecalibrator": ...
    def scale_for(self, radius: Tensor) -> Tensor: ...   # interp, clamp to [knot range]
```

**Integrity guardrail (falsifiable gate).** A function
`evaluate_recalibration_gate(before: dict, after: dict) -> dict` (per-band z_std before/after on the
**test** split) returns `accepted: bool`. Accept only if it moves ≥1 band's |z_std−1| down by a
margin AND pushes **no** band above z_std 1.3 on test. If rejected → keep `heteroscedastic`, report as
a failed attempt (like conformal). Default `noise_model` stays `heteroscedastic`; this is opt-in and
bit-identical when off.

**Tests (`tests/test_uq_monotone_recal.py`).**
- `test_recalibrator_monotone_and_positive`.
- `test_calibrated_input_scale_near_one` (already-calibrated synthetic → s(r)≈1).
- `test_overcovering_input_sharpens_toward_one` (synthetic z_std=0.3 band → post z_std closer to 1).
- `test_gate_rejects_overconfidence` (a recal that creates z_std>1.3 is rejected).
- persistence round-trip (save/load) when enabled.

**Acceptance.** per-band before/after z_std on test + the gate verdict, written to the calibration
report.

---

### M4 — Demonstrate the altitude-extrapolation edge vs the GP

**Why.** GP can match VESP-UQ in-support; the defensible edge is graceful degradation where there is
no training support (altitude OOD). Must be shown, not asserted.

**Files.** Extend `src/vesp/uq/uq_baseline_comparison.py` with an OOD mode; new flag on
`scripts/run_uq_baseline_comparison.py` (`--ood-train-band low|mid`).

**Method.** Train both VESP-UQ and `GPResidualUQ` on a restricted-altitude train subset (e.g. only
`r < 1.25`); evaluate per-band calibration on the **unseen high band**. Report z_std / PICP / Winkler
for both; the hypothesis is VESP-UQ degrades less.

**Integrity guardrail.** OOD band excluded from train by radius mask (assert disjoint via G2);
no refit on the OOD band.

**Tests.** `test_ood_split_is_disjoint`; comparison runs on the smoke config with an OOD band.

**Acceptance.** per-band OOD calibration table VESP-UQ vs GP; honest verdict.

---

## Pillar II — Integrity / anti-fabrication features

### G1 — No-orphan-number auditor (strongest anti-fabrication feature)

**Why.** Make every number in the report/LaTeX traceable to a source CSV under a checksummed
manifest, so a hand-entered number cannot survive CI.

**Files.** `src/vesp/uq/integrity/number_audit.py`, `scripts/run_number_audit.py`,
`tests/test_integrity_number_audit.py`.

**Signature.**
```python
def audit_report_numbers(
    report_path: Path, csv_dirs: list[Path], *, tol: float = 5e-3, min_digits: int = 2,
) -> dict:  # {"orphans": [...], "checked": int, "ok": bool}
def audit_latex_tables(tables_dir: Path, csv_dirs: list[Path], *, tol=5e-3) -> dict: ...
```

**Algorithm.** Regex-extract numeric tokens (floats with ≥`min_digits` significant digits; skip
section numbers, years, pure integers like seed counts via a small allowlist of contexts). Build a
multiset of all numeric values found in the source CSVs (parsing every cell). A report number is
"sourced" if some CSV value matches within relative `tol` (handles rounding/`mean ± std` rendering —
split on `±` and check each part). Any token with no match is an **orphan**. Also verify each CSV the
report claims to read is present in a `run_manifest.json` with a matching SHA-256 (reuse manifest
reader). `ok = not orphans and manifests_verified`.

**Edge cases.** Ignore numbers inside fenced code/commands; ignore `n/a`, `pending`; treat percentages
(`20%`) by also matching `0.20`. Document the allowlist in the module.

**Tests.** planted orphan in a fixture report → `ok False` with the orphan listed; a report whose
numbers all come from a fixture CSV → `ok True`; a manifest checksum mismatch → fails.

**Acceptance.** wired into CI after `run_journal_report.py`; fails on any orphan.

---

### G2 — Split-leakage & oracle-isolation guard

**Why.** Prevent (a) selecting/tuning on the test split and (b) the true-error oracle leaking into a
score — the two failure modes that silently inflate results.

**Files.** `src/vesp/uq/integrity/split_guard.py`; wire into `baselines.prepare`,
`suite.compute_run`, and `scoring`/`baselines` score builders.

**Design.**
```python
class Split(enum.Enum): TRAIN; VAL; TEST
@dataclass
class Tagged:           # thin wrapper carrying provenance without copying tensors
    data: Tensor; split: Split; role: str   # role in {"positions","error","oracle",...}
@contextmanager
def assert_no_test_access(): ...   # sets a thread-local flag; reads of TEST-tagged labels raise
def forbid_oracle(*arrays): ...    # raise if any array is tagged role=="oracle" (for score fns)
```
A minimal, **opt-in-at-the-seams** approach: tag the held/test error in `prepare`; wrap the
variant-selection region of `run_score_ablation` / suite selection in `assert_no_test_access()`; call
`forbid_oracle(...)` at the top of `vespuq_scores` / baseline score builders. Keep it lightweight (no
pervasive tensor wrapping — tag at the boundary objects only).

**Tests.** a constructed path that reads test labels during selection raises; the real selection path
passes; `forbid_oracle` raises when handed the oracle tensor.

**Acceptance.** suite + ablation run clean under the guard; leak fixtures trip it.

---

### G3 — Metric-range invariants (loud failure, never silent)

**Why.** A fabricated/buggy metric usually lands out of its valid range; that must abort, not be
written to a table.

**Files.** `src/vesp/uq/integrity/metric_invariants.py`; call at every metric-record point in
`benchmarking.py`, `suite.py` aggregators, `uq_baseline_comparison.py`.

**Signature.**
```python
_DOMAINS = {"auroc": (0.0, 1.0), "auprc": (0.0, 1.0), "capture_rate": (0.0, 1.0),
            "capture_auc_normalized": (0.0, 1.0001), "oracle_regret": (0.0, 1.0001),
            "spearman": (-1.0001, 1.0001), "picp_50": (0,1), ..., "z_std": (0.0, math.inf)}
def validate_metric(name: str, value: float | None, *, where: str = "") -> float | None: ...
    # None/NaN pass through (legitimately missing); finite out-of-domain -> ValueError(where, name, value)
def validate_row(row: Mapping[str, Any], *, where: str) -> None: ...  # validate all known keys
```

**Integration.** Call `validate_row` in `compute_run` before appending `ranking_rows` /
`decision_rows`, and in the aggregators. NaN/None are allowed (a band can be legitimately absent); only
*finite, out-of-domain* values raise.

**Tests.** injected AUROC=1.5 raises with a clear `where`; the real smoke suite passes;
property test: `detection_metrics`/`capture_auc`/`oracle_regret` outputs always in-domain on random
inputs (hypothesis).

**Acceptance.** suite aborts on any out-of-range metric with a precise message.

---

### G4 — Built-in negative controls (placebos) in every benchmark

**Why.** A placebo that must score at chance is a continuous leakage/fabrication detector.

**Files.** `src/vesp/uq/baselines/core.py` (add `label_shuffled_scores`), `suite.py` (add to
`DEFAULT_SELECTORS` + a placebo-assertion in `run_suite`).

**Design.** `random` already exists. Add `label_shuffled` = the true-error values permuted by a
seeded RNG (an *upper-bound-at-chance* control: it has the right marginal but no alignment). After
aggregation, assert per band: `|capture_rate(random) − rerun_fraction| < tol` and
`|spearman(random)| < tol` and the same for `label_shuffled`, with a documented tol (e.g. 0.15 at the
suite's n_orbits) scaled by `1/sqrt(n_orbits)`. A violation raises (fails the run).

**Tests.** placebos pass on synthetic; a deliberately-aligned "placebo" (leak) trips the assertion;
the tolerance scales with n.

**Acceptance.** every suite run self-checks its placebos; report shows them at chance.

---

### G5 — Determinism / reproducibility gate

**Why.** A byte-reproducible result cannot be quietly edited.

**Files.** `suite.py` (`--verify-reproducible` path), CI workflow, `tests/test_integrity_repro.py`.

**Design.** Run one (config, seed) twice; assert byte-identical `benchmark_runs.csv`,
`decision_quality.csv`, `calibration_summary.csv` (drop timing-only columns first via a normalizer).
Already partially covered for `benchmark_runs.csv` — generalize into a helper
`assert_reproducible(out_a, out_b, ignore_cols=("runtime_*",))`.

**Acceptance.** identical bytes on rerun; forced nondeterminism (e.g. unseeded RNG) is caught.

---

### G6 — Forbidden-claim linter

**Why.** Integrity includes not over-claiming in prose.

**Files.** `scripts/run_claim_lint.py`, `src/vesp/uq/integrity/claim_lint.py`,
`tests/test_integrity_claim_lint.py`.

**Design.** A constant `FORBIDDEN = (...)` (validated operational covariance, density recovery,
outperforms all baselines, guaranteed risk bound, trajectory correction, end-to-end ST-LRPS
validation). Scan the report + any `--manuscript path.tex`; a hit fails unless the line carries an
explicit `<!-- evidence: ... -->` tag. Reuse the forbidden list from
`VESP_UQ_JOURNAL_VALIDATION_PLAN.md` §Phase-14.

**Acceptance.** planted phrase fails; clean report passes; CI gate.

---

### G7 — Provenance-completeness checker

**Why.** Close the loop: manifest must match the files on disk.

**Files.** `src/vesp/uq/io/run_artifacts.py` (add `verify_manifest(dir) -> dict`),
`scripts/run_provenance_check.py`, `tests/test_integrity_provenance.py`.

**Design.** Recompute SHA-256/bytes for every file in `run_manifest.json["artifacts"]`; flag missing,
changed, or on-disk-but-unlisted files (orphans). `ok = no_missing and no_changed`. Run over every
study dir in CI.

**Acceptance.** a tampered byte is detected; clean dirs pass.

---

## Execution order (next session)

1. **G3 → G2 → G4** (metric invariants, leakage guard, placebos) — the safety net; ~all small, and
   they protect every later experiment. Wire into the suite first; confirm the existing smoke suite
   still passes under them.
2. **G1** (no-orphan-number auditor) — run it against the current `outputs/journal/` to baseline.
3. **M1 → M2** (ranking attack) behind the guards; a win is then trustworthy by construction.
4. **M3** (variance recalibration) with the falsifiable gate; **M4** (OOD vs GP).
5. **G5, G6, G7** (reproducibility, claim lint, provenance) as the CI closeout.

Each step: implement → unit tests green → `ruff`/`mypy` clean on touched files → run the smoke suite
under the new guards → record the outcome (positive or negative) in this doc.

## The integrity invariant (binding on every M-WP)
No method change is reported as an improvement unless: (a) selected on validation, measured on a
disjoint test split (G2); (b) every reported number is in-domain (G3) and traceable (G1); (c) placebos
still score at chance (G4); (d) the run is byte-reproducible (G5). A change that does not beat its
baseline is written up as a **negative result** — exactly as conformal and `altitude_binned` were.

## Per-WP status log (fill in as implemented)
| WP | status | outcome (measured) |
|----|--------|--------------------|
| G3 | **done** | `integrity/metric_invariants.py` (`validate_metric`/`validate_row`, `METRIC_DOMAINS`, `MetricRangeError`). Wired at every metric-record point: `suite.compute_run` (ranking/decision/calibration rows) + the three aggregators, and `uq_baseline_comparison.comparison_run` rows. None/NaN pass; finite or ±inf out-of-domain aborts with a precise `where`. Tests `tests/test_integrity_metric_invariants.py` (incl. a hypothesis property test that `detection_metrics`/`capture_auc`/`oracle_regret` stay in-domain). Smoke suite passes under it. |
| G2 | **done** | `integrity/split_guard.py` (`Split`, `Tagged`, `tag`/`reveal`, `forbid_oracle`, `assert_no_test_access`; `SplitLeakageError`/`OracleLeakageError`; thread-local, reentrant). Wired at the seams: `suite.compute_run` tags the true-error oracle as TEST and assembles all scores inside `assert_no_test_access()`, revealing it (`allow_test=True`) only for post-selection metric eval + the placebo; `forbid_oracle(...)` at the top of `baselines.vespuq_scores` and `baselines.assemble_baseline_scores`. Tests `tests/test_integrity_split_guard.py` (leak trips, clean path passes, oracle trips, thread-local, reentrant). **Note:** `ablation.py` val→test selection is already structurally val-only; wrapping its region is a low-value follow-up, deferred. |
| G4 | **done** | `baselines.label_shuffled_scores` (true error permuted: right marginal, no alignment) added to `DEFAULT_SELECTORS`; built outside the G2 guard (it deliberately uses the oracle). `suite.assert_placebos_at_chance` (+ `placebo_tolerance`, `PlaceboLeakageError`) asserts every placebo (`random`, `label_shuffled`) scores at chance at the primary budget per band — Spearman≈0 and capture≈rerun_fraction, tol = max(0.2, 3.5/√n_eff). Called in `run_suite`; a violation fails the run; results surfaced in `meta.placebo_checks`. Tests `tests/test_integrity_placebos.py`. Smoke suite placebos pass at chance. |
| G1 | **done** | `integrity/number_audit.py` (`audit_report_numbers`, `audit_latex_tables`, `collect_csv_values`, `verify_csv_manifests`) + `scripts/run_number_audit.py` (exits non-zero on any orphan / manifest issue). Audits every data number against the source-CSV multiset (relative tol with abs floor; `%` matches the `/100` form); skips ints / years / identifiers (`L90`, `p95`, `picp_90`) / refs (`Table 3`, `Phase-14`) / code fences; verifies each CSV is manifested with a matching SHA-256. **Baseline against the real `outputs/journal/journal_validation_report.md`: 543 numbers checked, 0 orphans, 28 CSVs manifested + matching** (with the 11 study dirs the report ingests passed as `--csv-dir`). Tests `tests/test_integrity_number_audit.py` (planted orphan fails, clean passes, tampered-manifest fails, identifiers/refs/code-fences not flagged, `%`↔fraction, LaTeX orphan). |
| M1 | **done (impl); verdict pending real run** | `scoring.py`: profile builders `radial_profile` (`|mean_error·r_hat| + sigma_radial`, diagonal projection via `local_radial_frame`), `anisotropy_multiplier` (`1+kappa(λmax/λmin−1)`, isotropic→1), `largest_eigenvalue_profile` (`sqrt(λmax)`); new first-class modes `radial_expected` / `anisotropy_gated` / `largest_eigenvalue` in `SCORING_FUNCTIONS` (+ `_DIRECTIONAL_MODES`, `_COVARIANCE_MODES`, `needs_covariance`). `score_sigma_profile` extended (`mean_error_vector`/`std_components`/`covariance`/`positions` kwargs, p95-aggregated, clear errors if inputs missing). `plugin.score_ensemble`/`score_trajectory`/`_score_profile` build `predict_covariance_3x3` **only when `needs_covariance`** (default scoring pays nothing); batched==sequential contract preserved. Ablation: `score_variants.PRODUCTION_SCORE_VARIANTS` maps `radial_expected_p95` / `anisotropy_gated_p95` back to production `SCORING_FUNCTIONS`, and the ablation calls `score_sigma_profile` rather than duplicating formulas. Tests `tests/test_uq_scoring.py` + `tests/test_score_variants_expanded.py` cover the registry link. **Within-bin "beats altitude" verdict needs the real 3-seed L60/L90 ablation run (`scripts/run_score_ablation.py`).** |
| M2 | **done (impl); verdict pending real run** | `scoring.py`: mode `expected_epistemic` (`expected_error * epistemic_fraction**gamma`, p95; `gamma=0` ≡ `expected_abs_p95`; `epistemic_fraction = epistemic_sigma/sigma` already on `UncertaintyPrediction`). Wired through `score_sigma_profile` + plugin scoring path; ablation variant `expected_epistemic_p95` added. Tests: gamma=0 reduces to expected-p95, aleatoric-point downweighting, epistemic_fraction∈[0,1] on a fitted plugin. **capture-AUC / oracle-regret vs `expected_abs_p95` & `min_altitude` needs the real ablation run.** |
| M3 | pending | — |
| M4 | pending | — |
| G5 | **done** | `integrity/reproducibility.py` (`normalize_csv_text` drops `runtime_*` cols; `compare_outputs`/`assert_reproducible`; `ReproducibilityError`) + `suite.run_reproducibility_check` (runs the suite twice into `run_a`/`run_b`, compares `benchmark_runs.csv`/`decision_quality.csv`/`calibration_summary.csv`) + `--verify-reproducible` flag on `run_vespuq_benchmark_suite.py` (exit-coded). Tests `tests/test_integrity_repro.py` incl. a smoke double-run that is byte-identical (3 tables) and a forced metric-diff that raises. |
| G6 | **done** | `integrity/claim_lint.py` (`FORBIDDEN` 7 patterns from `SCIENTIFIC_CLAIMS.md` "do not claim" + Phase-14: validated operational covariance, density recovery, "outperforms all baselines", guaranteed risk bound, trajectory correction, end-to-end ST-LRPS validation, learned/generative noise model; `scan_text`/`lint_report`) + `scripts/run_claim_lint.py`. A hit is excused by an `<!-- evidence: -->`/`% evidence:` tag or a disclaimer (`future work`, `not validated`, …) so the claims table can list a forbidden claim as future work; code fences ignored. Tests `tests/test_integrity_claim_lint.py`. **Baseline: real `journal_validation_report.md` → 0 violations.** |
| G7 | **done** | `io/run_artifacts.verify_manifest(run_dir)` (re-hashes every manifest artifact → verified/changed/missing; on-disk-but-unlisted reported, non-fatal; `ok = no missing/changed`) + `scripts/run_provenance_check.py`. Tests `tests/test_integrity_provenance.py` (clean verifies, tampered byte → changed, deleted → missing, stray file → unlisted-but-ok, no-manifest → not ok). **Baseline: `outputs/benchmark_suite` 19/19 verified, `uq_baseline_comparison` 4/4 verified, all clean.** **Follow-up (separate task):** `journal_report.py` writes its `.md`/`.tex` with no manifest → `outputs/journal/` is flagged NO MANIFEST; route it through `write_run_artifacts` to close G1/G7 on the journal dir. |
