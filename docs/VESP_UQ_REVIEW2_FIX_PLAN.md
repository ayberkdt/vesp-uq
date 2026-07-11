# VESP-UQ Review-2 Fix Plan (conformal double-scaling + fail-closed config + fair baselines)

Created: 2026-07-11
Status: R2WP-1..8 implemented 2026-07-11 (see Status Log); evidence sweeps (R2WP-1 table
regeneration, R2WP-4 L60/L90 ablation, R2WP-7 random-vs-spatial multi-seed) still pending on the
overnight runner

This plan responds to the second external review (2026-07-11). It is separate from
`VESP_UQ_REVIEW_RESPONSE_PLAN.md` (which answered the 2026-07-02 review) and narrower than
`VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`. Every review point below was re-verified against the
current tree before being planned; line anchors are from `main` at bef89ae.

The binding claim documents remain `SCIENTIFIC_CLAIMS.md` and `VESP_UQ_LIMITATIONS.md`. Nothing in
this plan expands the claim surface; R2WP-1 *shrinks* it until regenerated.

## Evidence Snapshot (verified anchors)

| # | Review point | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Conformal scale applied twice in the calibration report (P0). | **Confirmed.** | `_predict_covariance_block` already applies conformal to the covariance (`src/vesp/uq/plugin.py:1358`); `evaluate_calibration` takes that block result and applies `_apply_conformal_to_std` / `_apply_conformal_to_covariance` again (`plugin.py:1668-1675`). Reported std ≈ c²σ, reported cov ≈ c⁴Σ. |
| 1a | …but serving paths are internally consistent. | Nuance. | `predict_uncertainty` scales once (`plugin.py:1259`, cσ); `predict_covariance_3x3` scales once (c²Σ, sqrt-diag = cσ). `fit_conformal_calibration` clears `self.conformal_calibration` before predicting (`plugin.py:805-811`), so the *fit* is not contaminated. Only the **report** is wrong — post-conformal PICP/z_std/ellipsoid/sharpness tables overstate coverage and understate sharpness. |
| 2 | Heteroscedastic noise fit device mismatch on CUDA (P1). | **Confirmed.** | `AltitudeNoiseModel.fit` creates `log_a` / `raw_b` without `device=radii.device` (`src/vesp/extensions/probabilistic.py:227-228`). CUDA tests exist but skip on CPU-only CI, so this never fires there. |
| 3 | Config fails open on scientific-setting typos (P1). | **Confirmed.** | Unknown dtype silently → float32 (`plugin.py:264`); unparseable `lambda_l2` silently → 30.0 **and** mutates `reg_method` fixed→lcurve (`plugin.py:285-291`). Central config validation does not reject unknown keys in the `uq` block. Overlaps AWP-2 in `VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`. |
| 4 | Default split is point-level random, not spatial generalization (P1/P2). | **Confirmed for the default chain.** | Outer train/held: `src/vesp/uq/data.py:197`; internal train/val: `plugin.py:992` — both `torch.randperm`. Altitude-controlled / spatial harnesses exist (`altitude_controlled.py`, benchmarking suite) but are not the default evaluation chain, and the calibration report does not label which split regime produced it. |
| 5 | `expected_error` is an RMS magnitude, not E‖e‖ (P2). | **Confirmed.** | `plugin.py:1266`: `sqrt(‖μ‖² + σ²)` with `σ² = tr(Σ)` → this is `sqrt(E[‖e‖²])`, which upper-bounds `E[‖e‖]`. Fine as a ranking score; wrong name when compared against a physical acceleration-error budget. |
| 6 | `radial_expected` drops cross-covariance (P2). | **Confirmed.** | `radial_profile` uses the diagonal approximation `sqrt(Σ_k (r̂_k std_k)²)` (`src/vesp/uq/scoring.py:177-198`); `needs_covariance` deliberately exempts it (`scoring.py:161`). Correct form is `σ_r² = r̂ᵀ Σ_a r̂` and the 3×3 Σ is already available. |
| 7 | GP baseline comparison is not altitude-fair (new point). | **Confirmed.** | `GPResidualUQ` is a stationary RBF with homoscedastic per-component noise (`src/vesp/uq/baselines/gp.py`); it has no altitude-aware noise or nonstationarity, while VESP-UQ gets the altitude noise model + conformal. The docstring says this is intentional, but the *calibration* tables still read as "VESP-UQ beats GP" without an altitude-aware GP control. |
| 8 | Packaging: heavy mandatory deps, untested Python floor (P2/P3). | **Confirmed.** | `pyproject.toml:13-21`: `requires-python >= 3.10` but `torch` and `PyQt6` are core dependencies; CI (`.github/workflows/ci.yml`) tests 3.12 only. |
| 9 | `VESPUQPlugin` god object; ST-LRPS adapter boundary. | Already planned. | AWP-4 and AWP-7 in `VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`. Not duplicated here; R2WP-1's regression test is a prerequisite guard for the AWP-4 decomposition. |

---

## Work Packages

### R2WP-1 — Single conformal application + serving-parity regression test — **P0, S**

Why: every post-conformal calibration table currently published (PICP, z_std, ellipsoid coverage,
sharpness) measures c²σ / c⁴Σ instead of the served cσ / c²Σ. This is the one item that blocks
using conformal tables as paper results.

Decision: keep the scaling inside `_predict_covariance_block` (so the serving paths
`predict_uncertainty` / `predict_covariance_3x3` are unchanged) and **delete** the re-application
in `evaluate_calibration` (`plugin.py:1673-1675`) together with its now-false comment. This is the
minimal, serving-behavior-preserving fix.

Do:

1. Remove the second `_apply_conformal_to_std` / `_apply_conformal_to_covariance` call in
   `evaluate_calibration`; the block output is already scaled.
2. Add the serving-parity regression test (new `tests/test_uq_conformal_parity.py` or extend
   `tests/test_uq_conformal.py`): fit a small plugin with `conformal_apply=True` and a scale
   meaningfully ≠ 1 (e.g. synthetic residuals forcing c ≈ 2), then assert, on the same query set:
   - std used inside `evaluate_calibration` == `predict_uncertainty(...).std_components` (allclose);
   - cov used inside `evaluate_calibration` == `predict_covariance_3x3(...).covariance` (allclose);
   - with conformal **off**, the same identities hold (guards the no-op path).
   Expose the internals via a small hook (e.g. `evaluate_calibration(..., _return_arrays=True)` or a
   private helper both paths share) rather than duplicating the math in the test.
3. Add a directed unit test that would have caught the bug: with scale c, reported
   `mean_pred_sigma` must equal c·(raw σ), not c²·(raw σ).
4. Invalidate stale evidence: mark `benchmarks/vespuq_conformal_validation.md` and
   `conformal_validation_runs/` outputs as **pre-fix, std overstated by c** (banner at top), then
   regenerate via the existing conformal validation script. Note the fix in `CHANGELOG.md`.
5. Re-check downstream statements: any doc sentence citing post-conformal PICP/z_std numbers
   (README, SCIENTIFIC_CLAIMS, journal validation plan) gets re-derived numbers or a "stale,
   regeneration pending" marker. `scripts/run_claim_lint.py` must stay green.

Acceptance: parity test red on the old code, green after; regenerated conformal tables committed or
explicitly marked pending; no doc cites pre-fix post-conformal numbers without a stale marker.

### R2WP-2 — Heteroscedastic fit device correctness — **P1, XS**

Do:

1. `src/vesp/extensions/probabilistic.py:227-228`: create `log_a` and `raw_b` with
   `device=radii.device`.
2. Extend `tests/test_uq_gpu_parity.py` with a `noise_model="heteroscedastic"` `fit_error` case
   (CUDA-marked; skips on CPU CI but runs on the local CUDA box — run it once locally and record
   the result in the status log below).
3. CPU-side guard that does not need CUDA: assert after `fit` that the learned parameters'
   `.device` matches the input tensors' device (trivially true on CPU, but locks the contract and
   documents intent).

Acceptance: heteroscedastic `fit_error` runs on CUDA without device-mismatch; GPU parity suite
covers it.

### R2WP-3 — Fail-closed config for scientific settings — **P1, S** (extends AWP-2)

Why: a typo'd dtype or lambda must abort the run, not silently change the numerical experiment.
The lambda fallback is worst: it swaps the regularization *method* (fixed → lcurve).

Do:

1. `from_config` dtype: accept exactly {`float64`, `double`, `float32`, `single`}; anything else →
   `ValueError` naming the offending value.
2. `lambda_l2`: unparseable value → `ValueError`. Delete the silent `fixed → lcurve` mutation
   entirely.
3. Validate `uq` block keys against an explicit allowlist (schema or dataclass — align with the
   AWP-2 mechanism if that lands first; if AWP-2 already landed, fold these rules into it instead
   of adding a second validator). Unknown key → error listing valid keys. Same for
   `risk.scoring` values (must be in `SCORING_FUNCTIONS`).
4. Any *legitimate* defaulting that remains (absent optional key) must be recorded in the run
   manifest (AWP-1 output) as `config_defaults_applied`.
5. Tests: typo'd dtype raises; typo'd lambda raises; unknown uq key raises; valid config unchanged.
   Sweep existing `configs/*.yaml` through the validator to confirm none rely on fail-open behavior.

Acceptance: the three fail-open sites are fail-closed with tests; all shipped configs validate.

### R2WP-4 — Full-covariance radial projection in `radial_expected` — **P2, S**

Do:

1. Add `radial_profile_full(mean_error, covariance, positions)` computing
   `σ_r = sqrt(r̂ᵀ Σ r̂)`; keep the diagonal version available as an explicit ablation variant
   (`radial_expected_diag`).
2. Make `radial_expected` use the full projection; update `needs_covariance` so it requests the
   3×3 covariance (`scoring.py:161` exemption goes away — accept the compute cost; it is gated per
   scoring mode already).
3. Ablation: re-run the score-variant benchmark comparing full vs diagonal on L60/L90; report the
   ranking delta (Spearman between the two profiles + benchmark table row). If the delta is ~0,
   say so in the report — that itself is a useful robustness statement.
4. Unit test with a deliberately anisotropic, rotated covariance where diagonal and full projections
   disagree by construction.

Acceptance: `radial_expected` uses `r̂ᵀΣr̂`; diagonal kept only as named ablation; delta measured
and recorded.

### R2WP-5 — Rename `expected_error` to an RMS-honest name — **P2, S**

Do:

1. Canonical name: `rms_predictive_error` (formula `sqrt(‖μ‖² + tr(Σ))` in the docstring).
2. Keep `expected_error` as a deprecated alias on the prediction dataclasses/dicts for one release
   (property forwarding + `DeprecationWarning`), so persisted artifacts and external callers do not
   break; scoring-mode *names* (`expected_error`, `expected_epistemic`, …) stay as-is but their
   docstrings state the RMS formula explicitly.
3. Every report/table that prints the quantity next to a physical error budget must label it
   "RMS predictive error magnitude"; grep docs for "expected absolute error"-style phrasing.
4. Update `SCIENTIFIC_CLAIMS.md` wording if it uses "expected error" in the E‖e‖ sense.

Acceptance: no doc or report presents the quantity as an expected absolute error; alias warns;
tests green.

### R2WP-6 — Altitude-fair GP baseline — **P2, M**

Why: the current comparison gives VESP-UQ altitude-aware noise + conformal while the GP gets
neither, so "better calibration than GP" partly measures information access, not method quality.

Do:

1. Add an **altitude-aware GP variant** alongside the vanilla one in `src/vesp/uq/baselines/gp.py`
   (or a sibling module):
   - input feature: augment `(x, y, z)` with `log(h)` (h = r − 1) so the stationary kernel can
     express altitude dependence; and
   - heteroscedastic noise: reuse the same `AltitudeNoiseModel` power-law fit on GP residuals
     (identical post-hoc recalibration budget as the plugin gets).
2. Optionally also give it the same operational conformal scaling on the same held-out set — that
   makes the strongest "same information diet" control.
3. `uq_baseline_comparison.py` reports **three** columns: GP (vanilla), GP (altitude-aware), and
   VESP-UQ, on the same seeds/splits/metrics. Multi-seed mean ± std as today.
4. Rewrite the comparison narrative: the honest claim becomes whatever survives against the
   altitude-aware GP. If the altitude-aware GP matches VESP-UQ calibration, the differentiator to
   emphasize is cost/extrapolation structure — say that, don't bury it.

Acceptance: benchmark emits the three-way table; docs/claims cite the altitude-aware column, not
the vanilla one, wherever superiority is asserted.

### R2WP-7 — Spatial-generalization splits as the default evidence chain — **P1/P2, M–L**

Why: point-level random splits measure interpolation on a smooth correlated field. Paper-grade
generalization claims need spatially disjoint evaluation.

Do:

1. Implement first-class split strategies in `src/vesp/uq/data.py` behind a
   `split: {method: random|altitude_disjoint|angular_block|trajectory_group, ...}` config key
   (fail-closed per R2WP-3):
   - **altitude-disjoint**: train shells vs held shells with a buffer gap;
   - **angular block**: HEALPix-cell (or lat/lon block) holdout with a great-circle buffer;
   - **trajectory-group**: whole trajectories on one side only (for trajectory metrics).
2. The plugin's *internal* train/val split (`plugin.py:992`) inherits the same strategy when the
   outer split is spatial (at minimum: angular-block internal split when outer is angular), so the
   validation-calibrated components don't leak spatial neighbors either.
3. Stamp the split method + parameters into the calibration report and run manifest; a table whose
   split regime is unlabeled is not paper-usable.
4. Evidence runs (overnight runner already supports resumable sweeps): L60 + L90, ≥ 8 seeds,
   random vs altitude-disjoint vs angular-block. Expect coverage/sharpness to degrade under spatial
   splits — report the degradation as the honest generalization result; update
   `VESP_UQ_JOURNAL_VALIDATION_PLAN.md` checkboxes with actual numbers.
5. Cross-surrogate-artifact and residual-spectrum splits stay in the journal validation plan (they
   need new data, not new split code) — link, don't duplicate.

Acceptance: spatial splits selectable from config and stamped in outputs; multi-seed random-vs-
spatial comparison table exists; claims docs distinguish interpolation vs spatial-generalization
evidence.

### R2WP-8 — Packaging + CI floor honesty — **P3, S**

Do (choose per item, both acceptable):

1. Extras split: core deps `numpy/scipy/pandas/pyyaml`; `torch` → `vesp[torch]` (the UQ core
   imports torch, so document that `vesp[torch]` is required for UQ; headless analysis without
   torch only makes sense if the import graph allows it — verify before promising it);
   `PyQt6`+`matplotlib` → `vesp[ui]`; `all` aggregate. `PyQt6` must leave core deps regardless.
2. Python floor: either add 3.10/3.11 to the CI matrix (cheap: unit-test job only) or raise
   `requires-python` to `>=3.12`. Do not ship an untested floor.
3. ST-LRPS adapter extra (`vesp[stlrps]`) is AWP-7's scope — reference only.

Acceptance: `pip install vesp` no longer pulls PyQt6; every advertised Python version has a CI job.

---

## Recommended Order

1. **R2WP-1** (P0 — blocks all conformal paper tables; small diff, big evidence impact)
2. **R2WP-2** (XS, real runtime bug, one-line fix + test)
3. **R2WP-3** (fail-closed config; do before any new evidence runs so manifests are trustworthy)
4. **R2WP-4** then **R2WP-5** (scoring correctness + naming; cheap, independent)
5. **R2WP-6** (altitude-fair GP — needed before citing baseline superiority anywhere)
6. **R2WP-7** (heaviest compute; schedule the sweep on the overnight runner after 1–3 land so the
   regenerated evidence is produced exactly once, post-fix)
7. **R2WP-8** (any time; no science dependency)

Do **not** start the AWP-4 plugin decomposition mid-plan; land R2WP-1's parity test first — it is
precisely the guard that makes the decomposition safe.

## Invariants (binding on every R2WP)

- No claim expansion: `SCIENTIFIC_CLAIMS.md` / `VESP_UQ_LIMITATIONS.md` stay binding;
  `scripts/run_claim_lint.py` green after every WP.
- Any table produced before R2WP-1 that involves post-conformal std/cov is stale until regenerated;
  never mix pre-fix and post-fix numbers in one table.
- Every evidence run records: split method, seeds, config hash, conformal on/off, code version.
- Serving behavior (`predict_uncertainty` / `predict_covariance_3x3` outputs) changes in **no** WP
  except R2WP-4's scoring profile; R2WP-1 changes only the report path.

## Non-Goals

- Plugin decomposition (AWP-4), baseline/noise family consolidation (AWP-5), adapter extraction
  (AWP-7) — tracked in `VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`.
- RTN directional noise promotion — still held per `VESP_UQ_REVIEW_RESPONSE_PLAN.md`.
- New physics claims (curl/Helmholtz, orbit-determination covariance, operational rerun) — out of
  scope; limitations docs already bound them.

## Status Log (fill in as implemented)

| WP | Status | Evidence / commit |
| --- | --- | --- |
| R2WP-1 | **done** (code + tests + stale markers); conformal table regeneration pending | double application removed from `evaluate_calibration` (now shares `_calibration_arrays` with serving); `tests/test_uq_conformal_parity.py`; stale banner on `benchmarks/vespuq_conformal_validation.md`; SCIENTIFIC_CLAIMS note |
| R2WP-2 | **done**, verified on a CUDA machine | `device=radii.device` in `AltitudeNoiseModel.fit`; CUDA-vs-CPU parity + device-contract tests in `tests/test_uq_gpu_parity.py`. Note: on torch 2.5.1 the old code ran via 0-dim cross-device broadcasting (per-iteration syncs), not a hard crash |
| R2WP-3 | **done** | strict dtype via `common.config.get_dtype`; unparseable `lambda_l2` raises (fixed→lcurve mutation deleted); uq-block + sub-block key allowlists; `risk_scoring` validated at construction; `tests/test_uq_config_failclosed.py` incl. shipped-config sweep. Manifest `config_defaults_applied` recording deferred to AWP-1 |
| R2WP-4 | **done** (code); L60/L90 ranking-delta ablation pending | `radial_profile_full` (`r̂ᵀΣr̂`); `radial_expected` needs covariance; `radial_expected_diag` (+`_p95` variant) kept as ablation; anisotropic-disagreement unit test. Synthetic smoke: ranking unchanged (bias-dominated) |
| R2WP-5 | **done** (alias-forward variant) | canonical `rms_predictive_error` property on `UncertaintyPrediction`/`GPPrediction`; RMS formula + Jensen caveat in docstrings, SCIENTIFIC_CLAIMS, and report labels. Deviation from plan: hard field rename deferred (would churn persisted report schemas); no doc presents the quantity as expected absolute error |
| R2WP-6 | **done** (code); real-data three-way tables pending | `AltitudeAwareGPResidualUQ` (standardized log-h kernel feature + same `AltitudeNoiseModel` post-hoc budget); `gp_alt` column in `run_uq_baseline_comparison` CSV/decision tables; narrative requires citing `gp_alt` |
| R2WP-7 | **done** (code); random-vs-spatial multi-seed sweep pending | `data.split.method` = random / altitude_disjoint / angular_block / trajectory_group (fail-closed dispatcher); split info stamped into `fit_info["split"]` + `run_vespuq` report; CSV group-column support; tests in `tests/test_uq_data.py`. Internal-split inheritance (plugin val split following the outer spatial regime) NOT yet implemented — noted as follow-up |
| R2WP-8 | **done** | PyQt6/matplotlib → `ui`/`plots` extras; `requires-python >=3.12` (matches CI); ruff target py312; README install section updated. CI matrix expansion not taken (floor raised instead, per plan option 2) |
