# VESP-UQ Next Development Plan (post review-2 fixes)

Created: 2026-07-11, immediately after R2WP-1..8 landed on `main` (bef89ae..2642af5).
Status: DRAFT — nothing started.

This plan sequences what comes next across the open plan documents, now that the review-2 code
work is done. It supersedes nothing: each item still lives in its home document
(`VESP_UQ_REVIEW2_FIX_PLAN.md`, `VESP_UQ_METHOD_AND_INTEGRITY_PLAN.md`,
`VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`, `VESP_UQ_CLEANUP_PLAN.md`,
`VESP_UQ_PAPER_RIGOR_PLAN.md`) — this is the cross-plan ordering and the definition of "done
enough to write the paper". The binding claim documents remain `SCIENTIFIC_CLAIMS.md` and
`VESP_UQ_LIMITATIONS.md`.

## Where things stand (verified 2026-07-11)

- **Review-2 (R2WP-1..8): all code landed.** Full suite 715 passed / 1 skipped; ruff + mypy
  clean; CUDA parity verified. What remains from that plan is *evidence*, not code.
- **Stale evidence:** `benchmarks/vespuq_conformal_validation.md` carries a STALE banner (pre-fix
  double-scaled numbers) and must be regenerated before any conformal claim.
- **Integrity plan:** G1–G7 + M1/M2 done; **M3** (monotone variance recalibration) and **M4**
  (altitude-OOD edge vs GP) are spec'd but unimplemented.
- **Architecture plan:** AWP-5 done, AWP-6/7 deferred with triggers; **AWP-1** (manifested
  journal report), **AWP-2** (typed config), **AWP-4** (plugin decomposition) pending. AWP-4 is
  now unblocked: the R2WP-1 serving-parity test is exactly the guard the decomposition needed.
- **Cleanup plan:** Items 1–3 pending (spearman/pearson wrapper consolidation, dead ST-LRPS
  import fallback, MU_MOON constant decision).
- **Journal validation plan:** harnesses implemented; full real-run verdicts pending.

## Phase E — Evidence runs (compute-bound; unblocks every claim)

These were deliberately deferred until after the fixes so each number is produced exactly once,
post-fix. E1–E4 can be batched into one resumable overnight session (`run_overnight_metrics.py`
pattern); every output must carry the split/config stamp introduced in R2WP-7.

### E1 — Regenerate the conformal validation tables — **highest priority**
Run `scripts/run_conformal_validation.py` on the L60 and L90 configs (same bands/seeds as the
stale tables). Replace the stale tables in `benchmarks/vespuq_conformal_validation.md` (keep the
banner as a historical note above the OLD numbers or move them to an appendix), update the
SCIENTIFIC_CLAIMS stale-marker sentence, and re-derive the pass/fail verdict per band.
Expectation: post-fix conformal PICP/z_std will look *sharper* than the stale tables (the old
report overstated std by c) — the acceptance verdict may genuinely change; report either way.
- Acceptance: no STALE banner left on cited numbers; `run_claim_lint.py` green.

### E2 — Three-way GP comparison on real data (gp_alt verdict)
`scripts/run_uq_baseline_comparison.py` on L60 + L90, ≥8 seeds. The new `gp_alt` column is the
one that matters: whatever calibration/decision superiority survives against the altitude-fair
GP is the honest claim; where `gp_alt` matches VESP-UQ, the claim shifts to physics-structured
covariance / extrapolation / cost.
- Acceptance: three-column tables committed under `benchmarks/`; claims docs cite `gp_alt`.

### E3 — Radial full-vs-diag ranking ablation
`scripts/run_score_ablation.py` (variants now include `radial_expected_p95` and
`radial_expected_diag_p95`) on L60 + L90 multi-seed. Report the Spearman delta between the two
rankings. If the delta is ~0 on real data too (synthetic smoke was bias-dominated), say exactly
that — it is a robustness statement, not a failure.
- Acceptance: one ablation table row + a sentence in the benchmark notes.

### E4 — Random-vs-spatial split sweep
L60 + L90, ≥8 seeds, `data.split.method` ∈ {random, altitude_disjoint, angular_block}
(trajectory_group needs grouped CSVs — only if the data carries `traj_id`). Expect coverage and
sharpness to degrade under spatial splits; the degradation table IS the honest generalization
result. Update `VESP_UQ_JOURNAL_VALIDATION_PLAN.md` verdicts with the measured numbers.
- Acceptance: per-split calibration tables, split stamp visible in every artifact; journal
  validation plan updated from "pending" to measured verdicts.

## Phase F — Method follow-ups (small-to-medium code)

### F1 — Internal val split inherits the outer spatial regime (review-2 leftover) — **S**
The plugin's internal train/val split (`plugin.py`, `torch.randperm` in fit) is still random even
when the outer split is spatial, so validation-calibrated components (altitude noise, conformal,
supervisor) can leak spatial neighbors. Minimal design: `fit()` gains an optional
`val_mask`/`val_indices` argument; `run_vespuq`/`prepare` compute it with the same
`split_uq_samples_by_config` machinery (method inherited from `data.split`, applied to the train
subset) and pass it down. Stamp `fit_info["internal_split"]`. Bit-identical when not provided.
- Do this BEFORE E4 if cheap, otherwise record in E4's caveats that internal splits were random.

### F2 — M3: monotone variance recalibration — **M** (spec in integrity plan)
Now unblocked by R2WP-1 (calibration reports are trustworthy again — fitting a recalibrator to
the old double-scaled reports would have been garbage-in). Implement
`MonotoneVarianceRecalibrator` + the falsifiable acceptance gate exactly as spec'd
(`VESP_UQ_METHOD_AND_INTEGRITY_PLAN.md` §M3). Evaluate on the E1 outputs.
- Acceptance: per-band before/after z_std on test + gate verdict in the calibration report;
  rejected → documented negative, default stays `heteroscedastic`.

### F3 — M4: altitude-OOD edge vs GP — **S–M** (spec in integrity plan)
Extend `uq_baseline_comparison` with the OOD mode (`--ood-train-band`). One change to the
original spec: the comparison must now include **`gp_alt`** — the hypothesis "VESP-UQ degrades
less in the unseen band" is only interesting against the altitude-fair GP. Note `gp_alt`'s
altitude noise law extrapolates its power law too, so this is a real contest.
- Acceptance: per-band OOD calibration table (vespuq / gp / gp_alt); honest verdict either way.

### F4 — Cleanup plan items 1–3 — **S**
From `VESP_UQ_CLEANUP_PLAN.md`, order: Item 2 (delete dead identical ST-LRPS import fallback) →
Item 1 (consolidate the four spearman/pearson wrappers onto `ranking.py` cores, preserving each
guard) → Item 3 (MU_MOON: confirm the `lunaris` value, align the `worst_case` fallback; whether
core `lunar.py` adopts DE430 is the author's physics decision — flag, don't auto-fix).

## Phase G — Architecture (from the architecture plan, in leverage order)

### G1 — AWP-1: manifest the journal report + `config_defaults_applied` — **XS–S**
Also closes the R2WP-3 deferral: legitimate config defaulting gets recorded in the run manifest.
Do before the E-runs if trivial, so the regenerated evidence is born manifested.

### G2 — AWP-2: typed config with unknown-key rejection — **S**
Fold the R2WP-3 allowlists (`_UQ_CONFIG_KEYS` etc. in `plugin.py`) into the central schema in
`vesp.common.config` so there is ONE validator, not two. Behavior already fail-closed; this is
consolidation, not new policy.

### G3 — AWP-4: decompose `VESPUQPlugin` — **M**, behavior-preserving
Now safe to start: the conformal parity test + persistence round-trip tests are the regression
net. Target seams (from the architecture plan): PosteriorFitter / NoiseCalibrator /
ConformalCalibrator / CovariancePredictor / DomainSupportModel / RiskScorer behind the existing
facade. Hard rule: no public API or on-disk state-format change without a version bump.

### (deferred, unchanged triggers)
AWP-6 (covariance/scoring scale-out) — trigger: >~5k sources or directional scoring default.
AWP-7 / ST-LRPS `vesp[stlrps]` extra — trigger: packaging/CI need. R2WP-8's extras split is the
natural precedent when it fires.

## Phase H — Paper assembly (after E + F2/F3)

From `VESP_UQ_PAPER_RIGOR_PLAN.md` (WP-A/WP-B done): regenerate the journal report
(`scripts/run_journal_report.py`) from the post-fix evidence, then write the calibrated-covariance
narrative. The paper's evidence spine is: E1 (conformal, honest), E2 (gp_alt three-way), E4
(spatial generalization), plus M3/M4 verdicts (positive or documented-negative). The R2WP-5 RMS
naming rules apply to every table caption. `expected_error`-style quantities are ranking scores;
budgets use the RMS label.

## Recommended order

1. **G1** (manifest, XS — so evidence is born manifested) → **F1** (internal split, S — so E4 is
   leak-free)
2. **E1–E4** as one overnight batch (E1 first within the batch; it gates every conformal claim)
3. **F2** (M3) and **F3** (M4) — both consume E-run infrastructure; F3 extends E2's runner
4. **F4** + **G2** (cleanup + config consolidation; independent, fill-in work)
5. **G3** (AWP-4 decomposition) — last, on a quiet tree, behavior-preserving
6. **H** (paper) once 1–3 have verdicts

Rationale: everything above the paper is either evidence generation or making evidence
trustworthy; architecture consolidation (G2/G3) deliberately follows the evidence wave so no
refactor lands mid-measurement.

## Invariants (binding)

- No claim expansion; `run_claim_lint.py` green after every phase.
- Every evidence artifact records: split method (outer AND internal once F1 lands), seeds,
  config hash, conformal on/off, code version. Unstamped tables are not paper-usable.
- M3/M4 negative outcomes are documented negatives, not silent drops.
- G3 changes no public API and no persisted state format without a version bump.

## Status log

| Item | Status | Evidence / commit |
| --- | --- | --- |
| G1 (AWP-1 manifest + defaults recording) | pending | |
| F1 (internal split inheritance) | pending | |
| E1 (conformal table regen) | pending | |
| E2 (gp_alt three-way, real data) | pending | |
| E3 (radial full-vs-diag ablation) | pending | |
| E4 (random-vs-spatial sweep) | pending | |
| F2 (M3 monotone recalibration) | pending | |
| F3 (M4 OOD edge incl. gp_alt) | pending | |
| F4 (cleanup items 1–3) | pending | |
| G2 (AWP-2 typed config consolidation) | pending | |
| G3 (AWP-4 plugin decomposition) | pending | |
| H (paper assembly) | blocked on E + F2/F3 | |
