# VESP Architecture Improvement Plan

Created: 2026-07-01
Status: **AWP-5 implemented; remaining WPs pending/deferred**

Scope: the *architecture* of the maintained VESP-UQ surface (`src/vesp/uq`, `src/vesp/common`,
`src/vesp/extensions`, `src/vesp/ui`) plus its scientific-modeling ceiling. This is a
whole-system architecture audit, not a bug list — the code is clean (1 TODO in the whole
maintained tree, lint/mypy/tests green). The risks below are *structural*: places where the
current shape will resist the next change, silently mislead, or cap what the method can claim.

It is deliberately layered on top of the completed plans and does not re-open their claim
boundaries:

- `VESP_UQ_METHOD_AND_INTEGRITY_PLAN.md` (M1/M2 + G1–G7 done; **M3/M4 pending**)
- `VESP_UQ_NEXT_STEPS.md` (N1–N22 done; research backlog)
- `VESP_SYSTEM_HARDENING_PLAN.md` (H1–H8 done)
- `SCIENTIFIC_CLAIMS.md` / `VESP_UQ_LIMITATIONS.md` remain binding: no position-error
  prediction, no validated operational covariance, no invented units.

---

## 0. Diagnosis snapshot (verified anchors)

| # | Problem point | Evidence (verified) | Class |
|---|---------------|---------------------|-------|
| A1 | **`VESPUQPlugin` is a god object** | `src/vesp/uq/plugin.py` = 1919 lines, ~15 public methods (fit/predict/score/save/load/update + conformal + supervisor + covariance + noise + domain-support), state mutated imperatively across `fit_error` (`self.posterior`, `self.conformal_calibration`, `self.calibrated_supervisor`, `self.train_positions`, `self._domain_scale`, …) | maintainability |
| A2 | **Config is an untyped `dict` threaded everywhere** | `from_config(config: dict)`; **87** `config.get(...)` calls in `plugin.py`; `common/config.validate_config` checks *values* but never rejects **unknown keys**, and covers `body/model/loss` only — the entire `uq.*` block (scoring, conformal, covariance_mode, query_chunk_size, noise_model…) is unvalidated → a YAML typo silently falls back to a default | reliability |
| A3 | **`uq` package fragmentation / overlapping families** | Current status: the former scattered baseline modules are consolidated under `vesp.uq.baselines`; the RTN prototype workflow is merged into `rtn_noise.py`; drift entry points document their separate responsibilities; score-ablation variants reference the production scoring registry. | DRY / discoverability |
| B1 | **Modeling ceiling: over-coverage unfixed, edge undemonstrated** | `M3` (monotone variance recal for mid/high over-cover) and `M4` (GP OOD edge) are `pending` in the integrity plan; L90 ranking does **not** beat altitude; the noise law is a **2-param** altitude power law that "transfers usability not sharpness" (N11) | scientific capability |
| B2 | **Single fixed source geometry per fit** | geometry is chosen by an *offline sweep* (`geometry_calibration.py`), not integrated into the posterior; `make_shell_sources` is fixed at `fit` time; no per-fit geometry selection or model averaging | scientific capability |
| C1 | **Headline artifact escapes provenance** | `journal_report.py` writes `.md`/`.tex` with **no manifest** → `outputs/journal/` flagged NO MANIFEST (G7 log). G1/G7 (no-orphan-number + provenance) therefore do **not** cover the paper's own report | integrity |
| D1 | **Perf cliffs on the covariance/scoring path** | dense operator + `exact` covariance is O(sources²) memory; directional scoring (M1: `radial_expected`/`anisotropy_gated`) now *forces* a `predict_covariance_3x3` build; no matrix-free path (backlog beyond 5–10k sources); env fragility documented (torch thread oversubscription, C: fills, `TMPDIR` redirect) | scalability |
| E1 | **52k-line adapter liability** | `src/vesp/adapters/st_lrps` = 51.8k lines, **zero tests**, external undeclared `lunaris` dep, not importable in a clean env (`importorskip` → skips in CI). It dwarfs the maintained tree (17.8k) and is a packaging/search/maintenance drag | boundary hygiene |

---

## 1. Prioritization (leverage × risk)

**Tier 1 — do first (protect the paper + stop silent misconfig):**
- **C1** (manifest the journal report) — closes the one hole in an otherwise complete integrity net; XS effort, high credibility payoff.
- **A2** (config schema + unknown-key rejection) — cheapest way to eliminate a whole class of silent "why did my knob do nothing" failures across every run.
- **B1** (M3 + M4) — the *scientific* deliverables that decide whether the central reviewer question ("value beyond altitude / graceful OOD degradation") is answered. Already spec'd; needs implementation + the falsifiable gate.

**Tier 2 — structural health (unlocks safe future change):**
- **A1** (decompose the god object) — carve `VESPUQPlugin` into a fitted-state dataclass + focused collaborators without changing behavior.
- **A3** (consolidate the baseline/noise/drift families) — collapse the 4 baseline modules and the rtn/drift pairs behind one registry each.

**Tier 3 — scale & boundary:**
- **D1** (matrix-free covariance + lazy directional builds) — only when sources exceed ~5k or directional scoring becomes default.
- **B2** (geometry-aware posterior) — a research lift; sequence after M3/M4 land.
- **E1** (adapter boundary) — extract to an optional package / `extras` so the maintained tree stands alone.

---

## 2. Work packages

### AWP-1 — Manifest the journal report (C1) — **XS**
**Why.** The integrity architecture (G1 no-orphan-number, G7 provenance) is only as strong as its
weakest artifact, and the *headline* artifact — the manuscript report — is currently unmanifested.

**Do.** Route `journal_report.py`'s `.md`/`.tex`/`.csv` outputs through
`vesp.uq.io.run_artifacts.write_run_artifacts` (same path the suite already uses), so
`outputs/journal/run_manifest.json` exists with SHA-256 + byte size per file and the consumed
study CSVs as `inputs`. Then re-run `scripts/run_provenance_check.py` and
`scripts/run_number_audit.py` against `outputs/journal/` in CI.

**Acceptance.** `verify_manifest(outputs/journal)` → `ok: True`; G1 baseline (543 numbers, 0
orphans) now runs *with* a verified manifest instead of loose CSVs; regression test that the
journal dir is manifested.

---

### AWP-2 — Typed config with unknown-key rejection (A2) — **S**
**Why.** `validate_config` guards `body/model/loss` values but silently ignores unknown or
misspelled keys, and the whole `uq.*` block is unchecked. A typo (`covarianace_mode`,
`scoring: expected_p95x`) degrades to a default with no warning — the worst failure mode for a
reproducibility-first project.

**Do.**
- Introduce frozen dataclasses (or `TypedDict` + a validator) for the maintained config surface:
  `UQConfig` (scoring, conformal, covariance_mode, query_chunk_size, noise_model, risk.*),
  layered under the existing `body/model/loss`. Keep YAML as the on-disk format.
- Add **strict-mode** key checking to `validate_config`: any key not in the known schema raises
  (with a "did you mean…" suggestion via close-match). Provide `--allow-unknown-keys` escape hatch
  for forward-compat experiments only.
- Have `VESPUQPlugin.from_config` consume the typed object; shrink the 87 `.get()` sites to
  attribute access at the boundary.

**Acceptance.** A planted typo in a `uq.*` key fails `validate_config` with the suggestion; every
shipped config under `configs/vespuq/` validates clean; `from_config` no longer reads raw dicts;
mypy sees typed fields. No behavior change on valid configs (bit-identical smoke artifacts).

---

### AWP-3 — Land M3 (variance recalibration) + M4 (GP OOD edge) (B1) — **M**
**Why.** These are the two *scientific* gaps that cap the paper's claims. They are already
fully specified in `VESP_UQ_METHOD_AND_INTEGRITY_PLAN.md` §M3/§M4 with signatures, tests, and
falsifiable gates; they just need implementation under the existing G2/G3/G4/G5 guards.

**Do.** Implement exactly as spec'd:
- **M3** `MonotoneVarianceRecalibrator` in `extensions/probabilistic.py` + `noise_model:
  "monotone_spline"` branch in `plugin._fit_altitude_noise_model`, reusing the conformal
  application path (`_apply_conformal_to_std/_covariance`). Ship the **falsifiable gate**
  `evaluate_recalibration_gate` (accept only if ≥1 band's |z_std−1| drops AND no band exceeds
  1.3 on **test**). Default stays `heteroscedastic`; opt-in, bit-identical when off.
- **M4** OOD mode in `uq_baseline_comparison.py` + `--ood-train-band` flag: train VESP-UQ and
  `GPResidualUQ` on a radius-restricted subset, evaluate per-band on the unseen band, report
  z_std/PICP/Winkler for both. Assert train/OOD disjointness via G2.

**Acceptance.** per-band before/after z_std + gate verdict written to the calibration report and
picked up by G1; OOD comparison table (VESP-UQ vs GP) with an honest verdict (a null is written
up as a negative result, per the integrity invariant). Update the M3/M4 rows in the integrity
plan's status log.

---

### AWP-4 — Decompose `VESPUQPlugin` (A1) — **M** (behavior-preserving)
**Why.** A 1919-line class with imperatively-mutated state is the single biggest barrier to safe
change: every new method (M1/M2 already did this) reaches into shared mutable fields, and the
fit invariants ("conformal reset before refit", "domain scale invalidated") are implicit.

**Do (strangler-fig, no behavior change).**
- Extract a frozen `FittedState` dataclass holding the fit outputs (`posterior`, noise model,
  conformal calibration, calibrated supervisor, train/val geometry, `fit_info`). `fit_error`
  returns/assembles one immutable object instead of scattering `self.X = …`.
- Split the collaborators the god object currently inlines:
  `NoiseCalibrator` (altitude/binned/monotone), `ConformalCalibrator`, `SupervisorCalibrator`,
  `DomainSupport`, `CovariancePredictor`. `VESPUQPlugin` becomes a thin facade delegating to them
  — the public API (`fit/predict/score/save/load/update`) is unchanged and still the single entry.
- Persistence (`state_dict/save/load`) serializes `FittedState`, not ad-hoc attributes → the
  round-trip contract (`tests/test_uq_plugin_persistence.py`) becomes structural.

**Acceptance.** All existing plugin/persistence/scoring tests pass unchanged; predictions,
covariances, and scores are float-identical pre/post refactor; each collaborator is unit-testable
in isolation; `plugin.py` drops below ~800 lines with the rest in named modules.

---

### AWP-5 — Consolidate baseline / noise / drift families (A3) — **S–M**
**Why.** Four baseline modules and the rtn/drift pairs fragment one concept across files, making
"where do I add a baseline?" ambiguous and inviting copy-paste drift (the RNG-aliasing placebo bug
that G4 caught lived exactly in this kind of duplicated selector code).

**Do.**
- One `BASELINE_REGISTRY` (name → builder) behind a single `baselines/` subpackage; fold
  the former GP, assembly, expanded, and cheap heuristic modules into it as registered entries.
- Merge the RTN prototype artifact workflow into `rtn_noise` (numerical core + artifact workflow as two
  functions, one module) and document `drift_boundary` vs `drift_horizon` at the top of each (or
  merge if they share the horizon computation).
- Same registry pattern audit for `scoring.SCORING_FUNCTIONS` vs `score_variants.SCORE_VARIANTS`
  (these already partly share builders — make the ablation variants *reference* the production
  registry, never redefine).

**Acceptance.** module count in `uq/` drops; a new baseline is one registry entry + one test; no
import breaks (re-exports covered by a test); DRY-debt note added to the hardening plan's audit
outcome.

---

### AWP-6 — Scale the covariance/scoring path (D1) — **M**, *defer until needed*
**Why.** `exact` covariance is O(sources²) and directional scoring forces covariance builds; fine
today (≤~1k sources) but a cliff for the geometry-aware or larger-source future (B2, backlog).

**Do.** Add a matrix-free / low-rank covariance predictor (CG or Lanczos on `AᵀA + λI`) behind the
existing `uq.covariance_mode` enum (`exact | diagonal | lowrank` already exist — add `matrixfree`);
make directional scoring request only the diagonal/low-rank projection it actually uses
(`radial_profile` needs the radial variance, not the full 3×3). Add the thread-oversubscription /
`TMPDIR` guidance from the integrity plan into a `scripts/` preamble helper so large runs stop
tripping the environment.

**Acceptance.** `matrixfree` matches `exact` within tolerance on the smoke config; directional
scoring on a large ensemble no longer materializes full 3×3 when a projection suffices; a
documented cost curve.

---

### AWP-7 — Stand the maintained tree alone from the adapter (E1) — **M**, *boundary hygiene*
**Why.** 52k untested lines with an undeclared external dep is a packaging, search, and audit
drag on an otherwise tight 18k maintained core; it triples the tree a new contributor must ignore.

**Do.** Move `src/vesp/adapters/st_lrps` behind an optional install extra (`vesp[stlrps]`) or a
sibling namespace package, so `pip install -e .` for the UQ layer never pulls it. Keep the
documented seam (`load_surrogate_force_model`, N5) and its skip-guarded boundary test. No refactor
of the adapter internals — just relocate the boundary so the maintained tree is self-contained.

**Acceptance.** a clean `vesp` install imports/tests without the adapter present; the seam test
still runs where the adapter *is* installed; README/packaging note the split.

---

## 3. Recommended order

`AWP-1 (XS) → AWP-2 (S) → AWP-3 (M, scientific) → AWP-4 (M, refactor) → AWP-5 (S–M)`, then
**defer** `AWP-6` and `AWP-7` until a concrete trigger (sources > 5k, or a packaging/CI need).

Rationale: AWP-1/AWP-2 are cheap and remove silent-failure surface that would otherwise
contaminate everything after them. AWP-3 is the highest scientific leverage and is already
spec'd. AWP-4 is safest to do *after* AWP-2 (typed config shrinks the god object's surface) and
*before* AWP-5 (the registry consolidation is cleaner against decomposed collaborators). AWP-6/7
are real but not on the critical path to a stronger, more-credible paper.

## 4. Invariants (binding on every AWP)

- **No behavior change without a measured reason.** Refactors (AWP-4/5) must be bit-/float-identical
  on the smoke artifacts; capability changes (AWP-3) go through the integrity gates (G2 selection
  on val, G3 in-domain, G4 placebos at chance, G5 byte-reproducible) and are written up as a
  **negative result** if they don't beat baseline.
- **No new claim surface.** Nothing here loosens `SCIENTIFIC_CLAIMS.md`.
- **Every new artifact is manifested** (the lesson of C1): new outputs go through
  `write_run_artifacts` from the start.

## 5. Status log (fill in as implemented)
| WP | status | outcome (measured) |
|----|--------|--------------------|
| AWP-1 | pending | — |
| AWP-2 | pending | — |
| AWP-3 | pending | (M3/M4 spec'd in the integrity plan) |
| AWP-4 | pending | — |
| AWP-5 | done | `vesp.uq.baselines` package now owns cheap heuristics, run assembly helpers, expanded baselines, and `GPResidualUQ` behind `BASELINE_REGISTRY`; repo-internal imports no longer use the former top-level baseline modules. `rtn_noise.py` now owns both numerical scaling and manifest-backed prototype artifacts. `drift_boundary`/`drift_horizon` document separate responsibilities, score-ablation M1/M2 variants reference production `SCORING_FUNCTIONS` through `PRODUCTION_SCORE_VARIANTS`, and internal trajectory scoring imports now go directly through `scoring`/`selection`. |
| AWP-6 | deferred | trigger: sources > ~5k or directional scoring default |
| AWP-7 | deferred | trigger: packaging/CI need |
