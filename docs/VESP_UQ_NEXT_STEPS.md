# VESP-UQ — Next Steps Plan

A forward-looking, prioritized plan for the VESP-UQ layer, grounded in the current state of the
repository. This is a working roadmap, **not** a claims policy — the binding policy on what may be
claimed stays in [`SCIENTIFIC_CLAIMS.md`](SCIENTIFIC_CLAIMS.md) and the scope boundaries in
[`VESP_UQ_LIMITATIONS.md`](VESP_UQ_LIMITATIONS.md). Every item below must respect those constraints
(no position-error prediction claim, no validated operational orbit covariance, no invented units).

## Where we are

Recently completed (in the working tree, see commit grouping in N0):

- Conformal force-error calibration + sentinel false-negative audit (`vesp.uq.conformal`,
  `vesp.uq.audit`, `scripts/run_calibration_audit.py`).
- Physical acceleration-budget screening (`vesp.uq.physical_units`, `threshold_source:
  physical_budget`, `scripts/run_physical_budget_screening.py`) + optional conformal-corrected
  threshold.
- Propagation hardening: `vesp.uq.propagation` (MC) sign/eps consistency + scale-relative Cholesky
  jitter fix; new deterministic linearized `vesp.uq.linear_propagation` (STM) covariance.
- Unit-aware external trajectory loading (`vesp.uq.io.trajectory_loader`).
- Doc reconciliation (MC sampler + ST-LRPS adapter framed as *exploratory, not validated*) and a
  minimal CI workflow (`.github/workflows/ci.yml`).
- Test count: 335 → 364.

Open gaps found while surveying the code (evidence in parentheses):

- The newer scripts write bare output files and do **not** use the artifact/manifest +
  checksum system the main run uses (`grep` of `ensure_run_layout`/`write_run_manifest` over
  `scripts/run_calibration_audit.py`, `run_physical_budget_screening.py`,
  `run_force_error_benchmark.py`, `compare_risk_baselines.py` → none).
- No linter / formatter / type-check config anywhere (`pyproject.toml` has no `ruff`/`mypy`/`black`;
  no `.ruff.toml`/`.flake8`).
- `vesp.uq.linear_propagation` has a module + tests but **no driver script** and no benchmark doc,
  unlike the MC sampler (`scripts/run_propagation.py`).
- Several scripts are only exercised by the CI smoke step, with no pytest-level output assertions
  (`run_calibration_audit`, `run_force_error_benchmark`, `compare_risk_baselines`, `run_propagation`).
- The ST-LRPS adapter (`src/vesp/adapters/st_lrps`, 72 `.py` files) has **zero tests** and is
  exploratory wiring.

## Phases (prioritized)

### N0 — Commit the current Tier 1–5 work (prerequisite)

- **Why:** 18 modified + 4 new files are uncommitted; everything below should build on a clean base.
- **Action:** commit in logical groups —
  1. propagation hardening + CI + doc reconciliation,
  2. trajectory unit-awareness,
  3. doc-integrity (benchmarks README, units note, README),
  4. linearized covariance propagation,
  5. conformal-corrected physical budget.
- **Acceptance:** clean working tree; CI green on the pushed branch.
- **Effort:** XS.

### N1 — Reproducibility: route script outputs through the artifact/manifest system (HIGH)

- **Why:** reproducibility is a stated project value, but `run_calibration_audit`,
  `run_physical_budget_screening`, `run_force_error_benchmark`, `compare_risk_baselines` write bare
  JSON/MD/CSV with no run manifest, config snapshot, or SHA-256 checksums — unlike `vesp.uq.run`,
  which uses `ensure_run_layout` + `atomic_write_*` + `write_run_manifest`.
- **Action:** add a small shared helper (e.g. `vesp.uq.io.run_artifacts`) wrapping
  `vesp.common.artifacts` (`ensure_run_layout`, `atomic_write_json/text`, `compute_file_sha256`,
  `write_run_manifest`). Route every uq script through it; embed `config`, `seed`, and a UTC
  timestamp in each JSON; write a per-run manifest with output checksums.
- **Acceptance:** each script writes a manifest with config + checksums; a test asserts the manifest
  exists and checksums match; existing output filenames preserved.
- **Effort:** M. **Risk:** low (additive; no behavior change to the numbers).

### N2 — Code quality: lint + format check in CI (MEDIUM-HIGH)

- **Why:** stated "code quality" goal; no static analysis exists today (a stray placeholder import
  slipped in during recent work and was only caught by hand).
- **Action:** add a `ruff` config to `pyproject.toml` (lint + import hygiene + format check) scoped
  to `src/vesp/uq`, `scripts/`, `tests/`; add a CI lint job. Optionally add `mypy` on
  `src/vesp/uq` only (it is the most type-annotated area) as non-blocking to start.
- **Acceptance:** `ruff check` clean on the uq surface; CI runs it; no behavior change.
- **Effort:** S–M. **Risk:** low (may surface pre-existing lint to fix or ignore explicitly).

### N3 — Propagation consolidation: driver + benchmark doc (MEDIUM)

- **Why:** the linearized STM covariance (`vesp.uq.linear_propagation`) has no driver script or doc,
  and there is no documented MC-vs-STM comparison even though a test already shows they agree in the
  linear regime.
- **Action:**
  - add `scripts/run_linear_propagation.py` (parity with `run_propagation.py`): writes nominal
    states, `6x6` covariances, and position/velocity sigma, through the N1 artifact layer;
  - add `benchmarks/covariance_propagation.md`: MC vs STM agreement, cost trade-off (STM is
    deterministic / sampling-free; MC scales with sample count), and the **exploratory, not
    validated** framing + the force-risk⊥position-error caveat;
  - add the two propagation rows to `benchmarks/README.md`.
- **Acceptance:** script runs on the smoke config and writes artifacts; doc states the honest scope;
  CI smoke covers the new script.
- **Effort:** M. **Risk:** low.

### N4 — Script-level test coverage (MEDIUM)

- **Why:** `run_calibration_audit`, `run_force_error_benchmark`, `compare_risk_baselines`,
  `run_propagation` are only exercised by the CI smoke step (no pytest assertions on their output
  schemas), so their JSON/CSV contracts can drift silently.
- **Action:** add focused tests (tiny synthetic config, `tmp_path`) asserting each script's JSON keys
  / CSV header / row counts and a couple of invariants (e.g. flagged ⊆ trajectories). Reuse the
  `_tiny_screening_config` pattern already in `tests/test_uq_physical_budget_screening.py`.
- **Acceptance:** one test module per script (or a combined `tests/test_uq_scripts.py`); output
  schemas locked.
- **Effort:** M. **Risk:** low.

### N5 — ST-LRPS adapter boundary: bound it honestly (LOW, bounded)

- **Why:** `src/vesp/adapters/st_lrps` (72 files) is exploratory wiring with zero tests; its only
  VESP-UQ touchpoint is `scripts/run_stlrps_propagation.py` (uses the runtime force model as the MC
  base field).
- **Action (minimal):** add an import-safety test for the adapter package boundary the VESP-UQ side
  depends on (`vesp.adapters.st_lrps.runtime.force_model.load_surrogate_force_model` importable
  without heavy side effects), and a skip-guarded smoke test for `run_stlrps_propagation.py` (skip
  when no ST-LRPS artifact is present). Document explicitly that the rest of the adapter is
  out-of-scope vendored code.
- **Acceptance:** the VESP-UQ↔adapter seam has at least one test; the boundary is documented.
- **Effort:** S. **Risk:** low. (Full adapter testing is explicitly out of VESP-UQ scope.)

### N6 — Online force correction (Phase 5) — OPTIONAL, larger research item

- **Why:** the one remaining headline future-work item in the IAC plan: evaluate
  `a_corrected(x) = a_surrogate(x) + mean_error(x)` inside an integrator RHS, and benchmark the
  speed/accuracy trade-off (the docs warn that evaluating the full equivalent-source field every RHS
  call may erode the surrogate's speed advantage).
- **Action:** add an `a_corrected` RHS hook (reusing the same operator/sign/eps convention as the MC
  and STM propagators); a benchmark comparing surrogate vs surrogate+correction trajectories against
  a reference, reporting both accuracy delta **and** per-RHS cost. Frame honestly: the posterior mean
  is the ridge estimate, so this is a *force-model* correction, with no guaranteed long-horizon
  position-accuracy claim; report measured results only.
- **Acceptance:** benchmark runs on a synthetic reference; doc reports accuracy **and** cost with the
  honest caveats; tests cover the RHS hook's operator consistency.
- **Effort:** L. **Risk:** medium (scope + careful claims). Do only if explicitly desired.

## Recommended order

`N0 → N1 → N2 → N3 → N4`, with `N5`/`N6` optional. Rationale: commit first (N0); then the two
low-risk, high-value reproducibility/quality items (N1, N2) that harden everything already built;
then complete the propagation capability into a documented, tested deliverable (N3, N4). N5 bounds
an external subsystem honestly; N6 is the only item that adds new research scope and should be a
deliberate, separately-approved choice.

## Out of scope (and why)

- Full ST-LRPS adapter coverage/refactor — large vendored subsystem, outside the VESP-UQ
  calibration-layer focus (only its seam matters here; see N5).
- Anything that would claim position-error prediction, validated operational orbit covariance, or
  deterministic accuracy improvement — forbidden by `SCIENTIFIC_CLAIMS.md`.
- The Stage-3 MaxEnt / neural-density extensions — separate track from the force-risk/UQ layer.
