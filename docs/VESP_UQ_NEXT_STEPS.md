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

### N1 — Reproducibility: route script outputs through the artifact/manifest system — **DONE**

- **Why:** reproducibility is a stated project value, but `run_calibration_audit`,
  `run_physical_budget_screening`, `run_force_error_benchmark`, `compare_risk_baselines` wrote bare
  JSON/MD/CSV with no run manifest, config snapshot, or SHA-256 checksums.
- **Done:** added `vesp.uq.io.run_artifacts.write_run_artifacts` (atomic writes + injected
  `_provenance` per JSON + `run_manifest.json` with config snapshot, seed, environment, and per-file
  SHA-256 + byte size). Routed all four scripts through it and added a `write_run_manifest` to the
  main `vesp.uq.run`. Output filenames preserved. Tests in `tests/test_uq_run_artifacts.py` assert
  the manifest exists and its checksums match the files on disk (369 tests pass).

### N2 — Code quality: lint + format check in CI — **DONE**

- **Why:** stated "code quality" goal; no static analysis existed (a stray placeholder import
  slipped in during recent work and was only caught by hand).
- **Done:** added a `ruff` config to `pyproject.toml` (`select = E, F, I, W, B, UP`,
  `line-length 120`, `target-version py310`; `E402` ignored in `scripts/`+`tests/` for the
  sys.path-before-import pattern; `E501` left to the formatter). The gate is **lint-only** by
  deliberate scope choice — a `[tool.ruff.format]` section is present for local use but the formatter
  is intentionally **not** CI-gated, so this stays behavior-preserving rather than a ~1.5k-line mass
  reformat. Added a dedicated `lint` job to `.github/workflows/ci.yml`
  (`ruff check src/vesp/uq scripts tests`) and pinned `ruff==0.15.16` in the `dev` extra. Fixed the
  surfaced issues (autofixed import order/whitespace/unused imports + manual unused-variable,
  `zip(..., strict=True)`, and loop-closure binding fixes); `ruff check` is clean on the uq surface
  and all three scoped dirs, and the suite still passes (372 tests). `mypy` was left out (optional in
  the plan; can be added non-blocking later).

### N3 — Propagation consolidation: driver + benchmark doc — **DONE**

- **Why:** the linearized STM covariance (`vesp.uq.linear_propagation`) had no driver script or doc,
  and there was no documented MC-vs-STM comparison even though a test already showed they agree in
  the linear regime.
- **Done:**
  - added `scripts/run_linear_propagation.py` (parity with `run_propagation.py`): fits from a config,
    propagates a low circular orbit, and writes nominal states, `6x6` covariances, and
    position/velocity sigma through the N1 artifact layer (`linear_propagation.{json,md}` +
    `linear_propagation_states.csv` + `run_manifest.json`); params come from an optional
    `uq.propagation` config block overridable by CLI flags;
  - added `benchmarks/covariance_propagation.md`: MC-vs-STM agreement (converges to **0.008%** at
    `N = 8000` in the drift regime) and the cost trade-off (STM deterministic / sampling-free, ~70x
    faster; MC scales with sample count and carries `O(1/sqrt(N))` noise), with the **exploratory,
    not validated** framing + the force-risk⊥position-error caveat;
  - added the two propagation rows + reproduce commands to `benchmarks/README.md`.
- **Also:** CI smoke now runs the new script, and a focused test
  (`tests/test_uq_linear_propagation_script.py`) locks the artifact + covariance contract (manifest
  checksums, `6x6` shape, `J(0) = 0`, CSV header/row count).
- **Acceptance:** met — script runs on the smoke config and writes artifacts; the doc states the
  honest scope; CI smoke covers the new script.

### N4 — Script-level test coverage — **DONE**

- **Why:** `run_calibration_audit`, `run_force_error_benchmark`, `compare_risk_baselines` were only
  exercised by the CI smoke step (no pytest assertions on their output schemas), so their JSON/CSV
  contracts could drift silently.
- **Done:** added `tests/test_uq_scripts.py` — one test per artifact-writing script asserting the
  JSON keys, CSV header + row count, and invariants (`flagged ⊆ trajectories`,
  `n_flagged ≤ n_trajectories`, `is_position_error_benchmark` is False). `run_propagation` writes no
  files (nothing to lock) and its MC core is already covered by `tests/test_uq_propagation.py`, so it
  gets only an import-safety guard; `run_linear_propagation` / `run_physical_budget_screening` are
  locked by their own modules, and the artifact/manifest contract by `tests/test_uq_run_artifacts.py`.
- **Acceptance:** met — output schemas locked; suite at 378 tests.

### N5 — ST-LRPS adapter boundary: bound it honestly — **DONE**

- **Why:** `src/vesp/adapters/st_lrps` (~70 files) is exploratory wiring with zero tests; its only
  VESP-UQ touchpoint is `scripts/run_stlrps_propagation.py` (uses the runtime force model as the MC
  base field).
- **Done:** added `tests/test_stlrps_adapter_boundary.py` — an import-safety guard for the seam
  (`load_surrogate_force_model`) and the script, plus a skip-guarded artifact-load smoke
  (`VESP_STLRPS_MODEL_DIR`) asserting the exact interface VESP-UQ depends on (`mu_si`, `degree_min`,
  `predict_residual_accel_fixed`). The adapter depends on the external `lunaris` package (not
  vendored here, not a declared dep), so it is not importable in a clean VESP-UQ environment; the
  tests `importorskip` and therefore **skip in CI** by design (they run where the adapter is
  installed). Documented the boundary in `src/vesp/adapters/README.md` and a new
  `docs/VESP_UQ_LIMITATIONS.md` subsection ("Adapter scope: only the force-model seam is in scope"),
  and reconciled the stale local-vs-orbit-covariance note there now that the N3 propagators exist.
- **Acceptance:** met — the VESP-UQ↔adapter seam has tests; the boundary is documented. (Full adapter
  testing is explicitly out of VESP-UQ scope.)

### N6 — Online force correction (Phase 5) — **DONE** (exploratory)

- **Why:** the one remaining headline future-work item in the IAC plan: evaluate
  `a_corrected(x) = a_surrogate(x) + mean_error(x)` inside an integrator RHS, and benchmark the
  speed/accuracy trade-off (the docs warn that evaluating the full equivalent-source field every RHS
  call may erode the surrogate's speed advantage).
- **Done:** added `vesp.uq.correction.CorrectedForceField` (the `a_corrected` RHS hook, reusing the
  plugin's operator/sign/eps convention so `correction(x)` equals `predict_uncertainty(x).mean_error`
  exactly) and `integrate_trajectory` (RK4 matching the MC/STM propagators). Added
  `scripts/run_force_correction_benchmark.py`: on a synthetic world (truth = equivalent-source field),
  it integrates surrogate / corrected / reference orbits and reports the position-error reduction
  **and** the per-RHS cost through the N1 artifact layer. On the smoke config the correction cut the
  final position error ~**79×** at ~**17×** the per-RHS cost. Doc
  `benchmarks/online_force_correction.md` + a README row frame it honestly (force-model correction,
  best-case in-span synthetic, no long-horizon position-accuracy claim, measured numbers only). CI
  smoke runs it; `tests/test_uq_correction.py` pins operator consistency, the integrator, and the
  accuracy-improves / cost-increases / schema contract. Reconciled the "future work" note in
  `docs/VESP_UQ_LIMITATIONS.md`.
- **Acceptance:** met — benchmark runs on a synthetic reference; doc reports accuracy **and** cost
  with honest caveats; tests cover the RHS hook's operator consistency.

### N7 — Performance + persistence hardening — **DONE**

- **Why:** a survey of the prediction/screening hot path found (a) `score_ensemble` looped
  per trajectory — for a 512-orbit screen that meant 512 separate operator builds, 512 small
  posterior matmuls and 512 separate k-NN `cdist` calls; (b) `evaluate_calibration` built the
  dense operator **twice** for the same held-out positions (once directly, once inside
  `predict_covariance_3x3`); (c) `build_dense_operator` materialized `(Q, S, 3)` temporaries and
  paid two concatenation copies per build; (d) no prediction path was chunked over queries, so a
  large position set materialized the full `(3N, n_sources)` operator at once; and (e) the fitted
  layer had **no persistence** — every script refit from scratch, blocking fit-once/reuse
  workflows (screening, `CorrectedForceField`, the MC/STM propagators).
- **Done:**
  - **Batched ensemble scoring**: `score_ensemble` concatenates the ensemble, runs ONE
    query-chunked `predict_uncertainty` + ONE batched domain-support pass, then splits the
    profile per trajectory. Per-trajectory numbers are identical to the sequential path
    (equivalence locked by `tests/test_uq_batched_scoring.py`).
  - **Query chunking**: new `uq.query_chunk_size` knob (default 8192 positions/block) chunks
    `predict_uncertainty` / `predict_covariance_3x3` / `evaluate_calibration`, bounding operator
    memory on large query sets.
  - **Single operator build in calibration**: `evaluate_calibration` now feeds the row-level
    prediction AND the `3x3` covariance from one operator per chunk.
  - **Lean operator builder**: `build_dense_operator` writes per-axis `(Q, C)` blocks straight
    into a preallocated output — no `(Q, S, 3)` temporaries, no `torch.cat` copies. The
    arithmetic order is unchanged, so outputs are **bitwise identical** (verified across
    chunked/unchunked, potential/acceleration, eps/sign variants); measured **~2.1×** faster on a
    screening-shaped build (8192 pts × 512 sources: 199 → 95 ms).
  - **Net effect** on the 512-orbit × 64-point screening profile (n_sources = 512, exact
    covariance, domain support on, CPU): **84.7 → 49.7 µs per output point (~1.7×)** with
    identical risk scores.
  - **Persistence**: `VESPUQPlugin.state_dict()/save()/load()/from_state_dict()` — atomic,
    version-tagged, `torch.load(weights_only=True)`-safe payload carrying the posterior, altitude
    noise law, domain-support geometry, options and `fit_info`. `output.save_model: true` makes
    the main run write `vespuq_plugin.pt` (checksummed into `run_manifest.json`);
    `run_vespuq(config, return_plugin=True)` exposes the fitted plugin to callers. Round-trip
    equality locked by `tests/test_uq_plugin_persistence.py` (predictions, covariances,
    domain-support scores, trajectory scores, and `CorrectedForceField` corrections all match the
    pre-save plugin exactly).
- **Acceptance:** met — full suite green (17 new tests: equivalence, chunking, persistence
  round-trip, `save_model` artifact), `ruff check` clean, smoke artifacts unchanged in shape; no
  behavior change anywhere (bitwise-identical operator, float-identical scores).

### N8 — Train/serve separation + model lifecycle — **DONE**

- **Why:** the system had exactly one operating mode — every invocation refit the layer from
  calibration data before screening. Industrial UQ deployments separate **training** (produce a
  versioned model artifact + decision policy + provenance) from **serving** (load the artifact,
  score new ensembles repeatedly, never refit). N7's persistence made this possible; N8 built the
  lifecycle on it.
- **Done:**
  - **Input provenance**: `write_run_manifest(..., inputs=...)` — manifests now checksum the
    files a run CONSUMED (dataset CSV, trajectory CSV, model artifact) with the same SHA-256 +
    byte-size treatment as outputs; `vesp.uq.run` and the serve driver both record them.
  - **Decision policy + model card packaged with the model**: `VESPUQPlugin.save(...,
    extra_metadata=...)` / `plugin.user_metadata` (JSON-safe, `weights_only`-load safe,
    round-trips). The training driver embeds the resolved scoring mode, threshold (+ source /
    quantile / physical value), fallback rerun fraction, time weighting, units, and dataset
    SHA-256; `--save-model` CLI flag added. A model card (`vespuq_plugin_card.md`, built by
    `vesp.uq.reporting.build_model_card`) is written next to the artifact: intended use,
    provenance, fit + held-out calibration table, decision policy, and the claims-policy scope
    boundaries — card and model cannot drift apart because both come from the same run report.
  - **Serve driver**: `python -m vesp.uq.screen --model vespuq_plugin.pt
    (--trajectories ens.csv | --config cfg.yaml) --out dir` — loads the persisted layer, scores
    the ensemble (batched, no refit), applies the packaged decision policy with explicit
    precedence (CLI > model > default fraction), **refuses to apply a packaged threshold to a
    mismatched score scale**, uses the CSV's own residual force error as the only serve-time
    diagnostic (no invented oracle), and writes `screening_report.{json,md}` + score CSVs + a
    manifest with model/input checksums. Serve scores are row-for-row identical to the training
    driver on the same ensemble (locked by `tests/test_uq_screen_cli.py`).
  - **Exact sequential update**: `VESPUQPlugin.update_error(positions, error,
    [val_positions, val_error])` — closed-form conjugate update; with the same `lambda` and noise
    floor it **equals the batch refit on the concatenated data exactly** (pinned to fp precision
    by `tests/test_uq_sequential_update.py`, including two-updates-equal-one-batch). Fresh
    held-out data recalibrates the noise floor + altitude law exactly as `fit_error`; domain
    geometry extends; `fit_info` records `n_updates`. The L-curve is deliberately NOT re-run
    (documented in `docs/VESP_UQ_LIMITATIONS.md` with a re-validation warning).
- **Acceptance:** met — 16 new tests (9 serve CLI + 7 sequential update) plus card/manifest
  assertions; CI smoke now exercises the full train→serve chain
  (`vesp.uq.run --save-model` → `vesp.uq.screen`); `ruff check` clean; full suite green.

### N9 — Mission Console desktop UI — **DONE**

- **Why:** every capability (train, serve, inspect, update, provenance) was CLI-only; an
  operator-facing console makes the lifecycle manageable without memorizing commands, while
  keeping the CLIs the single source of truth.
- **Done:** `python ui/app_vespuq.py` launches a PyQt6 app (`src/vesp/ui`, ~6 pages on a
  dark nav-rail shell): Dashboard (model/run KPIs + recent runs), Train (config + overrides →
  temp-config `python -m vesp.uq.run` subprocess with live log, calibration table + KPI result
  panel), Screen (model picker + CSV/generated source + policy overrides →
  `python -m vesp.uq.screen` subprocess, flagged-row table), Model (fit/policy/provenance grids,
  rendered model card, uncertainty-vs-altitude matplotlib profile), Update (worker-thread
  `update_error` with the LIMITATIONS warning in-page, before→after summary), Runs
  (manifest/provenance browser incl. input checksums). Heavy work runs in subprocesses
  (`ProcessJob`/QProcess, cancellable) or worker threads (`FnWorker`); `vesp/__init__` made
  lazy (PEP 562) so the UI shell — and any torch-free import — no longer pays the torch import
  at startup. Tests (`tests/test_vespuq_ui.py`) pin module import safety, the no-heavy-imports
  contract (clean-subprocess check), run-scan classification, and the thin-launcher shape;
  they skip when PyQt6 is absent (CI) and never instantiate `QApplication`.
- **Acceptance:** met — UI tests + full suite green, `ruff` clean (launcher added to the E402
  per-file ignores alongside scripts/tests); the GUI itself needs an interactive desktop, so
  windowed verification happens on the user's machine (`python ui/app_vespuq.py`).

## Recommended order

`N0 → ~~N1~~ → ~~N2~~ → ~~N3~~ → ~~N4~~ → ~~N5~~ → ~~N6~~ → ~~N7~~ → ~~N8~~ → ~~N9~~`. **All
planned items (N1–N9) are done.** Rationale: commit first (N0); then the low-risk, high-value
reproducibility/quality items (N1, N2) that harden everything already built; then the propagation
capability as a documented, tested deliverable (N3) with script-level schema tests (N4); N5
bounds the external ST-LRPS subsystem honestly. N6 — the one new-research item — was done last,
on explicit request, as an **exploratory** force-model correction reporting measured accuracy
**and** cost with honest caveats. N7 hardened the layer's hot path and added fit-once/reuse
persistence without changing any reported number. N8 turned the layer into a deployable model
lifecycle: train/serve separation, packaged decision policy + model card, input provenance, and
an exact sequential update. N9 put an operator-facing desktop console on top of the same entry
points without forking any behavior into the UI.

## Next wave (N10+): planned, independent work items

Each item below is **independently executable** (no item blocks another; soft synergies are
noted), respects the claims policy, and has its own acceptance gate. Recommended order:
N10 → N12 → N13 → N11 → N15 → N16 → N14 → N17, but any can be picked up alone.

### ~~N10 — Dynamics-aware risk: `stm_dispersion` scoring mode (exploratory diagnostic)~~ **(DONE)**

- **Why:** the headline open finding is that pointwise force-risk does NOT rank long-horizon
  *position* error (expected; documented). The repo already has the machinery to test the
  obvious next hypothesis: weight the force-error posterior by trajectory *dynamics* using the
  existing linearized STM propagator (`vesp.uq.linear_propagation`, `P = J Sigma_sigma J^T`) and
  score each trajectory by its predicted position-dispersion scalar (e.g. max/final
  `sqrt(trace(P_rr))`).
- **Deliverables:** a `stm_dispersion` trajectory score (separate entry point, NOT wired into
  the default `SCORING_FUNCTIONS` unless it earns it); benchmark script reusing the 512-orbit
  ST-LRPS diagnostic + `compare_risk_baselines` harness; benchmark doc with the measured
  Spearman/capture vs the existing supervisor/expected scores and trivial baselines.
- **Claims guardrail:** framed as an exploratory *diagnostic* derived from the force-error
  posterior — a positive result is reported as rank correlation only; a null result is reported
  like the previous nulls. Never a position-error prediction claim.
- **Acceptance:** benchmark runs from one command through the artifact layer; doc states the
  numbers either way; tests pin the score's shape/finiteness and its exact reuse of the fitted
  posterior (sign/eps/weights). **Effort:** M-L.

### ~~N11 — Second band-limited residual dataset (surrogate-agnosticism evidence)~~ **(DONE)**

- **Why:** every real-data claim currently rests on ONE dataset (GRAIL gl0420a, degree-2..60
  residual). A second residual band (e.g. a degree-30 truncation surrogate → 31..90 residual)
  exercises a different error spectrum and tests that the calibration story is not tuned to one
  band.
- **Deliverables:** dataset builder invocation + new CSV under `data/` (or a documented
  generation command if too large to commit), a `configs/vespuq/vespuq_real_lunar_L90.yaml`,
  one full train→screen run, and a short results doc comparing per-band calibration vs the L60
  set.
- **Acceptance:** pipeline runs end-to-end on the new band; calibration table reported
  honestly (better or worse); dataset contract test. **Effort:** M.

### ~~N12 — Model comparison + drift report (registry promotion gate)~~ **(DONE)**

> Implemented: `vesp.uq.compare.compare_models` + `scripts/compare_models.py`
> (`model_comparison.{json,md}` through the artifact layer; both model files checksummed into
> the manifest's `inputs`). Verification pass fixed two latent defects before sign-off: the
> calibration side-by-side crashed on the report's scalar summary keys with full band coverage,
> and the CLI passed the `TrajectoryDataset` wrapper instead of the trajectory list to the
> screening-agreement path — both fixed and locked by `tests/test_uq_compare.py` (identity
> comparison with full band coverage, CLI run incl. `--trajectories`, manifest input checksums).

- **Why:** with persistence + sequential updates, the operational question becomes "is model B
  (updated/retrained) safe to promote over model A?" — industrial registries answer this with a
  side-by-side report, not a feeling.
- **Deliverables:** `vesp.uq.compare` + `scripts/compare_models.py`: two saved plugins + a
  held-out CSV → per-band calibration side-by-side, screening agreement on a shared ensemble
  (flag overlap, risk Spearman), posterior distance summaries, domain-support/coverage shift
  (drift), written via the artifact layer (`model_comparison.{json,md}` + manifest with both
  model checksums).
- **Acceptance:** comparing a model against itself yields identity metrics (overlap 1.0,
  Spearman 1.0); comparing pre/post `update_error` shows the expected n_train/noise deltas;
  schema locked by tests. **Effort:** M. *(Soft synergy: a UI "Compare" tab later.)*

### ~~N13 — IAC evidence pack: one-command paper bundle~~ **(DONE)**

> Implemented: `scripts/build_iac_pack.py` (full mode reruns the benchmark suite;
> `--collect-only` assembles existing outputs) producing `EVIDENCE.md` with a 9-claim map
> (incl. the N10 null diagnostic and the N11 second-band evidence), a checksummed manifest with
> the collected files as `inputs`, and a zip honoring `--out-dir`. CI smoke now builds the pack
> on the smoke config and asserts the core evidence files exist.

- **Why:** the IAC deliverable ultimately needs figures + tables; today they are scattered
  across per-benchmark run dirs. Reproducible-paper practice: one command regenerates the whole
  evidence bundle from configs, with provenance.
- **Deliverables:** `scripts/build_iac_pack.py` — runs (or collects, `--collect-only`) the
  benchmark suite, then assembles `outputs/iac_pack/` with the claim-mapped tables/figures
  (calibration per band, screening capture/lift, zero-alarm demo, MC-vs-STM agreement,
  correction accuracy-vs-cost), an `EVIDENCE.md` index mapping each artifact to the claim it
  supports (and to `SCIENTIFIC_CLAIMS.md` limits), and a manifest checksumming everything.
- **Acceptance:** one command produces the pack on the smoke config in CI; each table/figure
  traceable to a run manifest. **Effort:** M.

### ~~N14 — GPU verification + float32 screening benchmark (skip-guarded)~~ **(DONE)**

> Implemented: `tests/test_uq_gpu_parity.py` (CUDA-marked, skip cleanly on CPU-only machines;
> float64 parity at 1e-12) + `scripts/benchmark_gpu.py` + `benchmarks/gpu_verification.md` with
> the policy that headline numbers stay float64/CPU. Verification pass replaced the float32
> test's vacuous `rtol=1.0` tolerance with a real contract (relative-L2 < 5e-2 on
> mean-error/sigma + >90% risk-rank agreement — float32 is a ranking/throughput proxy only).

- **Why:** `device`/`dtype` knobs exist but are unverified on CUDA; screening throughput is a
  reported KPI and an easy 10-50x may be available where a GPU exists.
- **Deliverables:** CUDA-marked tests (skip cleanly when unavailable) for fit/predict/score
  parity vs CPU (tolerance-documented); a benchmark script reporting CPU vs GPU and
  float64-vs-float32 screening deltas (max relative risk-score error); docs stating the policy:
  headline calibration numbers stay float64/CPU-reproducible.
- **Acceptance:** suite stays green on CPU-only machines (all GPU tests skip); benchmark doc
  reports measured numbers from at least one environment. **Effort:** S-M.

### ~~N15 — Property-based invariant tests (hypothesis)~~ **(DONE)**

> Implemented: `tests/test_uq_properties.py` on a fixed `ci` hypothesis profile — threshold and
> top-k flag-count invariants for `select_reruns`, finite-sample conformal level bound,
> conformal-scale nonnegativity, aggregator monotonicity (`mean/p95 <= max`), and linear
> sigma-scaling of the mean score. `hypothesis` added to the dev extra and the CI install.

- **Why:** the scoring/selection/threshold layer is the safety-critical surface; example-based
  tests pin known cases, property tests pin the *invariants* (e.g. flag-count bounds, threshold
  monotonicity, weight-normalization invariance, batched==sequential as a property).
- **Deliverables:** `hypothesis` in the dev extra; property tests for `score_sigma_profile`,
  `select_reruns`, `calibrate_risk_threshold`, conformal scale, and the batched-scoring
  equivalence; CI unchanged otherwise.
- **Acceptance:** properties run deterministically in CI (fixed profiles/seeds); any
  discovered edge case fixed or documented. **Effort:** S-M.

### ~~N16 — Release engineering: CHANGELOG + v0.2.0 + CI wheel + `--version`~~ **(DONE)**

> Implemented: `CHANGELOG.md` (0.1.0 baseline + the full 0.2.0 entry covering N7–N17),
> `pyproject.toml` bumped to 0.2.0, `--version` on both CLIs via
> `vesp.common.version.package_version()` (importlib.metadata with a `0.0.0+source` sentinel),
> and a CI `package` job: `python -m build` → install the wheel in a clean venv → run
> `--version` + the train→serve smoke from the wheel → upload `dist/` as a workflow artifact.

### ~~N17 — Mission Console: Propagation page~~ **(DONE)**

> Implemented: `src/vesp/ui/pages/propagate.py` — config picker, STM/MC mode toggle, duration
> knob; runs `run_linear_propagation.py` (with `--out-dir` into a timestamped UI run dir) or
> `run_propagation.py` as cancellable subprocesses with live logs; plots position/velocity
> sigma growth from `linear_propagation_states.csv` (MC mode is log-only by design — the
> sampler writes no files); the exploratory-not-validated caveat is shown in-page. Registered
> in the nav rail; import-safety + heavy-import contracts extended in
> `tests/test_vespuq_ui.py`.

### Backlog (deliberately not scheduled)

- **Angular heteroscedastic noise refinement** — opt-in low-order angular misfit term; only if
  N11 surfaces region-dependent miscalibration (research risk: marginal gains).
- **Matrix-free / iterative solver scaling** — only needed beyond ~5-10k sources; large lift.
- **Full ST-LRPS validated integration** — still out of scope (claims policy; external dep).

## Out of scope (and why)

- Full ST-LRPS adapter coverage/refactor — large vendored subsystem, outside the VESP-UQ
  calibration-layer focus (only its seam matters here; see N5).
- Anything that would claim position-error prediction, validated operational orbit covariance, or
  deterministic accuracy improvement — forbidden by `SCIENTIFIC_CLAIMS.md`.
- The Stage-3 MaxEnt / neural-density extensions — separate track from the force-risk/UQ layer.
