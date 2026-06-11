# Changelog

All notable changes to the VESP-UQ layer. Versioning follows semver on the `vesp` package
surface; the binding science scope lives in `docs/SCIENTIFIC_CLAIMS.md` and
`docs/VESP_UQ_LIMITATIONS.md` and is unaffected by version numbers.

## 0.2.0 — 2026-06-11

The N7–N17 wave: performance hardening, a deployable train/serve model lifecycle, an operator
console, and the first N10+ research/operations items.

### Performance & correctness (N7)

- Batched ensemble scoring: `score_ensemble` runs one query-chunked predict + one
  domain-support pass over the concatenated ensemble (~1.7x on the 512-orbit screening profile;
  per-trajectory numbers identical).
- `uq.query_chunk_size` (default 8192) bounds dense-operator memory on every prediction path;
  `evaluate_calibration` builds its operator once per chunk instead of twice.
- `build_dense_operator` rewritten to preallocated per-axis writes — bitwise-identical outputs,
  ~2.1x faster.

### Model lifecycle (N7/N8)

- Fitted-plugin persistence: `VESPUQPlugin.save/load/state_dict/from_state_dict` (versioned,
  atomic, `weights_only`-safe). `output.save_model: true` / `--save-model` writes
  `vespuq_plugin.pt` plus a model card (`vespuq_plugin_card.md`); the training run's decision
  policy (scoring, resolved threshold + provenance, units) is embedded in the artifact.
- Train/serve separation: new serve CLI `python -m vesp.uq.screen` scores external CSVs or
  generated ensembles with a persisted model — no refitting; packaged thresholds refuse to
  apply across score scales.
- Exact sequential update: `update_error` conditions the posterior in closed form and equals
  the batch refit on concatenated data (same lambda/noise); optional fresh-held-out
  recalibration of the noise floor + altitude law.
- Run manifests checksum consumed **inputs** (datasets, trajectory CSVs, model artifacts) next
  to produced outputs — both in `vesp.common.artifacts.write_run_manifest` and the script-side
  `vesp.uq.io.run_artifacts.write_run_artifacts`.

### Mission Console (N9/N17)

- PyQt6 desktop app: `python ui/app_vespuq.py` (`src/vesp/ui`) — Dashboard, Train, Screen,
  Model, Update, Runs, and Propagate pages over the documented CLIs (subprocess jobs, live
  logs; UI never forks pipeline behavior). `vesp/__init__` is lazy (PEP 562): importing the UI
  shell (or any torch-free subpackage) no longer pays the torch import.

### Research & evidence (N10–N13)

- N10 `stm_dispersion`: exploratory dynamics-aware trajectory score (max position-dispersion
  from the linearized STM propagator); 512-scenario ST-LRPS diagnostic reports an honest
  **null** (Spearman ≈ -0.05) — deliberately NOT added to the default scoring modes.
- N11 second residual band: degree-31..90 dataset (degree-30 truncation surrogate) + config +
  full run report with a band-vs-band calibration comparison (conservative, not sharp, on the
  second band).
- N12 model comparison: `vesp.uq.compare` + `scripts/compare_models.py` — posterior distance,
  per-band calibration side-by-side, screening agreement (flag IoU, risk Spearman),
  domain-support drift; both model artifacts checksummed into the manifest.
- N13 IAC evidence pack: `scripts/build_iac_pack.py` assembles a claim-mapped, checksummed
  evidence bundle (`EVIDENCE.md` + manifest + zip) from the benchmark outputs.

### Verification (N14–N15)

- GPU parity tests (skip cleanly without CUDA) + `scripts/benchmark_gpu.py`; policy: headline
  calibration numbers stay float64/CPU-reproducible, float32 is a ranking/throughput proxy.
- Property-based invariant tests (hypothesis) over selection/threshold/scoring/conformal.

### Release engineering (N16)

- `--version` on the `vesp.uq.run` / `vesp.uq.screen` CLIs; CI builds wheel+sdist and
  smoke-tests the wheel in a clean venv; this changelog.

## 0.1.0 — 2026-06-09

Initial packaged state: Stage 1–2 equivalent-source feasibility framework; VESP-UQ calibration
layer (linear-Gaussian posterior, heteroscedastic altitude noise, trajectory risk screening,
conformal audit, physical budgets, MC/STM propagation — exploratory); artifact/manifest
reproducibility layer; CI (lint + tests + smoke); N1–N6 hardening.
