# VESP-UQ — IAC Experiment Plan

**An equivalent-source uncertainty calibration layer for lunar gravity residual surrogates.**

## Purpose

VESP-UQ is **not** a replacement for a fast residual-gravity surrogate and **not** a claim that
maximum entropy improves deterministic accuracy. It is an uncertainty and risk-calibration layer
that fits a surrogate's remaining force-model error

```text
e_a(x) = a_reference(x) - a_surrogate(x)
```

with a physics-consistent interior equivalent-source posterior, and then provides per-position
predictive uncertainty, trajectory-level risk scores, and selective high-fidelity rerun flags.
It is surrogate-agnostic: it only needs acceleration samples, not the surrogate's architecture.

## Pipeline

1. **Calibration samples** — `positions`, `surrogate_acceleration`, `reference_acceleration`
   (or the error directly). The band-limited residual dataset is already an error field: the
   degree-2..60 GRAIL residual *is* the error of a degree-60 truncation surrogate, so it is fit
   with `surrogate = 0`. See `vesp.uq.data`.
2. **Fit** (`VESPUQPlugin.fit` / `fit_error`) — equivalent-source ridge with automatic
   regularization (L-curve corner, `vesp.core.regularization.lcurve_lambda`), exact
   linear-Gaussian posterior over source strengths (`LinearGaussianPosterior`), and
   altitude-dependent heteroscedastic noise (`AltitudeNoiseModel`) fit on **held-out** residuals.
3. **Held-out calibration** (`evaluate_calibration`) — per-altitude-band PICP / z_std / NLL /
   CRPS, plus 3D ellipsoid (Mahalanobis / chi-square-3) coverage from the full `3x3` predictive
   covariance, and the low/high uncertainty ratio.
4. **Trajectory screening** (`score_trajectory`, `score_ensemble`, `run_risk_screening`) —
   aggregate predictive uncertainty along each trajectory; flag the riskiest subset.
5. **IAC outputs** (`vesp.uq.run`) — JSON + Markdown reports and CSV tables.

## What is implemented

- Surrogate-agnostic data interface (direct-error and reference/surrogate CSV modes; synthetic).
- Exact conjugate linear-Gaussian posterior; posterior mean == ridge point estimate.
- Automatic L-curve regularization; evidence (empirical-Bayes) and fixed modes.
- Altitude-dependent heteroscedastic calibration on held-out residuals.
- Component-wise calibration (PICP, z_std, NLL, CRPS) and **vector ellipsoid calibration**
  (Mahalanobis chi-square-3 coverage, mean/median d²) from a full `3x3` predictive covariance.
- Covariance speed modes: `exact`, `diagonal`, `lowrank` (top-k eigenpairs of the posterior
  covariance). Indicative cost for 4000 query points × 1280 sources: exact ≈ 600 ms,
  diagonal ≈ 230 ms (~2.6× faster), lowrank ≈ 260 ms.
- Trajectory scoring in three families: legacy **sigma** (`max`, `mean`, `low_alt_integral`,
  `time_above`, `combined`); **expected-force-error** (`expected_abs`(+`_p95`), `expected_low_alt`);
  and **supervisor** point risk = expected error x altitude weight x (1 + domain risk), in a
  *relative* form (`supervisor_rel`(+`_p95`), per-trajectory altitude normalization — for ranking)
  and an *absolute* form (`supervisor_abs`(+`_p95`), fixed altitude reference — for physical
  budgets / zero-alarm thresholds). `expected`/`supervisor`(+`_p95`) are backward-compatible aliases.
- Domain-support / OOD scoring (k-NN distance + radial + optional angular components).
- Selective rerun with three policies (top-fraction top-k, absolute threshold, threshold+cap) and
  capture-rate / precision / Spearman validation against a held-out ground-truth oracle.
- **External trajectory ingestion** (`vesp.uq.io.load_trajectory_csv`): score surrogate-generated
  ensembles from CSV (positions-only, or with surrogate/reference acceleration pairs for direct
  residual-force-error fitting/scoring).
- CSV artifacts: `calibration_by_band.csv`, `trajectory_scores.csv`, `flagged_trajectories.csv`,
  `fit_summary.json`; JSON + Markdown reports with an IAC claim summary.
- **Fitted-layer persistence** (`VESPUQPlugin.save` / `.load`, `output.save_model: true` or
  `--save-model` → `vespuq_plugin.pt` in the run manifest): fit once, reload for screening /
  correction / propagation without refitting; round-trip predictions are identical. The artifact
  packages the training run's **decision policy** (scoring, threshold + provenance, units) and
  ships with a **model card** (`vespuq_plugin_card.md`).
- **Train/serve separation**: `python -m vesp.uq.run` (training driver) vs
  `python -m vesp.uq.screen` (serve driver: load model, score external CSV or generated
  ensembles, apply/override the packaged policy, no refit). Manifests checksum consumed
  **inputs** (model, dataset/trajectory CSVs) as well as outputs.
- **Exact sequential update** (`VESPUQPlugin.update_error`): closed-form conjugate update equal
  to the batch refit on concatenated data (same `lambda`/noise); optional fresh-held-out
  recalibration of the noise floor + altitude law. The L-curve is not re-run (see
  `docs/VESP_UQ_LIMITATIONS.md`).
- **Batched + query-chunked prediction**: `score_ensemble` scores the whole ensemble in batched
  dense passes and all prediction paths chunk over queries (`uq.query_chunk_size`), bounding
  memory with per-trajectory numbers identical to sequential scoring.
- **Model comparison / promotion gate** (`vesp.uq.compare`, `scripts/compare_models.py`):
  posterior distance, per-band calibration side-by-side, screening agreement (flag IoU + risk
  Spearman), domain-support drift; both model artifacts checksummed into the manifest.
- **IAC evidence pack** (`scripts/build_iac_pack.py`): one command assembles the claim-mapped,
  checksummed evidence bundle (`EVIDENCE.md` maps each artifact to its claim and to the
  `SCIENTIFIC_CLAIMS.md` limits).
- **Second residual band (N11)**: the degree-31..90 set (a degree-30 truncation surrogate's
  error) calibrates without retuning -- conservatively, not sharply; see
  `benchmarks/vespuq_real_lunar_L90_report.md` for the band-vs-band table.
- **STM-dispersion diagnostic (N10, honest null)**: dynamics-weighting the force-error
  posterior via the linearized STM does NOT co-rank long-horizon ST-LRPS position error
  (Spearman ~ -0.05); kept out of the default scoring modes by design.

**Force-risk / OOD vs position-error diagnostic.** The core deliverable is *force-model risk /
OOD detection* — does the score flag low-altitude/OOD passes and rank the surrogate's true
*force* error (`scripts/run_force_error_benchmark.py`)? Whether the force-risk score happens to
co-rank a surrogate's long-horizon *position* error (`scripts/analyze_512_orbits.py`) is a
separate **diagnostic**, not a VESP-UQ claim; a null result there is expected when position error
is not force-model-error dominated.

## Exploratory (code exists, NOT a validated claim)

- **Monte Carlo orbit-dispersion sampling** (`vesp.uq.propagation`, `scripts/run_propagation.py`):
  draws source-strength samples from the posterior and propagates a batch to show the orbit-level
  spread implied by the force-error posterior. It samples the *local* force-model error only and is
  **not** a validated operational orbit-determination / state-covariance product (and force-risk does
  not rank long-horizon position error). See `docs/VESP_UQ_LIMITATIONS.md`.
- **Linearized (STM) covariance propagation** (`vesp.uq.linear_propagation`): the deterministic
  variational counterpart of the MC sampler, mapping the source posterior into a `6x6` state
  covariance (`P = J Sigma_sigma J^T`). Agrees with the MC sampler in the linear regime; same
  exploratory caveats (central-field gravity gradient by default; not validated orbit determination).
- **ST-LRPS adapter wiring** (`vesp.adapters.st_lrps`, `scripts/run_stlrps_propagation.py`): runs the
  ST-LRPS runtime force model as the MC sampler's base field. Exploratory wiring, not a validated
  integration result.

## What is not implemented (future work)

- A **validated** ST-LRPS (or other named surrogate) integration with an explicitly-tested
  orbit-accuracy / covariance-realism result.
- Validated operational orbit/state covariance realism (the MC and STM propagators above are
  exploratory only).

(The Phase-5 online correction `a_corrected = a_surrogate + mean_error` is now implemented as an
**exploratory** force-model correction — `vesp.uq.correction`, benchmarked with measured accuracy
**and** per-RHS cost in `benchmarks/online_force_correction.md`; it carries no long-horizon
position-accuracy claim. See `docs/VESP_UQ_LIMITATIONS.md`.)

## Safe claims

- VESP-UQ fits residual-force error from reference/surrogate acceleration pairs.
- It provides a physics-consistent equivalent-source posterior over the force-error field.
- It calibrates altitude-dependent predictive uncertainty (reduces low-altitude overconfidence).
- It scores surrogate-generated trajectories for risk and supports selective high-fidelity rerun.
- It is surrogate-agnostic at the acceleration interface level.

## Unsafe claims (do not make)

- That VESP-UQ improves deterministic trajectory/point-estimate accuracy (the mean is just ridge).
- That it replaces or is integrated with ST-LRPS or any other residual surrogate.
- That it performs operational orbit uncertainty propagation.
- That the equivalent sources represent true lunar internal density.
- That MaxEnt is the main successful point-estimator.

## Minimal IAC experiments

- **Experiment 1 — standalone residual-error calibration.** Goal: demonstrate the layer reduces
  low-altitude overconfidence. Report L-curve regularization, posterior uncertainty,
  heteroscedastic calibration, altitude-binned PICP90 / z_std, ellipsoid coverage, low/high ratio.
- **Experiment 2 — geometry & regularization ablation** (existing `geometry_shootout` /
  `regularizer_shootout`). Goal: show source placement and entropy do not cheaply solve the
  low-altitude bottleneck, so uncertainty calibration is necessary.
- **Experiment 3 — trajectory risk screening.** Goal: show selective high-fidelity reruns without
  destroying surrogate speed. Report risk score per trajectory, flagged fraction, whether flagged
  trajectories correspond to larger reference error, and post-processing runtime overhead.

Run:

```text
python -m vesp.uq.run --config configs/vespuq/vespuq_smoke.yaml       # tiny synthetic, seconds
python -m vesp.uq.run --config configs/vespuq/vespuq_real_lunar.yaml  # GRAIL gl0420a residual
```

The binding policy on claims is [`docs/SCIENTIFIC_CLAIMS.md`](SCIENTIFIC_CLAIMS.md); see also
[`docs/VESP_UQ_LIMITATIONS.md`](VESP_UQ_LIMITATIONS.md).
