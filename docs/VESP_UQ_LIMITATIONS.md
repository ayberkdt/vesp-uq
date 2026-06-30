# VESP-UQ — Limitations and Scope Boundaries

This document records what VESP-UQ does **not** do, so the IAC framing stays defensible.

## Not a better deterministic surrogate

The posterior mean equals the ridge / Tikhonov point estimate exactly. VESP-UQ does not improve
deterministic acceleration or trajectory accuracy and must not be presented as doing so. Its
contribution is **calibrated, altitude-aware error bars** and the trajectory risk screen they
enable. The entropy / MaxEnt point-estimate experiments remain only as an ablation: maximizing
entropy over the sources does not beat well-regularized ridge on accuracy.

## Not a true lunar density reconstruction

The interior equivalent sources are a **mathematical** distribution chosen to reproduce the
*exterior* residual-force error field. They are not a physical internal mass-density model and
carry no geophysical interpretation.

## Orbit uncertainty propagation: exploratory only, not validated

VESP-UQ supplies a position-dependent force-error covariance `Sigma_a(x)`. An **exploratory** Monte
Carlo orbit-dispersion sampler (`vesp.uq.propagation.VESPMonteCarloPropagator`,
`scripts/run_propagation.py`) draws source-strength samples from the posterior and propagates a
batch of trajectories to show the orbit-level *spread* implied by the force-error posterior. This is
**not** a validated operational orbit-determination or state-covariance product:

- it samples the *local force-model* error posterior; it does not model measurement processing,
  realistic process noise, or dynamic mismodelling beyond the fitted residual;
- the headline force-risk finding is that the VESP-UQ score does **not** rank a surrogate's
  long-horizon *position* error on the in-distribution set, so the dispersion must not be read as a
  calibrated position-error or covariance-realism claim;
- a **linearized (STM) variant** (`vesp.uq.linear_propagation.LinearForceErrorCovariancePropagator`)
  propagates the same posterior into a deterministic `6x6` state covariance via the variational
  equation (`P = J Sigma_sigma J^T`); it agrees with the MC sampler in the linear regime but is
  equally exploratory and carries the same caveats. It uses the central (point-mass) gravity
  gradient by default (finite-difference Jacobian for a custom base field) -- *not* a validated
  orbit-determination linearization.

The sampled field is kept exactly consistent with the fitted posterior (honors `acceleration_sign`,
softening `eps`, and source quadrature weights), so per-point sample mean/covariance match
`predict_uncertainty` / `predict_covariance_3x3`.

## ST-LRPS adapter: exploratory wiring, not a validated integration

An ST-LRPS surrogate adapter exists (`vesp.adapters.st_lrps`, the Sobolev-Trained Lunar Residual
Potential Surrogate package) and `scripts/run_stlrps_propagation.py` uses its runtime force model as
the `base_accel_fn` of the exploratory MC sampler above. This is **exploratory wiring**, not a
validated integration claim: there is no validated end-to-end orbit-accuracy or covariance-realism
result, and a null force-risk vs position-error correlation is *expected* (position error is often
not force-model-error dominated). Do not claim a validated ST-LRPS integration on the basis of this
wiring.

### Adapter scope: only the force-model seam is in scope (the rest is vendored)

The full `vesp.adapters.st_lrps` package (~70 files) is **vendored, exploratory** ST-LRPS code and is
**out of VESP-UQ scope**: it is not maintained, refactored, or tested as part of the calibration
layer, and it depends on the external `lunaris` package, so it is **not importable in a clean
VESP-UQ environment** (CI included). The **only** VESP-UQ↔adapter seam is
`vesp.adapters.st_lrps.runtime.force_model.load_surrogate_force_model` (used by
`scripts/run_stlrps_propagation.py` as the MC base field). That seam — and only that seam — is
bounded by VESP-UQ tests (`tests/test_stlrps_adapter_boundary.py`: an import-safety guard plus a
skip-guarded artifact-load smoke, both skipping when the adapter / `lunaris` / artifact are absent).
Full adapter testing or refactoring is explicitly **not** a VESP-UQ deliverable. See
`src/vesp/adapters/README.md`.

### ST-LRPS diagnostic vs integration

`scripts/analyze_512_orbits.py` *reads* precomputed `ST_LRPS_DT60` position-error metrics to ask a
single **diagnostic** question — does the VESP-UQ force-risk score happen to co-rank that
surrogate's long-horizon position error? This is **not** the same as an ST-LRPS integration:

- it consumes a static metrics CSV, it does **not** call or wrap ST-LRPS;
- VESP-UQ is not inside the ST-LRPS propagator's RHS, and `Sigma_a(x)` is not propagated;
- a null correlation there is *expected* (position error is often not force-model-error dominated)
  and is reported as a diagnostic, never as a VESP-UQ failure or a position-error claim.

This read-only diagnostic is distinct from the *exploratory wiring* above
(`run_stlrps_propagation.py`, which does run ST-LRPS as the MC base field). A **validated**
integration — an adapter that feeds `a_corrected`/`Sigma_a(x)` into the ST-LRPS workflow with an
explicitly-tested orbit-accuracy / covariance-realism result — is still a separate deliverable that
does not exist yet.

### Local force-error covariance vs orbit/state covariance propagation

VESP-UQ implements the **local** predictive acceleration-error covariance `Sigma_a(x)` (the full
`3x3` per-point covariance). The **exploratory** MC and linearized-STM propagators above map that
local posterior into an orbit/state covariance, but only as a sampling / linearization diagnostic. A
**validated operational** state/orbit covariance — one that models measurement processing, realistic
process noise, and dynamic mismodelling beyond the fitted residual — is **not** implemented and must
not be claimed.

## Online correction: implemented as an exploratory force-model correction

`a_corrected(x) = a_surrogate(x) + mean_error(x)` is now available as an **exploratory** force-model
correction (`vesp.uq.correction.CorrectedForceField`; benchmark
`scripts/run_force_correction_benchmark.py`; doc `benchmarks/online_force_correction.md`). The risk
*screen* still evaluates VESP-UQ only at **output trajectory points** (post-processing), not inside
every integrator RHS call; the reports expose `n_output_points_total` and
`score_us_per_output_point` to keep that explicit. The correction is the posterior **mean** (a ridge
point estimate), so it is a **force-model** correction with no guaranteed long-horizon
position-accuracy claim, and the benchmark reports the accuracy delta **and** the per-RHS cost —
evaluating the full equivalent-source field inside every RHS call can erode the surrogate's speed
advantage (the smoke run shows a large accuracy gain at ~17× the per-RHS cost). The synthetic truth
lies in the equivalent-source span, so those numbers are a **best-case** illustration, not a
validated result.

## Sequential update: exact for the posterior, NOT an automatic recalibration

`VESPUQPlugin.update_error` conditions the fitted posterior on new error samples in closed form;
with the same Tikhonov weight and noise floor it **equals the batch refit on the concatenated
data exactly** (pinned by `tests/test_uq_sequential_update.py`). What an update deliberately does
**not** do:

- it never re-runs the L-curve -- `lambda` stays at the original fit's value (re-selecting the
  prior weight would break the exact-update contract);
- it does not touch the global noise floor or the altitude noise law unless a **fresh held-out**
  `val_positions`/`val_error` pair is supplied (then both are recalibrated exactly as in
  `fit_error`).

So after a large update without fresh validation data, the per-band coverage numbers from the
original fit are stale: **re-validate** (`evaluate_calibration` on new held-out samples) before
relying on calibration claims, and prefer passing fresh validation data to the update itself.

## Measured extension gates, not automatic production features

The pre-results readiness gate (`scripts/run_vespuq_system_readiness.py`) may run diagnostics and
prototypes that deliberately stop short of production integration:

- exact log-factor attribution explains the existing multiplicative supervisor and uses masking
  validation; it is **not** SHAP/LIME and does not improve rerun selection by itself;
- the RTN-style covariance prototype is a held-out before/after gate artifact. It is not applied
  by `VESPUQPlugin` unless a future measured run promotes it explicitly;
- Helmholtz / non-conservative force-error extensions remain closed unless the curl diagnostic
  justifies them on a real surrogate residual field;
- source-geometry auto-selection is a calibration sweep over equivalent-source placement, not a
  physical density inference.

## Serve-time screening has no oracle

`python -m vesp.uq.screen` (the serve driver) scores trajectories with a persisted model and
deliberately provides **no nearest-neighbour ground-truth oracle** -- serving has no calibration
samples, and inventing an oracle there would blur the train/serve boundary. Unless the trajectory
CSV itself carries surrogate/reference acceleration pairs (then the residual force error is used
as a *diagnostic*), serve outputs are force-risk / OOD scores only. A threshold packaged with the
model is applied **only** when the serve-time scoring mode matches the mode it was calibrated
for; thresholds never transfer across score scales.

## Exact covariance can be expensive

The exact `3x3` predictive covariance costs `O(m · n_sources^2)` for `m` query points. For large
source counts use `covariance_mode="diagonal"` (drops source correlations, `O(m · n_sources)`,
~2-3× faster in practice) or `covariance_mode="lowrank"` (top-k eigenpairs of the posterior
covariance). Diagonal mode is an approximation: it ignores off-diagonal source correlations and
therefore can mis-estimate the predictive variance; it is intended for speed, not for the headline
calibration numbers.

## Calibration caveats

- The heteroscedastic noise law `sigma^2(h) = a · h^(-b)` is a simple 2-parameter power-law misfit
  model fit on held-out validation residuals. In-distribution it calibrates per band; at extreme
  out-of-distribution altitude it must **extrapolate** the law.
- Operational conformal calibration (`uq.conformal.apply: true`) is a split-conformal wrapper over
  the predictive force-error uncertainty. Its finite-sample coverage statement assumes the
  calibration residuals and future query residuals are exchangeable; it is not a license to claim
  position-error or orbit-covariance realism. When per-band scales are enabled, samples inside a
  fitted altitude band use that band's scale; under-populated bands and queries outside all fitted
  bands fall back to the global scale.
- Vector (ellipsoid) calibration assumes an approximately Gaussian 3D error; heavy-tailed or
  strongly non-Gaussian residuals will violate the chi-square-3 expectation.
- The trajectory-screening ground-truth oracle is a nearest-neighbour read from real samples;
  with a sparse held-out set the per-point true error is noisy. Use `oracle_source: heldout`
  (default, no leakage) and report which oracle was used.
