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

## Not operational orbit uncertainty propagation (yet)

VESP-UQ supplies a position-dependent force-error covariance `Sigma_a(x)`. It does **not** itself
propagate state covariance through an integrator. Consuming `Sigma_a(x)` as a force-model
process-noise input for STM / covariance propagation, or running full Monte Carlo orbit
uncertainty propagation, is future work and is not claimed unless that experiment is actually run.

## Not integrated with ST-LRPS (or any named surrogate)

VESP-UQ is surrogate-agnostic at the acceleration interface. No ST-LRPS adapter or integration
experiment exists in this repository. Do not claim integration unless such an adapter and
experiment are added and run.

## Online correction is future work

Adding `a_corrected(x) = a_surrogate(x) + mean_error(x)` inside an integrator's RHS is deferred
(Phase 5). The current risk screen evaluates VESP-UQ only at **output trajectory points**
(post-processing), not inside every integrator RHS call. The reports expose
`n_output_points_total` and `score_us_per_output_point` to keep this explicit. Online correction
must be benchmarked carefully because evaluating the full equivalent-source field inside every
RHS call may erode the surrogate's speed advantage.

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
- Vector (ellipsoid) calibration assumes an approximately Gaussian 3D error; heavy-tailed or
  strongly non-Gaussian residuals will violate the chi-square-3 expectation.
- The trajectory-screening ground-truth oracle is a nearest-neighbour read from real samples;
  with a sparse held-out set the per-point true error is noisy. Use `oracle_source: heldout`
  (default, no leakage) and report which oracle was used.
