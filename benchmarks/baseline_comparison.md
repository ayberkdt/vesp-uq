# Baseline Comparison for Trajectory Force-Risk Screening

This benchmark compares VESP-UQ trajectory risk scores against simple baseline selectors on a
single, fixed target: **trajectory-level true force-model error**. It answers a narrow, technical
question — *does the VESP-UQ score concentrate true force error better than trivial heuristics?* —
and nothing about long-horizon trajectory **position** error.

## What it tests

Each selector produces one scalar score per trajectory (higher = higher risk). The top
`rerun_fraction` are flagged and compared against the truly-high-force-error trajectories:

| Selector | Idea |
| --- | --- |
| `random` | chance-level reference (capture ≈ rerun_fraction, lift ≈ 1) |
| `min_altitude` | lowest periapsis ranks highest |
| `low_altitude_exposure` | fraction of points below `low_altitude_radius` |
| `domain_support` | mean per-point out-of-support (OOD) score (only if domain support is enabled) |
| `uncertainty_only` | mean predictive sigma (no bias / altitude / OOD weighting) |
| `altitude_residual_expected_ratio` | p95 of `expected_error / f_alt(radius)`, where `f_alt` is an altitude-only expected-error curve fit on plugin calibration geometry |
| `altitude_residual_expected_delta` | p95 of `expected_error - f_alt(radius)` |
| `calibrated_supervisor` | optional validation-calibrated supervisor (`calibrated_supervisor_p95`) when `uq.risk.calibrated_supervisor.enabled: true` |
| `supervisor` | full VESP-UQ supervisor (`supervisor_rel_p95`: expected error × altitude × domain) |

Reported per selector: Spearman vs true force error, capture rate, precision, lift over random,
mean true force error of flagged vs accepted trajectories, and their ratio.

If `uq.screening.time_weighting: kepler_r2` is enabled, the comparison applies the same
approximately time-proportional weights (`dt ~ r^2`) used by the main VESP-UQ trajectory scoring
to low-altitude exposure, VESP-UQ score reductions, altitude-residual p95/mean reductions, domain
support means, and the trajectory-level true force-error aggregation. This keeps the baseline
ranking target aligned with the main screening run when generated orbits are sampled uniformly in
true anomaly.

## Why minimum altitude is a strong simple baseline

Force-model error from a band-limited / truncated gravity surrogate typically grows toward low
altitude. A selector that simply ranks by lowest periapsis therefore captures much of the true
force-error signal with no model at all. `min_altitude` and `low_altitude_exposure` are the bars
VESP-UQ must clear: a supervisor score only earns its place if it ranks true force error **better**
than these heuristics, e.g. by additionally using predictive bias and out-of-support (OOD) risk in
directions/regimes where altitude alone is uninformative.

## Why true force-model error is the target

VESP-UQ scores expected *force-model* error and OOD risk. Evaluating it against force error is the
matched, honest test of the layer. Position error is a downstream, integrator-dependent quantity
that is often not force-error dominated; using it as the target would test a different (and not
claimed) capability. See [`position_error_diagnostic.md`](position_error_diagnostic.md) for that
separate diagnostic.

## How to run

```text
python scripts/compare_risk_baselines.py --config configs/vespuq/vespuq_smoke.yaml
python scripts/compare_risk_baselines.py --config configs/vespuq/vespuq_real_lunar.yaml --rerun-fraction 0.10
```

Writes `outputs/baselines/baseline_comparison.{json,csv,md}` plus
`altitude_incremental_value.csv`, `altitude_incremental_sweep.csv`, and
`baseline_comparison_paper.csv`. With an external trajectory CSV
(`uq.screening.trajectory_source: csv`) carrying surrogate/reference acceleration pairs, the true
force error is read directly from the residual; otherwise it uses the held-out nearest-neighbour
force-error oracle (no leakage).

The `altitude_incremental_value` block in the JSON/Markdown is the P0 diagnostic: for every score,
it reports the delta versus `min_altitude` in Spearman/lift, partial correlation after removing
minimum radius, within-altitude-bin Spearman, bootstrap confidence intervals, and a 5/10/20%
rerun-fraction sweep. The altitude-residual scores above are the P1 diagnostic: they ask whether
VESP-UQ expected error still ranks force error after the altitude-only expected-error trend is
removed.

When calibrated supervisor is enabled, the plugin also writes/uses a learned point-risk formula:
`expected_error^w * altitude_residual^w * (1 + w_epistemic * epistemic_fraction) *
(1 + w_domain * domain_risk)`. The small held-out grid includes `w_domain = 0`, so the domain
term is automatically disabled when it does not improve validation Spearman.

## How to interpret

- **VESP-UQ beats the heuristics** (higher Spearman / lift than `min_altitude` and
  `low_altitude_exposure`): the supervisor adds force-risk ranking value beyond altitude — the
  result to cite for the layer.
- **VESP-UQ does not beat them**: report it plainly. On in-distribution sets where force error is
  almost entirely altitude-driven, a min-altitude heuristic can match or exceed the supervisor;
  that is an honest negative for the *added* value of the score on that set, not a failure of the
  underlying calibration. Small ensembles also make capture rate / precision noisy — read Spearman
  and lift together, and prefer larger `n_orbits` for stable numbers.
- This is a **force-risk ranking** comparison only. It does not measure, and must not be read as,
  prediction of trajectory position error.
