# Conformal Calibration and Sentinel Auditing for VESP-UQ

This note documents the post-hoc reliability layer added to VESP-UQ:

- `src/vesp/uq/conformal.py` — split-conformal calibration of the predictive **force-error**
  uncertainty.
- `src/vesp/uq/audit.py` — sentinel auditing of accepted low-risk trajectories for **force-error**
  false negatives.
- `scripts/run_calibration_audit.py` — driver that fits VESP-UQ, calibrates, screens, audits, and
  writes `outputs/audit/{calibration_audit.json, calibration_audit.md, sentinel_audit.csv}`.

```
python scripts/run_calibration_audit.py --config configs/vespuq/vespuq_smoke.yaml
```

Everything below concerns **force-model error** (`a_reference - a_surrogate`). None of it is a
position-accuracy, trajectory-accuracy, or orbit-covariance diagnostic.

## Why add post-hoc conformal calibration

VESP-UQ produces predictive uncertainty from a linear-Gaussian posterior over equivalent-source
strengths plus a calibrated heteroscedastic noise floor. That makes it *itself a fitted uncertainty
model*: its nominal intervals are only as correct as those modelling assumptions. On held-out
residual samples the predictive band can under-cover (overconfident) or over-cover (too wide),
especially out of the calibration support where the assumptions are weakest.

Split conformal calibration is a distribution-free wrapper that does not assume the predictive
distribution is correct. Given paired held-out samples it computes the normalized residual
`s_i = true_error_i / predicted_error_i` and takes the conservative `(1 - alpha)` quantile (with the
standard finite-sample correction `ceil((n+1)(1-alpha))/n`) as a single multiplicative scale `c`.
The calibrated band `c · predicted_error` then empirically covers the true force error at the
requested level on exchangeable data. A larger learned scale means a *more conservative* interval;
`c >= 1` exactly when the raw predictions were under-covering.

## Why VESP-UQ does not guarantee correctness by itself

The plugin's posterior gives well-defined intervals *under its model*, but coverage is an empirical
property of the real residuals, not a property the model can assert about itself. The conformal
layer measures that empirical coverage and corrects it; it is a wrapper, not a replacement for
`VESPUQPlugin`. We therefore report **coverage before and after** calibration rather than claiming
the intervals are correct a priori.

## How held-out residuals are used for empirical coverage

The script splits the calibration samples into train / held-out exactly as the main run does, fits
the plugin on the train split, and evaluates on the held-out split:

- the **true error** is the predictive residual `observed_error − posterior_mean_error`;
- the **predicted error** is the matching predictive uncertainty, depending on `mode`:
  - `norm` — total predictive `sigma`, compared against the residual magnitude `||e||`;
  - `component_max` — per-component std, compared against `max_j |e_j|`;
  - `mahalanobis` — the realized Mahalanobis distance of the residual under the predictive `3×3`
    covariance, against the nominal 3-DOF radius `√3`.

`coverage_before` is the fraction of held-out samples inside the raw band; `coverage_after` is the
fraction inside the conformally scaled band. We only state that conformal calibration *improves
empirical held-out force-error coverage* when the measured `coverage_after` actually moves toward the
`1 - alpha` target — it is reported, never assumed.

## Why sentinel audits help for accepted low-risk trajectories

When a fixed rerun budget recomputes only the high-risk trajectories at higher fidelity, the
accepted (low-risk) set is taken on trust. Because the risk score is a fitted model, some genuinely
high force-error trajectories can land in the accepted set. A **sentinel audit** draws a small,
deterministic random sample from the accepted set so those misses can be *measured* on a fraction of
the budget instead of assumed away. Selection is reproducible for a given seed and never overlaps the
flagged set.

## How to interpret false negatives

A trajectory is **high force error** if its true force error is at or above the
`high_error_quantile` of the ensemble. A **false negative** is a high-force-error trajectory that was
accepted (not flagged). The report gives:

- `false_negative_rate` — missed high-error trajectories ÷ all high-error trajectories (the full
  offline accounting, available because the true force error is known for every trajectory here);
- `sentinel_false_negative_rate` — the high-error rate within the audited sentinel sample, i.e. the
  estimate an operational audit on a held-back budget would observe.

A high false-negative rate means the risk screen is letting real force-error trajectories through; a
low rate, confirmed on the sentinel sample, supports trusting the accepted set.

## What should not be claimed

- No guaranteed surrogate reliability — conformal coverage is empirical and exchangeability-dependent.
- No deterministic trajectory-accuracy improvement — the posterior mean is unchanged.
- No position-error prediction — all quantities are force-model error.
- No operational orbit-covariance propagation.
- No universal superiority over GP / PCE / other conformal UQ methods.
- Only this: conformal calibration improves empirical held-out **force-error** coverage when the
  measured `coverage_after` supports it, and sentinel audits provide an empirical false-negative
  estimate for accepted low-risk trajectories.
