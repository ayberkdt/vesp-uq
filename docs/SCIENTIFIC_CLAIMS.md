# Scientific Claims Policy

This file defines what the current MaxEnt-VESP repository may and may not claim. It is
binding on the README, commit messages, reports, papers, and any generated summary.
The guiding principle: **be experiment-first and scientifically honest. The goal is to
make the MaxEnt-VESP idea falsifiable, not to make it look finished.**

## What the project currently is

A **deterministic feasibility framework** for equivalent-source gravity modeling with a
**Stage 3A** deterministic, entropy-regularized point estimate. It is **not** the full
probabilistic MaxEnt-VESP-Net vision.

Implemented now:

- deterministic discrete equivalent-source VESP (single- and multi-shell),
- Newtonian potential kernel and analytic acceleration kernel,
- unit-safe potential/acceleration scaling,
- ridge / Tikhonov least-squares solving with optional moment constraints,
- diagnostics for source collapse, shell cancellation, monopole/dipole leakage,
- deterministic entropy regularization over source strengths (Stage 3A),
- constrained ("equal-data-fit", Skilling-Bryan) MaxEnt with OOD evaluation (Stage 3A.2),
- **Stage 3C: exact linear-Gaussian posterior over sources + calibration diagnostics**
  (`extensions/probabilistic.py`, `training/uncertainty.py`). Posterior mean == ridge; the
  covariance gives predictive error bars, validated by coverage (PICP) on held-out data.

Established results (report as-is, do not soften):

- Deterministic *point-estimate* MaxEnt (entropy maximization over sources) does **not**
  improve accuracy and is **strictly dominated by ridge** on OOD generalization (entropy
  spreads source mass; the low-altitude regime needs the localization ridge provides).
- At **matched data error** (experiment E7 shootout) the better regularizer depends on the
  pathology: on the collapse/**norm** disease L2 wins 18–2 (cancellation is a source-norm
  problem entropy does not touch); on a **concentration** setup entropy wins 10–6, sweeping
  `top_5pct_source_contribution` and `effective_source_count` — entropy's genuine niche is
  de-concentration, a health axis L2 does not control directly (at a higher data-error cost).
- The linear-Gaussian posterior's epistemic uncertainty **grows in OOD extrapolation**
  (≈86× larger predictive std at low altitude than high altitude on the synthetic OOD
  case). With empirical-Bayes (evidence) hyperparameters it is well calibrated
  in-distribution (z_std ≈ 1.09, PICP90 ≈ 0.88).
- **Stage 3C+ heteroscedastic (altitude-dependent) noise** (`noise_model: heteroscedastic`,
  `floor + a·h^(-b)`, fit on held-out validation residuals) **calibrates every altitude band
  in-distribution** on real lunar — the low band improves from PICP90 0.53 / z_std 4.13 to
  ≈0.86 / 1.22 (mid 0.74→0.93, high ~1.0). On a pure altitude-OOD split the calibration must
  extrapolate the law into an unseen band and is fundamentally limited.
- `lambda_l2: auto` selects the Tikhonov weight at the L-curve corner (lands in the stable
  knee, λ≈1e-3 on the synthetic multi-shell case).
- **Source geometry is only a weak lever for the low-altitude bottleneck** (experiment E8).
  Surface-near dense shells (α→0.98) **do not help and actively destabilize** (low/high error
  ratio and relative RMSE blow up, e.g. real-lunar `surface_dense` rel_acc 0.69, low/high 3262);
  the only modest gain is *more sources at the same radii* (~13% lower low-altitude error on
  real lunar, at worse conditioning). The low-altitude bottleneck is largely a degree-band-limit
  (≤60) / conditioning ceiling, not a cheap geometry fix → it should be **quantified** by the
  posterior (heteroscedastic Stage 3C+), not assumed reducible by geometry.

Implemented in the `vesp.uq` force-risk layer:

- the **local predictive acceleration-error covariance `Sigma_a(x)`** (the full `3x3` per-point
  covariance, `VESPUQPlugin.predict_covariance_3x3`), plus expected-force-error and
  domain-support / OOD scoring and trajectory-level selective-rerun screening,
- a **simple post-hoc altitude-dependent heteroscedastic recalibration** of the predictive noise
  (`floor + a·h^(-b)`, 2 parameters) fit on held-out validation residuals.

Not implemented yet (do not claim):

- full nonlinear / variational Bayesian posterior or sampling-based inference (Stage 3C is
  the **exact conjugate Gaussian** posterior for the linear model, not MCMC/VI),
- a **learned / full / generative / nonlinear** heteroscedastic noise model (only the simple
  2-parameter post-hoc recalibration above is implemented — do not call it learned/generative),
- **validated orbit/state covariance propagation**: the *local* force-error covariance `Sigma_a(x)`
  is implemented, and an *exploratory* Monte Carlo orbit-dispersion sampler exists
  (`vesp.uq.propagation`), but a *validated* operational state/orbit covariance (realistic process
  noise, measurement processing, covariance-realism result) is not — do not claim it,
- neural source-density network,
- irregular-body source placement.

## Allowed claims

- The method **represents exterior residual gravity fields using interior equivalent
  sources**.
- The exterior field is **harmonic outside the source domain**.
- Acceleration is **computed analytically** from the Newtonian kernel (synthetic path)
  or via finite differences from the spherical-harmonic model when ingesting real data
  (documented per dataset).
- The current MaxEnt implementation is **deterministic entropy-regularized source
  selection** (a single point estimate), warm-started from and compared against the
  ridge baseline.
- The learned source distribution is **an equivalent mathematical source map**, not a
  true density map.
- On the real lunar set, results are a **proof-of-concept on the GRAIL-derived
  band-limited lunar residual field** (a.k.a. "spherical-harmonic-derived lunar residual
  target").
- The method provides a **linear-Gaussian posterior over the equivalent sources** whose
  **measured** calibration may be reported (with the homoscedastic caveat), and whose
  **epistemic uncertainty correctly grows under low-altitude extrapolation**.

## Disallowed claims

- Do **not** claim true internal density recovery.
- Do **not** claim a full *nonlinear/variational/sampling* Bayesian posterior — Stage 3C
  is the exact conjugate Gaussian posterior for the (linear) model only.
- Uncertainty: you **may** report the linear-Gaussian posterior's *measured* calibration
  numbers (evidence + heteroscedastic: in-distribution per-band PICP90 ≈ 0.86–1.0, z_std ≈ 1).
  The heteroscedastic noise is a simple 2-parameter power-law (+floor) **post-hoc recalibration**
  on held-out residuals — do **not** call it a learned/full/generative noise model. Do **not**
  claim calibration on **altitude-OOD extrapolation** (calibrating an altitude band with no
  calibration data is fundamentally limited; report it as such).
- Do **not** claim operational/validated orbit uncertainty propagation. An exploratory Monte Carlo
  orbit-dispersion sampler exists (`vesp.uq.propagation`), but it samples the local force-error
  posterior only; it is not a validated operational covariance product and force-risk does not rank
  long-horizon position error.
- Do **not** claim "neural VESP-Net" unless a neural source-density model is
  implemented.
- Do **not** claim the real lunar run is an operational lunar gravity model.

## Reporting rules

- The headline comparison metric is **`relative_acceleration_rmse`** (unit-invariant).
  Absolute RMSE values are unit/coordinate dependent — always report the convention
  (`acceleration_metric_units`).
- **`acceptability_status` is a screening flag, not a scientific verdict.** A `GOOD`
  status only means no automatic red flag fired; it must never be used to hide mediocre
  metrics. Always read the underlying numbers.
- Every MaxEnt claim must be made **against the ridge baseline at the same L2, data
  split, source geometry and target scaling** (`entropy_weight = 0` is that baseline).
- An entropy regime is only "useful" if it improves source-distribution health
  (`shell_cancellation_ratio`, `top_5pct_source_contribution`,
  `shell_energy_balance_entropy_nats`, source concentration) **without** an unacceptable
  `relative_acceleration_rmse` increase. If entropy only worsens RMSE without improving
  diagnostics, **report that honestly**.
- Configs that are intentionally under-regularized or otherwise unstable must include
  `legacy` or `stress_test` in the filename, and their numbers must not be cited as the
  method's performance.

## The experimental questions (falsifiable)

| ID | Question | Experiment |
| --- | --- | --- |
| Q1 | Can a fixed-source equivalent-source model reproduce a known synthetic exterior residual field? | E0 `synthetic_exact_recovery` |
| Q2 | How sensitive is the fit to source-shell radius mismatch? | E1 `synthetic_shell_radius_mismatch` |
| Q3 | Does multi-shell fitting improve accuracy or create shell cancellation? | E2 `synthetic_multishell_truth` |
| Q4 | How much L2 regularization is needed to suppress ill-conditioned source solutions? | E3 `synthetic_l2_sweep`, real `real_lunar_l2_sweep` |
| Q5 | Does entropy regularization improve source-distribution health at acceptable data-error cost? | E4 `synthetic_entropy_pareto`, real `real_lunar_entropy_pareto` |
| Q6 | On lunar band-limited residual data, is the method numerically stable across altitude bands? | E5 real lunar proof-of-concept |
