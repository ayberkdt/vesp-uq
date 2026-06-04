# MaxEnt-VESP Stage 1-2 Feasibility Framework

## What Is This?

This repository is an early research framework for MaxEnt-VESP / VESP-Net. It
does not yet implement the full MaxEnt posterior framework. It implements the
Stage 1-2 feasibility framework for discrete equivalent-source gravity modeling.

The first question is deliberately narrow:

```text
Can fixed interior equivalent sources represent an exterior residual gravity
field accurately and stably enough to justify later MaxEnt / neural extensions?
```

## Current Scope: Stage 1-2 Only

Stage 1 uses fixed single-shell equivalent sources:

```text
Delta U(x) = sum_i w_i sigma_i / ||x - s_i||
Delta a(x) = sum_i w_i sigma_i (s_i - x) / ||x - s_i||^3
```

Stage 2 extends this to multiple interior shells:

```text
s_ji = alpha_j R_body u_i
Delta U(x) = sum_j sum_i w_ji sigma_ji / ||x - s_ji||
Delta a(x) = sum_j sum_i w_ji sigma_ji (s_ji - x) / ||x - s_ji||^3
```

## What Is Not Implemented Yet?

- Full Maximum Entropy posterior
- Probabilistic source distributions
- Neural source-density networks
- Angular SIREN source-density
- Uncertainty-aware orbit propagation
- Partition-function based inference

Extension placeholders are intentionally reserved for `entropy.py`,
`neural_density.py`, `probabilistic.py`, and `force_model.py`.

## Mathematical Formulation

The model is linear in `sigma`, so ridge/Tikhonov least squares is the default
feasibility solver. Neural/Adam optimization is available mainly for large
matrix-free cases where dense operators are impractical.

Regularized solve:

```text
min ||K sigma - d||^2
  + lambda_l2 ||sigma||^2
  + lambda_mono (sum_i w_i sigma_i)^2
  + lambda_dipole ||sum_i w_i sigma_i s_i||^2
```

Default acceleration sign convention:

```text
Delta a(x) = sum_i w_i sigma_i (s_i - x) / ||x - s_i||^3
```

Softening is optional:

```text
1 / sqrt(||x-s||^2 + eps^2)
```

Softening changes the physical Newtonian kernel and should only be used for
numerical stability tests.

## Units And Normalization

The canonical configuration is explicit about body scale:

```text
body.R_body
body.normalize_positions
body.position_units
body.potential_units
body.acceleration_units
```

Default synthetic experiments use normalized positions where the body radius is
`R_body = 1.0`. In that case `x`, source locations, altitude bins, and shell
radii are all dimensionless multiples of the body radius.

For real lunar datasets, the ingestion utility writes a sidecar metadata file
next to the CSV. The training loader uses this metadata to interpret position
and acceleration units instead of assuming every dataset is normalized.

Acceleration labels must be internally consistent:

```text
normalized gradient: d(DeltaU)/d(x/R_body)
physical acceleration: d(DeltaU)/d(distance) = normalized_gradient / R_body
```

This distinction is important because changing from normalized coordinates to
kilometers changes acceleration by a factor of `1 / R_body`.

## Source Geometry And Weights

The learned parameters are equivalent-source strengths `sigma`; they are not
interpreted as real density. Source positions are fixed before solving:

```text
single shell: s_i = alpha R_body u_i
multi shell:  s_ji = alpha_j R_body u_i
```

The default weight mode is `surface_area`, which assigns each shell a total
quadrature weight proportional to `4 pi (alpha R_body)^2`. Diagnostics report
effective source count, top-source concentration, monopole/dipole leakage, and
shell energy so that a low RMSE result can still be rejected if it is too
localized or physically brittle.

## Installation

```powershell
pip install -r requirements.txt
```

## Running Smoke Tests

```powershell
python scripts/smoke_test.py
pytest tests/
```

## Running Stage 1 Single-Shell Experiment

```powershell
python train_discrete.py --config configs/discrete_single_shell.yaml
```

## Running Stage 2 Multi-Shell Experiment

```powershell
python train_multishell.py --config configs/discrete_multishell.yaml
```

## Running Ablations

```powershell
python run_ablation.py --config configs/synthetic_stress_multishell.yaml
```

The ablation output includes:

```text
ablation_results.csv
ablation_summary.md
```

## End-of-Day Feasibility Suite

Run the compact decision suite:

```powershell
python run_feasibility.py --config configs/feasibility_suite.yaml
```

It tests:

- Same-family recovery
- Radius mismatch
- Multi-shell truth with single-shell model
- Multi-shell truth with multi-shell model
- Noisy observations
- High/low altitude OOD behavior

Outputs:

```text
outputs/feasibility/
    feasibility_results.csv
    maxent_readiness_report.md
```

## Expected Outputs

Each run writes:

```text
outputs/<run_name>/
    config.yaml
    sigma.pt
    metrics.json
    diagnostics.json
    altitude_binned_error.csv
    shell_energy.csv
    summary.txt
```

Top-level `.pt` checkpoints may also be produced for UI compatibility, but
generated outputs are ignored by git.

## Diagnostics Explanation

Core metrics:

- Potential RMSE
- Acceleration RMSE
- Relative acceleration RMSE
- Radial and cross-radial acceleration RMSE
- Vector angle error
- Altitude-binned acceleration error

Source diagnostics:

- Source norm
- Effective source count
- Top 1% / 5% source contribution
- Monopole leakage
- Dipole leakage
- Shell energy distribution
- Shell sigma norms

Effective source count is a localization diagnostic:

```text
p_i = |w_i sigma_i| / sum_j |w_j sigma_j|
N_eff = exp(-sum_i p_i log p_i)
```

It is not the full MaxEnt entropy objective.

## Synthetic Data

If no CSV path is supplied, the framework creates synthetic residual data from
hidden interior equivalent sources. This is useful for testing kernels, signs,
solvers, and recovery behavior. It is not observational validation.

## Real Residual Data Preparation

A GRAIL/PDS spherical harmonic ingestion utility is available:

```powershell
python -m experimental_vesp.real_gravity --model gl0420a --n-query 1024 --degree-min 2 --degree-max 60 --output data/lunar_grail_gl0420a_L60_residual.csv
```

General real-data pipeline:

1. Choose a high-degree reference gravity model.
2. Choose a low-degree baseline.
3. Compute residual potential and acceleration.
4. Save CSV as `x,y,z,DeltaU,Deltaax,Deltaay,Deltaaz`.
5. Fit with the VESP framework.

Positions are normalized by `R_body` in the default synthetic experiments. Real
CSV files should include the metadata sidecar generated by
`experimental_vesp.real_gravity`, especially when acceleration is exported in
physical units.

Real-data fitting templates:

```powershell
python -m experimental_vesp.train --config configs/real_lunar_gl0420a.yaml
python -m experimental_vesp.train --config configs/real_lunar_gl0420a_multishell.yaml
```

## Lunar Metadata Contract

Parts of the run/data architecture were adapted from the old
`LUNAR_SIMULATION` / ST-LRPS codebase without modifying that source folder.
The pieces carried over are intentionally small:

- a single-source lunar constants module
- strict lunar metadata validation
- strict SHADR/SHA coefficient-table parsing
- atomic artifact writes and run manifests

Real lunar CSV metadata now records `central_body`, GM, reference radius,
canonical scales, coefficient normalization state, and source gravity model
path. If a dataset claims to be lunar but its constants look Earth-scale or
otherwise inconsistent, loading fails early.

Each completed run writes:

```text
run_manifest.json
```

The manifest records config, compact metrics, generated artifacts, file sizes,
and SHA-256 checksums so later MaxEnt comparisons can be traced back to the
exact run outputs.

## Dense Operator Limit

For `N_query = 8192` and `N_source = 20000`:

```text
Potential operator:     8192 x 20000
Acceleration operator: 24576 x 20000
Joint operator:        32768 x 20000
```

Dense ridge is intended for small-to-medium experiments. Use chunked
matrix-free Adam or future iterative LSQR-style solvers for larger runs.

## Known Limitations

- Multi-shell does not automatically improve performance; it must be validated
  with ablations.
- Low-altitude stability can dominate the error budget.
- Learned source maps are equivalent mathematical sources, not real interior
  density.
- Current real spherical-harmonic acceleration uses finite differences for
  robustness; analytic gradients are a future speed improvement.

## Extension Roadmap

`entropy.py`:

- signed positive-negative entropy
- relative entropy / KL prior
- shell-wise entropy
- effective source entropy

`neural_density.py`:

- Angular MLP
- Angular SIREN
- SH encoding + MLP

`probabilistic.py`:

- posterior over sigma
- variational source distribution
- acceleration covariance

`force_model.py`:

- `predict_residual_accel(x)`
- `predict_residual_accel_with_uncertainty(x)`

Proceed to Stage 3 only if discrete and multi-shell VESP show enough
representational power on hard synthetic or real residual fields.

Recommended Stage 3 starting point:

1. Keep ridge/Tikhonov as the baseline.
2. Add deterministic entropy regularization over `sigma`.
3. Compare data-error vs entropy Pareto curves.
4. Only then move to neural density or probabilistic posterior experiments.
