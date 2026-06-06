# MaxEnt-VESP Stage 1-2 Feasibility Framework

## What Is This?

This repository is an early research framework for MaxEnt-VESP / VESP-Net. It
does not yet implement the full MaxEnt posterior framework. It implements the
Stage 1-2 feasibility framework for discrete equivalent-source gravity modeling.
This repository still does not implement full MaxEnt-VESP.

The first question is deliberately narrow:

```text
Can fixed interior equivalent sources represent an exterior residual gravity
field accurately and stably enough to justify later MaxEnt / neural extensions?
```

The next scientific decision is whether deterministic single-shell and
multi-shell equivalent-source VESP represent hard synthetic and real SH
residual fields accurately enough to justify Stage 3 entropy regularization.

## Project Structure

The code is packaged as `vesp` under a `src/` layout. Modules are grouped by
responsibility so each layer can evolve independently:

```text
src/vesp/
    common/      cross-cutting infrastructure
                   artifacts.py      atomic writes, run manifests, checksums
                   config.py         YAML config load / merge / validate
                   units.py          position/acceleration unit handling
                   lunar.py          lunar constants + metadata contract
    core/        math and model core
                   kernels.py        Newtonian kernel + dense operator
                   operators.py      potential/acceleration/joint operators
                   sources.py        source geometry (fibonacci, shells)
                   solvers.py        ridge / Tikhonov least squares
                   losses.py         moment losses, shell energy, composite
                   models.py         DiscreteVESP, MultiShellDiscreteVESP
                   metrics.py        RMSE / angle / relative error metrics
                   diagnostics.py    source localization diagnostics
    data/        datasets and IO
                   dataset.py        ResidualGravity data + CSV loading
                   synthetic.py      synthetic residual generation
                   splits.py         train/val/test splits
                   gravity_io.py     SHADR/SHA coefficient parsing
                   real_gravity.py   GRAIL/PDS spherical-harmonic ingestion
                   target_scaling.py target normalization scales
    training/    pipelines and CLI entry points
                   train.py          unified config-driven entry point
                   train_discrete.py single-run solve/train pipeline
                   train_multishell.py / run_ablation.py / feasibility.py
                   evaluate.py       evaluation + artifact writing
    analysis/    analysis.py, advanced_analysis.py (reports / plots / PDF)
    extensions/  Stage-3 scaffolds: entropy, neural_density,
                 probabilistic, force_model (not the full MaxEnt framework)
    app/         ui.py (PyQt6 workbench)

configs/         experiment YAML configs (single source of truth)
scripts/         dataset builders and orchestration helpers
tests/           pytest suite
data/            input gravity models and prepared residual CSVs
outputs/         generated run artifacts (git-ignored)
pyproject.toml   packaging (src layout) + pytest config
```

Intra-package imports use absolute `vesp.*` paths (e.g.
`from vesp.core.models import DiscreteVESP`). The thin root scripts
`train_discrete.py`, `train_multishell.py`, `run_ablation.py`, and
`run_feasibility.py` are convenience wrappers that delegate to the matching
`vesp.training.*` module.

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

## What Is Implemented Now?

- Stage 1-2: deterministic single- and multi-shell equivalent-source ridge/Tikhonov.
- **Stage 3A: deterministic discrete MaxEnt regularization** over the solved source
  strengths (entropy-regularized point estimate, warm-started from the ridge
  baseline) plus a data-error vs entropy Pareto sweep. See
  [Stage 3A](#stage-3a-discrete-maxent-regularization).

## What Is Not Implemented Yet?

- Full Maximum Entropy **posterior** (Stage 3A is a deterministic point estimate, not a
  calibrated distribution over sources)
- Probabilistic source distributions
- Neural source-density networks (Stage 3B)
- Angular SIREN source-density
- Uncertainty-aware orbit propagation
- Partition-function based inference

The remaining extension placeholders are reserved under `src/vesp/extensions/`
(`neural_density.py`, `probabilistic.py`, and `force_model.py`). `entropy.py` is now an
active Stage 3A component, not a scaffold.

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
For physical lunar CSVs, metadata sidecars are required. The loader uses
metadata to distinguish normalized positions, physical positions, normalized
gradients, and physical accelerations, and **converts the acceleration target
into the model coordinate system** (`prepare_data_for_model`). Without this
conversion the joint potential+acceleration solve would be internally
inconsistent by a factor of `R_body` and would silently abandon the potential
fit while only matching acceleration.

The config field `body.R_body` is the model radius scale. In normalized real
data configs it is usually `1.0`. The dataset metadata `R_body` is the physical
reference radius, usually in kilometers. The loader records both
`original_position_units` and `model_position_units` after conversion.

Acceleration labels must be internally consistent:

```text
normalized gradient: d(DeltaU)/d(x/R_body)
physical acceleration: d(DeltaU)/d(distance) = normalized_gradient / R_body
```

This distinction is important because changing from normalized coordinates to
kilometers changes acceleration by a factor of `1 / R_body`. The model always
predicts `dU/d(model coordinate)`, so the loader rescales the CSV acceleration
into the model convention: a physical `km/s^2` target is multiplied by `R_body`
(km) when the model works in normalized coordinates, and a normalized gradient is
divided by `R_body` when the model works in physical km. As a result, for
normalized-coordinate runs the reported acceleration metrics are in
**model normalized-gradient units** (`km^2/s^2` per normalized radius), not
`km/s^2`. Relative acceleration RMSE is unit-invariant and is the headline metric;
`acceleration_metric_units` in `metrics.json` / `summary.txt` records the
convention for each run.

## Target Scaling / Loss Normalization

Real residual potential and acceleration targets can differ by orders of
magnitude. For real-data configs, enable:

```yaml
loss:
  normalize_targets: true
  potential_scale: auto
  acceleration_scale: auto
```

Auto scales are computed from the train split only:

```text
potential_scale = sqrt(mean(DeltaU^2))
acceleration_scale = sqrt(mean(Deltaa_x^2 + Deltaa_y^2 + Deltaa_z^2))
```

The same scales are then used for validation/test row weighting. Ridge rows use
`sqrt(lambda_potential) / potential_scale` for potential and
`sqrt(lambda_acceleration) / acceleration_scale` for acceleration.

Target normalization only affects the solve/training objective. Reported RMSE
metrics are computed in the original raw target units after model prediction.
Every run writes `target_scales.json`; when normalization is disabled, both
scales are `1.0` and their sources are recorded as `disabled`.

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

The package uses a `src/` layout and is installed in editable mode so that the
`vesp` package and the `python -m vesp.*` entry points resolve from anywhere:

```powershell
pip install -r requirements.txt
pip install -e .
```

## Running Smoke Tests

```powershell
python scripts/smoke_test.py
pytest tests/
```

Before reporting numerical results, run the full deterministic checklist:

```powershell
python scripts/pre_results_check.py
```

## Running Stage 1 Single-Shell Experiment

```powershell
python -m vesp.training.train --config configs/discrete_single_shell.yaml
```

## Running Stage 2 Multi-Shell Experiment

```powershell
python -m vesp.training.train --config configs/discrete_multishell.yaml
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

## Interpreting Real Lunar Results

The first real lunar runs (`real_lunar_gl0420a*`) showed:

- high-altitude extrapolation is comparatively stable,
- low-altitude error dominates the error budget,
- unconstrained multi-shell fits can reduce RMSE but may collapse shell energy onto
  the innermost shell and inflate the source norm (`sigma_l2`).

Every run now self-diagnoses these failure modes. `summary.txt` / `metrics.json` /
`diagnostics.json` report low/mid/high altitude band RMSE and the
`low_to_high_error_ratio`, the shell-energy collapse metrics
(`dominant_shell_energy_fraction`, `shell_energy_entropy`, `shell_collapse_flag`),
the sigma-norm warning, and a single `acceptability_status`
(`GOOD | CONDITIONAL | REJECT_REGULARIZATION | REJECT_LOW_ALTITUDE |
REJECT_SOURCE_COLLAPSE | REJECT_NUMERICAL`). The status is a fast triage flag, **not**
a scientific decision. The screening rests on scale-invariant signals:
`REJECT_NUMERICAL` triggers on the dimensionless relative monopole/dipole leakage,
`REJECT_SOURCE_COLLAPSE` on shell-energy fractions, `REJECT_LOW_ALTITUDE` on the
low/high error ratio, and `CONDITIONAL` on relative acceleration RMSE / source
concentration. `REJECT_REGULARIZATION` is only a coarse `sigma_l2` gross-blow-up
gate (an absolute, coordinate-dependent magnitude), since healthy ridge fits already
reach `sigma_l2` of order 10.

Before considering Stage 3 MaxEnt, run the deterministic ablations (each supports a
fast `quick` mode and an exhaustive `--mode full`):

```powershell
python -m vesp.training.run_ablation --config configs/ablation_real_lunar_regularization.yaml
python -m vesp.training.run_ablation --config configs/ablation_real_lunar_shells.yaml
python -m vesp.training.run_ablation --config configs/ablation_real_lunar_lowalt_weighting.yaml
```

Optional low-altitude weighted training (boosts low-altitude rows in the solve only;
reported metrics stay in raw units):

```powershell
python -m vesp.training.train --config configs/real_lunar_gl0420a_lowalt_weighted.yaml
python -m vesp.training.train --config configs/real_lunar_gl0420a_multishell_lowalt_weighted.yaml
```

If the best **non-collapsed** deterministic run still has unacceptable low-altitude
error or source concentration, proceed to **Stage 3A: Discrete MaxEnt regularization**.
Do not jump directly to neural density. The default
`real_lunar_gl0420a_multishell.yaml` now ships conservative shells/regularization; the
prior settings are preserved as `real_lunar_gl0420a_multishell_legacy.yaml`.

## Stage 3A: Discrete MaxEnt Regularization

Stage 3A is the conservative first step of the MaxEnt roadmap. It keeps the Stage 1-2
ridge solution as a warm start and baseline, then refines the source strengths by
adding a maximum-entropy term to the same target-normalized, row-weighted data
objective:

```text
minimize   mean((A sigma - b)^2)
         + lambda_l2 * mean(sigma^2)
         + lambda_moment * (monopole^2 + lambda_dipole * dipole^2)
         - entropy_weight * H(sigma)
```

It is **not** the full MaxEnt posterior: it produces a single deterministic
entropy-regularized point estimate, not a calibrated distribution over sources.

Enable it with `solver.type: maxent` and `loss.entropy_weight` / `loss.entropy_mode`.
The convex objective is solved with L-BFGS (strong-Wolfe line search) from the ridge
warm start, which is robust to the ill-conditioned equivalent-source operator (Adam is
available via `maxent.optimizer: adam` for very large matrix-free problems).

Entropy modes (`loss.entropy_mode`):

- `positive_negative` (default) — signed MaxEnt over the source distribution; spreads
  source mass and reduces concentration (`top_5pct_source_contribution`).
- `abs` — entropy of the absolute source distribution.
- `relative_uniform` — KL divergence to a uniform prior.
- `shell_balance` — entropy of the per-shell energy distribution; directly resists
  shell-energy collapse (the dominant deterministic failure on the real lunar set).

Single MaxEnt run:

```powershell
python -m vesp.training.train --config configs/maxent_real_lunar_gl0420a_multishell.yaml
```

Data-error vs entropy Pareto sweep (the principled way to choose `entropy_weight`):

```powershell
python -m vesp.training.maxent_pareto --config configs/maxent_pareto_real_lunar.yaml
```

Outputs:

```text
outputs/maxent_pareto_real_lunar/
    pareto_curve.csv
    maxent_pareto_report.md
```

Every run (ridge or MaxEnt) now also reports entropy diagnostics
(`source_entropy_nats`, `max_possible_source_entropy_nats`,
`shell_energy_balance_entropy_nats`, `relative_entropy_to_uniform`, `entropy_weight`,
`entropy_mode`) so the ridge baseline (`entropy_weight=0`) and MaxEnt runs are directly
comparable along the Pareto curve.

MaxEnt entropy over `sigma` is a complementary regularizer, but on this dataset it is
**not** the primary fix for the apparent multi-shell "collapse" — see
[Diagnosing Multi-Shell Collapse](#diagnosing-multi-shell-collapse). The dominant
pathology there is a brittle inter-shell near-cancellation caused by under-regularization,
which proper Tikhonov (`lambda_l2`) removes at essentially zero data cost. Use MaxEnt as
a secondary tool once the L2 regularization is set correctly.

## Diagnosing Multi-Shell Collapse

The first real lunar multi-shell runs were flagged `REJECT_SOURCE_COLLAPSE` by the
shell-energy-fraction metric (one shell holding ~99% of `sum w sigma^2`). Investigation
showed this metric is **misleading**: it is radius-biased (a deep shell needs a large
`sigma` for the same field, so it accrues `sigma^2` energy) and blind to the real
failure.

The real failure is a **brittle inter-shell near-cancellation**. Adjacent shells (e.g.
0.70 and 0.86) are nearly linearly dependent for an exterior field, so an
under-regularized least-squares solve (`lambda_l2 = 1e-5`) fits the field with huge
opposing source strengths that nearly cancel: each shell's own field is ~70x the target
residual, and removing either shell sends the relative acceleration RMSE from 0.19 to
~70 (leave-one-shell-out). This is pure ill-conditioning (`sigma_l2 ~ 9.3`), and it lives
in the operator's near-null space (the SVD shows ~800 of 1280 singular values below
`1e-3 * sigma_max`).

Because the cancellation is in the near-null space, proper Tikhonov regularization
removes it for free. Sweeping `lambda_l2`:

| `lambda_l2` | rel. acc. RMSE | `shell_cancellation_ratio` | `sigma_l2` | status |
| ---: | ---: | ---: | ---: | --- |
| `1e-5` (old default) | 0.192 | 143 | 9.28 | REJECT_SOURCE_COLLAPSE |
| `1.0` | 0.219 | 7.3 | 0.19 | REJECT_SOURCE_COLLAPSE |
| `30` (new default) | 0.195 | 4.5 | 0.085 | REJECT_LOW_ALTITUDE |
| `1000` | 0.240 | 2.5 | 0.034 | REJECT_LOW_ALTITUDE |

So **the collapse was an implementation problem (under-regularization + a misleading
metric), not an architectural limit**. Most of the cancellation (143 -> ~4) is removed
at zero data cost; the small residual (~4 -> ~1.5) does cost data, reflecting the genuine
mild redundancy of adjacent shells. Two changes encode this:

- The default `lambda_l2` for multi-shell real configs is now `30` (was `1e-5`), and the
  regularization ablation grid spans `1e-4 ... 1e3` (the old `1e-7 ... 1e-4` grid could
  never find the fix).
- Every multi-shell evaluation now reports `shell_cancellation_ratio`
  (`sum_j RMS(field_j) / RMS(field_total)`) and `per_shell_field_rms`. The acceptability
  screen uses the cancellation ratio (a field-based, scale-invariant brittleness signal)
  as the primary `REJECT_SOURCE_COLLAPSE` trigger; the radius-biased energy fraction is
  only a fallback for runs that predate the metric.

## End-of-Day Feasibility Suite

Run the compact decision suite:

```powershell
python -m vesp.training.feasibility --config configs/feasibility_suite.yaml
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
    target_scales.json
    summary.txt
    run_manifest.json
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
- Shell cancellation ratio (multi-shell): `sum_j RMS(field_j) / RMS(field_total)`, plus
  `per_shell_field_rms`. ~1 for a healthy fit, >> 1 when adjacent shells cancel (brittle).

Source diagnostics:

- Source norm
- Effective source count
- Top 1% / 5% source contribution
- Relative monopole leakage (dimensionless: `|monopole| / total absolute source mass`)
- Relative dipole leakage (dimensionless: `||dipole|| / (total absolute source mass * mean source radius)`)
- Absolute monopole / dipole leakage (reported for continuity; scale-dependent)
- Shell energy distribution
- Shell sigma norms

The relative leakage metrics are unit- and scale-invariant, so the acceptability
screening thresholds them instead of the absolute values, which depend on both the
source magnitude and the coordinate convention.

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
python -m vesp.data.real_gravity --model gl0420a --n-query 1024 --degree-min 2 --degree-max 60 --output data/lunar_grail_gl0420a_L60_residual.csv
```

General real-data pipeline:

1. Choose a high-degree reference gravity model.
2. Choose a low-degree baseline.
3. Compute residual potential and acceleration.
4. Save CSV as `x,y,z,DeltaU,Deltaax,Deltaay,Deltaaz`.
5. Fit with the VESP framework.

Positions are normalized by `R_body` in the default synthetic experiments. Real
CSV files must include the metadata sidecar generated by
`vesp.data.real_gravity`, especially when acceleration is exported in
physical units. The loader rejects CSV input without explicit position-unit
metadata unless a caller intentionally opts out for a legacy test.

The ingestion utility also writes a compact diagnostics JSON next to the CSV,
including query count, radius range, position norms, potential RMS,
acceleration RMS, finite-difference step, reference radius, and GM.

Real-data fitting templates:

```powershell
python -m vesp.training.train --config configs/real_lunar_gl0420a.yaml
python -m vesp.training.train --config configs/real_lunar_gl0420a_ood.yaml
python -m vesp.training.train --config configs/real_lunar_gl0420a_multishell.yaml
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

All Stage 3 scaffolds live under `src/vesp/extensions/`.

`extensions/entropy.py` (**implemented, Stage 3A** — wired into `solver.type: maxent`
via `vesp.training.maxent`):

- signed positive-negative entropy
- relative entropy / KL prior
- shell-wise entropy and per-shell energy-balance entropy
- effective source entropy

`extensions/neural_density.py`:

- Angular MLP
- Angular SIREN
- SH encoding + MLP

`extensions/probabilistic.py`:

- posterior over sigma
- variational source distribution
- acceleration covariance

`extensions/force_model.py`:

- `predict_residual_accel(x)`
- `predict_residual_accel_with_uncertainty(x)`

Do not start Stage 3 MaxEnt until deterministic Stage 1-2 checks pass on hard
synthetic and small real residual datasets.

The conservative Stage 3 progression (steps 1-3 are now implemented as Stage 3A):

1. Keep ridge/Tikhonov as the baseline. *(done — warm start + `entropy_weight=0`)*
2. Add deterministic entropy regularization over `sigma`. *(done — `solver.type: maxent`)*
3. Compare data-error vs entropy Pareto curves. *(done — `vesp.training.maxent_pareto`)*
4. Only then move to neural density or probabilistic posterior experiments. *(future:
   Stage 3B `neural_density.py`, Stage 3C `probabilistic.py`)*
