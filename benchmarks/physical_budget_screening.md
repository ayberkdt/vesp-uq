# Physical Acceleration-Budget Screening for VESP-UQ

This note documents unit-aware physical acceleration-budget screening:

- `src/vesp/uq/physical_units.py` — explicit model<->physical acceleration conversion utilities.
- `src/vesp/uq/thresholds.py` — `threshold_source: physical_budget` (converts a physical budget into
  model-score units before `select_reruns`).
- `scripts/run_physical_budget_screening.py` — driver that fits, calibrates, screens against a
  physical budget, and writes
  `outputs/physical_budget/{physical_budget_screening.json, .md, physical_budget_scores.csv}`.

```
python scripts/run_physical_budget_screening.py --config configs/vespuq/vespuq_smoke.yaml \
    --budget 1e-8 --units m/s^2 --scoring expected_abs_p95
```

Everything here concerns **force-model acceleration error** (`a_reference - a_surrogate`). It is not
a position-accuracy or orbit-covariance diagnostic.

## Why physical acceleration budgets are needed

VESP-UQ trajectory risk scores are produced on the model's normalized-acceleration scale. That is
adequate for *ranking* which trajectories to rerun first, but operational screening is often phrased
as a physical tolerance: "rerun any trajectory whose estimated force-model error exceeds
`1e-8 m/s^2`." To honour that, the physical budget is converted into the model's score units and
compared against an absolute-scale risk score. Both model-normalized and physical values are kept in
every report.

## Ranking scores vs absolute physical thresholds

- **Relative supervisor modes** (`supervisor_rel`, `supervisor_rel_p95`) normalize altitude *per
  trajectory*. They answer "which orbits are riskiest *within this ensemble*?" — a ranking. The same
  numeric value does **not** mean the same physical thing across trajectories.
- **Absolute modes** (`expected_abs`, `expected_abs_p95`, `supervisor_abs`, `supervisor_abs_p95`)
  are on a fixed expected-force-error scale, so one threshold means the same thing for every
  trajectory. Only these can be tied to a physical budget.

## Why relative scores cannot be used with physical budgets

Because a relative score is renormalized per trajectory, converting a single physical budget into one
relative-score threshold would compare incommensurable quantities — the threshold would mean
different physical errors on different orbits. The code therefore raises a `ValueError` if a physical
budget is paired with a relative scoring mode.

## How to configure `acceleration_scale_m_s2`

Physical conversion is **never inferred** from body radius or GM. It is available only when the
config declares it explicitly, in one of two ways:

```yaml
body:
  # (a) scores are model-normalized; give an explicit scale (m/s^2 per one model-score unit)
  acceleration_units: model_normalized_accel
  acceleration_scale_m_s2: 1.0e-6
  # (b) OR declare the scores' physical unit directly:
  # acceleration_units: km/s^2   # then scores are treated as km/s^2 (scale = 1e3 m/s^2 per unit)
```

```yaml
uq:
  physical_budget:
    enabled: true
    value: 1.0e-8        # required when enabled
    units: m/s^2         # m/s^2 | km/s^2 | mm/s^2 | um/s^2
    scoring: expected_abs_p95
    max_rerun_fraction: null   # optional cap on the flagged count
```

If neither (a) nor (b) is present, conversion is unavailable and reports clearly state:
*"physical acceleration conversion unavailable; values are reported in model-normalized acceleration
units."* Requesting a physical budget without an available scale raises a clear error rather than
silently using normalized units.

## How to run the script

- CLI: `--budget`, `--units`, `--scoring`, `--max-rerun-fraction` (CLI flags override config).
- Config-only: set `uq.physical_budget.enabled: true` with a `value`, then run with just `--config`.

The smoke config ships a *synthetic* example scale (`acceleration_scale_m_s2: 1.0e-6`) so the script
is runnable in CI; it is illustrative, not a physical claim about the synthetic data.

## Interpreting zero-alarm and nonzero-alarm results

- **Zero alarms** — no trajectory's estimated force-risk reached the budget. A genuinely safe,
  in-distribution regime under a generous budget can correctly raise zero alarms; this is expected,
  not a failure.
- **Nonzero alarms** — one or more trajectories meet or exceed the budget and are flagged for
  high-fidelity rerun. If `max_rerun_fraction` is set and more trajectories exceed the budget than
  the cap allows, only the highest-risk ones up to the cap are flagged (the report records
  `max_rerun_fraction_capped: true` and the number above the budget).

## What should not be claimed

- No invented physical units — conversion requires explicit metadata.
- No silent conversion of normalized values without that metadata.
- No use of relative scoring modes for physical budgets.
- No trajectory-position accuracy improvement.
- No operational orbit-covariance propagation.
- No guarantee of safety — exceeding/not exceeding a budget is an estimate from a fitted risk model.
- State only that physical-budget screening flags trajectories whose estimated force-risk exceeds a
  user-defined acceleration-error tolerance, reported in both model-normalized and physical units.
