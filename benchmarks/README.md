# VESP-UQ Benchmarks

VESP-UQ is a **post-processing force-risk / OOD uncertainty-calibration layer** for lunar gravity
residual surrogates. It scores the *expected force-model error* and *out-of-support (OOD) risk* of
a trajectory so the riskiest samples can be sent to a high-fidelity rerun. It is **not** a
deterministic trajectory-accuracy improver, **not** a position-error predictor, **not** a
density-recovery model, and **not** an operational orbit-covariance propagator.

It matters *what each benchmark tests* — a result can be strong on one and null on another:

| Benchmark | File | What it tests |
| --- | --- | --- |
| Force-risk / OOD detection | [`force_ood_detection.md`](force_ood_detection.md) | does force-risk flag low-altitude / OOD passes and rank **true force error**? |
| Absolute-threshold screening | [`absolute_threshold_screening.md`](absolute_threshold_screening.md) | can an absolute physical budget flag **zero** (false-alarm behavior)? |
| Baseline comparison | [`baseline_comparison.md`](baseline_comparison.md) | does the VESP-UQ score beat trivial heuristics (min-altitude, exposure) at ranking **true force error**? |
| Conformal calibration + sentinel audit | [`calibration_audit.md`](calibration_audit.md) | does post-hoc conformal scaling improve held-out **force-error** coverage, and what is the false-negative rate among accepted low-risk trajectories? |
| Physical acceleration-budget screening | [`physical_budget_screening.md`](physical_budget_screening.md) | flag trajectories whose force-risk exceeds a physical acceleration-error budget (e.g. `1e-8 m/s^2`), converting the budget into model units via an explicit scale. |
| Force-error covariance propagation (STM) | [`covariance_propagation.md`](covariance_propagation.md) | deterministic linearized (STM) `6x6` **force-error** state covariance along a nominal orbit (exploratory; **not** validated orbit determination). |
| Force-error covariance propagation (MC) | [`covariance_propagation.md`](covariance_propagation.md) | Monte Carlo orbit-dispersion sample covariance of the **same** force-error posterior; cross-checks the STM result and agrees in the small-perturbation regime. |
| Online force-model correction | [`online_force_correction.md`](online_force_correction.md) | does `a_corrected = a_surrogate + posterior-mean force error` cut integrated **position** error vs the bare surrogate, and at what per-RHS cost? (exploratory force-model correction) |
| Position-error diagnostic | [`position_error_diagnostic.md`](position_error_diagnostic.md) | does force-risk *co-rank* long-horizon ST-LRPS **position** error? (diagnostic only) |
| GPU / Float32 Verification | [`gpu_verification.md`](gpu_verification.md) | throughput speedups and float32 precision degradation for CUDA hardware paths; states the policy that headline claims remain CPU float64. |

Two scoring families are used:
- **relative** (`supervisor_rel*`): per-trajectory altitude normalization — for *ranking* which
  orbits to rerun first within one ensemble (not cross-trajectory comparable).
- **absolute** (`expected_abs*`, `supervisor_abs*`): fixed altitude reference — for a *physical
  budget* that means the same thing across trajectories (zero-alarm screening).

## Headline takeaways

- VESP-UQ **detects low-altitude / OOD** passes and **ranks true force-model error** along
  trajectories (force-risk / OOD detection — the core claim).
- VESP-UQ supports **zero-alarm absolute-threshold screening** with a physical budget (a fixed
  top-fraction screen cannot).
- VESP-UQ does **not** rank long-horizon **ST-LRPS position error** on the in-distribution
  512-orbit diagnostic — and this is *expected*, because that position error is not
  force-model-error dominated there. The project does **not** claim position-error prediction.

## Reproduce

```text
python scripts/run_iac_benchmarks.py --config configs/vespuq/vespuq_smoke.yaml   # full suite -> outputs/iac/
python scripts/run_force_error_benchmark.py --config configs/vespuq/vespuq_real_lunar.yaml
python scripts/run_calibration_audit.py --config configs/vespuq/vespuq_smoke.yaml          # conformal coverage + sentinel audit
python scripts/run_physical_budget_screening.py --config configs/vespuq/vespuq_smoke.yaml \
    --budget 1e-8 --units m/s^2 --scoring expected_abs_p95                                  # physical acceleration budget
python scripts/run_linear_propagation.py --config configs/vespuq/vespuq_smoke.yaml          # STM force-error covariance (deterministic)
python scripts/run_propagation.py --config configs/vespuq/vespuq_real_lunar.yaml            # MC orbit-dispersion sampler (cross-check)
python scripts/run_force_correction_benchmark.py --config configs/vespuq/vespuq_smoke.yaml  # online force-model correction (accuracy vs cost)
python scripts/benchmark_gpu.py --config configs/vespuq/vespuq_smoke.yaml                   # GPU and float32 screening throughput vs precision
python scripts/analyze_512_orbits.py                                             # ST-LRPS position-error diagnostic
```

Each VESP-UQ script writes a `run_manifest.json` alongside its outputs, recording the config
snapshot, seed, environment, and a SHA-256 checksum + byte size for every emitted file, and embeds a
`_provenance` block in each JSON — so a result can be traced back to the exact config and verified.
