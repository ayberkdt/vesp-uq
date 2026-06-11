# VESP-UQ GPU/Float32 Screening Benchmark

- **Config:** `configs/vespuq/vespuq_smoke.yaml`
- **Trajectory points:** 512 orbits x 64 pts = 32768 queries
- **Sources:** 112

## Throughput (score_ensemble execution time)

| Environment | Total Time (s) | Throughput (µs/point) | Speedup vs CPU-float64 |
| --- | --- | --- | --- |
| cpu_float64 | 0.7470 | 22.80 | 1.00x |
| cpu_float32 | 0.6924 | 21.13 | 1.08x |
| cuda_float64 | 2.4624 | 75.15 | 0.30x |
| cuda_float32 | 2.6047 | 79.49 | 0.29x |

## Precision (Max Relative Error vs CPU-float64)

Max relative risk-score error (`|val - baseline| / |baseline|`) across the ensemble:

| Environment | Max Rel Error |
| --- | --- |
| cpu_float32 | 8.79e-07 |
| cuda_float64 | 2.13e-15 |
| cuda_float32 | 8.91e-07 |

## Policy Statement

**The headline calibration and scientific numbers must remain float64/CPU-reproducible.**
The measured boundaries of the fast paths:

- **float32 is a SCORING proxy only, with a float64-fitted posterior cast to float32** (the
  path this benchmark measures, and what `VESPUQPlugin.from_state_dict` with
  `options.dtype = "float32"` produces). On this problem the max relative risk-score error is
  ~9e-7 -- safe for ranking / bulk screening, not for headline numbers. The proxy contract
  (relative deviation < 5e-2 and >90% rank agreement) is pinned by
  `tests/test_uq_gpu_parity.py::test_gpu_float32_scoring_proxy_contract` on CUDA machines.
- **Fitting directly in float32 is NOT supported.** The Gram solve at the selected Tikhonov
  weights (lambda down to ~1e-8) exceeds float32 conditioning; the parity test measured O(1)
  (~240%) deviations from the float64 fit. The test suite only asserts that a float32 fit runs
  and stays finite -- it deliberately carries no parity claim.
- **CUDA float64 parity is exact to ~1e-12** (same fit, same predictions), but on this
  smoke-scale problem (112 sources, 32k points) the GPU shows **no speedup** (0.3x -- kernel
  launch + transfer overhead dominates small dense operators). GPU throughput is expected to
  pay off only at much larger source counts / query batches; measure before assuming.