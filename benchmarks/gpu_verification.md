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
While GPU and float32 throughputs represent significant speedups for deployment and internal exploratory screening, **the headline calibration and scientific numbers must remain float64/CPU-reproducible.** Float32 operations on inverted covariance matrices and equivalent-source operators typically accumulate $10^{-3}$ to $10^{-5}$ relative errors. This degradation is often acceptable for ranking and bulk screening but breaks strict scientific reproducibility guarantees.