# VESP-UQ vs Gaussian-Process UQ Baseline (WP-D)

Both models are fit on the same train split and evaluated on identical held-out calibration and trajectory-screening metrics. The GP is a strong, well-understood spatial UQ baseline; VESP-UQ adds physics-structured, altitude-aware covariance. Mean +/- std across seeds.

## Calibration (per band)

| band | region | model | z_std | PICP90 | ellipsoid PICP90 | radial z_std | calib_err_90 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vespuq_smoke | all | gp | 1.108 +/- 0.000 | 0.903 +/- 0.000 | 0.858 +/- 0.000 | 1.426 +/- 0.000 | 0.050 +/- 0.000 |
| vespuq_smoke | all | vespuq | 0.600 +/- 0.000 | 0.975 +/- 0.000 | 0.958 +/- 0.000 | 0.717 +/- 0.000 | 0.071 +/- 0.000 |
| vespuq_smoke | high | gp | 0.520 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.485 +/- 0.000 | 0.089 +/- 0.000 |
| vespuq_smoke | high | vespuq | 0.158 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.176 +/- 0.000 | 0.100 +/- 0.000 |
| vespuq_smoke | low | gp | 1.735 +/- 0.000 | 0.747 +/- 0.000 | 0.576 +/- 0.000 | 1.902 +/- 0.000 | 0.188 +/- 0.000 |
| vespuq_smoke | low | vespuq | 1.005 +/- 0.000 | 0.909 +/- 0.000 | 0.848 +/- 0.000 | 1.213 +/- 0.000 | 0.045 +/- 0.000 |
| vespuq_smoke | mid | gp | 0.924 +/- 0.000 | 0.919 +/- 0.000 | 0.927 +/- 0.000 | 1.058 +/- 0.000 | 0.027 +/- 0.000 |
| vespuq_smoke | mid | vespuq | 0.458 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.562 +/- 0.000 | 0.100 +/- 0.000 |

## Decision quality (trajectory screening)

| band | model | AUROC | AUPRC | capture-AUC (norm) | oracle-regret |
| --- | --- | ---: | ---: | ---: | ---: |
| vespuq_smoke | gp | 0.743 +/- 0.000 | 0.487 +/- 0.000 | 0.631 +/- 0.000 | 0.593 +/- 0.000 |
| vespuq_smoke | min_altitude | 0.764 +/- 0.000 | 0.442 +/- 0.000 | 0.470 +/- 0.000 | 0.562 +/- 0.000 |
| vespuq_smoke | vespuq | 0.667 +/- 0.000 | 0.167 +/- 0.000 | 0.305 +/- 0.000 | 0.829 +/- 0.000 |

## Runtime

| band | model | fit (s) | predict-held (s) |
| --- | --- | ---: | ---: |
| vespuq_smoke | gp | 0.013 +/- 0.000 | 0.0194 +/- 0.0000 |
| vespuq_smoke | vespuq | 0.459 +/- 0.000 | 0.0557 +/- 0.0000 |

Interpretation: where the GP matches or beats VESP-UQ on in-support calibration, VESP-UQ's contribution is its physics-structured covariance and altitude extrapolation, not a lower in-support error. Reported honestly either way; force-model error only.

