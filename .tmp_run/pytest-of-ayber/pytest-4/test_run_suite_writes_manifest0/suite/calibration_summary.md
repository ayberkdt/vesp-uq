# VESP-UQ Calibration Robustness (Table A)

Per-altitude-band held-out force-error calibration, mean +/- std across seeds. `z_std` ~ 1 is well-calibrated (>1 overconfident); `picp_90` should approach 0.90.

| band | region | n_seeds | rmse | mean_pred_std | z_std | picp_68 | picp_90 | ellipsoid_picp_90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vespuq_smoke | all | 1 | 2.445e-03 +/- 0.000e+00 | 2.877e-03 +/- 0.000e+00 | 0.600 +/- 0.000 | 0.911 +/- 0.000 | 0.975 +/- 0.000 | 0.958 +/- 0.000 |
| vespuq_smoke | high | 1 | 3.494e-04 +/- 0.000e+00 | 2.207e-03 +/- 0.000e+00 | 0.158 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| vespuq_smoke | low | 1 | 4.382e-03 +/- 0.000e+00 | 4.154e-03 +/- 0.000e+00 | 1.005 +/- 0.000 | 0.747 +/- 0.000 | 0.909 +/- 0.000 | 0.848 +/- 0.000 |
| vespuq_smoke | mid | 1 | 1.380e-03 +/- 0.000e+00 | 2.602e-03 +/- 0.000e+00 | 0.458 +/- 0.000 | 0.943 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |

## Component-wise calibration (radial vs tangential, WP-C)

The predictive error covariance split into the local radial vs tangential frame. The radial component carries the altitude-dependent force-model error; a band mis-calibrated mainly in the radial axis points to the noise law / geometry rather than an isotropic scale error. `z_std` ~ 1 calibrated; Winkler is the interval score (lower is better).

| band | region | radial z_std | tangential z_std | radial PICP90 | tangential PICP90 | calib_err_90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vespuq_smoke | all | 0.717 +/- 0.000 | 0.526 +/- 0.000 | 0.958 +/- 0.000 | 0.983 +/- 0.000 | 0.071 +/- 0.000 |
| vespuq_smoke | high | 0.176 +/- 0.000 | 0.149 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.100 +/- 0.000 |
| vespuq_smoke | low | 1.213 +/- 0.000 | 0.882 +/- 0.000 | 0.848 +/- 0.000 | 0.939 +/- 0.000 | 0.045 +/- 0.000 |
| vespuq_smoke | mid | 0.562 +/- 0.000 | 0.403 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.100 +/- 0.000 |

Primary rerun budget for ranking tables: 20%.

