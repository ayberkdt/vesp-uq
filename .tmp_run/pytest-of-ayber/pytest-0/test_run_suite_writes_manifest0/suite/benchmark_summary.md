# VESP-UQ Benchmark Suite Summary

- configs: vespuq_smoke  |  seeds: [0]  |  primary rerun budget: 20%
- git commit: `65cb5e3969b8c698740df50fd4ed9cfd45aec5be`  |  trajectories/run: 40
- true force error: `nn_oracle_heldout` (aggregator `p95`)

## Table B -- Ranking robustness (primary budget)

Mean +/- std across seeds at the 20% rerun budget. Target: trajectory true FORCE-model error (not position error).

| band | selector | capture | precision | lift | spearman | err_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vespuq_smoke | domain_support | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 +/- 0.00 | -0.015 +/- 0.000 | 0.87 +/- 0.00 |
| vespuq_smoke | knn_p95 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 +/- 0.00 | -0.009 +/- 0.000 | 0.91 +/- 0.00 |
| vespuq_smoke | label_shuffled | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 +/- 0.00 | 0.109 +/- 0.000 | 0.94 +/- 0.00 |
| vespuq_smoke | low_altitude_exposure | 0.500 +/- 0.000 | 0.250 +/- 0.000 | 2.50 +/- 0.00 | 0.304 +/- 0.000 | 1.17 +/- 0.00 |
| vespuq_smoke | min_altitude | 0.500 +/- 0.000 | 0.250 +/- 0.000 | 2.50 +/- 0.00 | 0.342 +/- 0.000 | 1.17 +/- 0.00 |
| vespuq_smoke | random | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 +/- 0.00 | -0.342 +/- 0.000 | 0.90 +/- 0.00 |
| vespuq_smoke | uncertainty_only | 0.500 +/- 0.000 | 0.250 +/- 0.000 | 2.50 +/- 0.00 | 0.230 +/- 0.000 | 1.13 +/- 0.00 |
| vespuq_smoke | vespuq_supervisor | 0.250 +/- 0.000 | 0.125 +/- 0.000 | 1.25 +/- 0.00 | 0.443 +/- 0.000 | 1.07 +/- 0.00 |

## Altitude-controlled incremental value

Partial correlation of each score with true force error after regressing out min-radius, and the matched-altitude paired sign test (concordance > 0.5 => signal beyond altitude).

| band | selector | partial_corr(given alt) | within-bin spearman | matched concordance |
| --- | --- | ---: | ---: | ---: |
| vespuq_smoke | domain_support | 0.043 +/- 0.000 | -0.017 +/- 0.000 | 0.389 +/- 0.000 |
| vespuq_smoke | min_altitude | 0.050 +/- 0.000 | 0.095 +/- 0.000 | n/a |
| vespuq_smoke | uncertainty_only | -0.016 +/- 0.000 | -0.163 +/- 0.000 | 0.333 +/- 0.000 |
| vespuq_smoke | vespuq_supervisor | -0.066 +/- 0.000 | 0.162 +/- 0.000 | 0.556 +/- 0.000 |

Interpretation: a positive partial correlation and a matched concordance above 0.5 mean the selector carries force-error ranking signal that is NOT explained by altitude alone. If altitude-only matches or beats VESP-UQ, the added value is the calibrated local covariance (Table A), not a superior scalar ranking. This is a force-risk diagnostic only.

