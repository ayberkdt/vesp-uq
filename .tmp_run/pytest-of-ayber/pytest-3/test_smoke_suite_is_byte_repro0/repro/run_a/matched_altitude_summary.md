# Matched-Altitude Pair Test (incremental value beyond altitude)

Trajectories are matched to within a min-radius caliper; for each matched pair the higher-scoring trajectory is compared to the lower-scoring one. A concordance rate above 0.50 means higher score tracks higher true FORCE error *at the same altitude*. Values are mean +/- std across seeds; the one-sided sign-test p-value tests concordance > 0.5.

| band | selector | n_pairs | concordance | mean d(true error) | sign-test p |
| --- | --- | ---: | ---: | ---: | ---: |
| vespuq_smoke | domain_support | 18 +/- 0 | 0.389 +/- 0.000 | -2.018e-03 +/- 0.000e+00 | 0.881 +/- 0.000 |
| vespuq_smoke | uncertainty_only | 18 +/- 0 | 0.333 +/- 0.000 | -2.386e-03 +/- 0.000e+00 | 0.951 +/- 0.000 |
| vespuq_smoke | vespuq_supervisor | 18 +/- 0 | 0.556 +/- 0.000 | 2.894e-03 +/- 0.000e+00 | 0.407 +/- 0.000 |

A concordance indistinguishable from 0.50 means the score adds no altitude-independent ranking signal in that band; the calibrated local covariance remains the contribution.

