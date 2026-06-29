# VESP-UQ Significance Tests (WP-A)

Each candidate selector vs the strong altitude comparator (`min_altitude`). `delta > 0` means the candidate beats altitude on that metric. Two complementary tests: a Wilcoxon signed-rank across seeds (exact), and a trajectory-level paired bootstrap (95% CI + p) on the first seed. A claim of 'beats altitude' requires the bootstrap CI to exclude 0; otherwise the verdict is 'indistinguishable from altitude' and the contribution is the calibrated covariance (Table A).

| band | candidate | metric | seed dDelta | Wilcoxon p | boot delta [95% CI] | boot p | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| vespuq_smoke | vespuq_supervisor | spearman | 0.101 | 1.000 | 0.101 [-0.085, 0.290] | 0.311 | indistinguishable |
| vespuq_smoke | vespuq_supervisor | capture | -0.250 | 1.000 | -0.250 [-0.800, 0.250] | 0.831 | indistinguishable |
| vespuq_smoke | vespuq_supervisor | auroc | -0.097 | 1.000 | -0.097 [-0.208, 0.126] | 0.634 | indistinguishable |
| vespuq_smoke | uncertainty_only | spearman | -0.113 | 1.000 | -0.113 [-0.249, 0.026] | 0.103 | indistinguishable |
| vespuq_smoke | uncertainty_only | capture | 0.000 | n/a | 0.000 [-0.400, 0.250] | 1.000 | indistinguishable |
| vespuq_smoke | uncertainty_only | auroc | -0.076 | 1.000 | -0.076 [-0.194, 0.021] | 0.153 | indistinguishable |

The Wilcoxon p across only a few seeds has low power; the bootstrap CI on a full trajectory ensemble is the primary significance evidence. Both target force-model error, not position error.

