# VESP-UQ Decision Quality (Table C)

Risk scores judged as the screening tool they are, against trajectory true FORCE-model error. AUROC/AUPRC detect the top-decile-error class (chance AUROC 0.5); capture-AUC is the budget-integrated capture (normalized: 1.0 = matches the oracle at every budget); oracle-regret is the captured-error gap to the oracle at the 20% budget (0 = optimal, 1 = random). Mean +/- std across seeds.

| band | selector | AUROC | AUPRC | capture-AUC (norm) | oracle-regret |
| --- | --- | ---: | ---: | ---: | ---: |
| vespuq_smoke | domain_support | 0.368 +/- 0.000 | 0.098 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 |
| vespuq_smoke | knn_p95 | 0.417 +/- 0.000 | 0.102 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 |
| vespuq_smoke | label_shuffled | 0.438 +/- 0.000 | 0.111 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 |
| vespuq_smoke | low_altitude_exposure | 0.667 +/- 0.000 | 0.406 +/- 0.000 | 0.375 +/- 0.000 | 0.562 +/- 0.000 |
| vespuq_smoke | min_altitude | 0.764 +/- 0.000 | 0.442 +/- 0.000 | 0.375 +/- 0.000 | 0.562 +/- 0.000 |
| vespuq_smoke | random | 0.236 +/- 0.000 | 0.079 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 |
| vespuq_smoke | uncertainty_only | 0.688 +/- 0.000 | 0.421 +/- 0.000 | 0.375 +/- 0.000 | 0.666 +/- 0.000 |
| vespuq_smoke | vespuq_supervisor | 0.667 +/- 0.000 | 0.167 +/- 0.000 | 0.125 +/- 0.000 | 0.829 +/- 0.000 |


