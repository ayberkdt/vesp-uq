# VESP-UQ OOD Altitude Sweep (force-risk / OOD detection)

Expected force error and domain-support risk along one direction at decreasing altitude.

| radius | expected_force_error | sigma | domain_risk |
| ---: | ---: | ---: | ---: |
| 1.02 | 3.632e-03 | 3.573e-03 | 0.026 |
| 1.05 | 9.852e-04 | 7.651e-04 | 0.000 |
| 1.10 | 6.809e-04 | 4.209e-04 | 0.000 |
| 1.20 | 5.329e-04 | 3.728e-04 | 0.000 |
| 1.35 | 4.366e-04 | 3.670e-04 | 0.000 |
| 1.50 | 3.973e-04 | 3.661e-04 | 0.000 |

- expected force error grows toward low altitude: **YES** (3.632e-03 at r=1.02 vs 3.973e-04 at r=1.5).

