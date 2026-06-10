# VESP-UQ Report - Equivalent-Source Force-Risk / OOD Calibration Layer

dataset: `data/lunar_grail_gl0420a_L60_residual.csv`
sources: 1280  |  reg: lcurve (lambda_l2=10)  |  noise_model: heteroscedastic  |  covariance_mode: exact  |  global noise_std=0.000211
units: risk_score=`model_normalized_accel`, acceleration=`km/s^2`, position=`normalized`  (Risk scores and expected force errors are in the model's normalized-acceleration units (dU/d(model coordinate)) by default. A physical conversion is applied only when explicit metadata is supplied (body.acceleration_scale_m_s2 or a physical body.acceleration_units); see the physical_conversion_* fields below. No physical scale is ever inferred.)
physical acceleration conversion: available (1 model unit = 1.000e+03 m/s^2, source `declared_physical_units`); model-normalized values are also retained.
altitude noise sigma^2(h)=a*h^(-b): a=2.194e-12, b=3.697 (h=r-1; larger b = faster growth toward surface)

## Experiment 1 - Standalone residual-error calibration

| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.308 | 1.925e-04 | 2.370e-04 | 2.945e-05 | 0.67 | 0.96 | 0.97 | 1.34 | -7.239 |
| low | 1.092 | 3.575e-04 | 3.210e-04 | 7.005e-05 | 1.09 | 0.87 | 0.87 | 3.56 | -6.614 |
| mid | 1.251 | 1.390e-04 | 2.141e-04 | 2.538e-05 | 0.64 | 0.97 | 0.99 | 1.24 | -7.323 |
| high | 1.475 | 5.782e-05 | 2.115e-04 | 1.094e-05 | 0.27 | 1.00 | 1.00 | 0.22 | -7.505 |

- Epistemic uncertainty grows toward low altitude: **YES** (low/high epistemic std ratio = 6.40, predictive sigma ratio = 1.52).

## Experiment 3 - Trajectory risk screening (force-risk vs supplied true-error metric)

- ensemble: 10000 trajectories (generated), 1200000 output points (scoring = `supervisor_rel`, oracle = `heldout`, true-error aggregator = `p95`, time-weighting = `none`, domain-support on)
- **Relative scoring mode** (`supervisor_rel` = `supervisor_rel`): for prioritization/ranking only, **not** absolute physical thresholding.
- selection: `fraction` (policy `topk`, requested 20.0%)
- flagged 2000/10000 (20.0%)
- expected force-error per orbit (ensemble mean | max): mean 4.526e-04 | max 2.155e-03 (model_normalized_accel)
- capture rate (top-decile true-error orbits flagged): **0.70**  | precision: 0.35  | lift over random: 3.49x
- Spearman(force-risk, supplied true-error metric): 0.77
- mean true error  flagged: 1.297e-03  vs  accepted: 8.108e-04  (ratio 1.60x)

### What these metrics mean

- **force-risk score** = the VESP-UQ trajectory risk (expected force-model error / OOD). The **supplied true-error metric** is an external diagnostic oracle (e.g. a position-error read) used only to *validate* ranking; VESP-UQ does not predict it by construction.
- **force-risk ranking** (Spearman, lift): does the force-risk score order orbits the way the supplied true-error metric does?
- **trajectory-error ranking** (capture rate, error ratio): do flagged orbits carry larger *true trajectory* error -- a different question from force-risk calibration.
- **false-alarm behavior**: under an absolute force-risk budget a safe set may flag zero; a fixed top-fraction always flags ~`rerun_fraction` by construction.
- **rerun prioritization**: relative supervisor modes *rank* which orbits to rerun first; absolute modes decide whether *any* orbit exceeds a physical budget.

## Runtime

- fit: 2.232 s  |  calibration eval: 0.110 s
- scoring: 15.283 ms/trajectory (127.36 us/output point, 1200000 points total)
- _VESP-UQ is evaluated at output trajectory points only, not inside every integrator RHS call._

## IAC claim summary

- **What was fitted?** An interior equivalent-source posterior over the residual-force error `e_a = a_reference - a_surrogate` (1280 sources, lcurve regularization).
- **What was calibrated?** Altitude-dependent predictive uncertainty (post-hoc heteroscedastic recalibration) on held-out validation residuals; the posterior mean equals the ridge point estimate.
- **Did low-altitude uncertainty increase?** Yes (low/high epistemic std ratio = 6.40).
- **PICP90 by band (low/mid/high):** 0.87 / 0.97 / 1.00.
- **Fraction of trajectories flagged:** 20.0% (selection `fraction`, capture rate 0.70, lift over random 3.49x).
- **Did flagged trajectories carry larger true error?** Yes (1.60x the accepted-set error).
- **Runtime overhead:** 15.283 ms/trajectory, 127.36 us/output point (post-processing only).
- **What should NOT be claimed:** not a better deterministic surrogate; not a position-error predictor; not true lunar density recovery; not operational orbit covariance propagation; not integrated with ST-LRPS. VESP-UQ is a force-risk / OOD uncertainty-calibration layer at the acceleration interface.

