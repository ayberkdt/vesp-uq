# VESP-UQ Report - Equivalent-Source Force-Risk / OOD Calibration Layer

dataset: `data/lunar_grail_gl0420a_L90_residual.csv`
sources: 1280  |  reg: lcurve (lambda_l2=0.0001)  |  noise_model: heteroscedastic  |  covariance_mode: exact  |  global noise_std=6.37e-05
units: risk_score=`model_normalized_accel`, acceleration=`km/s^2`, position=`normalized`  (Risk scores and expected force errors are in the model's normalized-acceleration units (dU/d(model coordinate)) by default. A physical conversion is applied only when explicit metadata is supplied (body.acceleration_scale_m_s2 or a physical body.acceleration_units); see the physical_conversion_* fields below. No physical scale is ever inferred.)
physical acceleration conversion: available (1 model unit = 1.000e+03 m/s^2, source `declared_physical_units`); model-normalized values are also retained.
altitude noise sigma^2(h)=a*h^(-b): a=1.645e-14, b=0.105 (h=r-1; larger b = faster growth toward surface)

## Experiment 1 - Standalone residual-error calibration

| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.308 | 4.949e-05 | 1.926e-04 | 1.574e-04 | 0.09 | 1.00 | 1.00 | 0.03 | -8.191 |
| low | 1.092 | 1.040e-04 | 5.806e-04 | 5.734e-04 | 0.17 | 1.00 | 1.00 | 0.09 | -6.887 |
| mid | 1.251 | 7.805e-06 | 9.757e-05 | 6.562e-05 | 0.07 | 1.00 | 1.00 | 0.01 | -8.394 |
| high | 1.475 | 1.311e-06 | 6.514e-05 | 1.175e-05 | 0.02 | 1.00 | 1.00 | 0.00 | -8.720 |

- Epistemic uncertainty grows toward low altitude: **YES** (low/high epistemic std ratio = 48.78, predictive sigma ratio = 8.91).

## Experiment 3 - Trajectory risk screening (force-risk vs supplied true-error metric)

- ensemble: 10000 trajectories (generated), 1200000 output points (scoring = `supervisor_rel`, oracle = `heldout`, true-error aggregator = `p95`, time-weighting = `none`, domain-support on)
- **Relative scoring mode** (`supervisor_rel` = `supervisor_rel`): for prioritization/ranking only, **not** absolute physical thresholding.
- selection: `fraction` (policy `topk`, requested 20.0%)
- flagged 2000/10000 (20.0%)
- expected force-error per orbit (ensemble mean | max): mean 1.917e-04 | max 9.921e-03 (model_normalized_accel)
- capture rate (top-decile true-error orbits flagged): **0.73**  | precision: 0.38  | lift over random: 3.64x
- Spearman(force-risk, supplied true-error metric): 0.84
- mean true error  flagged: 1.036e-04  vs  accepted: 1.986e-05  (ratio 5.22x)

### What these metrics mean

- **force-risk score** = the VESP-UQ trajectory risk (expected force-model error / OOD). The **supplied true-error metric** is an external diagnostic oracle (e.g. a position-error read) used only to *validate* ranking; VESP-UQ does not predict it by construction.
- **force-risk ranking** (Spearman, lift): does the force-risk score order orbits the way the supplied true-error metric does?
- **trajectory-error ranking** (capture rate, error ratio): do flagged orbits carry larger *true trajectory* error -- a different question from force-risk calibration.
- **false-alarm behavior**: under an absolute force-risk budget a safe set may flag zero; a fixed top-fraction always flags ~`rerun_fraction` by construction.
- **rerun prioritization**: relative supervisor modes *rank* which orbits to rerun first; absolute modes decide whether *any* orbit exceeds a physical budget.

## Runtime

- fit: 3.504 s  |  calibration eval: 0.190 s
- scoring: 18.938 ms/trajectory (157.81 us/output point, 1200000 points total)
- _VESP-UQ is evaluated at output trajectory points only, not inside every integrator RHS call._

## IAC claim summary

- **What was fitted?** An interior equivalent-source posterior over the residual-force error `e_a = a_reference - a_surrogate` (1280 sources, lcurve regularization).
- **What was calibrated?** Altitude-dependent predictive uncertainty (post-hoc heteroscedastic recalibration) on held-out validation residuals; the posterior mean equals the ridge point estimate.
- **Did low-altitude uncertainty increase?** Yes (low/high epistemic std ratio = 48.78).
- **PICP90 by band (low/mid/high):** 1.00 / 1.00 / 1.00.
- **Fraction of trajectories flagged:** 20.0% (selection `fraction`, capture rate 0.73, lift over random 3.64x).
- **Did flagged trajectories carry larger true error?** Yes (5.22x the accepted-set error).
- **Runtime overhead:** 18.938 ms/trajectory, 157.81 us/output point (post-processing only).
- **What should NOT be claimed:** not a better deterministic surrogate; not a position-error predictor; not true lunar density recovery; not operational orbit covariance propagation; not integrated with ST-LRPS. VESP-UQ is a force-risk / OOD uncertainty-calibration layer at the acceleration interface.

