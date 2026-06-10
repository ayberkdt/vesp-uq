# VESP-UQ IAC Benchmark Summary

config: `configs/vespuq/vespuq_real_lunar.yaml`

VESP-UQ is a force-risk / OOD calibration layer. The core benchmarks below test force-model
risk detection and selective rerun -- NOT trajectory position-error prediction.

- **force-error ranking** Spearman: 0.7548877951688779  (lift 4.05x) -- the core claim.
- **OOD altitude sweep**: expected force error grows toward low altitude: True.
- **absolute threshold**: zero-alarm capable: True.
- **calibration**: low-band PICP90 0.8731884057971014.
- **position-error diagnostic**: data_available (diagnostic only; not a VESP-UQ claim).

Not claimed: deterministic trajectory-accuracy improvement; position-error prediction;
operational orbit covariance propagation; ST-LRPS integration.

