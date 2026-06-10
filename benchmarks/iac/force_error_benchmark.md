# VESP-UQ Force-Error Ranking Benchmark

**This is a FORCE-ERROR benchmark, not a position-error benchmark.** It asks whether the
VESP-UQ force-risk score ranks the surrogate's true force-model error along a trajectory.

- scoring: `supervisor_rel_p95`  |  true force error: `nn_oracle_heldout` (aggregator `p95`)
- trajectories: 10000  |  top fraction flagged: 10%

- **Spearman(force-risk, true force error): 0.7549**
- capture rate: 0.4050  |  precision: 0.4090  |  lift over random: 4.05x
- mean true force error flagged: 1.354e-03  vs  accepted: 8.585e-04  (ratio 1.58x)

Interpretation: a positive Spearman / lift > 1 means the force-risk score concentrates the
surrogate's true force-model error -- the core VESP-UQ value (force-risk / OOD detection).
It does NOT imply prediction of long-horizon trajectory position error.

