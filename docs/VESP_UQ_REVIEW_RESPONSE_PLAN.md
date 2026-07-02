# VESP-UQ Review Response Plan

Created: 2026-07-02
Status: RWP-1 through RWP-6 cleaned; RTN promotion held pending stronger evidence

This plan responds to the external review in `pasted-text.txt`. It is intentionally narrower than
`VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`: the goal here is not a general refactor roadmap, but a
claim-safety and evidence plan for the review points that are materially correct.

The binding claim documents remain `SCIENTIFIC_CLAIMS.md` and `VESP_UQ_LIMITATIONS.md`. Nothing in
this plan expands the claim surface.

## Evidence Snapshot

Checked anchors:

| Review point | Verdict | Evidence |
| --- | --- | --- |
| Equivalent-source field is structurally conservative / curl-free. | Correct. | `src/vesp/core/kernels.py` builds acceleration from the Newtonian `1/r` gradient (`diff * inv_r^3`). VESP-UQ can represent conservative residual-force fields, not arbitrary vector fields. |
| Current curl gate exists, but the scattered proxy is not decision-grade. | Correct. | `src/vesp/uq/gate_diagnostics.py` implements Measurement C with a conservative-control subtraction. A quick L60/L90 run in `.tmp_run/vesp_gate_review_check` reported raw scaled curl `0.555 / 1.146`, conservative-control `0.332 / 0.907`, excess `0.223 / 0.239` on SH-derived residuals that should be conservative. The gate correctly refuses a Helmholtz extension and asks for a structured/analytic curl check. |
| Isotropic scalar altitude noise can hide directional miscalibration. | Correct. | Component metrics exist, and the quick gate reports radial-vs-tangential gaps up to `0.495` (`calibration_current.csv`). `src/vesp/uq/rtn_noise.py` is a prototype only; production remains scalar altitude noise plus optional conformal scaling. |
| Calibration is substantially post-hoc. | Correct, already partially documented. | `SCIENTIFIC_CLAIMS.md` and `VESP_UQ_LIMITATIONS.md` say the heteroscedastic noise law is held-out post-hoc recalibration. README still contains older Stage 3C language and should be tightened around the current L60/L90 evidence. |
| Operational conformal calibration fixes coverage uniformly. | Not supported. | `benchmarks/vespuq_conformal_validation.md` says only L90 mid passes the z_std/PICP90 target; L60 after conformal still over-covers (`PICP90` about `0.962-0.986`, `z_std` about `0.648-0.716`). |
| Epistemic posterior is not the dominant source of final predictive uncertainty. | Mostly correct, but should be reported carefully. | In the L60 conformal run, `mean_epistemic_std / mean_pred_std` is about `0.126-0.132` across bands. For L90 after conformal shrinkage this ratio is not semantically clean because the reported epistemic column is raw while total predictive std is conformally scaled. We need an explicit decomposition table before making a general statement. |
| "Surrogate-agnostic" evidence is too narrow. | Partially correct. | The code interface is surrogate-agnostic (`e_a = reference - surrogate`), and L90 is a second SH residual spectrum, but there is no committed, genuinely different neural surrogate residual benchmark. Claims should distinguish interface-agnostic design from empirically validated cross-surrogate performance. |
| Selective high-fidelity rerun value is not yet operationally validated. | Correct. | Current screening evidence is force-risk ranking / held-out force-error oracle. Orbit/state covariance, ST-LRPS wiring, and online correction are already marked exploratory. The README should not imply an operational rerun product. |
| Infrastructure/core ratio is high. | Correct but already covered elsewhere. | `VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md` covers `VESPUQPlugin` size, config typing, module fragmentation, covariance cost, and adapter boundary. Do not duplicate that work here. |

## Work Packages

### RWP-1 - Tighten Claim Language in README and Claims Docs

Why: The docs are honest in many places, but the front-door README still mixes older Stage 3C
language with current gate/conformal evidence. A reviewer will read the strongest sentence, not the
most careful caveat.

Do:

- Add a first-class "Conservative-field limitation" caveat to the VESP-UQ section: the current
  equivalent-source posterior represents gradients of a scalar potential. Non-conservative residual
  components are out of model class unless a future measured Helmholtz extension is justified.
- Reword "surrogate-agnostic" as "surrogate-interface agnostic" unless and until a genuinely
  different surrogate residual benchmark is committed.
- Replace broad "calibrated" phrasing with "held-out/post-hoc calibrated force-error uncertainty",
  with the exchangeability and altitude-OOD caveats kept visible.
- Update or cross-link current L60/L90 conformal validation: operational conformal is implemented,
  but not a uniform sharpness fix.
- Demote selective rerun language to force-risk prioritization unless an end-to-end high-fidelity
  rerun budget experiment exists.

Acceptance:

- `README.md`, `SCIENTIFIC_CLAIMS.md`, and `VESP_UQ_LIMITATIONS.md` agree on the conservative-field,
  post-hoc calibration, conformal partial-failure, and surrogate-evidence boundaries.
- `scripts/run_claim_lint.py` stays green.

Implemented:

- README now says "surrogate-interface agnostic", calls out the conservative-field limitation, and
  frames rerun language as force-risk prioritization / possible follow-up rather than a validated
  operational rerun product.
- `SCIENTIFIC_CLAIMS.md` and `VESP_UQ_LIMITATIONS.md` now carry the same conservative-field,
  post-hoc calibration, conformal partial-validation, and cross-surrogate evidence boundaries.
- Claim lint passes on `README.md`, `SCIENTIFIC_CLAIMS.md`, and `VESP_UQ_LIMITATIONS.md`.

### RWP-2 - Replace the Curl Proxy Gate with a Structured Conservative-Field Sanity Check

Why: The current Measurement C is valuable as a warning, but it produces material excess curl on a
known conservative SH-derived residual. That means it cannot be used to open a Helmholtz/non-
conservative architecture path.

Do:

- Keep the current scattered local-linear curl metric as a diagnostic artifact, not as a promotion
  gate.
- Add an analytic or structured sanity check for SH-derived residuals. Candidate options:
  finite-difference the scalar potential on matched local stencils, or evaluate curl in a basis where
  the expected curl is known to be zero.
- Define the failure threshold on known conservative controls before running any real neural
  residual.
- Only evaluate Helmholtz/non-conservative extensions after the conservative control is near-zero
  and a real surrogate residual shows excess curl above that control.

Acceptance:

- A known SH conservative residual passes the curl sanity check.
- The gate report explicitly says "proxy-limited" when the scattered metric disagrees with the
  structured control.
- No Helmholtz extension is implemented until this gate is credible.

Implemented:

- `gate_diagnostics.py` now keeps the scattered local-linear curl proxy as a diagnostic artifact and
  adds `measurement_c_sh_potential_curl_control`, a structured central-difference curl check built
  from the scalar-potential SH generator recorded in the dataset metadata.
- On the current L60/L90 quick gate, the scattered excess remains `0.223 / 0.239`, while the
  structured SH control is about `1e-16` and passes. The decision table therefore keeps Helmholtz
  closed and treats the scattered excess as proxy/sampling error.
- `tests/test_gate_diagnostics.py` pins this behavior.

### RWP-3 - Promote Directional Calibration Reporting Before Changing the Noise Model

Why: The review is right that scalar altitude noise can hide radial/tangential miscalibration. The
repo already has component metrics and an RTN-style prototype, so the next step is measured reporting
and a gated before/after, not a blind production feature.

Do:

- Surface `radial_z_std`, `tangential_z_std`, radial/tangential PICP90, and Winkler scores in the
  primary L60/L90 reports and README evidence tables.
- Run the existing `rtn_noise.py` prototype on L60 and L90 with train/held-out separation and report
  before/after component calibration.
- Reject the prototype if it creates over-confidence in any band, even if it improves one axis.
- If accepted, integrate as an opt-in covariance recalibration mode; default remains scalar
  heteroscedastic noise.

Acceptance:

- Component calibration tables exist for raw heteroscedastic, conformal, and RTN-prototype paths.
- At least one band improves radial/tangential calibration error without any band exceeding the
  over-confidence guardrail.
- If the gate fails, document the negative result and keep the production path unchanged.

Implemented / measured:

- The main VESP-UQ Markdown report now surfaces a "Component-wise calibration" table with radial
  and tangential `z_std`, PICP90, Winkler90, and mean calibration-error columns.
- `calibration_by_band.csv` now appends the same component-wise columns while preserving the
  existing legacy columns at the front of the file.
- The RTN/noise-model before/after gate was run in quick mode on L60/L90:
  `scripts/run_vespuq_rtn_noise_prototype.py --configs configs/vespuq/vespuq_real_lunar.yaml
  configs/vespuq/vespuq_real_lunar_L90.yaml --quick --out .tmp_run/vesp_rtn_review_check`.
  The shrink-allowed prototype improved the overall aggregate calibration error
  (`L60: 0.228 -> 0.0834`, `L90: 0.446 -> 0.291`) but both cases were `partial` because regional
  guardrails still held (`L60` low/mid, `L90` low).
- The conservative no-shrink variant was also checked and did not improve the overall cases
  (`L60: hold`, `L90: hold`).
- Result: do not promote RTN/noise-model recalibration into production. Production still defaults
  to the scalar heteroscedastic path plus optional conformal scaling.

### RWP-4 - Add an Uncertainty Decomposition Table

Why: The current reports can make it hard to see how much uncertainty comes from posterior
epistemic covariance versus scalar noise versus conformal scaling. That is exactly the reviewer's
sharpest framing critique.

Do:

- Add a per-band decomposition table with `mean_epistemic_std`, aleatoric/noise contribution,
  conformal scale, final `mean_pred_std`, and ratios computed with semantically aligned columns.
- Report raw and conformalized paths separately so ratios are not mixed after shrinkage.
- Add a short "honest reading" paragraph: VESP-UQ's final calibration may be dominated by empirical
  altitude noise/conformal terms even though the equivalent-source posterior supplies the physical
  covariance structure.

Acceptance:

- The report can answer: "How much of final sigma is posterior epistemic vs post-hoc noise?" for
  L60 and L90 without manual CSV arithmetic.
- No claim says the posterior covariance dominates unless the table supports it.

Implemented:

- The main VESP-UQ Markdown report now includes an "Uncertainty decomposition" table with final
  predictive standard deviation, raw posterior-epistemic standard deviation, `epi/pred`, approximate
  post-hoc remainder, `remainder/pred`, and the applied conformal scale.
- `calibration_by_band.csv` now appends `epistemic_to_pred_std_ratio`,
  `approx_posthoc_remainder_std`, `approx_posthoc_remainder_to_pred_std_ratio`, and
  `conformal_prediction_scale`.
- The report labels the remainder as an approximate diagnostic from band-mean standard deviations,
  so it does not overstate this as an exact variance decomposition.

### RWP-5 - Produce a Real Cross-Surrogate Evidence Gate or Narrow the Claim

Why: L90 is a useful second residual spectrum, but it is not the same as validating across a
different surrogate family. The interface is generic; the evidence is still mostly SH-derived.

Do one of:

- Add a committed benchmark residual from a genuinely different surrogate family (for example an
  ST-LRPS or SIREN residual with reference acceleration pairs), then run the same L60/L90-style
  calibration and screening reports.
- Or explicitly narrow all public claims to "surrogate-interface agnostic; empirically shown on
  SH-derived residual spectra."

Acceptance:

- Either a second-surrogate report exists with the same evidence contract, or the word
  "surrogate-agnostic" is consistently qualified in README, reports, and claim summaries.

Implemented:

- Chose the claim-narrowing path rather than inventing a weak second-surrogate artifact.
- README, model-card text, IAC plan, and user-facing UQ module docs now use
  "surrogate-interface agnostic" and state that current committed evidence is SH-derived.
- No public claim now treats L90 as a genuinely different surrogate-family validation.

### RWP-6 - Validate or Demote the High-Fidelity Rerun Value Proposition

Why: The practical value proposition is selective rerun prioritization. Current evidence supports
force-risk ranking, not a validated operational high-fidelity rerun loop.

Do:

- Define an end-to-end rerun-budget experiment: given a budget fraction, choose trajectories by
  VESP-UQ score, rerun or substitute high-fidelity force/reference data, and compare captured
  force-error reduction against altitude and random baselines.
- Keep the target force-error scoped. Do not promote position-error or orbit-covariance claims.
- If no such experiment is added now, adjust docs so "rerun" is framed as a plausible downstream
  use, not a validated operational contribution.

Acceptance:

- A rerun-budget table exists with capture, regret, and baseline comparisons, or docs explicitly
  demote the value proposition.

Implemented:

- Chose the demotion path rather than claiming an operational high-fidelity rerun loop.
- README, IAC plan, benchmark-summary text, physical-budget reports, and selection/scoring docs now
  frame this as force-risk follow-up prioritization / screening guidance.
- "Rerun" remains in API names and legacy CSV column names, but the surrounding documentation says
  the evidence is force-error ranking and held-out oracle diagnostics, not an operational rerun
  product.

## Recommended Order

1. RWP-1: tighten claims and README first. This is the cheapest credibility gain and prevents stale
   language from overstating the current evidence.
2. RWP-2: repair the curl gate before any non-conservative modeling discussion.
3. RWP-4: add uncertainty decomposition so the paper can explain what calibration is actually doing.
4. RWP-3: run the RTN prototype as a measured before/after gate.
5. RWP-5 and RWP-6: either produce stronger evidence or narrow the claims.

Current disposition: all six review-response work packages are complete or explicitly held by a
measured gate. Remaining work, if desired later, is new evidence generation: a genuinely different
surrogate-family residual benchmark and an end-to-end rerun-budget experiment.

## Non-Goals

- Do not implement Helmholtz/non-conservative sources from the current scattered curl proxy.
- Do not make RTN anisotropic noise the default until it passes a held-out, no-overconfidence gate.
- Do not duplicate the general architecture refactor work already captured in
  `VESP_ARCHITECTURE_IMPROVEMENT_PLAN.md`.
- Do not claim validated orbit covariance, position-error prediction, or operational ST-LRPS
  integration.
