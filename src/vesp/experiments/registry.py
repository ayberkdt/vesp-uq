"""Catalogue of the core experiments (E0–E5) and the questions they answer.

This is documentation-as-code: it ties each experiment id to the falsifiable
question from the project plan and to the YAML config that implements it. It is used
by the runner scripts (``--experiment E3``) and rendered into reports so a reader can
trace a result back to the question it was meant to answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentInfo:
    eid: str
    name: str
    question_id: str
    question: str
    config: str
    kind: str
    purpose: str


# Configs are paths relative to the repository root.
CORE_EXPERIMENTS: dict[str, ExperimentInfo] = {
    "E0": ExperimentInfo(
        eid="E0",
        name="synthetic_exact_recovery",
        question_id="Q1",
        question="Can a fixed-source equivalent-source model reproduce a known synthetic exterior residual field?",
        config="configs/experiments/synthetic_exact_recovery.yaml",
        kind="single",
        purpose="Sanity check the kernel, acceleration kernel, solver and metrics. Same source shell as truth.",
    ),
    "E1": ExperimentInfo(
        eid="E1",
        name="synthetic_shell_radius_mismatch",
        question_id="Q2",
        question="How sensitive is the fit to source-shell radius mismatch?",
        config="configs/experiments/synthetic_shell_radius_mismatch.yaml",
        kind="sweep",
        purpose="Fix truth shell, sweep model shell radius; compare RMSE, sigma_l2, concentration, low-altitude error.",
    ),
    "E2": ExperimentInfo(
        eid="E2",
        name="synthetic_multishell_truth",
        question_id="Q3",
        question="Does multi-shell fitting improve accuracy or create shell cancellation?",
        config="configs/experiments/synthetic_multishell_truth.yaml",
        kind="sweep",
        purpose="Multi-shell truth + multi-shell model under weak vs strong L2; measure shell_cancellation_ratio.",
    ),
    "E3": ExperimentInfo(
        eid="E3",
        name="synthetic_l2_sweep",
        question_id="Q4",
        question="How much L2 regularization is needed to suppress ill-conditioned source solutions?",
        config="configs/experiments/synthetic_l2_sweep.yaml",
        kind="sweep",
        purpose="Tikhonov sweep; find the stable Pareto-like region between data fit and source stability.",
    ),
    "E4": ExperimentInfo(
        eid="E4",
        name="synthetic_entropy_pareto",
        question_id="Q5",
        question="Does entropy regularization improve source distribution health at acceptable data-error cost?",
        config="configs/experiments/synthetic_entropy_pareto.yaml",
        kind="sweep",
        purpose="Entropy weight x mode sweep at fixed L2 vs the ridge baseline (entropy_weight=0).",
    ),
    "E5": ExperimentInfo(
        eid="E5",
        name="real_lunar_proof_of_concept",
        question_id="Q6",
        question="On lunar band-limited residual data, is the method numerically stable across altitude bands?",
        config="configs/experiments/real_lunar_ridge_baseline.yaml",
        kind="suite",
        purpose=(
            "GRAIL-derived band-limited lunar residual proof-of-concept: ridge baseline + L2 sweep "
            "+ entropy Pareto. NOT a true internal density reconstruction."
        ),
    ),
    "E6": ExperimentInfo(
        eid="E6",
        name="synthetic_maxent_constrained_ood",
        question_id="Q5",
        question="At equal data fit, does maximum-entropy source selection generalize better to OOD altitude than ridge?",
        config="configs/experiments/synthetic_maxent_constrained_ood.yaml",
        kind="sweep",
        purpose=(
            "The FAIR MaxEnt test: constrained (Skilling-Bryan) MaxEnt vs ridge at equal in-sample "
            "data misfit, judged on held-out low/high altitude generalization (not training RMSE)."
        ),
    ),
    "E7": ExperimentInfo(
        eid="E7",
        name="synthetic_regularizer_shootout",
        question_id="Q5",
        question="At matched data error, which regularizer (L2 vs entropy) buys more source-distribution health?",
        config="configs/experiments/synthetic_regularizer_shootout.yaml",
        kind="sweep",
        purpose=(
            "Regularizer shootout: ridge L2 knob vs constrained-MaxEnt entropy knob, aligned by data "
            "error, compared on health (cancellation / sigma_l2 / concentration / effective count). "
            "Run via scripts/regularizer_shootout.py. A concentrated variant gives entropy its best shot."
        ),
    ),
    "E8": ExperimentInfo(
        eid="E8",
        name="synthetic_geometry_shootout",
        question_id="Q2",
        question="Does source geometry (esp. surface-near density) reduce held-out low-altitude error?",
        config="configs/experiments/synthetic_geometry_shootout.yaml",
        kind="sweep",
        purpose=(
            "Source-geometry shootout: compare shell geometries (single / multi / deep / surface-dense / "
            "multi-resolution / denser), each with lambda_l2: auto, ranked by held-out low-altitude error. "
            "Tests the E7 forward hypothesis that the low-altitude bottleneck is geometry, not regularization. "
            "Run via scripts/geometry_shootout.py (optionally --with-calibration)."
        ),
    ),
}

# E5 is realised as several configs (ridge baseline, L2 sweep, entropy Pareto).
E5_CONFIGS = [
    "configs/experiments/real_lunar_ridge_baseline.yaml",
    "configs/experiments/real_lunar_l2_sweep.yaml",
    "configs/experiments/real_lunar_entropy_pareto.yaml",
]


def experiment_info(eid: str) -> ExperimentInfo:
    key = eid.strip().upper()
    if key not in CORE_EXPERIMENTS:
        raise KeyError(f"unknown experiment id {eid!r}; known: {sorted(CORE_EXPERIMENTS)}")
    return CORE_EXPERIMENTS[key]
