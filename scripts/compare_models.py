"""CLI to compare two VESP-UQ models and generate a drift report."""

import argparse
import sys
from pathlib import Path

from vesp.data.dataset import load_csv_dataset
from vesp.uq.compare import compare_models
from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.io import load_trajectory_csv
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.plugin import VESPUQPlugin


def build_comparison_md(report: dict, model_a_path: str, model_b_path: str) -> str:
    """Format the comparison dict as a markdown report."""
    md = [
        f"# VESP-UQ Model Comparison Report\n",
        f"- **Model A:** `{model_a_path}`",
        f"- **Model B:** `{model_b_path}`\n",
    ]
    
    post = report["posterior_distance"]
    md.extend([
        "## Posterior Distance (Drift)",
        f"- **Mean L2 Diff:** {post['mean_l2_diff']:.4e}",
        f"- **Covariance Frobenius Diff:** {post['cov_frob_diff']:.4e}",
        f"- **Noise Floor Variance Delta (B - A):** {post['noise_var_delta']:.4e}\n",
    ])
    
    ds = report["domain_shift"]
    md.extend([
        "## Domain Support Shift",
        "*(Scores for Model B's domain support evaluated on Model A's training set)*",
        f"- **Mean Score:** {ds['mean_score_on_A']:.4f}",
        f"- **Max Score:** {ds['max_score_on_A']:.4f}\n",
    ])
    
    cal = report["calibration"]
    if cal:
        md.append("## Calibration Comparison (Bands)")
        md.append("| Band | RMSE (A) | RMSE (B) | Mean Pred Std (A) | Mean Pred Std (B) | PICP90 (A) | PICP90 (B) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for band, metrics in cal.items():
            md.append(
                f"| {band} | {metrics['rmse']['A']:.4e} | {metrics['rmse']['B']:.4e} "
                f"| {metrics['mean_pred_std']['A']:.4e} | {metrics['mean_pred_std']['B']:.4e} "
                f"| {metrics['picp_90']['A']:.2f} | {metrics['picp_90']['B']:.2f} |"
            )
        md.append("\n")
        
    agr = report["screening_agreement"]
    if agr:
        md.extend([
            "## Screening Agreement",
            f"- **Risk Spearman Correlation:** {agr['risk_spearman']:.4f}",
            f"- **Flag Overlap (IoU):** {agr['flag_overlap']:.4f}",
            f"- **Flag Count:** {agr['n_flagged_A']} (A) vs {agr['n_flagged_B']} (B)\n",
        ])
        
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two VESP-UQ models.")
    parser.add_argument("--model-a", required=True, help="Path to first vespuq_plugin.pt")
    parser.add_argument("--model-b", required=True, help="Path to second vespuq_plugin.pt")
    parser.add_argument("--data", required=True, help="Held-out calibration CSV")
    parser.add_argument("--trajectories", help="Optional trajectories CSV for screening agreement")
    parser.add_argument("--out", required=True, help="Output directory for reports")
    args = parser.parse_args()

    print(f"Loading Model A: {args.model_a}")
    plugin_a = VESPUQPlugin.load(args.model_a, device="cpu")
    print(f"Loading Model B: {args.model_b}")
    plugin_b = VESPUQPlugin.load(args.model_b, device="cpu")

    print(f"Loading calibration data from: {args.data}")
    samples = load_uq_samples_from_csv(args.data)

    ensemble = None
    if args.trajectories:
        print(f"Loading trajectories from: {args.trajectories}")
        ensemble = load_trajectory_csv(args.trajectories)

    print("Computing comparisons...")
    report = compare_models(
        plugin_a=plugin_a,
        plugin_b=plugin_b,
        held_out_positions=samples.positions,
        held_out_error=samples.error,
        trajectory_ensemble=ensemble,
        scoring_mode="supervisor_rel",
        threshold_policy={"type": "fraction", "fraction": 0.2},
    )

    md_report = build_comparison_md(report, args.model_a, args.model_b)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    inputs = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "data": args.data,
    }
    if args.trajectories:
        inputs["trajectories"] = args.trajectories

    write_run_artifacts(
        out_dir=out_dir,
        tool="compare_models",
        json_files={"model_comparison.json": report},
        text_files={"model_comparison.md": md_report},
        config=inputs,
        seed=0,
    )
    print(f"Comparison reports written to {out_dir}")


if __name__ == "__main__":
    main()
