import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _load_comparison_csv(path: str, band: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" in df.columns:
        return df
    if "baseline" not in df.columns:
        raise ValueError("comparison CSV must contain either 'method' or 'baseline'")
    out = df.rename(
        columns={
            "baseline": "method",
            "lift_over_random": "lift",
            "force_error_ratio_flagged_to_accepted": "flagged_accepted_error_ratio",
        }
    ).copy()
    out["band"] = band
    out["flagged_fraction"] = out.get("rerun_fraction", np.nan)
    for col in ("capture_rate_std", "precision_std", "lift_std", "flagged_accepted_error_ratio_std"):
        if col not in out.columns:
            out[col] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot baseline comparisons.")
    parser.add_argument("--l60-csv", type=str, required=True)
    parser.add_argument("--l90-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/baselines")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_l60 = _load_comparison_csv(args.l60_csv, "L60")
    df_l90 = _load_comparison_csv(args.l90_csv, "L90")
    df = pd.concat([df_l60, df_l90], ignore_index=True)

    # Filter methods to keep only the required ones
    keep_methods = ["random", "min_altitude", "knn_p95", "supervisor"]
    method_labels = {
        "random": "Random",
        "min_altitude": "Altitude-only",
        "knn_p95": "kNN-only",
        "supervisor": "VESP-UQ"
    }
    
    df = df[df["method"].isin(keep_methods)].copy()
    df["method"] = df["method"].map(method_labels)
    # Order methods
    df["method"] = pd.Categorical(df["method"], categories=list(method_labels.values()), ordered=True)
    df = df.sort_values(["band", "method"])

    # Grouped bar chart: Capture Rate and Lift for L60 and L90
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    methods = list(method_labels.values())
    x = np.arange(len(methods))
    width = 0.35

    bands = ["L60", "L90"]
    colors = {"Capture Rate": "#4c72b0", "Lift over Random": "#dd8452"}

    for i, band in enumerate(bands):
        ax = axes[i]
        band_df = df[df["band"] == band]
        
        # Plot Capture Rate
        cap_rates = band_df["capture_rate"].values
        cap_err = band_df["capture_rate_std"].replace(np.nan, 0).values
        
        rects1 = ax.bar(x - width/2, cap_rates, width, yerr=cap_err, 
                        label='Capture Rate', color=colors["Capture Rate"],
                        capsize=3)
        
        # Plot Lift (on a secondary axis if it's very different, but since they are small, maybe we can normalize or use twinx)
        # Actually, Lift is around 1.0-5.0, Capture rate is 0.2-1.0. Twinx is better.
        ax2 = ax.twinx()
        lifts = band_df["lift"].values
        lift_err = band_df["lift_std"].replace(np.nan, 0).values
        
        rects2 = ax2.bar(x + width/2, lifts, width, yerr=lift_err, 
                         label='Lift over Random', color=colors["Lift over Random"],
                         capsize=3)

        ax.set_title(f"{band} Residual Screening")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel('Capture Rate')
        ax2.set_ylabel('Lift over Random')

        # Limits and formatting
        ax.set_ylim(0, 1.05)
        
        # Optional: Add chance level for capture rate
        ax.axhline(y=0.10, color='gray', linestyle='--', linewidth=1, zorder=0, label='Chance (10%)')
        
        # Legend (combine from both axes)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    out_file = out_dir / "baseline_comparison.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_file}")

if __name__ == "__main__":
    main()
