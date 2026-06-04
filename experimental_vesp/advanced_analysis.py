"""Prediction-level analysis for discrete VESP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .analysis import load_checkpoint_summary, make_markdown_report
from .data import ResidualGravityDataset
from .models import load_checkpoint
from .train_discrete import make_data


def _dtype_from_config(config: dict) -> torch.dtype:
    return torch.float64 if str(config.get("dtype", "float32")) == "float64" else torch.float32


def prediction_dataframe(
    checkpoint_path: str | Path,
    *,
    batch_size: int = 4096,
    source_chunk_size: int | None = None,
    device: str | torch.device = "cpu",
    max_points: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Recompute validation predictions and return a dataframe of residual diagnostics."""

    summary = load_checkpoint_summary(checkpoint_path)
    config = summary["config"]
    dtype = _dtype_from_config(config)
    _, val_data = make_data(config, dtype=dtype)
    if max_points is not None and val_data.positions.shape[0] > max_points:
        indices = torch.arange(max_points)
        val_data = val_data.subset(indices)

    model = load_checkpoint(str(checkpoint_path), map_location=device).to(device)
    val_data = val_data.to(device)
    loader = DataLoader(ResidualGravityDataset(val_data), batch_size=batch_size, shuffle=False)

    xs, true_u, pred_u, true_a, pred_a = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u, a = model(
                x,
                source_chunk_size=source_chunk_size or config.get("kernel", {}).get("source_chunk_size"),
                softening=float(config.get("kernel", {}).get("softening", 0.0)),
            )
            if u is None or a is None:
                raise RuntimeError("model must return potential and acceleration for advanced analysis")
            xs.append(x.detach().cpu())
            true_u.append(batch["potential"].detach().cpu())
            pred_u.append(u.detach().cpu())
            true_a.append(batch["acceleration"].detach().cpu())
            pred_a.append(a.detach().cpu())

    x_t = torch.cat(xs, dim=0)
    true_u_t = torch.cat(true_u, dim=0).reshape(-1)
    pred_u_t = torch.cat(pred_u, dim=0).reshape(-1)
    true_a_t = torch.cat(true_a, dim=0)
    pred_a_t = torch.cat(pred_a, dim=0)

    err_a = pred_a_t - true_a_t
    err_u = pred_u_t - true_u_t
    radius = torch.linalg.norm(x_t, dim=-1)
    radial = x_t / torch.clamp(radius.unsqueeze(-1), min=torch.finfo(x_t.dtype).eps)
    radial_err_scalar = torch.sum(err_a * radial, dim=-1)
    radial_err = radial_err_scalar.unsqueeze(-1) * radial
    cross_err = err_a - radial_err
    true_a_norm = torch.linalg.norm(true_a_t, dim=-1)
    pred_a_norm = torch.linalg.norm(pred_a_t, dim=-1)
    err_a_norm = torch.linalg.norm(err_a, dim=-1)
    rel_acc_err = err_a_norm / torch.clamp(true_a_norm, min=torch.finfo(x_t.dtype).eps)
    dot = torch.sum(pred_a_t * true_a_t, dim=-1)
    denom = torch.clamp(pred_a_norm * true_a_norm, min=torch.finfo(x_t.dtype).eps)
    angular_error_deg = torch.rad2deg(torch.arccos(torch.clamp(dot / denom, min=-1.0, max=1.0)))

    frame = pd.DataFrame(
        {
            "x": x_t[:, 0].numpy(),
            "y": x_t[:, 1].numpy(),
            "z": x_t[:, 2].numpy(),
            "radius": radius.numpy(),
            "altitude_normalized": (radius - 1.0).numpy(),
            "potential_true": true_u_t.numpy(),
            "potential_pred": pred_u_t.numpy(),
            "potential_error": err_u.numpy(),
            "acc_true_norm": true_a_norm.numpy(),
            "acc_pred_norm": pred_a_norm.numpy(),
            "acc_error_norm": err_a_norm.numpy(),
            "relative_acc_error": rel_acc_err.numpy(),
            "radial_error_signed": radial_err_scalar.numpy(),
            "radial_error_norm": torch.linalg.norm(radial_err, dim=-1).numpy(),
            "cross_error_norm": torch.linalg.norm(cross_err, dim=-1).numpy(),
            "angular_error_deg": angular_error_deg.numpy(),
        }
    )
    return frame, summary


def source_concentration_metrics(checkpoint_path: str | Path) -> dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sigma = ckpt["sigma"].detach().double()
    weights = ckpt["source_weights"].detach().double()
    shell_ids = ckpt["shell_ids"].detach().long()
    weighted_abs = torch.abs(weights * sigma)
    total = torch.sum(weighted_abs)
    if float(total) <= 0.0:
        return {
            "effective_source_count": 0.0,
            "effective_source_fraction": 0.0,
            "top_1pct_abs_fraction": 0.0,
            "top_5pct_abs_fraction": 0.0,
            "positive_abs_fraction": 0.0,
            "negative_abs_fraction": 0.0,
            "shell_abs_fractions": [],
        }

    p = weighted_abs / total
    entropy = -torch.sum(p * torch.log(torch.clamp(p, min=torch.finfo(p.dtype).tiny)))
    effective_n = torch.exp(entropy)
    sorted_p = torch.sort(p, descending=True).values
    n = sigma.numel()
    top_1 = max(1, int(round(0.01 * n)))
    top_5 = max(1, int(round(0.05 * n)))
    positive_abs = torch.sum(weighted_abs[sigma >= 0])
    negative_abs = torch.sum(weighted_abs[sigma < 0])
    shell_fractions = []
    for shell_id in range(int(torch.max(shell_ids).item()) + 1):
        shell_fractions.append(float(torch.sum(weighted_abs[shell_ids == shell_id]) / total))

    return {
        "effective_source_count": float(effective_n),
        "effective_source_fraction": float(effective_n / n),
        "top_1pct_abs_fraction": float(torch.sum(sorted_p[:top_1])),
        "top_5pct_abs_fraction": float(torch.sum(sorted_p[:top_5])),
        "positive_abs_fraction": float(positive_abs / total),
        "negative_abs_fraction": float(negative_abs / total),
        "shell_abs_fractions": shell_fractions,
    }


def advanced_metrics(frame: pd.DataFrame, checkpoint_path: str | Path) -> dict:
    q = frame[
        [
            "acc_error_norm",
            "relative_acc_error",
            "potential_error",
            "radial_error_signed",
            "cross_error_norm",
            "angular_error_deg",
        ]
    ].quantile([0.5, 0.9, 0.95, 0.99])
    radius_corr = frame["radius"].corr(frame["acc_error_norm"])
    abs_radial_bias = float(frame["radial_error_signed"].mean())
    near = frame.nsmallest(max(1, len(frame) // 5), "radius")
    far = frame.nlargest(max(1, len(frame) // 5), "radius")
    near_far_ratio = float(near["acc_error_norm"].mean() / max(far["acc_error_norm"].mean(), 1.0e-30))

    return {
        "n_eval": int(len(frame)),
        "acc_error_p50": float(q.loc[0.5, "acc_error_norm"]),
        "acc_error_p90": float(q.loc[0.9, "acc_error_norm"]),
        "acc_error_p95": float(q.loc[0.95, "acc_error_norm"]),
        "acc_error_p99": float(q.loc[0.99, "acc_error_norm"]),
        "relative_acc_error_p50": float(q.loc[0.5, "relative_acc_error"]),
        "relative_acc_error_p95": float(q.loc[0.95, "relative_acc_error"]),
        "potential_abs_error_p95": float(frame["potential_error"].abs().quantile(0.95)),
        "angular_error_deg_p95": float(q.loc[0.95, "angular_error_deg"]),
        "radius_error_corr": 0.0 if pd.isna(radius_corr) else float(radius_corr),
        "radial_error_bias": abs_radial_bias,
        "near_far_mean_error_ratio": near_far_ratio,
        **source_concentration_metrics(checkpoint_path),
    }


def _bin_by_radius(frame: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    bins = pd.cut(frame["radius"], bins=n_bins)
    grouped = frame.groupby(bins, observed=True)
    out = grouped.agg(
        radius_mid=("radius", "mean"),
        acc_error_mean=("acc_error_norm", "mean"),
        acc_error_p90=("acc_error_norm", lambda v: v.quantile(0.9)),
        radial_abs_mean=("radial_error_signed", lambda v: v.abs().mean()),
        cross_mean=("cross_error_norm", "mean"),
        relative_p90=("relative_acc_error", lambda v: v.quantile(0.9)),
        count=("radius", "size"),
    )
    return out.reset_index(drop=True)


def plot_checkpoint_diagnostics(
    frame: pd.DataFrame,
    summary: dict,
    metrics: dict,
    *,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = summary["name"]
    paths: list[Path] = []

    binned = _bin_by_radius(frame)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.plot(binned["radius_mid"], binned["acc_error_mean"], marker="o", label="mean")
    ax.plot(binned["radius_mid"], binned["acc_error_p90"], marker="s", label="p90")
    ax.set_xlabel("normalized radius")
    ax.set_ylabel("acceleration error norm")
    ax.set_title(f"{stem}: altitude error profile")
    ax.grid(True, alpha=0.28)
    ax.legend()
    path = output / f"{stem}_altitude_error.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.hist(frame["relative_acc_error"], bins=40, color="#2f7497", alpha=0.86)
    ax.set_xlabel("relative acceleration error")
    ax.set_ylabel("count")
    ax.set_title(f"{stem}: relative error distribution")
    ax.grid(True, alpha=0.22)
    path = output / f"{stem}_relative_error_hist.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    shell_fracs = metrics.get("shell_abs_fractions") or []
    if shell_fracs:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        ax.bar([str(i) for i in range(len(shell_fracs))], shell_fracs, color="#4f8f6f")
        ax.set_xlabel("shell index")
        ax.set_ylabel("fraction of |w sigma|")
        ax.set_ylim(0.0, max(1.0, max(shell_fracs) * 1.1))
        ax.set_title(f"{stem}: source magnitude by shell")
        ax.grid(True, axis="y", alpha=0.22)
        path = output / f"{stem}_shell_abs_fraction.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_comparison(reports: list[dict], *, output_dir: str | Path) -> Path | None:
    if not reports:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = [r["summary"]["name"] for r in reports]
    acc_rmse = [r["basic"].get("metrics", {}).get("acceleration_rmse", None) for r in reports]
    if any(v is None for v in acc_rmse):
        acc_rmse = [r["metrics"]["acc_error_p95"] for r in reports]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=140)
    ax.bar(names, acc_rmse, color="#245c7a")
    ax.set_yscale("log")
    ax.set_ylabel("acceleration RMSE / proxy")
    ax.set_title("Experiment comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    path = output / "comparison_acceleration_error.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_advanced_report(
    checkpoint_paths: Iterable[str | Path],
    *,
    output_dir: str | Path = "outputs/advanced_analysis",
    device: str | torch.device = "cpu",
    max_points: int | None = None,
) -> str:
    paths = [Path(path) for path in checkpoint_paths]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report_blocks = [make_markdown_report(paths), "", "# Prediction-Level Diagnostics", ""]
    reports = []
    for path in paths:
        frame, summary = prediction_dataframe(path, device=device, max_points=max_points)
        metrics = advanced_metrics(frame, path)
        figures = plot_checkpoint_diagnostics(frame, summary, metrics, output_dir=output)
        csv_path = output / f"{summary['name']}_validation_predictions.csv"
        frame.to_csv(csv_path, index=False)
        reports.append({"summary": summary, "metrics": metrics, "figures": figures, "csv": csv_path, "basic": summary})

        report_blocks.extend(
            [
                f"## {summary['name']} Deep Dive",
                "",
                f"- Validation points analyzed: {metrics['n_eval']}",
                f"- Acceleration error percentiles p50/p90/p95/p99: "
                f"{metrics['acc_error_p50']:.3e} / {metrics['acc_error_p90']:.3e} / "
                f"{metrics['acc_error_p95']:.3e} / {metrics['acc_error_p99']:.3e}",
                f"- Relative acceleration error p50/p95: "
                f"{metrics['relative_acc_error_p50']:.3e} / {metrics['relative_acc_error_p95']:.3e}",
                f"- Angular error p95: {metrics['angular_error_deg_p95']:.3e} deg",
                f"- Radius-error correlation: {metrics['radius_error_corr']:.3e}",
                f"- Near/far mean error ratio: {metrics['near_far_mean_error_ratio']:.3e}",
                f"- Radial signed error bias: {metrics['radial_error_bias']:.3e}",
                f"- Effective source count/fraction: "
                f"{metrics['effective_source_count']:.1f} / {metrics['effective_source_fraction']:.2%}",
                f"- Top 1% / 5% |w sigma| fraction: "
                f"{metrics['top_1pct_abs_fraction']:.2%} / {metrics['top_5pct_abs_fraction']:.2%}",
                f"- Positive/negative |w sigma| split: "
                f"{metrics['positive_abs_fraction']:.2%} / {metrics['negative_abs_fraction']:.2%}",
                f"- Validation prediction CSV: `{csv_path}`",
                "",
            ]
        )
        for fig_path in figures:
            report_blocks.append(f"![{fig_path.stem}]({fig_path.as_posix()})")
        report_blocks.append("")

    comparison = plot_comparison(reports, output_dir=output)
    if comparison is not None:
        report_blocks.extend(["## Advanced Comparison", "", f"![comparison]({comparison.as_posix()})", ""])

    report_blocks.extend(
        [
            "## Interpretation Notes",
            "",
            "- A high near/far ratio means the model is much less stable close to the body, even if global RMSE looks good.",
            "- A low effective source fraction means the equivalent source map is localized; this is where MaxEnt may later help.",
            "- A high top-1% or top-5% source fraction is a warning sign for ill-conditioned equivalent-source behavior.",
            "- Radius-error correlation should be read with altitude bins: strong negative correlation often means low-altitude weakness.",
            "",
        ]
    )
    return "\n".join(report_blocks)


def write_advanced_report(
    checkpoint_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    max_points: int | None = None,
) -> Path:
    output = Path(output_path)
    figure_dir = Path(output_dir) if output_dir is not None else output.with_suffix("").parent / (output.stem + "_assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = make_advanced_report(checkpoint_paths, output_dir=figure_dir, device=device, max_points=max_points)
    output.write_text(report, encoding="utf-8")
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", default="outputs/advanced_analysis_report.md")
    parser.add_argument("--assets-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args(argv)
    output = write_advanced_report(
        args.checkpoints,
        args.output,
        output_dir=args.assets_dir,
        device=args.device,
        max_points=args.max_points,
    )
    print(f"advanced_analysis_report: {output}")


if __name__ == "__main__":
    main()

