"""Prediction-level analysis for discrete VESP experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import textwrap
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .analysis import load_checkpoint_summary, make_markdown_report
from .data import ResidualGravityDataset
from .models import load_checkpoint
from .train_discrete import make_data


PDF_SCHEMA_VERSION = "vesp_analysis_pdf_v1"


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


def _metric(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def _summary_rows(checkpoint_paths: Iterable[str | Path]) -> list[list[str]]:
    rows = []
    for path in checkpoint_paths:
        summary = load_checkpoint_summary(path)
        config = summary.get("config", {})
        metrics = summary.get("metrics", {})
        diagnostics = metrics.get("diagnostics") or {}
        model = config.get("model", {}) if isinstance(config, dict) else {}
        data = config.get("data", {}) if isinstance(config, dict) else {}
        loss = config.get("loss", {}) if isinstance(config, dict) else {}
        shell_text = ", ".join(f"{v:.2f}" for v in summary.get("shell_radii", ())) or "-"
        data_path = data.get("path")
        rows.append(
            [
                summary["name"],
                f"{model.get('type', '-')} [{shell_text}]",
                Path(str(data_path)).name if data_path else str(data.get("type", "synthetic")),
                "on" if bool(loss.get("normalize_targets", False)) else "off",
                _metric(metrics, "acceleration_rmse"),
                _metric(metrics, "relative_acceleration_rmse"),
                _metric(metrics, "angle_deg_p95"),
                _metric(diagnostics, "top_5pct_source_contribution"),
            ]
        )
    return rows


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("#f7fafb")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", color="#14212b", va="top")
    y = 0.88
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=105) or [""]
        for chunk in wrapped:
            ax.text(0.0, y, chunk, fontsize=10.5, color="#26333d", va="top")
            y -= 0.04
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(11.0, 8.5))
                fig.patch.set_facecolor("#f7fafb")
                ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
                ax.axis("off")
                y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf: PdfPages, title: str, columns: list[str], rows: list[list[str]]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("#f7fafb")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.86])
    ax.axis("off")
    ax.text(0.0, 1.04, title, fontsize=18, fontweight="bold", color="#14212b", transform=ax.transAxes)
    table = ax.table(
        cellText=rows or [["-", "-", "-", "-", "-", "-", "-", "-"]],
        colLabels=columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dfe8ec")
            cell.set_text_props(weight="bold", color="#1b2a34")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f1f5f7")
        cell.set_edgecolor("#cfd9df")
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("#f7fafb")
    ax_title = fig.add_axes([0.06, 0.92, 0.88, 0.06])
    ax_title.axis("off")
    ax_title.text(0.0, 0.55, image_path.stem.replace("_", " "), fontsize=16, fontweight="bold", color="#14212b")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    pdf.savefig(fig)
    plt.close(fig)


def write_analysis_pdf(
    checkpoint_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    markdown: str | None = None,
    device: str | torch.device = "cpu",
    max_points: int | None = None,
    include_deep: bool = False,
) -> Path:
    """Write a compact PDF review bundle for selected checkpoints.

    When ``include_deep`` is true, prediction-level plots and CSVs are generated
    before the PDF is assembled. Otherwise any existing PNG files in
    ``output_dir`` are included.
    """

    paths = [Path(path) for path in checkpoint_paths]
    output = Path(output_path)
    assets_dir = Path(output_dir) if output_dir is not None else output.with_suffix("").parent / (output.stem + "_assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    report_text = markdown
    if include_deep:
        report_text = make_advanced_report(paths, output_dir=assets_dir, device=device, max_points=max_points)
    elif report_text is None:
        report_text = make_markdown_report(paths)

    figure_paths = sorted(assets_dir.glob("*.png"))
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with PdfPages(output) as pdf:
        _add_text_page(
            pdf,
            "VESP Analysis Review",
            [
                f"Schema: {PDF_SCHEMA_VERSION}",
                f"Created: {created}",
                f"Checkpoints: {len(paths)}",
                "Scope: deterministic Stage 1-2 equivalent-source feasibility review.",
                "This PDF summarizes run metrics, target scaling state, source concentration diagnostics, and generated plots.",
            ],
        )
        _add_table_page(
            pdf,
            "Run Summary",
            ["Run", "Model", "Data", "Target Norm", "Acc RMSE", "Rel Acc", "Angle p95", "Top 5%"],
            _summary_rows(paths),
        )
        notes = [line.strip() for line in (report_text or "").splitlines() if line.strip()]
        _add_text_page(pdf, "Interpretation Notes", notes[:80])
        for figure_path in figure_paths:
            _add_image_page(pdf, figure_path)
        metadata = pdf.infodict()
        metadata["Title"] = "VESP Analysis Review"
        metadata["Author"] = "MaxEnt-VESP Workbench"
        metadata["Subject"] = "Stage 1-2 feasibility diagnostics"
        metadata["Keywords"] = "VESP, equivalent source, target scaling, residual gravity"
    return output


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
    parser.add_argument("--pdf-output", default=None)
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
    if args.pdf_output:
        pdf = write_analysis_pdf(
            args.checkpoints,
            args.pdf_output,
            output_dir=args.assets_dir,
            markdown=output.read_text(encoding="utf-8"),
            device=args.device,
            max_points=args.max_points,
        )
        print(f"analysis_pdf: {pdf}")


if __name__ == "__main__":
    main()
