#!/usr/bin/env python3
"""N14 benchmark: GPU verification and float32 screening throughput/precision.

Times the screening `score_ensemble` hot-path across environments (CPU vs CUDA, float64 vs float32)
and calculates throughput speedups and max relative risk-score error against the standard CPU-float64 execution.

Outputs results into `outputs/gpu_benchmark/`.
"""

import argparse
import time
from pathlib import Path

import torch

from vesp.common.config import load_config
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.ensemble import generate_orbit_ensemble
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.run import run_vespuq


def build_benchmark_md(stats: dict, config_path: str) -> str:
    md = [
        "# VESP-UQ GPU/Float32 Screening Benchmark\n",
        f"- **Config:** `{config_path}`",
        f"- **Trajectory points:** {stats['n_trajectories']} orbits x {stats['n_points_per_traj']} pts = {stats['total_points']} queries",
        f"- **Sources:** {stats['n_sources']}\n",
        "## Throughput (score_ensemble execution time)\n",
        "| Environment | Total Time (s) | Throughput (µs/point) | Speedup vs CPU-float64 |",
        "| --- | --- | --- | --- |",
    ]
    base_us = stats["cpu_float64"]["us_per_point"]
    for env in ("cpu_float64", "cpu_float32", "cuda_float64", "cuda_float32"):
        if env not in stats:
            continue
        s = stats[env]
        md.append(
            f"| {env} | {s['time_s']:.4f} | {s['us_per_point']:.2f} | {base_us / s['us_per_point']:.2f}x |"
        )

    md.extend([
        "\n## Precision (Max Relative Error vs CPU-float64)\n",
        "Max relative risk-score error (`|val - baseline| / |baseline|`) across the ensemble:\n",
        "| Environment | Max Rel Error |",
        "| --- | --- |",
    ])
    for env in ("cpu_float32", "cuda_float64", "cuda_float32"):
        if env not in stats:
            continue
        err = stats[env].get("max_rel_error_to_base", 0.0)
        md.append(f"| {env} | {err:.2e} |")

    md.extend([
        "\n## Policy Statement",
        "While GPU and float32 throughputs represent significant speedups for deployment and internal exploratory screening, **the headline calibration and scientific numbers must remain float64/CPU-reproducible.** Float32 operations on inverted covariance matrices and equivalent-source operators typically accumulate $10^{-3}$ to $10^{-5}$ relative errors. This degradation is often acceptable for ranking and bulk screening but breaks strict scientific reproducibility guarantees."
    ])
    return "\n".join(md)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark GPU and float32 screening throughput.")
    parser.add_argument("--config", required=True, help="Path to VESP-UQ config (e.g. vespuq_smoke.yaml)")
    parser.add_argument("--out-dir", default="outputs/gpu_benchmark", help="Output directory")
    parser.add_argument("--orbits", type=int, default=512, help="Number of test orbits")
    parser.add_argument("--pts", type=int, default=64, help="Points per orbit")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    print(f"Fitting baseline plugin using {args.config}...")
    _, plugin_cpu_f64 = run_vespuq(config, return_plugin=True)
    n_sources = plugin_cpu_f64.sources.n_sources

    print(f"Generating benchmark ensemble ({args.orbits} orbits x {args.pts} pts)...")
    ens = generate_orbit_ensemble(n_orbits=args.orbits, n_points=args.pts, seed=42, dtype=torch.float64)
    trajectories_cpu_f64 = ens.trajectories

    total_points = args.orbits * args.pts

    def run_env(plugin, trajectories):
        # Warmup
        plugin.score_ensemble(trajectories[:2])
        # Timed execution
        t0 = time.perf_counter()
        scores = plugin.score_ensemble(trajectories)
        t1 = time.perf_counter()
        return scores, (t1 - t0)

    stats = {
        "n_trajectories": args.orbits,
        "n_points_per_traj": args.pts,
        "total_points": total_points,
        "n_sources": n_sources,
    }

    # 1. CPU Float64 (Baseline)
    print("Running CPU float64...")
    scores_base, time_base = run_env(plugin_cpu_f64, trajectories_cpu_f64)
    base_risk = torch.tensor([s.risk_score for s in scores_base], dtype=torch.float64)
    stats["cpu_float64"] = {"time_s": time_base, "us_per_point": (time_base / total_points) * 1e6}

    # 2. CPU Float32
    print("Running CPU float32...")
    state_f32 = plugin_cpu_f64.state_dict()
    state_f32["options"]["dtype"] = "float32"
    plugin_cpu_f32 = VESPUQPlugin.from_state_dict(state_f32, device="cpu")
    
    trajectories_cpu_f32 = [t.to(torch.float32) for t in trajectories_cpu_f64]
    scores_f32, time_f32 = run_env(plugin_cpu_f32, trajectories_cpu_f32)
    risk_f32 = torch.tensor([s.risk_score for s in scores_f32], dtype=torch.float64)
    err_f32 = float(torch.max(torch.abs(risk_f32 - base_risk) / torch.clamp(torch.abs(base_risk), min=1e-12)))
    stats["cpu_float32"] = {
        "time_s": time_f32,
        "us_per_point": (time_f32 / total_points) * 1e6,
        "max_rel_error_to_base": err_f32,
    }

    # 3. CUDA (if available)
    if torch.cuda.is_available():
        # CUDA Float64
        print("Running CUDA float64...")
        plugin_gpu_f64 = VESPUQPlugin.from_state_dict(plugin_cpu_f64.state_dict(), device="cuda")

        trajectories_gpu_f64 = [t.to("cuda") for t in trajectories_cpu_f64]
        scores_gf64, time_gf64 = run_env(plugin_gpu_f64, trajectories_gpu_f64)
        risk_gf64 = torch.tensor([s.risk_score for s in scores_gf64], dtype=torch.float64)
        err_gf64 = float(torch.max(torch.abs(risk_gf64 - base_risk) / torch.clamp(torch.abs(base_risk), min=1e-12)))
        stats["cuda_float64"] = {
            "time_s": time_gf64,
            "us_per_point": (time_gf64 / total_points) * 1e6,
            "max_rel_error_to_base": err_gf64,
        }

        # CUDA Float32
        print("Running CUDA float32...")
        plugin_gpu_f32 = VESPUQPlugin.from_state_dict(state_f32, device="cuda")

        trajectories_gpu_f32 = [t.to(device="cuda", dtype=torch.float32) for t in trajectories_cpu_f64]
        scores_gf32, time_gf32 = run_env(plugin_gpu_f32, trajectories_gpu_f32)
        risk_gf32 = torch.tensor([s.risk_score for s in scores_gf32], dtype=torch.float64)
        err_gf32 = float(torch.max(torch.abs(risk_gf32 - base_risk) / torch.clamp(torch.abs(base_risk), min=1e-12)))
        stats["cuda_float32"] = {
            "time_s": time_gf32,
            "us_per_point": (time_gf32 / total_points) * 1e6,
            "max_rel_error_to_base": err_gf32,
        }
    else:
        print("CUDA not available. Skipping GPU benchmarks.")

    md_report = build_benchmark_md(stats, args.config)
    print("\n" + md_report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_run_artifacts(
        out_dir=out_dir,
        tool="benchmark_gpu",
        json_files={"gpu_benchmark_report.json": stats},
        text_files={"gpu_benchmark_report.md": md_report},
        config={"orbits": args.orbits, "pts": args.pts},
    )
    print(f"\nArtifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
