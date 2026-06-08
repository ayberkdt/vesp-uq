#!/usr/bin/env python3
"""Run Phase 4: Operational Orbit Uncertainty Propagation."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import yaml

from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.propagation import VESPMonteCarloPropagator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vespuq/vespuq_real_lunar.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    
    print("1. Loading Data and Fitting VESP-UQ...")
    samples = load_uq_samples_from_csv(
        "data/lunar_grail_gl0420a_L60_residual.csv"
    )
    
    n_train = min(50000, samples.n)
    indices = np.random.RandomState(42).permutation(samples.n)[:n_train]
    
    train_pos = samples.positions[indices].to(device)
    train_err = samples.error[indices].to(device)
    
    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err)
    print("Fit complete.\n")

    print("2. Initializing Monte Carlo Propagator...")
    # Low circular orbit at 100km altitude (1.057 R_moon)
    r_initial = 1.057
    v_circular = np.sqrt(1.0 / r_initial)  # For mu = 1.0
    
    y0 = np.array([r_initial, 0.0, 0.0, 0.0, v_circular, 0.0], dtype=np.float64)
    
    propagator = VESPMonteCarloPropagator(
        plugin=plugin,
        n_samples=500,        # 500 parallel Monte Carlo trajectories
        dt_s=10.0,           # 10 second integration step
        mu=1.0,
        seed=42,
        device=device,
        dtype=torch.float64
    )
    
    print("3. Running Propagation...")
    # Orbital period is roughly 2*pi*sqrt(a^3 / mu) -> 2*pi*(1.057)^1.5 = 6.82 time units.
    # We propagate for 2 orbits (approx 14 time units)
    duration = 14.0
    output_dt = 0.5
    
    t_out, Y_out = propagator.propagate(y0, duration_s=duration, output_dt_s=output_dt)
    
    print("\n4. Results:")
    print(f"Time steps output: {len(t_out)}")
    print(f"State tensor shape: {Y_out.shape} (T, N, 6)")
    
    # Calculate final positional dispersion
    final_positions = Y_out[-1, :, :3]  # [N, 3]
    mean_pos = np.mean(final_positions, axis=0)
    std_pos = np.std(final_positions, axis=0)
    
    print("\nFinal Position Dispersion (Monte Carlo):")
    print(f"Mean Position: {mean_pos}")
    print(f"Std Deviation (X, Y, Z): {std_pos}")
    print(f"3D RMS Dispersion: {np.linalg.norm(std_pos):.6f} body radii")

if __name__ == "__main__":
    main()
