#!/usr/bin/env python3
"""
Phase 3/4: ST-LRPS + VESP-UQ Orbit Propagation

Loads an ST-LRPS force model from a previous LUNAR_SIMULATION run,
calibrates VESP-UQ on a residual dataset, and runs Monte Carlo propagation
combining both models.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import yaml

from vesp.adapters.st_lrps.runtime.force_model import load_surrogate_force_model
from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.propagation import VESPMonteCarloPropagator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vespuq/vespuq_real_lunar.yaml")
    parser.add_argument("--model_dir", type=str, default="C:/Users/ayber/Desktop/LUNAR_SIMULATION/outputs/training/100_1000km_ilk_deneme", help="Path to ST-LRPS run directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    
    print("1. Loading ST-LRPS Surrogate Model...")
    try:
        fm = load_surrogate_force_model(
            model_dir=args.model_dir,
            device="cpu", # Keep inference on CPU for now, or match device
            strict_contract=False,
            allow_legacy_contract=True,
            strict_domain=False
        )
        print(f"   Loaded ST-LRPS (mu_si = {fm.mu_si}, degree_min = {fm.degree_min})")
    except Exception as e:
        print(f"   Failed to load ST-LRPS from {args.model_dir}: {e}")
        print("   Proceeding without ST-LRPS (falling back to point mass).")
        fm = None

    print("\n2. Loading Data and Fitting VESP-UQ...")
    samples = load_uq_samples_from_csv(
        "data/lunar_grail_gl0420a_L60_residual.csv"
    )
    
    n_train = min(10000, samples.n)
    indices = np.random.RandomState(42).permutation(samples.n)[:n_train]
    
    # Position is in normalized body radii
    train_pos = samples.positions[indices].to(device)
    
    # Raw error in csv is km/s^2. 
    # Canonical Lunar Units: DU = 1738 km, TU = sqrt(DU^3 / GM).
    # GM_moon = 4902.800066 km^3/s^2.
    DU_km = 1738.0
    GM_km3_s2 = 4902.800066
    ACCEL_REF_KM_S2 = GM_km3_s2 / (DU_km**2)
    
    # Convert error to normalized units
    train_err_km_s2 = samples.error[indices].to(device)
    train_err_norm = train_err_km_s2 / ACCEL_REF_KM_S2
    
    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err_norm)
    print("   VESP-UQ Fit complete.\n")

    print("3. Initializing Monte Carlo Propagator...")
    
    # If ST-LRPS is loaded, define the wrapper
    if fm is not None:
        DU_m = 1738000.0
        ACCEL_REF_M_S2 = fm.mu_si / (DU_m**2)

        def st_lrps_base_accel(r_norm: torch.Tensor) -> torch.Tensor:
            # r_tu shape: (N, 3) or (3,)
            r_tu = r_norm.detach().cpu().numpy()
            # Convert r from CLU to meters
            r_m = r_tu * DU_m
            
            # Evaluate ST-LRPS model in meters to get residual acceleration
            da_m = fm.predict_residual_accel_fixed(r_m)

            # Point-mass baseline
            r_norm_val = np.linalg.norm(r_m, axis=-1, keepdims=True)
            r_norm_val = np.maximum(r_norm_val, 1.0)
            a_pm_m = -fm.mu_si * r_m / (r_norm_val ** 3)
            
            # Total acceleration
            a_m = a_pm_m + da_m
            
            # Convert acceleration from m/s^2 back to CLU
            a_tu = a_m / ACCEL_REF_M_S2
            return torch.tensor(a_tu, dtype=r_norm.dtype, device=r_norm.device)
    else:
        st_lrps_base_accel = None

    # Low circular orbit at ~100km altitude (1.057 R_moon)
    r_initial = 1.057
    v_circular = np.sqrt(1.0 / r_initial)  # For mu = 1.0
    y0 = np.array([r_initial, 0.0, 0.0, 0.0, v_circular, 0.0], dtype=np.float64)
    
    propagator = VESPMonteCarloPropagator(
        plugin=plugin,
        n_samples=500,        # parallel Monte Carlo trajectories
        dt_s=10.0,           # not 10 seconds anymore! dt is in TU.
        mu=1.0,
        seed=42,
        device=device,
        dtype=torch.float64,
        base_accel_fn=st_lrps_base_accel
    )
    
    # TU = 1034.33 s. So dt_s=10 means dt_norm = 10 / 1034.33
    TU_s = np.sqrt((DU_km**3) / GM_km3_s2)
    propagator.dt = 10.0 / TU_s
    
    print("4. Running Propagation...")
    # Orbital period is roughly 2*pi*sqrt(a^3 / mu) -> 2*pi*(1.057)^1.5 = 6.82 time units.
    # We propagate for 2 orbits (approx 14 time units)
    duration_tu = 14.0
    output_dt_tu = 0.5
    
    t_out, Y_out = propagator.propagate(y0, duration_s=duration_tu, output_dt_s=output_dt_tu)
    
    print("\n5. Results:")
    print(f"   Time steps output: {len(t_out)}")
    print(f"   State tensor shape: {Y_out.shape} (T, N, 6)")
    
    # Calculate final positional dispersion
    final_positions = Y_out[-1, :, :3]  # [N, 3]
    mean_pos = np.mean(final_positions, axis=0)
    std_pos = np.std(final_positions, axis=0)
    
    print("\nFinal Position Dispersion (Monte Carlo):")
    print(f"   Mean Position: {mean_pos}")
    print(f"   Std Deviation (X, Y, Z): {std_pos}")
    print(f"   3D RMS Dispersion: {np.linalg.norm(std_pos):.6e} body radii")
    print(f"   3D RMS Dispersion: {np.linalg.norm(std_pos) * DU_km * 1000:.2f} meters")

if __name__ == "__main__":
    main()
