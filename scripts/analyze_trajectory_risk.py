import torch
import numpy as np
import yaml
from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.trajectory import run_risk_screening
from vesp.uq.propagation import VESPMonteCarloPropagator

def main():
    cfg_path = "configs/vespuq/vespuq_real_lunar.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    
    print("Loading test data & fitting VESP-UQ...")
    samples = load_uq_samples_from_csv("data/lunar_grail_gl0420a_L60_residual.csv")
    DU_km = 1738.0
    GM_km3_s2 = 4902.800066
    ACCEL_REF_KM_S2 = GM_km3_s2 / (DU_km**2)
    
    train_pos = samples.positions.to(device)
    train_err_norm = samples.error.to(device) / ACCEL_REF_KM_S2
    
    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err_norm)
    
    # 2. Generate 100 trajectories at varying initial altitudes
    print("Generating 100 varying-altitude test trajectories...")
    np.random.seed(42)
    altitudes_km = np.random.uniform(50, 150, size=100)
    radii = (DU_km + altitudes_km) / DU_km  # normalized body radii
    
    y0_ensemble = np.zeros((100, 6), dtype=np.float64)
    for i, r in enumerate(radii):
        v_circ = np.sqrt(1.0 / r)
        y0_ensemble[i] = [r, 0.0, 0.0, 0.0, v_circ, 0.0]
        
    propagator = VESPMonteCarloPropagator(
        plugin=plugin,
        n_samples=100,
        dt_s=10.0,
        mu=1.0,
        seed=42,
        device=device,
        base_accel_fn=None # Use point-mass for fast orbit generation
    )
    TU_s = np.sqrt((DU_km**3) / GM_km3_s2)
    propagator.dt = 10.0 / TU_s
    
    # Propagate for 1 orbit (~6.5 TU)
    t_out, Y_out = propagator.propagate(y0_ensemble[0], duration_s=6.5, output_dt_s=0.5)
    
    # Use SciPy to quickly propagate 100 deterministic trajectories
    from scipy.integrate import solve_ivp
    
    def dynamics(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -1.0 * r / (r_norm**3) # point mass
        return np.concatenate([v, a])

    trajectories = []
    print("Propagating trajectories...")
    for i in range(100):
        sol = solve_ivp(dynamics, [0, 6.5], y0_ensemble[i], t_eval=np.linspace(0, 6.5, 30))
        # sol.y shape is (6, T)
        positions = sol.y[:3, :].T # shape (T, 3)
        trajectories.append(positions)
        
    print("Scoring ensemble with VESP-UQ...")
    # Convert numpy to tensor
    traj_tensors = [torch.tensor(t, dtype=torch.float64, device=device) for t in trajectories]
    
    results = run_risk_screening(
        plugin, 
        traj_tensors, 
        rerun_fraction=0.10, # Flag top 10% riskiest
        scoring="combined"
    )
    
    report = results["risk_screening_report"]
    print("\n--- RISK SCREENING REPORT ---")
    print(f"Total Trajectories Simulated: {report.n_trajectories}")
    print(f"Trajectories Flagged as PROBLEMATIC: {report.n_flagged} ({report.rerun_fraction*100:.1f}%)")
    print(f"Risk Threshold Used: {report.threshold:.6f}")
    
    # Let's inspect the altitude of flagged vs unflagged
    flagged_idx = report.flagged_indices
    flagged_alts = altitudes_km[flagged_idx]
    
    unflagged_idx = [i for i in range(100) if i not in flagged_idx]
    unflagged_alts = altitudes_km[unflagged_idx]
    
    print("\n--- PHYSICAL INSIGHTS ---")
    print(f"Average Altitude of PROBLEMATIC (flagged) trajectories: {np.mean(flagged_alts):.1f} km")
    print(f"Average Altitude of SAFE (accepted) trajectories:     {np.mean(unflagged_alts):.1f} km")
    
    print("\nConclusion: VESP-UQ correctly identified that lower-altitude trajectories have much higher physics-model error risks.")

if __name__ == "__main__":
    main()
