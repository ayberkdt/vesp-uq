import pandas as pd
import numpy as np
import torch
import yaml
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.trajectory import select_reruns, run_risk_screening
from scipy.integrate import solve_ivp

def kepler_to_cartesian(a_norm, e, inc_deg, raan_deg, argp_deg, ta_deg, mu=1.0):
    ta = np.radians(ta_deg)
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    
    r = a_norm * (1 - e**2) / (1 + e * np.cos(ta))
    
    x_orb = r * np.cos(ta)
    y_orb = r * np.sin(ta)
    
    p = a_norm * (1 - e**2)
    h = np.sqrt(mu * p)
    vx_orb = (mu / h) * -np.sin(ta)
    vy_orb = (mu / h) * (e + np.cos(ta))
    
    R3_W = np.array([[np.cos(raan), -np.sin(raan), 0],
                     [np.sin(raan), np.cos(raan), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(inc), -np.sin(inc)],
                     [0, np.sin(inc), np.cos(inc)]])
    R3_w = np.array([[np.cos(argp), -np.sin(argp), 0],
                     [np.sin(argp), np.cos(argp), 0],
                     [0, 0, 1]])
                     
    Q = R3_W @ R1_i @ R3_w
    r_vec = Q @ np.array([x_orb, y_orb, 0])
    v_vec = Q @ np.array([vx_orb, vy_orb, 0])
    return r_vec, v_vec

def dynamics(t, y):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -1.0 * r / (r_norm**3)
    return np.concatenate([v, a])

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
    TU_s = np.sqrt((DU_km**3) / GM_km3_s2)
    
    train_pos = samples.positions.to(device)
    train_err_norm = samples.error.to(device) / ACCEL_REF_KM_S2
    
    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err_norm)

    print("Loading 512 LUNAR test scenarios...")
    scenarios = pd.read_csv("data/test_512/scenarios.csv")
    metrics = pd.read_csv("data/test_512/metrics/gpu_batch_per_scenario_metrics.csv")
    
    st_lrps_metrics = metrics[metrics['model'] == 'ST_LRPS_DT60'].copy()
    if len(st_lrps_metrics) == 0:
        model_name = metrics['model'].iloc[0]
        st_lrps_metrics = metrics[metrics['model'] == model_name].copy()

    st_lrps_metrics = st_lrps_metrics.sort_values('scenario_id').reset_index(drop=True)
    scenarios = scenarios.sort_values('scenario_id').reset_index(drop=True)
    
    risk_scores = []
    true_errors = []
    
    print("Integrating 512 accurate point-mass trajectories...")
    # half a day is 12 hours = 43200 s
    duration_tu = 43200.0 / TU_s
    t_eval = np.linspace(0, duration_tu, 60) # evaluate 60 points along orbit
    
    traj_tensors = []
    for i in range(len(scenarios)):
        a_km = scenarios.loc[i, 'a_km']
        a_norm = a_km / DU_km
        e = scenarios.loc[i, 'e']
        inc_deg = scenarios.loc[i, 'inc_deg']
        raan_deg = scenarios.loc[i, 'raan_deg']
        argp_deg = scenarios.loc[i, 'argp_deg']
        ta_deg = scenarios.loc[i, 'ta_deg']
        
        r0, v0 = kepler_to_cartesian(a_norm, e, inc_deg, raan_deg, argp_deg, ta_deg)
        y0 = np.concatenate([r0, v0])
        
        sol = solve_ivp(dynamics, [0, duration_tu], y0, t_eval=t_eval)
        positions = sol.y[:3, :].T # shape (60, 3)
        pos_tensor = torch.tensor(positions, dtype=torch.float64, device=device)
        traj_tensors.append(pos_tensor)
        true_errors.append(st_lrps_metrics.loc[i, 'rms_pos_err_km'])
        
    print("Scoring the 512 accurate trajectories...")
    results = run_risk_screening(plugin, traj_tensors, rerun_fraction=0.10, scoring="combined", true_error=true_errors)
    report = results["risk_screening_report"]
    
    print("\n--- 512 LUNAR SCENARIOS RISK SCREENING REPORT ---")
    print(f"Total Trajectories: {len(scenarios)}")
    print(f"Spearman Rank Correlation (Risk vs True Error): {report.spearman_risk_vs_error:.4f}")
    print(f"Capture Rate (Top 10% Risk catching Top 10% Error): {report.capture_rate*100:.1f}%")
    print(f"Precision: {report.precision*100:.1f}%")
    
    print(f"\nMean True Error of Flagged (Top 10% Riskiest): {report.mean_error_flagged:.3f} km")
    print(f"Mean True Error of Accepted (Remaining 90%):   {report.mean_error_accepted:.3f} km")
    print(f"Ratio (Flagged Error / Accepted Error): {report.error_ratio_flagged_to_accepted:.2f}x")

if __name__ == '__main__':
    main()
