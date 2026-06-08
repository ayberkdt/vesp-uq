import torch
import numpy as np
import yaml
from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.plugin import VESPUQPlugin

def main():
    cfg_path = "configs/vespuq/vespuq_real_lunar.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    print("Loading test data...")
    samples = load_uq_samples_from_csv("data/lunar_grail_gl0420a_L60_residual.csv")

    DU_km = 1738.0
    GM_km3_s2 = 4902.800066
    ACCEL_REF_KM_S2 = GM_km3_s2 / (DU_km**2)

    n_total = samples.n
    print(f"Total samples available: {n_total}")
    
    indices = np.random.RandomState(42).permutation(n_total)
    n_train = int(n_total * 0.8)  # 80% train, 20% test
    if n_train > 10000:
        n_train = 10000
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pos = samples.positions[train_idx].to(device)
    train_err_km_s2 = samples.error[train_idx].to(device)
    train_err_norm = train_err_km_s2 / ACCEL_REF_KM_S2

    test_pos = samples.positions[test_idx].to(device)
    test_err_km_s2 = samples.error[test_idx].to(device)
    test_err_norm = test_err_km_s2 / ACCEL_REF_KM_S2

    print("Fitting VESP-UQ...")
    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err_norm)

    print("Evaluating over test set...")
    # Get true error magnitude
    true_err_mag = torch.norm(test_err_norm, dim=-1).cpu().numpy()

    # Get VESP-UQ predicted covariance
    cov_pred = plugin.predict_covariance_3x3(test_pos)
    cov_Q = cov_pred.covariance # shape [N, 3, 3]
    pred_variance = torch.diagonal(cov_Q, dim1=-2, dim2=-1).sum(dim=-1).cpu().numpy() # Trace
    pred_std = np.sqrt(pred_variance)

    # Calculate correlation
    correlation = np.corrcoef(true_err_mag, pred_std)[0, 1]
    print(f"\nPearson Correlation between Actual Error and VESP-UQ Predicted STD: {correlation:.4f}")

    # Top 1% highest uncertainty analysis
    sorted_idx = np.argsort(pred_std)[::-1]
    top_1_percent = sorted_idx[:int(len(sorted_idx) * 0.01)]
    bottom_99_percent = sorted_idx[int(len(sorted_idx) * 0.01):]

    mean_err_top1 = np.mean(true_err_mag[top_1_percent])
    mean_err_bottom99 = np.mean(true_err_mag[bottom_99_percent])

    print(f"\nMean Actual Error of Top 1% Most Uncertain Points: {mean_err_top1:.6e}")
    print(f"Mean Actual Error of Remaining 99% Points:         {mean_err_bottom99:.6e}")
    print(f"Ratio (Top 1% / Rest): {mean_err_top1 / mean_err_bottom99:.2f}x")

if __name__ == '__main__':
    main()
