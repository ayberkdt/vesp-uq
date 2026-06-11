import json
import subprocess
import sys

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq.compare import compare_models
from vesp.uq.plugin import VESPUQPlugin


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    pos = torch.randn(50, 3)
    pos = pos / torch.linalg.norm(pos, dim=-1, keepdim=True) * 1.5
    err = torch.randn(50, 3) * 1e-4

    val_pos = torch.randn(20, 3)
    val_pos = val_pos / torch.linalg.norm(val_pos, dim=-1, keepdim=True) * 1.5
    val_err = torch.randn(20, 3) * 1e-4

    return pos, err, val_pos, val_err


@pytest.fixture
def dummy_plugin(dummy_data):
    pos, err, val_pos, val_err = dummy_data
    sources = make_shell_sources([0.86], 64)
    plugin = VESPUQPlugin(sources, domain_support=True)
    plugin.fit_error(pos, err, val_positions=val_pos, val_error=val_err)
    return plugin


@pytest.fixture
def dummy_ensemble():
    return [
        torch.tensor([[1.1, 0, 0], [1.2, 0, 0]]),
        torch.tensor([[0, 1.5, 0], [0, 1.6, 0]]),
        torch.tensor([[0, 0, 1.1], [0, 0, 1.2]]),
        torch.tensor([[1.1, 1.1, 0], [1.2, 1.2, 0]]),
        torch.tensor([[-1.1, 0, 0], [-1.2, 0, 0]]),
    ]


def test_identity_comparison(dummy_plugin, dummy_data, dummy_ensemble):
    _, _, val_pos, val_err = dummy_data

    report = compare_models(
        dummy_plugin, dummy_plugin, val_pos, val_err, trajectory_ensemble=dummy_ensemble
    )

    assert report["posterior_distance"]["mean_l2_diff"] == 0.0
    assert report["posterior_distance"]["cov_frob_diff"] == 0.0
    assert report["posterior_distance"]["noise_var_delta"] == 0.0

    assert report["domain_shift"]["mean_score_on_A"] < 1.0
    assert report["domain_shift"]["max_score_on_A"] < 1.0

    assert report["screening_agreement"]["risk_spearman"] == pytest.approx(1.0)
    assert report["screening_agreement"]["flag_overlap"] == pytest.approx(1.0)
    assert report["screening_agreement"]["n_flagged_A"] == report["screening_agreement"]["n_flagged_B"]


def test_updated_model_comparison(dummy_plugin, dummy_data, dummy_ensemble, tmp_path):
    # Save the original plugin to copy it
    orig_path = tmp_path / "orig.pt"
    dummy_plugin.save(orig_path)

    plugin_b = VESPUQPlugin.load(orig_path)
    # Perform an update
    pos_new = torch.tensor([[1.5, 0.0, 0.0]])
    err_new = torch.tensor([[1e-4, 0.0, 0.0]])
    plugin_b.update_error(pos_new, err_new)

    _, _, val_pos, val_err = dummy_data
    report = compare_models(
        dummy_plugin, plugin_b, val_pos, val_err, trajectory_ensemble=dummy_ensemble
    )

    # After an update, they should not be identical
    assert report["posterior_distance"]["mean_l2_diff"] > 0.0
    assert report["domain_shift"]["mean_score_on_A"] < 1.0


def test_identity_comparison_with_full_band_coverage():
    # Regression: the calibration report carries SCALAR summary keys (low_high_*_ratio) next to
    # the per-band dicts; the comparison must skip them instead of subscripting floats. This
    # needs enough held-out samples that BOTH the low and high bands exist.
    g = torch.Generator().manual_seed(11)
    dirs = torch.randn(400, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = 1.03 + 0.57 * torch.rand(400, 1, generator=g, dtype=torch.float64)
    pos = dirs * radii
    err = 1.0e-4 * torch.randn(400, 3, generator=g, dtype=torch.float64)

    sources = make_shell_sources([0.86], 48, dtype=torch.float64)
    plugin = VESPUQPlugin(sources, reg_method="fixed", lambda_l2=1.0e-6)
    plugin.fit_error(pos, err)

    report = compare_models(plugin, plugin, pos, err)
    bands = report["calibration"]
    assert "low" in bands and "high" in bands  # the ratio keys existed -> loop survived them
    assert "low_high_epistemic_std_ratio" not in bands
    for metrics in bands.values():
        assert metrics["rmse"]["A"] == metrics["rmse"]["B"]


def test_compare_models_cli(tmp_path, dummy_plugin, dummy_data):
    p_a = tmp_path / "a.pt"
    p_b = tmp_path / "b.pt"
    dummy_plugin.save(p_a)
    dummy_plugin.save(p_b)

    # Create dummy csv for data
    data_csv = tmp_path / "data.csv"
    with open(data_csv, "w") as f:
        f.write("x,y,z,ax_ref,ay_ref,az_ref,ax_sur,ay_sur,az_sur\n")
        f.write("1,0,0,0,0,0,0,0,0\n")
    with open(str(data_csv) + ".metadata.json", "w") as f:
        json.dump({
            "position_units": "normalized",
        }, f)

    # Trajectory CSV exercises the screening-agreement path of the CLI (Format A).
    traj_csv = tmp_path / "traj.csv"
    with open(traj_csv, "w") as f:
        f.write("trajectory_id,t,x,y,z\n")
        for tid in range(3):
            for k in range(4):
                f.write(f"{tid},{k},{1.1 + 0.05 * k},{0.1 * tid},0\n")

    out_dir = tmp_path / "out"
    subprocess.check_call([
        sys.executable, "-m", "scripts.compare_models",
        "--model-a", str(p_a),
        "--model-b", str(p_b),
        "--data", str(data_csv),
        "--trajectories", str(traj_csv),
        "--out", str(out_dir)
    ])

    assert (out_dir / "model_comparison.json").exists()
    assert (out_dir / "model_comparison.md").exists()
    assert (out_dir / "run_manifest.json").exists()

    with open(out_dir / "model_comparison.json") as f:
        res = json.load(f)
    assert "posterior_distance" in res
    assert res["screening_agreement"]["flag_overlap"] == 1.0  # identical models, same ensemble

    # promotion decisions must trace to exact model bytes: both artifacts checksummed as inputs
    with open(out_dir / "run_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["inputs"]["model_a"]["sha256"]
    assert manifest["inputs"]["model_b"]["sha256"]
    assert manifest["inputs"]["trajectories"]["sha256"]
