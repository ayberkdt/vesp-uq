"""GPU (CUDA) parity and verification tests for the VESP-UQ layer.

Validates that fit, predict, and scoring operations on CUDA devices produce
results equivalent to the CPU baseline, up to documented float32/float64 tolerances.
"""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq.plugin import VESPUQPlugin


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    # Always generate on CPU to ensure bitwise-identical inputs across device tests
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=dtype, device="cpu")
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=dtype, device="cpu")).unsqueeze(-1)
    return dirs * radii


def _build_dataset(device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Builds a small synthetic training set. Base data is generated on CPU."""
    sources_cpu = make_shell_sources([0.75, 0.9], [24, 32], dtype=dtype, device="cpu")
    sigma_true = 0.02 * torch.randn(
        sources_cpu.n_sources, generator=torch.Generator(device="cpu").manual_seed(3), dtype=dtype, device="cpu"
    )
    positions_cpu = _query_shell(300, 1.05, 1.6, seed=1, dtype=dtype)
    A = build_acceleration_operator(positions_cpu, sources_cpu, eps=0.0, sign=1.0)
    error_cpu = (A @ sigma_true).reshape(3, positions_cpu.shape[0]).transpose(0, 1)

    sources = make_shell_sources([0.75, 0.9], [24, 32], dtype=dtype, device=device)
    positions = positions_cpu.to(device)
    error = error_cpu.to(device)
    return sources, positions, error


@pytest.fixture
def base_config():
    return {
        "reg_method": "fixed",
        "lambda_l2": 1.0e-8,
        "noise_model": "heteroscedastic",
        "val_fraction": 0.25,
        "risk_scoring": "supervisor_rel",
        "domain_support": True,
        "seed": 0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_fit_parity_float64(base_config):
    """Verify that fitting on CUDA float64 matches CPU float64 exactly."""
    sources_cpu, pos_cpu, err_cpu = _build_dataset(device="cpu", dtype=torch.float64)
    plugin_cpu = VESPUQPlugin(sources_cpu, **base_config)
    plugin_cpu.fit_error(pos_cpu, err_cpu)

    sources_gpu, pos_gpu, err_gpu = _build_dataset(device="cuda", dtype=torch.float64)
    plugin_gpu = VESPUQPlugin(sources_gpu, **base_config)
    plugin_gpu.fit_error(pos_gpu, err_gpu)

    # Check prediction parameters instead of internals
    test_pos_cpu = _query_shell(50, 1.1, 1.5, seed=42, dtype=torch.float64)
    test_pos_gpu = test_pos_cpu.to("cuda")

    out_cpu = plugin_cpu.predict_uncertainty(test_pos_cpu)
    out_gpu = plugin_gpu.predict_uncertainty(test_pos_gpu)

    for name in ("mean_error", "sigma", "epistemic_sigma", "expected_error", "risk_score"):
        torch.testing.assert_close(getattr(out_gpu, name).cpu(), getattr(out_cpu, name), rtol=1.0e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_score_ensemble_parity_float64(base_config):
    """Verify that `score_ensemble` perfectly matches on float64."""
    sources_cpu, pos_cpu, err_cpu = _build_dataset(device="cpu", dtype=torch.float64)
    plugin_cpu = VESPUQPlugin(sources_cpu, **base_config)
    plugin_cpu.fit_error(pos_cpu, err_cpu)

    sources_gpu, pos_gpu, err_gpu = _build_dataset(device="cuda", dtype=torch.float64)
    plugin_gpu = VESPUQPlugin(sources_gpu, **base_config)
    plugin_gpu.fit_error(pos_gpu, err_gpu)

    # Create dummy ensemble
    ens_cpu = []
    ens_gpu = []
    for s in range(5):
        traj = _query_shell(20, 1.05, 1.2, seed=100 + s, dtype=torch.float64)
        ens_cpu.append(traj)
        ens_gpu.append(traj.to("cuda"))

    scores_cpu = plugin_cpu.score_ensemble(ens_cpu)
    scores_gpu = plugin_gpu.score_ensemble(ens_gpu)

    for sc, sg in zip(scores_cpu, scores_gpu, strict=True):
        assert sg.risk_score == pytest.approx(sc.risk_score, rel=1e-12)



def _rel_l2(a32: torch.Tensor, a64: torch.Tensor) -> float:
    diff = a32.cpu().to(torch.float64) - a64
    return float(torch.linalg.norm(diff) / torch.linalg.norm(a64).clamp_min(1e-300))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_float32_scoring_proxy_contract(base_config):
    """The SUPPORTED float32 path: fit in float64, CAST the fitted posterior to float32 for
    GPU screening (exactly what `scripts/benchmark_gpu.py` measures). The proxy contract:
    predictions stay within a few percent and the risk RANKING agrees with float64.

    Fitting directly in float32 is NOT a supported contract -- the Gram solve at small
    lambda is too ill-conditioned for float32 (measured ~O(1) deviations) -- which is exactly
    why the headline policy keeps all fitting/calibration in float64.
    """

    sources_cpu, pos_cpu, err_cpu = _build_dataset(device="cpu", dtype=torch.float64)
    plugin_cpu = VESPUQPlugin(sources_cpu, **base_config)
    plugin_cpu.fit_error(pos_cpu, err_cpu)

    state_f32 = plugin_cpu.state_dict()
    state_f32["options"]["dtype"] = "float32"
    plugin_gpu = VESPUQPlugin.from_state_dict(state_f32, device="cuda")

    test_pos_cpu = _query_shell(50, 1.1, 1.5, seed=42, dtype=torch.float64)
    out_cpu = plugin_cpu.predict_uncertainty(test_pos_cpu)
    out_gpu = plugin_gpu.predict_uncertainty(test_pos_cpu.to("cuda", dtype=torch.float32))

    assert _rel_l2(out_gpu.mean_error, out_cpu.mean_error) < 5.0e-2
    assert _rel_l2(out_gpu.sigma, out_cpu.sigma) < 5.0e-2
    # ranking agreement is the use case float32 is allowed for (bulk screening / prioritization)
    order_cpu = torch.argsort(out_cpu.risk_score)
    order_gpu = torch.argsort(out_gpu.risk_score.cpu().to(torch.float64))
    agreement = float((order_cpu == order_gpu).float().mean())
    assert agreement > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_float32_fit_runs_but_carries_no_parity_claim(base_config):
    """Fitting directly in float32 on GPU must at least run and produce finite, positive
    uncertainties -- but parity with float64 is deliberately NOT asserted (unsupported path;
    the Gram matrix at lambda ~ 1e-8 exceeds float32 conditioning)."""

    sources_gpu, pos_gpu, err_gpu = _build_dataset(device="cuda", dtype=torch.float32)
    plugin_gpu = VESPUQPlugin(sources_gpu, **base_config)
    plugin_gpu.fit_error(pos_gpu, err_gpu)

    test_pos = _query_shell(50, 1.1, 1.5, seed=42, dtype=torch.float64).to("cuda", dtype=torch.float32)
    out = plugin_gpu.predict_uncertainty(test_pos)
    assert bool(torch.isfinite(out.mean_error).all())
    assert bool(torch.isfinite(out.sigma).all())
    assert bool((out.sigma > 0).all())
