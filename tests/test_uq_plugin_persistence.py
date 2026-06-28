"""Round-trip tests for fitted-plugin persistence (VESPUQPlugin.save / .load).

A loaded plugin must predict, score, and screen identically to the plugin that was saved --
persistence exists so operational consumers (screening, CorrectedForceField, propagators) can
reuse a calibrated layer without refitting.
"""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.extensions.probabilistic import BinnedAltitudeNoiseModel
from vesp.uq import VESPUQPlugin
from vesp.uq.correction import CorrectedForceField
from vesp.uq.plugin import PLUGIN_STATE_VERSION, PREDICTIVE_CONFORMAL_MODES


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _fitted_plugin(
    *,
    conformal_apply: bool = False,
    conformal_by_band: bool = False,
    conformal_by_region: bool = False,
    conformal_mode: str = "norm",
    noise_model: str = "heteroscedastic",
    calibrate_supervisor: bool = False,
) -> VESPUQPlugin:
    sources = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    positions = _query_shell(300, 1.05, 1.6, seed=1)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    plugin = VESPUQPlugin(
        sources,
        reg_method="fixed",
        lambda_l2=1.0e-8,
        noise_model=noise_model,
        altitude_noise_bins=5,
        val_fraction=0.25,
        risk_scoring="supervisor_rel",
        domain_support=True,
        conformal_apply=conformal_apply,
        conformal_mode=conformal_mode,
        conformal_by_band=conformal_by_band,
        conformal_by_region=conformal_by_region,
        conformal_bands={"low": [1.05, 1.325], "high": [1.325, 1.60]},
        conformal_min_band_n=5,
        conformal_min_region_n=5,
        calibrate_supervisor=calibrate_supervisor,
        seed=0,
    )
    plugin.fit_error(positions, error)
    return plugin


def test_save_load_round_trip_predicts_identically(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    assert path.exists()

    loaded = VESPUQPlugin.load(path)
    queries = _query_shell(64, 1.05, 1.6, seed=7)

    a = plugin.predict_uncertainty(queries)
    b = loaded.predict_uncertainty(queries)
    for name in ("mean_error", "sigma", "epistemic_sigma", "expected_error", "risk_score"):
        assert torch.allclose(getattr(a, name), getattr(b, name), rtol=1.0e-12, atol=0.0), name

    ca = plugin.predict_covariance_3x3(queries)
    cb = loaded.predict_covariance_3x3(queries)
    assert torch.allclose(ca.covariance, cb.covariance, rtol=1.0e-12, atol=0.0)

    # domain support state (train geometry + cached scales) must survive the round trip
    da = plugin.domain_support_score(queries)
    db = loaded.domain_support_score(queries)
    assert torch.allclose(da, db, rtol=1.0e-12, atol=0.0)

    sa = plugin.score_trajectory(queries)
    sb = loaded.score_trajectory(queries)
    assert sb.risk_score == pytest.approx(sa.risk_score, rel=1.0e-12)
    assert sb.scoring == sa.scoring

    # fit provenance and options ride along
    assert loaded.fit_info == plugin.fit_info
    assert loaded.risk_scoring == plugin.risk_scoring
    assert loaded.noise_model == plugin.noise_model
    assert loaded.altitude_noise is not None
    assert loaded.altitude_noise.log_a == pytest.approx(plugin.altitude_noise.log_a)
    assert loaded.posterior.lambda_l2 == pytest.approx(plugin.posterior.lambda_l2)


def test_conformal_enabled_without_apply_keeps_prediction_path_off():
    plugin = VESPUQPlugin.from_config({"model": {"n_source": 16}, "uq": {"conformal": {"enabled": True}}})
    assert plugin.conformal_apply is False
    assert plugin.conformal_calibration is None
    assert plugin.conformal_mode in PREDICTIVE_CONFORMAL_MODES
    assert plugin.sources.n_sources == 16


def test_operational_conformal_scales_uncertainty_not_mean():
    raw = _fitted_plugin()
    calibrated = _fitted_plugin(conformal_apply=True)
    assert calibrated.conformal_calibration is not None
    assert calibrated.conformal_calibration["enabled"] is True
    scale = calibrated.conformal_calibration["global"]["scale"]
    queries = _query_shell(64, 1.05, 1.6, seed=17)

    a = raw.predict_uncertainty(queries)
    b = calibrated.predict_uncertainty(queries)
    assert torch.allclose(b.mean_error, a.mean_error, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(b.std_components, a.std_components * scale, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(b.sigma, a.sigma * scale, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(
        b.expected_error,
        torch.sqrt((a.mean_error_magnitude * a.mean_error_magnitude + b.sigma * b.sigma).clamp_min(0.0)),
        rtol=1.0e-12,
        atol=0.0,
    )

    ca = raw.predict_covariance_3x3(queries)
    cb = calibrated.predict_covariance_3x3(queries)
    assert torch.allclose(cb.mean_error, ca.mean_error, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(cb.covariance, ca.covariance * (scale * scale), rtol=1.0e-12, atol=0.0)


def test_operational_mahalanobis_conformal_scales_covariance():
    raw = _fitted_plugin()
    calibrated = _fitted_plugin(conformal_apply=True, conformal_mode="mahalanobis")
    assert calibrated.conformal_calibration["mode"] == "mahalanobis"
    scale = calibrated.conformal_calibration["global"]["scale"]
    queries = _query_shell(32, 1.05, 1.6, seed=23)

    ca = raw.predict_covariance_3x3(queries)
    cb = calibrated.predict_covariance_3x3(queries)
    assert torch.allclose(cb.mean_error, ca.mean_error, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(cb.covariance, ca.covariance * (scale * scale), rtol=1.0e-12, atol=0.0)


def test_conformal_per_band_uses_band_scales_with_global_fallback():
    plugin = _fitted_plugin(conformal_apply=True, conformal_by_band=True)
    cal = plugin.conformal_calibration
    assert cal["scope"] == "per_band"
    used = [b for b in cal["bands"] if b.get("used")]
    assert used

    radii = torch.tensor([1.10, 1.45, 2.00], dtype=torch.float64)
    scales = plugin._conformal_scale_for_radius(radii)
    band_lookup = {b["name"]: b["scale"] for b in used}
    if "low" in band_lookup:
        assert scales[0].item() == pytest.approx(band_lookup["low"])
    if "high" in band_lookup:
        assert scales[1].item() == pytest.approx(band_lookup["high"])
    assert scales[2].item() == pytest.approx(cal["global"]["scale"])


def test_conformal_by_region_uses_region_scales_with_global_fallback():
    plugin = _fitted_plugin(conformal_apply=True, conformal_by_region=True)
    cal = plugin.conformal_calibration
    assert cal["scope"] == "per_region"
    used = [r for r in cal["regions"] if r.get("used")]
    assert used

    region = used[0]
    signs = torch.tensor(region["signs"], dtype=torch.float64)
    pos = signs.reshape(1, 3) / torch.linalg.norm(signs)
    pos = pos * 1.25
    scale = plugin._conformal_scale_for_query(torch.linalg.norm(pos, dim=-1), pos)
    assert scale[0].item() == pytest.approx(region["scale"])


def test_altitude_binned_noise_model_round_trips(tmp_path):
    plugin = _fitted_plugin(noise_model="altitude_binned")
    assert isinstance(plugin.altitude_noise, BinnedAltitudeNoiseModel)
    assert plugin.fit_info["altitude_noise_type"] == "altitude_binned"

    path = tmp_path / "binned_noise_plugin.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)
    assert isinstance(loaded.altitude_noise, BinnedAltitudeNoiseModel)
    assert loaded.altitude_noise.n_bins == plugin.altitude_noise.n_bins

    queries = _query_shell(32, 1.05, 1.6, seed=29)
    a = plugin.predict_uncertainty(queries)
    b = loaded.predict_uncertainty(queries)
    assert torch.allclose(b.sigma, a.sigma, rtol=1.0e-12, atol=0.0)


def test_calibrated_supervisor_round_trip_scores_identically(tmp_path):
    plugin = _fitted_plugin(calibrate_supervisor=True)
    assert plugin.calibrated_supervisor is not None
    assert plugin.calibrated_supervisor["enabled"] is True

    path = tmp_path / "calibrated_supervisor_plugin.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)
    assert loaded.calibrated_supervisor == plugin.calibrated_supervisor

    queries = _query_shell(32, 1.05, 1.6, seed=37)
    a = plugin.score_trajectory(queries, scoring="calibrated_supervisor_p95")
    b = loaded.score_trajectory(queries, scoring="calibrated_supervisor_p95")
    assert b.risk_score == pytest.approx(a.risk_score, rel=1.0e-12)
    assert not torch.isnan(torch.tensor(a.risk_score))


def test_conformal_save_load_round_trip_predicts_identically(tmp_path):
    plugin = _fitted_plugin(conformal_apply=True)
    path = tmp_path / "conformal_plugin.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)
    assert loaded.conformal_calibration == plugin.conformal_calibration

    queries = _query_shell(32, 1.05, 1.6, seed=19)
    a = plugin.predict_uncertainty(queries)
    b = loaded.predict_uncertainty(queries)
    assert torch.allclose(b.sigma, a.sigma, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(b.expected_error, a.expected_error, rtol=1.0e-12, atol=0.0)


def test_conformal_update_marks_stale_until_fresh_validation():
    plugin = _fitted_plugin(conformal_apply=True)
    assert plugin.fit_info["conformal_stale_after_update"] is False

    new_pos = _query_shell(12, 1.05, 1.6, seed=31)
    new_err = torch.zeros_like(new_pos)
    plugin.update_error(new_pos, new_err)
    assert plugin.fit_info["conformal_stale_after_update"] is True
    assert "without fresh validation" in plugin.fit_info["conformal_stale_reason"]

    val_pos = _query_shell(40, 1.05, 1.6, seed=32)
    val_err = torch.zeros_like(val_pos)
    plugin.update_error(new_pos, new_err, val_positions=val_pos, val_error=val_err)
    assert plugin.fit_info["conformal_stale_after_update"] is False


def test_loaded_plugin_drives_corrected_force_field(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)

    def zero_surrogate(x):
        return torch.zeros_like(x)

    field_orig = CorrectedForceField(plugin, surrogate_accel_fn=zero_surrogate)
    field_loaded = CorrectedForceField(loaded, surrogate_accel_fn=zero_surrogate)
    x = _query_shell(16, 1.1, 1.5, seed=9)
    assert torch.allclose(field_orig.correction(x), field_loaded.correction(x), rtol=1.0e-12, atol=0.0)


def test_unfitted_plugin_refuses_to_save(tmp_path):
    sources = make_shell_sources([0.8], 16, dtype=torch.float64)
    plugin = VESPUQPlugin(sources)
    with pytest.raises(RuntimeError, match="not fitted"):
        plugin.save(tmp_path / "nope.pt")


def test_load_rejects_foreign_and_future_payloads(tmp_path):
    plugin = _fitted_plugin()

    bad_format = tmp_path / "foreign.pt"
    torch.save({"format": "something.else", "version": 1}, bad_format)
    with pytest.raises(ValueError, match="format"):
        VESPUQPlugin.load(bad_format)

    state = plugin.state_dict()
    state["version"] = PLUGIN_STATE_VERSION + 1
    future = tmp_path / "future.pt"
    torch.save(state, future)
    with pytest.raises(ValueError, match="version"):
        VESPUQPlugin.load(future)


def test_state_dict_is_weights_only_safe(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    # the explicit contract: a clean, restricted unpickler must accept the payload
    state = torch.load(path, weights_only=True)
    assert state["format"] == "vesp.uq.plugin"
    rebuilt = VESPUQPlugin.from_state_dict(state)
    assert rebuilt.posterior is not None
