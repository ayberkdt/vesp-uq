import pytest

from experimental_vesp.lunar import (
    MU_MOON_SI,
    R_MOON_M,
    canonical_scales,
    is_lunar_body_signature,
    looks_like_lunar_metadata,
    validate_lunar_metadata_contract,
)


def test_lunar_signature_accepts_lunar_constants():
    assert is_lunar_body_signature(mu_si=MU_MOON_SI, r_ref_m=R_MOON_M)
    assert looks_like_lunar_metadata({"central_body": "moon", "mu_si": MU_MOON_SI, "r_ref_m": R_MOON_M})


def test_lunar_contract_rejects_non_lunar_label():
    with pytest.raises(ValueError, match="not lunar"):
        validate_lunar_metadata_contract({"central_body": "earth", "mu_si": 3.986004418e14})


def test_lunar_contract_rejects_bad_lunar_numbers():
    with pytest.raises(ValueError, match="constants do not look lunar"):
        validate_lunar_metadata_contract({"central_body": "moon", "mu_si": 3.986004418e14, "r_ref_m": 6_378_137.0})


def test_canonical_scales_are_positive():
    du, tu, vu = canonical_scales(mu_si=MU_MOON_SI, du_m=R_MOON_M)
    assert du == R_MOON_M
    assert tu > 0.0
    assert vu > 0.0
