"""Real spherical-harmonic gravity data ingestion.

This module targets PDS SHADR/SHA ASCII gravity models, especially the GRAIL
lunar gravity products. It can download a model, parse fully normalized
spherical harmonic coefficients, and expand a truncated residual field into the
CSV format consumed by the discrete VESP trainers.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesp.common.artifacts import atomic_write_json
from vesp.common.lunar import canonical_scales
from vesp.data.dataset import write_dataset_metadata
from vesp.data.gravity_io import read_shadr_ascii

PDS_GRAIL_SHADR_BASE = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr"

KNOWN_MODELS = {
    "gl0420a": {
        "tab_url": f"{PDS_GRAIL_SHADR_BASE}/jggrx_0420a_sha.tab",
        "label_url": f"{PDS_GRAIL_SHADR_BASE}/jggrx_0420a_sha.lbl",
        "description": "JPL GRAIL420C1A lunar gravity model, degree/order 420.",
    },
    "grgm1200a": {
        "tab_url": f"{PDS_GRAIL_SHADR_BASE}/gggrx_1200a_sha.tab",
        "label_url": f"{PDS_GRAIL_SHADR_BASE}/gggrx_1200a_sha.lbl",
        "description": "GSFC GRGM1200A lunar gravity model, degree/order 1200.",
    },
}


@dataclass
class SphericalHarmonicGravityModel:
    name: str
    reference_radius_km: float
    gm_km3_s2: float
    degree: int
    order: int
    c: np.ndarray
    s: np.ndarray
    normalization_state: int | None = None
    source_path: str | None = None
    column_order: str | None = None


def download_file(url: str, output_path: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output
    with urllib.request.urlopen(url) as response, output.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or "0")
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            copied += len(chunk)
            if total:
                print(f"downloaded {copied / total:.1%}", end="\r")
    if total:
        print()
    return output


def download_known_model(model_name: str, data_dir: str | Path = "data/gravity_models") -> tuple[Path, Path]:
    key = model_name.lower()
    if key not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(KNOWN_MODELS)}")
    info = KNOWN_MODELS[key]
    root = Path(data_dir)
    tab = download_file(info["tab_url"], root / Path(info["tab_url"]).name.lower())
    lbl = download_file(info["label_url"], root / Path(info["label_url"]).name.lower())
    return tab, lbl


def read_pds_sha(
    path: str | Path,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    strict: bool = True,
) -> SphericalHarmonicGravityModel:
    """Read a PDS SHA/TAB spherical harmonic model.

    The parser is intentionally strict by default. It auto-detects degree/order
    vs order/degree column order and validates coefficient coverage.
    """

    table = read_shadr_ascii(path, max_degree=max_degree, name=name, strict=strict)
    return SphericalHarmonicGravityModel(
        name=table.name,
        reference_radius_km=table.reference_radius_km,
        gm_km3_s2=table.gm_km3_s2,
        degree=table.degree,
        order=table.order,
        c=table.c,
        s=table.s,
        normalization_state=table.normalization_state,
        source_path=table.source_path,
        column_order=table.column_order,
    )


def build_legendre_coeffs(n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag = np.zeros(n_max + 1, dtype=np.float64)
    subdiag = np.zeros(n_max + 1, dtype=np.float64)
    A = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    B = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)

    if n_max >= 1:
        n = np.arange(1, n_max + 1, dtype=np.float64)
        diag[1:] = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
        subdiag[1:] = np.sqrt(2.0 * n + 1.0)

    for n_int in range(2, n_max + 1):
        n = float(n_int)
        m = np.arange(0.0, n - 1.0, dtype=np.float64)
        A[n_int, : n_int - 1] = np.sqrt(((2.0 * n - 1.0) * (2.0 * n + 1.0)) / ((n - m) * (n + m)))
        B[n_int, : n_int - 1] = np.sqrt(
            ((2.0 * n + 1.0) * (n - m - 1.0) * (n + m - 1.0)) / ((2.0 * n - 3.0) * (n + m) * (n - m))
        )

    # VESP uses geodesy convention: no Condon-Shortley phase (-1)^m
    scale_m = np.ones(n_max + 1, dtype=np.float64)
    if n_max >= 1:
        scale_m[1:] *= math.sqrt(2.0)

    return diag, subdiag, A, B, scale_m


def compute_normalized_legendre_matrix(degree_max: int, sin_lat: np.ndarray) -> np.ndarray:
    N = int(degree_max)
    K = sin_lat.shape[0]

    # cos_lat is strictly non-negative because -pi/2 <= lat <= pi/2
    cos_lat = np.sqrt(1.0 - sin_lat**2)

    diag, subdiag, A, B, scale_m = build_legendre_coeffs(N)

    P = np.zeros((N + 1, N + 1, K), dtype=np.float64)
    P[0, 0, :] = 1.0

    for n in range(1, N + 1):
        # Diagonal
        P[n, n, :] = diag[n] * cos_lat * P[n - 1, n - 1, :]
        # Sub-diagonal
        P[n, n - 1, :] = subdiag[n] * sin_lat * P[n - 1, n - 1, :]
        # Vertical
        if n >= 2:
            for m in range(n - 1):
                P[n, m, :] = A[n, m] * sin_lat * P[n - 1, m, :] - B[n, m] * P[n - 2, m, :]

    for m in range(N + 1):
        P[:, m, :] *= scale_m[m]

    return P


def residual_potential(
    model: SphericalHarmonicGravityModel,
    positions_normalized: np.ndarray,
    *,
    degree_min: int = 2,
    degree_max: int | None = None,
    remove_zonal: bool = False,
) -> np.ndarray:
    """Evaluate residual potential at normalized Cartesian query points.

    The returned potential is physical potential in km^2/s^2, but differentiated
    with respect to normalized coordinates when finite-difference acceleration is
    computed by ``residual_acceleration_finite_difference``.
    """

    x = positions_normalized[:, 0]
    y = positions_normalized[:, 1]
    z = positions_normalized[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(y, x)
    sin_lat = z / r
    max_l = min(model.degree, degree_max if degree_max is not None else model.degree)

    # Precompute Legendre matrix for all queries to avoid float64 lpmv overflow
    P_matrix = compute_normalized_legendre_matrix(max_l, sin_lat)

    series = np.zeros_like(r, dtype=np.float64)
    for degree in range(max(0, degree_min), max_l + 1):
        radial = r ** (-(degree + 1))
        degree_sum = np.zeros_like(r, dtype=np.float64)
        max_m = min(degree, model.order)
        for order in range(0, max_m + 1):
            if remove_zonal and order == 0:
                continue
            c_lm = model.c[degree, order]
            s_lm = model.s[degree, order]
            if c_lm == 0.0 and s_lm == 0.0:
                continue
            p_lm = P_matrix[degree, order, :]
            if order == 0:
                trig = c_lm
            else:
                trig = c_lm * np.cos(order * lon) + s_lm * np.sin(order * lon)
            degree_sum += p_lm * trig
        series += radial * degree_sum

    return (model.gm_km3_s2 / model.reference_radius_km) * series


def residual_acceleration_finite_difference(
    model: SphericalHarmonicGravityModel,
    positions_normalized: np.ndarray,
    *,
    degree_min: int = 2,
    degree_max: int | None = None,
    remove_zonal: bool = False,
    step: float = 1.0e-4,
) -> np.ndarray:
    """Finite-difference gradient of residual potential wrt normalized x,y,z."""

    acc = np.zeros_like(positions_normalized, dtype=np.float64)
    for axis in range(3):
        plus = positions_normalized.copy()
        minus = positions_normalized.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        u_plus = residual_potential(
            model,
            plus,
            degree_min=degree_min,
            degree_max=degree_max,
            remove_zonal=remove_zonal,
        )
        u_minus = residual_potential(
            model,
            minus,
            degree_min=degree_min,
            degree_max=degree_max,
            remove_zonal=remove_zonal,
        )
        acc[:, axis] = (u_plus - u_minus) / (2.0 * step)
    return acc


def random_exterior_points(
    n_points: int,
    *,
    radius_min: float = 1.03,
    radius_max: float = 1.60,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(radius_min, radius_max, size=(n_points, 1))
    return directions * radii


def write_residual_dataset_csv(
    output_path: str | Path,
    positions_normalized: np.ndarray,
    potential: np.ndarray,
    acceleration: np.ndarray,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "Delta U", "Delta a_x", "Delta a_y", "Delta a_z"])
        for x, u, a in zip(positions_normalized, potential, acceleration):
            writer.writerow([x[0], x[1], x[2], u, a[0], a[1], a[2]])
    return output


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _vector_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(values, dtype=np.float64), axis=1))))


def residual_dataset_diagnostics(
    *,
    positions_normalized: np.ndarray,
    potential: np.ndarray,
    acceleration: np.ndarray,
    model: SphericalHarmonicGravityModel,
    degree_min: int,
    degree_max: int,
    radius_min: float,
    radius_max: float,
    finite_difference_step: float,
    acceleration_output: str,
    acceleration_units: str,
) -> dict:
    norms = np.linalg.norm(positions_normalized, axis=1)
    diagnostics = {
        "n_query": int(positions_normalized.shape[0]),
        "radius_min": float(radius_min),
        "radius_max": float(radius_max),
        "degree_min": int(degree_min),
        "degree_max": int(degree_max),
        "position_norm_min": float(np.min(norms)),
        "position_norm_max": float(np.max(norms)),
        "potential_rms": _rms(potential),
        "acceleration_rms": _vector_rms(acceleration),
        "acceleration_output": acceleration_output,
        "acceleration_units": acceleration_units,
        "finite_difference_step": float(finite_difference_step),
        "reference_radius_km": float(model.reference_radius_km),
        "gm_km3_s2": float(model.gm_km3_s2),
    }
    if acceleration_output == "physical":
        rms = diagnostics["acceleration_rms"]
        if not math.isfinite(rms) or rms <= 0.0:
            raise ValueError("physical acceleration RMS must be finite and positive")
    return diagnostics


def write_residual_diagnostics(path: str | Path, diagnostics: dict) -> Path:
    output = Path(path)
    if output.suffix.lower() == ".csv":
        output = output.with_suffix(".diagnostics.json")
    atomic_write_json(output, diagnostics)
    return output


def build_real_lunar_dataset(
    *,
    model_name: str = "gl0420a",
    sha_path: str | Path | None = None,
    data_dir: str | Path = "data/gravity_models",
    output_path: str | Path = "data/lunar_grail_residual.csv",
    n_query: int = 1024,
    degree_min: int = 2,
    degree_max: int = 60,
    radius_min: float = 1.03,
    radius_max: float = 1.60,
    finite_difference_step: float = 1.0e-4,
    acceleration_output: str = "physical",
    remove_zonal: bool = False,
    seed: int = 42,
) -> Path:
    if sha_path is None:
        sha_path, _ = download_known_model(model_name, data_dir=data_dir)
    model = read_pds_sha(sha_path, max_degree=degree_max, name=model_name, strict=True)
    points = random_exterior_points(n_query, radius_min=radius_min, radius_max=radius_max, seed=seed)
    potential = residual_potential(
        model,
        points,
        degree_min=degree_min,
        degree_max=degree_max,
        remove_zonal=remove_zonal,
    )
    grad_normalized = residual_acceleration_finite_difference(
        model,
        points,
        degree_min=degree_min,
        degree_max=degree_max,
        remove_zonal=remove_zonal,
        step=finite_difference_step,
    )
    if acceleration_output == "physical":
        acceleration = grad_normalized / model.reference_radius_km
        acceleration_units = "km/s^2"
        acceleration_kind = "physical"
    elif acceleration_output == "normalized_gradient":
        acceleration = grad_normalized
        acceleration_units = "km^2/s^2 per normalized radius"
        acceleration_kind = "normalized_gradient"
    else:
        raise ValueError("acceleration_output must be 'physical' or 'normalized_gradient'")
    output = write_residual_dataset_csv(output_path, points, potential, acceleration)
    diagnostics = residual_dataset_diagnostics(
        positions_normalized=points,
        potential=potential,
        acceleration=acceleration,
        model=model,
        degree_min=degree_min,
        degree_max=degree_max,
        radius_min=radius_min,
        radius_max=radius_max,
        finite_difference_step=finite_difference_step,
        acceleration_output=acceleration_kind,
        acceleration_units=acceleration_units,
    )
    diagnostics_path = write_residual_diagnostics(output, diagnostics)
    du_m, tu_s, vu_m_s = canonical_scales(mu_si=model.gm_km3_s2 * 1.0e9, du_m=model.reference_radius_km * 1000.0)
    write_dataset_metadata(
        output,
        {
            "metadata_schema": "vesp_lunar_residual_csv_v1",
            "central_body": "moon",
            "target_name": "MOON",
            "position_units": "normalized",
            "potential_units": "km^2/s^2",
            "acceleration_units": acceleration_units,
            "acceleration_output": acceleration_kind,
            "R_body": model.reference_radius_km,
            "R_body_units": "km",
            "r_ref_m": model.reference_radius_km * 1000.0,
            "gm_km3_s2": model.gm_km3_s2,
            "mu_si": model.gm_km3_s2 * 1.0e9,
            "DU_m": du_m,
            "TU_s": tu_s,
            "VU_m_s": vu_m_s,
            "coordinate_system": "body-fixed normalized Cartesian",
            "gravity_model": model.name,
            "gravity_model_path": model.source_path,
            "normalization_state": model.normalization_state,
            "column_order": model.column_order,
            "degree_min": degree_min,
            "degree_max": degree_max,
        },
    )
    print(
        "real_lunar_diagnostics: "
        f"n_query={diagnostics['n_query']} "
        f"r=[{diagnostics['position_norm_min']:.4f}, {diagnostics['position_norm_max']:.4f}] "
        f"potential_rms={diagnostics['potential_rms']:.6e} "
        f"acceleration_rms={diagnostics['acceleration_rms']:.6e} "
        f"path={diagnostics_path}"
    )
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build real lunar residual gravity CSV data from PDS GRAIL SHA models.")
    parser.add_argument("--model", default="gl0420a", choices=sorted(KNOWN_MODELS))
    parser.add_argument("--sha-path", default=None)
    parser.add_argument("--data-dir", default="data/gravity_models")
    parser.add_argument("--output", default="data/lunar_grail_residual.csv")
    parser.add_argument("--n-query", type=int, default=1024)
    parser.add_argument("--degree-min", type=int, default=2)
    parser.add_argument("--degree-max", type=int, default=60)
    parser.add_argument("--radius-min", type=float, default=1.03)
    parser.add_argument("--radius-max", type=float, default=1.60)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-4)
    parser.add_argument("--acceleration-output", choices=["physical", "normalized_gradient"], default="physical")
    parser.add_argument("--remove-zonal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output = build_real_lunar_dataset(
        model_name=args.model,
        sha_path=args.sha_path,
        data_dir=args.data_dir,
        output_path=args.output,
        n_query=args.n_query,
        degree_min=args.degree_min,
        degree_max=args.degree_max,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        finite_difference_step=args.finite_difference_step,
        acceleration_output=args.acceleration_output,
        remove_zonal=args.remove_zonal,
        seed=args.seed,
    )
    print(f"real_lunar_dataset: {output}")


if __name__ == "__main__":
    main()
