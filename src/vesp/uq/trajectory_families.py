"""Controlled trajectory families for the VESP-UQ diversity study (WP9).

The default screening ensemble (:func:`vesp.uq.ensemble.generate_orbit_ensemble`) uses Haar-random
orientations, so inclination and eccentricity are not controllable. A journal reviewer will ask
whether VESP-UQ's value holds across *distinct* orbit families, so this module generates families
with controlled periapsis/apoapsis, inclination, and arc coverage:

* ``low_alt_near_circular`` -- low periapsis, near-zero eccentricity,
* ``eccentric_perilune`` -- low periapsis, high apoapsis (large eccentricity),
* ``polar`` / ``equatorial`` / ``inclined`` -- controlled inclination bands,
* ``descent_arc`` -- a perilune-centred partial arc (open, not a closed orbit),
* ``high_alt_transfer`` -- entirely high-altitude arcs,
* ``ood_low_alt`` -- periapsis at/below the training-support edge (out of distribution).

Trajectories are in model-normalized body radii, deterministic given a seed. Generation reuses the
same conic geometry as the default ensemble; only the orientation/eccentricity sampling differs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

__all__ = ["FAMILIES", "FamilySpec", "generate_family", "family_descriptor"]


@dataclass(frozen=True)
class FamilySpec:
    r_peri_range: tuple[float, float]
    r_apo_range: tuple[float, float]
    inclination_deg_range: tuple[float, float]
    arc_fraction: float = 1.0  # 1.0 = full closed orbit; <1 = perilune-centred open arc
    note: str = ""


# Inclination is measured from the equatorial (xy) plane: 0 deg equatorial, 90 deg polar.
FAMILIES: dict[str, FamilySpec] = {
    "low_alt_near_circular": FamilySpec((1.05, 1.15), (1.05, 1.18), (0.0, 90.0), 1.0,
                                        "low periapsis, near-circular"),
    "eccentric_perilune": FamilySpec((1.03, 1.10), (1.40, 1.60), (0.0, 90.0), 1.0,
                                     "low perilune pass, high apoapsis"),
    "polar": FamilySpec((1.05, 1.25), (1.25, 1.55), (80.0, 100.0), 1.0, "polar inclination"),
    "equatorial": FamilySpec((1.05, 1.25), (1.25, 1.55), (0.0, 15.0), 1.0, "equatorial inclination"),
    "inclined": FamilySpec((1.05, 1.25), (1.25, 1.55), (30.0, 60.0), 1.0, "mid inclination"),
    "descent_arc": FamilySpec((1.03, 1.10), (1.45, 1.60), (0.0, 90.0), 0.33,
                              "open perilune-centred descent arc"),
    "high_alt_transfer": FamilySpec((1.40, 1.52), (1.52, 1.60), (0.0, 90.0), 1.0,
                                    "entirely high-altitude arcs"),
    "ood_low_alt": FamilySpec((1.00, 1.03), (1.10, 1.40), (0.0, 90.0), 1.0,
                              "periapsis at/below training-support edge (OOD)"),
}


@dataclass
class Family:
    name: str
    trajectories: list[torch.Tensor]
    periapsis: torch.Tensor
    apoapsis: torch.Tensor
    inclination_deg: torch.Tensor
    eccentricity: torch.Tensor
    initial_states: torch.Tensor | None = None  # (N, 6) [r, v] at periapsis, mu=1 normalized
    period: torch.Tensor | None = None          # (N,) Keplerian period (mu=1)


def _rotation(omega_raan: torch.Tensor, inc: torch.Tensor, arg_peri: torch.Tensor) -> torch.Tensor:
    """3-1-3 (Rz(RAAN) Rx(inc) Rz(argp)) rotation matrices for ``(n,)`` angle tensors."""

    def rz(a):
        c, s = torch.cos(a), torch.sin(a)
        z, o = torch.zeros_like(a), torch.ones_like(a)
        return torch.stack([
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ], dim=-2)

    def rx(a):
        c, s = torch.cos(a), torch.sin(a)
        z, o = torch.zeros_like(a), torch.ones_like(a)
        return torch.stack([
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ], dim=-2)

    return rz(omega_raan) @ rx(inc) @ rz(arg_peri)


def generate_family(
    name: str,
    *,
    n_orbits: int = 2000,
    n_points: int = 120,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> Family:
    """Generate one controlled trajectory family (see :data:`FAMILIES`)."""

    if name not in FAMILIES:
        raise ValueError(f"unknown family {name!r}; choices: {sorted(FAMILIES)}")
    spec = FAMILIES[name]
    # Stable per-family offset (Python's str hash is salted per process -> not reproducible).
    name_offset = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 997
    g = torch.Generator().manual_seed(int(seed) + 31 * name_offset)

    u_peri = torch.rand(n_orbits, generator=g, dtype=dtype)
    u_apo = torch.rand(n_orbits, generator=g, dtype=dtype)
    r_peri = spec.r_peri_range[0] + u_peri * (spec.r_peri_range[1] - spec.r_peri_range[0])
    apo_lo = torch.clamp(r_peri, min=spec.r_apo_range[0])
    r_apo = apo_lo + u_apo * torch.clamp(torch.tensor(spec.r_apo_range[1], dtype=dtype) - apo_lo, min=0.0)

    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri).clamp_min(torch.finfo(dtype).tiny)
    p = a * (1.0 - e * e)

    i_lo, i_hi = spec.inclination_deg_range
    inc_deg = i_lo + torch.rand(n_orbits, generator=g, dtype=dtype) * (i_hi - i_lo)
    inc = inc_deg * (math.pi / 180.0)
    raan = torch.rand(n_orbits, generator=g, dtype=dtype) * (2.0 * math.pi)
    argp = torch.rand(n_orbits, generator=g, dtype=dtype) * (2.0 * math.pi)
    rotations = _rotation(raan, inc, argp)

    span = spec.arc_fraction * 2.0 * math.pi
    if spec.arc_fraction >= 1.0:
        theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, dtype=dtype)[:-1]
    else:
        theta = torch.linspace(-0.5 * span, 0.5 * span, n_points, dtype=dtype)  # perilune-centred

    # Initial state at periapsis (theta=0) for trajectory propagation: r0 = R @ [r_peri,0,0],
    # v0 = R @ [0, v_peri, 0] (prograde), with the vis-viva periapsis speed (mu=1 normalized).
    v_peri = torch.sqrt((2.0 / r_peri - 1.0 / a).clamp_min(0.0))  # mu = 1
    period = 2.0 * math.pi * torch.sqrt(a ** 3)                   # mu = 1
    states = torch.empty(n_orbits, 6, dtype=dtype)

    trajectories: list[torch.Tensor] = []
    for k in range(n_orbits):
        r = p[k] / (1.0 + e[k] * torch.cos(theta))
        plane = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        trajectories.append(plane @ rotations[k].transpose(0, 1))
        r0 = rotations[k] @ torch.tensor([r_peri[k], 0.0, 0.0], dtype=dtype)
        v0 = rotations[k] @ torch.tensor([0.0, v_peri[k], 0.0], dtype=dtype)
        states[k] = torch.cat([r0, v0])

    return Family(name=name, trajectories=trajectories, periapsis=r_peri, apoapsis=r_apo,
                  inclination_deg=inc_deg, eccentricity=e, initial_states=states, period=period)


def family_descriptor(fam: Family) -> dict:
    """Summary ranges (counts, altitude, inclination, eccentricity) for a generated family."""

    min_r = torch.stack([torch.linalg.norm(t, dim=-1).min() for t in fam.trajectories])
    max_r = torch.stack([torch.linalg.norm(t, dim=-1).max() for t in fam.trajectories])
    return {
        "family": fam.name,
        "n_trajectories": len(fam.trajectories),
        "min_radius_low": float(min_r.min()),
        "min_radius_high": float(min_r.max()),
        "max_radius_high": float(max_r.max()),
        "inclination_deg_low": float(fam.inclination_deg.min()),
        "inclination_deg_high": float(fam.inclination_deg.max()),
        "eccentricity_low": float(fam.eccentricity.min()),
        "eccentricity_high": float(fam.eccentricity.max()),
    }
