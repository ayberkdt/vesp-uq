"""Surrogate-agnostic sample interface for VESP-UQ.

VESP-UQ only needs, at each calibration position, the residual-force error
``e_a(x) = a_reference(x) - a_surrogate(x)``. This module loads/holds those samples in two
equivalent modes:

A) **Direct error mode** -- the CSV already stores the error (or residual) acceleration; the
   surrogate is implicitly zero (this is the current band-limited residual dataset, where the
   stored acceleration *is* the degree-truncation surrogate's error).
B) **Reference/surrogate mode** -- the CSV stores both ``a_reference`` and ``a_surrogate`` and
   the error is computed as their difference.

Nothing here knows anything about the surrogate's architecture: it is an acceleration-level
interface only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

# Column aliases for the explicit reference/surrogate/error CSV format.
_POS_ALIASES = {"x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z")}
_REF_ALIASES = {
    "ax_ref": ("ax_ref", "a_x_ref", "ax_reference", "Delta a_x_ref"),
    "ay_ref": ("ay_ref", "a_y_ref", "ay_reference", "Delta a_y_ref"),
    "az_ref": ("az_ref", "a_z_ref", "az_reference", "Delta a_z_ref"),
}
_SUR_ALIASES = {
    "ax_sur": ("ax_sur", "a_x_sur", "ax_surrogate", "ax_sur"),
    "ay_sur": ("ay_sur", "a_y_sur", "ay_surrogate"),
    "az_sur": ("az_sur", "a_z_sur", "az_surrogate"),
}
_ERR_ALIASES = {
    "ax_err": ("ax_err", "a_x_err", "ax_error", "Delta a_x", "Delta_a_x", "delta_a_x", "dax"),
    "ay_err": ("ay_err", "a_y_err", "ay_error", "Delta a_y", "Delta_a_y", "delta_a_y", "day"),
    "az_err": ("az_err", "a_z_err", "az_error", "Delta a_z", "Delta_a_z", "delta_a_z", "daz"),
}


@dataclass
class UQSamples:
    """Calibration samples for the VESP-UQ layer.

    ``error`` is always populated (``reference - surrogate`` when both are given). ``reference``
    and ``surrogate`` are kept when available; in direct-error mode ``surrogate`` is zeros and
    ``reference`` equals ``error``.
    """

    positions: torch.Tensor  # (N, 3)
    error: torch.Tensor  # (N, 3)
    reference: torch.Tensor | None = None
    surrogate: torch.Tensor | None = None
    metadata: dict | None = None

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def radius(self) -> torch.Tensor:
        return torch.linalg.norm(self.positions, dim=-1)

    def subset(self, indices: torch.Tensor) -> "UQSamples":
        return UQSamples(
            positions=self.positions[indices],
            error=self.error[indices],
            reference=self.reference[indices] if self.reference is not None else None,
            surrogate=self.surrogate[indices] if self.surrogate is not None else None,
            metadata=self.metadata,
        )

    def to(self, device: torch.device | str) -> "UQSamples":
        return UQSamples(
            positions=self.positions.to(device),
            error=self.error.to(device),
            reference=self.reference.to(device) if self.reference is not None else None,
            surrogate=self.surrogate.to(device) if self.surrogate is not None else None,
            metadata=self.metadata,
        )


def validate_uq_samples(samples: UQSamples) -> UQSamples:
    """Check shapes/finiteness; raise a clear ``ValueError`` otherwise. Returns the samples."""

    for name, t in (("positions", samples.positions), ("error", samples.error)):
        if t.ndim != 2 or t.shape[-1] != 3:
            raise ValueError(f"UQSamples.{name} must have shape (N, 3), got {tuple(t.shape)}")
    if samples.positions.shape[0] != samples.error.shape[0]:
        raise ValueError("UQSamples.positions and .error must have the same number of rows")
    if samples.n == 0:
        raise ValueError("UQSamples is empty")
    if not torch.isfinite(samples.positions).all() or not torch.isfinite(samples.error).all():
        raise ValueError("UQSamples contains non-finite positions or error values")
    return samples


def _resolve(fieldnames: set[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str] | None:
    """Return logical->actual column map if *all* logical names resolve, else ``None``."""

    selected: dict[str, str] = {}
    for logical, opts in aliases.items():
        match = next((o for o in opts if o in fieldnames), None)
        if match is None:
            return None
        selected[logical] = match
    return selected


def load_uq_samples_from_csv(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    mode: str = "auto",
) -> UQSamples:
    """Load VESP-UQ calibration samples from a CSV.

    ``mode``:
      - ``"auto"`` (default): use error columns if present, else reference+surrogate.
      - ``"error"``: require error/residual columns (``ax_err``/``Delta a_x`` ...).
      - ``"reference_surrogate"``: require both reference and surrogate columns.

    Recognized columns (first alias shown): ``x, y, z``; reference ``ax_ref, ay_ref, az_ref``;
    surrogate ``ax_sur, ay_sur, az_sur``; error/residual ``ax_err, ay_err, az_err`` (also the
    legacy ``Delta a_x`` residual names). Raises a clear ``ValueError`` if required columns are
    missing.
    """

    if mode not in {"auto", "error", "reference_surrogate"}:
        raise ValueError("mode must be 'auto', 'error', or 'reference_surrogate'")
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        fields = set(reader.fieldnames)
        pos_cols = _resolve(fields, _POS_ALIASES)
        if pos_cols is None:
            raise ValueError(f"CSV {path} is missing position columns x, y, z (found {sorted(fields)})")
        ref_cols = _resolve(fields, _REF_ALIASES)
        sur_cols = _resolve(fields, _SUR_ALIASES)
        err_cols = _resolve(fields, _ERR_ALIASES)

        use_refsur = (mode == "reference_surrogate") or (mode == "auto" and err_cols is None)
        if use_refsur:
            if ref_cols is None or sur_cols is None:
                raise ValueError(
                    f"CSV {path} needs reference columns (ax_ref, ay_ref, az_ref) and surrogate "
                    f"columns (ax_sur, ay_sur, az_sur) for reference/surrogate mode"
                )
        else:
            if err_cols is None:
                raise ValueError(
                    f"CSV {path} needs error columns (ax_err, ay_err, az_err) -- or the legacy "
                    f"'Delta a_x/y/z' residual columns -- for direct error mode"
                )

        pos_rows, ref_rows, sur_rows, err_rows = [], [], [], []
        for row in reader:
            pos_rows.append([float(row[pos_cols[c]]) for c in ("x", "y", "z")])
            if use_refsur:
                ref_rows.append([float(row[ref_cols[c]]) for c in ("ax_ref", "ay_ref", "az_ref")])
                sur_rows.append([float(row[sur_cols[c]]) for c in ("ax_sur", "ay_sur", "az_sur")])
            else:
                err_rows.append([float(row[err_cols[c]]) for c in ("ax_err", "ay_err", "az_err")])

    if not pos_rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    positions = torch.tensor(pos_rows, dtype=dtype)
    if use_refsur:
        reference = torch.tensor(ref_rows, dtype=dtype)
        surrogate = torch.tensor(sur_rows, dtype=dtype)
        error = reference - surrogate
        meta = {"mode": "reference_surrogate", "path": str(path)}
    else:
        error = torch.tensor(err_rows, dtype=dtype)
        reference = error.clone()
        surrogate = torch.zeros_like(error)
        meta = {"mode": "error", "path": str(path)}
    return validate_uq_samples(UQSamples(positions, error, reference, surrogate, metadata=meta))


def split_uq_samples(
    samples: UQSamples, *, train_fraction: float = 0.7, seed: int = 0
) -> tuple[UQSamples, UQSamples]:
    """Deterministic random train/held-out split (reproducible for a given ``seed``)."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    perm = torch.randperm(samples.n, generator=torch.Generator().manual_seed(int(seed)))
    n_train = int(round(train_fraction * samples.n))
    return samples.subset(perm[:n_train]), samples.subset(perm[n_train:])


def make_synthetic_uq_samples(
    *,
    n: int = 512,
    n_truth_sources: int = 24,
    truth_shell: float = 0.7,
    query_r_range: tuple[float, float] = (1.03, 1.6),
    noise_std: float = 1.0e-4,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> UQSamples:
    """Generate a tiny synthetic error field from interior truth sources (for smoke runs/tests).

    The error is the analytic acceleration of random interior point sources plus a small
    homoscedastic noise floor, so it lives in the equivalent-source span and the layer can fit
    and calibrate it. ``surrogate`` is zero; ``reference`` equals the error.
    """

    from vesp.core.operators import build_acceleration_operator
    from vesp.core.sources import make_shell_sources

    g = torch.Generator().manual_seed(int(seed))
    dirs = torch.randn(n, 3, generator=g, dtype=dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (query_r_range[0] + (query_r_range[1] - query_r_range[0]) * torch.rand(n, generator=g, dtype=dtype))
    positions = dirs * radii.unsqueeze(-1)

    truth = make_shell_sources([truth_shell], n_truth_sources, dtype=dtype)
    sigma_truth = 0.02 * torch.randn(truth.n_sources, generator=g, dtype=dtype)
    A = build_acceleration_operator(positions, truth, eps=0.0, sign=1.0)
    error_flat = A @ sigma_truth
    error = error_flat.reshape(3, n).transpose(0, 1).contiguous()
    if noise_std > 0.0:
        error = error + noise_std * torch.randn(n, 3, generator=g, dtype=dtype)
    return validate_uq_samples(
        UQSamples(
            positions=positions,
            error=error,
            reference=error.clone(),
            surrogate=torch.zeros_like(error),
            metadata={"mode": "synthetic", "n_truth_sources": n_truth_sources, "truth_shell": truth_shell},
        )
    )
