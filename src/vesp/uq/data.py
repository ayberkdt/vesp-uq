"""Surrogate-interface-agnostic sample interface for VESP-UQ.

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
import math
from collections.abc import Mapping
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

    def subset(self, indices: torch.Tensor) -> UQSamples:
        return UQSamples(
            positions=self.positions[indices],
            error=self.error[indices],
            reference=self.reference[indices] if self.reference is not None else None,
            surrogate=self.surrogate[indices] if self.surrogate is not None else None,
            metadata=self.metadata,
        )

    def to(self, device: torch.device | str) -> UQSamples:
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


def _resolve(fieldnames: set[str], aliases: Mapping[str, tuple[str, ...]]) -> dict[str, str] | None:
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
    with open(path, encoding="utf-8-sig", newline="") as f:
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

        # optional group / trajectory-id column (enables the trajectory_group split, R2WP-7)
        group_col = next(
            (c for c in ("group", "traj_id", "trajectory_id", "orbit_id") if c in fields), None
        )

        pos_rows, ref_rows, sur_rows, err_rows, grp_rows = [], [], [], [], []
        for row in reader:
            pos_rows.append([float(row[pos_cols[c]]) for c in ("x", "y", "z")])
            if group_col is not None:
                grp_rows.append(row[group_col])
            if use_refsur:
                assert ref_cols is not None and sur_cols is not None
                ref_rows.append([float(row[ref_cols[c]]) for c in ("ax_ref", "ay_ref", "az_ref")])
                sur_rows.append([float(row[sur_cols[c]]) for c in ("ax_sur", "ay_sur", "az_sur")])
            else:
                assert err_cols is not None
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
    if grp_rows:
        meta["groups"] = grp_rows  # aligned with the FULL sample set (not with subsets)
    return validate_uq_samples(UQSamples(positions, error, reference, surrogate, metadata=meta))


def split_uq_samples(
    samples: UQSamples, *, train_fraction: float = 0.7, seed: int = 0
) -> tuple[UQSamples, UQSamples]:
    """Deterministic random train/held-out split (reproducible for a given ``seed``).

    Point-level random splits measure INTERPOLATION on a smooth, spatially correlated field --
    training neighbors can sit arbitrarily close to held-out points. For spatial-generalization
    evidence use :func:`split_uq_samples_by_config` with an ``altitude_disjoint`` /
    ``angular_block`` / ``trajectory_group`` method instead (R2WP-7).
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    perm = torch.randperm(samples.n, generator=torch.Generator().manual_seed(int(seed)))
    n_train = int(round(train_fraction * samples.n))
    return samples.subset(perm[:n_train]), samples.subset(perm[n_train:])


# --------------------------------------------------------------------- spatial splits (R2WP-7)
UQ_SPLIT_METHODS = ("random", "altitude_disjoint", "angular_block", "trajectory_group")


def _require_both_sides(train_mask: torch.Tensor, held_mask: torch.Tensor, method: str) -> None:
    if int(train_mask.sum()) == 0 or int(held_mask.sum()) == 0:
        raise ValueError(
            f"split method {method!r} produced an empty side "
            f"(train={int(train_mask.sum())}, held={int(held_mask.sum())}); adjust its parameters"
        )


def split_uq_samples_altitude_disjoint(
    samples: UQSamples,
    *,
    held_quantile: tuple[float, float] = (0.0, 0.3),
    buffer: float = 0.02,
    seed: int = 0,  # noqa: ARG001 -- kept for a uniform split signature; the split is deterministic
) -> tuple[UQSamples, UQSamples, dict]:
    """Altitude-disjoint split: hold out a contiguous radius band, train on the rest.

    ``held_quantile`` selects the held band by radius quantiles (default: the lowest 30% of
    radii, the OOD-toward-surface case). Train points within ``buffer`` (radius units) of the
    held band are DROPPED, so no training neighbor sits at the band edge.
    """

    q_lo, q_hi = float(held_quantile[0]), float(held_quantile[1])
    if not 0.0 <= q_lo < q_hi <= 1.0:
        raise ValueError("held_quantile must satisfy 0 <= lo < hi <= 1")
    r = samples.radius
    lo = torch.quantile(r, q_lo) if q_lo > 0.0 else r.min()
    hi = torch.quantile(r, q_hi) if q_hi < 1.0 else r.max()
    held_mask = (r >= lo) & (r <= hi)
    train_mask = (r < lo - buffer) | (r > hi + buffer)
    _require_both_sides(train_mask, held_mask, "altitude_disjoint")
    info = {
        "method": "altitude_disjoint",
        "held_quantile": [q_lo, q_hi],
        "held_radius_range": [float(lo), float(hi)],
        "buffer": float(buffer),
        "n_train": int(train_mask.sum()),
        "n_held": int(held_mask.sum()),
        "n_dropped_buffer": int(samples.n - int(train_mask.sum()) - int(held_mask.sum())),
    }
    idx = torch.arange(samples.n)
    return samples.subset(idx[train_mask]), samples.subset(idx[held_mask]), info


def split_uq_samples_angular_block(
    samples: UQSamples,
    *,
    train_fraction: float = 0.7,
    n_blocks: int = 12,
    buffer_deg: float = 0.0,
    seed: int = 0,
) -> tuple[UQSamples, UQSamples, dict]:
    """Angular-block split: spherical-Voronoi cells held out whole (block-level holdout).

    ``n_blocks`` seeded random unit vectors define Voronoi cells on the direction sphere; whole
    cells are assigned to the held side until it reaches ``1 - train_fraction`` of the points.
    With ``buffer_deg > 0``, train points within that great-circle distance of any held point are
    dropped, so the two sides are separated by an angular gap.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if int(n_blocks) < 2:
        raise ValueError("n_blocks must be >= 2")
    g = torch.Generator().manual_seed(int(seed))
    dirs = samples.positions / torch.linalg.norm(samples.positions, dim=-1, keepdim=True)
    centers = torch.randn(int(n_blocks), 3, generator=g, dtype=dirs.dtype)
    centers = centers / torch.linalg.norm(centers, dim=-1, keepdim=True)
    assign = torch.argmax(dirs @ centers.transpose(0, 1), dim=-1)

    target_held = (1.0 - train_fraction) * samples.n
    block_order = torch.randperm(int(n_blocks), generator=g)
    held_blocks: list[int] = []
    n_held = 0
    for b in block_order.tolist():
        count = int((assign == b).sum())
        if count == 0:
            continue
        held_blocks.append(b)
        n_held += count
        if n_held >= target_held:
            break
    held_mask = torch.isin(assign, torch.tensor(held_blocks))
    if int(held_mask.sum()) == samples.n:  # never hold everything
        raise ValueError("angular_block split held out every point; increase n_blocks")
    train_mask = ~held_mask

    n_dropped = 0
    if buffer_deg > 0.0 and bool(held_mask.any()):
        cos_thresh = math.cos(math.radians(float(buffer_deg)))
        held_dirs = dirs[held_mask]
        keep = train_mask.clone()
        idx_train = torch.arange(samples.n)[train_mask]
        for a in range(0, idx_train.numel(), 4096):  # chunked NN so memory stays bounded
            blk = idx_train[a : a + 4096]
            max_cos = (dirs[blk] @ held_dirs.transpose(0, 1)).max(dim=-1).values
            keep[blk[max_cos >= cos_thresh]] = False
        n_dropped = int(train_mask.sum() - keep.sum())
        train_mask = keep
    _require_both_sides(train_mask, held_mask, "angular_block")
    info = {
        "method": "angular_block",
        "n_blocks": int(n_blocks),
        "held_blocks": sorted(held_blocks),
        "buffer_deg": float(buffer_deg),
        "n_train": int(train_mask.sum()),
        "n_held": int(held_mask.sum()),
        "n_dropped_buffer": n_dropped,
    }
    idx = torch.arange(samples.n)
    return samples.subset(idx[train_mask]), samples.subset(idx[held_mask]), info


def split_uq_samples_trajectory_group(
    samples: UQSamples,
    groups,
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[UQSamples, UQSamples, dict]:
    """Group-level split: every sample of a group (e.g. one trajectory) lands on one side only.

    ``groups`` is a length-``N`` sequence of hashable group labels aligned with ``samples``.
    Groups are shuffled deterministically and assigned whole to the train side until it reaches
    ``train_fraction`` of the points.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    labels = list(groups)
    if len(labels) != samples.n:
        raise ValueError(f"groups must have length {samples.n}, got {len(labels)}")
    unique = sorted(set(labels), key=str)
    if len(unique) < 2:
        raise ValueError("trajectory_group split needs at least 2 distinct groups")
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(unique), generator=g).tolist()
    counts = {u: 0 for u in unique}
    for lab in labels:
        counts[lab] += 1
    train_groups: set = set()
    n_train = 0
    for gi in order:
        u = unique[gi]
        if n_train >= train_fraction * samples.n:
            break
        train_groups.add(u)
        n_train += counts[u]
    if len(train_groups) == len(unique):  # keep at least one group held out
        train_groups.discard(unique[order[-1]])
    train_mask = torch.tensor([lab in train_groups for lab in labels], dtype=torch.bool)
    held_mask = ~train_mask
    _require_both_sides(train_mask, held_mask, "trajectory_group")
    info = {
        "method": "trajectory_group",
        "n_groups": len(unique),
        "n_train_groups": len(train_groups),
        "n_held_groups": len(unique) - len(train_groups),
        "n_train": int(train_mask.sum()),
        "n_held": int(held_mask.sum()),
    }
    idx = torch.arange(samples.n)
    return samples.subset(idx[train_mask]), samples.subset(idx[held_mask]), info


def split_uq_samples_by_config(
    samples: UQSamples,
    split_cfg: dict | None,
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
    groups=None,
) -> tuple[UQSamples, UQSamples, dict]:
    """Dispatch a train/held split from a ``data.split`` config block (fail-closed).

    ``split_cfg`` is ``{"method": <one of UQ_SPLIT_METHODS>, ...method params...}``; ``None`` or
    a missing method means the legacy point-level random split. The returned info dict records
    the method and its parameters -- callers MUST stamp it into their report/manifest so no
    calibration table circulates without its split regime (R2WP-7).
    """

    cfg = dict(split_cfg or {})
    method = str(cfg.pop("method", "random")).lower()
    if method not in UQ_SPLIT_METHODS:
        raise ValueError(f"unknown split method {method!r}; must be one of {UQ_SPLIT_METHODS}")
    if method == "random":
        _reject_unknown(cfg, set(), "data.split(random)")
        train, held = split_uq_samples(samples, train_fraction=train_fraction, seed=seed)
        info = {
            "method": "random",
            "train_fraction": float(train_fraction),
            "n_train": train.n,
            "n_held": held.n,
        }
        return train, held, info
    if method == "altitude_disjoint":
        _reject_unknown(cfg, {"held_quantile", "buffer"}, "data.split(altitude_disjoint)")
        return split_uq_samples_altitude_disjoint(
            samples,
            held_quantile=tuple(cfg.get("held_quantile", (0.0, 0.3))),
            buffer=float(cfg.get("buffer", 0.02)),
            seed=seed,
        )
    if method == "angular_block":
        _reject_unknown(cfg, {"n_blocks", "buffer_deg"}, "data.split(angular_block)")
        return split_uq_samples_angular_block(
            samples,
            train_fraction=train_fraction,
            n_blocks=int(cfg.get("n_blocks", 12)),
            buffer_deg=float(cfg.get("buffer_deg", 0.0)),
            seed=seed,
        )
    # trajectory_group
    _reject_unknown(cfg, set(), "data.split(trajectory_group)")
    if groups is None:
        groups = (samples.metadata or {}).get("groups")
    if groups is None:
        raise ValueError(
            "split method 'trajectory_group' needs group labels: pass groups= or load a CSV "
            "with a group/trajectory-id column"
        )
    return split_uq_samples_trajectory_group(
        samples, groups, train_fraction=train_fraction, seed=seed
    )


def _reject_unknown(cfg: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} key(s) {unknown}; valid keys: {sorted(allowed)}")


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
