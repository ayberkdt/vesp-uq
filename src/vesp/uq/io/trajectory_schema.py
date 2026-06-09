"""Schema for externally supplied surrogate trajectory ensembles.

Two CSV formats are recognized (see :mod:`vesp.uq.io.trajectory_loader`):

- **Format A (positions only)** -- columns ``trajectory_id, t, x, y, z``. Enough for
  post-processing risk scoring of surrogate-generated trajectories.
- **Format B (positions + acceleration pairs)** -- Format A plus
  ``ax_sur, ay_sur, az_sur, ax_ref, ay_ref, az_ref``. Can both *fit* the residual-force error
  (``residual = reference - surrogate``) and be scored.

Units default to verbatim (typically normalized body radii / model-normalized acceleration). When
explicit metadata is supplied, :func:`~vesp.uq.io.trajectory_loader.load_trajectory_csv` can convert
positions (via a ``PositionScaler``) and physical accelerations (via an ``AccelerationScale``) into
the model's working units; ``metadata["units"]`` records exactly what conversion was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Required / optional columns. Aliases keep the loader tolerant of minor header variants.
ID_COLUMN = "trajectory_id"
TIME_COLUMN = "t"
POSITION_COLUMNS = ("x", "y", "z")
SURROGATE_COLUMNS = ("ax_sur", "ay_sur", "az_sur")
REFERENCE_COLUMNS = ("ax_ref", "ay_ref", "az_ref")

REQUIRED_BASE_COLUMNS = (ID_COLUMN, *POSITION_COLUMNS)  # t is optional but recommended

_ID_ALIASES = (ID_COLUMN, "traj_id", "trajectory", "id")
_TIME_ALIASES = (TIME_COLUMN, "time", "time_s", "t_s")
_POS_ALIASES = {"x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z")}
_SUR_ALIASES = {
    "ax_sur": ("ax_sur", "a_x_sur", "ax_surrogate"),
    "ay_sur": ("ay_sur", "a_y_sur", "ay_surrogate"),
    "az_sur": ("az_sur", "a_z_sur", "az_surrogate"),
}
_REF_ALIASES = {
    "ax_ref": ("ax_ref", "a_x_ref", "ax_reference"),
    "ay_ref": ("ay_ref", "a_y_ref", "ay_reference"),
    "az_ref": ("az_ref", "a_z_ref", "az_reference"),
}


@dataclass
class TrajectoryDataset:
    """An ensemble of externally generated surrogate trajectories.

    ``trajectories[i]`` is an ``(N_i, 3)`` position tensor (variable length allowed). When the CSV
    carries acceleration pairs, ``surrogate_accelerations`` / ``reference_accelerations`` /
    ``residual_accelerations`` (``reference - surrogate``) are populated and aligned per
    trajectory; otherwise they are ``None``. ``times[i]`` is the per-point time column (or
    ``None`` if no time column was present).
    """

    trajectories: list[torch.Tensor]
    trajectory_ids: list
    times: list[torch.Tensor] | None = None
    surrogate_accelerations: list[torch.Tensor] | None = None
    reference_accelerations: list[torch.Tensor] | None = None
    residual_accelerations: list[torch.Tensor] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_points(self) -> int:
        return int(sum(int(t.shape[0]) for t in self.trajectories))

    @property
    def has_accelerations(self) -> bool:
        return self.surrogate_accelerations is not None and self.reference_accelerations is not None
