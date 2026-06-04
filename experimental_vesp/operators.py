"""Dense operator builders for Stage 1-2 linear VESP solves.

Dense operators are useful for feasibility experiments. They become expensive
quickly: with N_query=8192 and N_source=20000, the joint potential+acceleration
operator has 32768 x 20000 entries. Use chunked/matrix-free optimization for
larger runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels import build_dense_operator
from .sources import SourceGeometry


@dataclass
class OperatorBundle:
    operator: torch.Tensor
    target: torch.Tensor
    column_scale: torch.Tensor | None = None

    def unscale_solution(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.column_scale is None:
            return sigma
        return sigma / self.column_scale


def build_potential_operator(
    x: torch.Tensor,
    sources: SourceGeometry,
    *,
    eps: float = 0.0,
    source_chunk_size: int | None = None,
) -> torch.Tensor:
    return build_dense_operator(
        x,
        sources.positions.to(x.device, dtype=x.dtype),
        sources.weights.to(x.device, dtype=x.dtype),
        source_chunk_size=source_chunk_size,
        softening=eps,
        include_potential=True,
        include_acceleration=False,
    )


def build_acceleration_operator(
    x: torch.Tensor,
    sources: SourceGeometry,
    *,
    eps: float = 0.0,
    sign: float = 1.0,
    source_chunk_size: int | None = None,
) -> torch.Tensor:
    return build_dense_operator(
        x,
        sources.positions.to(x.device, dtype=x.dtype),
        sources.weights.to(x.device, dtype=x.dtype),
        source_chunk_size=source_chunk_size,
        softening=eps,
        include_potential=False,
        include_acceleration=True,
        acceleration_sign=sign,
    )


def stack_targets(
    *,
    potential: torch.Tensor | None,
    acceleration: torch.Tensor | None,
    use_potential: bool,
    use_acceleration: bool,
) -> torch.Tensor:
    rows = []
    if use_potential:
        if potential is None:
            raise ValueError("potential target required")
        rows.append(potential.reshape(-1))
    if use_acceleration:
        if acceleration is None:
            raise ValueError("acceleration target required")
        rows.extend([acceleration[:, axis].reshape(-1) for axis in range(3)])
    if not rows:
        raise ValueError("at least one target block is required")
    return torch.cat(rows)


def build_joint_operator(
    x: torch.Tensor,
    sources: SourceGeometry,
    *,
    potential: torch.Tensor | None = None,
    acceleration: torch.Tensor | None = None,
    use_potential: bool = True,
    use_acceleration: bool = True,
    potential_weight: float = 1.0,
    acceleration_weight: float = 1.0,
    eps: float = 0.0,
    sign: float = 1.0,
    source_chunk_size: int | None = None,
    column_normalize: bool = False,
) -> OperatorBundle:
    operator = build_dense_operator(
        x,
        sources.positions.to(x.device, dtype=x.dtype),
        sources.weights.to(x.device, dtype=x.dtype),
        source_chunk_size=source_chunk_size,
        softening=eps,
        include_potential=use_potential,
        include_acceleration=use_acceleration,
        acceleration_sign=sign,
    )
    target = stack_targets(
        potential=potential,
        acceleration=acceleration,
        use_potential=use_potential,
        use_acceleration=use_acceleration,
    ).to(x.device, dtype=x.dtype)

    row_weights = []
    if use_potential:
        row_weights.append(torch.full((x.shape[0],), float(potential_weight), dtype=x.dtype, device=x.device))
    if use_acceleration:
        row_weights.extend(
            [torch.full((x.shape[0],), float(acceleration_weight), dtype=x.dtype, device=x.device) for _ in range(3)]
        )
    weights = torch.sqrt(torch.cat(row_weights))
    operator = operator * weights.unsqueeze(-1)
    target = target * weights

    column_scale = None
    if column_normalize:
        column_scale = torch.clamp(torch.linalg.norm(operator, dim=0), min=torch.finfo(x.dtype).eps)
        operator = operator / column_scale.unsqueeze(0)

    return OperatorBundle(operator=operator, target=target, column_scale=column_scale)

