"""Train/validation/test split utilities for altitude-aware VESP studies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import ResidualGravityData


@dataclass
class DataSplits:
    train: ResidualGravityData
    val: ResidualGravityData
    test_high: ResidualGravityData | None = None
    test_low: ResidualGravityData | None = None


def _subset(data: ResidualGravityData, mask_or_idx: torch.Tensor) -> ResidualGravityData:
    return data.subset(mask_or_idx)


def _nonempty_indices(mask: torch.Tensor, label: str) -> torch.Tensor:
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        raise ValueError(f"split band '{label}' produced no samples")
    return idx


def random_split(data: ResidualGravityData, *, train_fraction: float = 0.8, seed: int = 0) -> DataSplits:
    generator = torch.Generator().manual_seed(seed)
    n = data.positions.shape[0]
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(train_fraction * n))
    return DataSplits(train=_subset(data, perm[:n_train]), val=_subset(data, perm[n_train:]))


def altitude_band_split(
    data: ResidualGravityData,
    *,
    train_r_range: tuple[float, float],
    val_r_range: tuple[float, float] | None = None,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> DataSplits:
    radii = torch.linalg.norm(data.positions, dim=-1)
    train_mask = (radii >= train_r_range[0]) & (radii <= train_r_range[1])
    band = _subset(data, _nonempty_indices(train_mask, "train"))
    if val_r_range is None or val_r_range == train_r_range:
        return random_split(band, train_fraction=train_fraction, seed=seed)
    val_mask = (radii >= val_r_range[0]) & (radii <= val_r_range[1])
    return DataSplits(train=band, val=_subset(data, _nonempty_indices(val_mask, "val")))


def ood_high_altitude_split(
    data: ResidualGravityData,
    *,
    train_r_range: tuple[float, float],
    test_high_r_range: tuple[float, float],
    train_fraction: float = 0.8,
    seed: int = 0,
) -> DataSplits:
    splits = altitude_band_split(data, train_r_range=train_r_range, train_fraction=train_fraction, seed=seed)
    radii = torch.linalg.norm(data.positions, dim=-1)
    mask = (radii >= test_high_r_range[0]) & (radii <= test_high_r_range[1])
    splits.test_high = _subset(data, _nonempty_indices(mask, "test_high"))
    return splits


def ood_low_altitude_split(
    data: ResidualGravityData,
    *,
    train_r_range: tuple[float, float],
    test_low_r_range: tuple[float, float],
    train_fraction: float = 0.8,
    seed: int = 0,
) -> DataSplits:
    splits = altitude_band_split(data, train_r_range=train_r_range, train_fraction=train_fraction, seed=seed)
    radii = torch.linalg.norm(data.positions, dim=-1)
    mask = (radii >= test_low_r_range[0]) & (radii <= test_low_r_range[1])
    splits.test_low = _subset(data, _nonempty_indices(mask, "test_low"))
    return splits


def make_splits(data: ResidualGravityData, config: dict) -> DataSplits:
    split_cfg = config.get("split", {})
    data_cfg = config.get("data", {})
    split_type = str(split_cfg.get("type", "random")).lower()
    train_fraction = float(split_cfg.get("train_fraction", data_cfg.get("train_fraction", 0.8)))
    seed = int(config.get("seed", data_cfg.get("seed", 0)))
    if split_type == "random":
        return random_split(data, train_fraction=train_fraction, seed=seed)
    if split_type in {"altitude_band", "altitude"}:
        return altitude_band_split(
            data,
            train_r_range=tuple(split_cfg.get("train_r_range", [1.05, 1.50])),
            val_r_range=tuple(split_cfg.get("val_r_range", split_cfg.get("train_r_range", [1.05, 1.50]))),
            train_fraction=train_fraction,
            seed=seed,
        )
    if split_type == "altitude_ood":
        splits = ood_high_altitude_split(
            data,
            train_r_range=tuple(split_cfg.get("train_r_range", [1.05, 1.50])),
            test_high_r_range=tuple(split_cfg.get("test_high_r_range", [1.50, 2.00])),
            train_fraction=train_fraction,
            seed=seed,
        )
        low = split_cfg.get("test_low_r_range")
        if low is not None:
            radii = torch.linalg.norm(data.positions, dim=-1)
            mask = (radii >= low[0]) & (radii <= low[1])
            splits.test_low = _subset(data, _nonempty_indices(mask, "test_low"))
        return splits
    raise ValueError(f"unknown split type: {split_type}")
