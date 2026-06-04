"""Strict PDS SHADR/SHA gravity-model loading.

This is a dependency-light adaptation of the LUNAR_SIMULATION gravity I/O
architecture. It keeps parsing and validation separate from residual-data
generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_NUM_DELIM_TRANS = str.maketrans({",": " ", "\t": " "})


@dataclass(frozen=True)
class GravityCoefficientTable:
    name: str
    source_path: str
    reference_radius_km: float
    gm_km3_s2: float
    degree: int
    order: int
    normalization_state: int | None
    column_order: str
    c: np.ndarray
    s: np.ndarray


def _parse_nums(line: str) -> list[float]:
    text = line.replace("D", "E").replace("d", "E").translate(_NUM_DELIM_TRANS).strip()
    if not text:
        return []
    try:
        return [float(part) for part in text.split()]
    except ValueError:
        return [float(match.group(0).replace("D", "E").replace("d", "E")) for match in _FLOAT_RE.finditer(line)]


def _infer_column_swap(samples: Iterable[tuple[int, int]]) -> bool:
    pairs = list(samples)
    if not pairs:
        return False
    violations_nm = sum(first < second for first, second in pairs)
    violations_mn = sum(second < first for first, second in pairs)
    return violations_mn < violations_nm


def _is_integer_like(value: float) -> bool:
    return abs(value - round(value)) < 1.0e-9


def _parse_header(line: str) -> tuple[float, float, int, int, int | None]:
    nums = _parse_nums(line)
    if len(nums) < 5:
        raise ValueError("SHADR header requires at least radius, GM, omega, degree, and order fields")
    reference_radius_km = float(nums[0])
    gm_km3_s2 = float(nums[1])
    degree = int(nums[3])
    order = int(nums[4])
    normalization_state = int(nums[5]) if len(nums) > 5 and _is_integer_like(nums[5]) else None
    if reference_radius_km <= 0.0:
        raise ValueError(f"invalid reference radius: {reference_radius_km}")
    if gm_km3_s2 <= 0.0:
        raise ValueError(f"invalid GM: {gm_km3_s2}")
    if degree < 0 or order < 0:
        raise ValueError(f"invalid degree/order: {degree}/{order}")
    return reference_radius_km, gm_km3_s2, degree, order, normalization_state


def read_shadr_ascii(
    path: str | Path,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    coeff_start_line: int = 3,
    sample_size: int = 1000,
    strict: bool = True,
    require_normalization_state: int | None = 1,
) -> GravityCoefficientTable:
    """Read a PDS SHADR/SHA ASCII gravity table.

    The parser auto-detects whether the first two coefficient columns are
    degree/order or order/degree, accepts D-style exponents, and in strict mode
    checks triangular indexing, duplicates, and coefficient coverage.
    """

    path = Path(path)
    if coeff_start_line <= 0:
        raise ValueError("coeff_start_line must be >= 1")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_degree is not None and int(max_degree) < 0:
        raise ValueError("max_degree must be >= 0")
    if not path.is_file():
        raise FileNotFoundError(f"gravity model not found: {path}")

    header_seen = False
    reference_radius_km = 0.0
    gm_km3_s2 = 0.0
    file_degree = 0
    file_order = 0
    normalization_state: int | None = None
    n_use = 0
    c: np.ndarray | None = None
    s: np.ndarray | None = None
    seen: np.ndarray | None = None
    buffered: list[tuple[int, int, float, float, int]] = []
    swap: bool | None = None
    stored_count = 0

    def maybe_fail(message: str) -> None:
        if strict:
            raise ValueError(message)

    def store(raw1: int, raw2: int, val_c: float, val_s: float, line_no: int) -> None:
        nonlocal stored_count
        assert c is not None and s is not None
        n, m = (raw2, raw1) if swap else (raw1, raw2)
        if n > n_use:
            return
        if n < 0 or m < 0 or m > n or m > n_use:
            maybe_fail(f"invalid triangular index at line {line_no}: n={n}, m={m}, n_use={n_use}")
            return
        if strict:
            assert seen is not None
            if seen[n, m]:
                raise ValueError(f"duplicate coefficient for n={n}, m={m} at line {line_no}")
            seen[n, m] = 1
            stored_count += 1
        c[n, m] = float(val_c)
        s[n, m] = float(val_s)

    with path.open("r", encoding="ascii", errors="strict" if strict else "ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not header_seen:
                if not line.strip():
                    continue
                reference_radius_km, gm_km3_s2, file_degree, file_order, normalization_state = _parse_header(line)
                if (
                    strict
                    and require_normalization_state is not None
                    and normalization_state is not None
                    and int(normalization_state) != int(require_normalization_state)
                ):
                    raise ValueError(
                        f"unsupported normalization_state={normalization_state}; "
                        f"expected {require_normalization_state}"
                    )
                degree_cap = max(file_degree, file_order)
                n_use = min(degree_cap, int(max_degree)) if max_degree is not None else degree_cap
                c = np.zeros((n_use + 1, n_use + 1), dtype=np.float64)
                s = np.zeros_like(c)
                seen = np.zeros_like(c, dtype=np.uint8) if strict else None
                header_seen = True
                continue

            if line_no < coeff_start_line:
                nums = _parse_nums(line)
                if len(nums) < 4 or not (_is_integer_like(nums[0]) and _is_integer_like(nums[1])):
                    continue
                coeff_start_line = line_no

            if not line.strip():
                continue

            nums = _parse_nums(line)
            if len(nums) < 4:
                maybe_fail(f"malformed coefficient line {line_no}: {line.rstrip()[:160]}")
                continue
            if not (_is_integer_like(nums[0]) and _is_integer_like(nums[1])):
                maybe_fail(f"non-integer coefficient index at line {line_no}: {line.rstrip()[:160]}")
                continue
            raw1 = int(round(nums[0]))
            raw2 = int(round(nums[1]))
            val_c = float(nums[2])
            val_s = float(nums[3])

            if swap is None:
                buffered.append((raw1, raw2, val_c, val_s, line_no))
                if len(buffered) >= min(sample_size, max(64, 3 * (n_use + 1))):
                    swap = _infer_column_swap((a, b) for a, b, _, _, _ in buffered)
                    for item in buffered:
                        store(*item)
                    buffered.clear()
                continue

            store(raw1, raw2, val_c, val_s, line_no)

    if not header_seen or c is None or s is None:
        raise ValueError(f"failed to parse SHADR header: {path}")
    if swap is None:
        swap = _infer_column_swap((a, b) for a, b, _, _, _ in buffered)
        for item in buffered:
            store(*item)
        buffered.clear()

    if strict:
        assert seen is not None
        expected_with_c00 = (n_use + 1) * (n_use + 2) // 2
        has_c00 = bool(seen[0, 0])
        expected = expected_with_c00 if has_c00 else expected_with_c00 - 1
        if stored_count != expected:
            missing_examples: list[tuple[int, int]] = []
            for n in range(n_use + 1):
                for m in range(n + 1):
                    if n == 0 and m == 0 and not has_c00:
                        continue
                    if seen[n, m] == 0:
                        missing_examples.append((n, m))
                        if len(missing_examples) >= 10:
                            break
                if len(missing_examples) >= 10:
                    break
            raise ValueError(
                "SHADR coefficient table incomplete after parsing. "
                f"path={path}, n_use={n_use}, stored={stored_count}, expected={expected}, "
                f"first_missing={missing_examples}"
            )

    return GravityCoefficientTable(
        name=name or path.stem,
        source_path=str(path),
        reference_radius_km=float(reference_radius_km),
        gm_km3_s2=float(gm_km3_s2),
        degree=int(n_use),
        order=min(int(file_order), int(n_use)),
        normalization_state=normalization_state,
        column_order="order_degree" if swap else "degree_order",
        c=np.ascontiguousarray(c, dtype=np.float64),
        s=np.ascontiguousarray(s, dtype=np.float64),
    )


__all__ = ["GravityCoefficientTable", "read_shadr_ascii"]
