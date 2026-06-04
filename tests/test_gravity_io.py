from pathlib import Path

import numpy as np
import pytest

from experimental_vesp.gravity_io import read_shadr_ascii


def _write_table(path: Path, rows: list[str]) -> Path:
    text = "\n".join(rows) + "\n"
    path.write_text(text, encoding="ascii")
    return path


def test_read_shadr_ascii_degree_order(tmp_path):
    path = _write_table(
        tmp_path / "mini_sha.tab",
        [
            " 0.1738000000000000E+04, 0.4902800126160000E+04, 0.0, 2, 2, 1, 0, 0",
            " 1, 0, 1.0E-3, 0.0",
            " 1, 1, 2.0E-3, 3.0E-3",
            " 2, 0, 4.0E-3, 0.0",
            " 2, 1, 5.0E-3, 6.0E-3",
            " 2, 2, 7.0E-3, 8.0E-3",
        ],
    )
    table = read_shadr_ascii(path, max_degree=2, strict=True)
    assert table.degree == 2
    assert table.order == 2
    assert table.column_order == "degree_order"
    assert np.isclose(table.c[2, 2], 7.0e-3)
    assert np.isclose(table.s[2, 2], 8.0e-3)


def test_read_shadr_ascii_order_degree_auto_detect(tmp_path):
    path = _write_table(
        tmp_path / "mini_swapped.sha",
        [
            "1738.0, 4902.80012616, 0.0, 2, 2, 1, 0, 0",
            "0, 1, 1.0E-3, 0.0",
            "1, 1, 2.0E-3, 3.0E-3",
            "0, 2, 4.0E-3, 0.0",
            "1, 2, 5.0E-3, 6.0E-3",
            "2, 2, 7.0E-3, 8.0E-3",
        ],
    )
    table = read_shadr_ascii(path, max_degree=2, strict=True)
    assert table.column_order == "order_degree"
    assert np.isclose(table.c[2, 1], 5.0e-3)
    assert np.isclose(table.s[2, 1], 6.0e-3)


def test_read_shadr_ascii_strict_rejects_incomplete_table(tmp_path):
    path = _write_table(
        tmp_path / "incomplete.tab",
        [
            "1738.0, 4902.80012616, 0.0, 2, 2, 1, 0, 0",
            "1, 0, 1.0E-3, 0.0",
            "1, 1, 2.0E-3, 3.0E-3",
            "2, 0, 4.0E-3, 0.0",
            "2, 1, 5.0E-3, 6.0E-3",
        ],
    )
    with pytest.raises(ValueError, match="incomplete"):
        read_shadr_ascii(path, max_degree=2, strict=True)
