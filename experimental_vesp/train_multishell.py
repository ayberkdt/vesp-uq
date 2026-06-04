"""Legacy wrapper for the unified train entrypoint."""

from __future__ import annotations

from typing import Iterable

from .train import main as unified_main


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        argv = ["--config", "configs/discrete_multishell.yaml"]
    unified_main(argv)


if __name__ == "__main__":
    main()
