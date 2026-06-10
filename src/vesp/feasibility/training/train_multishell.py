"""Legacy wrapper for the unified train entrypoint."""

from __future__ import annotations

from collections.abc import Iterable

from vesp.feasibility.training.train import main as unified_main


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        argv = ["--config", "configs/feasibility/discrete_multishell.yaml"]
    unified_main(argv)


if __name__ == "__main__":
    main()
