"""Train or solve a multi-shell discrete VESP baseline."""

from __future__ import annotations

import argparse
from typing import Iterable

from .models import MultiShellDiscreteVESP
from .train_discrete import load_config, run


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experimental_vesp/configs/discrete_multishell.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config), model_cls=MultiShellDiscreteVESP)


if __name__ == "__main__":
    main()

