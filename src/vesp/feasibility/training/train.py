"""Unified Stage 1-2 training/solve entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from vesp.common.config import load_config
from vesp.core.models import MultiShellDiscreteVESP
from vesp.feasibility.training.train_discrete import run


def run_from_config(config: dict) -> dict:
    model_type = config.get("model", {}).get("type", "discrete")
    if model_type == "multishell":
        return run(config, model_cls=MultiShellDiscreteVESP)
    return run(config)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()

