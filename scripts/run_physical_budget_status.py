"""VESP-UQ physical acceleration-budget screening status (WP11).

Reports, per config, whether model->m/s^2 scaling metadata is present and -- when it is -- runs the
absolute force-error budget screen at a physical tolerance, reporting alarms / fraction / false
positives-negatives. Degenerate alarm sets are flagged as implemented-but-not-activated. No
physical scaling is invented.

    python scripts/run_physical_budget_status.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --tolerance-m-s2 1e-8 --out outputs/physical_budget_status/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.physical_budget_status import DEFAULT_TOLERANCE_M_S2, run_physical_budget_status


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ physical-budget screening status (WP11).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-m-s2", type=float, default=DEFAULT_TOLERANCE_M_S2)
    parser.add_argument("--out", default="outputs/physical_budget_status/")
    args = parser.parse_args(argv)

    configs = []
    for path in args.configs:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        configs.append(cfg)

    result = run_physical_budget_status(
        configs, seed=args.seed, tolerance_m_s2=args.tolerance_m_s2, out_dir=args.out,
    )
    for r in result["rows"]:
        status = "activated" if r.get("operationally_activated") else ("not activated" if not r["activated"] else "degenerate")
        print(f"{r['band']}: {status}")
    print(f"saved_physical_budget_status: {Path(args.out) / 'physical_budget_status.md'}")


if __name__ == "__main__":
    main()
