"""Altitude-controlled incremental-value diagnostics for VESP-UQ (WP4, standalone).

Answers the reviewer question "does VESP-UQ add information *beyond* altitude?" with three
altitude-held-fixed diagnostics, aggregated over seeds:

* within-altitude-bin Spearman of each score vs true force error (altitude_bin_ranking.csv/png),
* partial correlation of each score with true force error given min-radius
  (partial_correlation_summary.csv),
* matched-altitude paired sign test (matched_altitude_pairs.csv, matched_altitude_summary.md).

Everything targets trajectory true FORCE-model error, never position error.

    python scripts/run_altitude_controlled.py --config configs/vespuq/vespuq_real_lunar.yaml \
        --seeds 0 1 2 3 4 --out outputs/altitude_controlled/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.suite import (
    ALTITUDE_CONTROL_SELECTORS,
    DEFAULT_FRACTIONS,
    _altitude_bin_csv,
    _matched_pairs_csv,
    _matched_summary_md,
    _partial_correlation_csv,
    _plot_altitude_bins,
    aggregate_altitude_controlled,
    band_label,
    compute_run,
    git_commit_hash,
)

_QUICK_SEEDS = (0, 1)
_QUICK_N_ORBITS = 200


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ altitude-controlled incremental-value diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="outputs/altitude_controlled/")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"config not found: {args.config}")
    cfg = load_config(str(path))
    cfg.setdefault("_config_path", str(path))
    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds
    if args.quick:
        screen = cfg.setdefault("uq", {}).setdefault("screening", {})
        screen["n_orbits"] = min(int(screen.get("n_orbits", _QUICK_N_ORBITS)), _QUICK_N_ORBITS)

    # The altitude-control selectors are a subset; reuse the suite per-seed compute.
    runs = [
        compute_run(cfg, seed=s, rerun_fractions=DEFAULT_FRACTIONS, selectors=ALTITUDE_CONTROL_SELECTORS)
        for s in seeds
    ]
    ac_agg = aggregate_altitude_controlled(runs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if not args.no_plots and _plot_altitude_bins(runs, out_dir / "altitude_bin_ranking.png"):
        artifact_files["altitude_bin_ranking.png"] = out_dir / "altitude_bin_ranking.png"

    write_run_artifacts(
        out_dir,
        tool="run_altitude_controlled",
        config=cfg,
        json_files={"altitude_controlled_meta.json": {
            "band": band_label(cfg),
            "seeds": seeds,
            "git_commit": git_commit_hash(),
            "selectors": list(ALTITUDE_CONTROL_SELECTORS),
        }},
        text_files={
            "altitude_bin_ranking.csv": _altitude_bin_csv(runs),
            "partial_correlation_summary.csv": _partial_correlation_csv(ac_agg),
            "matched_altitude_pairs.csv": _matched_pairs_csv(runs),
            "matched_altitude_summary.md": _matched_summary_md(ac_agg),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    print(_matched_summary_md(ac_agg))
    print(f"saved_altitude_controlled: {out_dir / 'matched_altitude_summary.md'}")


if __name__ == "__main__":
    main()
