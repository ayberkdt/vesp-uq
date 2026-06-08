"""E8 driver: run the source-geometry shootout and write a ranking verdict.

    python scripts/geometry_shootout.py --config configs/experiments/synthetic_geometry_shootout.yaml
    python scripts/geometry_shootout.py --config configs/experiments/synthetic_geometry_shootout.yaml --with-calibration

Runs every geometry through the standard experiment runner (each with lambda_l2: auto so it is
fairly regularized), writes the usual suite artifacts, then ranks geometries by held-out
low-altitude error (`geometry_verdict.md`, `geometry_ranking.csv`). With ``--with-calibration``
it also runs the Stage 3C uncertainty eval on the best + baseline geometry and appends a
per-band calibration table (does better geometry -> better low-altitude calibration?).
"""

from __future__ import annotations

import argparse
import csv
import io
import contextlib
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vesp.common.config import merge_defaults
from vesp.experiments.geometry import _find_baseline, geometry_report
from vesp.experiments.runner import _set_nested, git_commit_hash, load_experiment_config, run_experiment
from vesp.experiments.summarize import write_suite_artifacts


def _write_ranking_csv(path: Path, ranking: list[dict]) -> None:
    from vesp.experiments.geometry import REPORT_COLUMNS

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)


def _overrides_by_name(experiment_cfg: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for spec in experiment_cfg.get("sweep", []):
        if isinstance(spec, dict) and "name" in spec:
            out[str(spec["name"])] = spec.get("set", {})
    return out


def _calibration_for(experiment_cfg: dict, name: str, overrides: dict, out_dir: Path) -> dict | None:
    from vesp.training.uncertainty import run_uncertainty_eval

    cfg = merge_defaults(deepcopy(experiment_cfg["base_config"]))
    for dotted, value in overrides.items():
        _set_nested(cfg, dotted, value)
    cfg.setdefault("uncertainty", {})["hyperparams"] = "evidence"
    cfg["output"] = {"output_dir": str(out_dir), "run_name": name}
    with contextlib.redirect_stdout(io.StringIO()):
        report = run_uncertainty_eval(cfg)
    return report


def _calibration_table(experiment_cfg: dict, ranking: list[dict], rows: list[dict], out_dir: Path) -> str:
    overrides = _overrides_by_name(experiment_cfg)
    best = ranking[0] if ranking else None
    baseline = _find_baseline(rows)
    targets = []
    for tag, row in (("best", best), ("baseline", baseline)):
        if row is not None and row.get("run_name") in overrides:
            targets.append((tag, str(row["run_name"])))
    # de-dup if best == baseline
    seen = set()
    targets = [(t, n) for t, n in targets if not (n in seen or seen.add(n))]
    if not targets:
        return ""
    lines = [
        "",
        "## Calibration link (Stage 3C, evidence) — best vs baseline geometry",
        "",
        "| geometry | band | picp_90 | z_std | rmse | mean_epistemic_std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for tag, name in targets:
        report = _calibration_for(experiment_cfg, name, overrides[name], out_dir)
        if report is None:
            continue
        for band, m in report.get("bands", {}).items():
            lines.append(
                f"| {name} ({tag}) | {band} | {m.get('picp_90', float('nan')):.2f} | "
                f"{m.get('z_std', float('nan')):.2f} | {m.get('rmse', float('nan')):.3e} | "
                f"{m.get('mean_epistemic_std', float('nan')):.3e} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="an E8 geometry shootout experiment YAML")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--with-calibration", action="store_true", help="also run Stage 3C calibration on best+baseline")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    experiment_cfg = load_experiment_config(config_path)
    name = str(experiment_cfg["experiment"]["name"])
    suite_dir = Path(args.output_root) if args.output_root else (ROOT / "outputs" / "suites" / name)
    runs_dir = suite_dir / "runs"

    git_commit = git_commit_hash()
    result = run_experiment(
        experiment_cfg, output_root=runs_dir, config_path=config_path, quick=args.quick, git_commit=git_commit
    )
    write_suite_artifacts(
        suite_dir, result.rows, suite_name=name, experiments=[name], git_commit=git_commit, make_plots=not args.no_plots
    )

    markdown, ranking = geometry_report(result.rows)
    verdict_path = suite_dir / "geometry_verdict.md"
    # write the verdict first so an optional calibration failure can never lose it
    verdict_path.write_text(markdown, encoding="utf-8")
    _write_ranking_csv(suite_dir / "geometry_ranking.csv", ranking)

    if args.with_calibration:
        try:
            markdown += _calibration_table(experiment_cfg, ranking, result.rows, suite_dir / "calibration")
            verdict_path.write_text(markdown, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - calibration is an optional appendix
            print(f"calibration step failed (non-fatal): {exc}")

    # encoding-safe console output (Windows consoles may not be UTF-8)
    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"geometry_verdict: {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
