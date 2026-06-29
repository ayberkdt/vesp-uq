"""One-command paper evidence run: benchmark suite -> GP baseline -> journal report + figures.

Runs the three heavy stages SEQUENTIALLY (concurrent multi-threaded torch oversubscribes cores and
thrashes) on the real GRAIL L60/L90 residual with the tuned configs. Fire-and-forget: each stage is
isolated so one failure does not lose the others, everything is logged to ``outputs/full_run.log``,
and a final summary lists what was produced. No metric is invented -- every number traces to a study
CSV under a checksummed manifest.

    python scripts/run_full_paper_run.py

Sized to finish well under ~2 hours on a typical multi-core CPU; tune with the flags below.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L60 = ROOT / "configs" / "vespuq" / "vespuq_real_lunar.yaml"
L90 = ROOT / "configs" / "vespuq" / "vespuq_real_lunar_L90.yaml"


def _load(paths):
    from vesp.common.config import load_config

    cfgs = []
    for p in paths:
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        cfgs.append(cfg)
    return cfgs


def _log(msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Full VESP-UQ paper evidence run (sequential).")
    parser.add_argument("--out", default="outputs", help="outputs root")
    parser.add_argument("--suite-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gp-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-orbits", type=int, default=3000,
                        help="screening orbits per run (caps cost; same for suite + GP)")
    parser.add_argument("--skip-suite", action="store_true")
    parser.add_argument("--skip-gp", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args(argv)

    # headless plotting; keep torch from oversubscribing if the user did not set it
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "full_run.log"
    t_start = time.perf_counter()
    _log(f"START full paper run | suite_seeds={args.suite_seeds} gp_seeds={args.gp_seeds} "
         f"n_orbits={args.n_orbits}", log_path)

    stages: list[tuple[str, str]] = []  # (stage, status)

    # ---- Stage 1: benchmark suite (ranking + significance + decision + calibration) ----
    if not args.skip_suite:
        try:
            from vesp.uq.suite import run_suite

            t0 = time.perf_counter()
            _log("STAGE 1/3 benchmark suite (L60+L90) starting ...", log_path)
            run_suite(
                _load([L60, L90]),
                seeds=tuple(args.suite_seeds),
                n_orbits=args.n_orbits,
                out_dir=out_root / "benchmark_suite",
                make_plots=True,
                progress=True,
            )
            _log(f"STAGE 1 done in {(time.perf_counter() - t0) / 60:.1f} min "
                 f"-> {out_root / 'benchmark_suite'}", log_path)
            stages.append(("benchmark_suite", "ok"))
        except Exception as exc:  # isolate: a stage failure must not lose the others
            _log(f"STAGE 1 FAILED: {exc!r}", log_path)
            stages.append(("benchmark_suite", f"FAILED: {exc!r}"))

    # ---- Stage 2: VESP-UQ vs GP baseline ----
    if not args.skip_gp:
        try:
            from vesp.uq.uq_baseline_comparison import run_uq_baseline_comparison

            t0 = time.perf_counter()
            _log("STAGE 2/3 GP baseline comparison starting ...", log_path)
            run_uq_baseline_comparison(
                _load([L60, L90]),
                seeds=tuple(args.gp_seeds),
                n_orbits=args.n_orbits,
                out_dir=out_root / "uq_baseline_comparison",
            )
            _log(f"STAGE 2 done in {(time.perf_counter() - t0) / 60:.1f} min "
                 f"-> {out_root / 'uq_baseline_comparison'}", log_path)
            stages.append(("uq_baseline_comparison", "ok"))
        except Exception as exc:
            _log(f"STAGE 2 FAILED: {exc!r}", log_path)
            stages.append(("uq_baseline_comparison", f"FAILED: {exc!r}"))

    # ---- Stage 3: journal report + LaTeX tables + paper figures ----
    if not args.skip_report:
        try:
            from vesp.uq.figures import render_paper_figures
            from vesp.uq.journal_report import write_report

            t0 = time.perf_counter()
            _log("STAGE 3/3 journal report + figures starting ...", log_path)
            result = write_report(out_root, out_dir=out_root / "journal")
            render_paper_figures(
                benchmark_dir=out_root / "benchmark_suite",
                baseline_dir=out_root / "uq_baseline_comparison",
                out_dir=out_root / "journal" / "figures",
            )
            _log(f"STAGE 3 done in {(time.perf_counter() - t0) / 60:.1f} min | "
                 f"verdict={result['verdict']['overall']} | "
                 f"available={result['available']} | pending={result['missing']}", log_path)
            stages.append(("journal_report", "ok"))
        except Exception as exc:
            _log(f"STAGE 3 FAILED: {exc!r}", log_path)
            stages.append(("journal_report", f"FAILED: {exc!r}"))

    total_min = (time.perf_counter() - t_start) / 60
    _log("=" * 60, log_path)
    _log(f"FULL RUN COMPLETE in {total_min:.1f} min", log_path)
    for name, status in stages:
        _log(f"  {name}: {status}", log_path)
    _log("Key outputs:", log_path)
    _log(f"  {out_root / 'benchmark_suite' / 'benchmark_summary.md'}", log_path)
    _log(f"  {out_root / 'benchmark_suite' / 'significance_summary.md'}", log_path)
    _log(f"  {out_root / 'benchmark_suite' / 'decision_quality.md'}", log_path)
    _log(f"  {out_root / 'benchmark_suite' / 'calibration_summary.md'}", log_path)
    _log(f"  {out_root / 'uq_baseline_comparison' / 'uq_baseline_comparison.md'}", log_path)
    _log(f"  {out_root / 'journal' / 'journal_validation_report.md'}", log_path)


if __name__ == "__main__":
    main()
