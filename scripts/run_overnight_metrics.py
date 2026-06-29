"""Resumable overnight driver for the VESP-UQ benchmark suite.

The benchmark suite (:mod:`vesp.uq.suite`) computes every ``(config, seed)`` run, holds them all in
memory, and writes the journal tables only at the very end. For a long real-data sweep (many seeds x
20k orbits x two altitude bands) that means a crash at 4am loses the whole night's compute. This
driver wraps the exact same per-run computation and aggregation but **checkpoints every
``(config, seed)`` to disk as it finishes**, so re-running with the same ``--out`` resumes from where
it stopped instead of restarting.

Layout under ``--out``::

    <out>/
      L60_seed_000/  run.pt  run_meta.json  done.ok          # completed run
      L60_seed_001/  failed.txt                              # crashed run (skipped, reported)
      ...
      heartbeat.json                                         # last seed, elapsed, ETA, git, memory
      overnight.log                                          # append-only progress log
      benchmark_summary.md, calibration_summary.md, ...      # final aggregated journal artifacts

A run dir with ``done.ok`` is reloaded (not recomputed); a dir with only ``failed.txt`` is retried.
The final aggregation tolerates a partial set, so a sweep that loses one seed still produces every
table from the seeds that did complete. Everything targets trajectory true FORCE-model error.

Examples
--------
    # smoke wiring check
    python scripts/run_overnight_metrics.py --configs configs/vespuq/vespuq_smoke.yaml --quick \
        --out outputs/overnight/smoke

    # the real overnight sweep (resume by re-running the same command after a crash)
    python scripts/run_overnight_metrics.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 --n-orbits 10000 \
        --out outputs/overnight/L60_L90_10k_5seed
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from vesp.common.config import load_config
from vesp.uq.suite import (
    DEFAULT_FRACTIONS,
    DEFAULT_SELECTORS,
    aggregate_and_write,
    band_label,
    compute_run,
    git_commit_hash,
)

_QUICK_SEEDS = (0, 1)
_QUICK_N_ORBITS = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_mb() -> float | None:
    """Best-effort resident-set size in MiB (None when psutil is unavailable)."""

    try:
        import psutil
    except Exception:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _load_configs(paths, *, quick: bool, n_orbits: int | None) -> list[dict]:
    configs = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        screen = cfg.setdefault("uq", {}).setdefault("screening", {})
        if quick:
            screen["n_orbits"] = min(int(screen.get("n_orbits", _QUICK_N_ORBITS)), _QUICK_N_ORBITS)
        elif n_orbits is not None:
            screen["n_orbits"] = int(n_orbits)
        configs.append(cfg)
    return configs


def _run_dir_name(band: str, seed: int, *, index: int, unique_bands: bool) -> str:
    """Per-run checkpoint dir; prefix with the config index when two configs share a band label."""

    prefix = band if unique_bands else f"cfg{index}_{band}"
    return f"{prefix}_seed_{seed:03d}"


class _Tee:
    """Mirror progress lines to stdout and an append-only log file."""

    def __init__(self, log_path: Path):
        self._fh = open(log_path, "a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"{_now_iso()} {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _write_heartbeat(out_dir: Path, payload: dict) -> None:
    tmp = out_dir / "heartbeat.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "heartbeat.json")


def run_overnight(
    configs: list[dict],
    *,
    seeds,
    rerun_fractions,
    selectors,
    out_dir: Path,
    primary_fraction: float,
    make_plots: bool,
) -> dict:
    """Compute every ``(config, seed)`` with on-disk checkpointing, then aggregate the completed set."""

    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Tee(out_dir / "overnight.log")
    seeds = [int(s) for s in seeds]
    bands = [band_label(c) for c in configs]
    unique_bands = len(set(bands)) == len(bands)
    git = git_commit_hash()

    jobs = [(i, cfg, seed) for i, cfg in enumerate(configs) for seed in seeds]
    total = len(jobs)
    log(f"[overnight] start: {total} runs ({len(configs)} configs x {len(seeds)} seeds) "
        f"| git={git} | out={out_dir}")

    runs: list[dict] = []
    completed_keys: list[tuple[int, int]] = []  # (config_index, seed) of runs that produced a result
    suite_t0 = time.perf_counter()
    n_done = n_loaded = n_failed = 0

    for job_idx, (cfg_index, cfg, seed) in enumerate(jobs, start=1):
        band = bands[cfg_index]
        run_dir = out_dir / _run_dir_name(band, seed, index=cfg_index, unique_bands=unique_bands)
        run_dir.mkdir(parents=True, exist_ok=True)
        done_marker = run_dir / "done.ok"
        ckpt = run_dir / "run.pt"

        if done_marker.exists() and ckpt.exists():
            run = torch.load(ckpt, map_location="cpu", weights_only=False)
            runs.append(run)
            completed_keys.append((cfg_index, seed))
            n_loaded += 1
            log(f"[overnight] {job_idx}/{total} resume ({band} seed={seed}) -- loaded checkpoint")
        else:
            # stale/failed dir: clear the failure marker before retrying
            (run_dir / "failed.txt").unlink(missing_ok=True)
            t0 = time.perf_counter()
            try:
                run = compute_run(cfg, seed=seed, rerun_fractions=rerun_fractions,
                                  selectors=selectors, reference_fraction=primary_fraction)
            except Exception:  # noqa: BLE001 -- one bad seed must not abort the night
                tb = traceback.format_exc()
                (run_dir / "failed.txt").write_text(
                    f"{_now_iso()}\nband={band} seed={seed}\n\n{tb}", encoding="utf-8")
                n_failed += 1
                log(f"[overnight] {job_idx}/{total} FAILED ({band} seed={seed}): "
                    f"{tb.strip().splitlines()[-1]}")
                _write_heartbeat(out_dir, _heartbeat_payload(
                    out_dir, job_idx, total, band, seed, suite_t0, git,
                    n_done, n_loaded, n_failed, status="failed"))
                continue
            dt = time.perf_counter() - t0
            # atomic-ish checkpoint: write payload, then the done marker last
            torch.save(run, ckpt)
            (run_dir / "run_meta.json").write_text(json.dumps({
                "band": run["band"], "seed": run["seed"], "config_path": run.get("config_path"),
                "n_trajectories": run["n_trajectories"], "n_points": run["n_points"],
                "true_force_error_source": run["true_force_error_source"],
                "timings": run["timings"], "elapsed_seconds": dt, "git_commit": git,
            }, indent=2), encoding="utf-8")
            done_marker.write_text(_now_iso(), encoding="utf-8")
            runs.append(run)
            completed_keys.append((cfg_index, seed))
            n_done += 1
            log(f"[overnight] {job_idx}/{total} done ({band} seed={seed}, {dt:.0f}s)")

        _write_heartbeat(out_dir, _heartbeat_payload(
            out_dir, job_idx, total, band, seed, suite_t0, git,
            n_done, n_loaded, n_failed, status="running"))

    if not runs:
        log("[overnight] no runs completed -- nothing to aggregate")
        log.close()
        raise SystemExit("all runs failed; see failed.txt under each run dir")

    # Aggregate only over the seeds that actually completed (per band), so the placebo chance-band
    # tolerance reflects the real sample size.
    completed_seeds = sorted({seed for (_, seed) in completed_keys})
    log(f"[overnight] aggregating {len(runs)} runs over seeds {completed_seeds} "
        f"(done={n_done} resumed={n_loaded} failed={n_failed})")
    result = aggregate_and_write(
        runs,
        configs=configs,
        seeds=completed_seeds,
        rerun_fractions=rerun_fractions,
        selectors=selectors,
        primary_fraction=primary_fraction,
        out_dir=out_dir,
        make_plots=make_plots,
    )
    log(f"[overnight] wrote journal artifacts to {out_dir}")
    log.close()
    return result


def _heartbeat_payload(out_dir, job_idx, total, band, seed, suite_t0, git,
                       n_done, n_loaded, n_failed, *, status: str) -> dict:
    elapsed = time.perf_counter() - suite_t0
    processed = max(1, job_idx)
    eta = elapsed / processed * (total - job_idx)
    return {
        "updated_at": _now_iso(),
        "status": status,
        "last_band": band,
        "last_seed": seed,
        "completed": job_idx,
        "total": total,
        "n_computed": n_done,
        "n_resumed": n_loaded,
        "n_failed": n_failed,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "memory_mb": _memory_mb(),
        "git_commit": git,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Resumable overnight VESP-UQ benchmark driver.")
    parser.add_argument("--configs", nargs="+", required=True, help="one or more VESP-UQ YAML configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rerun-fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--out", required=True, help="run directory (re-run the same --out to resume)")
    parser.add_argument("--primary-fraction", type=float, default=0.20)
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override uq.screening.n_orbits for all configs (speed vs power)")
    parser.add_argument("--quick", action="store_true", help="2 seeds + capped n_orbits (wiring check)")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG plots (CSV/MD only)")
    args = parser.parse_args(argv)

    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds
    configs = _load_configs(args.configs, quick=args.quick, n_orbits=args.n_orbits)

    result = run_overnight(
        configs,
        seeds=seeds,
        rerun_fractions=args.rerun_fractions,
        selectors=args.selectors,
        out_dir=Path(args.out),
        primary_fraction=args.primary_fraction,
        make_plots=not args.no_plots,
    )
    meta = result["meta"]
    if meta["missing_selectors"]:
        print(f"warning: selectors not available on this plugin/config: {meta['missing_selectors']}",
              file=sys.stderr)
    print(f"bands={meta['bands']} seeds={meta['seeds']} primary={meta['primary_fraction']:.0%} "
          f"git={meta['git_commit']}")
    print(f"saved_overnight_metrics: {Path(args.out) / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
