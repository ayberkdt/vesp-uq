# VESP-UQ Cleanup Plan (DRY / silent-bug follow-ups)

Drafted 2026-07-05 after the `average_ranks` consolidation pass. This document lists the remaining
cleanup items so a follow-up session can implement them directly.

## Already done (context)

The tie-aware average-rank algorithm was duplicated in four places. It is now a single leaf module
`src/vesp/uq/ranking.py` (torch-only), imported by `benchmarking`, `altitude_controlled`,
`gate_diagnostics`, `selection`, and `plugin`. Two inline `torch.unique(...)` rank copies (in
`selection._spearman` and `plugin._spearman_rank_corr`) were deleted. A latent device bug was fixed
in the process: `average_ranks` allocated its scatter target with a CPU-only `torch.empty`, which
broke for CUDA inputs (e.g. `select_reruns` on a cuda score); it now allocates on `v.device`. Full
test suite green.

Suggested order below: **Item 2 → Item 1 → Item 3** (risk-free warm-up, then the real DRY work, then
a science decision that needs the author).

---

## Item 1 — Consolidate Spearman/Pearson wrappers (DRY, medium value, medium risk)

The rank math is now shared, but four correlation wrappers remain. **Their guards differ — do not
blindly merge; each behavior must be preserved.**

| Wrapper | Location | Guard behavior |
|---|---|---|
| `altitude_controlled.pearson` / `spearman` | `src/vesp/uq/altitude_controlled.py:49,64` | nan if `numel<2` or **any** non-finite (no masking); canonical clean style |
| `gate_diagnostics._pearson` / `_rankdata` / `_spearman` | `src/vesp/uq/gate_diagnostics.py:52,67,77` | **masks** non-finite pairwise, requires `>=3` finite pairs |
| `selection._spearman` | `src/vesp/uq/selection.py:63` | aligns `b` to `a.device`; nan if `numel<2` or any non-finite; manual `sqrt` denom |
| `plugin._spearman_rank_corr` (staticmethod) | `src/vesp/uq/plugin.py:480` | **raises** `ValueError` on length mismatch; nan if `numel<2` or any non-finite |

**Approach.** Add `pearson(a, b)` and `spearman(a, b)` cores to `ranking.py` (leaf, torch-only). Keep
each call site as a thin wrapper that applies **its own** guard (mask / `min_n=2`-or-`3` /
device-align / raise) and then delegates to the shared core, so no behavior drifts. Add a cross-impl
parity test. Verify `test_gate_diagnostics`, `test_altitude_controlled`, `test_uq_trajectory`, and
`test_uq_plugin*` stay green.

## Item 2 — Remove dead duplicate import fallback (safe, low value)

`src/vesp/adapters/st_lrps/data/dataset_parameters.py:43-50` — the `try` and `except Exception`
blocks run the **identical** three `from lunaris...` imports, so the `except` can never behave
differently (no-op fallback, `# pragma: no cover`). Introduced verbatim at the ST-LRPS port
(commit `c737ddb`). Either collapse to plain imports, or restore the intended real fallback if one
existed upstream. Confirm against the upstream LUNAR_SIMULATION structure before touching (this is a
ported file).

## Item 3 — Investigate MU_MOON constant discrepancy (INVESTIGATE, do not auto-fix — science decision)

- `src/vesp/common/lunar.py:15` → `MU_MOON_SI = 4_904_869_500_000.0` (4.9048695e12)
- `src/vesp/adapters/st_lrps/paper_evidence/worst_case.py:27` fallback → `4.902800066e12` (standard DE430 GM_moon)
- adapter canonical → `lunaris.common.constants.MU_MOON` (value not yet confirmed)

~0.04% apart. The two subsystems (VESP core vs ST-LRPS adapter) may legitimately differ, but the
`worst_case` **numeric fallback** diverges from the adapter's real value — if that import ever fails,
orbital-period estimates shift silently. Actions:

1. Confirm the `lunaris` `MU_MOON` value.
2. Make the `worst_case` fallback equal the adapter value, or drop the numeric fallback (import-only).
3. Whether core `lunar.py` should adopt the standard 4.9028e12 is a **physics decision** — flag to
   the author; do not change unilaterally.
