from __future__ import annotations

from vesp.uq.readiness import summarize_readiness


def test_summarize_readiness_ready_with_partial_rtn_not_promoted():
    summary, rows = summarize_readiness(
        gate_result={"cases": [{"consistency_rows": [{"status": "ok"}]}]},
        attribution_result={
            "cases": [{"band": "L60"}],
            "masking_rows": [{"top_mask_beats_random": True}, {"top_mask_beats_random": False}],
        },
        rtn_result={"cases": [{"case_decision": "partial"}]},
        geometry_result={"verdict": {"L60": {"best_geometry": "surface_dense"}}},
        verifications=[{"ok": True, "verified": ["a"], "unlisted": []}],
    )

    assert summary["status"] == "ready_for_controlled_result_runs"
    assert summary["rtn_production_promotion"] == "hold"
    assert summary["geometry_auto_selection"] == "ran"
    assert summary["attribution_mask_win_rate"] == 0.5
    statuses = {row["check"]: row["status"] for row in rows}
    assert statuses["geometry_auto_selection"] == "ok"
    assert statuses["prototype_promotion"] == "hold"


def test_summarize_readiness_blocks_on_consistency_warning():
    summary, rows = summarize_readiness(
        gate_result={"cases": [{"consistency_rows": [{"status": "warn"}]}]},
        attribution_result={"cases": [{"band": "L60"}], "masking_rows": [{"top_mask_beats_random": True}]},
        rtn_result={"cases": [{"case_decision": "candidate"}]},
        verifications=[{"ok": True, "verified": ["a"], "unlisted": []}],
    )

    assert summary["status"] == "not_ready"
    assert "gate_diagnostics_abc" in summary["blockers"]
    assert {row["check"]: row["status"] for row in rows}["gate_diagnostics_abc"] == "block"
