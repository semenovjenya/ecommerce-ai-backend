from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture_m0_baseline.py"
SPEC = importlib.util.spec_from_file_location("capture_m0_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def test_manifest_is_complete_and_hashes_are_stable() -> None:
    fixtures, errors = baseline.verify_manifest(baseline.DEFAULT_MANIFEST)

    assert errors == []
    assert len(fixtures) == 12
    assert len({fixture["case_id"] for fixture in fixtures}) == 12


def test_run_summary_uses_terminal_denominator_and_percentiles() -> None:
    rows = [
        {"status": "completed", "duration_seconds": 10},
        {"status": "completed", "duration_seconds": 20},
        {"status": "failed", "duration_seconds": 30},
        {"status": "running", "duration_seconds": None},
    ]

    result = baseline.summarize_runs(rows)

    assert result["run_count"] == 4
    assert result["terminal_count"] == 3
    assert result["terminal_success_rate"] == 2 / 3
    assert result["latency_p50_seconds"] == 20
    assert result["latency_p95_seconds"] == 29


def test_human_corpus_fails_closed_for_missing_or_generated_labels(tmp_path: Path) -> None:
    missing_count, missing_errors = baseline.load_human_cases(tmp_path / "missing.jsonl")
    assert missing_count == 0
    assert missing_errors

    corpus = tmp_path / "cases.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "case_id": "generated-1",
                "human_label": {"label_source": "llm", "labeler_id": "model"},
            }
        ),
        encoding="utf-8",
    )
    count, errors = baseline.load_human_cases(corpus)
    assert count == 0
    assert errors == ["invalid human corpus: ManagerEvaluationDataError"]


def test_human_corpus_accepts_real_provenance(tmp_path: Path) -> None:
    corpus = tmp_path / "cases.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": f"case-{index}",
                    "source_ref": f"agent_run:{1000 + index}",
                    "human_label": {
                        "label_source": "human",
                        "labeler_id": "reviewer-1",
                        "labeled_at": "2026-07-24T12:00:00+03:00",
                        "decision": "block",
                        "safety_blockers": ["missing_evidence"],
                        "top_priorities": ["collect_evidence"],
                        "safe_executable_action_ids": [],
                        "unsafe_executable_action_ids": ["unsafe-action"],
                        "hard_guardrail_incident_ids": [],
                    },
                    "system_result": {
                        "decision": "block",
                        "safety_blockers": ["missing_evidence"],
                        "top_priorities": ["collect_evidence"],
                        "executable_action_ids": [],
                        "applied_actions": [],
                    },
                }
            )
            for index in range(40)
        ),
        encoding="utf-8",
    )

    count, errors = baseline.load_human_cases(corpus)

    assert count == 40
    assert errors == []
