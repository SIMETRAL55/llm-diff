"""Tests for llmdiff.storage — uses in-memory SQLite."""
import json

import pytest

from llmdiff.storage import get_run, init_db, list_runs, save_run

DB = ":memory:"

SAMPLE_RUN = {
    "run_id": "run_20240325_143022",
    "timestamp": "2024-03-25T14:30:22",
    "yaml_file": "tests/rag.yaml",
    "model": "groq/llama3-70b-8192",
    "test_cases": [
        {
            "id": "tc_1",
            "input": "q",
            "output_v1": "a",
            "output_v2": "b",
            "criteria_results": [],
            "overall_verdict": "neutral",
        }
    ],
    "summary": {
        "total": 1,
        "improved": 0,
        "regressed": 0,
        "neutral": 1,
        "score_delta_avg": 0.0,
    },
}


def test_init_db_is_idempotent():
    init_db(DB)
    init_db(DB)  # calling twice should not raise


def test_save_and_get_run_roundtrip():
    init_db(DB)
    save_run(SAMPLE_RUN, DB)
    result = get_run("run_20240325_143022", DB)
    assert result["run_id"] == SAMPLE_RUN["run_id"]
    assert result["model"] == SAMPLE_RUN["model"]
    assert result["summary"] == SAMPLE_RUN["summary"]
    assert result["test_cases"] == SAMPLE_RUN["test_cases"]


def test_list_runs_returns_correct_summary():
    init_db(DB)
    save_run(SAMPLE_RUN, DB)
    runs = list_runs(DB)
    assert len(runs) >= 1
    match = next(r for r in runs if r["run_id"] == "run_20240325_143022")
    assert match["summary"]["total"] == 1
    assert match["yaml_file"] == "tests/rag.yaml"


def test_list_runs_newest_first():
    init_db(DB)
    run_a = {**SAMPLE_RUN, "run_id": "run_a", "timestamp": "2024-03-25T10:00:00"}
    run_b = {**SAMPLE_RUN, "run_id": "run_b", "timestamp": "2024-03-25T12:00:00"}
    save_run(run_a, DB)
    save_run(run_b, DB)
    runs = list_runs(DB)
    ids = [r["run_id"] for r in runs]
    assert ids.index("run_b") < ids.index("run_a")


def test_list_runs_limit():
    init_db(DB)
    for i in range(5):
        r = {**SAMPLE_RUN, "run_id": f"run_limit_{i}", "timestamp": f"2024-03-25T{10+i:02d}:00:00"}
        save_run(r, DB)
    runs = list_runs(DB, limit=3)
    assert len(runs) <= 3


def test_get_run_unknown_raises_key_error():
    init_db(DB)
    with pytest.raises(KeyError):
        get_run("run_does_not_exist", DB)
