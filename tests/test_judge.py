"""Tests for llmdiff.judge — no real API calls."""
from unittest.mock import MagicMock, patch

import pytest

from llmdiff.judge import judge_pair, judge_test_case, judge_run


def _mock_resp(text):
    m = MagicMock()
    m.choices[0].message.content = text
    return m


def _side_effects(*texts):
    return [_mock_resp(t) for t in texts]


@patch("llmdiff.judge.litellm.completion")
def test_unanimous_b_is_improved_high_confidence(mock_comp):
    mock_comp.side_effect = _side_effects("B\nv2 is better.", "B\nv2 is better.", "B\nv2 is better.")
    result = judge_pair("q", "out_v1", "out_v2", "criterion", "groq/llama3-70b-8192")
    assert result["verdict"] == "improved"
    assert result["confidence"] == "high"
    assert result["score_v2"] == 1.0
    assert result["score_v1"] == 0.0
    assert result["delta"] == 1.0


@patch("llmdiff.judge.litellm.completion")
def test_majority_a_is_regressed_medium_confidence(mock_comp):
    mock_comp.side_effect = _side_effects("A\nv1 is better.", "A\nv1 is better.", "B\nv2 is better.")
    result = judge_pair("q", "out_v1", "out_v2", "criterion", "groq/llama3-70b-8192")
    assert result["verdict"] == "regressed"
    assert result["confidence"] == "medium"
    assert result["score_v1"] == 1.0
    assert result["score_v2"] == 0.0
    assert result["delta"] == -1.0


@patch("llmdiff.judge.litellm.completion")
def test_all_different_falls_back_to_tie_low_confidence(mock_comp):
    mock_comp.side_effect = _side_effects("A\nr1", "B\nr2", "tie\nr3")
    result = judge_pair("q", "out_v1", "out_v2", "criterion", "groq/llama3-70b-8192")
    assert result["confidence"] == "low"


@patch("llmdiff.judge.litellm.completion")
def test_tie_scores_are_point_five(mock_comp):
    mock_comp.side_effect = _side_effects("tie\nequal", "tie\nequal", "tie\nequal")
    result = judge_pair("q", "out_v1", "out_v2", "criterion", "groq/llama3-70b-8192")
    assert result["score_v1"] == 0.5
    assert result["score_v2"] == 0.5
    assert result["delta"] == 0.0
    assert result["verdict"] == "neutral"


@patch("llmdiff.judge.litellm.completion")
def test_delta_calculation(mock_comp):
    # B wins: delta should be +1.0
    mock_comp.side_effect = _side_effects("B\nr", "B\nr", "B\nr")
    result = judge_pair("q", "v1", "v2", "criterion", "model")
    assert result["delta"] == pytest.approx(1.0)


@patch("llmdiff.judge.litellm.completion")
def test_judge_test_case_improved_verdict(mock_comp):
    # 3 criteria all returning B (improved)
    mock_comp.side_effect = _side_effects(*["B\nr"] * 9)
    runner_result = {"id": "tc_1", "input": "q", "output_v1": "v1", "output_v2": "v2"}
    result = judge_test_case(runner_result, ["c1", "c2", "c3"], "model")
    assert result["overall_verdict"] == "improved"
    assert len(result["criteria_results"]) == 3


@patch("llmdiff.judge.litellm.completion")
def test_judge_test_case_regressed_verdict(mock_comp):
    # All criteria returning A (regressed)
    mock_comp.side_effect = _side_effects(*["A\nr"] * 9)
    runner_result = {"id": "tc_1", "input": "q", "output_v1": "v1", "output_v2": "v2"}
    result = judge_test_case(runner_result, ["c1", "c2", "c3"], "model")
    assert result["overall_verdict"] == "regressed"


@patch("llmdiff.judge.litellm.completion")
def test_judge_test_case_neutral_verdict(mock_comp):
    # All criteria tie → delta=0 → neutral
    mock_comp.side_effect = _side_effects(*["tie\nr"] * 9)
    runner_result = {"id": "tc_1", "input": "q", "output_v1": "v1", "output_v2": "v2"}
    result = judge_test_case(runner_result, ["c1", "c2", "c3"], "model")
    assert result["overall_verdict"] == "neutral"


@patch("llmdiff.judge.litellm.completion")
def test_judge_run_returns_full_shape(mock_comp):
    mock_comp.side_effect = _side_effects(*["B\nr"] * 30)
    config = {
        "model": "groq/llama3-70b-8192",
        "judge_model": "groq/llama3-70b-8192",
        "yaml_file": "tests.yaml",
        "test_cases": [
            {"id": "tc_1", "criteria": ["c1"]},
            {"id": "tc_2", "criteria": ["c1"]},
        ],
    }
    runner_results = [
        {"id": "tc_1", "input": "q", "output_v1": "v1", "output_v2": "v2"},
        {"id": "tc_2", "input": "q", "output_v1": "v1", "output_v2": "v2"},
    ]
    result = judge_run(runner_results, config)
    assert result["run_id"].startswith("run_")
    assert "timestamp" in result
    assert "summary" in result
    assert result["summary"]["total"] == 2
    assert len(result["test_cases"]) == 2
