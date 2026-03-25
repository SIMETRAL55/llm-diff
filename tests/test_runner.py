"""Tests for llmdiff.runner — no real API calls."""
from unittest.mock import MagicMock, patch

import pytest

from llmdiff.runner import run_test_cases

BASIC_CONFIG = {
    "model": "groq/llama3-70b-8192",
    "test_cases": [
        {
            "id": "tc_001",
            "input": "What is the default chunk size?",
            "context": "The default chunk size is 1000 characters.",
            "prompt_v1": "Context: {context}\nQuestion: {input}",
            "prompt_v2": "Answer concisely. Context: {context}\nQ: {input}",
        }
    ],
}


def _mock_completion(content="Mock answer"):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("llmdiff.runner.litellm.completion")
def test_calls_litellm_with_correct_model(mock_completion):
    mock_completion.return_value = _mock_completion()
    run_test_cases(BASIC_CONFIG)
    calls = mock_completion.call_args_list
    assert all(c.kwargs["model"] == "groq/llama3-70b-8192" for c in calls)


@patch("llmdiff.runner.litellm.completion")
def test_substitutes_input_and_context(mock_completion):
    mock_completion.return_value = _mock_completion()
    run_test_cases(BASIC_CONFIG)
    calls = mock_completion.call_args_list
    # Each test case generates 2 calls (v1 and v2)
    assert len(calls) == 2
    for call in calls:
        content = call.kwargs["messages"][0]["content"]
        assert "What is the default chunk size?" in content
        assert "The default chunk size is 1000 characters." in content


@patch("llmdiff.runner.litellm.completion")
def test_failed_test_case_does_not_crash(mock_completion):
    mock_completion.side_effect = Exception("API error")
    config = {
        "model": "groq/llama3-70b-8192",
        "test_cases": [
            {
                "id": "tc_fail",
                "input": "q",
                "context": "",
                "prompt_v1": "{input}",
                "prompt_v2": "{input}",
            }
        ],
    }
    results = run_test_cases(config)
    assert len(results) == 1
    assert results[0]["id"] == "tc_fail"
    assert "ERROR" in results[0]["output_v1"]


@patch("llmdiff.runner.litellm.completion")
def test_output_shape_matches_schema(mock_completion):
    mock_completion.return_value = _mock_completion("answer text")
    results = run_test_cases(BASIC_CONFIG)
    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"id", "input", "output_v1", "output_v2"}
    assert r["id"] == "tc_001"
    assert r["output_v1"] == "answer text"
    assert r["output_v2"] == "answer text"


@pytest.mark.parametrize("model", [
    "groq/llama3-70b-8192",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku-20240307",
])
@patch("llmdiff.runner.litellm.completion")
def test_model_string_passed_through_unchanged(mock_completion, model):
    mock_completion.return_value = _mock_completion()
    config = {
        "model": model,
        "test_cases": [
            {
                "id": "tc_x",
                "input": "q",
                "context": "",
                "prompt_v1": "{input}",
                "prompt_v2": "{input}",
            }
        ],
    }
    run_test_cases(config)
    for call in mock_completion.call_args_list:
        assert call.kwargs["model"] == model
