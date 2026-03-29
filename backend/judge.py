import logging
import re
import time
from datetime import datetime
from typing import Callable

import litellm

from backend import config as _cfg

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are evaluating two AI-generated responses for a single quality criterion.

Input: {input}

Criterion: {criterion}

Response A (v1):
{output_v1}

Response B (v2):
{output_v2}

Which response better satisfies the criterion? Reply with exactly one of: A, B, or tie
Then on the next line, one sentence explaining your reasoning.
"""

ABSOLUTE_JUDGE_PROMPT = """\
You are evaluating an AI assistant's output against a reference answer.

Question: {input}
Reference answer: {reference_answer}
Output to evaluate: {output}

Does this output correctly and completely answer the question given the reference answer?
Respond with only a number between 0.0 and 1.0, where 1.0 means perfect and 0.0 means completely wrong.
"""


def _parse_verdict(text: str) -> tuple[str, str]:
    """Extract (verdict, reasoning) from a judge model response."""
    lines = text.strip().splitlines()
    verdict = "tie"
    reasoning = ""

    for line in lines:
        token = line.strip().lower()
        if token in ("a", "b", "tie"):
            verdict = token
            break
        # Handle cases like "A." or "**A**" or "verdict: A"
        match = re.search(r"\b(a|b|tie)\b", token)
        if match:
            verdict = match.group(1)
            break

    # Reasoning is everything after the first verdict line
    for i, line in enumerate(lines):
        if re.search(r"\b(a|b|tie)\b", line.strip().lower()):
            reasoning = " ".join(lines[i + 1:]).strip()
            break

    return verdict, reasoning or "No reasoning provided."


def _call_judge(prompt: str, judge_model: str) -> str:
    """Make a single judge model call with retry logic. Returns raw text."""
    for attempt in range(4):
        try:
            response = litellm.completion(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            if _cfg.JUDGE_SLEEP > 0:
                time.sleep(_cfg.JUDGE_SLEEP)
            return text
        except litellm.RateLimitError:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
            logger.warning(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return ""
    return ""


def judge_pair(
    input: str,
    output_v1: str,
    output_v2: str,
    criterion: str,
    judge_model: str,
    reference_answer: str | None = None,
    votes: int = 1,
) -> dict:
    """Judge a single criterion between two outputs.

    When reference_answer is provided, uses absolute scoring mode (0.0–1.0 per output).
    Otherwise uses A/B comparison mode with majority vote (controlled by `votes`).
    votes=1 is fastest; votes=3 gives more reliable results via majority vote.

    Returns a dict matching the criteria_results shape in CLAUDE.md.
    """
    if reference_answer is not None:
        # Absolute scoring mode: score each output independently against the reference
        def _score_output(output: str) -> float:
            prompt = ABSOLUTE_JUDGE_PROMPT.format(
                input=input,
                reference_answer=reference_answer,
                output=output,
            )
            scores = []
            for _ in range(max(1, votes - 1)):
                text = _call_judge(prompt, judge_model)
                try:
                    scores.append(float(text.strip()))
                except (ValueError, AttributeError):
                    scores.append(0.5)
            return sum(scores) / len(scores)

        score_v1 = round(_score_output(output_v1), 4)
        score_v2 = round(_score_output(output_v2), 4)
        delta = round(score_v2 - score_v1, 4)

        if delta > 0.1:
            verdict = "improved"
        elif delta < -0.1:
            verdict = "regressed"
        else:
            verdict = "neutral"

        return {
            "criterion": criterion,
            "score_v1": score_v1,
            "score_v2": score_v2,
            "delta": delta,
            "verdict": verdict,
            "reasoning": f"Absolute scores — v1: {score_v1}, v2: {score_v2}",
            "confidence": "high",
        }

    # A/B comparison mode: majority vote
    prompt = JUDGE_PROMPT.format(
        input=input,
        criterion=criterion,
        output_v1=output_v1,
        output_v2=output_v2,
    )

    verdicts = []
    reasoning_samples = []

    for _ in range(votes):
        text = _call_judge(prompt, judge_model)
        if text:
            verdict, reasoning = _parse_verdict(text)
        else:
            verdict, reasoning = "tie", "Judge call failed."
        verdicts.append(verdict)
        reasoning_samples.append(reasoning)

    # Majority vote
    counts = {"a": verdicts.count("a"), "b": verdicts.count("b"), "tie": verdicts.count("tie")}
    majority = max(counts, key=counts.get)

    if counts["a"] == counts["b"] == counts["tie"]:
        majority = "tie"

    # Confidence
    top_count = counts[majority]
    if top_count == 3:
        confidence = "high"
    elif top_count == 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Map verdict
    verdict_map = {"a": "regressed", "b": "improved", "tie": "neutral"}
    verdict = verdict_map[majority]

    # Scores: winner=1.0, loser=0.0, tie=0.5 each
    if majority == "a":
        score_v1, score_v2 = 1.0, 0.0
    elif majority == "b":
        score_v1, score_v2 = 0.0, 1.0
    else:
        score_v1, score_v2 = 0.5, 0.5

    delta = round(score_v2 - score_v1, 4)

    # Use the reasoning from the majority-agreeing call, or the first one
    majority_idx = next(
        (i for i, v in enumerate(verdicts) if v == majority),
        0,
    )
    reasoning = reasoning_samples[majority_idx]

    return {
        "criterion": criterion,
        "score_v1": score_v1,
        "score_v2": score_v2,
        "delta": delta,
        "verdict": verdict,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def judge_test_case(
    runner_result: dict,
    criteria: list[str],
    judge_model: str,
    votes: int = 1,
    criterion_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Judge all criteria for one test case.

    criterion_callback(index, total, criterion_text) is called before each criterion
    is judged, so the caller can display live progress.

    Returns the full test case result shape with criteria_results and overall_verdict.
    """
    reference_answer = runner_result.get("reference_answer")
    criteria_results = []
    for i, criterion in enumerate(criteria):
        if criterion_callback:
            criterion_callback(i, len(criteria), criterion)
        result = judge_pair(
            input=runner_result["input"],
            output_v1=runner_result["output_v1"],
            output_v2=runner_result["output_v2"],
            criterion=criterion,
            judge_model=judge_model,
            reference_answer=reference_answer,
            votes=votes,
        )
        criteria_results.append(result)

    if criteria_results:
        avg_delta = sum(r["delta"] for r in criteria_results) / len(criteria_results)
    else:
        avg_delta = 0.0

    if avg_delta > 0.1:
        overall_verdict = "improved"
    elif avg_delta < -0.1:
        overall_verdict = "regressed"
    else:
        overall_verdict = "neutral"

    return {
        "id": runner_result["id"],
        "input": runner_result["input"],
        "output_v1": runner_result["output_v1"],
        "output_v2": runner_result["output_v2"],
        "criteria_results": criteria_results,
        "overall_verdict": overall_verdict,
    }


def judge_run(
    runner_results: list[dict],
    config: dict,
    test_case_callback: Callable[[dict], None] | None = None,
    criterion_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Judge all test cases in a run and return the full run result shape.

    test_case_callback(judged_case) is called after each test case is fully judged.
    criterion_callback(index, total, criterion_text) is called before each criterion.
    """
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()
    judge_model = config.get("judge_model", config.get("model"))
    votes = int(config.get("judge_votes", _cfg.JUDGE_VOTES))

    # Build a lookup: test case id → criteria list
    tc_criteria = {tc["id"]: tc.get("criteria", []) for tc in config.get("test_cases", [])}

    judged_cases = []
    for result in runner_results:
        criteria = tc_criteria.get(result["id"], [])
        judged = judge_test_case(result, criteria, judge_model, votes=votes, criterion_callback=criterion_callback)
        judged_cases.append(judged)
        if test_case_callback:
            test_case_callback(judged)

    # Summary
    verdicts = [tc["overall_verdict"] for tc in judged_cases]
    improved = verdicts.count("improved")
    regressed = verdicts.count("regressed")
    neutral = verdicts.count("neutral")

    all_deltas = [
        cr["delta"]
        for tc in judged_cases
        for cr in tc["criteria_results"]
    ]
    score_delta_avg = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "yaml_file": config.get("yaml_file", ""),
        "model": config.get("model", ""),
        "test_cases": judged_cases,
        "summary": {
            "total": len(judged_cases),
            "improved": improved,
            "regressed": regressed,
            "neutral": neutral,
            "score_delta_avg": score_delta_avg,
        },
    }
