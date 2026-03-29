import json
import os
import sys

import yaml
import click
from dotenv import load_dotenv

load_dotenv()

from backend import version
from backend.runner import run_test_cases
from backend.judge import judge_run
from backend import storage
from backend import config as _cfg

DB_PATH = _cfg.DB_PATH


def _check_thresholds(summary: dict, threshold_config: dict) -> tuple[bool, str]:
    """Evaluate threshold config against a run summary. Returns (passed, message)."""
    total = summary["total"]
    regressions = summary["regressed"]
    pass_rate = (summary["improved"] + summary["neutral"]) / total if total > 0 else 1.0

    failures = []
    min_pass_rate = threshold_config.get("min_pass_rate")
    max_regression_count = threshold_config.get("max_regression_count")

    if min_pass_rate is not None and pass_rate < min_pass_rate:
        failures.append(f"pass rate {pass_rate:.0%} < required {min_pass_rate:.0%}")
    if max_regression_count is not None and regressions > max_regression_count:
        failures.append(f"{regressions} regressions > allowed {max_regression_count}")

    if failures:
        return False, f"FAIL: {', '.join(failures)}"
    return True, "✓ All thresholds passed"


@click.group()
@click.version_option(version, "--version", "-V")
def main():
    """LLM Diff — prompt regression testing for RAG pipelines."""


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True))
def run(yaml_file):
    """Run test cases, judge results, and store to history."""
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    config["yaml_file"] = yaml_file
    storage.init_db(DB_PATH)

    click.echo(f"Running {len(config['test_cases'])} test cases with {config['model']}...")
    with click.progressbar(length=len(config["test_cases"]), label="Running") as bar:
        results = []
        for tc in config["test_cases"]:
            result = run_test_cases({"model": config["model"], "test_cases": [tc]})
            results.extend(result)
            bar.update(1)

    click.echo(f"Judging with {config.get('judge_model', config['model'])}...")
    run_result = judge_run(results, config)
    storage.save_run(run_result, DB_PATH)

    s = run_result["summary"]
    click.echo(
        f"Done. {s['improved']} improved · {s['regressed']} regressed · {s['neutral']} neutral"
        f"  (run_id: {run_result['run_id']})"
    )

    if "threshold" in config:
        passed, msg = _check_thresholds(s, config["threshold"])
        click.echo(msg)
        if not passed:
            sys.exit(1)


def _render_test_case(tc: dict) -> None:
    """Render a single test case verdict with per-criterion breakdown."""
    verdict = tc["overall_verdict"]
    criteria_results = tc.get("criteria_results", [])

    if criteria_results:
        avg_delta = sum(r["delta"] for r in criteria_results) / len(criteria_results)
    else:
        avg_delta = 0.0

    improved_count = sum(1 for r in criteria_results if r["verdict"] == "improved")
    regressed_count = sum(1 for r in criteria_results if r["verdict"] == "regressed")
    total_criteria = len(criteria_results)

    if verdict == "improved":
        icon = click.style("✅", fg="green")
        tc_id = click.style(tc["id"], fg="green")
        summary = click.style(
            f"{improved_count}/{total_criteria} criteria improved (+{avg_delta:.2f} avg delta)",
            fg="green",
        )
    elif verdict == "regressed":
        icon = click.style("❌", fg="red")
        tc_id = click.style(tc["id"], fg="red")
        worst = min(criteria_results, key=lambda r: r["delta"])
        summary = click.style(
            f"{regressed_count} criterion regressed ({worst['criterion'][:30]} {avg_delta:+.2f})",
            fg="red",
        )
    else:
        icon = click.style("➡ ", fg="white", dim=True)
        tc_id = click.style(tc["id"], dim=True)
        summary = click.style("no significant change", dim=True)

    click.echo(f"{icon} {tc_id} — {summary}")

    for cr in criteria_results:
        delta = cr["delta"]
        crit_name = cr["criterion"]
        if cr["verdict"] == "improved":
            mark = click.style("  ✓", fg="green")
            detail = click.style(f"{crit_name} ({delta:+.1f})", fg="green")
        elif cr["verdict"] == "regressed":
            mark = click.style("  ✗", fg="red")
            detail = click.style(f"{crit_name} ({delta:+.1f})", fg="red")
        else:
            mark = click.style("  →", dim=True)
            detail = click.style(f"{crit_name} ({delta:+.1f})", dim=True)
        click.echo(f"{mark} {detail}")

    click.echo()


def _render_diff(run_result: dict) -> None:
    """Render a colored terminal diff from a full run result."""
    for tc in run_result["test_cases"]:
        _render_test_case(tc)

    s = run_result["summary"]
    click.echo(click.style("─" * 37, dim=True))
    improved_txt = click.style(f"{s['improved']} improved", fg="green")
    regressed_txt = click.style(f"{s['regressed']} regressed", fg="red")
    neutral_txt = click.style(f"{s['neutral']} neutral", dim=True)
    click.echo(f"{improved_txt}  ·  {regressed_txt}  ·  {neutral_txt}")
    click.echo()
    click.echo("Run `llmdiff serve` to explore results in the dashboard")


def _status(msg: str, *, erase: bool = False) -> None:
    """Print a status line. If erase=True, overwrite it on the next call."""
    if erase:
        # Pad to 80 chars so previous longer line is fully overwritten
        click.echo(f"\r{msg:<80}", nl=False)
    else:
        click.echo(msg)


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option("--ci", is_flag=True, default=False, help="CI mode: suppress rich output, print one-line result + JSON")
def compare(yaml_file, ci):
    """Run test cases and print a colored diff to the terminal."""
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    config["yaml_file"] = yaml_file
    storage.init_db(DB_PATH)

    total = len(config["test_cases"])
    judge_model_name = config.get("judge_model", config["model"])

    # ── Run phase: one test case at a time ───────────────────────────────────
    if not ci:
        click.echo(f"Model: {config['model']}  |  Judge: {judge_model_name}  |  {total} test cases\n")

    results = []
    for idx, tc in enumerate(config["test_cases"], 1):
        if not ci:
            _status(f"[{idx}/{total}] {tc['id']}  running...", erase=True)
        result = run_test_cases({"model": config["model"], "test_cases": [tc]})
        results.extend(result)

    if not ci:
        # Clear the last status line before judging output
        click.echo(f"\r{' ' * 80}\r", nl=False)

    # ── Judge phase: stream results as each test case finishes ────────────────
    judged_cases: list[dict] = []
    n_judged = 0

    def _on_criterion(i: int, total_crit: int, text: str) -> None:
        if not ci:
            tc_id = results[n_judged]["id"] if n_judged < len(results) else "…"
            short = text[:40] + "…" if len(text) > 40 else text
            _status(f"[{n_judged + 1}/{total}] {tc_id}  judging {i + 1}/{total_crit}: {short}", erase=True)

    def _on_test_case(judged: dict) -> None:
        nonlocal n_judged
        if not ci:
            # Clear the status line, then print the final verdict for this case
            click.echo(f"\r{' ' * 80}\r", nl=False)
            _render_test_case(judged)
        judged_cases.append(judged)
        n_judged += 1

    run_result = judge_run(
        results,
        config,
        test_case_callback=_on_test_case,
        criterion_callback=_on_criterion,
    )
    storage.save_run(run_result, DB_PATH)

    if ci:
        s = run_result["summary"]
        threshold_config = config.get("threshold", {})
        if threshold_config:
            passed, msg = _check_thresholds(s, threshold_config)
        else:
            passed, msg = True, "PASS"
        print(msg)
        print(json.dumps(run_result))
        if not passed:
            sys.exit(1)
    else:
        # Summary footer
        s = run_result["summary"]
        click.echo(click.style("─" * 37, dim=True))
        improved_txt = click.style(f"{s['improved']} improved", fg="green")
        regressed_txt = click.style(f"{s['regressed']} regressed", fg="red")
        neutral_txt = click.style(f"{s['neutral']} neutral", dim=True)
        click.echo(f"{improved_txt}  ·  {regressed_txt}  ·  {neutral_txt}")
        click.echo()
        click.echo("Run `llmdiff serve` to explore results in the dashboard")

        if "threshold" in config:
            passed, msg = _check_thresholds(s, config["threshold"])
            click.echo(msg)
            if not passed:
                sys.exit(1)


@main.command()
def history():
    """List past runs as a formatted table."""
    storage.init_db(DB_PATH)
    runs = storage.list_runs(DB_PATH)

    if not runs:
        click.echo("No runs yet. Run `llmdiff compare your_tests.yaml` to get started.")
        return

    header = click.style(
        f"{'RUN ID':<28}  {'TIMESTAMP':<20}  {'MODEL':<28}  {'↑':>3}  {'↓':>3}  {'→':>3}",
        bold=True,
    )
    click.echo(header)
    click.echo("─" * 90)

    for r in runs:
        s = r["summary"]
        run_id = r["run_id"]
        ts = r["timestamp"][:19].replace("T", " ")
        model = r["model"][:28]
        improved = click.style(str(s.get("improved", 0)), fg="green")
        regressed = click.style(str(s.get("regressed", 0)), fg="red")
        neutral = click.style(str(s.get("neutral", 0)), dim=True)
        click.echo(f"{run_id:<28}  {ts:<20}  {model:<28}  {improved:>3}  {regressed:>3}  {neutral:>3}")


@main.command()
def serve():
    """Start the web UI at localhost:7331."""
    click.echo(f"Starting LLM Diff dashboard at http://localhost:{_cfg.PORT}")
    from backend.server import start
    start()


# ── Demo mode ────────────────────────────────────────────────────────────────

DEMO_TEST_CASES = [
    {
        "id": "demo_001",
        "input": "What is LangChain's default chunk size?",
        "output_v1": (
            "LangChain's default chunk size is 1000 characters. The RecursiveCharacterTextSplitter "
            "uses this as its default, along with a chunk_overlap of 200 characters to maintain "
            "context between adjacent chunks. This is a reasonable default for most document types."
        ),
        "output_v2": "The default chunk_size is 1000 characters with 200 character overlap.",
        "criteria_results": [
            {
                "criterion": "Answer is factually correct (1000 characters)",
                "score_v1": 1.0,
                "score_v2": 1.0,
                "delta": 0.0,
                "verdict": "neutral",
                "reasoning": "Both correctly state 1000 characters.",
                "confidence": "high",
            },
            {
                "criterion": "Response is under 100 words",
                "score_v1": 0.0,
                "score_v2": 1.0,
                "delta": 1.0,
                "verdict": "improved",
                "reasoning": "v2 is concise; v1 is verbose.",
                "confidence": "high",
            },
            {
                "criterion": "Answer cites the context, not prior knowledge",
                "score_v1": 0.5,
                "score_v2": 0.5,
                "delta": 0.0,
                "verdict": "neutral",
                "reasoning": "Both reference context implicitly.",
                "confidence": "medium",
            },
        ],
        "overall_verdict": "improved",
    },
    {
        "id": "demo_002",
        "input": "How do I handle rate limit errors from OpenAI?",
        "output_v1": (
            "Use exponential backoff. The tenacity library makes this easy. "
            "Start with 1 second, double each retry, add jitter, cap at 60 seconds."
        ),
        "output_v2": (
            "Just catch the error and sleep for a few seconds before retrying. "
            "A fixed delay of 5 seconds usually works fine."
        ),
        "criteria_results": [
            {
                "criterion": "Mentions exponential backoff",
                "score_v1": 1.0,
                "score_v2": 0.0,
                "delta": -1.0,
                "verdict": "regressed",
                "reasoning": "v1 recommends exponential backoff; v2 uses fixed delay.",
                "confidence": "high",
            },
            {
                "criterion": "Mentions tenacity or equivalent retry library",
                "score_v1": 1.0,
                "score_v2": 0.0,
                "delta": -1.0,
                "verdict": "regressed",
                "reasoning": "v1 mentions tenacity; v2 does not.",
                "confidence": "high",
            },
            {
                "criterion": "Does not recommend a fixed sleep as primary strategy",
                "score_v1": 1.0,
                "score_v2": 0.0,
                "delta": -1.0,
                "verdict": "regressed",
                "reasoning": "v2 explicitly recommends a fixed 5 second sleep.",
                "confidence": "high",
            },
        ],
        "overall_verdict": "regressed",
    },
    {
        "id": "demo_003",
        "input": "What embedding model is recommended for semantic search over code?",
        "output_v1": (
            "text-embedding-3-small is a good baseline. For code-specific tasks, "
            "voyage-code-2 outperforms general models. Both support 8192 tokens."
        ),
        "output_v2": (
            "voyage-code-2 and text-embedding-3-small are both strong options for code search, "
            "each supporting up to 8192 tokens per input."
        ),
        "criteria_results": [
            {
                "criterion": "Names at least one specific embedding model",
                "score_v1": 1.0,
                "score_v2": 1.0,
                "delta": 0.0,
                "verdict": "neutral",
                "reasoning": "Both name specific models.",
                "confidence": "high",
            },
            {
                "criterion": "Mentions the token limit (8192)",
                "score_v1": 1.0,
                "score_v2": 1.0,
                "delta": 0.0,
                "verdict": "neutral",
                "reasoning": "Both correctly cite 8192 tokens.",
                "confidence": "high",
            },
            {
                "criterion": "Does not hallucinate models not in the context",
                "score_v1": 1.0,
                "score_v2": 1.0,
                "delta": 0.0,
                "verdict": "neutral",
                "reasoning": "Both stay within context.",
                "confidence": "high",
            },
        ],
        "overall_verdict": "neutral",
    },
]


@main.command()
def demo():
    """Run a demo with no API key required."""
    from datetime import datetime

    storage.init_db(DB_PATH)

    run_id = "run_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()

    verdicts = [tc["overall_verdict"] for tc in DEMO_TEST_CASES]
    all_deltas = [
        cr["delta"]
        for tc in DEMO_TEST_CASES
        for cr in tc["criteria_results"]
    ]
    score_delta_avg = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0

    run_result = {
        "run_id": run_id,
        "timestamp": timestamp,
        "yaml_file": "demo",
        "model": "demo/mock",
        "test_cases": DEMO_TEST_CASES,
        "summary": {
            "total": len(DEMO_TEST_CASES),
            "improved": verdicts.count("improved"),
            "regressed": verdicts.count("regressed"),
            "neutral": verdicts.count("neutral"),
            "score_delta_avg": score_delta_avg,
        },
    }

    click.echo()
    _render_diff(run_result)

    storage.save_run(run_result, DB_PATH)
    click.echo()
    click.echo('Demo complete. Run `llmdiff serve` to see the dashboard.')
