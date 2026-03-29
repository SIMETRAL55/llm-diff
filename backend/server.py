import asyncio
import json
import os
import re
from pathlib import Path
from functools import partial

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from backend import storage
from backend.runner import run_test_cases
from backend.judge import judge_test_case

DB_PATH = os.environ.get("LLMDIFF_DB_PATH", "~/.llmdiff/history.db")
PORT = int(os.environ.get("LLMDIFF_PORT", "7331"))

# Directory where YAML test files are allowed to be loaded from.
# Only files within this directory (non-recursively following symlinks) are
# permitted when the UI submits a run request.
_ALLOWED_YAML_DIR = Path(os.environ.get("LLMDIFF_YAML_DIR", str(Path.home() / ".llmdiff" / "tests"))).resolve()

# Allowlist of model-string prefixes accepted by the streaming endpoint.
# Operators can extend this list via the LLMDIFF_ALLOWED_PROVIDERS env var
# (comma-separated prefixes, e.g. "groq/,openai/,ollama/").
_DEFAULT_ALLOWED_PROVIDERS = {
    "groq/",
    "openai/",
    "anthropic/",
    "ollama/",
    "gemini/",
    "vertex_ai/",
    "cohere/",
    "mistral/",
    "together_ai/",
    "bedrock/",
    "azure/",
}

_env_providers = os.environ.get("LLMDIFF_ALLOWED_PROVIDERS", "")
ALLOWED_PROVIDERS: frozenset[str] = frozenset(
    p.strip() for p in _env_providers.split(",") if p.strip()
) or frozenset(_DEFAULT_ALLOWED_PROVIDERS)


def _validate_model_string(model: str) -> None:
    """Raise ValueError if model string does not match an allowed provider prefix."""
    if not any(model.startswith(prefix) for prefix in ALLOWED_PROVIDERS):
        raise ValueError(
            f"Model '{model}' uses a disallowed provider. "
            f"Allowed prefixes: {sorted(ALLOWED_PROVIDERS)}"
        )


def _resolve_yaml_path(yaml_file: str) -> Path:
    """Resolve yaml_file to an absolute path and verify it stays within
    _ALLOWED_YAML_DIR. Raises ValueError on path-traversal attempts."""
    # Accept absolute paths only within the allowed dir, or bare filenames
    # relative to the allowed dir.
    candidate = Path(yaml_file)
    if not candidate.is_absolute():
        candidate = _ALLOWED_YAML_DIR / candidate

    resolved = candidate.resolve()

    # Ensure the resolved path is inside the allowed directory.
    try:
        resolved.relative_to(_ALLOWED_YAML_DIR)
    except ValueError:
        raise ValueError(
            f"yaml_file path '{yaml_file}' is outside the allowed directory "
            f"'{_ALLOWED_YAML_DIR}'. Set LLMDIFF_YAML_DIR to change this."
        )

    if not resolved.exists():
        raise FileNotFoundError(f"YAML file not found: {resolved}")

    return resolved


app = FastAPI(title="LLM Diff")

_STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def startup():
    storage.init_db(DB_PATH)


@app.get("/")
def index():
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "LLM Diff — run `npm run build` in frontend/ first"}, status_code=503)


@app.get("/api/runs")
def api_list_runs():
    return storage.list_runs(DB_PATH)


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    # Validate run_id contains only safe characters before passing to storage.
    if not re.match(r"^[A-Za-z0-9_\-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    try:
        return storage.get_run(run_id, DB_PATH)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


class StreamRequest(BaseModel):
    yaml_file: str

    @field_validator("yaml_file")
    @classmethod
    def yaml_file_must_be_safe(cls, v: str) -> str:
        # Basic sanity check before full path resolution in the endpoint.
        if not v or ".." in v or v.startswith("/etc") or v.startswith("/proc"):
            raise ValueError("Invalid yaml_file value")
        return v


@app.post("/api/runs/stream")
async def api_stream_run(req: StreamRequest):
    # Resolve and validate path — raises ValueError on traversal attempts.
    try:
        yaml_path = _resolve_yaml_path(req.yaml_file)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async def event_generator():
        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            return

        # Validate model strings before passing to LiteLLM (SSRF guard).
        for field in ("model", "judge_model"):
            model_val = config.get(field) or config.get("model", "")
            if model_val:
                try:
                    _validate_model_string(model_val)
                except ValueError as exc:
                    yield {"event": "error", "data": json.dumps({"error": str(exc)})}
                    return

        config["yaml_file"] = str(yaml_path)
        test_cases = config.get("test_cases", [])
        judge_model = config.get("judge_model", config.get("model"))

        from datetime import datetime
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().isoformat()

        yield {
            "event": "start",
            "data": json.dumps({"total": len(test_cases), "run_id": run_id}),
        }

        judged_cases = []
        tc_criteria = {tc["id"]: tc.get("criteria", []) for tc in test_cases}

        loop = asyncio.get_event_loop()
        for tc in test_cases:
            runner_results = await loop.run_in_executor(
                None, run_test_cases, {"model": config["model"], "test_cases": [tc]}
            )
            if runner_results:
                result = runner_results[0]
                criteria = tc_criteria.get(result["id"], [])
                judged = await loop.run_in_executor(
                    None, partial(judge_test_case, result, criteria, judge_model)
                )
                judged_cases.append(judged)
                yield {"event": "result", "data": json.dumps(judged)}

        verdicts = [tc["overall_verdict"] for tc in judged_cases]
        all_deltas = [
            cr["delta"]
            for tc in judged_cases
            for cr in tc.get("criteria_results", [])
        ]
        score_delta_avg = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0

        summary = {
            "total": len(judged_cases),
            "improved": verdicts.count("improved"),
            "regressed": verdicts.count("regressed"),
            "neutral": verdicts.count("neutral"),
            "score_delta_avg": score_delta_avg,
        }

        run_result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "yaml_file": str(yaml_path),
            "model": config.get("model", ""),
            "test_cases": judged_cases,
            "summary": summary,
        }

        # Always write to the server-configured DB path — never trust client-supplied paths.
        storage.save_run(run_result, DB_PATH)

        yield {"event": "done", "data": json.dumps(summary)}

    return EventSourceResponse(event_generator())


def start():
    import uvicorn
    # Bind to localhost only. To expose on a network interface, set the
    # LLMDIFF_HOST env var explicitly (e.g. LLMDIFF_HOST=0.0.0.0 for Docker).
    host = os.environ.get("LLMDIFF_HOST", "127.0.0.1")
    uvicorn.run("backend.server:app", host=host, port=PORT, reload=False)


# Mount static files last so API routes take priority
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
