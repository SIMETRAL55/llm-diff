import json
import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = os.environ.get("LLMDIFF_DB_PATH", "~/.llmdiff/history.db")

# Cache a single shared connection for in-memory databases so all callers
# see the same data (each sqlite3.connect(":memory:") creates an isolated DB).
_memory_conn: sqlite3.Connection | None = None


def _connect(db_path: str) -> sqlite3.Connection:
    global _memory_conn
    if db_path == ":memory:":
        if _memory_conn is None:
            _memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
        return _memory_conn
    path = str(Path(db_path).expanduser())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _close(con: sqlite3.Connection, db_path: str) -> None:
    if db_path != ":memory:":
        con.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create tables if they don't exist. Idempotent."""
    con = _connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            yaml_file   TEXT NOT NULL,
            model       TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            result_json  TEXT NOT NULL
        )
    """)
    con.commit()
    _close(con, db_path)


def save_run(run_result: dict, db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert a run result into the database."""
    con = _connect(db_path)
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_result["run_id"],
            run_result["timestamp"],
            run_result.get("yaml_file", ""),
            run_result.get("model", ""),
            json.dumps(run_result["summary"]),
            json.dumps(run_result),
        ),
    )
    con.commit()
    _close(con, db_path)


def get_run(run_id: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Retrieve a single run by ID. Raises KeyError if not found."""
    con = _connect(db_path)
    row = con.execute(
        "SELECT result_json FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    _close(con, db_path)
    if row is None:
        raise KeyError(f"Run not found: {run_id}")
    return json.loads(row[0])


def list_runs(db_path: str = DEFAULT_DB_PATH, limit: int = 50) -> list[dict]:
    """Return summary list of runs, newest first."""
    con = _connect(db_path)
    rows = con.execute(
        "SELECT run_id, timestamp, yaml_file, model, summary_json "
        "FROM runs ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    _close(con, db_path)
    return [
        {
            "run_id": r[0],
            "timestamp": r[1],
            "yaml_file": r[2],
            "model": r[3],
            "summary": json.loads(r[4]),
        }
        for r in rows
    ]
