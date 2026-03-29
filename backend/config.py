"""Central configuration for LLM Diff.

All values can be overridden via environment variables (e.g. in .env):

  LLMDIFF_JUDGE_VOTES=3          # number of judge calls per criterion (1=fast, 3=reliable)
  LLMDIFF_JUDGE_SLEEP=6          # seconds to sleep after each judge call (for rate limiting)
  LLMDIFF_DB_PATH=~/.llmdiff/history.db
  LLMDIFF_PORT=7331
"""
import os

# How many judge calls to make per criterion. Majority vote is used.
# 1 = fastest (no voting), 3 = most reliable (recommended for important runs).
JUDGE_VOTES: int = int(os.environ.get("LLMDIFF_JUDGE_VOTES", "1"))

# Seconds to sleep after each judge call. Set to ~6 when using Gemini free tier
# (10 RPM limit). Leave at 0 for paid APIs.
JUDGE_SLEEP: float = float(os.environ.get("LLMDIFF_JUDGE_SLEEP", "0"))

# Path to the SQLite database file.
DB_PATH: str = os.environ.get("LLMDIFF_DB_PATH", "~/.llmdiff/history.db")

# Port for the web dashboard.
PORT: int = int(os.environ.get("LLMDIFF_PORT", "7331"))
