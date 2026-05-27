"""Cross-worker rate limiter backed by a tiny sqlite file.

The query API's main `papers.db` is opened immutable in production, so we
keep the bucket state in a separate writable sqlite file. All uvicorn
workers see the same bucket for the same client IP, which is the whole
point — an in-memory bucket per worker would be 4× too generous on a
4-worker deployment and would let an attacker rotate workers to bypass
the limit.

Token-bucket semantics:
- `tokens` accumulate at PAPERLISTS_RATE_PER_MIN/60.0 per second
- Each accepted request consumes 1 token
- Bursts are capped at PAPERLISTS_RATE_BURST
- Stale rows (last_seen > PAPERLISTS_RATE_STALE_SEC ago) are GC'd
  opportunistically on every Nth call
- Hard cap of PAPERLISTS_RATE_BUCKET_MAX rows; oldest evicted first

The implementation is intentionally simple — one SELECT, one UPSERT per
request, all in a single WAL-mode db. At ~60 req/min/IP this is well
under the cost of the actual FTS5 query that follows.
"""
from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from pathlib import Path

RATE_DB_PATH = Path(
    os.environ.get("PAPERLISTS_RATELIMIT_DB", "/tmp/paperlists-ratelimit.db")
).resolve()
RATE_PER_MIN = float(os.environ.get("PAPERLISTS_RATE_PER_MIN", "60"))
RATE_BURST = float(os.environ.get("PAPERLISTS_RATE_BURST", "20"))
BUCKET_MAX = int(os.environ.get("PAPERLISTS_RATE_BUCKET_MAX", "10000"))
STALE_SEC = float(os.environ.get("PAPERLISTS_RATE_STALE_SEC", str(60 * 30)))
GC_PROBABILITY = float(os.environ.get("PAPERLISTS_RATE_GC_PROBABILITY", "0.01"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_buckets (
    ip         TEXT PRIMARY KEY,
    tokens     REAL NOT NULL,
    last_seen  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_last_seen ON rate_buckets(last_seen);
"""

# Per-worker connection. sqlite3 connections aren't thread-safe by default
# but FastAPI's threadpool may call us from multiple threads; we serialize
# inside the process via a lock and rely on WAL + IMMEDIATE for cross-
# worker safety.
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _ensure_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    RATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RATE_DB_PATH, timeout=2.0, isolation_level=None, check_same_thread=False)
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=2000;")
    conn.executescript(_SCHEMA)
    _conn = conn
    return conn


def _maybe_gc(conn: sqlite3.Connection, now: float) -> None:
    """Opportunistically drop stale + over-cap rows. Cheap on the median
    request because we only run with probability GC_PROBABILITY."""
    if random.random() > GC_PROBABILITY:
        return
    conn.execute("DELETE FROM rate_buckets WHERE last_seen < ?", (now - STALE_SEC,))
    n = conn.execute("SELECT COUNT(*) FROM rate_buckets").fetchone()[0]
    if n > BUCKET_MAX:
        # Drop the oldest (n - BUCKET_MAX) rows.
        conn.execute(
            "DELETE FROM rate_buckets WHERE ip IN ("
            "  SELECT ip FROM rate_buckets ORDER BY last_seen ASC LIMIT ?"
            ")",
            (n - BUCKET_MAX,),
        )


def check_and_consume(ip: str) -> tuple[bool, int]:
    """Return (allowed, retry_after_sec). retry_after_sec is meaningful
    only when allowed=False."""
    now = time.time()
    with _lock:
        conn = _ensure_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            # Couldn't grab the writer lock in time — fail open rather than
            # crash; the limiter is a soft guardrail, not a hard barrier.
            return True, 0
        try:
            row = conn.execute(
                "SELECT tokens, last_seen FROM rate_buckets WHERE ip=?", (ip,)
            ).fetchone()
            if row is None:
                tokens, last = RATE_BURST, now
            else:
                tokens, last = row
                tokens = min(RATE_BURST, tokens + (now - last) * (RATE_PER_MIN / 60.0))

            if tokens < 1.0:
                # Persist the refilled-but-still-insufficient state so the
                # client doesn't keep "starting fresh" by spacing requests.
                conn.execute(
                    "INSERT INTO rate_buckets(ip, tokens, last_seen) VALUES(?,?,?) "
                    "ON CONFLICT(ip) DO UPDATE SET tokens=excluded.tokens, last_seen=excluded.last_seen",
                    (ip, tokens, now),
                )
                conn.execute("COMMIT")
                retry = int((1.0 - tokens) * 60.0 / RATE_PER_MIN) + 1
                return False, retry

            conn.execute(
                "INSERT INTO rate_buckets(ip, tokens, last_seen) VALUES(?,?,?) "
                "ON CONFLICT(ip) DO UPDATE SET tokens=excluded.tokens, last_seen=excluded.last_seen",
                (ip, tokens - 1.0, now),
            )
            _maybe_gc(conn, now)
            conn.execute("COMMIT")
            return True, 0
        except Exception:
            conn.execute("ROLLBACK")
            raise


def reset_for_tests() -> None:
    """Test helper: drop all bucket state."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        try:
            RATE_DB_PATH.unlink()
        except FileNotFoundError:
            pass
