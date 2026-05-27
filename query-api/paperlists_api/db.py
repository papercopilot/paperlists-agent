"""Thin sqlite connection helpers."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("PAPERLISTS_DB", "papers.db")).resolve()
DB_CACHE_KIB = int(os.environ.get("PAPERLISTS_DB_CACHE_KIB", "65536"))
DB_MMAP_BYTES = int(os.environ.get("PAPERLISTS_DB_MMAP_BYTES", str(256 * 1024 * 1024)))
DB_IMMUTABLE = os.environ.get("PAPERLISTS_DB_IMMUTABLE", "0") == "1"


def _row_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


@contextmanager
def connect():
    immutable = "&immutable=1" if DB_IMMUTABLE else ""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro{immutable}", uri=True, timeout=30)
    conn.row_factory = _row_factory
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size=-{DB_CACHE_KIB}")
    conn.execute(f"PRAGMA mmap_size={DB_MMAP_BYTES}")
    try:
        yield conn
    finally:
        conn.close()
