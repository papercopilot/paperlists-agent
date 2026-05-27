from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from paperlists_api import queries
from paperlists_api.indexer import build_index


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_corpus_manifest_works_without_pipeline_table(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "Reasoning paper", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.corpus_manifest(conn)

    assert out["total_papers"] == 1
    assert out["source_manifest_available"] is False
    assert out["sources"] == []
    assert out["pipeline_runs"] == []
    assert "iclr" in out["conferences"]


def test_corpus_manifest_exposes_source_manifest_when_present(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "Reasoning paper", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE source_manifest (
                conf TEXT,
                year INTEGER,
                fetched_at TEXT,
                source_url TEXT,
                row_count INTEGER,
                hash TEXT,
                pipeline_status TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO source_manifest
                (conf, year, fetched_at, source_url, row_count, hash, pipeline_status)
            VALUES
                ('iclr', 2025, '2026-05-27T00:00:00Z',
                 'https://github.com/papercopilot/paperlists', 1, 'sha256:test', 'ok')
            """
        )
        out = queries.corpus_manifest(conn)

    assert out["source_manifest_available"] is True
    assert out["sources"] == [
        {
            "conf": "iclr",
            "year": 2025,
            "fetched_at": "2026-05-27T00:00:00Z",
            "source_url": "https://github.com/papercopilot/paperlists",
            "row_count": 1,
            "hash": "sha256:test",
            "pipeline_status": "ok",
        }
    ]


def test_corpus_manifest_route_returns_manifest(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from paperlists_api import main

    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "Reasoning paper", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    @contextmanager
    def _test_connect():
        conn = _connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr(main, "connect", _test_connect)

    resp = TestClient(main.app).get("/v1/corpus_manifest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_papers"] == 1
    assert body["source_manifest_available"] is False
