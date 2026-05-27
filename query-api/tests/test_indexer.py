from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from paperlists_api.indexer import build_index, discover_files


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_discover_files_does_not_require_hardcoded_conference_list(tmp_path: Path) -> None:
    _write_json(tmp_path / "newconf" / "newconf2026.json", [{"id": "a", "title": "A"}])
    _write_json(tmp_path / "tools" / "tools2026.json", [{"id": "ignored", "title": "Ignored"}])

    assert discover_files(tmp_path) == [("newconf", 2026, tmp_path / "newconf" / "newconf2026.json")]


def test_build_index_preserves_records_without_id_field(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "siggraph" / "siggraph2024.json",
        [
            {"psid": "sess1", "doi": "10.1145/one", "title": "Paper One", "author": "Ada Lovelace", "keywords": "graphics"},
            {"doi": "10.1145/example", "title": "Paper Two", "author": "Grace Hopper", "keywords": "rendering"},
            {"psid": "sess1", "url_paper": "https://example.test/paper3", "title": "Paper Three", "author": "Alan Turing", "keywords": "simulation"},
            {"psid": "sess1", "title": "Paper Four", "author": "Katherine Johnson", "keywords": "simulation"},
        ],
    )

    db_path = tmp_path / "papers.db"
    stats = build_index(tmp_path, db_path, force=True)

    assert stats["rows_indexed"] == 4
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT paper_id, source_path, source_index FROM papers ORDER BY source_index"
    ).fetchall()
    conn.close()

    assert rows[0] == ("10.1145/one", "siggraph/siggraph2024.json", 0)
    assert rows[1] == ("10.1145/example", "siggraph/siggraph2024.json", 1)
    assert rows[2] == ("https://example.test/paper3", "siggraph/siggraph2024.json", 2)
    assert rows[3][0].startswith("generated:")
    assert rows[3][1:] == ("siggraph/siggraph2024.json", 3)


def test_build_index_tries_stable_alternate_ids_before_dedup_hash(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "siggraph" / "siggraph2025.json",
        [
            {"id": "shared-session", "doi": "10.1145/first", "title": "Paper One"},
            {"id": "shared-session", "doi": "10.1145/second", "title": "Paper Two"},
            {"id": "shared-session", "openreview": "https://openreview.net/forum?id=third", "title": "Paper Three"},
        ],
    )

    db_path = tmp_path / "papers.db"
    stats = build_index(tmp_path, db_path, force=True)

    assert stats["rows_indexed"] == 3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT paper_id FROM papers ORDER BY source_index").fetchall()
    conn.close()

    assert [row[0] for row in rows] == [
        "shared-session",
        "10.1145/second",
        "https://openreview.net/forum?id=third",
    ]


def test_build_index_checkpoints_wal_before_close(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "A", "keywords": "reasoning"}],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    wal_path = Path(f"{db_path}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0
