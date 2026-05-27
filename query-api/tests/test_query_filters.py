from __future__ import annotations

import json
import sqlite3
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


def test_research_endpoints_share_default_rejected_filter(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2024.json",
        [
            {
                "id": "accepted-2024",
                "title": "LLM reasoning accepted 2024",
                "author": "Ada Lovelace",
                "aff": "Example University",
                "keywords": "llm reasoning; chain-of-thought",
                "status": "Poster",
            },
            {
                "id": "rejected-2024",
                "title": "LLM reasoning rejected 2024",
                "author": "Grace Hopper",
                "aff": "Example Lab",
                "keywords": "llm reasoning; rejected",
                "status": "reject",
            },
        ],
    )
    _write_json(
        tmp_path / "nips" / "nips2025.json",
        [
            {
                "id": "accepted-2025",
                "title": "LLM reasoning accepted 2025",
                "author": "Katherine Johnson",
                "aff": "Example University",
                "keywords": "llm reasoning; test-time scaling",
                "status": "Poster",
            },
            {
                "id": "withdrawn-2025",
                "title": "LLM reasoning withdrawn 2025",
                "author": "Alan Turing",
                "aff": "Example Lab",
                "keywords": "llm reasoning; withdrawn",
                "status": "WITHDRAWN",
            },
        ],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        search_2025 = queries.search_papers(
            conn, q="LLM reasoning", year_from=2025, year_to=2025
        )
        trend = queries.topic_trend(conn, q="LLM reasoning", year_from=2024, year_to=2025)
        evolution = queries.topic_evolution(conn, q="LLM reasoning", year_from=2024, year_to=2025)
        landscape = queries.field_landscape(conn, q="LLM reasoning", year=2025)
        diff = queries.compare_periods(
            conn,
            q="LLM reasoning",
            period_a=(2024, 2024),
            period_b=(2025, 2025),
        )

        assert search_2025["total"] == 1
        assert [row["papers"] for row in trend["series"]] == [1, 1]
        assert [window["n_papers"] for window in evolution["windows"]] == [1, 1]
        assert landscape["n_papers"] == 1
        assert diff["period_a"]["n_papers"] == 1
        assert diff["period_b"]["n_papers"] == 1

        full_evolution = queries.topic_evolution(
            conn,
            q="LLM reasoning",
            year_from=2024,
            year_to=2025,
            exclude_rejected=False,
        )
        full_landscape = queries.field_landscape(
            conn, q="LLM reasoning", year=2025, exclude_rejected=False
        )
        full_diff = queries.compare_periods(
            conn,
            q="LLM reasoning",
            period_a=(2024, 2024),
            period_b=(2025, 2025),
            exclude_rejected=False,
        )

        assert [window["n_papers"] for window in full_evolution["windows"]] == [2, 2]
        assert full_landscape["n_papers"] == 2
        assert full_diff["period_a"]["n_papers"] == 2
        assert full_diff["period_b"]["n_papers"] == 2


def test_landscape_and_period_diff_support_conference_filter(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "iclr", "title": "LLM reasoning", "keywords": "llm reasoning", "status": "Poster"}],
    )
    _write_json(
        tmp_path / "nips" / "nips2025.json",
        [{"id": "nips", "title": "LLM reasoning", "keywords": "llm reasoning", "status": "Poster"}],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        landscape = queries.field_landscape(
            conn, q="LLM reasoning", year=2025, conferences=["iclr"]
        )
        diff = queries.compare_periods(
            conn,
            q="LLM reasoning",
            period_a=(2025, 2025),
            period_b=(2025, 2025),
            conferences=["nips"],
        )

    assert landscape["n_papers"] == 1
    assert landscape["venue_distribution"] == [("iclr", 1)]
    assert diff["period_a"]["n_papers"] == 1
    assert diff["period_b"]["n_papers"] == 1
