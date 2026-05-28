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


def test_compare_periods_includes_venue_diff(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2021.json",
        [{"id": "old", "title": "rlhf", "keywords": "rlhf", "status": "Poster"}],
    )
    _write_json(
        tmp_path / "acl" / "acl2025.json",
        [{"id": "new", "title": "rlhf", "keywords": "rlhf", "status": "Long"}],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        diff = queries.compare_periods(
            conn,
            q="rlhf",
            period_a=(2020, 2022),
            period_b=(2023, 2025),
        )

    assert diff["venue_diff"]["faded"] == [("iclr", 1)]
    assert diff["venue_diff"]["emerged"] == [("acl", 1)]


def test_topic_evolution_uses_title_terms_when_keywords_are_missing(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "nips" / "nips2020.json",
        [
            {
                "id": "rag",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "status": "Poster",
            }
        ],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.topic_evolution(
            conn,
            q="retrieval augmented generation",
            year_from=2020,
            year_to=2020,
        )

    keywords = [term for term, _ in out["windows"][0]["top_keywords"]]
    assert "retrieval" in keywords
    assert "augmented" in keywords
    assert "generation" in keywords


def test_author_trajectory_excludes_rejected_by_default_and_supports_filters(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {
                "id": "accepted",
                "title": "Accepted reasoning",
                "author": "Yejin Choi",
                "status": "Poster",
                "gs_citation": 10,
            },
            {
                "id": "rejected",
                "title": "Rejected reasoning",
                "author": "Yejin Choi",
                "status": "Reject",
                "gs_citation": 100,
            },
        ],
    )
    _write_json(
        tmp_path / "acl" / "acl2025.json",
        [
            {
                "id": "acl-paper",
                "title": "ACL reasoning",
                "author": "Yejin Choi",
                "status": "Findings",
                "gs_citation": 20,
            }
        ],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.author_trajectory(conn, name="Yejin Choi")
        titles = [paper["title"] for year in out["by_year"] for paper in year["papers"]]
        assert titles == ["ACL reasoning", "Accepted reasoning"]
        assert out["exclude_rejected"] is True

        iclr = queries.author_trajectory(
            conn, name="Yejin Choi", conferences=["iclr"]
        )
        iclr_titles = [
            paper["title"] for year in iclr["by_year"] for paper in year["papers"]
        ]
        assert iclr_titles == ["Accepted reasoning"]

        raw = queries.author_trajectory(
            conn, name="Yejin Choi", exclude_rejected=False
        )
        raw_titles = [
            paper["title"] for year in raw["by_year"] for paper in year["papers"]
        ]
        assert raw_titles == ["Rejected reasoning", "ACL reasoning", "Accepted reasoning"]


def test_author_trajectory_fails_closed_for_overly_broad_names(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {"id": "a", "title": "Paper A", "author": "Li", "status": "Poster"},
            {"id": "b", "title": "Paper B", "author": "Li", "status": "Poster"},
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        try:
            queries.author_trajectory(conn, name="Li", max_matches=1)
        except queries.TooManyMatchesError as exc:
            assert exc.endpoint == "author_trajectory"
            assert exc.matches == 2
            assert exc.max_matches == 1
        else:
            raise AssertionError("expected TooManyMatchesError")


def test_api_rejects_reversed_year_ranges() -> None:
    from fastapi.testclient import TestClient
    from paperlists_api import main as m

    client = TestClient(m.app)
    resp = client.get(
        "/v1/topic_evolution",
        params={"q": "reasoning", "year_from": 2025, "year_to": 2024},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "year_from must be <= year_to"

    resp = client.get(
        "/v1/compare_periods",
        params={
            "q": "reasoning",
            "period_a_from": 2025,
            "period_a_to": 2024,
            "period_b_from": 2023,
            "period_b_to": 2024,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "period_a_from must be <= period_a_to"
