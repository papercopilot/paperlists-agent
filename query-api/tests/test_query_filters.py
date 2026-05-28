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
    assert diff["match_mode"] == "phrase"
    assert diff["query_expression"] == '"rlhf"'


def test_api_accepts_token_and_match_mode_for_sensitivity_checks(tmp_path: Path, monkeypatch) -> None:
    from contextlib import contextmanager
    from fastapi.testclient import TestClient
    from paperlists_api import main as m

    m.ratelimit.reset_for_tests()
    _write_json(
        tmp_path / "iclr" / "iclr2022.json",
        [{"id": "old", "title": "Test time adaptation with scaling laws", "status": "Poster"}],
    )
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "new", "title": "Test Time Scaling", "status": "Poster"}],
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

    monkeypatch.setattr(m, "connect", _test_connect)
    client = TestClient(m.app)

    default_resp = client.get(
        "/v1/topic_trend",
        params={"q": "test time scaling", "year_from": 2022, "year_to": 2025},
    )
    broad_resp = client.get(
        "/v1/topic_trend",
        params={
            "q": "test time scaling",
            "year_from": 2022,
            "year_to": 2025,
            "match_mode": "token_and",
        },
    )

    assert default_resp.status_code == 200
    assert broad_resp.status_code == 200
    assert default_resp.json()["match_mode"] == "phrase"
    assert broad_resp.json()["match_mode"] == "token_and"
    assert [row["year"] for row in default_resp.json()["series"]] == [2025]
    assert [row["year"] for row in broad_resp.json()["series"]] == [2022, 2025]


def test_alias_or_match_mode_expands_known_acronyms(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "acl" / "acl2024.json",
        [
            {
                "id": "acro",
                "title": "RAG improves retrieval",
                "keywords": "rag; retrieval",
                "status": "Long",
            },
            {
                "id": "full",
                "title": "Retrieval Augmented Generation for QA",
                "keywords": "retrieval augmented generation; qa",
                "status": "Long",
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        phrase = queries.search_papers(conn, q="RAG")
        expanded = queries.search_papers(conn, q="RAG", match_mode="alias_or")

    assert phrase["total_matches"] == 1
    assert expanded["match_mode"] == "alias_or"
    assert expanded["query_alias_expanded"] is True
    assert expanded["query_aliases"] == ["RAG", "retrieval augmented generation"]
    assert expanded["query_expression"] == '"RAG" OR "retrieval augmented generation"'
    assert expanded["total_matches"] == 2


def test_alias_or_match_mode_expands_rlhf_phrase_set(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "acl" / "acl2024.json",
        [
            {"id": "acro", "title": "RLHF Alignment", "keywords": "rlhf", "status": "Long"},
            {
                "id": "with-from",
                "title": "Reinforcement Learning from Human Feedback",
                "keywords": "alignment",
                "status": "Long",
            },
            {
                "id": "without-from",
                "title": "Reinforcement Learning Human Feedback",
                "keywords": "alignment",
                "status": "Long",
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        acro = queries.search_papers(conn, q="RLHF", match_mode="alias_or")
        full = queries.search_papers(
            conn,
            q="reinforcement learning from human feedback",
            match_mode="alias_or",
        )

    expected_aliases = [
        "RLHF",
        "reinforcement learning from human feedback",
        "reinforcement learning human feedback",
    ]
    assert acro["query_aliases"] == expected_aliases
    assert acro["total_matches"] == 3
    assert full["query_aliases"] == [
        "reinforcement learning from human feedback",
        "reinforcement learning human feedback",
        "rlhf",
    ]
    assert full["total_matches"] == 3


def test_alias_or_match_mode_expands_combined_aliases(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "acl" / "acl2024.json",
        [
            {"id": "a", "title": "LLM RAG", "status": "Long"},
            {"id": "b", "title": "Large Language Model RAG", "status": "Long"},
            {"id": "c", "title": "LLM Retrieval Augmented Generation", "status": "Long"},
            {
                "id": "d",
                "title": "Large Language Model Retrieval Augmented Generation",
                "status": "Long",
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        expanded = queries.search_papers(conn, q="LLM RAG", match_mode="alias_or")

    assert "large language model retrieval augmented generation" in expanded["query_aliases"]
    assert expanded["total_matches"] == 4


def test_search_fails_closed_for_overly_broad_queries(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {"id": "a", "title": "Model A", "status": "Poster"},
            {"id": "b", "title": "Model B", "status": "Poster"},
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)
    monkeypatch.setattr(queries, "MAX_SEARCH_MATCHES", 1)

    with _connect(db_path) as conn:
        try:
            queries.search_papers(conn, q="model")
        except queries.TooManyMatchesError as exc:
            assert exc.endpoint == "search"
            assert exc.matches == 2
            assert exc.max_matches == 1
        else:
            raise AssertionError("expected TooManyMatchesError")


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
    assert "knowledge" in keywords
    assert "intensive" in keywords
    assert "nlp" in keywords
    assert "retrieval" not in keywords
    assert "generation" not in keywords
    suppressed = [term for term, _ in out["windows"][0]["suppressed_query_keywords"]]
    assert "retrieval" in suppressed
    assert "augmented" in suppressed
    assert "generation" in suppressed
    assert out["query_noise_filter"]["suppressed_count"] == 3


def test_compare_periods_uses_title_terms_when_keywords_are_missing(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "nips" / "nips2024.json",
        [{"id": "old", "title": "LLM Reasoning with Chain of Thought", "status": "Poster"}],
    )
    _write_json(
        tmp_path / "nips" / "nips2025.json",
        [{"id": "new", "title": "LLM Reasoning with Reinforcement Learning", "status": "Poster"}],
    )

    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        diff = queries.compare_periods(
            conn,
            q="LLM reasoning",
            period_a=(2024, 2024),
            period_b=(2025, 2025),
        )

    assert ("reinforcement", 1) in diff["keyword_diff"]["emerged"]
    assert ("chain", 1) in diff["keyword_diff"]["faded"]
    suppressed = diff["keyword_diff_suppressed_query_terms"]
    assert ("llm", 1, 1) in suppressed["sustained"]
    assert ("reasoning", 1, 1) in suppressed["sustained"]
    assert diff["query_noise_filter"]["suppressed_count"] == 2


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
        first_paper = out["by_year"][0]["papers"][0]
        assert first_paper["author_position"] == 1
        assert first_paper["n_authors"] == 1

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

    m.ratelimit.reset_for_tests()
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
