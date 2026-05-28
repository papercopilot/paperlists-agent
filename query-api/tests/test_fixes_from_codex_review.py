"""Regression tests for the fixes from the codex review pass.

Each test pins a specific finding so a future refactor can't silently
un-fix it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from paperlists_api import queries
from paperlists_api.indexer import build_index


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- P1: indexer dedup ----------

def test_indexer_dedup_keeps_distinct_records_with_colliding_id(tmp_path: Path) -> None:
    """codex P1: siggraph2025.json has the same `id` for 4-5 paper rows.
    The original UNIQUE(conf, year, paper_id) + INSERT OR IGNORE silently
    dropped them. We now generate a sha1 fallback per collision."""
    _write_json(
        tmp_path / "siggraph" / "siggraph2025.json",
        [
            {"id": "shared-id", "title": "Paper A", "author": "Alice"},
            {"id": "shared-id", "title": "Paper B", "author": "Bob"},
            {"id": "shared-id", "title": "Paper C", "author": "Carol"},
            {"id": "unique-id", "title": "Paper D", "author": "Dave"},
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        # All 4 distinct papers must be indexed.
        rows = conn.execute("SELECT title, paper_id FROM papers ORDER BY title").fetchall()
        titles = [r["title"] for r in rows]
        assert titles == ["Paper A", "Paper B", "Paper C", "Paper D"]
        # The fallback rekeyed at least 2 of the 3 colliding records. It may
        # use a later stable ID candidate (doi/openreview/...) or a generated
        # hash when no stable alternate exists.
        rekeyed = [r["paper_id"] for r in rows if r["paper_id"] != "shared-id"]
        assert len(rekeyed) >= 3
        assert len({r["paper_id"] for r in rows}) == 4
        # No empty paper_ids.
        for r in rows:
            assert r["paper_id"] and r["paper_id"].strip()


# ---------- P2: FTS5 sanitizer ----------

@pytest.mark.parametrize("bad_query", [
    "foo AND NOT bar",            # bare FTS5 operators
    'unclosed"quote',             # unbalanced double-quote
    "in-context learning",        # leading `-` would parse as NOT
    "LLM-reasoning",
    "-exclude this",
    "NEAR(foo bar)",
    'title:"injection"',          # column-scoped phrase injection
])
def test_sanitize_fts_neutralizes_dangerous_inputs(tmp_path: Path, bad_query: str) -> None:
    """codex P2: previously these inputs raised sqlite3.OperationalError →
    HTTP 500. Now they should either match zero results or match legitimate
    terms, but never raise."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "x", "title": "foo bar baz", "keywords": "test", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        # Should not raise. The result count is allowed to be anything ≥ 0.
        out = queries.search_papers(conn, q=bad_query)
        assert "total_matches" in out
        assert out["total_matches"] >= 0


# ---------- P2: search response shape ----------

def test_search_returns_total_matches_and_pagination_metadata(tmp_path: Path) -> None:
    """codex P2: clients can't paginate without total_matches; the field
    was previously named `total`. Keep `total` as a back-compat alias."""
    rows = [
        {"id": f"p{i}", "title": f"reasoning paper {i}", "keywords": "reasoning", "status": "Poster"}
        for i in range(7)
    ]
    _write_json(tmp_path / "iclr" / "iclr2025.json", rows)
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.search_papers(conn, q="reasoning", limit=3, offset=0)
        assert out["total_matches"] == 7
        assert out["total"] == 7  # alias for one release
        assert out["returned"] == 3
        assert out["limit"] == 3
        assert out["offset"] == 0
        assert out["has_more"] is True

        out2 = queries.search_papers(conn, q="reasoning", limit=10, offset=0)
        assert out2["has_more"] is False


# ---------- P2: compare_periods schema ----------

def test_compare_periods_exposes_flat_year_fields(tmp_path: Path) -> None:
    """codex P2 + my own flag: clients expect period_a.year_from /
    period_a.year_to; previously only `years: [a, b]` was emitted."""
    _write_json(
        tmp_path / "iclr" / "iclr2023.json",
        [{"id": "a", "title": "reasoning", "keywords": "reasoning", "status": "Poster"}],
    )
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "b", "title": "reasoning", "keywords": "reasoning", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.compare_periods(
            conn, q="reasoning",
            period_a=(2023, 2023),
            period_b=(2025, 2025),
        )
        for k in ("years", "year_from", "year_to", "n_papers"):
            assert k in out["period_a"], f"period_a missing {k}"
            assert k in out["period_b"], f"period_b missing {k}"
        assert out["period_a"]["year_from"] == 2023
        assert out["period_a"]["year_to"] == 2023
        assert out["period_b"]["year_from"] == 2025
        assert out["period_b"]["year_to"] == 2025


# ---------- P2: landmark fallback ----------

def test_topic_evolution_landmark_fallback_when_citations_sparse(tmp_path: Path) -> None:
    """codex P2: current-year papers have ~0 citations, so filtering by
    `if r['cites']` strips them. We now switch to rating_avg + status."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {
                "id": "spotlight-paper", "title": "Spotlight reasoning paper",
                "keywords": "reasoning", "status": "Spotlight",
                "rating_avg": 8.0, "gs_citation": 0,
            },
            {
                "id": "poster-paper", "title": "Poster reasoning paper",
                "keywords": "reasoning", "status": "Poster",
                "rating_avg": 6.0, "gs_citation": None,
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.topic_evolution(
            conn, q="reasoning", year_from=2025, year_to=2025, window=1,
        )
        assert len(out["windows"]) == 1
        w = out["windows"][0]
        # Must surface BOTH papers (old code would have shown 0 landmarks).
        titles = [p["title"] for p in w["landmark_papers"]]
        assert "Spotlight reasoning paper" in titles
        assert "Poster reasoning paper" in titles
        # Higher-rating Spotlight should rank above lower-rating Poster.
        assert titles[0] == "Spotlight reasoning paper"
        # Caller can see WHY the ranking was chosen.
        assert w["ranking_basis"] == "rating_avg+status_fallback"


# ---------- P1: top_papers exclude_rejected ----------

# ---------- Round 2 (codex re-review) ----------

def test_landmark_fallback_actually_demotes_low_cite_paper(tmp_path: Path) -> None:
    """codex round-2 P2a: previous fix reported `ranking_basis: rating_avg+
    status_fallback` but still sorted by `(cite, rating, bonus)` — so a 1-cite
    weak paper still beat a 0-cite Spotlight. The sort key must actually
    change in the fallback regime."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {
                "id": "weak-but-cited",
                "title": "Weak paper with one citation",
                "keywords": "reasoning",
                "status": "Poster",
                "rating_avg": 3.0,
                "gs_citation": 1,
            },
            {
                "id": "strong-spotlight",
                "title": "Strong Spotlight reasoning paper",
                "keywords": "reasoning",
                "status": "Spotlight",
                "rating_avg": 8.0,
                "gs_citation": 0,
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.topic_evolution(
            conn, q="reasoning", year_from=2025, year_to=2025, window=1,
        )
        w = out["windows"][0]
        assert w["ranking_basis"] == "rating_avg+status_fallback"
        # The Spotlight must rank first despite 0 citations.
        assert w["landmark_papers"][0]["title"] == "Strong Spotlight reasoning paper"
        assert w["landmark_papers"][1]["title"] == "Weak paper with one citation"


def test_landmark_uses_citation_sort_above_threshold(tmp_path: Path) -> None:
    """Symmetric check: when max_cite >= 20, citation ordering wins even
    against higher-rated papers."""
    _write_json(
        tmp_path / "iclr" / "iclr2024.json",
        [
            {
                "id": "cited-classic",
                "title": "Cited classic",
                "keywords": "reasoning",
                "status": "Poster",
                "rating_avg": 5.0,
                "gs_citation": 500,
            },
            {
                "id": "fresh-spotlight",
                "title": "Fresh Spotlight",
                "keywords": "reasoning",
                "status": "Spotlight",
                "rating_avg": 9.0,
                "gs_citation": 5,
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.topic_evolution(
            conn, q="reasoning", year_from=2024, year_to=2024, window=1,
        )
        w = out["windows"][0]
        assert w["ranking_basis"] == "gs_citation"
        assert w["landmark_papers"][0]["title"] == "Cited classic"


def test_raw_mode_enables_fts5_operators(tmp_path: Path) -> None:
    """codex round-2 P2b: with `raw=False` (default) FTS5 operators like
    OR and column filters are neutralized inside a safe phrase. With
    `raw=True` they regain their FTS5 meaning."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {"id": "a", "title": "diffusion model", "status": "Poster"},
            {"id": "b", "title": "transformer model", "status": "Poster"},
            {"id": "c", "title": "completely unrelated graph paper", "status": "Poster"},
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        # Default sanitizer: "diffusion OR transformer" is treated as a
        # literal safe phrase, so neither one-term paper matches.
        default_out = queries.search_papers(conn, q="diffusion OR transformer")
        assert default_out["total_matches"] == 0

        # raw=True: FTS5 parses OR as the boolean operator, returning both.
        raw_out = queries.search_papers(conn, q="diffusion OR transformer", raw=True)
        assert raw_out["total_matches"] == 2
        titles = {r["title"] for r in raw_out["results"]}
        assert titles == {"diffusion model", "transformer model"}


def test_default_multiword_query_is_phrase_not_token_and(tmp_path: Path) -> None:
    """Default topic matching must not fabricate longitudinal trends for
    emerging multi-word directions by matching generic terms far apart."""
    _write_json(
        tmp_path / "iclr" / "iclr2022.json",
        [
            {
                "id": "false-history",
                "title": "Test time adaptation with scaling laws",
                "keywords": "adaptation",
                "status": "Poster",
            }
        ],
    )
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {
                "id": "real-topic",
                "title": "Test Time Scaling for Reasoning",
                "keywords": "test time scaling",
                "status": "Poster",
            }
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        default = queries.topic_trend(
            conn, q="test time scaling", year_from=2022, year_to=2025
        )
        assert default["match_mode"] == "phrase"
        assert default["query_expression"] == '"test time scaling"'
        assert [(row["year"], row["papers"]) for row in default["series"]] == [
            (2025, 1)
        ]

        token_and = queries.topic_trend(
            conn,
            q="test time scaling",
            year_from=2022,
            year_to=2025,
            match_mode="token_and",
        )
        assert token_and["match_mode"] == "token_and"
        assert token_and["query_expression"] == '"test" "time" "scaling"'
        assert [(row["year"], row["papers"]) for row in token_and["series"]] == [
            (2022, 1),
            (2025, 1),
        ]


def test_raw_mode_malformed_query_raises_typed_error(tmp_path: Path) -> None:
    """raw=True with malformed input should raise FTSQueryError (HTTP 400 at
    the API layer), not a generic 500."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "diffusion model", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        with pytest.raises(queries.FTSQueryError):
            queries.search_papers(conn, q='unclosed"phrase', raw=True)


@pytest.mark.parametrize("bad_raw_query", [
    'badcol:diffusion',       # column-filter typo → "no such column"
    'foo -bar',                # leading-`-` operator → "no such column: bar"
    '*foo',                    # bad special query → "unknown special query"
    '"unclosed',               # unbalanced quote → "unterminated string"
    'AND foo',                 # bare operator → "fts5: syntax error"
    '(unbalanced',             # unbalanced paren → "fts5: syntax error"
    'OR bar',
])
def test_raw_mode_classifies_known_fts5_parse_errors_as_400(tmp_path: Path, bad_raw_query: str) -> None:
    """codex round-4 P2: parse-error marker list must cover everything
    FTS5 actually emits for malformed user input — not just the obvious
    `fts5: syntax error`. Otherwise raw=true bad queries 500."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "diffusion model", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        with pytest.raises(queries.FTSQueryError):
            queries.search_papers(conn, q=bad_raw_query, raw=True)


def test_raw_mode_no_such_column_schema_errors_still_propagate(tmp_path: Path) -> None:
    """`no such column` can come from FTS syntax or from real SQL drift. Dotted
    SQL column names should not be relabelled as user-invalid FTS input."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "diffusion model", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            queries._run_fts(
                conn,
                """
                SELECT p.status_typo
                FROM papers_fts
                JOIN papers p ON p.id = papers_fts.rowid
                WHERE papers_fts MATCH ?
                """,
                ["diffusion"],
                raw=True,
            )


def test_proxy_auto_detection_covers_major_platforms(monkeypatch) -> None:
    """codex round-4 P3: auto-detect should fire for Railway, HF Spaces,
    Fly, Render, Vercel, Cloud Run, Azure App Service."""
    import importlib
    from paperlists_api import main as m
    for env_var in [
        "RAILWAY_ENVIRONMENT", "SPACE_ID", "FLY_APP_NAME",
        "RENDER", "VERCEL", "K_SERVICE", "WEBSITE_SITE_NAME",
    ]:
        # Clean state.
        for v in m._PLATFORM_PROXY_MARKERS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.delenv("PAPERLISTS_TRUST_PROXY", raising=False)
        monkeypatch.setenv(env_var, "1")
        importlib.reload(m)
        assert m._TRUST_PROXY, f"{env_var} should auto-enable trust_proxy"
        assert m._DETECTED_PROXY_NAME == env_var


def test_proxy_explicit_off_overrides_auto_detection(monkeypatch) -> None:
    """If the operator sets PAPERLISTS_TRUST_PROXY=0, auto-detect must
    not re-enable it. Otherwise the explicit safety opt-out is broken."""
    import importlib
    from paperlists_api import main as m
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("PAPERLISTS_TRUST_PROXY", "0")
    importlib.reload(m)
    assert not m._TRUST_PROXY


def test_client_ip_prefers_x_real_ip_when_proxy_is_trusted(monkeypatch) -> None:
    """Railway documents X-Real-IP as the client IP header; use it before XFF
    when proxy trust is enabled."""
    from types import SimpleNamespace
    from paperlists_api import main as m

    monkeypatch.setattr(m, "_TRUST_PROXY", True)
    req = SimpleNamespace(
        headers={
            "x-real-ip": "203.0.113.8",
            "x-forwarded-for": "198.51.100.9, 10.0.0.1",
        },
        client=SimpleNamespace(host="10.0.0.2"),
    )
    assert m._client_ip(req) == "203.0.113.8"

    req.headers.pop("x-real-ip")
    assert m._client_ip(req) == "198.51.100.9"

    monkeypatch.setattr(m, "_TRUST_PROXY", False)
    assert m._client_ip(req) == "10.0.0.2"


def test_search_rejects_unbounded_offset_before_querying_db() -> None:
    from fastapi.testclient import TestClient
    from paperlists_api import main as m

    resp = TestClient(m.app).get("/v1/search", params={"q": "model", "offset": m.MAX_OFFSET + 1})
    assert resp.status_code == 422


def test_server_errors_dont_masquerade_as_invalid_query(tmp_path: Path) -> None:
    """codex round-3 P3: previously, ANY sqlite OperationalError in _run_fts
    was mapped to FTSQueryError → HTTP 400 invalid_query. If the FTS table
    went missing post-deploy, a 5xx-worthy bug would be hidden as a 400.
    Now: in default (sanitized) mode, errors propagate as-is."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [{"id": "a", "title": "diffusion model", "status": "Poster"}],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        # Simulate schema drift by dropping the FTS table.
        conn.execute("DROP TABLE papers_fts")
        # Default-mode query against a broken DB should leak the real
        # error, not be relabelled as user input invalid.
        with pytest.raises(sqlite3.OperationalError):
            queries.search_papers(conn, q="diffusion")


def test_ratelimit_db_state_shared_across_invocations(tmp_path: Path) -> None:
    """codex round-2 P1a: the rate-limit bucket must be sqlite-backed so
    multiple uvicorn workers see the same state. We can't fork workers in
    a unit test, but we can prove the state lives in a sqlite file by
    closing the connection between calls and observing token consumption
    persists."""
    import os, importlib
    db_path = tmp_path / "rl.db"
    os.environ["PAPERLISTS_RATELIMIT_DB"] = str(db_path)
    os.environ["PAPERLISTS_RATE_PER_MIN"] = "60"
    os.environ["PAPERLISTS_RATE_BURST"] = "3"
    try:
        # Re-import so the module picks up the env override.
        from paperlists_api import ratelimit as rl
        importlib.reload(rl)
        rl.reset_for_tests()
        # 3 requests should pass.
        for _ in range(3):
            allowed, _ = rl.check_and_consume("1.2.3.4")
            assert allowed
        # 4th should fail.
        allowed, retry = rl.check_and_consume("1.2.3.4")
        assert not allowed
        assert retry >= 1

        # Simulate worker restart by closing the connection and re-opening.
        # State should persist (the sqlite file is the source of truth).
        if rl._conn is not None:
            rl._conn.close()
            rl._conn = None
        allowed, _ = rl.check_and_consume("1.2.3.4")
        assert not allowed, "bucket state must survive a worker restart"
    finally:
        os.environ.pop("PAPERLISTS_RATELIMIT_DB", None)
        os.environ.pop("PAPERLISTS_RATE_PER_MIN", None)
        os.environ.pop("PAPERLISTS_RATE_BURST", None)
        importlib.reload(rl)
        rl.reset_for_tests()


def test_top_papers_excludes_rejected_by_default(tmp_path: Path) -> None:
    """codex P1: top_papers had no exclude_rejected filter. For OpenReview
    venues, Reject papers can have high gs_citation and pollute the ranking."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {
                "id": "rejected-but-cited",
                "title": "Rejected but cited",
                "status": "Reject", "gs_citation": 500,
            },
            {
                "id": "accepted",
                "title": "Accepted poster",
                "status": "Poster", "gs_citation": 100,
            },
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)

    with _connect(db_path) as conn:
        out = queries.top_papers(conn, conf="iclr", year=2025, by="gs_citation")
        titles = [r["title"] for r in out["results"]]
        assert titles == ["Accepted poster"]
        assert out["exclude_rejected"] is True

        out_raw = queries.top_papers(
            conn, conf="iclr", year=2025, by="gs_citation", exclude_rejected=False,
        )
        titles_raw = [r["title"] for r in out_raw["results"]]
        assert titles_raw == ["Rejected but cited", "Accepted poster"]


def test_broad_analysis_queries_raise_too_many_matches(tmp_path: Path, monkeypatch) -> None:
    """Broad aggregation endpoints should count first and fail closed instead
    of materializing an unbounded match set in a Railway worker."""
    _write_json(
        tmp_path / "iclr" / "iclr2025.json",
        [
            {"id": f"p{i}", "title": f"learning paper {i}", "keywords": "learning", "status": "Poster"}
            for i in range(3)
        ],
    )
    db_path = tmp_path / "papers.db"
    build_index(tmp_path, db_path, force=True)
    monkeypatch.setattr(queries, "MAX_ANALYSIS_MATCHES", 2)

    with _connect(db_path) as conn:
        with pytest.raises(queries.TooManyMatchesError) as evolution_err:
            queries.topic_evolution(conn, q="learning", year_from=2025, year_to=2025)
        assert evolution_err.value.endpoint == "topic_evolution"
        assert evolution_err.value.matches == 3
        assert evolution_err.value.max_matches == 2

        with pytest.raises(queries.TooManyMatchesError) as compare_err:
            queries.compare_periods(
                conn,
                q="learning",
                period_a=(2024, 2024),
                period_b=(2025, 2025),
            )
        assert compare_err.value.endpoint == "compare_periods"

        with pytest.raises(queries.TooManyMatchesError) as landscape_err:
            queries.field_landscape(conn, q="learning", year=2025)
        assert landscape_err.value.endpoint == "field_landscape"
