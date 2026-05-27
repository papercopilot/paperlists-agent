"""MCP server for the papercopilot/paperlists corpus.

Talks to any paperlists API deployment via PAPERLISTS_API_URL. A contributor-
hosted demo exists for evaluation, but the server does not silently default to
it so upstream users do not send queries to non-papercopilot infrastructure by
accident. Designed as a thin client so the same code works
across Claude Code / Claude Desktop / Cursor / Codex / any MCP host.

Run via stdio:
    uvx --from . paperlists-mcp
    # or
    python -m paperlists_mcp.server

Env vars:
    PAPERLISTS_API_URL   required; demo: https://api-production-18d3.up.railway.app
    PAPERLISTS_TIMEOUT   default: 30 (seconds)
"""
from __future__ import annotations

import os
from urllib.parse import quote
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

DEMO_API_URL = "https://api-production-18d3.up.railway.app"
API_URL = os.environ.get("PAPERLISTS_API_URL", "").rstrip("/")
TIMEOUT = float(os.environ.get("PAPERLISTS_TIMEOUT", "30"))

mcp = FastMCP("paperlists")
_client = (
    httpx.Client(base_url=API_URL, timeout=TIMEOUT, headers={"User-Agent": "paperlists-mcp/0.1"})
    if API_URL else None
)


def _get(path: str, **params) -> dict:
    if _client is None:
        return {
            "error": "missing_api_url",
            "detail": (
                "Set PAPERLISTS_API_URL to a paperlists API deployment. "
                f"For demo testing only, use {DEMO_API_URL}."
            ),
        }
    # Drop None params so they don't override defaults on the server side.
    clean = {k: v for k, v in params.items() if v is not None}
    try:
        resp = _client.get(path, params=clean)
    except httpx.HTTPError as e:
        return {"error": "network_error", "detail": str(e)}
    if resp.status_code >= 400:
        # Surface the API's structured error (e.g. invalid_query) when present.
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text[:500]}
        return {"error": body.get("error", f"HTTP {resp.status_code}"), **body}
    return resp.json()


@mcp.tool()
def list_coverage() -> dict:
    """List which conferences and years are indexed.

    Returns total paper count and a per-conference breakdown. Always call this
    first if you're unsure whether a venue/year is available — it costs no
    quota and saves wasted searches.
    """
    return _get("/v1/coverage")


@mcp.tool()
def corpus_manifest() -> dict:
    """Return corpus freshness and provenance metadata.

    This is the integration point for the papercopilot data-fetching
    pipeline. Today it returns build metadata and coverage. When the API
    database includes a `source_manifest` table, it also returns per-source
    fetched_at/source_url/row_count/hash/status records.
    """
    return _get("/v1/corpus_manifest")


@mcp.tool()
def search_papers(
    query: str,
    conferences: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    exclude_rejected: bool = True,
    limit: int = 25,
    offset: int = 0,
    order_by: str = "relevance",
    include_abstract: bool = False,
    raw: bool = False,
) -> dict:
    """Full-text search over title / abstract / keywords / authors.

    Args:
        query: Free-text query. By default, input is split into terms and
            AND'd together (safe for arbitrary user input). To use FTS5
            operators (`foo OR bar`, `"exact phrase"`, `title:diffusion`,
            `reason*`), set `raw=True`.
        conferences: Comma-separated venue list (e.g. "iclr,nips,icml").
            Omit to search all.
        year_from / year_to: Inclusive year filter.
        exclude_rejected: Drop Reject/Withdraw entries (recommended).
        limit: Max results (1-200).
        offset: Pagination offset. Combine with `has_more` in the response
            to walk through long result sets — call again with
            `offset = previous_offset + previous_limit` until `has_more` is
            false.
        order_by: "relevance" | "year_desc" | "citation_desc" | "rating_desc".
        include_abstract: Set true only if you need full abstracts — they
            cost ~1KB per result.
        raw: Enable full FTS5 syntax. Malformed expressions return HTTP 400
            (surfaced as `{error: "invalid_query"}` in the result).

    Returns a dict with `total_matches`, `returned`, `offset`, `limit`,
    `has_more`, and `results`. Use `has_more` to paginate.
    """
    return _get(
        "/v1/search",
        q=query, conferences=conferences,
        year_from=year_from, year_to=year_to,
        exclude_rejected=exclude_rejected,
        limit=limit, offset=offset, order_by=order_by,
        include_abstract=include_abstract,
        raw=raw,
    )


@mcp.tool()
def get_paper(conf: str, paper_id: str) -> dict:
    """Fetch a single paper's full record including abstract and affiliations.

    Args:
        conf: Lowercase venue (e.g. "iclr", "nips").
        paper_id: The paper's `id` field from the JSON (OpenReview ID, etc.).
    """
    return _get(f"/v1/paper/{quote(conf, safe='')}/{quote(paper_id, safe='')}")


@mcp.tool()
def topic_trend(
    query: str,
    conferences: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Yearly publication volume + citation-weighted volume for a topic.

    Use this to see how a research area has grown or declined over time.
    Returns a `series` of `{year, papers, citations, by_conf}` records.

    Example queries: "diffusion model", "retrieval augmented generation",
    "mixture of experts", "constitutional AI". Set `raw=True` for FTS5
    operator support.
    """
    return _get(
        "/v1/topic_trend",
        q=query, conferences=conferences,
        year_from=year_from, year_to=year_to,
        exclude_rejected=exclude_rejected, raw=raw,
    )


@mcp.tool()
def topic_evolution(
    query: str,
    year_from: int,
    year_to: int,
    window: int = 1,
    top_k: int = 15,
    conferences: Optional[str] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Track how a research area evolves: per-window top co-occurring keywords,
    top venues, and landmark papers.

    This is the single best tool for answering "how did <field> change between
    year X and year Y?" It surfaces topic drift inside a query — e.g. asking
    `topic_evolution("retrieval augmented generation", 2020, 2024, window=1)`
    will show RAG morphing from dense-passage-retrieval era into LLM-coupled
    pipelines.

    Each window includes a `ranking_basis` field — `"gs_citation"` when
    citations are meaningful, `"rating_avg+status_fallback"` for the current
    year where citations are near-zero. In the fallback regime, landmarks
    are sorted by acceptance status (Oral > Spotlight > Poster) and rating,
    not citations.

    Args:
        window: years per bucket (1 = annual, 2 = biennial, ...).
        top_k: keywords/venues per window.
        exclude_rejected: Drop Reject/Withdraw entries (recommended).
        raw: Enable full FTS5 syntax in the query.
    """
    return _get(
        "/v1/topic_evolution",
        q=query, year_from=year_from, year_to=year_to,
        window=window, top_k=top_k, conferences=conferences,
        exclude_rejected=exclude_rejected, raw=raw,
    )


@mcp.tool()
def compare_periods(
    query: str,
    period_a_from: int,
    period_a_to: int,
    period_b_from: int,
    period_b_to: int,
    top_k: int = 15,
    conferences: Optional[str] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Diff a topic between two year ranges. Returns three buckets per
    dimension (keywords, authors, affiliations):

    - **emerged**: present in period B but not period A
    - **faded**: present in period A but not period B
    - **sustained**: present in both, sorted by total volume

    Use when you have a hypothesis like "RLHF moved from RL conferences to
    NLP conferences between 2022 and 2024" — this tool will confirm or refute.

    Each period in the response exposes both `years: [a, b]` and flat
    `year_from`/`year_to` fields. Set `raw=True` for FTS5 operator support.
    """
    return _get(
        "/v1/compare_periods",
        q=query,
        period_a_from=period_a_from, period_a_to=period_a_to,
        period_b_from=period_b_from, period_b_to=period_b_to,
        top_k=top_k,
        conferences=conferences,
        exclude_rejected=exclude_rejected, raw=raw,
    )


@mcp.tool()
def author_trajectory(
    name: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> dict:
    """Papers by an author across years, useful for tracking a researcher's
    topical pivots or productivity. Match is on the `authors` field — use
    the canonical name as it appears in publications.
    """
    return _get(
        "/v1/author_trajectory",
        name=name, year_from=year_from, year_to=year_to,
    )


@mcp.tool()
def field_landscape(
    query: str,
    year: int,
    top_k: int = 10,
    conferences: Optional[str] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Snapshot a research field in a specific year: top papers (by citation),
    top authors, top affiliations, top keywords, venue distribution.

    Use for "state of <field> in <year>" summaries or to identify the dominant
    labs in a subfield at a point in time. `raw=True` enables FTS5 operators.
    """
    return _get(
        "/v1/field_landscape",
        q=query, year=year, top_k=top_k,
        conferences=conferences, exclude_rejected=exclude_rejected, raw=raw,
    )


@mcp.tool()
def conference_stats(conf: str, year: int) -> dict:
    """Acceptance breakdown + rating/citation distribution for a single
    venue-year. Useful for "how selective was ICLR 2024?" type questions."""
    return _get(f"/v1/conference_stats/{conf}/{year}")


@mcp.tool()
def top_papers(
    conf: str, year: int,
    by: str = "gs_citation",
    top_k: int = 20,
    exclude_rejected: bool = True,
) -> dict:
    """Top-N papers from a single venue-year, ranked by `gs_citation` or
    `rating`. Returns title, authors, paper_id, URLs.

    `exclude_rejected` defaults to True so Reject/Withdraw entries don't
    pollute the ranking on OpenReview venues (ICLR/ICML/COLM). Set False
    for raw corpus diagnostics.
    """
    return _get(
        f"/v1/top_papers/{conf}/{year}",
        by=by, top_k=top_k, exclude_rejected=exclude_rejected,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
