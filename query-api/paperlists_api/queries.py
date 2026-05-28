"""Query primitives over the sqlite index.

Kept separate from the FastAPI layer so the same functions can be reused
from CLI tests, notebooks, or an offline `--mode local` MCP fallback.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Iterable, Optional

EXCLUDED_STATUSES_DEFAULT = ("Withdraw", "Reject", "Withdrawn", "Rejected", "Desk Reject")
_EXCLUDED_STATUSES_LOWER = tuple(s.lower() for s in EXCLUDED_STATUSES_DEFAULT)
MAX_ANALYSIS_MATCHES = 50_000
MAX_AUTHOR_TRAJECTORY_MATCHES = 500


class FTSQueryError(ValueError):
    """User query couldn't be parsed by FTS5. API layer maps this to HTTP 400."""


class TooManyMatchesError(ValueError):
    """Analysis query matched too many rows to aggregate safely in one worker."""

    def __init__(self, *, endpoint: str, matches: int, max_matches: int):
        self.endpoint = endpoint
        self.matches = matches
        self.max_matches = max_matches
        super().__init__(
            f"{endpoint} matched {matches} papers; narrow the years, venues, "
            f"or query terms below {max_matches}"
        )


# FTS5 input is treated as a single quoted phrase after tokenization.
# This neutralizes the FTS5 query language entirely: operators like AND/OR/NOT/
# NEAR, prefix-NOT (`-foo`), column filters (`title:`), and unbalanced quotes
# can no longer fall through and either change semantics or raise
# sqlite3.OperationalError.
#
# The precision trade-off is intentional for longitudinal analysis: a query like
# "test time scaling" should not match any paper that happens to contain the
# generic terms "test", "time", and "scaling" far apart in the abstract, because
# that fabricates a historical trend for an emerging direction. Use `raw=True`
# when the caller needs broader FTS5 syntax.
_FTS_KEEP = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _fts_tokens(q: str) -> list[str]:
    """Split user text into FTS5-safe tokens. Drops punctuation; hyphens
    become whitespace so `in-context` doesn't read as the NOT-operator."""
    if not q:
        return []
    cleaned = _FTS_KEEP.sub(" ", q.replace("-", " "))
    return [t for t in cleaned.split() if t]


def sanitize_fts(q: str) -> str:
    """Turn user text into a safe FTS5 expression.

    Tokens are joined inside one double-quoted phrase so FTS5 operators
    (AND/OR/NOT/NEAR, column filters, `-`-prefix-NOT, prefix `*`) and
    unbalanced quotes can never fall through.

    **Trade-off:** this is stricter than the previous token-AND default, and
    documented FTS5 power features (`foo OR bar`,
    `"exact phrase"`, `title:diffusion`, `reason*`) are NOT honored under
    this default. Use `raw=True` on the endpoint to opt back in to full
    FTS5 syntax (syntax errors then return HTTP 400).
    """
    tokens = _fts_tokens(q)
    if not tokens:
        return ""
    return f'"{" ".join(tokens)}"'


def sanitize_fts_phrase_in(column: str, q: str) -> str:
    """Build an FTS5 column-scoped phrase: `column:"tok1 tok2"`. Used for
    authors lookup where we want a single ordered phrase, not term-AND."""
    tokens = _fts_tokens(q)
    if not tokens:
        return ""
    return f'{column}:"{" ".join(tokens)}"'


def _prepare_query(q: str, raw: bool) -> str:
    """Map user input to an FTS5 expression. With `raw=True` we trust the
    caller and let FTS5 parse the input as-is; the syntax-error → 400
    mapping in `_run_fts` still keeps malformed input from 500'ing."""
    if raw:
        raw_query = (q or "").strip()
        _validate_raw_fts_columns(raw_query)
        return raw_query
    return sanitize_fts(q)


# Known FTS5 parse-error fingerprints. Anything *not* in this set is a
# server-side problem (missing table, locked db, schema drift, ...) and
# should bubble up as 5xx rather than be mislabelled `invalid_query`.
#
# Empirically derived by feeding bad input to FTS5; tests below pin the set.
_FTS_PARSE_ERROR_MARKERS = (
    "fts5: syntax error",          # AND/OR/NEAR misuse, unbalanced parens
    "fts5: parse error",
    "malformed match",
    "unterminated string",         # unbalanced double-quote
    "parse error",
    "unknown special query",       # `*foo` and other special-query syntax errors
    "fts5: unknown",               # variants of the above
)
_ALLOWED_RAW_FTS_COLUMNS = {"title", "abstract", "keywords", "authors"}
_RAW_COLUMN_FILTER_RE = re.compile(r"(?:^|[\s(])([A-Za-z_][A-Za-z0-9_]*)\s*:")
_SQLITE_NO_SUCH_COLUMN_RE = re.compile(r"no such column:\s*([A-Za-z_][\w.]*)")


def _raw_fts_user_column_error(msg: str, raw_query: str) -> bool:
    """Return true when sqlite's `no such column` came from FTS5 syntax.

    FTS5 reports both `badcol:value` and `foo -bar` as `no such column`,
    which looks identical to a real SQL/schema bug. Treat it as user input
    only when the identifier appears in the raw MATCH expression in one of
    those FTS-specific forms. Dotted names (`p.status`) are SQL columns and
    should propagate as server errors.
    """
    m = _SQLITE_NO_SUCH_COLUMN_RE.search(msg)
    if not m:
        return False
    col = m.group(1).lower()
    if "." in col:
        return False
    q = raw_query.lower()
    return bool(
        re.search(rf"(?:^|[\s(]){re.escape(col)}\s*:", q)
        or re.search(rf"(?:^|[\s(])-{re.escape(col)}\b", q)
    )


def _validate_raw_fts_columns(q: str) -> None:
    for m in _RAW_COLUMN_FILTER_RE.finditer(q or ""):
        col = m.group(1).lower()
        if col not in _ALLOWED_RAW_FTS_COLUMNS:
            raise FTSQueryError(
                f"unknown FTS5 column {col!r}; allowed columns are "
                f"{', '.join(sorted(_ALLOWED_RAW_FTS_COLUMNS))}"
            )


def _run_fts(
    conn: sqlite3.Connection,
    sql: str,
    params: list,
    *,
    raw: bool = False,
) -> list:
    """Execute an FTS5-backed query.

    When `raw=True` the caller passed user input verbatim into the MATCH
    expression, so FTS5 parse errors are user-facing and we translate them
    to FTSQueryError (HTTP 400 at the API layer). Any other sqlite error
    (missing table, schema drift, db locked) propagates as-is so it
    surfaces as a 5xx, not a misleading `invalid_query`.

    When `raw=False` (default), the input was produced by `sanitize_fts`
    which only emits quoted phrase tokens — that grammar is closed and
    cannot fail. So we don't catch anything: if it fails, it's a server
    bug and should be loud."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        if not raw:
            raise
        msg = str(e).lower()
        raw_query = str(params[0] if params else "")
        if _raw_fts_user_column_error(msg, raw_query):
            raise FTSQueryError(str(e)) from e
        if any(marker in msg for marker in _FTS_PARSE_ERROR_MARKERS):
            raise FTSQueryError(str(e)) from e
        # Looked like an operational/server error even in raw mode — leak it.
        raise


def _enforce_analysis_match_cap(
    conn: sqlite3.Connection,
    count_sql: str,
    params: list,
    *,
    raw: bool,
    endpoint: str,
    max_matches: Optional[int] = None,
) -> int:
    """Count FTS matches before broad in-memory aggregations.

    Trend endpoints intentionally compute keyword / author / affiliation
    counters in Python because their output shape is nested and agent-facing.
    A broad query like "learning" over 15+ years can match a large fraction of
    the corpus, so count first and fail closed instead of materializing every
    row in a Railway worker.
    """
    if max_matches is None:
        max_matches = MAX_ANALYSIS_MATCHES
    rows = _run_fts(conn, count_sql, params, raw=raw)
    matches = int(rows[0]["n"]) if rows else 0
    if matches > max_matches:
        raise TooManyMatchesError(
            endpoint=endpoint,
            matches=matches,
            max_matches=max_matches,
        )
    return matches


def _conf_filter(confs: Optional[list[str]]) -> tuple[str, list]:
    if not confs:
        return "", []
    placeholders = ",".join("?" * len(confs))
    return f" AND p.conf IN ({placeholders})", [c.lower() for c in confs]


def _year_filter(year_from: Optional[int], year_to: Optional[int]) -> tuple[str, list]:
    parts, params = [], []
    if year_from is not None:
        parts.append("p.year >= ?")
        params.append(year_from)
    if year_to is not None:
        parts.append("p.year <= ?")
        params.append(year_to)
    if not parts:
        return "", []
    return " AND " + " AND ".join(parts), params


def _exclude_rejected_filter(exclude: bool, *, alias: str = "p") -> tuple[str, list]:
    if not exclude:
        return "", []
    placeholders = ",".join("?" * len(_EXCLUDED_STATUSES_LOWER))
    col = f"{alias}.status" if alias else "status"
    return f" AND ({col} IS NULL OR LOWER({col}) NOT IN ({placeholders}))", list(_EXCLUDED_STATUSES_LOWER)


def _row_to_card(row: dict, *, include_abstract: bool) -> dict:
    out = {
        "conf": row["conf"],
        "year": row["year"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "authors": row["authors"],
        "status": row["status"],
        "track": row["track"],
        "site": row["site"],
        "openreview": row["openreview"],
        "pdf": row["pdf"],
        "rating_avg": row["rating_avg"],
        "gs_citation": row["gs_citation"],
        "keywords": row["keywords"],
        "primary_area": row["primary_area"],
    }
    if include_abstract:
        out["abstract"] = row["abstract"]
    return out


def search_papers(
    conn: sqlite3.Connection,
    *,
    q: str,
    conferences: Optional[list[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    exclude_rejected: bool = True,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "relevance",
    include_abstract: bool = False,
    raw: bool = False,
) -> dict:
    """Full-text search across title/abstract/keywords/authors.

    Default (`raw=False`): input is tokenized into one quoted phrase. Safe for
    arbitrary user input and precise enough for longitudinal topic queries.

    With `raw=True`: input is passed to FTS5 as-is, so callers can use
    `foo OR bar`, `"exact phrase"`, `title:diffusion`, `reason*`. Malformed
    expressions raise FTSQueryError (HTTP 400 at the API layer).
    """
    q_clean = _prepare_query(q, raw)
    if not q_clean:
        return {
            "total_matches": 0, "returned": 0, "offset": offset,
            "limit": limit, "has_more": False, "total": 0, "results": [],
        }

    conf_sql, conf_params = _conf_filter(conferences)
    year_sql, year_params = _year_filter(year_from, year_to)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)

    order_sql = {
        "relevance": "bm25(papers_fts)",
        "year_desc": "p.year DESC, bm25(papers_fts)",
        "citation_desc": "p.gs_citation DESC NULLS LAST, bm25(papers_fts)",
        "rating_desc": "p.rating_avg DESC NULLS LAST, bm25(papers_fts)",
    }.get(order_by, "bm25(papers_fts)")

    sql = f"""
        SELECT p.*
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?
          {conf_sql}{year_sql}{excl_sql}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
    """
    params = [q_clean, *conf_params, *year_params, *excl_params, limit, offset]
    rows = _run_fts(conn, sql, params, raw=raw)

    count_sql = f"""
        SELECT COUNT(*) AS n
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?
          {conf_sql}{year_sql}{excl_sql}
    """
    total = _run_fts(conn, count_sql, [q_clean, *conf_params, *year_params, *excl_params], raw=raw)[0]["n"]

    cards = [_row_to_card(r, include_abstract=include_abstract) for r in rows]
    return {
        # Primary, agent-friendly field name (matches the rest of the API).
        "total_matches": total,
        "returned": len(cards),
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(cards)) < total,
        # Back-compat alias — keep one release, then remove.
        "total": total,
        "results": cards,
    }


def get_paper(conn: sqlite3.Connection, conf: str, paper_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM papers WHERE conf=? AND paper_id=? LIMIT 1",
        (conf.lower(), paper_id),
    ).fetchone()
    if not row:
        return None
    out = _row_to_card(row, include_abstract=True)
    out["affiliations"] = row["affiliations"]
    out["confidence_avg"] = row["confidence_avg"]
    return out


def coverage(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT conf, year, COUNT(*) AS n FROM papers GROUP BY conf, year ORDER BY conf, year"
    ).fetchall()
    by_conf: dict[str, dict] = {}
    total = 0
    for r in rows:
        d = by_conf.setdefault(r["conf"], {"years": {}, "total": 0})
        d["years"][r["year"]] = r["n"]
        d["total"] += r["n"]
        total += r["n"]
    built_at = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
    return {
        "total_papers": total,
        "conferences": by_conf,
        "built_at": float(built_at["value"]) if built_at else None,
    }


def corpus_manifest(conn: sqlite3.Connection) -> dict:
    """Return corpus freshness/provenance metadata for agent clients.

    v1 intentionally works before Jing's data-fetching pipeline is wired in:
    the API returns coverage + build metadata even when the optional
    `source_manifest` table does not exist yet. Once the pipeline writes that
    table, the same endpoint starts exposing per-source provenance without a
    schema migration in the API layer.
    """
    cov = coverage(conn)
    built_at = cov.get("built_at")
    manifest_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_manifest'"
    ).fetchone() is not None

    sources: list[dict] = []
    if manifest_exists:
        columns = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(source_manifest)").fetchall()
        ]
        wanted = [
            "conf", "year", "fetched_at", "source_url",
            "row_count", "hash", "pipeline_status",
        ]
        selected = [c for c in wanted if c in columns]
        if selected:
            order_cols = [c for c in ("conf", "year", "fetched_at") if c in selected]
            order_sql = f" ORDER BY {', '.join(order_cols)}" if order_cols else ""
            rows = conn.execute(
                f"SELECT {', '.join(selected)} FROM source_manifest{order_sql}"
            ).fetchall()
            sources = [dict(row) for row in rows]

    return {
        "built_at": built_at,
        "total_papers": cov["total_papers"],
        "conferences": cov["conferences"],
        "source_manifest_available": manifest_exists,
        "sources": sources,
        # Reserved for the agentic fetch pipeline's run-level status monitor.
        "pipeline_runs": [],
    }


# ---------- Research-evolution endpoints ----------

def topic_trend(
    conn: sqlite3.Connection,
    *,
    q: str,
    conferences: Optional[list[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Yearly paper count + citation-weighted volume for a topic query.

    `raw=True` passes the query to FTS5 verbatim (full operator support);
    default sanitizes input into one quoted phrase.
    """
    q_clean = _prepare_query(q, raw)
    if not q_clean:
        return {"query": q, "series": []}

    conf_sql, conf_params = _conf_filter(conferences)
    year_sql, year_params = _year_filter(year_from, year_to)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)

    sql = f"""
        SELECT p.year AS year,
               p.conf AS conf,
               COUNT(*) AS papers,
               COALESCE(SUM(p.gs_citation), 0) AS citations
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?
          {conf_sql}{year_sql}{excl_sql}
        GROUP BY p.year, p.conf
        ORDER BY p.year, p.conf
    """
    params = [q_clean, *conf_params, *year_params, *excl_params]
    rows = _run_fts(conn, sql, params, raw=raw)

    # Also yearly totals (any conference matching) for denominator/visualization.
    yearly: dict[int, dict] = {}
    for r in rows:
        bucket = yearly.setdefault(r["year"], {"year": r["year"], "papers": 0, "citations": 0, "by_conf": {}})
        bucket["papers"] += r["papers"]
        bucket["citations"] += r["citations"]
        bucket["by_conf"][r["conf"]] = {"papers": r["papers"], "citations": r["citations"]}
    return {
        "query": q,
        "series": [yearly[y] for y in sorted(yearly)],
    }


# Stop-words to drop when extracting keyword/term drift.
_STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or such that the their then there these to was were will with we our you your this they them than but not no any all can may use using used new novel model models method methods learning deep neural network networks based approach paper task tasks
""".split())


def _tokenize_keywords(s: str) -> Iterable[str]:
    if not s:
        return []
    # paperlists separates keywords with semicolons; fall back to splitting on commas/newlines.
    parts = re.split(r"[;,\n]", s)
    out = []
    for p in parts:
        kw = p.strip().lower()
        if kw and kw not in _STOPWORDS and len(kw) > 1:
            out.append(kw)
    return out


def _tokenize_title_terms(s: str) -> Iterable[str]:
    """Fallback terms for rows whose source metadata lacks `keywords`."""
    if not s:
        return []
    out = []
    for tok in _fts_tokens(s.lower()):
        if tok not in _STOPWORDS and len(tok) > 2:
            out.append(tok)
    return out


def topic_evolution(
    conn: sqlite3.Connection,
    *,
    q: str,
    year_from: int,
    year_to: int,
    window: int = 1,
    top_k: int = 15,
    conferences: Optional[list[str]] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Per-year (or per-window) top co-occurring keywords and top venues.

    For each year window, fetch papers matching `q` and aggregate keyword
    frequencies. Surfaces topic drift inside a research area.

    `raw=True` enables full FTS5 syntax in the query (operators, prefix,
    column filters); default sanitizes input.
    """
    q_clean = _prepare_query(q, raw)
    if not q_clean or year_from > year_to:
        return {"query": q, "windows": []}

    conf_sql, conf_params = _conf_filter(conferences)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)
    from_where_sql = f"""
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?
          AND p.year BETWEEN ? AND ?
          {conf_sql}{excl_sql}
    """
    params = [q_clean, year_from, year_to, *conf_params, *excl_params]
    total_matches = _enforce_analysis_match_cap(
        conn,
        f"SELECT COUNT(*) AS n {from_where_sql}",
        params,
        raw=raw,
        endpoint="topic_evolution",
    )
    rows = _run_fts(
        conn,
        f"""
        SELECT p.keywords AS keywords, p.conf AS conf, p.title AS title,
               p.gs_citation AS cites, p.rating_avg AS rating, p.status AS status,
               p.year AS year, p.paper_id AS paper_id
        {from_where_sql}
        """,
        params,
        raw=raw,
    )

    buckets: dict[int, dict] = {}
    y = year_from
    while y <= year_to:
        w_end = min(y + window - 1, year_to)
        buckets[y] = {
            "year_from": y,
            "year_to": w_end,
            "n_papers": 0,
            "keywords": Counter(),
            "venues": Counter(),
            "candidates": [],
            "max_cite": 0,
        }
        y = w_end + 1

    # Status weight: prefer accepted-with-distinction → poster → unknown.
    _STATUS_BONUS = {
        "oral": 3.0, "spotlight": 2.0, "poster": 1.0,
        "accept": 1.0, "accepted": 1.0,
    }

    for r in rows:
        start = year_from + ((r["year"] - year_from) // window) * window
        bucket = buckets[start]
        bucket["n_papers"] += 1
        terms = list(_tokenize_keywords(r["keywords"]))
        if not terms:
            terms = list(_tokenize_title_terms(r["title"]))
        for kw in terms:
            bucket["keywords"][kw] += 1
        if r["conf"]:
            bucket["venues"][r["conf"]] += 1
        # Track every candidate; rank later with a citation-or-rating blend.
        bucket["candidates"].append(r)
        if (r["cites"] or 0) > bucket["max_cite"]:
            bucket["max_cite"] = r["cites"] or 0

    def _row_features(row: dict) -> tuple:
        cite = row["cites"] or 0
        rating = row["rating"] or 0.0
        status = (row["status"] or "").lower()
        bonus = _STATUS_BONUS.get(status, 0.0)
        return cite, rating, bonus

    def _key_by_citation(row: dict) -> tuple:
        cite, rating, bonus = _row_features(row)
        return (cite, rating, bonus)

    def _key_by_rating(row: dict) -> tuple:
        # In the fallback regime ratings + acceptance bonus dominate. We
        # still include citation as the deepest tiebreaker so 1-cite vs
        # 0-cite among otherwise-identical papers stays deterministic.
        cite, rating, bonus = _row_features(row)
        return (bonus, rating, cite)

    windows = []
    for y in sorted(buckets):
        bucket = buckets[y]
        # Citation-only is unfair for the current year (papers <1 year old have
        # near-zero gs_citation). When the window's max citation is low, the
        # landmark ranking degenerates to "earliest-published venue wins" —
        # one weak paper with a single cite would beat a strong Spotlight
        # with 0 cites. Switch to a rating-and-status sort in that regime.
        if bucket["max_cite"] >= 20:
            ranking_basis = "gs_citation"
            key_fn = _key_by_citation
        else:
            ranking_basis = "rating_avg+status_fallback"
            key_fn = _key_by_rating
        landmarks = sorted(bucket["candidates"], key=key_fn, reverse=True)[:5]
        windows.append({
            "year_from": bucket["year_from"],
            "year_to": bucket["year_to"],
            "n_papers": bucket["n_papers"],
            "ranking_basis": ranking_basis,
            "top_keywords": bucket["keywords"].most_common(top_k),
            "top_venues": bucket["venues"].most_common(10),
            "landmark_papers": [
                {
                    "conf": r["conf"], "year": r["year"], "paper_id": r["paper_id"],
                    "title": r["title"],
                    "gs_citation": r["cites"],
                    "rating_avg": r["rating"],
                    "status": r["status"],
                }
                for r in landmarks
            ],
        })

    return {"query": q, "window": window, "total_matches": total_matches, "windows": windows}


def author_trajectory(
    conn: sqlite3.Connection,
    *,
    name: str,
    conferences: Optional[list[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    exclude_rejected: bool = True,
    max_matches: Optional[int] = None,
) -> dict:
    """List papers by an author across years, grouped for trajectory analysis."""
    if not name or not name.strip():
        return {"name": name, "total_papers": 0, "by_year": []}

    # We match against the `authors` FTS column as an exact-phrase token.
    q_phrase = sanitize_fts_phrase_in("authors", name)
    if not q_phrase:
        return {"name": name, "total_papers": 0, "by_year": []}
    conf_sql, conf_params = _conf_filter(conferences)
    year_sql, year_params = _year_filter(year_from, year_to)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)
    from_where_sql = f"""
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?
          {conf_sql}{year_sql}{excl_sql}
    """
    params = [q_phrase, *conf_params, *year_params, *excl_params]
    _enforce_analysis_match_cap(
        conn,
        f"SELECT COUNT(*) AS n {from_where_sql}",
        params,
        raw=False,
        endpoint="author_trajectory",
        max_matches=max_matches or MAX_AUTHOR_TRAJECTORY_MATCHES,
    )
    rows = _run_fts(
        conn,
        f"""
        SELECT p.year, p.conf, p.paper_id, p.title, p.authors, p.gs_citation, p.status
        {from_where_sql}
        ORDER BY p.year DESC, p.gs_citation DESC NULLS LAST
        """,
        params,
    )

    by_year: dict[int, list] = {}
    for r in rows:
        # Confirm name actually appears (case-insensitive) — FTS porter stemming can over-match.
        if name.lower() not in (r["authors"] or "").lower():
            continue
        by_year.setdefault(r["year"], []).append({
            "conf": r["conf"], "paper_id": r["paper_id"], "title": r["title"],
            "authors": r["authors"], "gs_citation": r["gs_citation"], "status": r["status"],
        })

    return {
        "name": name,
        "exclude_rejected": exclude_rejected,
        "total_papers": sum(len(v) for v in by_year.values()),
        "by_year": [
            {"year": y, "papers": by_year[y]} for y in sorted(by_year, reverse=True)
        ],
    }


def field_landscape(
    conn: sqlite3.Connection,
    *,
    q: str,
    year: int,
    top_k: int = 10,
    conferences: Optional[list[str]] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Single-year snapshot for a field: top papers, top authors, top affiliations,
    top keywords. Useful for 'state of <field> in <year>' summaries.

    `raw=True` enables full FTS5 syntax."""
    q_clean = _prepare_query(q, raw)
    if not q_clean:
        return {"query": q, "year": year}

    conf_sql, conf_params = _conf_filter(conferences)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)
    from_where_sql = f"""
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ? AND p.year = ?
          {conf_sql}{excl_sql}
    """
    params = [q_clean, year, *conf_params, *excl_params]
    _enforce_analysis_match_cap(
        conn,
        f"SELECT COUNT(*) AS n {from_where_sql}",
        params,
        raw=raw,
        endpoint="field_landscape",
    )
    rows = _run_fts(
        conn,
        f"""
        SELECT p.conf, p.year, p.paper_id, p.title, p.authors, p.affiliations,
               p.keywords, p.gs_citation, p.rating_avg, p.status, p.openreview, p.site
        {from_where_sql}
        """,
        params,
        raw=raw,
    )

    author_counter: Counter[str] = Counter()
    aff_counter: Counter[str] = Counter()
    kw_counter: Counter[str] = Counter()
    venue_counter: Counter[str] = Counter()
    for r in rows:
        for a in (r["authors"] or "").split(";"):
            a = a.strip()
            if a:
                author_counter[a] += 1
        for af in (r["affiliations"] or "").split(";"):
            af = af.strip()
            if af:
                aff_counter[af] += 1
        for kw in _tokenize_keywords(r["keywords"]):
            kw_counter[kw] += 1
        if r["conf"]:
            venue_counter[r["conf"]] += 1

    top_papers = sorted(
        rows, key=lambda r: (r["gs_citation"] or 0, r["rating_avg"] or 0), reverse=True
    )[:top_k]

    return {
        "query": q,
        "year": year,
        "n_papers": len(rows),
        "top_papers": [
            {"conf": r["conf"], "paper_id": r["paper_id"], "title": r["title"],
             "authors": r["authors"], "gs_citation": r["gs_citation"],
             "rating_avg": r["rating_avg"], "openreview": r["openreview"], "site": r["site"]}
            for r in top_papers
        ],
        "top_authors": author_counter.most_common(top_k),
        "top_affiliations": aff_counter.most_common(top_k),
        "top_keywords": kw_counter.most_common(top_k),
        "venue_distribution": venue_counter.most_common(),
    }


def compare_periods(
    conn: sqlite3.Connection,
    *,
    q: str,
    period_a: tuple[int, int],
    period_b: tuple[int, int],
    top_k: int = 15,
    conferences: Optional[list[str]] = None,
    exclude_rejected: bool = True,
    raw: bool = False,
) -> dict:
    """Diff a topic between two year ranges. Returns keywords/authors/affiliations
    that emerged, disappeared, or stayed across the two periods.

    `raw=True` enables full FTS5 syntax."""
    def _period_meta(p: tuple[int, int], n: int) -> dict:
        # Expose both shapes so clients can use either `years[0]/[1]` or
        # the flat `year_from`/`year_to` form.
        return {
            "years": list(p),
            "year_from": p[0],
            "year_to": p[1],
            "n_papers": n,
        }

    q_clean = _prepare_query(q, raw)
    if not q_clean:
        empty = {"emerged": [], "faded": [], "sustained": []}
        return {
            "query": q,
            "period_a": _period_meta(period_a, 0),
            "period_b": _period_meta(period_b, 0),
            "keyword_diff": empty,
            "author_diff": empty,
            "affiliation_diff": empty,
            "venue_diff": empty,
        }

    lo = min(period_a[0], period_b[0])
    hi = max(period_a[1], period_b[1])
    conf_sql, conf_params = _conf_filter(conferences)
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected)
    from_where_sql = f"""
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ? AND p.year BETWEEN ? AND ?
          {conf_sql}{excl_sql}
    """
    params = [q_clean, lo, hi, *conf_params, *excl_params]
    total_matches = _enforce_analysis_match_cap(
        conn,
        f"SELECT COUNT(*) AS n {from_where_sql}",
        params,
        raw=raw,
        endpoint="compare_periods",
    )
    rows = _run_fts(
        conn,
        f"""
        SELECT p.authors, p.affiliations, p.keywords, p.conf, p.year
        {from_where_sql}
        """,
        params,
        raw=raw,
    )

    def _empty_bucket() -> dict:
        return {
            "n_papers": 0,
            "authors": Counter(),
            "affiliations": Counter(),
            "keywords": Counter(),
            "venues": Counter(),
        }

    a = _empty_bucket()
    b = _empty_bucket()

    def _add(bucket: dict, r: dict) -> None:
        bucket["n_papers"] += 1
        for author in (r["authors"] or "").split(";"):
            author = author.strip()
            if author:
                bucket["authors"][author] += 1
        for aff in (r["affiliations"] or "").split(";"):
            aff = aff.strip()
            if aff:
                bucket["affiliations"][aff] += 1
        for kw in _tokenize_keywords(r["keywords"]):
            bucket["keywords"][kw] += 1
        if r["conf"]:
            bucket["venues"][r["conf"]] += 1

    for r in rows:
        if period_a[0] <= r["year"] <= period_a[1]:
            _add(a, r)
        if period_b[0] <= r["year"] <= period_b[1]:
            _add(b, r)

    def _diff(ca: Counter, cb: Counter, k: int):
        emerged = [(x, cb[x]) for x in cb if x not in ca]
        emerged.sort(key=lambda t: t[1], reverse=True)
        faded = [(x, ca[x]) for x in ca if x not in cb]
        faded.sort(key=lambda t: t[1], reverse=True)
        sustained = [(x, ca[x], cb[x]) for x in ca if x in cb]
        sustained.sort(key=lambda t: t[1] + t[2], reverse=True)
        return {
            "emerged": emerged[:k],
            "faded": faded[:k],
            "sustained": sustained[:k],
        }

    return {
        "query": q,
        "total_matches": total_matches,
        "period_a": _period_meta(period_a, a["n_papers"]),
        "period_b": _period_meta(period_b, b["n_papers"]),
        "keyword_diff": _diff(a["keywords"], b["keywords"], top_k),
        "author_diff": _diff(a["authors"], b["authors"], top_k),
        "affiliation_diff": _diff(a["affiliations"], b["affiliations"], top_k),
        "venue_diff": _diff(a["venues"], b["venues"], top_k),
    }


def conference_stats(conn: sqlite3.Connection, *, conf: str, year: int) -> dict:
    rows = conn.execute(
        "SELECT status, track, rating_avg, gs_citation FROM papers WHERE conf=? AND year=?",
        (conf.lower(), year),
    ).fetchall()
    status_counter: Counter[str] = Counter()
    track_counter: Counter[str] = Counter()
    ratings = []
    cites = []
    for r in rows:
        status_counter[r["status"] or "(unknown)"] += 1
        track_counter[r["track"] or "(unknown)"] += 1
        if r["rating_avg"] is not None:
            ratings.append(r["rating_avg"])
        if r["gs_citation"] is not None:
            cites.append(r["gs_citation"])

    def _summary(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "n": n,
            "min": xs_sorted[0],
            "p25": xs_sorted[n // 4],
            "median": xs_sorted[n // 2],
            "p75": xs_sorted[(3 * n) // 4],
            "max": xs_sorted[-1],
            "mean": sum(xs_sorted) / n,
        }

    return {
        "conf": conf,
        "year": year,
        "n_papers": len(rows),
        "status_breakdown": status_counter.most_common(),
        "track_breakdown": track_counter.most_common(),
        "rating_summary": _summary(ratings),
        "citation_summary": _summary([float(c) for c in cites]),
    }


def top_papers(
    conn: sqlite3.Connection,
    *,
    conf: str,
    year: int,
    by: str = "gs_citation",
    top_k: int = 20,
    exclude_rejected: bool = True,
) -> dict:
    by_col = {
        "gs_citation": "gs_citation",
        "rating": "rating_avg",
        "rating_avg": "rating_avg",
    }.get(by, "gs_citation")
    excl_sql, excl_params = _exclude_rejected_filter(exclude_rejected, alias="")
    rows = conn.execute(
        f"""
        SELECT conf, year, paper_id, title, authors, status, track, site,
               openreview, rating_avg, gs_citation
        FROM papers
        WHERE conf=? AND year=? AND {by_col} IS NOT NULL
          {excl_sql}
        ORDER BY {by_col} DESC
        LIMIT ?
        """,
        (conf.lower(), year, *excl_params, top_k),
    ).fetchall()
    return {
        "conf": conf,
        "year": year,
        "ranked_by": by_col,
        "exclude_rejected": exclude_rejected,
        "results": [dict(r) for r in rows],
    }
