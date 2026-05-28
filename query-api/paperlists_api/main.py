"""FastAPI service exposing the paperlists corpus over HTTPS.

Designed for hosted deployment (Railway / HF Spaces / Fly.io) so that
casual users — including AI agents via the MCP wrapper — can query the
full corpus without downloading the ~830MB of raw JSON.

Endpoints are intentionally compact and trend-focused, since plain
keyword search is already well served by papercopilot.com.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__, queries, ratelimit
from .db import DB_PATH, connect
from .queries import FTSQueryError, TooManyMatchesError

API_TITLE = "Paperlists Query API"
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MAX_OFFSET = 10_000
_MATCH_MODE_PATTERN = f"^({'|'.join(queries.MATCH_MODES)})$"


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_info() -> dict:
    """Expose enough deploy identity for clients to reject stale endpoints."""
    return {
        "version": __version__,
        "git_sha": _first_env(
            "PAPERLISTS_GIT_SHA",
            "RAILWAY_GIT_COMMIT_SHA",
            "GIT_COMMIT_SHA",
            "SOURCE_VERSION",
        ),
        "git_branch": _first_env(
            "PAPERLISTS_GIT_BRANCH",
            "RAILWAY_GIT_BRANCH",
            "GIT_BRANCH",
        ),
        "deployment_id": _first_env(
            "PAPERLISTS_DEPLOYMENT_ID",
            "RAILWAY_DEPLOYMENT_ID",
            "RAILWAY_SNAPSHOT_ID",
        ),
        "environment": _first_env(
            "RAILWAY_ENVIRONMENT",
            "RAILWAY_ENVIRONMENT_NAME",
            "PAPERLISTS_ENVIRONMENT",
        ),
    }

app = FastAPI(
    title=API_TITLE,
    version=__version__,
    description=(
        "AI-native query layer over the papercopilot/paperlists corpus. "
        "Trend-focused endpoints (topic_trend, topic_evolution, "
        "compare_periods, author_trajectory, field_landscape) are first-class — "
        "raw keyword search remains available as /v1/search."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(FTSQueryError)
async def _fts_query_error_handler(request: Request, exc: FTSQueryError):
    return JSONResponse(
        {"error": "invalid_query", "detail": str(exc)},
        status_code=400,
    )


@app.exception_handler(TooManyMatchesError)
async def _too_many_matches_error_handler(request: Request, exc: TooManyMatchesError):
    return JSONResponse(
        {
            "error": "too_many_matches",
            "detail": str(exc),
            "matches": exc.matches,
            "max_matches": exc.max_matches,
            "endpoint": exc.endpoint,
        },
        status_code=400,
    )

# ---------- Cross-worker rate limiter ----------
# Backed by a tiny sqlite file (see paperlists_api/ratelimit.py). All
# uvicorn workers see the same bucket for the same IP, so the per-IP quota
# survives a 4-worker Railway deployment. Single-worker setups work too —
# the sqlite file is just an in-process map at that point.
#
# Trust-proxy resolution:
#   "1"/"true"/"yes"   → always trust X-Forwarded-For
#   "0"/"false"/"no"   → never trust XFF
#   "auto" (default)   → trust if we detect a known trusted-proxy host
#                        (Railway, HF Spaces, Fly, Render, Vercel, Cloud Run,
#                        Azure). Avoids the failure mode where
#                        all users collapse into one bucket because the
#                        operator forgot to set PAPERLISTS_TRUST_PROXY=1 on
#                        a Railway deploy.
_TRUST_PROXY_ENV = os.environ.get("PAPERLISTS_TRUST_PROXY", "auto").lower()
# Heuristic: these env vars are auto-injected by the platform's edge proxy.
# If any is present we are behind that platform's proxy and XFF is
# trustworthy. Operators on platforms NOT in this list MUST set
# PAPERLISTS_TRUST_PROXY=1 explicitly — otherwise every visitor will share
# one rate-limit bucket because req.client.host is the proxy's address.
_PLATFORM_PROXY_MARKERS = (
    "RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID",    # Railway
    "SPACE_ID",                                       # HuggingFace Spaces
    "FLY_APP_NAME",                                   # Fly.io
    "RENDER",                                         # Render
    "VERCEL",                                         # Vercel
    "K_SERVICE", "K_REVISION",                        # Google Cloud Run / Knative
    "WEBSITE_SITE_NAME",                              # Azure App Service
)
_DETECTED_PROXY_NAME = next(
    (name for name in _PLATFORM_PROXY_MARKERS if os.environ.get(name)), None
)
if _TRUST_PROXY_ENV in ("1", "true", "yes"):
    _TRUST_PROXY = True
elif _TRUST_PROXY_ENV in ("0", "false", "no"):
    _TRUST_PROXY = False
else:  # "auto" (default) or anything unrecognized
    _TRUST_PROXY = bool(_DETECTED_PROXY_NAME)


def _client_ip(req: Request) -> str:
    """Return the IP to bucket against.

    Critical: this MUST NOT trust user-supplied X-Forwarded-For unless we
    know we're behind a proxy that overwrites it. Uvicorn is started
    without `--forwarded-allow-ips=*`, so its default (trust 127.0.0.1
    only) keeps `req.client.host` honest. When `_TRUST_PROXY` is set
    (explicitly via env or auto-detected from a platform marker), we read
    Railway documents `X-Real-IP` as the client IP header; many other hosts
    use XFF. Prefer `X-Real-IP`, then take the left-most XFF entry.
    """
    if _TRUST_PROXY:
        real_ip = req.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip() or "unknown"
        fwd = req.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip() or "unknown"
    return req.client.host if req.client else "unknown"


def _validate_year_range(start: Optional[int], end: Optional[int], label: str) -> None:
    if start is not None and end is not None and start > end:
        raise HTTPException(
            status_code=400,
            detail=f"{label}_from must be <= {label}_to",
        )


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in ("/", "/healthz", "/v1/coverage", "/v1/corpus_manifest"):
        return await call_next(request)
    ip = _client_ip(request)
    allowed, retry = ratelimit.check_and_consume(ip)
    if not allowed:
        return JSONResponse(
            {"error": "rate_limited", "retry_after_sec": retry},
            status_code=429,
            headers={"Retry-After": str(retry)},
        )
    return await call_next(request)


# ---------- Routes ----------


@app.get("/")
def root():
    return {
        "name": API_TITLE,
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/v1/coverage",
            "/v1/corpus_manifest",
            "/v1/search",
            "/v1/paper/{conf}/{paper_id}",
            "/v1/topic_trend",
            "/v1/topic_evolution",
            "/v1/author_trajectory",
            "/v1/field_landscape",
            "/v1/compare_periods",
            "/v1/conference_stats/{conf}/{year}",
            "/v1/top_papers/{conf}/{year}",
        ],
        "source": "https://github.com/papercopilot/paperlists",
        "api": _build_info(),
    }


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "db": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "api": _build_info(),
    }


@app.get("/v1/coverage")
def coverage():
    with connect() as conn:
        return queries.coverage(conn)


@app.get("/v1/corpus_manifest")
def corpus_manifest():
    with connect() as conn:
        manifest = queries.corpus_manifest(conn)
    manifest["api"] = _build_info()
    return manifest


_RAW_DESC = (
    "If true, pass the query string to FTS5 verbatim — enables operators "
    "(`foo OR bar`, `\"exact phrase\"`, `title:diffusion`, `reason*`) at "
    "the cost of HTTP 400 on syntax errors. Default false: input is "
    "tokenized into one safe quoted phrase."
)
_MATCH_MODE_DESC = (
    "Non-raw matching strategy. phrase is high precision and the default; "
    "token_and is broader; alias_or expands known acronym/name variants "
    "(for example RAG OR retrieval augmented generation)."
)


@app.get("/v1/search")
def search(
    q: str = Query(..., min_length=1, max_length=200, description="Query string. By default input is treated as one safe phrase; set raw=true for full FTS5 syntax."),
    conferences: Optional[str] = Query(None, description="Comma-separated conf list, e.g. 'iclr,nips,icml'."),
    year_from: Optional[int] = Query(None, ge=1990, le=2100),
    year_to: Optional[int] = Query(None, ge=1990, le=2100),
    exclude_rejected: bool = Query(True),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0, le=MAX_OFFSET, description="Pagination offset. Capped to keep deep FTS pagination bounded."),
    order_by: str = Query("relevance", pattern="^(relevance|year_desc|citation_desc|rating_desc)$"),
    include_abstract: bool = Query(False, description="Include abstract in each result. Off by default to control egress."),
    raw: bool = Query(False, description=_RAW_DESC),
    match_mode: str = Query(queries.MATCH_MODE_PHRASE, pattern=_MATCH_MODE_PATTERN, description=_MATCH_MODE_DESC),
):
    _validate_year_range(year_from, year_to, "year")
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.search_papers(
            conn,
            q=q, conferences=confs,
            year_from=year_from, year_to=year_to,
            exclude_rejected=exclude_rejected,
            limit=limit, offset=offset, order_by=order_by,
            include_abstract=include_abstract,
            raw=raw, match_mode=match_mode,
        )


@app.get("/v1/paper/{conf}/{paper_id:path}")
def get_paper(conf: str, paper_id: str):
    with connect() as conn:
        paper = queries.get_paper(conn, conf, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@app.get("/v1/topic_trend")
def topic_trend(
    q: str = Query(..., min_length=1, max_length=200),
    conferences: Optional[str] = Query(None),
    year_from: Optional[int] = Query(None, ge=1990, le=2100),
    year_to: Optional[int] = Query(None, ge=1990, le=2100),
    exclude_rejected: bool = Query(True),
    raw: bool = Query(False, description=_RAW_DESC),
    match_mode: str = Query(queries.MATCH_MODE_PHRASE, pattern=_MATCH_MODE_PATTERN, description=_MATCH_MODE_DESC),
):
    _validate_year_range(year_from, year_to, "year")
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.topic_trend(
            conn, q=q, conferences=confs,
            year_from=year_from, year_to=year_to,
            exclude_rejected=exclude_rejected, raw=raw,
            match_mode=match_mode,
        )


@app.get("/v1/topic_evolution")
def topic_evolution(
    q: str = Query(..., min_length=1, max_length=200),
    year_from: int = Query(..., ge=1990, le=2100),
    year_to: int = Query(..., ge=1990, le=2100),
    window: int = Query(1, ge=1, le=5),
    top_k: int = Query(15, ge=1, le=50),
    conferences: Optional[str] = Query(None),
    exclude_rejected: bool = Query(True),
    raw: bool = Query(False, description=_RAW_DESC),
    match_mode: str = Query(queries.MATCH_MODE_PHRASE, pattern=_MATCH_MODE_PATTERN, description=_MATCH_MODE_DESC),
):
    _validate_year_range(year_from, year_to, "year")
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.topic_evolution(
            conn, q=q, year_from=year_from, year_to=year_to,
            window=window, top_k=top_k, conferences=confs,
            exclude_rejected=exclude_rejected, raw=raw,
            match_mode=match_mode,
        )


@app.get("/v1/author_trajectory")
def author_trajectory(
    name: str = Query(..., min_length=2, max_length=120),
    conferences: Optional[str] = Query(None),
    year_from: Optional[int] = Query(None, ge=1990, le=2100),
    year_to: Optional[int] = Query(None, ge=1990, le=2100),
    exclude_rejected: bool = Query(True),
):
    _validate_year_range(year_from, year_to, "year")
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.author_trajectory(
            conn, name=name, conferences=confs,
            year_from=year_from, year_to=year_to,
            exclude_rejected=exclude_rejected,
        )


@app.get("/v1/field_landscape")
def field_landscape(
    q: str = Query(..., min_length=1, max_length=200),
    year: int = Query(..., ge=1990, le=2100),
    top_k: int = Query(10, ge=1, le=50),
    conferences: Optional[str] = Query(None),
    exclude_rejected: bool = Query(True),
    raw: bool = Query(False, description=_RAW_DESC),
    match_mode: str = Query(queries.MATCH_MODE_PHRASE, pattern=_MATCH_MODE_PATTERN, description=_MATCH_MODE_DESC),
):
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.field_landscape(
            conn, q=q, year=year, top_k=top_k,
            conferences=confs, exclude_rejected=exclude_rejected,
            raw=raw, match_mode=match_mode,
        )


@app.get("/v1/compare_periods")
def compare_periods(
    q: str = Query(..., min_length=1, max_length=200),
    period_a_from: int = Query(..., ge=1990, le=2100),
    period_a_to: int = Query(..., ge=1990, le=2100),
    period_b_from: int = Query(..., ge=1990, le=2100),
    period_b_to: int = Query(..., ge=1990, le=2100),
    top_k: int = Query(15, ge=1, le=50),
    conferences: Optional[str] = Query(None),
    exclude_rejected: bool = Query(True),
    raw: bool = Query(False, description=_RAW_DESC),
    match_mode: str = Query(queries.MATCH_MODE_PHRASE, pattern=_MATCH_MODE_PATTERN, description=_MATCH_MODE_DESC),
):
    _validate_year_range(period_a_from, period_a_to, "period_a")
    _validate_year_range(period_b_from, period_b_to, "period_b")
    confs = [c.strip().lower() for c in conferences.split(",")] if conferences else None
    with connect() as conn:
        return queries.compare_periods(
            conn, q=q,
            period_a=(period_a_from, period_a_to),
            period_b=(period_b_from, period_b_to),
            top_k=top_k,
            conferences=confs,
            exclude_rejected=exclude_rejected,
            raw=raw,
            match_mode=match_mode,
        )


@app.get("/v1/conference_stats/{conf}/{year}")
def conference_stats(conf: str, year: int):
    with connect() as conn:
        return queries.conference_stats(conn, conf=conf, year=year)


@app.get("/v1/top_papers/{conf}/{year}")
def top_papers(
    conf: str, year: int,
    by: str = Query("gs_citation", pattern="^(gs_citation|rating|rating_avg)$"),
    top_k: int = Query(20, ge=1, le=100),
    exclude_rejected: bool = Query(True, description="Exclude Reject/Withdraw/Desk Reject from the ranking. Set False to inspect raw rejects for OpenReview venues."),
):
    with connect() as conn:
        return queries.top_papers(
            conn, conf=conf, year=year, by=by,
            top_k=top_k, exclude_rejected=exclude_rejected,
        )
