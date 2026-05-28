# paperlists-api

A small FastAPI service that exposes the [papercopilot/paperlists](https://github.com/papercopilot/paperlists) corpus over HTTPS or localhost, so AI agents (via the companion MCP server, the bundled Skill, or any HTTP client) can run **trend-focused** queries without loading hundreds of JSON files on every call.

## What it does

Built around the observation that flat keyword search is already well served by [papercopilot.com](https://papercopilot.com). The API's first-class verbs are about **how research areas evolve**:

| Endpoint | Use case |
|---|---|
| `GET /v1/topic_trend` | Yearly paper count + citation-weighted volume for a topic |
| `GET /v1/topic_evolution` | Per-year (or per-window) top co-occurring keywords + landmark papers |
| `GET /v1/compare_periods` | Diff a topic across two year ranges — what emerged / faded / sustained |
| `GET /v1/author_trajectory` | Papers by an author across years |
| `GET /v1/field_landscape` | Single-year snapshot for a field: top papers, authors, affiliations, keywords |
| `GET /v1/conference_stats/{conf}/{year}` | Acceptance breakdown + rating/citation summary stats |
| `GET /v1/top_papers/{conf}/{year}` | Ranked by citation or rating |
| `GET /v1/search` | Standard FTS5 search (title/abstract/keywords/authors) |
| `GET /v1/paper/{conf}/{paper_id}` | Single record (full schema, including abstract) |
| `GET /v1/coverage` | What conferences/years are indexed |
| `GET /v1/corpus_manifest` | Corpus freshness/provenance contract for the data pipeline |

Trend-style endpoints and `author_trajectory` default to
`exclude_rejected=true`, matching `/v1/search` so survey counts do not mix
accepted/poster papers with rejected or withdrawn submissions. Set
`exclude_rejected=false` for raw corpus diagnostics. Endpoints that summarize
a field (`topic_trend`, `topic_evolution`, `compare_periods`,
`field_landscape`, `author_trajectory`, and `search`) also accept
`conferences=iclr,nips,icml` style comma-separated filters where relevant.

Implementation: sqlite FTS5 over a flattened `papers` table built from the repo's JSON files. Index lives in `papers.db`. Build it locally with:

```bash
cd query-api
uv run python -m paperlists_api.indexer /path/to/paperlists ./papers.db
PAPERLISTS_DB=$PWD/papers.db uv run uvicorn paperlists_api.main:app --reload
```

Then `open http://localhost:8000/docs` for the interactive API browser.

By default, free-text queries are tokenized into one safe FTS5 phrase. This is
stricter than token-AND matching, but it avoids fabricating historical trends
for emerging multi-word topics such as "test time scaling". Set `raw=true`
when you intentionally need FTS5 operators or broader boolean matching.

For Skill verification, use the same API through the bundled client:

```bash
cd ../skill
PAPERLISTS_API_URL=http://127.0.0.1:8000 python3 scripts/paperlists.py coverage
PAPERLISTS_API_URL=http://127.0.0.1:8000 python3 scripts/paperlists.py corpus_manifest
PAPERLISTS_API_URL=http://127.0.0.1:8000 python3 scripts/paperlists.py topic_evolution q="agent" year_from=2020 year_to=2025 window=1
```

## Deployment

### Railway
- Contributor-hosted live demo for evaluation:
  `https://api-production-18d3.up.railway.app`.
- Use the repo root as the Railway upload/build root.
- The Docker build fetches the paperlists JSON corpus from GitHub inside Railway and builds `papers.db` there. This avoids uploading hundreds of MB of tracked JSON through `railway up`.
- Railway reads root `railway.json` and root `Dockerfile`.
- Runtime defaults to **4 Uvicorn workers** (`WEB_CONCURRENCY=4`). The rate-limit state is sqlite-backed (`paperlists_api/ratelimit.py`, separate writable file at `/tmp/paperlists-ratelimit.db`) so all workers share one bucket per IP. No risk of `N×limit` bypass from sticky worker routing.
- `PAPERLISTS_TRUST_PROXY` defaults to `"auto"`, which auto-enables proxy-header trust whenever a known platform marker is present in the env (Railway / HF Spaces / Fly / Render / Vercel / Cloud Run / Azure App Service). Set it to `"1"` or `"0"` to override. On any host **not** in that list, set `PAPERLISTS_TRUST_PROXY=1` explicitly or every visitor will share one rate-limit bucket because `req.client.host` resolves to the proxy address. The app prefers `X-Real-IP` (Railway) and falls back to the left-most `X-Forwarded-For`. Uvicorn is **not** started with `--forwarded-allow-ips=*` — that flag would let uvicorn itself rewrite `request.client.host` from XFF *before* the middleware runs, defeating the trust gate.
- Current Railway build indexed 237,735 papers from 292 source files into a
  ~623 MB sqlite DB.
- Free hobby tier (~$5/mo credit) is sufficient for the demo.
- Egress is the main cost driver. Three mitigations baked in:
  1. `include_abstract` defaults to `false` on `/v1/search` — abstracts are only sent on `/v1/paper/{conf}/{id}`.
  2. Token-bucket rate limiter (default 60 req/min/IP, burst 20). Tune via `PAPERLISTS_RATE_PER_MIN`, `PAPERLISTS_RATE_BURST`. Bucket size capped at 10k rows with 30 min stale eviction (`PAPERLISTS_RATE_BUCKET_MAX`, `PAPERLISTS_RATE_STALE_SEC`). GC sweeps run probabilistically (~1% of requests).
  3. Response field whitelist on cards — large fields like `bibtex`, `reviewers`, `or_profile` are never returned.
- Broad analysis endpoints (`topic_evolution`, `compare_periods`, and
  `field_landscape`) count matches before in-memory aggregation and return
  `400 {"error":"too_many_matches"}` above 50k rows. `/v1/search` caps
  `offset` at 10k to keep deep pagination from tying up workers.

### HF Spaces / Fly.io / self-hosted
Same Dockerfile, just point at a different host. The index step is the slow part (~1–2 min on cold build).

## Future / out of scope for v1
- Incremental re-indexing on JSON updates (current build is full rebuild)
- Affiliation normalization across `aff_unique_norm` variants
- Embedding-based semantic search (would require a vector DB; deferred)
- Cloudflare Workers + D1 port for near-zero idle cost
