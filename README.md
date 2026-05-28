# paperlists-agent

AI-native query and agent tooling for the
[papercopilot/paperlists](https://github.com/papercopilot/paperlists)
conference-paper corpus.

This repo is the dedicated home for the agent layer that started in
[`papercopilot/paperlists#29`](https://github.com/papercopilot/paperlists/pull/29).
It keeps the data corpus and the agent/query product separate:

| Surface | Directory | Use it when |
|---|---|---|
| FastAPI query service | [`query-api/`](query-api/) | You want HTTPS or localhost access to the corpus |
| MCP server | [`mcp-server/`](mcp-server/) | You want Claude Code, Cursor, Codex, Claude Desktop, or another MCP host to query papers |
| Cross-tool Skill + CLI | [`skill/`](skill/) | You want a portable markdown skill and stdlib-only command-line client |

The core verbs are research-evolution oriented, not just keyword search:

- `topic_trend`: yearly topic volume and citation-weighted volume
- `topic_evolution`: per-year/per-window keywords, venues, and landmark papers
- `compare_periods`: emerged/faded/sustained terms, authors, and affiliations
- `author_trajectory`: papers by author across years
- `field_landscape`: single-year field snapshot
- `corpus_manifest`: corpus freshness/provenance contract for the data pipeline

## Quick start

Use the hosted demo only for evaluation:

```bash
export PAPERLISTS_API_URL=https://api-production-18d3.up.railway.app
python3 skill/scripts/paperlists.py coverage
python3 skill/scripts/paperlists.py corpus_manifest  # confirm api.version/build identity
python3 skill/scripts/paperlists.py topic_evolution q="LLM reasoning" year_from=2024 year_to=2025 conferences=iclr,nips,icml,acl,emnlp match_mode=token_and
```

For longitudinal claims, require `corpus_manifest.api.version >= 0.2.0` (or a
known deploy git SHA). Older demos used token-AND query semantics without
`match_mode`, `query_expression`, `venue_diff`, or query-noise metadata.

For a local API:

```bash
cd query-api
uv run python -m paperlists_api.indexer /path/to/paperlists ./papers.db
PAPERLISTS_DB=$PWD/papers.db uv run uvicorn paperlists_api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs`.

## Deploy

The root `Dockerfile` fetches the upstream paperlists JSON archive during build
and bakes a sqlite FTS5 index into the runtime image. This avoids committing
or uploading the raw data.

Railway can deploy from the repo root:

```bash
railway up
```

Runtime knobs:

- `WEB_CONCURRENCY=4`
- `PAPERLISTS_RATE_PER_MIN=60`
- `PAPERLISTS_RATE_BURST=20`
- `PAPERLISTS_TRUST_PROXY=auto`
- `PAPERLISTS_DB=/app/papers.db`

## Development

```bash
cd query-api
uv run --extra dev pytest -q
uv run python -m compileall paperlists_api ../mcp-server/paperlists_mcp ../skill/scripts/paperlists.py
```

The API currently indexes 237,735 papers across 31 venues in the hosted demo.
Local `papers.db` files are generated artifacts and must not be committed.
