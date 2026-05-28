---
name: paperlists
description: |
  Search and analyze how AI research has evolved over time, using the
  papercopilot/paperlists corpus (237k+ papers across 31 venues,
  2010-present). First-class verbs are trend-focused — topic_trend,
  topic_evolution, compare_periods, author_trajectory, field_landscape —
  not just keyword search. Backed by a hosted HTTPS API; no local data
  download required.
triggers:
  - "how has <topic> evolved"
  - "trend of <topic> over years"
  - "papers on <topic> in <conference>"
  - "who is publishing on <topic>"
  - "compare <topic> between <year_a> and <year_b>"
  - "state of <field> in <year>"
  - "top papers at <conference> <year>"
  - "rating distribution of <conference>"
  - "research evolution"
  - "literature survey"
---

# Paperlists — AI research evolution tracker

Use this skill whenever the user asks how a research area has changed
over time, who publishes in a field, what landmark papers exist in a
venue/year, or wants to do **literature-survey-style analysis** without
clicking through papercopilot.com / OpenReview manually.

## When to use

Strong fit:
- "How has retrieval-augmented generation evolved 2020–2025?"
- "Compare mixture-of-experts research between 2019–2021 and 2022–2024."
- "Top-cited diffusion model papers at NeurIPS 2023."
- "Yann LeCun's papers since 2020."
- "State of mechanistic interpretability in 2024."

Weak fit (other tools are better):
- A single paper lookup by exact title → just google it.
- Live ArXiv preprints → this corpus is conference-published only.
- Full-text PDF reading → fetch the `pdf` URL separately.

## How it works

The skill is backed by a hosted HTTPS API at
**`$PAPERLISTS_API_URL`**. For demo testing only, use
`https://api-production-18d3.up.railway.app`; production releases should point
at papercopilot-owned or self-hosted infrastructure.
You can call it three ways:

1. **MCP** — if `paperlists-mcp` is registered with your host, just use
   the tool names below. Prefer this when available.
2. **HTTP via the bundled script** — `scripts/paperlists.py <endpoint> [k=v ...]`.
   Use when MCP isn't configured.
3. **Direct curl** — for one-off ad-hoc queries.

For fully local/offline runs, start the query API against a local sqlite index
and point this skill at it:

```bash
cd query-api
python -m paperlists_api.indexer /path/to/paperlists ./papers.db
PAPERLISTS_DB=$PWD/papers.db uvicorn paperlists_api.main:app --port 8000
```

Then call the same script or MCP server with:

```bash
cd ../skill
PAPERLISTS_API_URL=http://127.0.0.1:8000 python3 scripts/paperlists.py coverage
```

The output contract is identical between Railway-hosted and local sqlite modes.

## Tools / endpoints

| Tool | Endpoint | Purpose |
|---|---|---|
| `list_coverage` | `GET /v1/coverage` | What's indexed — call first if uncertain |
| `corpus_manifest` | `GET /v1/corpus_manifest` | Corpus freshness/provenance metadata |
| `search_papers` | `GET /v1/search` | FTS5 search over title/abstract/keywords/authors |
| `get_paper` | `GET /v1/paper/{conf}/{paper_id}` | Single paper full record |
| `topic_trend` | `GET /v1/topic_trend` | Yearly volume + citation-weight for a query |
| `topic_evolution` | `GET /v1/topic_evolution` | Per-window top keywords, venues, landmark papers |
| `compare_periods` | `GET /v1/compare_periods` | Diff a topic across two year ranges |
| `author_trajectory` | `GET /v1/author_trajectory` | Researcher's papers grouped by year |
| `field_landscape` | `GET /v1/field_landscape` | Single-year snapshot of a field |
| `conference_stats` | `GET /v1/conference_stats/{conf}/{year}` | Acceptance + rating/citation summary |
| `top_papers` | `GET /v1/top_papers/{conf}/{year}` | Ranked by citation or rating |

Common params: `q` (query), `conferences` (comma list like `iclr,nips,icml`),
`year_from`/`year_to`, and `exclude_rejected` (default `true` for search,
trend, ranking, and `author_trajectory` endpoints — including `top_papers`). Pass
`exclude_rejected=false` only when you explicitly want raw corpus diagnostics
that include Reject / Withdraw entries. Abstracts are off by default to control
egress — pass `include_abstract=true` only if you need the full text. Query
endpoints also accept `match_mode=phrase` (default) or `match_mode=token_and`
for broader sensitivity checks. Use `match_mode=alias_or` when the direction has
common acronym/name variants, such as `RAG` and `retrieval augmented generation`.

### Response shape notes

- `corpus_manifest` returns `{built_at, total_papers, conferences, sources, pipeline_runs}`.
  Today `sources` is empty unless the sqlite DB contains the optional
  `source_manifest` table written by the data-fetching pipeline.
- `search_papers` returns `{total_matches, returned, offset, limit, has_more, results, ...}`.
  Use `has_more` and `offset` to paginate; never assume `results` is exhaustive.
  (`total` is kept as a back-compat alias for one release; prefer `total_matches`.)
  The hosted API caps `offset` at 10k and rejects overly broad searches with
  `too_many_matches` to protect the Railway deployment; narrow by phrase, year,
  venue, or aliases instead of asking for generic terms like `model`.
- Query endpoints echo `match_mode`, `query_expression`, and alias/filter
  metadata. Treat big differences between `phrase`, `token_and`, and `alias_or`
  as a sensitivity warning, not as interchangeable counts.
- `topic_evolution`, `compare_periods`, and `field_landscape` suppress keyword
  variants that merely restate the query. Inspect `query_noise_filter`,
  `suppressed_query_keywords`, or `keyword_diff_suppressed_query_terms` when
  you need to explain what was filtered.
- For longitudinal summaries, prefer `topic_evolution.keyword_drift.grew`,
  `.emerged`, and `.faded` over manually eyeballing every window. This is the
  intended surface for claims like "reinforcement learning grew 6 -> 91 under
  LLM reasoning"; then cite window-level `top_keywords` as supporting detail.
- `compare_periods` returns each period as `{years: [a, b], year_from, year_to, n_papers}`
  — both shapes are populated, pick whichever is more ergonomic. It also
  returns `venue_diff` alongside keyword, author, and affiliation diffs.
- `topic_evolution` adds `ranking_basis` to each window: `"gs_citation"` when
  citations are meaningful, `"rating_avg+status_fallback"` for the current year
  where citations are near-zero. Treat landmark order as a heuristic in the
  fallback regime.
- Bad FTS5 input (unbalanced quotes, raw operators) returns HTTP 400 with
  `{error: "invalid_query"}`. The MCP/skill wrappers turn this into a normal
  result object so agents can retry with a cleaner query.
- Overly broad analysis calls return HTTP 400 with `{error: "too_many_matches"}`.
  Narrow the year range, venues, or query before retrying.

## Worked patterns

### Pattern 1 — Research evolution in one shot
```bash
scripts/paperlists.py topic_evolution q="retrieval augmented generation" year_from=2020 year_to=2025 window=1 conferences=iclr,nips,icml,acl,emnlp
```
Returns per-year top keywords (e.g. dense passage → llm → multi-hop),
top venues (emnlp → iclr/nips), and landmark cited papers each year.
**This is usually the single best call** for "how did <field> evolve" questions.

### Pattern 2 — Topic drift between two eras
```bash
scripts/paperlists.py compare_periods q="vision transformer" period_a_from=2018 period_a_to=2020 period_b_from=2022 period_b_to=2024 conferences=iclr,nips,icml
```
Returns `emerged` / `faded` / `sustained` for keywords, authors, and
affiliations. Great for "what's new" or "what's been abandoned" narratives.

### Pattern 3 — Surveying a venue-year
```bash
scripts/paperlists.py top_papers conf=nips year=2023 by=gs_citation top_k=20
scripts/paperlists.py conference_stats conf=iclr year=2024
```

### Pattern 4 — Author-centric
```bash
scripts/paperlists.py author_trajectory name="Yann LeCun" year_from=2020 conferences=iclr,nips,icml
```
Use the canonical full name as it appears on publications. Very broad names
return `too_many_matches`; narrow by full name, year range, or venue. Each paper
includes `author_position` and `n_authors`, so down-rank senior-author tail
papers when the user asks what a researcher is directly driving.

## Efficiency tips for agents

- **Always `list_coverage` first** if you're unsure whether a venue/year is
  available. It's free and saves wasted calls.
- **Don't fetch abstracts unless you need them.** `include_abstract=true`
  multiplies response size ~30×.
- **Prefer `topic_evolution` over many `search_papers` calls** when the user
  wants temporal analysis — one call gives you the structured story.
- **Set `limit` aggressively low** (5–10) on exploratory `search_papers`
  calls; raise it only after the user confirms direction.
- **Default sanitizer**: hyphens are handled server-side
  ("in-context learning" works). Operators (`OR`, `NOT`, `NEAR`), prefix
  (`reason*`), column filters (`title:diffusion`), and explicit quote syntax
  are **NOT** parsed in the default mode — input is tokenized into one safe
  phrase. Use `match_mode=token_and` for a broader sensitivity check,
  `match_mode=alias_or` for known acronym/name variants, and pass `raw=true`
  only when you need full FTS5 syntax (malformed expressions then return HTTP 400).
- **For high-stakes longitudinal claims**, compare `match_mode=phrase` against
  `match_mode=token_and` and `match_mode=alias_or`. If they change the story
  materially, report the result as query-sensitive and inspect examples before
  making a strong claim.

## Bundled script

`scripts/paperlists.py` is a minimal Python CLI that talks to the same API
the MCP server uses. Useful when MCP isn't available. See `scripts/paperlists.py --help`.
