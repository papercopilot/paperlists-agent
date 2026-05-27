# paperlists-mcp

An MCP (Model Context Protocol) server that exposes the [papercopilot/paperlists](https://github.com/papercopilot/paperlists) corpus to AI agents — Claude Code, Claude Desktop, Cursor, Codex, OpenAI ChatGPT (via MCP), Hermes, anything that speaks MCP.

A thin wrapper around any hosted [`paperlists-api`](../query-api/) service:
zero data download, zero local index. ~5 MB install footprint. Set
`PAPERLISTS_API_URL` to a papercopilot-owned, self-hosted, or demo API endpoint.

## Why MCP?

`papercopilot.com` already answers "find me this paper". The MCP server is built around the harder question agents actually ask: **how has this research area evolved?** Tools like `topic_trend`, `topic_evolution`, `compare_periods`, and `field_landscape` are first-class — they answer questions that would otherwise require dozens of manual searches.

## Install

```bash
# from this directory
pip install -e .
# or:
uvx --from . paperlists-mcp
```

`PAPERLISTS_API_URL` is required at runtime. For demo testing only, use
`https://api-production-18d3.up.railway.app`.

## Register with your MCP client

### Claude Code / Claude Desktop
Add to `~/.config/claude/claude_desktop_config.json` (or use `claude mcp add`):

```jsonc
{
  "mcpServers": {
    "paperlists": {
      "command": "uvx",
      "args": ["paperlists-mcp"],
      "env": {
        "PAPERLISTS_API_URL": "https://api-production-18d3.up.railway.app"
      }
    }
  }
}
```

### Cursor / Codex / others
Same `command` + `args`. Cursor reads `~/.cursor/mcp.json`; Codex reads `~/.codex/mcp.json` (key names vary slightly across hosts but the launch command is identical).

## Tools

| Tool | What it answers |
|---|---|
| `list_coverage` | What venues/years are indexed |
| `corpus_manifest` | Corpus freshness/provenance metadata for agent clients |
| `search_papers` | Plain FTS5 search across title/abstract/keywords/authors |
| `get_paper` | Fetch a single paper's full record (including abstract) |
| **`topic_trend`** | Yearly volume + citation-weight for a research area |
| **`topic_evolution`** | Per-window top keywords / venues / landmark papers — surfaces topic drift |
| **`compare_periods`** | Diff a topic across two year ranges: emerged / faded / sustained |
| `author_trajectory` | A researcher's papers grouped by year |
| `field_landscape` | Single-year snapshot of a field |
| `conference_stats` | Acceptance + rating/citation distribution |
| `top_papers` | Ranked by citation or rating |

`search_papers` and the trend-style tools default to `exclude_rejected=true`.
Use `exclude_rejected=false` only for raw corpus diagnostics. Topic tools also
accept comma-separated `conferences` filters such as `iclr,nips,icml`.

## Running against a local index

If you have the paperlists repo cloned and want to avoid the hosted API entirely (offline use, very large queries, custom modifications):

```bash
# in query-api/
python -m paperlists_api.indexer /path/to/paperlists papers.db
PAPERLISTS_DB=$PWD/papers.db uvicorn paperlists_api.main:app --port 8000 &
PAPERLISTS_API_URL=http://localhost:8000 paperlists-mcp
```

The MCP server's contract is identical against either backend.
The same local API override also works for the bundled Skill:

```bash
cd ../skill
PAPERLISTS_API_URL=http://127.0.0.1:8000 python3 scripts/paperlists.py coverage
```
