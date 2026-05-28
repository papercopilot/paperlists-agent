"""FTS query sanitization and explicit match-mode helpers."""
from __future__ import annotations

import re

MATCH_MODE_PHRASE = "phrase"
MATCH_MODE_TOKEN_AND = "token_and"
MATCH_MODE_ALIAS_OR = "alias_or"
MATCH_MODES = (MATCH_MODE_PHRASE, MATCH_MODE_TOKEN_AND, MATCH_MODE_ALIAS_OR)

# FTS5 input is treated as a single quoted phrase after tokenization.
# This neutralizes the FTS5 query language entirely in default mode.
_FTS_KEEP = re.compile(r"[^\w\s]+", flags=re.UNICODE)

# Canonical alias phrase sets. Each group is bi-directional and phrase-level:
# if any variant appears in a query, `alias_or` searches all variants.
ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("llm", "large language model"),
    ("vit", "vision transformer"),
    ("rag", "retrieval augmented generation"),
    ("cot", "chain of thought"),
    ("moe", "mixture of experts"),
    ("ssm", "state space model"),
    ("rl", "reinforcement learning"),
    (
        "rlhf",
        "reinforcement learning from human feedback",
        "reinforcement learning human feedback",
    ),
    ("vlm", "vision language model"),
    ("mllm", "multimodal large language model"),
    ("nerf", "neural radiance field"),
    ("gnn", "graph neural network"),
    ("vqa", "visual question answering"),
)


def normalize_stem(t: str) -> str:
    """Cheap singular-form normalizer: enough for query/keyword dedup."""
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def fts_tokens(q: str) -> list[str]:
    """Split user text into FTS5-safe tokens.

    Punctuation is dropped; hyphens become whitespace so `in-context` does not
    read as the FTS5 NOT operator.
    """
    if not q:
        return []
    cleaned = _FTS_KEEP.sub(" ", q.replace("-", " "))
    return [t for t in cleaned.split() if t]


def _token_stems(q: str) -> list[str]:
    return [normalize_stem(t.lower()) for t in fts_tokens(q)]


def sanitize_fts(q: str) -> str:
    """Turn user text into one safe FTS5 phrase expression."""
    tokens = fts_tokens(q)
    if not tokens:
        return ""
    return f'"{" ".join(tokens)}"'


def sanitize_fts_token_and(q: str) -> str:
    """Turn user text into a safe token-AND FTS5 expression."""
    return " ".join(f'"{t}"' for t in fts_tokens(q))


def sanitize_fts_alias_or(q: str) -> str:
    """Build a safe OR query over known acronym/name aliases."""
    variants = [sanitize_fts(v) for v in query_alias_variants(q)]
    variants = [v for v in variants if v]
    return " OR ".join(variants)


def sanitize_fts_phrase_in(column: str, q: str) -> str:
    """Build an FTS5 column-scoped phrase: `column:"tok1 tok2"`."""
    tokens = fts_tokens(q)
    if not tokens:
        return ""
    return f'{column}:"{" ".join(tokens)}"'


def query_alias_variants(q: str) -> list[str]:
    """Return safe query text variants implied by known alias groups."""
    tokens = fts_tokens(q)
    if not tokens:
        return []
    base = " ".join(tokens)
    base_key = " ".join(normalize_stem(t.lower()) for t in tokens)
    variants: dict[str, str] = {base_key: base}
    lowered = [normalize_stem(t.lower()) for t in tokens]
    matches = []

    for group in ALIAS_GROUPS:
        tokenized_variants = [(variant, fts_tokens(variant)) for variant in group]
        stem_variants = [
            (variant, toks, [normalize_stem(t.lower()) for t in toks])
            for variant, toks in tokenized_variants
        ]
        for _variant, toks, stems in stem_variants:
            n = len(stems)
            for idx in range(0, len(tokens) - n + 1):
                if lowered[idx:idx + n] == stems:
                    matches.append((idx, idx + n, _variant, stem_variants))

    longest_matches = []
    for idx, end, variant, stem_variants in matches:
        if any(
            other_idx <= idx
            and other_end >= end
            and (other_end - other_idx) > (end - idx)
            for other_idx, other_end, _other_variant, _other_variants in matches
        ):
            continue
        longest_matches.append((idx, end, variant, stem_variants))
    longest_matches.sort(key=lambda m: m[0])

    def _add_variant(parts: list[str]) -> None:
        phrase = " ".join(parts)
        key = " ".join(normalize_stem(t.lower()) for t in fts_tokens(phrase))
        variants.setdefault(key, phrase)

    def _expand(match_idx: int, cursor: int, parts: list[str]) -> None:
        if len(variants) >= 32:
            return
        if match_idx >= len(longest_matches):
            _add_variant([*parts, *tokens[cursor:]])
            return

        idx, end, variant, stem_variants = longest_matches[match_idx]
        if idx < cursor:
            _expand(match_idx + 1, cursor, parts)
            return
        prefix = [*parts, *tokens[cursor:idx]]
        for replacement, _rtoks, _rstems in stem_variants:
            repl_tokens = tokens[idx:end] if replacement == variant else fts_tokens(replacement)
            _expand(match_idx + 1, end, [*prefix, *repl_tokens])

    _expand(0, 0, [])

    out = [base]
    out.extend(sorted(v for k, v in variants.items() if k != base_key))
    return out


def query_alias_meta(q: str, raw: bool, match_mode: str) -> dict:
    if raw:
        return {}
    aliases = query_alias_variants(q)
    if len(aliases) <= 1:
        return {}
    return {
        "query_aliases": aliases,
        "query_alias_expanded": match_mode == MATCH_MODE_ALIAS_OR,
    }
