"""Keyword extraction and query-restatement filtering helpers."""
from __future__ import annotations

import re
from typing import Iterable

from .match_modes import ALIAS_GROUPS, fts_tokens, normalize_stem

# Stop-words to drop when extracting keyword/term drift.
STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or such that the their then there these to was were will with we our you your this they them than but not no any all can may use using used new novel model models method methods learning deep neural network networks based approach paper task tasks
""".split())


def query_stem_tokens(q: str) -> set[str]:
    """Return normalized tokens implied by the user's query and aliases."""
    if not q:
        return set()
    base = {normalize_stem(t) for t in fts_tokens(q.lower())}
    out = set(base)
    for group in ALIAS_GROUPS:
        variant_stems = [
            {normalize_stem(t) for t in fts_tokens(variant.lower())}
            for variant in group
        ]
        if any(stems.issubset(base) for stems in variant_stems):
            for stems in variant_stems:
                out |= stems
    return out


def is_query_noise(keyword: str, qstems: set[str]) -> bool:
    """True iff the keyword's content tokens are all implied by the query."""
    if not qstems:
        return False
    toks = {normalize_stem(t) for t in fts_tokens(keyword.lower())}
    if not toks:
        return False
    return toks.issubset(qstems)


def filter_query_noise(counted: list[tuple], qstems: set[str]) -> list[tuple]:
    """Filter [(term, count, ...)] tuples by `is_query_noise`."""
    if not qstems:
        return counted
    return [item for item in counted if not is_query_noise(item[0], qstems)]


def partition_query_noise(counted: list[tuple], qstems: set[str]) -> tuple[list[tuple], list[tuple]]:
    """Split counted keyword tuples into (kept, query-restatement noise)."""
    if not qstems:
        return counted, []
    kept = []
    suppressed = []
    for item in counted:
        if is_query_noise(item[0], qstems):
            suppressed.append(item)
        else:
            kept.append(item)
    return kept, suppressed


def query_noise_meta(qstems: set[str], suppressed: list[tuple]) -> dict:
    return {
        "query_noise_filter": {
            "enabled": bool(qstems),
            "query_stems": sorted(qstems),
            "suppressed_count": len(suppressed),
        }
    }


def tokenize_keywords(s: str) -> Iterable[str]:
    if not s:
        return []
    # paperlists separates keywords with semicolons; fall back to commas/newlines.
    parts = re.split(r"[;,\n]", s)
    out = []
    for p in parts:
        kw = p.strip().lower()
        if kw and kw not in STOPWORDS and len(kw) > 1:
            out.append(kw)
    return out


def tokenize_title_terms(s: str) -> Iterable[str]:
    """Fallback terms for rows whose source metadata lacks `keywords`."""
    if not s:
        return []
    out = []
    for tok in fts_tokens(s.lower()):
        if tok not in STOPWORDS and len(tok) > 2:
            out.append(tok)
    return out
