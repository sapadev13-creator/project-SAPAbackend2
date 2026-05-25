"""
Indeks frasa keyword — hindari loop 18k+ frasa per teks (Excel batch).
"""

from __future__ import annotations

from collections import defaultdict

from sapa_api.fuzzy_match import fuzzy_phrase_match


def build_phrase_index(
    trait_keywords: dict[str, list[str]],
) -> dict[str, list[tuple[str, str]]]:
    """Kelompokkan frasa multi-kata menurut kata pertama (sorted: terpanjang dulu)."""
    by_first: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for trait, phrases in trait_keywords.items():
        for phrase in phrases:
            if " " not in phrase:
                continue
            first = phrase.split()[0]
            by_first[first].append((trait, phrase))
    for first in by_first:
        by_first[first].sort(key=lambda x: (-len(x[1]), x[1]))
    return dict(by_first)


def match_trait_phrases(
    text_lower: str,
    by_first: dict[str, list[tuple[str, str]]],
    *,
    phrase_matches_variant,
    fuzzy_match,
    use_fuzzy: bool = True,
) -> list[tuple[str, str]]:
    """
    Kembalikan (trait, phrase) yang cocok.
    Hanya memeriksa frasa yang kata pertamanya muncul di teks.
    """
    matched: list[tuple[str, str]] = []
    seen: set[str] = set()

    for first, candidates in by_first.items():
        if first not in text_lower:
            continue
        for trait, phrase in candidates:
            if phrase in seen:
                continue
            if phrase in text_lower:
                matched.append((trait, phrase))
                seen.add(phrase)
                continue
            if phrase_matches_variant(phrase, text_lower):
                matched.append((trait, phrase))
                seen.add(phrase)
                continue
            if use_fuzzy and fuzzy_match(text_lower, phrase):
                matched.append((trait, phrase))
                seen.add(phrase)

    return matched
