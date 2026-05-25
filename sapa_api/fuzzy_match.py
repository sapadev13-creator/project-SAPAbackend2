"""Koreksi typo: cocokkan token/frasa ke vocab terdekat."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from sapa_api import state
from sapa_api.config import (
    FUZZY_MIN_RATIO,
    FUZZY_PHRASE_MIN_RATIO,
    FUZZY_SHORT_MIN_RATIO,
)
from sapa_api.crisis import CRISIS_CRITICAL_PHRASES, CRISIS_HIGH_PHRASES
from sapa_api.text_utils import (
    POSITIVE_CONTEXT_PHRASES,
    STOPWORDS_ID,
    is_meaningful_token,
    tokenize,
)

# Kata pendek penting (krisis / OCEAN) — boleh dikoreksi meski < 4 huruf
SHORT_FUZZY_WHITELIST = frozenset({
    "bunuh", "diri", "mati", "ingin", "pengen", "mau", "akan", "putus", "asa",
    "benci", "sedih", "marah", "takut", "cemas", "suka", "senang",
})


@dataclass
class TypoCorrection:
    original: str
    corrected: str
    confidence: float


@dataclass
class NormalizedText:
    original: str
    normalized: str
    corrections: list[TypoCorrection] = field(default_factory=list)

    @property
    def had_typos(self) -> bool:
        return len(self.corrections) > 0


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _collect_vocabulary() -> tuple[set[str], list[str]]:
    words: set[str] = set(SHORT_FUZZY_WHITELIST)

    try:
        from sapa_api.keywords import TRAIT_KEYWORDS
        for kws in TRAIT_KEYWORDS.values():
            for kw in kws:
                if " " in kw:
                    words.update(tokenize(kw))
                elif len(kw) >= 3:
                    words.add(kw)
    except Exception:
        pass

    for phrase in POSITIVE_CONTEXT_PHRASES + CRISIS_CRITICAL_PHRASES + CRISIS_HIGH_PHRASES:
        words.update(tokenize(phrase))

    if state.ONT_META:
        for meta in state.ONT_META:
            words.update(t for t in str(meta["lexeme"]).split("_") if len(t) >= 3)

    phrases: list[str] = list(POSITIVE_CONTEXT_PHRASES) + list(CRISIS_CRITICAL_PHRASES)
    phrases += list(CRISIS_HIGH_PHRASES)
    try:
        from sapa_api.keywords import TRAIT_KEYWORDS
        for kws in TRAIT_KEYWORDS.values():
            for kw in kws:
                if " " in kw:
                    phrases.append(kw)
    except Exception:
        pass

    return words, sorted(set(phrases), key=len, reverse=True)


def build_fuzzy_index():
    """Bangun indeks vocab (dipanggil saat startup)."""
    words, phrases = _collect_vocabulary()
    state.FUZZY_VOCAB = words
    state.FUZZY_PHRASES = phrases

    index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for w in words:
        if not w:
            continue
        key = (w[0], len(w))
        index[key].append(w)
        for delta in (-1, 1):
            index[(w[0], len(w) + delta)].append(w)
    state.FUZZY_INDEX = dict(index)
    return len(words), len(phrases)


def _candidate_pool(token: str) -> list[str]:
    if not state.FUZZY_INDEX or not state.FUZZY_VOCAB:
        return []
    if token in state.FUZZY_VOCAB:
        return [token]

    pool: list[str] = []
    if token:
        for length in (len(token) - 1, len(token), len(token) + 1):
            if length < 3:
                continue
            pool.extend(state.FUZZY_INDEX.get((token[0], length), []))
    return list(set(pool))


def _pattern_typo_fix(token: str) -> tuple[str | None, float]:
    """Pola typo umum: huruf ganda, dll."""
    if len(token) >= 4 and token[-1] == token[-2]:
        trimmed = token[:-1]
        if trimmed in state.FUZZY_VOCAB:
            return trimmed, 0.92
    if len(token) >= 5 and token[-2] == token[-1]:
        trimmed = token[:-1]
        if trimmed in state.FUZZY_VOCAB:
            return trimmed, 0.9
    return None, 0.0


def closest_word(token: str) -> tuple[str | None, float]:
    if token in STOPWORDS_ID:
        return token, 1.0

    fixed, conf = _pattern_typo_fix(token)
    if fixed:
        return fixed, conf

    allow_short = token in SHORT_FUZZY_WHITELIST or len(token) >= 4
    if not allow_short and len(token) < 4:
        return None, 0.0

    min_ratio = FUZZY_SHORT_MIN_RATIO if len(token) <= 5 else FUZZY_MIN_RATIO
    best, best_score = None, 0.0
    pool = set(_candidate_pool(token))
    if token in state.FUZZY_VOCAB:
        pool.add(token)

    for cand in pool:
        if abs(len(cand) - len(token)) > 2 and len(token) > 4:
            continue
        score = _ratio(token, cand)
        if score < min_ratio:
            continue
        if best is None:
            best, best_score = cand, score
            continue
        if score > best_score + 0.02:
            best, best_score = cand, score
        elif score >= best_score - 0.01 and len(cand) < len(best):
            best, best_score = cand, score

    if best and best != token and best_score >= min_ratio:
        return best, best_score
    if token in state.FUZZY_VOCAB:
        return token, 1.0
    return None, 0.0


def correct_tokens(tokens: list[str]) -> tuple[list[str], list[TypoCorrection]]:
    corrected: list[str] = []
    fixes: list[TypoCorrection] = []

    for tok in tokens:
        match, conf = closest_word(tok)
        if match and match != tok:
            corrected.append(match)
            fixes.append(TypoCorrection(tok, match, round(conf, 3)))
        else:
            corrected.append(tok)

    return corrected, fixes


def normalize_text_typos(text: str) -> NormalizedText:
    original = text
    tokens = tokenize(text)
    if not tokens:
        return NormalizedText(original=original, normalized=original)

    corrected_tokens, fixes = correct_tokens(tokens)
    normalized = " ".join(corrected_tokens)

    # Koreksi frasa utuh (sliding window) jika token sudah hampir benar
    normalized, phrase_fixes = _correct_phrases_in_text(normalized)
    fixes.extend(phrase_fixes)

    return NormalizedText(
        original=original,
        normalized=normalized,
        corrections=fixes,
    )


def _correct_phrases_in_text(text_lower: str) -> tuple[str, list[TypoCorrection]]:
    if not state.FUZZY_PHRASES:
        return text_lower, []

    tokens = tokenize(text_lower)
    fixes: list[TypoCorrection] = []
    changed = tokens[:]

    for phrase in state.FUZZY_PHRASES:
        if phrase in text_lower:
            continue
        parts = phrase.split()
        n = len(parts)
        if n < 2:
            continue
        for i in range(len(tokens) - n + 1):
            window = " ".join(tokens[i : i + n])
            if _ratio(window, phrase) >= FUZZY_PHRASE_MIN_RATIO and window != phrase:
                fixes.append(TypoCorrection(window, phrase, round(_ratio(window, phrase), 3)))
                # replace span in changed
                changed = (
                    changed[:i]
                    + parts
                    + changed[i + n :]
                )
                tokens = changed
                text_lower = " ".join(tokens)
                break

    return " ".join(changed), fixes


def fuzzy_phrase_match(text_lower: str, phrase: str) -> bool:
    """Cek frasa exact atau fuzzy di teks."""
    if phrase in text_lower:
        return True
    parts = phrase.split()
    n = len(parts)
    tokens = tokenize(text_lower)
    if len(tokens) < n:
        return False
    for i in range(len(tokens) - n + 1):
        window = " ".join(tokens[i : i + n])
        if _ratio(window, phrase) >= FUZZY_PHRASE_MIN_RATIO:
            return True
    return False


def fuzzy_token_in_counter(token: str, counter: dict) -> int:
    """Hitung frekuensi token + varian typo di counter."""
    total = counter.get(token, 0)
    if token in state.FUZZY_VOCAB:
        return total
    match, conf = closest_word(token)
    if match and match != token and conf >= FUZZY_MIN_RATIO:
        return total + counter.get(match, 0)
    return total
