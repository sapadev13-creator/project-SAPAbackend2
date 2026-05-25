"""Deteksi & penyesuaian teks risiko tinggi (bunuh diri, dsb.)."""

import re

from sapa_api.text_utils import clamp_ocean, has_crisis_language, ocean_only, tokenize

# Frasa kritis — prioritas tertinggi
CRISIS_CRITICAL_PHRASES = (
    "bunuh diri",
    "membunuh diri",
    "ingin bunuh diri",
    "pengen bunuh diri",
    "mau bunuh diri",
    "akan bunuh diri",
    "ingin bunuh",
    "pengen bunuh",
    "mau bunuh",
    "ingin mati",
    "pengen mati",
    "mau mati",
    "tidak ingin hidup",
    "tidak mau hidup",
    "putus asa",
    "akhiri hidup",
    "mengakhiri hidup",
    "melukai diri",
    "menyakiti diri",
    "menyerah hidup",
)

CRISIS_HIGH_PHRASES = (
    "ingin mati",
    "pengen mati",
    "depresi berat",
    "tidak ada harapan",
    "lebih baik mati",
    "hidup tidak berarti",
    "menyerah",
)

CRISIS_HIGHLIGHT_TOKENS = frozenset({
    "bunuh", "diri", "mati", "pengen", "ingin", "mau", "akan",
    "putus", "asa", "menyerah", "melukai", "menyakiti", "akhiri",
})

_CRISIS_INTENT_BUNUH = re.compile(
    r"\b(pengen|ingin|mau|akan|pengin|pingin)\s+bunuh\b", re.I
)
_CRISIS_BUNUH_DIRI = re.compile(r"\bbunuh\s+diri\b", re.I)


def detect_crisis_level(text: str) -> str:
    """
    Returns: 'none' | 'high' | 'critical'
    """
    from sapa_api.fuzzy_match import fuzzy_phrase_match

    t = text.lower().strip()
    if not t:
        return "none"

    for phrase in CRISIS_CRITICAL_PHRASES:
        if phrase in t or fuzzy_phrase_match(t, phrase):
            return "critical"

    if _CRISIS_INTENT_BUNUH.search(t) or _CRISIS_BUNUH_DIRI.search(t):
        return "critical"

    tokens = set(tokenize(t))
    if "bunuh" in tokens and ("diri" in tokens or "sendiri" in tokens):
        return "critical"

    for phrase in CRISIS_HIGH_PHRASES:
        if phrase in t or fuzzy_phrase_match(t, phrase):
            return "high"

    if has_crisis_language(t):
        return "high"

    return "none"


def crisis_highlight_tokens(text: str) -> set[str]:
    """Token yang harus di-highlight pada teks krisis."""
    level = detect_crisis_level(text)
    if level == "none":
        return set()
    found = set()
    for tok in tokenize(text.lower()):
        if tok in CRISIS_HIGHLIGHT_TOKENS or len(tok) >= 5:
            if tok in ("rasanya", "seperti", "banget", "sangat"):
                continue
            if level == "critical" and tok in CRISIS_HIGHLIGHT_TOKENS:
                found.add(tok)
            elif level == "high" and tok in ("bunuh", "mati", "diri", "pengen", "ingin", "putus", "asa"):
                found.add(tok)
    if level == "critical":
        for tok in tokenize(text.lower()):
            if tok in CRISIS_HIGHLIGHT_TOKENS:
                found.add(tok)
    return found


def apply_crisis_adjustment(scores: dict, text: str) -> tuple[dict, str]:
    """Override skor untuk teks risiko — tidak dibatasi delta normal."""
    level = detect_crisis_level(text)
    if level == "none":
        return scores, level

    out = scores.copy()
    ocean = ocean_only(out)

    if level == "critical":
        out["EXTREME_ALERT"] = 5.0
        out["N"] = max(ocean.get("N", 3.0), 4.85)
        out["E"] = min(ocean.get("E", 3.0), 1.0)
        out["A"] = min(ocean.get("A", 3.0), 2.0)
        out["C"] = min(ocean.get("C", 3.0), 2.0)
        out["O"] = min(ocean.get("O", 3.0), 3.0)
    elif level == "high":
        out["EXTREME_ALERT"] = max(out.get("EXTREME_ALERT", 0), 4.0)
        out["N"] = max(ocean.get("N", 3.0), 4.5)
        out["E"] = min(ocean.get("E", 3.0), 1.8)
        out["A"] = min(ocean.get("A", 3.0), 2.5)

    out = clamp_ocean(out)
    out["EXTREME_ALERT"] = round(min(out.get("EXTREME_ALERT", 0), 5.0), 3)
    return out, level
