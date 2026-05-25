"""
Intensifier, negasi, dan kata sambung (connector) untuk modifikasi skor OCEAN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sapa_api.config import OCEAN_TRAITS
from sapa_api.text_utils import clamp_ocean, ocean_only, tokenize

# Kata penguat tinggi
INTENSIFIER_HIGH: dict[str, float] = {
    "sangat": 1.35,
    "sekali": 1.3,
    "banget": 1.3,
    "amat": 1.35,
    "super": 1.25,
    "terlalu": 1.4,
    "extremely": 1.35,
    "really": 1.25,
    "very": 1.25,
}

# Kata penguat rendah / moderat
INTENSIFIER_LOW: dict[str, float] = {
    "agak": 0.75,
    "cukup": 0.85,
    "lumayan": 0.85,
    "sedikit": 0.65,
    "relatif": 0.8,
}

INTENSIFIERS = {**INTENSIFIER_HIGH, **INTENSIFIER_LOW}

NEGATION = frozenset({
    "tidak", "bukan", "gak", "nggak", "ngga", "tak", "belum", "jangan",
    "tanpa", "no", "not", "never", "tak", "ndak", "enggak",
})

# Kata sambung / konteks hubungan antarklausa
CONNECTOR_CAUSE = frozenset({
    "karena", "sebab", "sehingga", "maka", "akibat", "lantaran", "dikarenakan",
})
CONNECTOR_CONTRAST = frozenset({
    "tapi", "tetapi", "namun", "meski", "meskipun", "walau", "walaupun",
    "padahal", "nyatanya", "justru",
})
CONNECTOR_ADDITIVE = frozenset({
    "dan", "serta", "juga", "lagipula", "bahkan", "plus", "sambil",
})
CONNECTOR_RESULT = frozenset({
    "jadi", "maka", "akhirnya", "akibatnya", "sehingga",
})

ALL_CONNECTORS = CONNECTOR_CAUSE | CONNECTOR_CONTRAST | CONNECTOR_ADDITIVE | CONNECTOR_RESULT

# Kata sentimen yang sering dipengaruhi intensifier/negasi
SENTIMENT_ANCHORS = frozenset({
    "cemas", "stres", "stress", "khawatir", "gelisah", "panik", "takut", "sedih",
    "marah", "bahagia", "senang", "tenang", "optimis", "depresi", "frustasi",
    "terganggu", "kepikiran", "overthinking", "empati", "kreatif", "disiplin",
    "rajin", "sosial", "ramah", "percaya", "kooperatif", "adaptif", "fleksibel",
    "menyesuaikan", "bergaul", "introvert", "ekstrovert",
})

# Negasi + kata → efek OCEAN (bukan sekadar invert generik)
NEGATED_POSITIVE_CALM = frozenset({"tenang", "santai", "rileks", "damai", "stabil"})
NEGATED_DISTRESS = frozenset({"cemas", "stres", "khawatir", "sedih", "marah", "panik"})


@dataclass
class ModifierHit:
    modifier: str
    modifier_type: str  # intensifier_high | intensifier_low | negation | connector
    target: str
    factor: float
    ocean_hint: str | None = None
    connector_role: str | None = None


@dataclass
class ModifierAnalysis:
    hits: list[ModifierHit] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _split_by_connectors(text_lower: str) -> list[str]:
    tokens = tokenize(text_lower)
    if not tokens:
        return [text_lower] if text_lower else []

    segments: list[str] = []
    buf: list[str] = []
    for t in tokens:
        if t in ALL_CONNECTORS and buf:
            segments.append(" ".join(buf))
            buf = []
        elif t not in ALL_CONNECTORS:
            buf.append(t)
    if buf:
        segments.append(" ".join(buf))
    return segments or [text_lower]


def analyze_modifiers(text: str) -> ModifierAnalysis:
    text_lower = text.lower()
    tokens = tokenize(text_lower)
    hits: list[ModifierHit] = []

    for i, tok in enumerate(tokens):
        if tok in ALL_CONNECTORS:
            role = "contrast"
            if tok in CONNECTOR_CAUSE:
                role = "cause"
            elif tok in CONNECTOR_ADDITIVE:
                role = "additive"
            elif tok in CONNECTOR_RESULT:
                role = "result"
            after = " ".join(tokens[i + 1 : i + 6])
            hits.append(
                ModifierHit(
                    modifier=tok,
                    modifier_type="connector",
                    target=after or "(lanjutan)",
                    factor=1.0,
                    connector_role=role,
                )
            )

        if tok in NEGATION and i + 1 < len(tokens):
            target = tokens[i + 1]
            if target in SENTIMENT_ANCHORS or target in NEGATED_DISTRESS | NEGATED_POSITIVE_CALM:
                ocean = None
                if target in NEGATED_DISTRESS:
                    ocean = "N"
                elif target in NEGATED_POSITIVE_CALM:
                    ocean = "N"
                hits.append(
                    ModifierHit(
                        modifier=tok,
                        modifier_type="negation",
                        target=target,
                        factor=-0.85,
                        ocean_hint=ocean,
                    )
                )

        if tok in INTENSIFIERS and i + 1 < len(tokens):
            target = tokens[i + 1]
            if target in SENTIMENT_ANCHORS:
                hits.append(
                    ModifierHit(
                        modifier=tok,
                        modifier_type=(
                            "intensifier_high" if tok in INTENSIFIER_HIGH else "intensifier_low"
                        ),
                        target=target,
                        factor=INTENSIFIERS[tok],
                    )
                )

    # Frasa ber-intensifier: "sangat cemas berat"
    for high_word, factor in INTENSIFIER_HIGH.items():
        for anchor in SENTIMENT_ANCHORS:
            pattern = f"{high_word} {anchor}"
            if pattern in text_lower:
                if not any(h.target == anchor and h.modifier == high_word for h in hits):
                    hits.append(
                        ModifierHit(
                            modifier=high_word,
                            modifier_type="intensifier_high",
                            target=anchor,
                            factor=factor,
                        )
                    )

    segments = _split_by_connectors(text_lower)
    summary = {
        "intensifier_count": sum(1 for h in hits if h.modifier_type.startswith("intensifier")),
        "negation_count": sum(1 for h in hits if h.modifier_type == "negation"),
        "connector_count": sum(1 for h in hits if h.modifier_type == "connector"),
        "segment_count": len(segments),
    }
    return ModifierAnalysis(hits=hits, segments=segments, summary=summary)


def phrase_modifier_scale(text_lower: str, phrase: str) -> tuple[float, list[ModifierHit]]:
    """Skala bobot untuk frasa berdasarkan intensifier/negasi di depannya."""
    idx = text_lower.find(phrase)
    if idx < 0:
        return 1.0, []

    before = tokenize(text_lower[:idx])
    related: list[ModifierHit] = []
    scale = 1.0

    if not before:
        return scale, related

    window = before[-3:]
    for i, tok in enumerate(window):
        if tok in INTENSIFIERS:
            scale *= INTENSIFIERS[tok]
            related.append(
                ModifierHit(
                    modifier=tok,
                    modifier_type=(
                        "intensifier_high" if tok in INTENSIFIER_HIGH else "intensifier_low"
                    ),
                    target=phrase.split()[0] if phrase else "",
                    factor=INTENSIFIERS[tok],
                )
            )
        if tok in NEGATION:
            scale *= -0.7
            related.append(
                ModifierHit(
                    modifier=tok,
                    modifier_type="negation",
                    target=phrase.split()[0] if phrase else "",
                    factor=-0.7,
                )
            )

    return max(0.25, min(1.6, scale)), related


def apply_modifier_ocean_adjustment(
    scores: dict,
    text: str,
    confidence_scale: float = 1.0,
) -> tuple[dict, ModifierAnalysis]:
    """Sesuaikan skor OCEAN dari intensifier, negasi, dan konteks connector."""
    analysis = analyze_modifiers(text)
    adjusted = scores.copy()
    text_lower = text.lower()

    for hit in analysis.hits:
        if hit.modifier_type == "negation" and hit.ocean_hint:
            if hit.target in NEGATED_DISTRESS:
                adjusted["N"] = adjusted.get("N", 3.0) - 0.35 * confidence_scale
            elif hit.target in NEGATED_POSITIVE_CALM:
                adjusted["N"] = adjusted.get("N", 3.0) + 0.3 * confidence_scale

        if hit.modifier_type.startswith("intensifier") and hit.target in NEGATED_DISTRESS:
            boost = (hit.factor - 1.0) * 0.4 * confidence_scale
            adjusted["N"] = adjusted.get("N", 3.0) + boost
            if hit.factor >= 1.3:
                adjusted["E"] = adjusted.get("E", 3.0) - boost * 0.3

    # Segmen setelah connector kontras → beri bobot lebih pada klausa kedua
    if len(analysis.segments) >= 2:
        second = analysis.segments[-1]
        if any(c in tokenize(text_lower) for c in CONNECTOR_CONTRAST):
            if any(w in second for w in ("cemas", "stres", "terganggu", "khawatir")):
                adjusted["N"] = adjusted.get("N", 3.0) + 0.2 * confidence_scale
            if any(w in second for w in ("senang", "bahagia", "optimis")):
                adjusted["N"] = adjusted.get("N", 3.0) - 0.15 * confidence_scale
                adjusted["A"] = adjusted.get("A", 3.0) + 0.1 * confidence_scale

    adjusted = clamp_ocean(adjusted)
    return adjusted, analysis


def modifiers_to_dict(analysis: ModifierAnalysis) -> dict:
    return {
        "summary": analysis.summary,
        "segments": analysis.segments,
        "modifiers": [
            {
                "modifier": h.modifier,
                "type": h.modifier_type,
                "target": h.target,
                "factor": round(h.factor, 3),
                "ocean_hint": h.ocean_hint,
                "connector_role": h.connector_role,
            }
            for h in analysis.hits
        ],
    }


def modifier_explanation_note(analysis: ModifierAnalysis) -> str:
    if not analysis.hits:
        return ""
    parts = []
    for h in analysis.hits[:3]:
        if h.modifier_type.startswith("intensifier"):
            parts.append(f"«{h.modifier} {h.target}» (penguat ×{h.factor:.2f})")
        elif h.modifier_type == "negation":
            parts.append(f"«{h.modifier} {h.target}» (negasi)")
        elif h.modifier_type == "connector":
            parts.append(f"«{h.modifier}» ({h.connector_role})")
    if not parts:
        return ""
    return " Modifikator linguistik: " + ", ".join(parts) + "."
