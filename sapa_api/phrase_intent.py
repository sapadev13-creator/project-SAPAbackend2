"""
Klasifikasi intent linguistik — menentukan dominan OCEAN & persona sebelum max(skor model).

Intent dipisah agar kombinasi kata tidak salah arah (mis. validasi empatik → A, bukan O).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sapa_api.config import OCEAN_TRAITS
from sapa_api.crisis import detect_crisis_level
from sapa_api.text_utils import (
    has_crisis_language,
    has_distress_language,
    has_empathy_validation_context,
    has_positive_context,
    has_relationship_affection_context,
    tokenize,
)

# --- Frasa per intent (prioritas tinggi = dicek lebih dulu) ---

CRISIS_PHRASES = (
    "bunuh diri", "ingin bunuh diri", "pengen bunuh diri", "mau bunuh diri",
    "ingin mati", "pengen mati", "tidak ingin hidup", "melukai diri",
    "menyakiti diri", "akhiri hidup",
)

ANXIETY_PHRASES = (
    "cemas menghadapi", "menghadapi hari esok", "hari esok",
    "mudah terganggu", "kepikiran", "overthinking", "stres berat",
    "sangat cemas", "cemas berat", "khawatir berlebihan",
    "merasa cemas", "merasa stres", "tidak tenang", "gelisah berat",
)

SAD_PHRASES = (
    "putus harapan", "merasa sedih", "merasa hampa", "merasa kosong",
    "terpuruk", "tidak bersemangat", "kehilangan motivasi",
)

ANGER_PHRASES = (
    "mudah marah", "naik pitam", "meledak emosi", "frustrasi berat",
    "kesal berat", "sulit mengendalikan emosi",
)

ADAPTIVE_PHRASES = (
    "mudah menyesuaikan diri", "menyesuaikan diri", "lingkungan baru",
    "beradaptasi", "adaptif", "fleksibel", "nyaman di keramaian",
)

CREATIVE_PHRASES = (
    "suka ide baru", "imajinatif", "kreatif", "inovatif", "ide baru",
    "suka bereksperimen", "open minded", "berpikir kreatif",
    "out of the box", "visioner",
)

DISCIPLINE_PHRASES = (
    "disiplin", "terorganisir", "tepat waktu", "bertanggung jawab",
    "rajin", "terencana", "sistematis", "fokus menyelesaikan",
    "manajemen waktu", "deadline", "produktif",
)

ACHIEVEMENT_PHRASES = (
    "ambisius", "berprestasi", "suka tantangan", "goal oriented",
    "prestasi", "ingin sukses", "berorientasi hasil",
)

SOCIAL_DEPENDENCY_PHRASES = (
    "butuh teman", "butuh teman untuk", "curhat setiap hari",
    "sangat butuh teman", "butuh dukungan sosial", "butuh orang lain",
    "butuh validasi", "sangat butuh curhat",
)

NEGATIVE_SOCIAL_PHRASES = (
    "sulit percaya orang", "sulit percaya", "menghindari keramaian",
    "tidak nyaman sosial", "menarik diri", "hindari keramaian",
    "isolasi sosial", "konflik interpersonal", "tidak suka keramaian",
)

ROMANTIC_AFFECTION_PHRASES = (
    "mencintai pasangan", "mencintai kekasih", "cinta mati-matian", "kasih sayang",
    "penuh kasih sayang", "merasa sayang", "sayang pada", "bersama kekasih",
    "bersama pacar", "dengan pasangan", "dekat dengannya", "hubungan dekat",
    "hubungan romantis", "quality time dengan pasangan", "orang tersayang",
    "romantis dan", "sangat romantis", "rindu kekasih", "rindu pacar",
)

SOCIAL_POSITIVE_PHRASES = (
    "suka bertemu orang", "aktif bersosialisasi", "komunikatif",
    "percaya diri sosial", "suka ngobrol", "networking",
    "suka bergaul", "bergaul dan aktif",
)

# Kategori Excel → dimensi OCEAN dominan yang diharapkan
TRAIT_CATEGORY_TO_OCEAN: dict[str, str] = {
    "EXTREME_NEGATIVE": "N",
    "ANXIETY_EMO": "N",
    "SAD_EMO": "N",
    "ANGER_EMO": "N",
    "MENTAL_UNSTABLE_N": "N",
    "EMO_NEGATIVE": "N",
    "NEGATIVE_SOCIAL": "N",
    "EMPATHY_HARMONY_A": "A",
    "RELATIONSHIP_AFFECTION": "A",
    "TRUST": "A",
    "COLLABORATION": "A",
    "EMO_POSITIVE": "A",
    "POSITIVE_SOCIAL": "E",
    "EXTRAVERSION_E": "E",
    "E_SOCIAL_DEPENDENCY": "E",
    "CREATIVE_DISCUSSION_A": "O",
    "INTROSPECTION": "O",
    "DISCIPLINE_C": "C",
    "ACHIEVEMENT": "C",
}

INTENT_TO_OCEAN: dict[str, str] = {
    "crisis": "N",
    "anxiety": "N",
    "sad": "N",
    "anger": "N",
    "empathy_validation": "A",
    "adaptive": "A",
    "social_positive": "E",
    "social_dependency": "E",
    "negative_social": "N",
    "creative": "O",
    "discipline": "C",
    "achievement": "C",
    "mixed_positive": "A",
    "relationship_affection": "A",
}


@dataclass
class TextIntent:
    primary: str
    secondary: str | None = None
    matched_phrases: list[tuple[str, str]] = field(default_factory=list)
    """(intent_key, matched_subphrase)"""
    category_hits: dict[str, int] = field(default_factory=dict)
    """Excel trait category -> hit count"""

    @property
    def expected_ocean(self) -> str:
        if self.primary in INTENT_TO_OCEAN:
            return INTENT_TO_OCEAN[self.primary]
        return "A"

    def ocean_boosts(self) -> dict[str, float]:
        """Delta tambahan per dimensi dari intent."""
        boosts = {k: 0.0 for k in OCEAN_TRAITS}
        p = self.primary
        if p == "crisis":
            boosts.update({"N": 0.5, "E": -0.2, "A": -0.15, "O": -0.1})
        elif p == "anxiety":
            boosts.update({"N": 0.35, "E": -0.12, "O": -0.08})
        elif p == "sad":
            boosts.update({"N": 0.3, "E": -0.1})
        elif p == "anger":
            boosts.update({"N": 0.3, "A": -0.1})
        elif p == "empathy_validation":
            boosts.update({"A": 0.4, "N": -0.2, "O": -0.22, "E": 0.05})
        elif p == "adaptive":
            boosts.update({"A": 0.25, "E": 0.2, "N": -0.12, "O": -0.05})
        elif p == "social_positive":
            boosts.update({"E": 0.45, "A": 0.12, "N": -0.1, "O": -0.05})
        elif p == "social_dependency":
            boosts.update({"E": 0.38, "A": 0.15, "N": 0.08, "O": -0.05})
        elif p == "negative_social":
            boosts.update({"N": 0.32, "E": -0.22, "A": -0.12, "O": -0.05})
        elif p == "creative":
            boosts.update({"O": 0.3, "E": 0.08})
        elif p == "discipline":
            boosts.update({"C": 0.42, "N": -0.1, "O": -0.08})
        elif p == "achievement":
            boosts.update({"C": 0.28, "E": 0.1, "O": 0.08})
        elif p == "mixed_positive":
            boosts.update({"A": 0.2, "E": 0.1, "N": -0.1})
        elif p == "relationship_affection":
            boosts.update({"A": 0.48, "E": 0.14, "N": -0.1, "O": -0.2})
        return boosts


def _first_match(text_lower: str, phrases: tuple[str, ...]) -> str | None:
    for p in phrases:
        if p in text_lower:
            return p
    return None


def _score_categories(category_hits: dict[str, int]) -> dict[str, float]:
    ocean_scores: dict[str, float] = {k: 0.0 for k in OCEAN_TRAITS}
    for cat, count in category_hits.items():
        ocean = TRAIT_CATEGORY_TO_OCEAN.get(cat)
        if ocean:
            ocean_scores[ocean] += count
    return ocean_scores


def classify_text_intent(
    text: str,
    matched_phrases: list[tuple[str, str]] | None = None,
    matched_excel: list[tuple[str, str]] | None = None,
) -> TextIntent:
    """
    Tentukan intent utama dari teks + match kategori Excel (trait, phrase).
    """
    t = text.lower().strip()
    matched: list[tuple[str, str]] = []
    cat_hits: dict[str, int] = {}

    excel_matches = matched_phrases or matched_excel
    if excel_matches:
        for cat, phrase in excel_matches:
            cat_hits[cat] = cat_hits.get(cat, 0) + 1

    crisis_level = detect_crisis_level(t)
    if crisis_level == "critical" or _first_match(t, CRISIS_PHRASES):
        p = _first_match(t, CRISIS_PHRASES) or "krisis"
        matched.append(("crisis", p))
        return TextIntent("crisis", matched_phrases=matched, category_hits=cat_hits)

    if has_empathy_validation_context(t):
        p = _first_match(t, (
            "perasaan yang didengarkan", "terasa lebih ringan",
            "merasa didengarkan", "ringan hati",
        )) or "validasi"
        matched.append(("empathy_validation", p))
        return TextIntent("empathy_validation", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, ANXIETY_PHRASES):
        matched.append(("anxiety", p))
        if not has_positive_context(t) or "cemas" in t or "stres" in t:
            return TextIntent("anxiety", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, SAD_PHRASES):
        matched.append(("sad", p))
        if not has_positive_context(t):
            return TextIntent("sad", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, ANGER_PHRASES):
        matched.append(("anger", p))
        return TextIntent("anger", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, ADAPTIVE_PHRASES):
        matched.append(("adaptive", p))
        return TextIntent("adaptive", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, CREATIVE_PHRASES):
        matched.append(("creative", p))
        return TextIntent("creative", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, DISCIPLINE_PHRASES):
        matched.append(("discipline", p))
        return TextIntent("discipline", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, ACHIEVEMENT_PHRASES):
        matched.append(("achievement", p))
        return TextIntent("achievement", matched_phrases=matched, category_hits=cat_hits)

    if has_relationship_affection_context(t) or (
        cat_hits.get("RELATIONSHIP_AFFECTION", 0) >= 1
        and _first_match(t, ROMANTIC_AFFECTION_PHRASES)
    ):
        p = _first_match(t, ROMANTIC_AFFECTION_PHRASES) or "afeksi hubungan"
        matched.append(("relationship_affection", p))
        return TextIntent("relationship_affection", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, SOCIAL_DEPENDENCY_PHRASES):
        matched.append(("social_dependency", p))
        return TextIntent("social_dependency", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, NEGATIVE_SOCIAL_PHRASES):
        matched.append(("negative_social", p))
        return TextIntent("negative_social", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, SOCIAL_POSITIVE_PHRASES):
        matched.append(("social_positive", p))
        return TextIntent("social_positive", matched_phrases=matched, category_hits=cat_hits)

    # Fallback: kategori Excel dominan (jika ada match frasa)
    if cat_hits:
        ocean_from_cat = _score_categories(cat_hits)
        best_ocean = max(ocean_from_cat, key=ocean_from_cat.get)
        if ocean_from_cat[best_ocean] >= 1:
            intent_map = {
                "N": (
                    "negative_social"
                    if cat_hits.get("NEGATIVE_SOCIAL")
                    else ("anxiety" if has_distress_language(t) else "sad")
                ),
                "A": (
                    "empathy_validation"
                    if cat_hits.get("EMPATHY_HARMONY_A")
                    else (
                        "relationship_affection"
                        if cat_hits.get("RELATIONSHIP_AFFECTION")
                        or has_relationship_affection_context(t)
                        else "mixed_positive"
                    )
                ),
                "E": (
                    "social_dependency"
                    if cat_hits.get("E_SOCIAL_DEPENDENCY")
                    else "social_positive"
                ),
                "O": "creative",
                "C": "discipline" if cat_hits.get("DISCIPLINE_C") else "achievement",
            }
            primary = intent_map.get(best_ocean, "mixed_positive")
            if primary == "anxiety" and has_positive_context(t) and not has_distress_language(t):
                primary = "mixed_positive"
            return TextIntent(primary, matched_phrases=matched, category_hits=cat_hits)

    if has_relationship_affection_context(t):
        matched.append(("relationship_affection", "afeksi hubungan"))
        return TextIntent("relationship_affection", matched_phrases=matched, category_hits=cat_hits)

    if has_positive_context(t):
        return TextIntent("mixed_positive", matched_phrases=matched, category_hits=cat_hits)

    if has_distress_language(t) and not has_crisis_language(t):
        return TextIntent("anxiety", matched_phrases=matched, category_hits=cat_hits)

    return TextIntent("neutral", matched_phrases=matched, category_hits=cat_hits)


def apply_intent_ocean_adjustment(
    scores: dict,
    intent: TextIntent,
    confidence_scale: float = 1.0,
) -> dict:
    from sapa_api.text_utils import clamp_ocean, ocean_only

    out = scores.copy()
    if intent.primary == "neutral":
        return out
    for dim, delta in intent.ocean_boosts().items():
        out[dim] = out.get(dim, 3.0) + delta * confidence_scale
    return clamp_ocean(out)


def dominant_from_intent(
    scores: dict,
    intent: TextIntent,
    construct_ocean: str | None = None,
) -> str | None:
    """Dominan OCEAN dari intent + konstruk (intent menang jika kuat)."""
    from sapa_api.text_utils import ocean_only

    if intent.primary == "crisis":
        return "N"
    if intent.primary == "relationship_affection":
        return "A"
    if intent.primary in INTENT_TO_OCEAN:
        expected = INTENT_TO_OCEAN[intent.primary]
        ocean = ocean_only(scores)
        bias = 0.14 if expected == "A" else 0.08
        boosted = ocean.get(expected, 0) + bias
        others = max(ocean.get(k, 0) for k in OCEAN_TRAITS if k != expected)
        if boosted >= others - 0.03:
            return expected
    if construct_ocean and intent.primary in ("neutral", "mixed_positive"):
        return construct_ocean
    if intent.category_hits:
        ocean_from_cat = _score_categories(intent.category_hits)
        if max(ocean_from_cat.values()) >= 1:
            return max(ocean_from_cat, key=ocean_from_cat.get)
    return None
