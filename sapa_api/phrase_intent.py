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
    has_social_enjoyment_context,
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

EMO_BURDEN_PHRASES = (
    "hidup suram", "suram banget", "bobrok banget", "hidup bobrok",
    "mental hancur", "hancur banget", "ancur banget", "down bad",
    "mental bobrok", "perasaan suram", "lelah hidup", "ancur total",
    "hidup hancur", "mood suram", "emosi suram", "perasaan bobrok",
)

ANGER_PHRASES = (
    "mudah marah", "naik pitam", "meledak emosi", "frustrasi berat",
    "kesal berat", "sulit mengendalikan emosi",
)

RAGE_OVERFLOW_PHRASES = (
    "luapan kemarahan", "marah meledak", "ledakan kemarahan", "kemarahan meledak",
    "marah tidak terkontrol", "marah berlebihan", "ledakan emosi marah",
    "sulit menahan marah", "marah tak terkendali", "emosi marah meledak",
    "kemarahan yang meluap", "marah sampai meledak",
    "mengacau banget", "ngacau banget", "ngamuk banget", "ngegas banget",
    "emosi meledak", "marah banget", "kesel parah",
)

EMO_SURGE_PHRASES = (
    "luapan emosi", "emosi meluap", "emosi tak terkendali", "emosi berlebihan",
    "gelombang emosi", "emosi yang meluap", "emosi tidak terkendali",
    "emosi yang berlebihan", "emosi naik tajam", "emosi meledak keluar",
)

CRITICAL_HOSTILE_PHRASES = (
    "sangat kritis", "kritis dan tajam", "suka mencaci", "suka menyerang",
    "nada sinis", "kritik tajam", "suka mempermalukan", "kritis terhadap orang",
    "menyerang orang", "suka mengejek", "komentar sinis", "kritik menusuk",
    "bacot mulu", "bacot terus", "tolol banget", "bullshit banget", "bulshit banget",
    "ngaco banget", "nyindir terus", "sok tau", "toxic banget", "kaga pernah inget dosa",
    "gak pernah inget dosa", "munafik banget", "sok suci padahal",
)

HATRED_PHRASES = (
    "kebencian", "penuh kebencian", "membenci", "benci berat", "rasa benci",
    "menyimpan kebencian", "kebencian mendalam", "benci sekali", "membenci orang",
    "rasa kebencian", "benci dan dendam", "kebencian yang dalam",
    "bangsat lu", "bangsat banget", "keparat lu", "keparat emang", "brengsek banget",
    "benci banget", "benci lu", "muak banget", "muak sama", "jijik banget",
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
    "menikmati acara sosial", "acara sosial", "seminar", "komunitas",
    "pertemuan besar", "event sosial", "suka bertemu orang",
    "aktif bersosialisasi", "komunikatif", "percaya diri sosial",
    "suka ngobrol", "networking", "suka bergaul", "bergaul dan aktif",
)

# Kategori Excel → dimensi OCEAN dominan yang diharapkan
TRAIT_CATEGORY_TO_OCEAN: dict[str, str] = {
    "EXTREME_NEGATIVE": "N",
    "ANXIETY_EMO": "N",
    "SAD_EMO": "N",
    "ANGER_EMO": "N",
    "RAGE_OVERFLOW": "N",
    "EMO_SURGE": "N",
    "CRITICAL_HOSTILE": "N",
    "HATRED_EMO": "N",
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
    "emotional_burden": "N",
    "anger": "N",
    "rage": "N",
    "emotion_surge": "N",
    "critical_hostile": "N",
    "hatred": "N",
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
        elif p == "emotional_burden":
            boosts.update({"N": 0.32, "E": -0.12, "A": -0.1, "O": 0.04})
        elif p == "anger":
            boosts.update({"N": 0.3, "A": -0.1})
        elif p == "rage":
            boosts.update({"N": 0.42, "A": -0.22, "C": -0.15})
        elif p == "emotion_surge":
            boosts.update({"N": 0.38, "A": -0.15, "C": -0.12})
        elif p == "critical_hostile":
            boosts.update({"N": 0.35, "A": -0.28, "E": -0.1})
        elif p == "hatred":
            boosts.update({"N": 0.45, "A": -0.35, "E": -0.15, "C": -0.1})
        elif p == "empathy_validation":
            boosts.update({"A": 0.4, "N": -0.2, "O": -0.22, "E": 0.05})
        elif p == "adaptive":
            boosts.update({"A": 0.25, "E": 0.2, "N": -0.12, "O": -0.05})
        elif p == "social_positive":
            boosts.update({"E": 0.58, "A": 0.1, "N": -0.1, "O": -0.08})
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
            boosts.update({"A": 0.32, "E": 0.08, "N": -0.12, "O": -0.1})
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

    if p := _first_match(t, EMO_BURDEN_PHRASES):
        matched.append(("emotional_burden", p))
        return TextIntent("emotional_burden", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, HATRED_PHRASES):
        matched.append(("hatred", p))
        return TextIntent("hatred", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, RAGE_OVERFLOW_PHRASES):
        matched.append(("rage", p))
        return TextIntent("rage", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, CRITICAL_HOSTILE_PHRASES):
        matched.append(("critical_hostile", p))
        return TextIntent("critical_hostile", matched_phrases=matched, category_hits=cat_hits)

    if p := _first_match(t, EMO_SURGE_PHRASES):
        matched.append(("emotion_surge", p))
        return TextIntent("emotion_surge", matched_phrases=matched, category_hits=cat_hits)

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

    if has_social_enjoyment_context(t) or (
        cat_hits.get("POSITIVE_SOCIAL", 0) >= 1
        and _first_match(t, SOCIAL_POSITIVE_PHRASES)
    ):
        p = _first_match(t, SOCIAL_POSITIVE_PHRASES) or "acara sosial"
        matched.append(("social_positive", p))
        return TextIntent("social_positive", matched_phrases=matched, category_hits=cat_hits)

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
                    "hatred"
                    if cat_hits.get("HATRED_EMO")
                    else (
                        "rage"
                        if cat_hits.get("RAGE_OVERFLOW")
                        else (
                            "critical_hostile"
                            if cat_hits.get("CRITICAL_HOSTILE")
                            else (
                                "emotion_surge"
                                if cat_hits.get("EMO_SURGE")
                                else (
                                    "anger"
                                    if cat_hits.get("ANGER_EMO")
                                    else (
                                        "emotional_burden"
                                        if cat_hits.get("EMO_NEGATIVE")
                                        else (
                                            "negative_social"
                                            if cat_hits.get("NEGATIVE_SOCIAL")
                                            else ("anxiety" if has_distress_language(t) else "sad")
                                        )
                                    )
                                )
                            )
                        )
                    )
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

    from sapa_api.text_utils import dominant_from_adjusted_scores

    if intent.primary == "neutral":
        return None
    ocean = ocean_only(scores)
    expected = INTENT_TO_OCEAN.get(intent.primary)
    if not expected:
        return None
    if ocean.get(expected, 0) >= max(ocean.values()) - 0.08:
        return expected
    return dominant_from_adjusted_scores(scores, intent.primary)
