"""
Pemetaan kategori keywords_traits.xlsx → persona (hasil akurat per kombinasi kata).
"""

from __future__ import annotations

from sapa_api.config import TRAIT_LIST_NAMES
from sapa_api.crisis import detect_crisis_level
from sapa_api.phrase_intent import TextIntent, classify_text_intent
from sapa_api.text_utils import (
    has_empathy_validation_context,
    has_relationship_affection_context,
    ocean_only,
)

# (label persona, deskripsi, prioritas dasar)
CATEGORY_PERSONA: dict[str, tuple[str, str, int]] = {
    "EXTREME_NEGATIVE": (
        "Krisis Emosional Tinggi",
        "menunjukkan tanda tekanan psikologis berat, putus asa, atau risiko menyakiti diri",
        200,
    ),
    "ANXIETY_EMO": (
        "Cemas & Overthinking",
        "mudah khawatir, banyak berpikir, reflektif terhadap masalah",
        125,
    ),
    "SAD_EMO": (
        "Melankolis & Berduka",
        "cenderung sedih, kehilangan semangat, dan merasa hampa",
        120,
    ),
    "ANGER_EMO": (
        "Tempramental",
        "emosional, mudah tersulut, impulsif saat tertekan",
        120,
    ),
    "MENTAL_UNSTABLE_N": (
        "Burnout & Labil Emosional",
        "kelelahan mental, mood tidak stabil, dan mudah drop",
        118,
    ),
    "EMO_NEGATIVE": (
        "Luka Emosional Mendalam",
        "menyimpan beban batin, perasaan berat, dan sulit dilepaskan",
        115,
    ),
    "NEGATIVE_SOCIAL": (
        "Menghindar & Isolasi Sosial",
        "cenderung menarik diri, tidak nyaman sosial, atau konflik interpersonal",
        115,
    ),
    "EMPATHY_HARMONY_A": (
        "Empatik & Terdengar",
        "merasa didengar, divalidasi, dan lebih ringan — berorientasi pada keharmonisan",
        140,
    ),
    "RELATIONSHIP_AFFECTION": (
        "Romantis Afektif",
        "hangat, penuh perhatian, berorientasi pada hubungan dekat",
        130,
    ),
    "TRUST": (
        "Terbuka & Saling Percaya",
        "jujur, transparan, dan membangun rasa aman dalam relasi",
        125,
    ),
    "COLLABORATION": (
        "Kolaboratif & Kooperatif",
        "nyaman kerja tim, gotong royong, dan sinergi dengan orang lain",
        120,
    ),
    "EMO_POSITIVE": (
        "Ceria & Optimis",
        "cenderung bahagia, bersemangat, dan melihat sisi positif kehidupan",
        125,
    ),
    "POSITIVE_SOCIAL": (
        "Ekstrovert Sosial",
        "percaya diri, aktif berinteraksi, mudah bergaul",
        128,
    ),
    "EXTRAVERSION_E": (
        "Ekstrovert Energik",
        "energik, komunikatif, dan nyaman menjadi pusat perhatian",
        128,
    ),
    "E_SOCIAL_DEPENDENCY": (
        "Afiliasi & Butuh Dukungan Sosial",
        "butuh kehadiran orang lain, interaksi, dan validasi dari lingkungan sosial",
        122,
    ),
    "CREATIVE_DISCUSSION_A": (
        "Visioner Kreatif",
        "imajinatif, visioner, suka ide baru dan diskusi kreatif",
        120,
    ),
    "INTROSPECTION": (
        "Reflektif & Introspektif",
        "sering merenung, mencari makna, dan menyelami perasaan diri",
        118,
    ),
    "DISCIPLINE_C": (
        "Perfeksionis Produktif",
        "teliti, terstruktur, disiplin, dan menuntut standar tinggi",
        120,
    ),
    "ACHIEVEMENT": (
        "Ambisius & Berorientasi Prestasi",
        "fokus pada target, prestasi, dan pencapaian jangka panjang",
        120,
    ),
    "ADAPTIVE_FLEX": (
        "Adaptif & Fleksibel",
        "mudah beradaptasi, terbuka pada lingkungan baru, dan kooperatif",
        135,
    ),
}

# Intent terdeteksi → kategori dataset (prioritas di atas hit skor Excel)
INTENT_PRIMARY_TO_CATEGORY: dict[str, str | None] = {
    "crisis": "EXTREME_NEGATIVE",
    "empathy_validation": "EMPATHY_HARMONY_A",
    "adaptive": "ADAPTIVE_FLEX",
    "anxiety": "ANXIETY_EMO",
    "sad": "SAD_EMO",
    "anger": "ANGER_EMO",
    "creative": "CREATIVE_DISCUSSION_A",
    "discipline": "DISCIPLINE_C",
    "achievement": "ACHIEVEMENT",
    "social_positive": "POSITIVE_SOCIAL",
    "social_dependency": "E_SOCIAL_DEPENDENCY",
    "negative_social": "NEGATIVE_SOCIAL",
    "mixed_positive": "EMO_POSITIVE",
    "relationship_affection": "RELATIONSHIP_AFFECTION",
    "neutral": None,
}


def _category_hit_scores(
    intent: TextIntent,
    evidence: dict | None,
) -> dict[str, float]:
    scores: dict[str, float] = {c: float(v) for c, v in intent.category_hits.items()}
    if evidence:
        for cat in TRAIT_LIST_NAMES:
            hits = evidence.get(cat) or []
            if hits:
                scores[cat] = scores.get(cat, 0) + len(hits) * 1.5
    return scores


def _dominant_category(scores: dict[str, float], intent: TextIntent) -> str | None:
    if not scores:
        mapped = INTENT_PRIMARY_TO_CATEGORY.get(intent.primary)
        return mapped
    top = max(scores.items(), key=lambda x: x[1])
    if top[1] < 1.0:
        return INTENT_PRIMARY_TO_CATEGORY.get(intent.primary)
    # Jika intent primary punya kategori, beri bobot tambahan
    mapped = INTENT_PRIMARY_TO_CATEGORY.get(intent.primary)
    if mapped and scores.get(mapped, 0) >= top[1] - 0.5:
        return mapped
    return top[0]


def _compatible(
    ps: dict,
    category: str,
    crisis: bool,
    validation: bool,
    romantic: bool = False,
) -> bool:
    if category == "EXTREME_NEGATIVE":
        return crisis or ps.get("EXTREME_ALERT", 0) >= 2.5
    if category == "ADAPTIVE_FLEX":
        return (
            ps.get("A", 0) >= 2.7
            and ps.get("N", 0) <= 3.9
            and ps.get("EXTREME_ALERT", 0) < 2.5
        )
    if category == "EMPATHY_HARMONY_A":
        return validation or ps.get("A", 0) >= 2.7
    if category in ("ANXIETY_EMO", "SAD_EMO", "ANGER_EMO", "MENTAL_UNSTABLE_N", "EMO_NEGATIVE"):
        return ps.get("N", 0) >= 2.8
    if category in ("POSITIVE_SOCIAL", "EXTRAVERSION_E", "E_SOCIAL_DEPENDENCY"):
        return ps.get("E", 0) >= 2.3 or ps.get("A", 0) >= 2.8
    if category in ("CREATIVE_DISCUSSION_A", "INTROSPECTION"):
        return ps.get("O", 0) >= 2.8
    if category in ("DISCIPLINE_C", "ACHIEVEMENT"):
        return ps.get("C", 0) >= 2.7
    if category == "EMO_POSITIVE":
        return ps.get("N", 0) <= 3.5 and (ps.get("A", 0) >= 2.7 or ps.get("E", 0) >= 2.5)
    if category == "RELATIONSHIP_AFFECTION":
        return romantic and (
            ps.get("A", 0) >= 2.8
            and ps.get("A", 0) >= ps.get("O", 0) + 0.12
            and ps.get("A", 0) >= ps.get("E", 0) - 0.1
            and ps.get("N", 0) <= 3.8
            and ps.get("EXTREME_ALERT", 0) < 2.5
        )
    if category == "TRUST":
        return ps.get("A", 0) >= 2.75 and not romantic
    return True


def resolve_persona_from_dataset(
    scores: dict,
    text: str,
    intent: TextIntent | None = None,
    evidence: dict | None = None,
) -> tuple[str, str, str] | None:
    """
    Returns (label, description, dominant_category) atau None jika fallback ke rules lama.
    """
    text_lower = text.lower()
    if intent is None:
        intent = classify_text_intent(text)

    ps = ocean_only(scores)
    ps["EXTREME_ALERT"] = scores.get("EXTREME_ALERT", 0)
    crisis_tier = detect_crisis_level(text_lower)
    crisis = crisis_tier in ("critical", "high") or ps.get("EXTREME_ALERT", 0) >= 3.0
    validation = has_empathy_validation_context(text_lower) or intent.primary == "empathy_validation"
    romantic = (
        has_relationship_affection_context(text_lower)
        or intent.primary == "relationship_affection"
    )

    if crisis or intent.primary == "crisis":
        cat = "EXTREME_NEGATIVE"
    elif intent.primary in INTENT_PRIMARY_TO_CATEGORY:
        cat = INTENT_PRIMARY_TO_CATEGORY[intent.primary]
    else:
        cat_scores = _category_hit_scores(intent, evidence)
        cat = _dominant_category(cat_scores, intent)

    if (
        romantic
        and cat not in ("RELATIONSHIP_AFFECTION", "EXTREME_NEGATIVE")
        and intent.category_hits.get("RELATIONSHIP_AFFECTION", 0) >= 1
    ):
        cat = "RELATIONSHIP_AFFECTION"

    if not cat or cat not in CATEGORY_PERSONA:
        return None
    if not _compatible(ps, cat, crisis, validation, romantic=romantic):
        return None

    label, desc, base_prio = CATEGORY_PERSONA[cat]
    ocean = ocean_only(ps)
    dom_ocean = max(ocean, key=ocean.get)
    dominance = ocean[dom_ocean] - sum(ocean.values()) / 5
    final_prio = base_prio + dominance * 0.1

    return label, desc, cat


def dominant_keyword_category(
    intent: TextIntent | None,
    evidence: dict | None,
) -> str | None:
    if intent is None:
        return None
    forced = INTENT_PRIMARY_TO_CATEGORY.get(intent.primary)
    if forced:
        return forced
    scores = _category_hit_scores(intent, evidence)
    return _dominant_category(scores, intent)
