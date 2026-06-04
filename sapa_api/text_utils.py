"""Utilitas tokenisasi, konteks teks, dan kecukupan sampel."""

import re
from dataclasses import dataclass

from sapa_api.config import OCEAN_TRAITS

# Kata fungsi / generik — tidak dipakai untuk match keyword tunggal
GENERIC_STOPWORDS = frozenset({
    "di", "ke", "dari", "yang", "dan", "atau", "pada", "untuk", "dengan",
    "ini", "itu", "nya", "adalah", "akan", "sudah", "belum", "juga", "saja",
    "the", "a", "an", "is", "are", "was", "were",
    "saya", "aku", "gue", "kamu", "dia", "mereka", "kita", "kami", "kalian",
    "mereka", "lo", "lu", "gw", "gue", "ane", "ente",
    "mudah", "merasa", "rasanya", "rasa", "seperti",
    "memang", "kalau", "kalo",
    "bila", "apabila", "supaya", "agar", "biar", "hal", "halnya", "sesuatu",
    "suatu", "beberapa", "banyak", "semua", "setiap", "hanya", "cuma",
    "masih", "lagi", "sudah", "pernah", "selalu", "kadang", "kadang-kadang",
    "hari", "besok", "kemarin", "nanti", "sekarang", "dalam", "atas", "bawah",
    "the", "and", "or", "but", "very", "really",
})

STOPWORDS_ID = GENERIC_STOPWORDS

# Frasa positif adaptif — harus spesifik (bukan "mudah" saja)
POSITIVE_CONTEXT_PHRASES = (
    "mudah menyesuaikan diri",
    "menyesuaikan diri",
    "lingkungan baru",
    "beradaptasi",
    "adaptif",
    "fleksibel",
    "suka belajar",
    "senang belajar",
    "bahagia",
    "berkolaborasi",
    "kerja sama",
    "optimis",
    "semangat",
    "bangga",
)

# Validasi empatik / kelegaan — A & EMO_POSITIVE, bukan O kreatif
# Romantis / afeksi hubungan dekat — A (bukan validasi empatik atau sekadar ceria)
RELATIONSHIP_AFFECTION_PHRASES = (
    "mencintai pasangan",
    "mencintai kekasih",
    "cinta mati-matian",
    "cinta mati matian",
    "kasih sayang",
    "penuh kasih sayang",
    "merasa sayang",
    "sayang pada",
    "sayang padamu",
    "sayang kepadanya",
    "bersama kekasih",
    "bersama pacar",
    "dengan pasangan",
    "dekat dengannya",
    "dekat dengan pasangan",
    "selalu dekat",
    "hubungan dekat",
    "hubungan romantis",
    "quality time dengan pasangan",
    "orang tersayang",
    "pada orang tersayang",
    "romantis dan",
    "sangat romantis",
    "afeksi hubungan",
    "hubungan hangat dan mesra",
    "mesra dengan pasangan",
    "rindu kekasih",
    "rindu pacar",
)

# Menikmati interaksi / acara sosial — E (Extraversion), bukan sekadar O atau A
SOCIAL_ENJOYMENT_PHRASES = (
    "menikmati acara sosial",
    "acara sosial",
    "suka acara sosial",
    "senang acara sosial",
    "seminar",
    "komunitas",
    "pertemuan besar",
    "event sosial",
    "kumpul komunitas",
    "aktif di komunitas",
    "suka seminar",
    "suka networking",
    "suka bertemu orang",
    "aktif bersosialisasi",
    "suka bergaul",
    "bergaul dan aktif",
)

EMPATHY_VALIDATION_PHRASES = (
    "perasaan yang didengarkan",
    "perasaan yang divalidasi",
    "perasaan yang diterima",
    "perasaan yang dipahami",
    "perasaan yang dihargai",
    "didengarkan dengan penuh perhatian",
    "terasa lebih ringan",
    "terasa ringan",
    "merasa lebih ringan",
    "merasa didengarkan",
    "merasa divalidasi",
    "merasa dipahami",
    "merasa dihargai",
    "merasa diterima",
    "beban terasa ringan",
    "hati terasa ringan",
    "ringan hati",
    "lega",
    "tenang hati",
    "ada yang mendengarkan",
    "didengarkan tanpa dihakimi",
    "perasaan diterima",
    "perasaan dipahami",
    "merasa lega",
    "beban jadi ringan",
    "hati jadi ringan",
)

# Prestasi / ketelitian — C, bukan O
DISCIPLINE_CONTEXT_PHRASES = (
    "tepat waktu", "terorganisir", "manajemen waktu", "to do list",
    "deadline", "rajin", "disiplin", "sistematis", "terencana",
)

# Kreativitas eksplisit — O
CREATIVE_CONTEXT_PHRASES = (
    "ide baru", "suka ide", "kreatif", "imajinatif", "inovatif",
    "visioner", "out of the box", "bereksperimen",
)

# Bahasa krisis — bunuh diri / menyakiti diri (BUKAN cemas biasa)
CRISIS_LANGUAGE_HINTS = (
    "bunuh diri", "membunuh diri", "pengen bunuh", "ingin bunuh",
    "mau bunuh", "ingin mati", "pengen mati", "tidak ingin hidup",
    "melukai diri", "menyakiti diri", "akhiri hidup", "mengakhiri hidup",
)

# Distres emosional — menaikkan N, bukan krisis
DISTRESS_LANGUAGE_HINTS = (
    "cemas", "stres", "stress", "gelisah", "khawatir", "panik", "takut",
    "sedih", "depresi", "marah", "kemarahan", "benci", "kebencian", "kecewa",
    "frustasi", "kritis", "kepikiran", "overthinking", "terganggu", "putus asa",
    "emosi", "dendam", "murka", "geram", "bacot", "tolol", "bangsat", "keparat",
    "brengsek", "mengacau", "ngaco", "ngamuk", "suram", "bobrok", "bullshit",
    "bulshit", "muak", "jijik", "toxic",
)

MIN_SINGLE_KEYWORD_LEN = 4
MIN_MEANINGFUL_TOKENS_TRAIT = 6
MIN_MEANINGFUL_TOKENS_STATE = 3


@dataclass
class TextSufficiency:
    word_count: int
    meaningful_count: int
    analysis_mode: str  # trait | mixed | state
    confidence_scale: float
    disclaimer: str | None


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def is_generic_token(token: str) -> bool:
    return token in GENERIC_STOPWORDS or len(token) < MIN_SINGLE_KEYWORD_LEN


def is_meaningful_token(token: str) -> bool:
    return not is_generic_token(token)


def meaningful_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if is_meaningful_token(t)]


def has_empathy_validation_context(text_lower: str) -> bool:
    return any(p in text_lower for p in EMPATHY_VALIDATION_PHRASES)


def has_social_enjoyment_context(text_lower: str) -> bool:
    if has_distress_language(text_lower) or has_crisis_language(text_lower):
        return False
    if any(p in text_lower for p in SOCIAL_ENJOYMENT_PHRASES):
        return True
    social_hits = sum(
        1
        for w in ("sosial", "seminar", "komunitas", "networking", "bergaul")
        if w in text_lower
    )
    return social_hits >= 2 and (
        "menikmati" in text_lower
        or "senang" in text_lower
        or "suka" in text_lower
        or "aktif" in text_lower
    )


def has_relationship_affection_context(text_lower: str) -> bool:
    if has_empathy_validation_context(text_lower):
        return False
    if has_distress_language(text_lower) or has_crisis_language(text_lower):
        return False
    if any(p in text_lower for p in RELATIONSHIP_AFFECTION_PHRASES):
        return True
    romantic_tokens = (
        "kekasih", "pacar", "pasangan", "romantis", "mencintai",
    )
    tokens = set(tokenize(text_lower))
    if tokens & set(romantic_tokens):
        if "cinta" in text_lower or "sayang" in text_lower or "kasih" in text_lower:
            return True
        if "kekasih" in tokens or "pacar" in tokens:
            return True
    return False


def has_positive_context(text_lower: str) -> bool:
    return (
        has_empathy_validation_context(text_lower)
        or any(p in text_lower for p in POSITIVE_CONTEXT_PHRASES)
    )


def has_crisis_language(text_lower: str) -> bool:
    return any(p in text_lower for p in CRISIS_LANGUAGE_HINTS)


def has_distress_language(text_lower: str) -> bool:
    return any(p in text_lower for p in DISTRESS_LANGUAGE_HINTS)


def has_negative_context(text_lower: str) -> bool:
    """Kompatibilitas: krisis ATAU distres."""
    return has_crisis_language(text_lower) or has_distress_language(text_lower)


def assess_text_sufficiency(text: str) -> TextSufficiency:
    tokens = tokenize(text)
    meaningful = meaningful_tokens(text)
    n = len(meaningful)
    wc = len(tokens)

    if n < MIN_MEANINGFUL_TOKENS_STATE:
        return TextSufficiency(
            word_count=wc,
            meaningful_count=n,
            analysis_mode="state",
            confidence_scale=0.35,
            disclaimer=(
                "Teks sangat singkat: hasil lebih mencerminkan keadaan emosi sementara (state) "
                "bukan profil kepribadian (trait) jangka panjang. Disarankan minimal 2–3 kalimat "
                "atau ~30 kata bermakna."
            ),
        )
    if n < MIN_MEANINGFUL_TOKENS_TRAIT:
        return TextSufficiency(
            word_count=wc,
            meaningful_count=n,
            analysis_mode="mixed",
            confidence_scale=0.65,
            disclaimer=(
                "Teks masih relatif pendek: interpretasi trait bersifat indikatif. "
                "Untuk profil lebih stabil, gunakan teks lebih panjang dan kontekstual."
            ),
        )
    return TextSufficiency(
        word_count=wc,
        meaningful_count=n,
        analysis_mode="trait",
        confidence_scale=1.0,
        disclaimer=None,
    )


def ocean_only(scores: dict) -> dict:
    return {k: float(scores[k]) for k in OCEAN_TRAITS if k in scores}


def align_scores_to_intent_dominance(
    scores: dict,
    intent_primary: str | None,
    margin: float = 0.1,
) -> dict:
    """
    Pastikan dimensi intent memiliki skor tertinggi (dominant_trait = argmax OCEAN).
    Dipanggil setelah limit_ocean_delta agar penyesuaian tidak terpotong.
    """
    if not intent_primary or intent_primary == "neutral":
        return scores

    intent_top = {
        "crisis": "N",
        "anxiety": "N",
        "sad": "N",
        "anger": "N",
        "negative_social": "N",
        "empathy_validation": "A",
        "relationship_affection": "A",
        "adaptive": "A",
        "mixed_positive": "A",
        "social_positive": "E",
        "social_dependency": "E",
        "creative": "O",
        "discipline": "C",
        "achievement": "C",
    }
    target = intent_top.get(intent_primary)
    if not target:
        return scores

    out = scores.copy()
    ocean = ocean_only(out)
    top_val = max(ocean.values())
    if ocean.get(target, 0) >= top_val - 0.01:
        return out

    out[target] = round(min(5.0, top_val + margin), 3)
    return out


def dominant_from_adjusted_scores(
    scores: dict,
    intent_primary: str | None = None,
    tie_margin: float = 0.03,
) -> str:
    """
    Trait dominan = dimensi OCEAN dengan skor tertinggi pada prediction_adjusted.
    Intent hanya dipakai sebagai tie-break jika selisih <= tie_margin.
    """
    ocean = ocean_only(scores)
    if not ocean:
        return "O"
    top_val = max(ocean.values())
    leaders = [k for k, v in ocean.items() if v >= top_val - tie_margin]
    if len(leaders) == 1:
        return leaders[0]
    intent_ocean = {
        "crisis": "N",
        "anxiety": "N",
        "sad": "N",
        "anger": "N",
        "negative_social": "N",
        "empathy_validation": "A",
        "relationship_affection": "A",
        "adaptive": "A",
        "social_positive": "E",
        "social_dependency": "E",
        "creative": "O",
        "discipline": "C",
        "achievement": "C",
        "mixed_positive": "A",
    }
    if intent_primary and intent_primary in intent_ocean:
        preferred = intent_ocean[intent_primary]
        if preferred in leaders:
            return preferred
    return max(ocean, key=ocean.get)


def clamp_ocean(scores: dict, lo: float = 1.0, hi: float = 5.0) -> dict:
    out = scores.copy()
    for k in OCEAN_TRAITS:
        if k in out:
            out[k] = round(max(lo, min(hi, out[k])), 3)
    return out


def limit_ocean_delta(raw: dict, adjusted: dict, max_delta: float = 1.15) -> dict:
    out = adjusted.copy()
    for k in OCEAN_TRAITS:
        if k not in raw or k not in out:
            continue
        delta = out[k] - raw[k]
        out[k] = round(raw[k] + max(-max_delta, min(max_delta, delta)), 3)
    return out


def scale_adjustment_delta(raw: dict, adjusted: dict, confidence: float) -> dict:
    """Campurkan adjustment ke raw sesuai confidence (teks pendek = lebih dekat ke model)."""
    out = adjusted.copy()
    conf = max(0.25, min(1.0, confidence))
    for k in OCEAN_TRAITS:
        if k not in raw or k not in out:
            continue
        out[k] = round(raw[k] + (out[k] - raw[k]) * conf, 3)
    if "EXTREME_ALERT" in adjusted:
        out["EXTREME_ALERT"] = round(adjusted["EXTREME_ALERT"] * conf, 3)
    return out
