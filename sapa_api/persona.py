import re
from collections import Counter

from sapa_api.config import OCEAN_TRAITS
from sapa_api.crisis import crisis_highlight_tokens, detect_crisis_level
from sapa_api.text_utils import (
    TextSufficiency,
    has_crisis_language,
    has_distress_language,
    has_empathy_validation_context,
    has_relationship_affection_context,
    has_positive_context,
    is_meaningful_token,
    ocean_only,
)
from sapa_api.sentiment_modifiers import analyze_modifiers
from sapa_api.phrase_intent import TextIntent, classify_text_intent
from sapa_api.persona_categories import resolve_persona_from_dataset
from sapa_api.trait_constructs import dominant_from_constructs


def _collect_highlights(evidence: dict, semantic_matches=None) -> set[str]:
    highlights = set()
    for key, items in evidence.items():
        for e in items:
            match_type = e.get("match_type", "exact")
            if match_type == "embedding":
                continue
            for t in e.get("matched_tokens", []):
                tl = t.lower()
                if is_meaningful_token(tl):
                    highlights.add(tl)
    if semantic_matches:
        for m in semantic_matches:
            if m.get("match_type") != "phrase":
                continue
            for t in m.get("lexeme_tokens", []):
                if is_meaningful_token(t):
                    highlights.add(t.lower())
    return highlights


def highlight_keywords_in_text(text: str, evidence: dict, semantic_matches=None):
    tokens = re.findall(r"\w+|\W+", text)
    highlights = _collect_highlights(evidence, semantic_matches)
    highlights |= crisis_highlight_tokens(text)
    mod = analyze_modifiers(text)
    for h in mod.hits:
        if h.modifier_type.startswith("intensifier") or h.modifier_type == "negation":
            highlights.add(h.modifier.lower())
            if h.target and len(h.target) >= 4:
                highlights.add(h.target.lower())
    return "".join(
        f"<mark>{t}</mark>" if t.lower() in highlights else t for t in tokens
    )


def extract_keywords(text, top_n=5):
    words = [
        w for w, _ in Counter(re.findall(r"\w+", text.lower())).most_common(top_n * 2)
        if is_meaningful_token(w)
    ]
    return words[:top_n]


def _semantic_snippet(semantic_matches, limit=3):
    if not semantic_matches:
        return ""
    lexemes = [m["lexeme"].replace("_", " ") for m in semantic_matches[:limit]]
    return ", ".join(lexemes)


def _is_crisis_context(adjusted: dict, text_lower: str) -> bool:
    if detect_crisis_level(text_lower) == "critical":
        return True
    alert = adjusted.get("EXTREME_ALERT", 0)
    ocean = ocean_only(adjusted)
    return (
        alert >= 3.5
        and ocean.get("N", 0) >= 4.3
        and has_crisis_language(text_lower)
    )


def _has_distress_not_crisis(text_lower: str, adjusted: dict) -> bool:
    return (
        has_distress_language(text_lower)
        and not has_crisis_language(text_lower)
        and detect_crisis_level(text_lower) == "none"
        and adjusted.get("EXTREME_ALERT", 0) < 2.5
    )


def generate_explanation_suggestion_super(
    text,
    adjusted,
    evidence,
    semantic_matches=None,
    sufficiency: TextSufficiency | None = None,
    construct_matches=None,
    text_intent: TextIntent | None = None,
):
    text_lower = text.lower()
    if text_intent is None:
        text_intent = classify_text_intent(text)
    ocean = ocean_only(adjusted)
    dominant = max(ocean, key=ocean.get)
    snippet = ", ".join(extract_keywords(text)[:3])
    sem_snip = _semantic_snippet(semantic_matches)
    if sem_snip:
        snippet = f"{snippet}, {sem_snip}" if snippet else sem_snip

    if _is_crisis_context(adjusted, text_lower):
        explanation = (
            f"⚠ Kalimat ini mengandung indikasi risiko tinggi terkait keselamatan diri. "
            f"Kecenderungan Neuroticism (N) dominan. "
            f"Pola linguistik: {snippet}."
        )
        suggestion = (
            "Sangat disarankan segera menghubungi layanan kesehatan mental, "
            "konselor, atau orang terpercaya. Jika darurat, hubungi layanan bantuan krisis."
        )
    elif _has_distress_not_crisis(text_lower, adjusted):
        constructs = ", ".join(
            m.phrase for m in (construct_matches or [])[:3]
        ) or snippet
        mod = analyze_modifiers(text_lower)
        mod_txt = ""
        if mod.hits:
            m0 = mod.hits[0]
            if m0.modifier_type.startswith("intensifier"):
                mod_txt = f" Intensitas linguistik («{m0.modifier}») memperkuat nuansa {m0.target}."
        explanation = (
            f"Teks menunjukkan kecenderungan Neuroticism (N) — kecemasan, stres, "
            f"atau kerentanan emosional (bukan indikasi krisis bunuh diri). "
            f"Indikator: {constructs}.{mod_txt}"
        )
        suggestion = (
            "Pertimbangkan manajemen stres, journaling, mindfulness, atau konseling "
            "jika pola ini berlangsung dan mengganggu fungsi sehari-hari."
        )
    elif text_intent.primary in (
        "anxiety", "sad", "anger", "crisis", "creative", "discipline",
        "achievement", "social_positive", "social_dependency", "negative_social",
        "relationship_affection", "adaptive", "empathy_validation",
    ):
        intent_labels = {
            "anxiety": "kecemasan (N)",
            "sad": "kesedihan (N)",
            "anger": "kemarahan (N)",
            "crisis": "risiko krisis (N)",
            "creative": "keterbukaan ide (O)",
            "discipline": "ketelitian (C)",
            "achievement": "orientasi prestasi (C)",
            "social_positive": "sosial aktif (E)",
            "social_dependency": "afiliasi & butuh dukungan (E)",
            "negative_social": "menghindar & isolasi sosial (N)",
            "relationship_affection": "romantis & afeksi hubungan (A)",
            "adaptive": "adaptasi positif (A/E)",
            "empathy_validation": "validasi emosional (A)",
        }
        label = intent_labels.get(text_intent.primary, text_intent.primary)
        constructs = ", ".join(
            m.phrase for m in (construct_matches or [])[:3]
        ) or snippet
        explanation = (
            f"Kalimat diklasifikasi sebagai pola {label}. "
            f"Indikator linguistik: {constructs}."
        )
        suggestion = (
            "Interpretasi mengikuti kombinasi kata dan intent teks, "
            "bukan skor model mentah saja."
        )
    elif has_empathy_validation_context(text_lower):
        constructs = ", ".join(
            m.phrase for m in (construct_matches or [])[:3]
        ) or snippet
        explanation = (
            f"Kalimat ini menggambarkan validasi emosional dan kelegaan (Agreeableness / A), "
            f"bukan kreativitas terbuka (O) atau distres (N). Indikator: {constructs}."
        )
        suggestion = (
            "Lanjutkan ruang aman untuk mengekspresikan perasaan; dukungan mendengarkan "
            "membantu menjaga kestabilan emosi."
        )
    elif has_positive_context(text_lower):
        explanation = (
            f"Kalimat ini menunjukkan pola positif/adaptif dengan kecenderungan {dominant} "
            f"(misalnya: {snippet})."
        )
        suggestion = (
            f"Pertahankan sikap seperti {snippet} untuk mendukung trait {dominant} "
            "secara seimbang."
        )
    else:
        sem_note = ""
        if semantic_matches:
            top = semantic_matches[0]
            sem_note = (
                f" Ontologi semantik: «{top['lexeme'].replace('_', ' ')}» "
                f"({top['sub_trait']}, similarity {top['similarity']})."
            )
        explanation = (
            f"Kalimat ini menunjukkan kecenderungan {dominant} karena pola seperti "
            f"{snippet}.{sem_note}"
        )
        suggestion = (
            f"Mengamati hal seperti {snippet} dapat membantu memahami trait {dominant}."
        )
    return explanation, suggestion


def determine_dominant_contextual(adjusted, evidence):
    scores = ocean_only(adjusted)
    scores["E"] += len(evidence.get("POSITIVE_SOCIAL", [])) * 0.3
    scores["A"] += len(evidence.get("EMPATHY_HARMONY_A", [])) * 0.4
    scores["C"] += len(evidence.get("DISCIPLINE_C", [])) * 0.4
    scores["O"] += len(evidence.get("CREATIVE_DISCUSSION_A", [])) * 0.4
    scores["N"] += len(evidence.get("ANXIETY_EMO", [])) * 0.5
    return max(scores, key=scores.get)


def normalize_scores(scores, min_val=1.0, max_val=5.0):
    for k in scores:
        scores[k] = max(min(scores[k], max_val), min_val)
    return scores


def _persona_scores(scores: dict) -> dict:
    ocean = ocean_only(scores)
    ocean["EXTREME_ALERT"] = scores.get("EXTREME_ALERT", 0)
    return ocean


ADAPTIVE_CONSTRUCT_PHRASES = frozenset({
    "mudah menyesuaikan diri",
    "menyesuaikan diri",
    "lingkungan baru",
    "beradaptasi",
    "adaptif",
    "fleksibel",
})


def _is_adaptive_persona(ps: dict, construct_matches=None) -> bool:
    if construct_matches and any(
        m.phrase in ADAPTIVE_CONSTRUCT_PHRASES for m in construct_matches
    ):
        return (
            ps.get("A", 0) >= 2.8
            and ps.get("N", 0) <= 3.8
            and ps.get("EXTREME_ALERT", 0) < 2
        )
    return (
        ps.get("E", 0) >= 2.5
        and ps.get("A", 0) >= 3.0
        and ps.get("N", 0) <= 3.8
        and ps.get("EXTREME_ALERT", 0) < 2
    )


PERSONA_RULES = [
    ("Empatik & Terdengar",
     lambda s: s.get("A", 0) >= 2.85 and s.get("N", 0) <= 3.7 and s.get("O", 0) <= 3.85,
     "merasa didengar, divalidasi, dan lebih ringan — berorientasi pada keharmonisan"),
    ("Adaptif & Fleksibel", lambda s: _is_adaptive_persona(s),
     "mudah beradaptasi, terbuka pada lingkungan baru, dan kooperatif"),
    ("Krisis Emosional Tinggi",
     lambda s: s.get("EXTREME_ALERT", 0) >= 3.5 and s.get("N", 0) >= 4.3,
     "menunjukkan tanda tekanan psikologis berat, putus asa, atau risiko menyakiti diri"),
    ("Depresi Mendalam",
     lambda s: s.get("EXTREME_ALERT", 0) >= 2.5 and s.get("E", 0) <= 2.5 and s.get("N", 0) >= 4.0,
     "menarik diri, kehilangan motivasi, dan mengalami kesedihan intens"),
    ("Burnout Mental Berat",
     lambda s: s.get("EXTREME_ALERT", 0) >= 2.5 and s.get("C", 0) <= 2.8 and s.get("N", 0) >= 3.8,
     "kelelahan emosional ekstrem, kehilangan arah, dan kehabisan energi"),
    ("Cemas & Overthinking",
     lambda s: s.get("N", 0) >= 3.6 and s.get("O", 0) >= 3.0,
     "mudah khawatir, banyak berpikir, reflektif terhadap masalah"),
    ("Tempramental",
     lambda s: s.get("N", 0) >= 4.0 and s.get("A", 0) <= 2.9 and s.get("C", 0) <= 3.0,
     "emosional, mudah tersulut, impulsif saat tertekan"),
    ("Melankolis Reflektif",
     lambda s: s.get("N", 0) >= 3.5 and s.get("O", 0) >= 3.3 and s.get("E", 0) <= 3.0,
     "sering merenung, introspektif, sensitif terhadap perasaan"),
    ("Stabil Emosional",
     lambda s: s.get("N", 0) <= 2.5,
     "tenang, terkendali, mampu mengelola tekanan dengan baik"),
    ("Tangguh Mental",
     lambda s: s.get("C", 0) >= 3.5 and s.get("N", 0) <= 3.0,
     "kuat secara mental, tidak mudah menyerah, fokus solusi"),
    ("Ekstrovert Sosial",
     lambda s: s.get("E", 0) >= 3.8 and s.get("A", 0) >= 3.2,
     "percaya diri, aktif berinteraksi, mudah bergaul"),
    ("Sosial Ekspresif",
     lambda s: s.get("E", 0) >= 3.5 and s.get("O", 0) >= 3.2,
     "komunikatif, ekspresif, suka berbagi ide"),
    ("Introvert Mandiri",
     lambda s: s.get("E", 0) <= 2.8 and s.get("C", 0) >= 3.2,
     "mandiri, fokus, nyaman bekerja sendiri"),
    ("Romantis Afektif",
     lambda s: s.get("A", 0) >= 3.2 and s.get("A", 0) >= s.get("O", 0) + 0.25,
     "hangat, penuh perhatian, berorientasi pada hubungan (Agreeableness)"),
    ("Empatik Caregiver",
     lambda s: s.get("A", 0) >= 3.7 and s.get("N", 0) <= 3.2,
     "peduli, protektif, senang membantu orang lain"),
    ("Relasional Selektif",
     lambda s: s.get("A", 0) >= 3.3 and s.get("E", 0) <= 3.0,
     "ramah namun berhati-hati dalam memilih relasi"),
    ("Visioner Kreatif",
     lambda s: s.get("O", 0) >= 3.7 and s.get("O", 0) >= s.get("A", 0) + 0.2,
     "imajinatif, visioner, berpikir jauh ke depan"),
    ("Pemikir Inovatif",
     lambda s: s.get("O", 0) >= 3.5 and s.get("E", 0) >= 3.2,
     "aktif menciptakan ide baru dan solusi kreatif"),
    ("Reflektif Analitis",
     lambda s: s.get("O", 0) >= 3.3 and s.get("C", 0) >= 3.3,
     "mendalam, sistematis, kritis dalam berpikir"),
    ("Perfeksionis Produktif",
     lambda s: s.get("C", 0) >= 3.8 and s.get("C", 0) >= s.get("N", 0) + 0.2,
     "teliti, terstruktur, menuntut standar tinggi"),
    ("Ambisius Visioner",
     lambda s: s.get("C", 0) >= 3.5 and s.get("O", 0) >= 3.5,
     "berorientasi prestasi, berpikir strategis"),
    ("Pragmatis Efisien",
     lambda s: s.get("C", 0) >= 3.3 and s.get("E", 0) >= 3.2,
     "praktis, fokus hasil, cepat mengambil keputusan"),
    ("Gigih & Persisten",
     lambda s: s.get("C", 0) >= 3.4 and s.get("N", 0) <= 3.0,
     "konsisten, tahan tekanan, tidak mudah menyerah"),
    ("Seimbang Adaptif",
     lambda s: all(2.8 <= s[k] <= 3.6 for k in OCEAN_TRAITS),
     "fleksibel, stabil, mampu menyesuaikan diri di berbagai situasi"),
]


def generate_persona_profile(
    scores: dict,
    text: str = "",
    sufficiency: TextSufficiency | None = None,
    construct_matches=None,
    text_intent: TextIntent | None = None,
    evidence: dict | None = None,
):
    ps = _persona_scores(scores)
    text_lower = text.lower()
    if text_intent is None:
        text_intent = classify_text_intent(text)

    resolved = resolve_persona_from_dataset(
        scores, text, intent=text_intent, evidence=evidence
    )
    if resolved:
        label, desc, _cat = resolved
        return [f"Kepribadian : {label} — {desc}"]

    positive = has_positive_context(text_lower)
    validation = has_empathy_validation_context(text_lower) or text_intent.primary == "empathy_validation"
    romantic = (
        has_relationship_affection_context(text_lower)
        or text_intent.primary == "relationship_affection"
    )
    crisis = _is_crisis_context(scores, text_lower)
    crisis_tier = detect_crisis_level(text_lower)
    distress = _has_distress_not_crisis(text_lower, scores)
    best_label = "Seimbang"
    best_desc = "adaptif, fleksibel, dan tidak ekstrem pada satu trait"
    best_priority = -1

    construct_dom = dominant_from_constructs(ocean_only(ps), construct_matches or [])

    for label, cond, desc in PERSONA_RULES:
        if label == "Empatik & Terdengar":
            if not validation or not cond(ps):
                continue
        elif label == "Adaptif & Fleksibel":
            if not positive or not _is_adaptive_persona(ps, construct_matches):
                continue
        elif label == "Krisis Emosional Tinggi":
            if not crisis:
                continue
        elif label == "Cemas & Overthinking":
            if text_intent.primary not in ("anxiety",) and (
                not distress and construct_dom != "N"
            ):
                continue
            if text_intent.primary != "anxiety" and not cond(ps):
                continue
        elif label == "Tempramental":
            if crisis or not cond(ps):
                continue
            if distress and text_intent.primary not in ("anger",):
                continue
        elif label == "Perfeksionis Produktif":
            if text_intent.primary == "discipline":
                if ps.get("C", 0) < 2.95:
                    continue
            elif not cond(ps):
                continue
        elif label == "Ekstrovert Sosial":
            if text_intent.primary == "social_positive":
                if ps.get("E", 0) < 2.4:
                    continue
            elif not cond(ps):
                continue
        elif label == "Romantis Afektif":
            if not romantic or not cond(ps):
                continue
        elif not cond(ps):
            continue
        if label == "Empatik & Terdengar" and validation:
            prio = 105
        elif label == "Adaptif & Fleksibel":
            prio = 100
        elif label == "Krisis Emosional Tinggi":
            prio = 200 if crisis_tier == "critical" else 120
        elif label == "Cemas & Overthinking" and (
            text_intent.primary == "anxiety" or distress
        ):
            prio = 120 if text_intent.primary == "anxiety" else 110
        elif label == "Tempramental" and text_intent.primary == "anger":
            prio = 115
        elif label == "Perfeksionis Produktif" and text_intent.primary == "discipline":
            prio = 100
        elif label == "Ekstrovert Sosial" and text_intent.primary == "social_positive":
            prio = 100
        elif label == "Romantis Afektif" and text_intent.primary == "relationship_affection":
            prio = 125
        elif label == "Seimbang Adaptif":
            if text_intent.primary in (
                "anxiety", "sad", "anger", "crisis", "empathy_validation",
                "creative", "discipline", "achievement", "social_positive",
                "relationship_affection",
            ):
                continue
            prio = 15
        else:
            prio = 20
        if crisis_tier == "critical" and label in ("Tempramental", "Melankolis Reflektif"):
            continue
        if distress and label in ("Visioner Kreatif", "Melankolis Reflektif"):
            continue
        if validation and label in (
            "Tempramental", "Melankolis Reflektif", "Visioner Kreatif",
            "Pemikir Inovatif", "Reflektif Analitis",
        ):
            continue
        if text_intent.primary == "anxiety" and label in (
            "Visioner Kreatif", "Pemikir Inovatif", "Tempramental", "Melankolis Reflektif",
        ):
            continue
        if text_intent.primary in ("discipline", "achievement") and label in (
            "Visioner Kreatif", "Melankolis Reflektif", "Cemas & Overthinking",
        ):
            continue
        if text_intent.primary == "creative" and label in (
            "Tempramental", "Cemas & Overthinking", "Empatik & Terdengar",
        ):
            continue
        if sufficiency and sufficiency.analysis_mode == "state" and label not in (
            "Cemas & Overthinking", "Stabil Emosional", "Seimbang Adaptif", "Krisis Emosional Tinggi"
        ):
            prio -= 30
        ocean = ocean_only(ps)
        dominant_trait = max(ocean, key=ocean.get)
        dominance = ocean[dominant_trait] - sum(
            v for k, v in ocean.items() if k != dominant_trait
        ) / 4
        score = prio + dominance
        if score > best_priority:
            best_priority = score
            best_label = label
            best_desc = desc

    return [f"Kepribadian : {best_label} — {best_desc}"]


def generate_global_conclusion(avg, dominant):
    O, C, E, A, N = avg["O"], avg["C"], avg["E"], avg["A"], avg["N"]
    conclusion = (
        f"Secara keseluruhan, trait kepribadian paling dominan adalah {dominant}. Individu ini cenderung "
    )
    if avg.get("EXTREME_ALERT", 0) >= 3.5:
        conclusion = (
            "Terdapat indikasi tekanan emosional yang sangat tinggi dan risiko kesehatan mental."
        )
        suggestion = (
            "Sangat disarankan untuk segera mencari dukungan profesional, "
            "berbicara dengan orang terpercaya, atau menghubungi layanan bantuan psikologis."
        )
        return conclusion, suggestion

    if dominant == "O":
        conclusion += "kreatif, reflektif, dan terbuka terhadap ide baru."
    elif dominant == "C":
        conclusion += "terstruktur, disiplin, konsisten, dan bertanggung jawab."
    elif dominant == "E":
        conclusion += "aktif secara sosial, komunikatif, dan energik."
    elif dominant == "A":
        conclusion += "kooperatif, empatik, dan menjaga keharmonisan sosial."
    elif dominant == "N":
        conclusion += "sensitif terhadap tekanan emosional dan mudah mengalami stres."

    suggestion = "Disarankan untuk "
    if dominant == "C":
        suggestion += "memanfaatkan kemampuan perencanaan dan kedisiplinan, tetapi tetap fleksibel."
    elif dominant == "O":
        suggestion += "menyalurkan kreativitas ke aktivitas produktif dan eksplorasi ide."
    elif dominant == "E":
        suggestion += "mengoptimalkan kemampuan komunikasi, kepemimpinan, dan refleksi diri."
    elif dominant == "A":
        suggestion += "mempertahankan empati sambil belajar bersikap tegas saat dibutuhkan."
    elif dominant == "N":
        suggestion += "melatih regulasi emosi melalui manajemen stres, mindfulness, atau journaling rutin."

    if N <= 2.5:
        conclusion += " Tingkat kestabilan emosi tergolong baik."
    elif N >= 3.6:
        conclusion += " Terdapat kecenderungan emosi negatif yang cukup tinggi."

    return conclusion, suggestion
