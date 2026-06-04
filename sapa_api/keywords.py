import re
from collections import Counter

import pandas as pd

from sapa_api.config import KEYWORDS_XLSX, TRAIT_LIST_NAMES
from sapa_api.crisis import CRISIS_CRITICAL_PHRASES, detect_crisis_level
from sapa_api.fuzzy_match import fuzzy_phrase_match
from sapa_api.text_utils import (
    assess_text_sufficiency,
    clamp_ocean,
    has_crisis_language,
    has_distress_language,
    has_empathy_validation_context,
    has_relationship_affection_context,
    has_positive_context,
    has_social_enjoyment_context,
    align_scores_to_intent_dominance,
    dominant_from_adjusted_scores,
    is_meaningful_token,
    limit_ocean_delta,
    ocean_only,
    scale_adjustment_delta,
    tokenize,
)
from sapa_api.sentiment_modifiers import phrase_modifier_scale
from sapa_api.phrase_intent import (
    apply_intent_ocean_adjustment,
    classify_text_intent,
    dominant_from_intent,
)
from sapa_api.trait_constructs import (
    apply_construct_adjustments,
    dominant_from_constructs,
)
from sapa_api.keyword_index import build_phrase_index, match_trait_phrases

print("Mencoba membuka file:", KEYWORDS_XLSX)
_kw_df = pd.read_excel(KEYWORDS_XLSX)
if len(_kw_df.columns) >= 2:
    _kw_df = _kw_df.iloc[:, :2]
    _kw_df.columns = ["Keyword / Phrase", "Trait / Kategori"]

TRAIT_KEYWORDS: dict[str, list[str]] = {}
for _, row in _kw_df.iterrows():
    trait = str(row["Trait / Kategori"]).strip()
    word = str(row["Keyword / Phrase"]).strip().lower()
    if " " not in word and not is_meaningful_token(word):
        continue
    TRAIT_KEYWORDS.setdefault(trait, []).append(word)

for _name in TRAIT_LIST_NAMES:
    globals()[_name] = TRAIT_KEYWORDS.get(_name, [])

for trait, words in TRAIT_KEYWORDS.items():
    print(f"{trait}: {len(words)} kata")

KEYWORD_TRAIT_MAP = {
    # Optimized weights untuk prediksi lebih akurat (v2.1)
    "ANGER_EMO": {"N": 0.5, "A": -0.2, "C": -0.1},
    "RAGE_OVERFLOW": {"N": 0.78, "A": -0.38, "C": -0.28, "E": 0.08},
    "EMO_SURGE": {"N": 0.68, "A": -0.22, "C": -0.18, "E": 0.05},
    "CRITICAL_HOSTILE": {"N": 0.58, "A": -0.42, "O": 0.06, "C": -0.12, "E": -0.08},
    "HATRED_EMO": {"N": 0.72, "A": -0.48, "E": -0.18, "C": -0.12, "O": -0.05},
    "SAD_EMO": {"N": 0.4, "O": 0.05, "E": -0.1},
    "ANXIETY_EMO": {"N": 0.55, "E": -0.2, "O": -0.1},
    "MENTAL_UNSTABLE_N": {"N": 0.8, "C": -0.2},
    "NEGATIVE_SOCIAL": {"N": 0.4, "A": -0.3, "E": -0.2, "C": -0.1},
    "POSITIVE_SOCIAL": {"E": 0.58, "A": 0.18, "N": -0.15, "O": 0.05},
    "EXTRAVERSION_E": {"E": 0.62, "A": 0.15, "N": -0.15, "O": 0.05},
    "E_SOCIAL_DEPENDENCY": {"E": 0.4, "A": 0.2, "N": 0.05},
    "COLLABORATION": {"A": 0.6, "E": 0.3, "C": 0.2, "N": -0.1},
    "RELATIONSHIP_AFFECTION": {"A": 0.78, "E": 0.18, "O": -0.08, "N": -0.1},
    "EMPATHY_HARMONY_A": {"A": 0.7, "N": -0.3, "E": 0.1},
    "TRUST": {"A": 0.5, "N": -0.15, "E": 0.1},
    "CREATIVE_DISCUSSION_A": {"O": 0.6, "E": 0.2, "A": 0.1},
    "INTROSPECTION": {"O": 0.35, "N": 0.1, "E": -0.08},
    "DISCIPLINE_C": {"C": 0.85, "N": -0.2, "E": -0.1},
    "ACHIEVEMENT": {"C": 0.6, "E": 0.2, "O": 0.15, "N": -0.1},
    "EMO_POSITIVE": {"A": 0.3, "E": 0.3, "N": -0.2, "O": 0.1},
    "EXTREME_NEGATIVE": {"N": 2.0, "E": -0.8, "A": -0.6, "C": -0.6, "O": -0.4},
    "EMO_NEGATIVE": {"N": 0.5, "E": -0.15, "A": -0.15, "O": 0.05, "C": -0.1},
}

# Saat keyword sama ada di banyak sheet-kategori, hanya kategori prioritas tertinggi
TRAIT_MATCH_PRIORITY = (
    "EXTREME_NEGATIVE",
    "HATRED_EMO",
    "RAGE_OVERFLOW",
    "CRITICAL_HOSTILE",
    "EMO_SURGE",
    "ANXIETY_EMO",
    "SAD_EMO",
    "ANGER_EMO",
    "MENTAL_UNSTABLE_N",
    "EMO_NEGATIVE",
    "NEGATIVE_SOCIAL",
    "EXTRAVERSION_E",
    "POSITIVE_SOCIAL",
    "EMO_POSITIVE",
    "COLLABORATION",
    "RELATIONSHIP_AFFECTION",
    "EMPATHY_HARMONY_A",
    "TRUST",
    "CREATIVE_DISCUSSION_A",
    "INTROSPECTION",
    "DISCIPLINE_C",
    "ACHIEVEMENT",
    "E_SOCIAL_DEPENDENCY",
)

_CRISIS_EXTREME_PHRASES = frozenset(CRISIS_CRITICAL_PHRASES) | frozenset({
    "bunuh diri", "pengen bunuh diri", "mau bunuh diri", "pengen mati",
    "mau mati", "tidak mau hidup", "ingin menghilang",
})

POSITIVE_TRAITS = frozenset({
    "POSITIVE_SOCIAL", "EXTRAVERSION_E", "EMO_POSITIVE", "COLLABORATION",
    "RELATIONSHIP_AFFECTION", "EMPATHY_HARMONY_A", "TRUST",
    "CREATIVE_DISCUSSION_A", "ACHIEVEMENT", "E_SOCIAL_DEPENDENCY",
})

NEGATIVE_TRAITS = frozenset({
    "ANGER_EMO", "RAGE_OVERFLOW", "EMO_SURGE", "CRITICAL_HOSTILE", "HATRED_EMO",
    "SAD_EMO", "ANXIETY_EMO", "NEGATIVE_SOCIAL",
    "EXTREME_NEGATIVE", "MENTAL_UNSTABLE_N", "EMO_NEGATIVE",
})


def _phrase_is_crisis_extreme(phrase: str) -> bool:
    if phrase in _CRISIS_EXTREME_PHRASES:
        return True
    return any(c in phrase for c in ("bunuh diri", "ingin bunuh", "pengen bunuh", "mau bunuh"))


def _build_single_word_trait_map() -> dict[str, str]:
    """Satu token -> satu kategori (hindari N dobel dari 156 konflik)."""
    word_traits: dict[str, list[str]] = {}
    for trait, keywords in TRAIT_KEYWORDS.items():
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        for word in keywords:
            if " " in word or not is_meaningful_token(word):
                continue
            word_traits.setdefault(word, []).append(trait)
    resolved: dict[str, str] = {}
    for word, traits in word_traits.items():
        for t in TRAIT_MATCH_PRIORITY:
            if t in traits:
                resolved[word] = t
                break
        else:
            resolved[word] = traits[0]
    return resolved


_SINGLE_WORD_TRAIT = _build_single_word_trait_map()
_PHRASE_INDEX = build_phrase_index(TRAIT_KEYWORDS)

_BUNUH_VARIANTS = (
    ("ingin bunuh diri", ("pengen bunuh diri", "mau bunuh diri", "pengen bunuh")),
    ("ingin bunuh", ("pengen bunuh", "mau bunuh", "pengen bunuh diri")),
)


def _phrase_matches_variant(phrase: str, text_lower: str) -> bool:
    if phrase in text_lower:
        return True
    for canonical, variants in _BUNUH_VARIANTS:
        if phrase == canonical or phrase in variants:
            return any(v in text_lower for v in variants) or canonical in text_lower
    return False


def _apply_trait_weights(adjusted: dict, weights: dict, scale: float):
    for ocean_dim, w in weights.items():
        adjusted[ocean_dim] = adjusted.get(ocean_dim, 3.0) + w * scale


def adjust_ocean_by_keywords(
    scores: dict,
    text: str,
    confidence_scale: float = 1.0,
    *,
    use_fuzzy_phrases: bool = True,
):
    adjusted = {**ocean_only(scores), "EXTREME_ALERT": 0.0}
    text_lower = text.lower()
    tokens = tokenize(text)
    counter = Counter(tokens)
    positive_ctx = has_positive_context(text_lower)
    crisis_lang = has_crisis_language(text_lower)
    distress_lang = has_distress_language(text_lower)
    matched_phrases: list[tuple[str, str]] = []

    adjusted, construct_matches = apply_construct_adjustments(
        adjusted, text, scale=confidence_scale
    )

    excel_scale = 0.55 * confidence_scale
    for word, freq in counter.items():
        if not is_meaningful_token(word) or freq < 1:
            continue
        trait = _SINGLE_WORD_TRAIT.get(word)
        if not trait:
            continue
        weights = KEYWORD_TRAIT_MAP[trait]
        if trait == "EXTREME_NEGATIVE":
            if not crisis_lang:
                continue
            adjusted["EXTREME_ALERT"] += freq * 1.0 * confidence_scale
        _apply_trait_weights(adjusted, weights, freq * 0.5 * excel_scale)

    for trait, phrase in match_trait_phrases(
        text_lower,
        _PHRASE_INDEX,
        phrase_matches_variant=_phrase_matches_variant,
        fuzzy_match=fuzzy_phrase_match,
        use_fuzzy=use_fuzzy_phrases,
    ):
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        if trait == "EXTREME_NEGATIVE" and not (
            crisis_lang or _phrase_is_crisis_extreme(phrase)
        ):
            continue

        weights = KEYWORD_TRAIT_MAP[trait]
        matched_phrases.append((trait, phrase))
        scale = excel_scale
        mod_scale, _ = phrase_modifier_scale(text_lower, phrase)
        scale *= max(0.3, mod_scale)
        if trait == "EXTREME_NEGATIVE" and crisis_lang:
            adjusted["EXTREME_ALERT"] += 2.5 * confidence_scale * mod_scale
        elif trait in POSITIVE_TRAITS:
            scale *= 1.05
        if mod_scale < 0:
            inv = {k: -v for k, v in weights.items()}
            _apply_trait_weights(adjusted, inv, abs(scale))
        else:
            _apply_trait_weights(adjusted, weights, scale)

        if trait == "EMPATHY_HARMONY_A" and has_empathy_validation_context(text_lower):
            adjusted["A"] = adjusted.get("A", 3.0) + 0.28 * confidence_scale
            adjusted["N"] = adjusted.get("N", 3.0) - 0.15 * confidence_scale
            adjusted["O"] = adjusted.get("O", 3.0) - 0.18 * confidence_scale

        if trait == "RELATIONSHIP_AFFECTION" and has_relationship_affection_context(text_lower):
            adjusted["A"] = adjusted.get("A", 3.0) + 0.35 * confidence_scale
            adjusted["E"] = adjusted.get("E", 3.0) + 0.1 * confidence_scale
            adjusted["N"] = adjusted.get("N", 3.0) - 0.1 * confidence_scale
            adjusted["O"] = adjusted.get("O", 3.0) - 0.18 * confidence_scale

    if positive_ctx and not distress_lang:
        adjusted["EXTREME_ALERT"] = min(adjusted["EXTREME_ALERT"], 0.5)
        if any(t in POSITIVE_TRAITS for t, _ in matched_phrases):
            adjusted["E"] += 0.15 * confidence_scale
            adjusted["A"] += 0.15 * confidence_scale

    if not crisis_lang:
        adjusted["EXTREME_ALERT"] = min(adjusted["EXTREME_ALERT"], 1.5 * confidence_scale)

    intent = classify_text_intent(text, matched_phrases=matched_phrases)
    adjusted = apply_intent_ocean_adjustment(adjusted, intent, confidence_scale)

    adjusted = clamp_ocean(adjusted)
    if detect_crisis_level(text_lower) == "none":
        adjusted = limit_ocean_delta(scores, adjusted, max_delta=0.95 * confidence_scale + 0.2)

    adjusted = align_scores_to_intent_dominance(adjusted, intent.primary, margin=0.12)
    adjusted = clamp_ocean(adjusted)
    adjusted["EXTREME_ALERT"] = round(min(adjusted.get("EXTREME_ALERT", 0), 5.0), 3)

    dominant = dominant_from_adjusted_scores(adjusted, intent.primary)
    return dominant, adjusted, construct_matches, intent


def apply_emotional_keyword_adjustment(
    text: str, scores: dict, o_reduce: float = 0.1, confidence_scale: float = 1.0
):
    adjusted = scores.copy()
    text_lower = text.lower()
    tokens = tokenize(text)
    counter = Counter(tokens)
    positive_ctx = has_positive_context(text_lower)
    distress = has_distress_language(text_lower)

    if distress:
        n_boost = 0.25 * confidence_scale
        adjusted["N"] = adjusted.get("N", 3.0) + n_boost
        if "cemas" in tokens or "stres" in tokens or "terganggu" in tokens:
            adjusted["E"] = adjusted.get("E", 3.0) - 0.08 * confidence_scale
        return clamp_ocean(adjusted)

    if has_social_enjoyment_context(text_lower):
        adjusted["E"] = adjusted.get("E", 3.0) + 0.15 * confidence_scale
        adjusted["A"] = adjusted.get("A", 3.0) + 0.04 * confidence_scale
        adjusted["O"] = adjusted.get("O", 3.0) - 0.08 * confidence_scale
        adjusted["N"] = adjusted.get("N", 3.0) - 0.06 * confidence_scale
        return clamp_ocean(adjusted)

    if has_relationship_affection_context(text_lower):
        adjusted["A"] = adjusted.get("A", 3.0) + 0.2 * confidence_scale
        adjusted["E"] = adjusted.get("E", 3.0) + 0.08 * confidence_scale
        adjusted["N"] = adjusted.get("N", 3.0) - 0.08 * confidence_scale
        adjusted["O"] = adjusted.get("O", 3.0) - 0.16 * confidence_scale
        return clamp_ocean(adjusted)

    if positive_ctx:
        a_boost = 0.18 if has_empathy_validation_context(text_lower) else 0.12
        adjusted["A"] = adjusted.get("A", 3.0) + a_boost * confidence_scale
        adjusted["E"] = adjusted.get("E", 3.0) + 0.08 * confidence_scale
        adjusted["N"] = adjusted.get("N", 3.0) - 0.15 * confidence_scale
        if has_empathy_validation_context(text_lower):
            adjusted["O"] = adjusted.get("O", 3.0) - 0.2 * confidence_scale
        return clamp_ocean(adjusted)

    neg_score = sum(
        counter[w] for w in tokens
        if is_meaningful_token(w) and w in (
            ANGER_EMO + RAGE_OVERFLOW + EMO_SURGE + CRITICAL_HOSTILE + HATRED_EMO
            + SAD_EMO + ANXIETY_EMO
        )
    )
    pos_score = sum(
        counter[w] for w in tokens
        if is_meaningful_token(w) and w in EMO_POSITIVE + POSITIVE_SOCIAL + TRUST
    )

    if neg_score > pos_score and has_distress_language(text_lower):
        diff = min(0.8, (neg_score - pos_score) * 0.1) * confidence_scale
        adjusted["N"] += diff
        adjusted["E"] -= diff * 0.3
    elif pos_score > neg_score:
        diff = min(0.8, (pos_score - neg_score) * 0.1) * confidence_scale
        adjusted["A"] += diff
        adjusted["E"] += diff * 0.2
        adjusted["N"] -= diff * 0.2

    if not positive_ctx and not distress:
        creative_hits = sum(
            counter[w] for w in EMO_POSITIVE + CREATIVE_DISCUSSION_A + INTROSPECTION
            if is_meaningful_token(w) and w in counter
        )
        if creative_hits == 0:
            adjusted["O"] = adjusted.get("O", 3.0) - o_reduce * confidence_scale

    return clamp_ocean(adjusted)


def determine_dominant_trait(
    scores: dict,
    text: str,
    construct_matches=None,
    intent=None,
) -> str:
    text_lower = text.lower()
    if intent is None:
        intent = classify_text_intent(text)

    work = align_scores_to_intent_dominance(scores, intent.primary, margin=0.12)
    work = clamp_ocean(work)
    ocean = ocean_only(work)
    crisis_level = detect_crisis_level(text_lower)
    if crisis_level in ("critical", "high") and ocean.get("N", 0) >= max(ocean.values()) - 0.2:
        return "N"
    if work.get("EXTREME_ALERT", 0) >= 3.5 and has_crisis_language(text_lower):
        if ocean.get("N", 0) >= max(ocean.values()) - 0.15:
            return "N"

    return dominant_from_adjusted_scores(work, intent.primary)
