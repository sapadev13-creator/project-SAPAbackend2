import torch
from fastapi import HTTPException

from sapa_api import state
from sapa_api.config import DEVICE, MAX_LEN
from sapa_api.crisis import apply_crisis_adjustment, detect_crisis_level
from sapa_api.fuzzy_match import normalize_text_typos
from sapa_api.keywords import (
    adjust_ocean_by_keywords,
    apply_emotional_keyword_adjustment,
    determine_dominant_trait,
)
from sapa_api.lexical import build_lexical_vector_with_analysis
from sapa_api.persona import (
    generate_explanation_suggestion_super,
    generate_persona_profile,
    highlight_keywords_in_text,
)
from sapa_api.persona_categories import dominant_keyword_category as _dominant_keyword_category
from sapa_api.semantic import (
    adjust_ocean_by_semantic,
    aggregate_semantic_subtraits,
    matches_to_expansion_candidates,
)
from sapa_api.text_utils import (
    align_scores_to_intent_dominance,
    assess_text_sufficiency,
    clamp_ocean,
    limit_ocean_delta,
    scale_adjustment_delta,
)
from sapa_api.sentiment_modifiers import (
    apply_modifier_ocean_adjustment,
    modifier_explanation_note,
    modifiers_to_dict,
)
from sapa_api.trait_constructs import construct_evidence
from sapa_api.viz import generate_ocean_chart


def _model_raw_scores(text: str, lexical) -> dict:
    enc = state.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = state.model(
            enc["input_ids"],
            enc["attention_mask"],
            lexical,
        )
    return {
        "O": round(out[0, 0].item(), 3),
        "C": round(out[0, 1].item(), 3),
        "E": round(out[0, 2].item(), 3),
        "A": round(out[0, 3].item(), 3),
        "N": round(out[0, 4].item(), 3),
    }


def compute_ocean_prediction(text: str):
    """Inferensi + konstruk Big Five + adjustment (dengan penyesuaian teks pendek)."""
    if state.tokenizer is None or state.model is None:
        raise HTTPException(503, "Model not ready")

    typo_norm = normalize_text_typos(text)
    work_text = typo_norm.normalized
    sufficiency = assess_text_sufficiency(work_text)
    conf = sufficiency.confidence_scale

    lexical, coverage, subtraits, evidence, semantic_matches = (
        build_lexical_vector_with_analysis(work_text)
    )
    lexical = lexical.to(DEVICE)
    if lexical.dim() == 1:
        lexical = lexical.unsqueeze(0)

    raw = _model_raw_scores(work_text, lexical)
    _, adjusted, construct_matches, text_intent = adjust_ocean_by_keywords(
        raw, work_text, conf
    )
    adjusted = adjust_ocean_by_semantic(
        adjusted, semantic_matches, work_text, conf, text_intent=text_intent
    )
    adjusted = apply_emotional_keyword_adjustment(work_text, adjusted, confidence_scale=conf)
    adjusted, modifier_analysis = apply_modifier_ocean_adjustment(
        adjusted, work_text, confidence_scale=conf
    )

    crisis_level = detect_crisis_level(work_text)
    if crisis_level == "none":
        adjusted = limit_ocean_delta(raw, adjusted, max_delta=0.95 * conf + 0.2)
    adjusted, crisis_level = apply_crisis_adjustment(adjusted, work_text)

    if sufficiency.analysis_mode != "trait" and crisis_level == "none":
        adjusted = scale_adjustment_delta(raw, adjusted, conf)

    adjusted = align_scores_to_intent_dominance(adjusted, text_intent.primary, margin=0.12)
    adjusted = clamp_ocean(adjusted)
    dominant = determine_dominant_trait(
        adjusted, work_text, construct_matches, intent=text_intent
    )
    explanation, suggestion = generate_explanation_suggestion_super(
        work_text,
        adjusted,
        evidence,
        semantic_matches,
        sufficiency=sufficiency,
        construct_matches=construct_matches,
        text_intent=text_intent,
    )

    mod_note = modifier_explanation_note(modifier_analysis)
    if mod_note:
        explanation = explanation.rstrip(".") + "." + mod_note
    if sufficiency.disclaimer:
        explanation = f"{sufficiency.disclaimer} {explanation}"

    construct_ev = construct_evidence(construct_matches)
    if construct_ev:
        evidence = {**evidence, "big_five_constructs": construct_ev}

    return {
        "input_text": text,
        "text_normalized": work_text,
        "typo_corrections": [
            {"original": c.original, "corrected": c.corrected, "confidence": c.confidence}
            for c in typo_norm.corrections
        ],
        "analysis_mode": sufficiency.analysis_mode,
        "confidence_scale": sufficiency.confidence_scale,
        "meaningful_token_count": sufficiency.meaningful_count,
        "disclaimer": sufficiency.disclaimer,
        "raw": raw,
        "adjusted": adjusted,
        "dominant": dominant,
        "coverage": coverage,
        "subtraits": subtraits,
        "evidence": evidence,
        "semantic_matches": semantic_matches,
        "semantic_subtraits": aggregate_semantic_subtraits(semantic_matches),
        "ontology_expansion_candidates": matches_to_expansion_candidates(
            semantic_matches, top_k=8
        ),
        "explanation": explanation,
        "suggestion": suggestion,
        "highlighted_text": highlight_keywords_in_text(
            text, evidence, semantic_matches
        ),
        "text_intent": text_intent.primary,
        "dominant_keyword_category": _dominant_keyword_category(
            text_intent, evidence
        ),
        "personality_profile": generate_persona_profile(
            adjusted,
            work_text,
            sufficiency=sufficiency,
            construct_matches=construct_matches,
            text_intent=text_intent,
            evidence=evidence,
        ),
        "crisis_level": crisis_level,
        "sentiment_modifiers": modifiers_to_dict(modifier_analysis),
    }


def run_ocean_pipeline(text: str, username: str | None = None):
    """Pipeline untuk Twitter / Excel — membungkus compute_ocean_prediction."""
    pred = compute_ocean_prediction(text)
    try:
        chart = generate_ocean_chart(pred["adjusted"])
    except Exception:
        chart = None

    return {
        "username": username,
        "highlighted_text": pred["highlighted_text"],
        "prediction_adjusted": pred["adjusted"],
        "dominant_trait": pred["dominant"],
        "dominant_keyword_category": pred.get("dominant_keyword_category"),
        "text_intent": pred.get("text_intent"),
        "personality_profile": pred["personality_profile"],
        "explanation": pred["explanation"],
        "suggestion": pred["suggestion"],
        "ocean_chart_base64": chart,
        "ontology_analysis": {
            "coverage_percent": pred["coverage"],
            "active_subtraits": pred["subtraits"],
            "semantic_subtraits": pred["semantic_subtraits"],
        },
        "ontology_expansion_candidates": pred["ontology_expansion_candidates"],
        "lexical_evidence": pred["evidence"],
        "text_normalized": pred.get("text_normalized"),
        "typo_corrections": pred.get("typo_corrections", []),
        "analysis_mode": pred.get("analysis_mode"),
        "confidence_scale": pred.get("confidence_scale"),
        "meaningful_token_count": pred.get("meaningful_token_count"),
        "disclaimer": pred.get("disclaimer"),
        "crisis_level": pred.get("crisis_level", "none"),
        "big_five_constructs": pred.get("evidence", {}).get("big_five_constructs", []),
        "sentiment_modifiers": pred.get("sentiment_modifiers", {}),
    }
