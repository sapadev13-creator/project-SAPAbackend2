"""
Pemrosesan batch / Excel — inferensi model & semantic di-chunk agar lebih cepat.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from fastapi import HTTPException

from sapa_api import state
from sapa_api.config import DEVICE, MAX_LEN, OCEAN_TRAITS
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
from sapa_api.persona_categories import dominant_keyword_category
from sapa_api.semantic import (
    adjust_ocean_by_semantic,
    aggregate_semantic_subtraits,
    encode_texts_batch,
    matches_to_expansion_candidates,
)
from sapa_api.sentiment_modifiers import (
    apply_modifier_ocean_adjustment,
    modifier_explanation_note,
    modifiers_to_dict,
)
from sapa_api.text_utils import (
    align_scores_to_intent_dominance,
    assess_text_sufficiency,
    clamp_ocean,
    limit_ocean_delta,
    scale_adjustment_delta,
)
from sapa_api.trait_constructs import construct_evidence
from sapa_api.viz import generate_ocean_chart

DEFAULT_BATCH_SIZE = 16


def _model_raw_scores_batch(
    texts: list[str],
    lexical_tensors: list[torch.Tensor],
) -> list[dict]:
    if state.tokenizer is None or state.model is None:
        raise HTTPException(503, "Model not ready")

    enc = state.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    lexical_batch = torch.stack(lexical_tensors, dim=0).to(DEVICE)

    with torch.no_grad():
        out = state.model(
            enc["input_ids"],
            enc["attention_mask"],
            lexical_batch,
        )

    rows = []
    for i in range(len(texts)):
        rows.append({
            "O": round(out[i, 0].item(), 3),
            "C": round(out[i, 1].item(), 3),
            "E": round(out[i, 2].item(), 3),
            "A": round(out[i, 3].item(), 3),
            "N": round(out[i, 4].item(), 3),
        })
    return rows


def compute_ocean_prediction_lite(
    text: str,
    *,
    precomputed_raw: dict | None = None,
    precomputed_lexical: tuple | None = None,
    text_embedding=None,
    skip_semantic_embedding: bool = False,
    skip_chart: bool = True,
    skip_heavy_evidence: bool = False,
) -> dict:
    """Satu teks; opsi tanpa chart / semantic embedding penuh (mode Excel cepat)."""
    if state.tokenizer is None or state.model is None:
        raise HTTPException(503, "Model not ready")

    typo_norm = normalize_text_typos(text)
    work_text = typo_norm.normalized
    sufficiency = assess_text_sufficiency(work_text)
    conf = sufficiency.confidence_scale

    if precomputed_lexical is not None:
        lexical, coverage, subtraits, evidence, semantic_matches = precomputed_lexical
    else:
        semantic_vec = None
        if text_embedding is not None:
            semantic_vec = text_embedding
        lexical, coverage, subtraits, evidence, semantic_matches = (
            build_lexical_vector_with_analysis(
                work_text,
                semantic_text_vec=semantic_vec,
                skip_semantic_embedding=skip_semantic_embedding,
            )
        )

    if precomputed_raw is not None:
        raw = precomputed_raw
    else:
        lex = lexical if lexical.dim() == 2 else lexical.unsqueeze(0)
        raw = _model_raw_scores_batch([work_text], [lex.squeeze(0)])[0]

    _, adjusted, construct_matches, text_intent = adjust_ocean_by_keywords(
        raw,
        work_text,
        conf,
        use_fuzzy_phrases=not skip_semantic_embedding,
    )
    adjusted = adjust_ocean_by_semantic(
        adjusted, semantic_matches, work_text, conf, text_intent=text_intent
    )
    adjusted = apply_emotional_keyword_adjustment(work_text, adjusted, confidence_scale=conf)
    adjusted, modifier_analysis = apply_modifier_ocean_adjustment(
        work_text, adjusted, confidence_scale=conf
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

    personality_profile = generate_persona_profile(
        adjusted,
        work_text,
        sufficiency=sufficiency,
        construct_matches=construct_matches,
        text_intent=text_intent,
        evidence=evidence if not skip_heavy_evidence else None,
    )

    highlighted = highlight_keywords_in_text(text, evidence, semantic_matches)

    out: dict[str, Any] = {
        "input_text": text,
        "text_normalized": work_text,
        "prediction_adjusted": {k: adjusted[k] for k in OCEAN_TRAITS if k in adjusted},
        "dominant_trait": dominant,
        "text_intent": text_intent.primary,
        "dominant_keyword_category": dominant_keyword_category(text_intent, evidence),
        "personality_profile": personality_profile,
        "explanation": explanation,
        "suggestion": suggestion,
        "highlighted_text": highlighted,
        "crisis_level": crisis_level,
        "analysis_mode": sufficiency.analysis_mode,
        "confidence_scale": sufficiency.confidence_scale,
        "meaningful_token_count": sufficiency.meaningful_count,
        "disclaimer": sufficiency.disclaimer,
        "sentiment_modifiers": modifiers_to_dict(modifier_analysis),
    }

    if not skip_heavy_evidence:
        out["ontology_analysis"] = {
            "coverage_percent": coverage,
            "active_subtraits": subtraits,
            "semantic_subtraits": aggregate_semantic_subtraits(semantic_matches),
        }
        out["ontology_expansion_candidates"] = matches_to_expansion_candidates(
            semantic_matches, top_k=8
        )
        out["lexical_evidence"] = evidence

    if not skip_chart:
        try:
            out["ocean_chart_base64"] = generate_ocean_chart(adjusted)
        except Exception:
            out["ocean_chart_base64"] = None

    return out


def process_texts_batch(
    items: list[tuple[int, str]],
    *,
    fast: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_charts: bool = False,
    include_heavy_details: bool = False,
) -> tuple[list[dict], dict]:
    """
    items: [(row_index, original_text), ...]
    Kembalikan (hasil per baris, statistik waktu).
    """
    t0 = time.perf_counter()
    results: list[dict] = []
    skip_sem = fast
    skip_chart = not include_charts
    skip_heavy = fast and not include_heavy_details

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        work_texts: list[str] = []
        originals: list[str] = []
        indices: list[int] = []

        for idx, original in chunk:
            typo = normalize_text_typos(original)
            work_texts.append(typo.normalized)
            originals.append(original)
            indices.append(idx)

        embeddings = None
        if not skip_sem:
            embeddings = encode_texts_batch(work_texts)

        lexical_tensors: list[torch.Tensor] = []
        lexical_bundles = []
        for i, wt in enumerate(work_texts):
            vec = None
            if embeddings is not None:
                vec = embeddings[i]
            bundle = build_lexical_vector_with_analysis(
                wt,
                semantic_text_vec=vec,
                skip_semantic_embedding=skip_sem,
            )
            lexical_bundles.append(bundle)
            lex = bundle[0]
            lexical_tensors.append(lex.squeeze(0) if lex.dim() == 2 else lex)

        raw_scores = _model_raw_scores_batch(work_texts, lexical_tensors)

        for i, idx in enumerate(indices):
            pred = compute_ocean_prediction_lite(
                originals[i],
                precomputed_raw=raw_scores[i],
                precomputed_lexical=lexical_bundles[i],
                text_embedding=embeddings[i] if embeddings is not None else None,
                skip_semantic_embedding=skip_sem,
                skip_chart=skip_chart,
                skip_heavy_evidence=skip_heavy,
            )
            pred["row_index"] = idx
            results.append(pred)

    elapsed = time.perf_counter() - t0
    stats = {
        "rows": len(results),
        "seconds": round(elapsed, 2),
        "rows_per_second": round(len(results) / elapsed, 2) if elapsed > 0 else 0,
        "fast_mode": fast,
        "batch_size": batch_size,
    }
    return results, stats
