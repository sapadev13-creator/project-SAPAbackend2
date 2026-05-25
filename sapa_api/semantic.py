"""Semantic ontology: embedding similarity + phrase matching (ketat)."""

import re
from collections import defaultdict

import numpy as np
import torch

from sapa_api import state
from sapa_api.config import (
    DEVICE,
    MAX_LEN,
    SEMANTIC_LEXICAL_WEIGHT,
    SEMANTIC_OCEAN_WEIGHT,
    SEMANTIC_PHRASE_BOOST,
    SEMANTIC_THRESHOLD,
    SEMANTIC_TOP_K,
    SUBTRAIT_OCEAN_PREFIX,
)
from sapa_api.text_utils import (
    has_crisis_language,
    has_distress_language,
    has_empathy_validation_context,
    has_positive_context,
    has_relationship_affection_context,
    is_meaningful_token,
    limit_ocean_delta,
    ocean_only,
    tokenize,
)

NEGATIVE_LEXEME_PARTS = frozenset({
    "bunuh", "mati", "sedih", "marah", "cemas", "takut", "depresi", "putus",
    "asa", "menyerah", "kecewa", "benci", "panik", "gila", "stress", "stres",
})


def subtrait_to_ocean(sub_trait: str) -> str | None:
    for prefix, dim in SUBTRAIT_OCEAN_PREFIX.items():
        if sub_trait.startswith(prefix):
            return dim
    return None


def _is_negative_lexeme(lexeme: str) -> bool:
    parts = lexeme.lower().split("_")
    return any(p in NEGATIVE_LEXEME_PARTS for p in parts)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)


def prepare_embedding_index():
    if state.ONT_EMBEDDINGS is None:
        return
    emb = np.asarray(state.ONT_EMBEDDINGS, dtype=np.float32)
    state.ONT_EMB_NORM = _normalize_rows(emb)


def encode_text(text: str) -> np.ndarray | None:
    batch = encode_texts_batch([text])
    if batch is None:
        return None
    return batch[0]


def encode_texts_batch(texts: list[str]) -> np.ndarray | None:
    """Embedding CLS untuk banyak teks sekaligus (lebih cepat untuk Excel batch)."""
    if not texts or state.tokenizer is None or state.model is None:
        return None
    enc = state.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = state.model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        vecs = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def _phrase_ontology_hits(tokens: list[str], token_set: set[str]) -> dict[int, float]:
    """Hanya frasa utuh di teks — tidak pakai subset token longgar."""
    hits: dict[int, float] = {}
    if not state.ONT_META:
        return hits

    text_underscore = "_".join(tokens)
    for i, meta in enumerate(state.ONT_META):
        lexeme = str(meta["lexeme"])
        parts = lexeme.split("_")
        if len(parts) >= 2 and lexeme in text_underscore:
            hits[i] = max(hits.get(i, 0), SEMANTIC_PHRASE_BOOST)
        elif len(parts) == 1 and is_meaningful_token(parts[0]) and parts[0] in token_set:
            if len(parts[0]) >= 6:
                hits[i] = max(hits.get(i, 0), 0.72)
    return hits


def compute_semantic_matches(
    text: str,
    top_k: int = SEMANTIC_TOP_K,
    threshold: float = SEMANTIC_THRESHOLD,
    text_vec: np.ndarray | None = None,
    *,
    skip_embedding: bool = False,
) -> list[dict]:
    if state.ONT_EMB_NORM is None or state.ONT_META is None:
        return []

    text_lower = text.lower()
    tokens = tokenize(text)
    token_set = set(tokens)
    positive_ctx = has_positive_context(text_lower)
    validation_ctx = has_empathy_validation_context(text_lower)
    combined: dict[int, float] = _phrase_ontology_hits(tokens, token_set)

    embed_threshold = 0.52 if not combined else threshold
    if not skip_embedding and text_vec is None:
        text_vec = encode_text(text)
    if text_vec is not None:
        if text_vec.ndim == 2:
            text_vec = text_vec[0]
        sims = state.ONT_EMB_NORM @ text_vec
        for idx in np.where(sims >= embed_threshold)[0]:
            idx = int(idx)
            meta = state.ONT_META[idx]
            if validation_ctx and meta["sub_trait"].startswith("o_"):
                continue
            if positive_ctx and meta["sub_trait"].startswith("n_") and sims[idx] < 0.62:
                continue
            combined[idx] = max(combined.get(idx, 0), float(sims[idx]))

    if not combined:
        return []

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    matches = []
    for idx, sim in ranked:
        meta = state.ONT_META[idx]
        lexeme = meta["lexeme"]
        sub_trait = meta["sub_trait"]
        if has_empathy_validation_context(text_lower) and sub_trait.startswith("o_"):
            continue
        if positive_ctx and sub_trait.startswith("n_") and not _is_negative_lexeme(lexeme):
            continue
        matches.append({
            "lexeme": lexeme,
            "sub_trait": sub_trait,
            "ocean_dim": subtrait_to_ocean(sub_trait),
            "similarity": round(sim, 3),
            "match_type": "phrase" if sim >= SEMANTIC_PHRASE_BOOST - 0.05 else "embedding",
            "lexeme_tokens": [t for t in lexeme.split("_") if is_meaningful_token(t)],
        })
    return matches


def apply_semantic_to_lexical(
    vec: torch.Tensor,
    semantic_matches: list[dict],
    evidence: dict,
    subtrait_scores: dict,
) -> set[str]:
    matched_tokens: set[str] = set()
    for m in semantic_matches:
        sub_trait = m["sub_trait"]
        if sub_trait not in state.subtrait2id:
            continue
        sid = state.subtrait2id[sub_trait]
        score = m["similarity"] * SEMANTIC_LEXICAL_WEIGHT * 0.65
        vec[sid] += score
        subtrait_scores[sub_trait] = subtrait_scores.get(sub_trait, 0) + score
        matched_tokens.update(m["lexeme_tokens"])
        evidence.setdefault("semantic_ontology", []).append({
            "lexeme": m["lexeme"],
            "matched_tokens": m["lexeme_tokens"],
            "score": round(score, 3),
            "similarity": m["similarity"],
            "match_type": m["match_type"],
            "sub_trait": sub_trait,
        })
    return matched_tokens


def adjust_ocean_by_semantic(
    scores: dict,
    semantic_matches: list[dict],
    text: str = "",
    confidence_scale: float = 1.0,
    text_intent=None,
) -> dict:
    adjusted = scores.copy()
    ocean_boost: dict[str, float] = defaultdict(float)
    text_lower = text.lower()
    positive_ctx = has_positive_context(text_lower)
    validation_ctx = has_empathy_validation_context(text_lower)
    romantic_ctx = has_relationship_affection_context(text_lower)
    intent_primary = getattr(text_intent, "primary", None) if text_intent else None

    for m in semantic_matches:
        ocean = m.get("ocean_dim")
        if not ocean:
            continue
        sim = m["similarity"]
        sub_trait = m.get("sub_trait", "")
        if (validation_ctx or romantic_ctx) and (sub_trait.startswith("o_") or ocean == "O"):
            continue
        if romantic_ctx and ocean == "A":
            sim *= 1.35
        if intent_primary in ("empathy_validation", "adaptive", "anxiety", "sad", "anger", "crisis"):
            if intent_primary != "creative" and ocean == "O" and sub_trait.startswith("o_"):
                continue
            if intent_primary in ("empathy_validation", "adaptive", "relationship_affection") and ocean == "A":
                sim *= 1.3
            if intent_primary == "relationship_affection" and ocean == "O":
                continue
            if intent_primary in ("anxiety", "sad", "anger", "crisis") and ocean == "N":
                sim *= 1.15
        if validation_ctx and ocean == "A":
            sim *= 1.25
        ocean_boost[ocean] += sim * SEMANTIC_OCEAN_WEIGHT * confidence_scale

        if (
            ocean == "N"
            and _is_negative_lexeme(m["lexeme"])
            and "bunuh" in m["lexeme"]
            and sim >= 0.68
            and has_crisis_language(text_lower)
        ):
            adjusted["EXTREME_ALERT"] = adjusted.get("EXTREME_ALERT", 0) + sim * 0.35

    for ocean, boost in ocean_boost.items():
        adjusted[ocean] = adjusted.get(ocean, 3.0) + min(boost, 0.45 * confidence_scale)

    if positive_ctx or not has_crisis_language(text_lower):
        if not has_distress_language(text_lower) or not has_crisis_language(text_lower):
            adjusted["EXTREME_ALERT"] = min(adjusted.get("EXTREME_ALERT", 0), 1.0)

    adjusted = limit_ocean_delta(scores, adjusted, max_delta=0.9)
    for k in ["O", "C", "E", "A", "N"]:
        if k in adjusted:
            adjusted[k] = max(1.0, min(5.0, adjusted[k]))
    return adjusted


def semantic_coverage_percent(semantic_matches: list[dict], token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    covered = set()
    for m in semantic_matches:
        covered.update(m["lexeme_tokens"])
    return round(len(covered) / token_count * 100, 2)


def matches_to_expansion_candidates(semantic_matches: list[dict], top_k: int = 5) -> list[dict]:
    return [
        {
            "candidate_from": m["lexeme"],
            "suggested_subtrait": m["sub_trait"],
            "similarity": m["similarity"],
            "ocean_dim": m.get("ocean_dim"),
            "match_type": m.get("match_type", "embedding"),
        }
        for m in semantic_matches[:top_k]
    ]


def aggregate_semantic_subtraits(semantic_matches: list[dict]) -> dict:
    agg = defaultdict(float)
    for m in semantic_matches:
        agg[m["sub_trait"]] += m["similarity"]
    return dict(sorted(agg.items(), key=lambda x: -x[1]))
