import re
from collections import defaultdict

import torch

from sapa_api import state
from sapa_api.text_utils import is_meaningful_token
from sapa_api.semantic import (
    apply_semantic_to_lexical,
    compute_semantic_matches,
    matches_to_expansion_candidates,
    semantic_coverage_percent,
)


def build_lexical_vector_with_analysis(
    text: str,
    *,
    semantic_text_vec=None,
    skip_semantic_embedding: bool = False,
):
    vec = torch.zeros(state.LEXICAL_SIZE)
    tokens = re.findall(r"\w+", text.lower())
    token_set = set(tokens)

    matched_tokens = set()
    subtrait_scores = defaultdict(float)
    evidence = defaultdict(list)

    # Fast path: use inverted index token -> pattern_ids (built at startup)
    patterns = getattr(state, "LEXICON_PATTERNS", None)
    inv = getattr(state, "TOKEN_TO_PATTERN_IDS", None)
    if patterns and inv:
        candidate_ids = set()
        for t in token_set:
            ids = inv.get(t)
            if ids:
                candidate_ids.update(ids)

        for pid in candidate_ids:
            p = patterns[pid]
            overlap = {t for t in (p["tokens"] & token_set) if is_meaningful_token(t)}
            if not overlap:
                continue
            ratio = len(overlap) / len(p["tokens"])
            if ratio == 1.0:
                score = 2.0 * p["strength"]
            elif ratio >= 0.5:
                score = 0.5 * p["strength"]
            else:
                continue
            subtrait = p["sub_trait"]
            sid = state.subtrait2id[subtrait]
            vec[sid] += score
            subtrait_scores[subtrait] += score
            matched_tokens |= overlap
            evidence[subtrait].append({
                "lexeme": p["lexeme"],
                "matched_tokens": list(overlap),
                "score": round(score, 3),
                "match_type": "exact",
            })
    else:
        # Fallback: older loop
        for subtrait, pats in state.LEXICON.items():
            sid = state.subtrait2id[subtrait]
            for p in pats:
                overlap = {t for t in (p["tokens"] & token_set) if is_meaningful_token(t)}
                if not overlap:
                    continue

                ratio = len(overlap) / len(p["tokens"])
                if ratio == 1.0:
                    score = 2.0 * p["strength"]
                elif ratio >= 0.5:
                    score = 0.5 * p["strength"]
                else:
                    continue

                vec[sid] += score
                subtrait_scores[subtrait] += score
                matched_tokens |= overlap
                evidence[subtrait].append({
                    "lexeme": p["lexeme"],
                    "matched_tokens": list(overlap),
                    "score": round(score, 3),
                    "match_type": "exact",
                })

    semantic_matches = compute_semantic_matches(
        text,
        text_vec=semantic_text_vec,
        skip_embedding=skip_semantic_embedding,
    )
    semantic_tokens = apply_semantic_to_lexical(
        vec, semantic_matches, evidence, subtrait_scores
    )
    matched_tokens |= semantic_tokens

    vec = torch.log1p(vec)
    if vec.sum() > 0:
        vec = vec / vec.sum()

    exact_coverage = len(matched_tokens) / max(len(token_set), 1) * 100
    semantic_cov = semantic_coverage_percent(semantic_matches, len(token_set))
    coverage = round(min(100.0, exact_coverage + semantic_cov * 0.35), 2)

    return (
        vec.unsqueeze(0),
        coverage,
        dict(sorted(subtrait_scores.items(), key=lambda x: -x[1])),
        dict(evidence),
        semantic_matches,
    )


def expand_ontology_candidates(text: str, top_k=5, threshold=0.7):
    """Backward-compatible; memakai pipeline semantic terpadu."""
    matches = compute_semantic_matches(text, top_k=top_k * 2, threshold=min(threshold, 0.42))
    filtered = [m for m in matches if m["similarity"] >= threshold]
    if filtered:
        return matches_to_expansion_candidates(filtered, top_k=top_k)
    return matches_to_expansion_candidates(matches, top_k=top_k)
