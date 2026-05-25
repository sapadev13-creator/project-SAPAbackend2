"""
Konstruk Big Five berbasis frasa — lebih representatif daripada kata generik.
"""

from dataclasses import dataclass

from sapa_api.fuzzy_match import fuzzy_phrase_match
from sapa_api.text_utils import tokenize

# (frasa, bobot adjustment per dimensi OCEAN)
CONSTRUCT_PHRASES: dict[str, list[tuple[str, dict[str, float]]]] = {
    "N": [
        ("mudah terganggu", {"N": 0.85, "E": -0.15}),
        ("terganggu dengan", {"N": 0.7}),
        ("cemas menghadapi", {"N": 0.75, "E": -0.1}),
        ("sangat cemas", {"N": 0.8}),
        ("cemas berat", {"N": 0.85}),
        ("merasa cemas", {"N": 0.7}),
        ("merasa stres", {"N": 0.75, "E": -0.1}),
        ("kepikiran", {"N": 0.55, "O": 0.15}),
        ("overthinking", {"N": 0.7}),
        ("khawatir", {"N": 0.55}),
        ("gelisah", {"N": 0.6}),
        ("panik", {"N": 0.75, "E": -0.2}),
        ("putus asa", {"N": 0.9, "E": -0.3, "A": -0.2}),
        ("tidak tenang", {"N": 0.65}),
        ("emosional tidak stabil", {"N": 0.8}),
    ],
    "O": [
        ("terbuka dengan ide", {"O": 0.7, "E": 0.15}),
        ("suka bereksperimen", {"O": 0.75}),
        ("imajinatif", {"O": 0.7}),
        ("kreatif", {"O": 0.55}),
        ("ide baru", {"O": 0.5}),
        ("penasaran", {"O": 0.55}),
        ("suka belajar hal baru", {"O": 0.65}),
    ],
    "C": [
        ("disiplin", {"C": 0.65, "N": -0.1}),
        ("terencana", {"C": 0.6}),
        ("bertanggung jawab", {"C": 0.65, "A": 0.15}),
        ("rajin", {"C": 0.55}),
        ("fokus menyelesaikan", {"C": 0.6}),
        ("tepat waktu", {"C": 0.55}),
    ],
    "E": [
        ("menikmati acara sosial", {"E": 0.72, "A": 0.12, "O": 0.05}),
        ("acara sosial", {"E": 0.65, "A": 0.1}),
        ("pertemuan besar", {"E": 0.6, "A": 0.08}),
        ("suka bergaul", {"E": 0.65, "A": 0.2}),
        ("percaya diri sosial", {"E": 0.7}),
        ("aktif bersosialisasi", {"E": 0.65}),
        ("suka bicara", {"E": 0.55}),
        ("energik", {"E": 0.5}),
    ],
    "A": [
        ("mudah menyesuaikan diri", {"A": 0.5, "E": 0.35, "N": -0.1}),
        ("perasaan yang didengarkan", {"A": 0.85, "N": -0.2, "O": -0.15}),
        ("perasaan yang divalidasi", {"A": 0.85, "N": -0.2, "O": -0.15}),
        ("perasaan yang diterima", {"A": 0.8, "N": -0.18, "O": -0.12}),
        ("terasa lebih ringan", {"A": 0.75, "N": -0.25, "O": -0.12}),
        ("merasa didengarkan", {"A": 0.8, "N": -0.2, "O": -0.1}),
        ("merasa divalidasi", {"A": 0.8, "N": -0.2, "O": -0.1}),
        ("ringan hati", {"A": 0.65, "N": -0.2, "E": 0.1}),
        ("empati", {"A": 0.6, "N": -0.1}),
        ("peduli orang", {"A": 0.55}),
        ("kooperatif", {"A": 0.5}),
        ("suka membantu", {"A": 0.5, "E": 0.15}),
        ("mencintai pasangan", {"A": 0.75, "E": 0.18, "O": -0.15, "N": -0.1}),
        ("mencintai kekasih", {"A": 0.72, "E": 0.16, "O": -0.14, "N": -0.1}),
        ("bersama kekasih", {"A": 0.68, "E": 0.14, "O": -0.12, "N": -0.08}),
        ("kasih sayang", {"A": 0.65, "E": 0.12, "O": -0.12, "N": -0.08}),
        ("penuh kasih sayang", {"A": 0.7, "E": 0.14, "O": -0.14, "N": -0.1}),
        ("merasa sayang", {"A": 0.62, "E": 0.1, "O": -0.1, "N": -0.06}),
        ("orang tersayang", {"A": 0.58, "E": 0.1, "O": -0.1}),
        ("hubungan dekat", {"A": 0.6, "E": 0.1, "O": -0.1}),
        ("quality time dengan pasangan", {"A": 0.55, "E": 0.14, "O": -0.1}),
        ("romantis dan", {"A": 0.6, "O": -0.15, "E": 0.08}),
    ],
}

# Kata tunggal yang BOLEH dipakai jika benar-benar indikator trait (bukan generik)
TRAIT_SINGLE_WORDS: dict[str, frozenset[str]] = {
    "N": frozenset({
        "cemas", "stres", "stress", "gelisah", "khawatir", "panik", "takut",
        "sedih", "depresi", "marah", "kecewa", "frustasi", "overthinking",
    }),
    "O": frozenset({"kreatif", "imajinatif", "penasaran", "eksploratif", "inovatif"}),
    "C": frozenset({"disiplin", "rajin", "teliti", "teratur", "sistematis"}),
    "E": frozenset({"ekstrovert", "sosial", "komunikatif", "percaya", "ramah"}),
    "A": frozenset({"empati", "kooperatif", "sabar", "harmonis", "peduli"}),
}


@dataclass
class ConstructMatch:
    phrase: str
    ocean: str
    weights: dict[str, float]


def match_trait_constructs(text: str) -> list[ConstructMatch]:
    t = text.lower()
    matches: list[ConstructMatch] = []

    for ocean, phrases in CONSTRUCT_PHRASES.items():
        for phrase, weights in phrases:
            if phrase in t or fuzzy_phrase_match(t, phrase):
                matches.append(ConstructMatch(phrase, ocean, weights))

    tokens = set(tokenize(t))
    for ocean, words in TRAIT_SINGLE_WORDS.items():
        for w in words:
            if w in tokens:
                matches.append(
                    ConstructMatch(w, ocean, {ocean: 0.45})
                )

    return matches


def apply_construct_adjustments(scores: dict, text: str, scale: float = 1.0) -> tuple[dict, list[ConstructMatch]]:
    adjusted = scores.copy()
    matches = match_trait_constructs(text)

    for m in matches:
        for dim, w in m.weights.items():
            adjusted[dim] = adjusted.get(dim, 3.0) + w * scale

    return adjusted, matches


def dominant_from_constructs(
    scores: dict, matches: list[ConstructMatch]
) -> str | None:
    if not matches:
        return None

    dim_scores: dict[str, float] = {}
    for m in matches:
        dim_scores[m.ocean] = dim_scores.get(m.ocean, 0) + sum(m.weights.values())

    if not dim_scores:
        return None
    return max(dim_scores, key=dim_scores.get)


def construct_evidence(matches: list[ConstructMatch]) -> list[dict]:
    return [
        {
            "phrase": m.phrase,
            "ocean_dim": m.ocean,
            "weights": m.weights,
            "match_type": "construct",
        }
        for m in matches
    ]
