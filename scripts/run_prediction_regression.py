"""
Regresi prediksi (tanpa model GPU) — raw skor tetap, uji keyword+intent+persona.

  python scripts/run_prediction_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skor model contoh (mirip produksi)
RAW = {"O": 3.69, "C": 2.54, "E": 1.72, "A": 2.88, "N": 3.14}

CASES = [
    ("perasaan yang didengarkan terasa lebih ringan", "A", "Empatik"),
    ("saya cemas menghadapi hari esok", "N", "Cemas"),
    ("saya sangat cemas menghadapi hari esok", "N", "Cemas"),
    ("saya mudah menyesuaikan diri di lingkungan baru", "A", "Adaptif"),
    ("saya ingin bunuh diri", "N", "Krisis"),
    ("saya lelah hidup dan putus harapan", "N", None),
    ("suka ide baru dan kreatif berpikir", "O", "Visioner"),
    ("saya disiplin dan tepat waktu menyelesaikan tugas", "C", "Perfeksionis"),
    ("suka bergaul dan aktif bersosialisasi", "E", "Ekstrovert"),
    ("merasa bahagia dan optimis hari ini", "A", "Ceria"),
    ("butuh teman untuk curhat setiap hari", "E", "Afiliasi"),
    ("sulit percaya orang dan menghindari keramaian", "N", "Menghindar"),
    ("mudah marah dan kesal berat", "N", "Tempramental"),
    ("saya sangat mencintai pasangan dan ingin selalu dekat dengannya", "A", "Romantis"),
    ("merasa sayang dan hangat saat bersama kekasih", "A", "Romantis"),
    ("romantis dan penuh kasih sayang pada orang tersayang", "A", "Romantis"),
    ("cinta mati-matian pada pacar", "A", "Romantis"),
    (
        "Saya menikmati acara sosial seperti seminar, komunitas, dan pertemuan besar.",
        "E",
        "Ekstrovert",
    ),
]


def predict(text: str) -> dict:
    from sapa_api.keywords import (
        adjust_ocean_by_keywords,
        apply_emotional_keyword_adjustment,
        determine_dominant_trait,
    )
    from sapa_api.sentiment_modifiers import apply_modifier_ocean_adjustment
    from sapa_api.crisis import apply_crisis_adjustment
    from sapa_api.persona import generate_persona_profile
    from sapa_api.text_utils import (
        align_scores_to_intent_dominance,
        assess_text_sufficiency,
        clamp_ocean,
    )

    s = assess_text_sufficiency(text)
    _, adj, cm, intent = adjust_ocean_by_keywords(dict(RAW), text, s.confidence_scale)
    adj = apply_emotional_keyword_adjustment(text, adj, s.confidence_scale)
    adj, _ = apply_modifier_ocean_adjustment(adj, text, s.confidence_scale)
    adj, _ = apply_crisis_adjustment(adj, text)
    adj = align_scores_to_intent_dominance(adj, intent.primary, margin=0.12)
    adj = clamp_ocean(adj)
    dom = determine_dominant_trait(adj, text, cm, intent=intent)
    persona = generate_persona_profile(
        adj, text, s, cm, text_intent=intent, evidence={}
    )
    return {
        "dominant": dom,
        "intent": intent.primary,
        "persona": persona[0] if persona else "",
        "O": round(adj["O"], 2),
        "C": round(adj.get("C", 3), 2),
        "E": round(adj["E"], 2),
        "A": round(adj["A"], 2),
        "N": round(adj["N"], 2),
    }


def main():
    ok = 0
    fail = 0
    print("=== Prediction regression ===\n")
    for text, exp_dom, exp_persona_hint in CASES:
        r = predict(text)
        dom_ok = r["dominant"] == exp_dom
        persona_ok = True
        if exp_persona_hint:
            persona_ok = exp_persona_hint in r["persona"]
        status = "OK" if dom_ok and persona_ok else "FAIL"
        if status == "OK":
            ok += 1
        else:
            fail += 1
        print(f"[{status}] {text[:50]}")
        print(f"       dom={r['dominant']} (exp {exp_dom}) intent={r['intent']}")
        print(f"       O={r['O']} A={r['A']} N={r['N']}")
        print(f"       persona: {r['persona'][:70]}...")
        if not dom_ok or not persona_ok:
            print(f"       ^ expected dom={exp_dom} persona~{exp_persona_hint}")
        print()
    print(f"Passed {ok}/{len(CASES)} | Failed {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
