"""
Analisis Kualitas Prediksi & Dataset Keywords.

Mengidentifikasi masalah:
1. Keyword duplikat antar kategori
2. Keyword dengan kombinasi kata yang lemah
3. Ketidakseimbangan distribusi per trait
4. Konflikt mapping antar kategori
5. Fuzzy match issues

Jalankan: python scripts/analyze_prediction_quality.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "keywords_traits.xlsx"

KW_COL = "Keyword / Phrase"
TR_COL = "Trait / Kategori"

# Mapping dari trait ke OCEAN adjustment
KEYWORD_TRAIT_MAP = {
    "ANGER_EMO": {"N": 0.45, "A": -0.15},
    "SAD_EMO": {"N": 0.35, "O": 0.05},
    "ANXIETY_EMO": {"N": 0.5, "E": -0.15},
    "MENTAL_UNSTABLE_N": {"N": 0.7},
    "NEGATIVE_SOCIAL": {"N": 0.35, "A": -0.25, "E": -0.15},
    "POSITIVE_SOCIAL": {"E": 0.35, "A": 0.35, "N": -0.15},
    "EXTRAVERSION_E": {"E": 0.5, "A": 0.2, "N": -0.1},
    "E_SOCIAL_DEPENDENCY": {"E": 0.35, "A": 0.15},
    "COLLABORATION": {"A": 0.5, "E": 0.25, "C": 0.15},
    "RELATIONSHIP_AFFECTION": {"A": 0.6, "E": 0.15},
    "EMPATHY_HARMONY_A": {"A": 0.65, "N": -0.25},
    "TRUST": {"A": 0.4, "N": -0.08},
    "CREATIVE_DISCUSSION_A": {"O": 0.5, "E": 0.15},
    "INTROSPECTION": {"O": 0.4, "N": 0.08},
    "DISCIPLINE_C": {"C": 0.75, "N": -0.15},
    "ACHIEVEMENT": {"C": 0.5, "E": 0.15},
    "EMO_POSITIVE": {"A": 0.25, "E": 0.25, "N": -0.15},
    "EXTREME_NEGATIVE": {"N": 1.8, "E": -0.6, "A": -0.5, "C": -0.5},
    "EMO_NEGATIVE": {"N": 0.42, "E": -0.12, "A": -0.1, "O": 0.05},
}

# Prioritas saat keyword duplikat
TRAIT_PRIORITY = (
    "EXTREME_NEGATIVE",
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


def load_keywords_file():
    """Load keywords dari file Excel."""
    df = pd.read_excel(XLSX)
    data = defaultdict(list)
    for _, row in df.iterrows():
        trait = str(row[TR_COL]).strip()
        keyword = str(row[KW_COL]).strip().lower()
        if keyword and trait:
            data[trait].append(keyword)
    return data


def analyze_keywords():
    """Analisis lengkap dataset keywords."""
    data = load_keywords_file()
    
    print("=" * 80)
    print("ANALISIS KUALITAS KEYWORDS DATASET")
    print("=" * 80)
    
    # 1. Distribusi per trait
    print("\n1. DISTRIBUSI KEYWORD PER TRAIT")
    print("-" * 80)
    trait_counts = {t: len(kws) for t, kws in data.items()}
    avg_count = np.mean(list(trait_counts.values()))
    
    for trait in TRAIT_PRIORITY:
        count = trait_counts.get(trait, 0)
        status = "✓" if count >= avg_count * 0.7 else "⚠" if count > 0 else "✗"
        target = 180 if trait != "EXTREME_NEGATIVE" else 42
        pct = (count / target * 100) if target else 0
        print(f"  {status} {trait:30s} : {count:3d} kata  ({pct:5.1f}% dari {target})")
    
    total_keywords = sum(trait_counts.values())
    print(f"\n  Total keywords: {total_keywords}")
    
    # 2. Keyword duplikat antar kategori
    print("\n2. KEYWORD DUPLIKAT ANTAR KATEGORI")
    print("-" * 80)
    all_keywords = {}
    for trait, keywords in data.items():
        for kw in keywords:
            if kw not in all_keywords:
                all_keywords[kw] = []
            all_keywords[kw].append(trait)
    
    duplicates = {kw: traits for kw, traits in all_keywords.items() if len(traits) > 1}
    conflict_count = 0
    if duplicates:
        print(f"  Ditemukan {len(duplicates)} keyword duplikat:")
        for kw in sorted(duplicates.keys())[:20]:  # Tampilkan 20 pertama
            traits = duplicates[kw]
            priority_trait = next((t for t in TRAIT_PRIORITY if t in traits), traits[0])
            status = "✓" if priority_trait == traits[0] else "⚠"
            print(f"    {status} '{kw}':")
            for t in traits:
                marker = " (prioritas)" if t == priority_trait else ""
                print(f"         - {t}{marker}")
            conflict_count += 1
    else:
        print("  ✓ Tidak ada keyword duplikat")
    
    # 3. Analisis kualitas keyword individual
    print("\n3. ANALISIS KUALITAS KEYWORD")
    print("-" * 80)
    
    # Kata tunggal vs frasa
    single_words = 0
    multi_word_phrases = 0
    for keywords in data.values():
        for kw in keywords:
            if " " in kw:
                multi_word_phrases += 1
            else:
                single_words += 1
    
    print(f"  Kata tunggal: {single_words} ({single_words*100/(single_words+multi_word_phrases):.1f}%)")
    print(f"  Frasa multi-kata: {multi_word_phrases} ({multi_word_phrases*100/(single_words+multi_word_phrases):.1f}%)")
    
    # Panjang keyword
    all_kws = [kw for kws in data.values() for kw in kws]
    lengths = [len(kw) for kw in all_kws]
    print(f"\n  Panjang rata-rata: {np.mean(lengths):.1f} karakter")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    
    very_short = [kw for kw in all_kws if len(kw) < 3]
    very_long = [kw for kw in all_kws if len(kw) > 60]
    if very_short:
        print(f"  ⚠ Keyword sangat pendek (<3 char): {len(very_short)} kata")
    if very_long:
        print(f"  ⚠ Keyword sangat panjang (>60 char): {len(very_long)} kata")
        for kw in very_long[:5]:
            print(f"      - '{kw}'")
    
    # 4. OCEAN weight distribution
    print("\n4. DISTRIBUSI OCEAN WEIGHT PER TRAIT")
    print("-" * 80)
    
    trait_impact = defaultdict(lambda: {"N": 0, "E": 0, "O": 0, "C": 0, "A": 0})
    for trait, keywords in data.items():
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        weights = KEYWORD_TRAIT_MAP[trait]
        for dim, weight in weights.items():
            trait_impact[trait][dim] = weight
    
    for trait in TRAIT_PRIORITY:
        if trait not in KEYWORD_TRAIT_MAP:
            continue
        impact = trait_impact[trait]
        dims_str = " | ".join(f"{d}:{v:+.2f}" for d, v in sorted(impact.items()) if v != 0)
        print(f"  {trait:30s}: {dims_str}")
    
    # 5. Rekomendasi perbaikan
    print("\n5. REKOMENDASI PERBAIKAN PRIORITAS")
    print("-" * 80)
    
    recommendations = []
    
    # Cek trait dengan keyword kurang
    for trait in TRAIT_PRIORITY:
        count = trait_counts.get(trait, 0)
        target = 180 if trait != "EXTREME_NEGATIVE" else 42
        if count < target * 0.5:
            recommendations.append(f"  • {trait} ({count}/{target}): TAMBAH minimal {target - count} keyword")
    
    # Cek trait dengan banyak duplikat
    trait_conflicts = defaultdict(int)
    for traits in duplicates.values():
        for t in traits:
            trait_conflicts[t] += 1
    
    for trait, count in sorted(trait_conflicts.items(), key=lambda x: -x[1])[:5]:
        recommendations.append(f"  • {trait}: AUDIT {count} keyword duplikat")
    
    # Cek keyword quality
    if very_short:
        recommendations.append(f"  • HAPUS {len(very_short)} keyword sangat pendek (<3 char)")
    
    if very_long:
        recommendations.append(f"  • REVIEW {len(very_long)} keyword sangat panjang (>60 char)")
    
    # Check for weak combinations
    weak_traits = [t for t in TRAIT_PRIORITY if trait_counts.get(t, 0) < avg_count * 0.6]
    if weak_traits:
        recommendations.append(f"  • Traits dengan sedikit keyword: {', '.join(weak_traits[:3])}")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("  ✓ Dataset sudah dalam kondisi baik")
    
    print("\n" + "=" * 80)
    return data, duplicates, trait_counts


def analyze_phrase_combinations(data):
    """Analisis kombinasi kata dalam frasa."""
    print("\n6. ANALISIS KOMBINASI KATA DALAM FRASA")
    print("-" * 80)
    
    phrase_analysis = defaultdict(list)
    for trait, keywords in data.items():
        for kw in keywords:
            if " " in kw:
                words = kw.split()
                phrase_analysis[len(words)].append((kw, trait))
    
    print(f"  Distribusi panjang frasa:")
    for length in sorted(phrase_analysis.keys()):
        phrases = phrase_analysis[length]
        print(f"    {length} kata: {len(phrases)} frasa")
        if length >= 7:
            print(f"      ⚠ Terlalu panjang - contoh: {phrases[0][0]}")
    
    print(f"\n  Rekomendasi:")
    print(f"    • Ideal: frasa 2-4 kata (untuk matching yang akurat)")
    print(f"    • Hindari: frasa >5 kata atau narasi panjang")
    too_long = [p for ps in phrase_analysis.values() for p, t in ps if len(p) > 55]
    if too_long:
        print(f"    • Ditemukan {len(too_long)} frasa >55 karakter - pertimbangkan shortening")


if __name__ == "__main__":
    data, duplicates, counts = analyze_keywords()
    analyze_phrase_combinations(data)
