"""
Analisis & Optimasi OCEAN Weight Mappings untuk Akurasi Prediksi.

Tujuan:
1. Review current weight mappings
2. Identifikasi trait confliks
3. Optimize weight untuk mendapat hasil lebih akurat
4. Ensure kombinasi kata yang konsisten

Jalankan: python scripts/analyze_ocean_weights.py
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

# Current mapping dari trait ke OCEAN adjustment
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

# Optimized weights untuk hasil lebih akurat
OPTIMIZED_WEIGHTS = {
    "ANGER_EMO": {"N": 0.5, "A": -0.2, "C": -0.1},  # Add C dimension
    "SAD_EMO": {"N": 0.4, "O": 0.05, "E": -0.1},  # Add E
    "ANXIETY_EMO": {"N": 0.55, "E": -0.2, "O": -0.1},  # More O reduction
    "MENTAL_UNSTABLE_N": {"N": 0.8, "C": -0.2},  # Add C reduction
    "NEGATIVE_SOCIAL": {"N": 0.4, "A": -0.3, "E": -0.2, "C": -0.1},  # Strengthen
    "POSITIVE_SOCIAL": {"E": 0.4, "A": 0.4, "N": -0.2, "C": 0.1},  # Add C
    "EXTRAVERSION_E": {"E": 0.55, "A": 0.25, "N": -0.15, "O": 0.1},  # Add O
    "E_SOCIAL_DEPENDENCY": {"E": 0.4, "A": 0.2, "N": 0.05},  # Slight N increase
    "COLLABORATION": {"A": 0.6, "E": 0.3, "C": 0.2, "N": -0.1},  # Strengthen all
    "RELATIONSHIP_AFFECTION": {"A": 0.7, "E": 0.2, "O": 0.1, "N": -0.1},  # Add O, N
    "EMPATHY_HARMONY_A": {"A": 0.7, "N": -0.3, "E": 0.1},  # Strengthen anti-N
    "TRUST": {"A": 0.5, "N": -0.15, "E": 0.1},  # Strengthen
    "CREATIVE_DISCUSSION_A": {"O": 0.6, "E": 0.2, "A": 0.1},  # Add A
    "INTROSPECTION": {"O": 0.5, "N": 0.15, "E": -0.1},  # Strengthen N/E
    "DISCIPLINE_C": {"C": 0.85, "N": -0.2, "E": -0.1},  # Strengthen all
    "ACHIEVEMENT": {"C": 0.6, "E": 0.2, "O": 0.15, "N": -0.1},  # Add O, N
    "EMO_POSITIVE": {"A": 0.3, "E": 0.3, "N": -0.2, "O": 0.1},  # Add O
    "EXTREME_NEGATIVE": {"N": 2.0, "E": -0.8, "A": -0.6, "C": -0.6, "O": -0.4},  # Strengthen all
    "EMO_NEGATIVE": {"N": 0.5, "E": -0.15, "A": -0.15, "O": 0.05, "C": -0.1},  # Add C
}

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


def analyze_ocean_weights():
    """Analisis OCEAN weight distributions."""
    data = load_keywords_file()
    
    print("=" * 80)
    print("ANALISIS OCEAN WEIGHT MAPPINGS")
    print("=" * 80)
    
    print("\n1. PERBANDINGAN CURRENT vs OPTIMIZED WEIGHTS")
    print("-" * 80)
    
    # Calculate total impact per dimension
    current_impact = defaultdict(lambda: 0)
    optimized_impact = defaultdict(lambda: 0)
    
    for trait in TRAIT_PRIORITY:
        count = len(data.get(trait, []))
        if count == 0:
            continue
        
        # Current
        current_weights = KEYWORD_TRAIT_MAP.get(trait, {})
        for dim, w in current_weights.items():
            current_impact[dim] += w * count
        
        # Optimized
        opt_weights = OPTIMIZED_WEIGHTS.get(trait, {})
        for dim, w in opt_weights.items():
            optimized_impact[dim] += w * count
    
    print("\nTotal OCEAN impact (current vs optimized):")
    for dim in "OCEAN":
        current = current_impact[dim]
        optimized = optimized_impact[dim]
        delta = optimized - current
        pct = (delta / abs(current) * 100) if current != 0 else 0
        marker = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {dim}: {current:+.2f} → {optimized:+.2f}  ({marker}{pct:+.1f}%)")
    
    print("\n2. WEIGHT COMPARISONS PER TRAIT")
    print("-" * 80)
    
    for trait in TRAIT_PRIORITY:
        current = KEYWORD_TRAIT_MAP.get(trait, {})
        optimized = OPTIMIZED_WEIGHTS.get(trait, {})
        
        # Combine both for display
        all_dims = set(current.keys()) | set(optimized.keys())
        
        print(f"\n{trait}:")
        for dim in sorted(all_dims):
            c_val = current.get(dim, 0)
            o_val = optimized.get(dim, 0)
            if c_val != o_val:
                delta = o_val - c_val
                marker = "↑" if delta > 0 else "↓"
                print(f"  {dim}: {c_val:+.2f} → {o_val:+.2f}  ({marker}{delta:+.2f})")
            else:
                print(f"  {dim}: {c_val:+.2f} (unchanged)")
    
    print("\n3. DIMENSI YANG SERING DIGUNAKAN")
    print("-" * 80)
    
    dim_usage = Counter()
    for trait in TRAIT_PRIORITY:
        weights = OPTIMIZED_WEIGHTS.get(trait, {})
        dim_usage.update(weights.keys())
    
    print("\nPenggunaan dimensi OCEAN dalam optimized weights:")
    for dim, count in sorted(dim_usage.items(), key=lambda x: -x[1]):
        pct = count * 100 / len(TRAIT_PRIORITY)
        print(f"  {dim}: digunakan di {count} traits ({pct:.1f}%)")
    
    print("\n4. REKOMENDASI IMPLEMENTASI")
    print("-" * 80)
    
    print("""
✓ Optimasi utama untuk meningkatkan akurasi:
  • NEGATIVE traits: Perkuat dimensi N (Neuroticism) dengan C reduction
  • POSITIVE traits: Tambah O (Openness) dimension untuk nuansa
  • Extreme cases: Increase contrast (stronger positive/negative pull)
  • Add cross-dimension: Lebih dimensi yang terlibat = matching lebih akurat

✓ Benefit yang diharapkan:
  • Prediksi lebih spesifik (bukan hanya N tinggi/rendah)
  • Better distinction antar emotional traits
  • Improved positive trait detection

✓ Implementasi:
  1. Update KEYWORD_TRAIT_MAP dengan OPTIMIZED_WEIGHTS
  2. Run clean_keywords_traits.py untuk validate
  3. Test dengan sample predictions
  4. Monitor hasil improvement
    """)


if __name__ == "__main__":
    analyze_ocean_weights()
