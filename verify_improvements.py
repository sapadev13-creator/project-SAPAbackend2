"""
VERIFICATION SCRIPT - Memastikan semua perubahan sudah di-apply dengan benar.

Jalankan: python verify_improvements.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify_keyword_updates():
    """Verify bahwa KEYWORD_TRAIT_MAP sudah diupdate."""
    print("=" * 90)
    print("VERIFIKASI IMPROVEMENT OPTIMIZATION")
    print("=" * 90)
    
    # Import dari file yang sudah diupdate
    sys.path.insert(0, str(ROOT))
    from sapa_api.keywords import KEYWORD_TRAIT_MAP
    
    print("\n✓ File berhasil diimport")
    
    # Expected values setelah optimization
    expected_updates = {
        "ANGER_EMO": {
            "expected": {"N": 0.5, "A": -0.2, "C": -0.1},
            "description": "Tambah C dimension, perkuat N dan A"
        },
        "SAD_EMO": {
            "expected": {"N": 0.4, "O": 0.05, "E": -0.1},
            "description": "Tambah E dimension"
        },
        "ANXIETY_EMO": {
            "expected": {"N": 0.55, "E": -0.2, "O": -0.1},
            "description": "Perkuat N, E, O"
        },
        "POSITIVE_SOCIAL": {
            "expected": {"E": 0.4, "A": 0.4, "N": -0.2, "C": 0.1},
            "description": "Tambah C dan N dimensions"
        },
        "COLLABORATION": {
            "expected": {"A": 0.6, "E": 0.3, "C": 0.2, "N": -0.1},
            "description": "Perkuat semua, tambah N"
        },
        "CREATIVE_DISCUSSION_A": {
            "expected": {"O": 0.6, "E": 0.2, "A": 0.1},
            "description": "Perkuat O, tambah A"
        },
        "DISCIPLINE_C": {
            "expected": {"C": 0.85, "N": -0.2, "E": -0.1},
            "description": "Perkuat C, tambah E reduction"
        },
        "EXTREME_NEGATIVE": {
            "expected": {"N": 2.0, "E": -0.8, "A": -0.6, "C": -0.6, "O": -0.4},
            "description": "Perkuat N, E, A, C, tambah O"
        },
    }
    
    print("\n1. VALIDASI PERUBAHAN KEYWORDS")
    print("-" * 90)
    
    all_ok = True
    for trait, info in expected_updates.items():
        actual = KEYWORD_TRAIT_MAP.get(trait, {})
        expected = info["expected"]
        
        status = "✓" if actual == expected else "✗"
        print(f"\n{status} {trait}")
        print(f"   Deskripsi: {info['description']}")
        
        if actual == expected:
            print(f"   Status: ✅ CORRECT - {actual}")
        else:
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            all_ok = False
    
    print("\n" + "-" * 90)
    
    if all_ok:
        print("\n✅ SEMUA PERUBAHAN BERHASIL DI-APPLY!")
    else:
        print("\n⚠️ ADA YANG BELUM SESUAI - Cek perubahan di keywords.py")
    
    # Count traits with multi-dimension
    print("\n2. STATISTIK OPTIMIZATION")
    print("-" * 90)
    
    multi_dim_traits = sum(1 for w in KEYWORD_TRAIT_MAP.values() if len(w) >= 3)
    total_traits = len(KEYWORD_TRAIT_MAP)
    
    print(f"\nTotal traits: {total_traits}")
    print(f"Traits dengan 3+ dimensi: {multi_dim_traits} ({multi_dim_traits*100/total_traits:.0f}%)")
    
    # Analyze dimension usage
    dim_usage = {}
    for trait, weights in KEYWORD_TRAIT_MAP.items():
        for dim in weights.keys():
            dim_usage[dim] = dim_usage.get(dim, 0) + 1
    
    print(f"\nDimensi OCEAN usage:")
    for dim in "OCEAN":
        count = dim_usage.get(dim, 0)
        pct = count * 100 / total_traits
        print(f"  {dim}: {count} traits ({pct:.0f}%)")
    
    print("\n3. EXAMPLE CHANGES")
    print("-" * 90)
    
    # Show example of before/after
    examples = [
        ("ANGER_EMO", "Emotional trait dengan C dimension"),
        ("COLLABORATION", "Positive trait dengan balanced dimensions"),
        ("EXTREME_NEGATIVE", "Crisis trait dengan O dimension"),
    ]
    
    for trait, desc in examples:
        weights = KEYWORD_TRAIT_MAP[trait]
        dims_str = " | ".join(f"{d}:{v:+.2f}" for d, v in sorted(weights.items()) if v != 0)
        print(f"\n{trait}:")
        print(f"  Deskripsi: {desc}")
        print(f"  Weights: {dims_str}")
        print(f"  Impact: Multi-dimensional ({len(weights)} dimensions)")
    
    print("\n" + "=" * 90)
    print("\n✅ VERIFICATION COMPLETE!")
    print("\nNext steps:")
    print("  1. Restart aplikasi: uvicorn main:app --reload")
    print("  2. Test dengan sample predictions: python test_predictions.py")
    print("  3. Verify improvement dibanding version sebelumnya")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    try:
        verify_keyword_updates()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPastikan:")
        print("  1. Anda di directory project root")
        print("  2. sapa_api/keywords.py sudah diupdate")
        print("  3. Python environment sudah di-setup")
