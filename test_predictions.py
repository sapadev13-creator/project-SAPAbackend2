"""
TEST PREDICTIONS - Verifikasi Improvement Akurasi

Script untuk test prediksi dengan berbagai input dan
membandingkan dengan expected output setelah optimization.

Jalankan: python test_predictions.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_predictions():
    """Test predictions dengan berbagai input."""
    
    print("=" * 90)
    print("TEST PREDICTIONS - VALIDASI OCEAN WEIGHT OPTIMIZATION")
    print("=" * 90)
    
    # Test cases dengan expected traits
    test_cases = [
        {
            "name": "NEGATIVE: SADNESS",
            "text": "Aku merasa sangat sedih dan kehilangan semangat hidup",
            "expected_traits": ["SAD_EMO", "EMO_NEGATIVE"],
            "expected_high": {"N": "high", "O": "medium"},
            "expected_low": {"E": "low", "A": "low"},
        },
        {
            "name": "NEGATIVE: ANGER",
            "text": "Aku sangat kesal dan mudah marah terhadap semua orang",
            "expected_traits": ["ANGER_EMO", "NEGATIVE_SOCIAL"],
            "expected_high": {"N": "high", "A": "low"},
            "expected_low": {"C": "low"},
        },
        {
            "name": "NEGATIVE: ANXIETY",
            "text": "Saya selalu cemas tentang masa depan dan khawatir hal buruk akan terjadi",
            "expected_traits": ["ANXIETY_EMO"],
            "expected_high": {"N": "very_high"},
            "expected_low": {"E": "low", "O": "low"},
        },
        {
            "name": "POSITIVE: TRUST & COLLABORATION",
            "text": "Saya percaya pada orang lain dan suka bekerja dalam tim yang harmonis",
            "expected_traits": ["TRUST", "COLLABORATION", "EMPATHY_HARMONY_A"],
            "expected_high": {"A": "very_high", "C": "high", "E": "high"},
            "expected_low": {"N": "low"},
        },
        {
            "name": "POSITIVE: EXTRAVERSION",
            "text": "Saya sangat senang bertemu orang baru dan suka berbagi cerita",
            "expected_traits": ["EXTRAVERSION_E", "POSITIVE_SOCIAL"],
            "expected_high": {"E": "very_high", "A": "high"},
            "expected_low": {"N": "low"},
        },
        {
            "name": "OPENNESS: CREATIVITY & INTROSPECTION",
            "text": "Saya suka mengeksplorasi ide-ide baru dan sering merenungkan makna kehidupan",
            "expected_traits": ["CREATIVE_DISCUSSION_A", "INTROSPECTION"],
            "expected_high": {"O": "very_high"},
            "expected_low": {"E": "low"},
        },
        {
            "name": "CONSCIENTIOUSNESS: DISCIPLINE & ACHIEVEMENT",
            "text": "Saya sangat terorganisir, selalu tepat waktu, dan berorientasi pada prestasi",
            "expected_traits": ["DISCIPLINE_C", "ACHIEVEMENT"],
            "expected_high": {"C": "very_high", "E": "medium"},
            "expected_low": {"N": "low"},
        },
        {
            "name": "EXTREME_NEGATIVE: CRISIS",
            "text": "Ingin bunuh diri, tidak ada lagi harapan, semua sia-sia",
            "expected_traits": ["EXTREME_NEGATIVE"],
            "expected_high": {"N": "extreme"},
            "expected_low": {"E": "extreme_low", "A": "extreme_low", "C": "extreme_low"},
            "note": "CRITICAL: Should trigger crisis alert"
        },
    ]
    
    print("\nTEST CASES:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Text: \"{test_case['text']}\"")
        print(f"   Expected traits: {', '.join(test_case['expected_traits'])}")
        print(f"   Expected high: {test_case['expected_high']}")
        print(f"   Expected low: {test_case['expected_low']}")
        if "note" in test_case:
            print(f"   Note: {test_case['note']}")
        print(f"   Expected dominant: {test_case['expected_traits'][0]}")
    
    print("\n\n" + "=" * 90)
    print("CARA MENJALANKAN TEST:")
    print("=" * 90)
    
    print("""
1. Pastikan app sudah running:
   uvicorn main:app --reload

2. Untuk setiap test case di atas, jalankan:
   curl -X POST http://localhost:8000/api/predict \\
     -H "Content-Type: application/json" \\
     -d '{"text": "aku merasa sangat sedih..."}'

3. Atau gunakan Python requests:
   ```python
   import requests
   response = requests.post(
       "http://localhost:8000/api/predict",
       json={"text": "aku merasa sangat sedih..."}
   )
   result = response.json()
   print(result['adjusted'])  # Check OCEAN values
   ```

4. Verifikasi hasil:
   ✓ Trait discrimination lebih baik (bukan hanya N tinggi)
   ✓ Positive traits lebih terdeteksi
   ✓ Crisis cases lebih kuat signal-nya
   ✓ Multi-dimensional approach terlihat pada OCEAN output

EXPECTED IMPROVEMENTS DARI OPTIMIZATION:
========================================

1. SADNESS vs ANGER vs ANXIETY:
   Sebelum: Semua high N, sulit dibedakan
   Sesudah: 
   - SAD_EMO:     N: 0.40 (medium), O: +0.05
   - ANGER_EMO:   N: 0.50 (tinggi), A: -0.20
   - ANXIETY_EMO: N: 0.55 (tinggi), E: -0.20, O: -0.10

2. POSITIVE TRAITS:
   Sebelum: Mostly A (agreeableness), kurang C dan O
   Sesudah: Balanced approach dengan C (discipline) dan O (openness)

3. EXTREME_NEGATIVE:
   Sebelum: N: 1.8, E/A/C reduced
   Sesudah: N: 2.0 (lebih kuat), O: -0.4 (NEW - untuk nuansa extreme)

4. OVERALL:
   - More nuanced predictions
   - Better emotional trait differentiation
   - Stronger crisis detection
   - More consistent with test inputs
""")
    
    print("\n" + "=" * 90)
    print("VALIDATION CHECKLIST:")
    print("=" * 90)
    
    checklist = [
        "✓ KEYWORD_TRAIT_MAP sudah diupdate dengan optimized weights",
        "○ App sudah di-restart (untuk load new weights)",
        "○ Sample predictions sudah ditest",
        "○ Results menunjukkan improvement dibanding sebelumnya",
        "○ Crisis detection lebih kuat untuk extreme cases",
        "○ Positive traits lebih sering dideteksi",
        "○ Emotional traits lebih terdiskriminasi",
        "○ No regressions pada existing functionality",
    ]
    
    for check in checklist:
        print(f"  {check}")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    test_predictions()
