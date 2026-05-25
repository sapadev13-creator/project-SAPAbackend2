"""
Optimasi Frasa Keywords untuk Prediksi Lebih Akurat.

Strategi:
1. Shorten long phrases (>5 kata / >55 char) ke 2-4 kata
2. Pisahkan konteks kompleks menjadi frasa terpisah
3. Improve semantic clarity
4. Preserve intent dari frasa original

Jalankan: python scripts/optimize_phrases.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "keywords_traits.xlsx"
OUTPUT = ROOT / "keywords_traits_optimized.xlsx"

KW_COL = "Keyword / Phrase"
TR_COL = "Trait / Kategori"

# Mapping frasa panjang ke frasa pendek yang lebih optimal
PHRASE_OPTIMIZATIONS = {
    # ANXIETY_EMO optimizations
    "cemas akan hal yang mungkin tidak terjadi": "cemas berlebihan",
    "khawatir tentang hal-hal yang akan datang": "khawatir masa depan",
    "sering merasa tegang atau khawatir": "sering cemas",
    "mudah terkejut atau terganggu": "mudah terkejut",
    "sulit bersantai atau tenang": "sulit tenang",
    
    # SAD_EMO optimizations
    "merasa sedih atau murung": "merasa sedih",
    "sering merasa sedih atau kecewa": "sering sedih",
    "sulit menemukan kesenangan": "sulit senang",
    "cenderung merasa depresi": "cenderung depresi",
    "kehilangan minat pada hal yang dulunya disukai": "kehilangan minat",
    
    # ANGER_EMO optimizations
    "mudah kesal atau jengkel": "mudah kesal",
    "sering marah pada hal kecil": "mudah marah",
    "cepat meluap saat emosi": "cepat meluap",
    "suka berbicara kasar saat marah": "berbicara kasar",
    "aku cepat kesal saat hal tidak sesuai": "cepat kesal",
    
    # NEGATIVE_SOCIAL optimizations
    "tidak suka berinteraksi dengan banyak orang": "избегают interaksi banyak orang",
    "sulit memulai percakapan": "sulit memulai percakapan",
    "merasa canggung saat bertemu orang baru": "merasa canggung",
    "lebih suka sendirian daripada dengan orang lain": "suka sendirian",
    "sulit untuk mengekspresikan perasaan": "sulit ekspresikan",
    
    # EXTRAVERSION_E optimizations (keep these as they're already good)
    
    # POSITIVE_SOCIAL optimizations
    "senang menghabiskan waktu dengan orang lain": "senang bersama orang",
    "suka berbagi cerita dan pengalaman": "suka berbagi cerita",
    "merasa energi saat bersama teman": "energi bersama teman",
    "tidak takut mengajukan ide dalam grup": "percaya diri di grup",
    
    # COLLABORATION optimizations
    "kerja tim terasa menyenangkan saat tidak ada rasa takut": "kerja tim menyenangkan",
    "senang bekerja dalam proyek bersama": "senang kerja tim",
    "bisa berkompromi saat ada perbedaan pendapat": "suka berkompromi",
    "menghargai kontribusi orang lain di tim": "menghargai kontribusi",
    
    # TRUST optimizations
    "percaya pada niat baik orang lain": "percaya orang lain",
    "tidak cepat menuduh atau meragukan orang": "tidak meragukan",
    
    # CREATIVE_DISCUSSION_A optimizations
    "suka mengeksplorasi ide-ide baru": "suka ide baru",
    "terbuka pada perspektif berbeda": "terbuka perspektif",
    "menikmati diskusi mendalam": "suka diskusi",
    
    # DISCIPLINE_C optimizations
    "selalu tepat waktu dalam pekerjaan": "tepat waktu",
    "terorganisir dan sistematis": "terorganisir",
    "konsisten mengikuti rencana": "konsisten rencana",
    
    # ACHIEVEMENT optimizations
    "berorientasi pada hasil konkret": "fokus hasil",
    "suka menyelesaikan proyek sampai selesai": "menyelesaikan proyek",
    "termotivasi oleh pencapaian": "termotivasi prestasi",
    
    # EXTREME_NEGATIVE optimizations (keep crisis phrases - already optimized)
    "aku memikirkan ulang kata-kataku sendiri lebih lama dari orang lain mengingatnya": "overthinking",
    "aku terlalu memikirkan kehilangan sebelum kehilangan itu nyata": "khawatir kehilangan",
    
    # INTROSPECTION optimizations
    "suka merenungkan makna hidup": "suka renungan",
    "memikirkan tujuan dan nilai dalam hidup": "pikirkan tujuan",
    
    # EMO_POSITIVE optimizations
    "merasa bahagia atau puas": "merasa bahagia",
    "senang dengan pencapaian diri": "senang prestasi",
    
    # E_SOCIAL_DEPENDENCY optimizations
    "merasa tidak nyaman saat sendirian": "tidak nyaman sendirian",
    "membutuhkan validasi dari orang lain": "butuh validasi",
    
    # RELATIONSHIP_AFFECTION optimizations
    "peduli dan perhatian pada teman": "peduli teman",
    "senang membantu atau menolong": "senang membantu",
}

# Frasa kompleks yang perlu dipecah jadi lebih dari satu frasa
PHRASE_SPLITS = {
    # Buat variasi dari frasa kompleks
    "kerja tim terasa hangat saat tidak ada yang merasa ditinggal": [
        "kerja tim hangat",
        "tidak ada ditinggal",
    ],
    "aku sering berbicara dengan nada tinggi saat emosi": [
        "berbicara nada tinggi",
        "emosi tinggi",
    ],
}


def optimize_phrases():
    """Optimize semua frasa panjang."""
    df = pd.read_excel(XLSX)
    
    optimized_rows = []
    changes = []
    
    for idx, row in df.iterrows():
        keyword = str(row[KW_COL]).strip().lower()
        trait = str(row[TR_COL]).strip()
        
        if not keyword or not trait:
            continue
        
        optimized_rows.append({
            KW_COL: keyword,
            TR_COL: trait,
        })
        
        # Check for optimizations
        if keyword in PHRASE_OPTIMIZATIONS:
            new_kw = PHRASE_OPTIMIZATIONS[keyword]
            changes.append({
                'trait': trait,
                'original': keyword,
                'optimized': new_kw,
                'type': 'shortened',
            })
        elif keyword in PHRASE_SPLITS:
            # Add original
            for split_phrase in PHRASE_SPLITS[keyword]:
                optimized_rows.append({
                    KW_COL: split_phrase,
                    TR_COL: trait,
                })
                changes.append({
                    'trait': trait,
                    'original': keyword,
                    'optimized': split_phrase,
                    'type': 'split',
                })
    
    print(f"✓ Diperoleh {len(changes)} optimasi frasa")
    
    # Buat versi dengan optimasi applied
    df_optimized = pd.DataFrame(optimized_rows)
    
    # Apply optimizations
    df_optimized[KW_COL] = df_optimized[KW_COL].apply(
        lambda x: PHRASE_OPTIMIZATIONS.get(x, x)
    )
    
    df_optimized.to_excel(OUTPUT, index=False)
    print(f"✓ File optimasi disimpan ke: {OUTPUT}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RINGKASAN OPTIMASI FRASA")
    print("=" * 80)
    
    shortened = [c for c in changes if c['type'] == 'shortened']
    split = [c for c in changes if c['type'] == 'split']
    
    if shortened:
        print(f"\nDIPENDEK ({len(shortened)} frasa):")
        for change in shortened[:10]:
            print(f"  '{change['original']}'")
            print(f"    → '{change['optimized']}'")
            print()
    
    if split:
        print(f"\nDIPISAH ({len(split)} frasa):")
        for change in split[:10]:
            print(f"  '{change['original']}'")
            print(f"    → '{change['optimized']}'")
            print()
    
    print("=" * 80)
    print(f"\n✓ Total keywords sebelum: {len(df)}")
    print(f"✓ Total keywords sesudah: {len(df_optimized)}")
    print(f"✓ Penambahan dari split: +{len(df_optimized) - len(df)} keyword")


if __name__ == "__main__":
    optimize_phrases()
