"""
Audit & rapikan keywords_traits.xlsx agar prediksi lebih konsisten.

Jalankan dari root proyek:
  python scripts/clean_keywords_traits.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "keywords_traits.xlsx"
BACKUP = ROOT / "keywords_traits.backup.xlsx"
APP_COPY = ROOT / "app" / "keywords_traits.xlsx"

KW_COL = "Keyword / Phrase"
TR_COL = "Trait / Kategori"

# Urutan prioritas saat keyword sama muncul di banyak kategori (kategori pertama menang)
TRAIT_PRIORITY = [
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
]

# Hanya frasa/kata ini yang boleh tetap di EXTREME_NEGATIVE (krisis / risiko tinggi)
CRISIS_EXTREME_PHRASES = frozenset({
    "bunuh diri", "ingin bunuh diri", "pengen bunuh diri", "mau bunuh diri",
    "ingin bunuh", "pengen bunuh", "mau bunuh", "ingin mati", "pengen mati",
    "mau mati", "tidak ingin hidup", "tidak mau hidup", "melukai diri",
    "menyakiti diri", "akhiri hidup", "akhiri hidupku", "lepaskan nyawa",
    "putus asa ingin mati", "putus asa total", "hidup tidak berarti",
    "hidup tak berarti", "ingin berhenti hidup", "tak ingin lanjut hidup",
    "ingin menghilang", "ingin hilang saja",
})

# Frasa di EXTREME yang dipindah ke kategori distres (bukan krisis bunuh diri)
EXTREME_REASSIGN_TO_SAD = frozenset({
    "putus harapan", "kehilangan harapan", "habis harapan", "putus asa berat",
    "terpuruk ekstrem", "terpuruk parah", "hancur batin", "jiwa runtuh",
    "hidup runtuh", "hidup kosong", "hampa segalanya", "ingin menyerah",
    "rasanya ingin menyerah", "menyerah sepenuhnya", "lelah hidup",
    "lelah bertahan", "lelah mental", "jiwa kelelahan",
})

EXTREME_REASSIGN_TO_ANXIETY = frozenset({
    "tertekan", "terstres", "terlalu berat", "terlalu sakit", "dunia menekan",
    "dunia menekan berat", "dunia gelap", "gelap batin", "gelap pikiran",
    "terjebak gelap", "terjebak keputusasaan", "terjebak putus asa",
})

GENERIC_DROP = frozenset({
    "tim", "iba", "kamu", "kita", "kami", "dia", "lo", "lu", "gue", "gw",
    "the", "and", "or", "yang", "ini", "itu", "dan", "atau", "bisa", "akan",
    "hari", "besok", "nanti", "sekarang", "emosi", "hal", "sesuatu",
})

TYPO_DROP = frozenset({"mebunuh", "memati", "memenyerah"})


def _normalize(kw: str) -> str:
    return " ".join(str(kw).strip().lower().split())


def _pick_trait(traits: list[str]) -> str:
    for t in TRAIT_PRIORITY:
        if t in traits:
            return t
    return traits[0]


def _is_crisis_extreme(kw: str) -> bool:
    if kw in CRISIS_EXTREME_PHRASES:
        return True
    return any(c in kw for c in ("bunuh diri", "ingin bunuh", "pengen bunuh", "mau bunuh"))


def _reassign_extreme(kw: str) -> str | None:
    if kw in EXTREME_REASSIGN_TO_SAD:
        return "SAD_EMO"
    if kw in EXTREME_REASSIGN_TO_ANXIETY:
        return "ANXIETY_EMO"
    if kw in TYPO_DROP:
        return None
    if kw in GENERIC_DROP:
        return None
    return None


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[KW_COL] = df[KW_COL].astype(str).map(_normalize)
    df[TR_COL] = df[TR_COL].astype(str).str.strip()

    stats = {
        "rows_in": len(df),
        "dropped_generic": 0,
        "dropped_typo": 0,
        "extreme_reassigned": 0,
        "extreme_removed_non_crisis": 0,
        "deduped_cross_trait": 0,
    }

    # Drop generic / typo
    mask_drop = df[KW_COL].isin(GENERIC_DROP | TYPO_DROP)
    stats["dropped_generic"] = int(df[df[KW_COL].isin(GENERIC_DROP)].shape[0])
    stats["dropped_typo"] = int(df[df[KW_COL].isin(TYPO_DROP)].shape[0])
    df = df[~mask_drop]

    # Reassign / filter EXTREME_NEGATIVE
    rows = []
    for _, row in df.iterrows():
        kw, tr = row[KW_COL], row[TR_COL]
        if tr == "EXTREME_NEGATIVE":
            new_tr = _reassign_extreme(kw)
            if new_tr:
                stats["extreme_reassigned"] += 1
                rows.append({KW_COL: kw, TR_COL: new_tr})
                continue
            if not _is_crisis_extreme(kw):
                stats["extreme_removed_non_crisis"] += 1
                if any(x in kw for x in ("sedih", "lelah", "putus", "hampa", "kosong", "menyerah")):
                    rows.append({KW_COL: kw, TR_COL: "SAD_EMO"})
                elif any(x in kw for x in ("cemas", "gelisah", "tekan", "gelap", "stres")):
                    rows.append({KW_COL: kw, TR_COL: "ANXIETY_EMO"})
                else:
                    rows.append({KW_COL: kw, TR_COL: "MENTAL_UNSTABLE_N"})
                continue
        rows.append({KW_COL: kw, TR_COL: tr})
    df = pd.DataFrame(rows)

    # Satu keyword -> satu kategori (prioritas)
    by_kw: dict[str, list[str]] = {}
    for kw, tr in zip(df[KW_COL], df[TR_COL]):
        by_kw.setdefault(kw, []).append(tr)

    deduped = []
    for kw, traits in by_kw.items():
        unique = list(dict.fromkeys(traits))
        if len(unique) > 1:
            stats["deduped_cross_trait"] += len(unique) - 1
        deduped.append({KW_COL: kw, TR_COL: _pick_trait(unique)})

    out = pd.DataFrame(deduped)
    out = out.drop_duplicates(subset=[KW_COL, TR_COL])
    out = out.sort_values([TR_COL, KW_COL]).reset_index(drop=True)
    stats["rows_out"] = len(out)
    return out, stats


def main():
    if not XLSX.exists():
        raise FileNotFoundError(XLSX)

    df = pd.read_excel(XLSX)
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = [KW_COL, TR_COL]

    cleaned, stats = clean_dataframe(df)

    pd.read_excel(XLSX).to_excel(BACKUP, index=False)
    cleaned.to_excel(XLSX, index=False)
    if APP_COPY.parent.exists():
        cleaned.to_excel(APP_COPY, index=False)

    print("=== clean_keywords_traits ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nBackup: {BACKUP}")
    print(f"Written: {XLSX}")
    if APP_COPY.exists():
        print(f"Synced: {APP_COPY}")
    print("\nPer kategori:")
    print(cleaned[TR_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
