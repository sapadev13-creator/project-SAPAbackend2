"""
Pastikan frasa validasi empatik hanya di EMPATHY_HARMONY_A / EMO_POSITIVE,
bukan di kategori yang menaikkan O atau N.

  python scripts/fix_empathy_phrases_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sapa_api.text_utils import EMPATHY_VALIDATION_PHRASES

XLSX = ROOT / "keywords_traits.xlsx"
APP = ROOT / "app" / "keywords_traits.xlsx"
KW, TR = "Keyword / Phrase", "Trait / Kategori"

KEEP_IN = frozenset({"EMPATHY_HARMONY_A", "EMO_POSITIVE", "TRUST", "RELATIONSHIP_AFFECTION"})
REMOVE_FROM = frozenset({
    "CREATIVE_DISCUSSION_A", "INTROSPECTION", "EMO_NEGATIVE", "ANXIETY_EMO",
    "SAD_EMO", "MENTAL_UNSTABLE_N",
})


def main():
    df = pd.read_excel(XLSX)
    df.columns = [KW, TR][: len(df.columns)]
    df[KW] = df[KW].astype(str).str.strip().str.lower()

    moved = 0
    rows = []
    for _, row in df.iterrows():
        kw, tr = row[KW], row[TR]
        if any(p in kw for p in EMPATHY_VALIDATION_PHRASES) or kw in EMPATHY_VALIDATION_PHRASES:
            if tr in REMOVE_FROM:
                rows.append({KW: kw, TR: "EMPATHY_HARMONY_A"})
                moved += 1
                continue
            if tr not in KEEP_IN and tr != "POSITIVE_SOCIAL":
                rows.append({KW: kw, TR: "EMPATHY_HARMONY_A"})
                moved += 1
                continue
        rows.append({KW: kw, TR: tr})

    out = pd.DataFrame(rows).drop_duplicates(subset=[KW, TR])
    out = out.sort_values([TR, KW]).reset_index(drop=True)
    out.to_excel(XLSX, index=False)
    if APP.parent.exists():
        out.to_excel(APP, index=False)
    print(f"Dipindah/diselaraskan: {moved} baris")
    print(out[TR].value_counts().sort_index())


if __name__ == "__main__":
    main()
