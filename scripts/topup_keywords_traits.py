"""Top-up kategori yang masih di bawah target setelah balance. python scripts/topup_keywords_traits.py"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from balance_keywords_traits import (  # noqa: E402
    APP_COPY,
    BACKUP,
    KW_COL,
    TR_COL,
    TOP_UP_UNIQUE,
    XLSX,
    _dedupe_cross_trait,
    _fill_group,
    _normalize,
    _target_for,
)

if __name__ == "__main__":
    df = pd.read_excel(XLSX)
    df.columns = [KW_COL, TR_COL]
    df[KW_COL] = df[KW_COL].map(_normalize)

    for trait in sorted(df[TR_COL].unique()):
        target = _target_for(trait)
        sub = df[df[TR_COL] == trait]
        if len(sub) < target:
            filled = _fill_group(sub, trait, target)
            df = pd.concat([df[df[TR_COL] != trait], filled], ignore_index=True)

    df = _dedupe_cross_trait(df)
    for trait in sorted(df[TR_COL].unique()):
        target = _target_for(trait)
        sub = df[df[TR_COL] == trait]
        if len(sub) < target:
            filled = _fill_group(sub, trait, target)
            df = pd.concat([df[df[TR_COL] != trait], filled], ignore_index=True)
    df = _dedupe_cross_trait(df).sort_values([TR_COL, KW_COL])

    df.to_excel(XLSX, index=False)
    if APP_COPY.parent.exists():
        df.to_excel(APP_COPY, index=False)
    print(df[TR_COL].value_counts().sort_index())
    print("total", len(df))
