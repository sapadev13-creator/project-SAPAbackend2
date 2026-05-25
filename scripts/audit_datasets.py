"""
Laporan kualitas dataset (keywords + ontology).
  python scripts/audit_datasets.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sapa_api.text_utils import GENERIC_STOPWORDS, is_meaningful_token

def audit_keywords(path: Path):
    df = pd.read_excel(path)
    df.columns = ["Keyword / Phrase", "Trait / Kategori"][: len(df.columns)]
    kw, tr = "Keyword / Phrase", "Trait / Kategori"
    df[kw] = df[kw].astype(str).str.strip().str.lower()
    cross = df.groupby(kw)[tr].nunique()
    conflicts = int((cross > 1).sum())
    intra_dup = int(df.duplicated(subset=[kw, tr]).sum())
    global_dup = int(df.duplicated(subset=[kw]).sum())
    unique_kw = df[kw].nunique()
    generic = df[df[kw].apply(lambda w: w in GENERIC_STOPWORDS or (" " not in w and not is_meaningful_token(w)))]
    extreme = df[df[tr] == "EXTREME_NEGATIVE"]
    print(f"\n=== {path.name} ===")
    print(f"Baris: {len(df)} | Keyword unik: {unique_kw}")
    print(f"Duplikat (kw+kategori sama): {intra_dup}")
    print(f"Duplikat keyword global: {global_dup}")
    print(f"Konflik antar-kategori: {conflicts}")
    print(f"Kata generik (loader skip): {len(generic)}")
    print(f"EXTREME_NEGATIVE: {len(extreme)}")
    print(df[tr].value_counts().to_string())


def audit_ontology(path: Path):
    ont = pd.read_csv(path)
    multi = ont.groupby("lexeme")["sub_trait"].nunique()
    print(f"\n=== {path.name} ===")
    print(f"Baris: {len(ont)}, lexeme unik: {ont.lexeme.nunique()}")
    print(f"Lexeme -> banyak sub_trait: {(multi > 1).sum()}")


def main():
    audit_keywords(ROOT / "keywords_traits.xlsx")
    if (ROOT / "app" / "keywords_traits.xlsx").exists():
        audit_keywords(ROOT / "app" / "keywords_traits.xlsx")
    audit_ontology(ROOT / "ontology_clean.csv")


if __name__ == "__main__":
    main()
