"""
Audit & deduplikasi keywords_traits — setiap frasa unik (global, satu kategori).

  python scripts/dedupe_keywords_dataset.py
  python scripts/dedupe_keywords_dataset.py --input keywords_traits_enriched.xlsx
  python scripts/dedupe_keywords_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KW_COL = "Keyword / Phrase"
TR_COL = "Trait / Kategori"

TRAIT_PRIORITY = [
    "EXTREME_NEGATIVE",
    "ANXIETY_EMO",
    "SAD_EMO",
    "ANGER_EMO",
    "MENTAL_UNSTABLE_N",
    "EMO_NEGATIVE",
    "NEGATIVE_SOCIAL",
    "EMPATHY_HARMONY_A",
    "RELATIONSHIP_AFFECTION",
    "TRUST",
    "EMO_POSITIVE",
    "POSITIVE_SOCIAL",
    "EXTRAVERSION_E",
    "COLLABORATION",
    "E_SOCIAL_DEPENDENCY",
    "CREATIVE_DISCUSSION_A",
    "INTROSPECTION",
    "DISCIPLINE_C",
    "ACHIEVEMENT",
]


def _norm(kw: str) -> str:
    return " ".join(str(kw).strip().lower().split())


def _pick_trait(traits: list[str]) -> str:
    for t in TRAIT_PRIORITY:
        if t in traits:
            return t
    return traits[0]


def audit_df(df: pd.DataFrame) -> dict:
    kw = df[KW_COL].map(_norm)
    tr = df[TR_COL].astype(str).str.strip()
    intra = int(df.assign(_kw=kw, _tr=tr).duplicated(subset=["_kw", "_tr"]).sum())
    global_dup_rows = int(kw.duplicated().sum())
    cross = kw.groupby(kw).apply(lambda s: tr.loc[s.index].nunique())
    conflicts = int((cross > 1).sum())
    empty = int(((kw == "") | kw.isna()).sum())
    return {
        "rows": len(df),
        "unique_kw": kw.nunique(),
        "intra_dup_rows": intra,
        "global_dup_rows": global_dup_rows,
        "cross_category_keywords": conflicts,
        "empty_kw": empty,
    }


def dedupe_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    df = df.copy()
    df[KW_COL] = df[KW_COL].astype(str).map(_norm)
    df[TR_COL] = df[TR_COL].astype(str).str.strip()
    df = df[(df[KW_COL] != "") & df[KW_COL].notna()]

    # Hapus duplikat persis (kw + kategori sama)
    before_intra = len(df)
    df = df.drop_duplicates(subset=[KW_COL, TR_COL], keep="first")
    removed_intra = before_intra - len(df)

    # Satu keyword -> satu kategori (prioritas)
    by_kw: dict[str, list[str]] = {}
    for kw, tr in zip(df[KW_COL], df[TR_COL]):
        if tr not in by_kw.get(kw, []):
            by_kw.setdefault(kw, []).append(tr)

    removed_cross = 0
    deduped = []
    for kw, traits in by_kw.items():
        if len(traits) > 1:
            removed_cross += len(traits) - 1
        deduped.append({KW_COL: kw, TR_COL: _pick_trait(traits)})

    out = pd.DataFrame(deduped)
    out = out.drop_duplicates(subset=[KW_COL])
    out = out.sort_values([TR_COL, KW_COL]).reset_index(drop=True)

    stats = {
        "rows_in": rows_in,
        "removed_intra": removed_intra,
        "removed_cross_category": removed_cross,
        "rows_out": len(out),
    }
    return out, stats


def load_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = [KW_COL, TR_COL]
    return df


def process_file(path: Path, dry_run: bool, backup: bool) -> None:
    if not path.exists():
        print(f"[skip] tidak ada: {path}")
        return

    raw = load_xlsx(path)
    before = audit_df(raw)
    cleaned, stats = dedupe_df(raw)
    after = audit_df(cleaned)

    print(f"\n=== {path.name} ===")
    print("Sebelum:", before)
    print("Aksi:", stats)
    print("Sesudah:", after)

    if dry_run:
        return

    if backup and path.name == "keywords_traits.xlsx":
        bak = path.with_name("keywords_traits.backup.xlsx")
        shutil.copy2(path, bak)
        print(f"Backup: {bak}")

    cleaned.to_excel(path, index=False)
    print(f"Disimpan: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "keywords_traits.xlsx")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--sync-app", action="store_true", default=True)
    parser.add_argument("--all-copies", action="store_true", help="Juga dedupe enriched + app")
    args = parser.parse_args()

    targets = [args.input]
    if args.all_copies:
        targets.extend([
            ROOT / "keywords_traits_enriched.xlsx",
            ROOT / "app" / "keywords_traits.xlsx",
        ])
    elif args.sync_app:
        app = ROOT / "app" / "keywords_traits.xlsx"
        if args.input.resolve() == (ROOT / "keywords_traits.xlsx").resolve():
            targets.append(app)

    master_path = (ROOT / "keywords_traits.xlsx").resolve()
    for p in targets:
        if p.resolve() == master_path or p.name != "keywords_traits.xlsx":
            process_file(p, args.dry_run, backup=not args.no_backup and p.resolve() == master_path)

    if not args.dry_run and args.sync_app and master_path.exists():
        master_df = load_xlsx(master_path)
        app = ROOT / "app" / "keywords_traits.xlsx"
        if app.parent.exists():
            master_df.to_excel(app, index=False)
            print(f"\nSinkron app ← master: {app} ({len(master_df)} baris)")


if __name__ == "__main__":
    main()
