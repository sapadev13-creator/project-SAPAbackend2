"""
Perkaya keywords_traits.xlsx hingga ~TARGET entri per kategori.

  python scripts/enrich_keywords_to_target.py
  python scripts/enrich_keywords_to_target.py --target 1000 --extreme 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phrase_generators import (  # noqa: E402
    GENERATORS,
    TRAIT_MARKERS,
    generate_bulk_exclusive,
    generate_massive_marker,
    _norm,
    _ok,
)
from sapa_api.config import TRAIT_LIST_NAMES  # noqa: E402

XLSX = ROOT / "keywords_traits.xlsx"
ENRICHED = ROOT / "keywords_traits_enriched.xlsx"
BACKUP = ROOT / "keywords_traits.backup.xlsx"
APP = ROOT / "app" / "keywords_traits.xlsx"
KW, TR = "Keyword / Phrase", "Trait / Kategori"

TRAIT_PRIORITY = [
    "EXTREME_NEGATIVE", "ANXIETY_EMO", "SAD_EMO", "ANGER_EMO", "MENTAL_UNSTABLE_N",
    "EMO_NEGATIVE", "NEGATIVE_SOCIAL", "EMPATHY_HARMONY_A", "RELATIONSHIP_AFFECTION",
    "TRUST", "EMO_POSITIVE", "POSITIVE_SOCIAL", "EXTRAVERSION_E", "COLLABORATION",
    "E_SOCIAL_DEPENDENCY", "CREATIVE_DISCUSSION_A", "INTROSPECTION", "DISCIPLINE_C",
    "ACHIEVEMENT",
]


def _owner(kw: str, traits: list[str]) -> str:
    for t in TRAIT_PRIORITY:
        if t in traits:
            return t
    return traits[0]


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    by_kw: dict[str, list[str]] = {}
    for r in rows:
        by_kw.setdefault(r[KW], []).append(r[TR])
    out = []
    for kw, traits in by_kw.items():
        out.append({KW: kw, TR: _owner(kw, list(dict.fromkeys(traits)))})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--extreme", type=int, default=120)
    args = parser.parse_args()

    src = ENRICHED if ENRICHED.exists() else XLSX
    if src.exists():
        df = pd.read_excel(src)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = [KW, TR]
        df[KW] = df[KW].astype(str).map(_norm)
        df[TR] = df[TR].astype(str).str.strip()
        try:
            pd.read_excel(XLSX).to_excel(BACKUP, index=False)
        except Exception:
            pass
    else:
        df = pd.DataFrame(columns=[KW, TR])

    existing_global = set(df[KW])
    by_trait: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        by_trait.setdefault(r[TR], set()).add(r[KW])

    new_rows: list[dict] = []
    stats = {}

    for trait in TRAIT_LIST_NAMES:
        target = args.extreme if trait == "EXTREME_NEGATIVE" else args.target
        have = by_trait.get(trait, set())
        need = max(0, target - len(have))
        stats[trait] = {"before": len(have), "target": target, "added": 0}

        if need <= 0:
            continue

        gen = GENERATORS.get(trait)
        if not gen:
            continue

        generated = gen(existing_global | have, need + 200)
        for kw in generated:
            if len(have) >= target:
                break
            if kw in existing_global or kw in have:
                continue
            have.add(kw)
            existing_global.add(kw)
            new_rows.append({KW: kw, TR: trait})
            stats[trait]["added"] += 1

        # Pass 2: generator eksklusif per trait
        while len(have) < target:
            extra = generate_bulk_exclusive(
                trait, existing_global | have, target - len(have) + 100
            )
            if not extra:
                break
            added_this_round = 0
            for kw in extra:
                if len(have) >= target:
                    break
                if kw in existing_global or kw in have:
                    continue
                have.add(kw)
                existing_global.add(kw)
                new_rows.append({KW: kw, TR: trait})
                stats[trait]["added"] += 1
                added_this_round += 1
            if added_this_round == 0:
                break

    combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined[combined[KW].map(_ok)]
    combined = combined.drop_duplicates(subset=[KW, TR])
    rows = combined.to_dict("records")
    final = pd.DataFrame(_dedupe_rows(rows)).sort_values([TR, KW]).reset_index(drop=True)

    # Pass 3: top-up trait yang masih di bawah target
    for trait in TRAIT_LIST_NAMES:
        tcount = int((final[TR] == trait).sum())
        tgt = args.extreme if trait == "EXTREME_NEGATIVE" else args.target
        if tcount >= tgt:
            continue
        have_t = set(final.loc[final[TR] == trait, KW])
        glob = set(final[KW])
        need = tgt - tcount
        extra_rows = []
        for kw in generate_bulk_exclusive(trait, glob, need + 300):
            if len(have_t) >= tgt:
                break
            if kw in glob:
                continue
            have_t.add(kw)
            glob.add(kw)
            extra_rows.append({KW: kw, TR: trait})
        if extra_rows:
            final = pd.concat([final, pd.DataFrame(extra_rows)], ignore_index=True)
            final = pd.DataFrame(_dedupe_rows(final.to_dict("records"))).sort_values(
                [TR, KW]
            ).reset_index(drop=True)

    # Pass 4: massive marker combos untuk trait yang masih < target
    glob = set(final[KW])
    for trait in TRAIT_LIST_NAMES:
        tgt = args.extreme if trait == "EXTREME_NEGATIVE" else args.target
        tcount = int((final[TR] == trait).sum())
        if tcount >= tgt:
            continue
        markers = TRAIT_MARKERS.get(trait)
        if not markers:
            continue
        have_t = set(final.loc[final[TR] == trait, KW])
        extra_rows = []
        for kw in generate_massive_marker(trait, markers, glob, tgt - tcount + 500):
            if len(have_t) >= tgt:
                break
            if kw in glob:
                continue
            have_t.add(kw)
            glob.add(kw)
            extra_rows.append({KW: kw, TR: trait})
        if extra_rows:
            final = pd.concat([final, pd.DataFrame(extra_rows)], ignore_index=True)
            final = pd.DataFrame(_dedupe_rows(final.to_dict("records"))).sort_values(
                [TR, KW]
            ).reset_index(drop=True)

    try:
        final.to_excel(XLSX, index=False)
        if APP.parent.exists():
            final[[KW, TR]].to_excel(APP, index=False)
        out_path = XLSX
    except PermissionError:
        alt = ROOT / "keywords_traits_enriched.xlsx"
        final.to_excel(alt, index=False)
        out_path = alt
        print(f"File terkunci. Disimpan ke: {alt}")

    print(f"=== enrich_keywords_to_target (target={args.target}) ===")
    print(f"Output: {out_path}")
    print(f"Total baris: {len(final)}\n")
    vc = final[TR].value_counts().sort_index()
    print(vc.to_string())
    print("\nDetail penambahan:")
    for trait in TRAIT_LIST_NAMES:
        s = stats.get(trait, {})
        if s:
            print(f"  {trait}: {s.get('before',0)} + {s.get('added',0)} -> "
                  f"{vc.get(trait, s.get('before',0))} (target {s.get('target')})")


if __name__ == "__main__":
    main()
