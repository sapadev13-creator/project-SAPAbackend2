"""
Rapikan keywords_traits.xlsx:
- Pisah kolom Intent (konstruk | narasi | kata_tunggal)
- Hapus narasi panjang (>55 char, diawali aku/ada/kenapa)
- Pindahkan frasa validasi ke EMPATHY, distres ke ANXIETY/SAD
- Dedupe antar kategori

  python scripts/reorganize_dataset_by_intent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sapa_api.text_utils import EMPATHY_VALIDATION_PHRASES, is_meaningful_token

XLSX = ROOT / "keywords_traits.xlsx"
APP = ROOT / "app" / "keywords_traits.xlsx"
BACKUP = ROOT / "keywords_traits.backup.xlsx"

KW, TR, INTENT = "Keyword / Phrase", "Trait / Kategori", "Intent"

NARRATIVE_PREFIXES = ("aku ", "ada ", "kenapa ", "baru ", "setiap ", "kalau ", "gimana ")

TRAIT_PRIORITY = [
    "EXTREME_NEGATIVE", "ANXIETY_EMO", "SAD_EMO", "ANGER_EMO", "MENTAL_UNSTABLE_N",
    "EMO_NEGATIVE", "NEGATIVE_SOCIAL", "EMPATHY_HARMONY_A", "RELATIONSHIP_AFFECTION",
    "TRUST", "EMO_POSITIVE", "POSITIVE_SOCIAL", "EXTRAVERSION_E", "COLLABORATION",
    "E_SOCIAL_DEPENDENCY", "CREATIVE_DISCUSSION_A", "INTROSPECTION", "DISCIPLINE_C",
    "ACHIEVEMENT",
]


def _normalize(kw: str) -> str:
    return " ".join(str(kw).strip().lower().split())


def _classify_intent_row(kw: str, trait: str) -> str:
    if any(p in kw for p in EMPATHY_VALIDATION_PHRASES):
        return "validasi_empati"
    if kw.startswith(NARRATIVE_PREFIXES) or len(kw) > 55:
        return "narasi"
    if " " not in kw:
        return "kata_tunggal"
    return "frasa_konstruk"


def _should_drop(kw: str, intent: str) -> bool:
    if intent == "narasi":
        return True
    if len(kw) < 3:
        return True
    if " " not in kw and not is_meaningful_token(kw):
        return True
    if "timsenang" in kw or "  " in kw:
        return True
    return False


def _owner_trait(kw: str, traits: list[str]) -> str:
    if any(x in kw for x in ("bunuh", "ingin mati", "melukai diri")):
        return "EXTREME_NEGATIVE" if "EXTREME_NEGATIVE" in traits else traits[0]
    if any(p in kw for p in EMPATHY_VALIDATION_PHRASES):
        return "EMPATHY_HARMONY_A" if "EMPATHY_HARMONY_A" in traits else traits[0]
    if kw.startswith("butuh") or "ketergantungan sosial" in kw:
        return "E_SOCIAL_DEPENDENCY" if "E_SOCIAL_DEPENDENCY" in traits else traits[0]
    if kw.startswith(("tim ", "suka tim", "kolabor")):
        return "COLLABORATION" if "COLLABORATION" in traits else traits[0]
    if kw.startswith("prestasi"):
        return "ACHIEVEMENT" if "ACHIEVEMENT" in traits else traits[0]
    if kw.startswith("batin "):
        return "EMO_NEGATIVE" if "EMO_NEGATIVE" in traits else traits[0]
    if kw.startswith("sosial "):
        return "POSITIVE_SOCIAL" if "POSITIVE_SOCIAL" in traits else traits[0]
    for t in TRAIT_PRIORITY:
        if t in traits:
            return t
    return traits[0]


def main():
    df = pd.read_excel(XLSX)
    df = df.iloc[:, :2]
    df.columns = [KW, TR]

    rows = []
    dropped = 0
    for _, r in df.iterrows():
        kw, tr = _normalize(r[KW]), str(r[TR]).strip()
        intent = _classify_intent_row(kw, tr)
        if _should_drop(kw, intent):
            dropped += 1
            continue
        if any(p in kw for p in EMPATHY_VALIDATION_PHRASES) and tr in (
            "INTROSPECTION", "CREATIVE_DISCUSSION_A", "EMO_NEGATIVE", "ANXIETY_EMO",
        ):
            tr = "EMPATHY_HARMONY_A"
            intent = "validasi_empati"
        rows.append({KW: kw, TR: tr, INTENT: intent})

    clean = pd.DataFrame(rows)
    by_kw: dict[str, list] = {}
    for _, r in clean.iterrows():
        by_kw.setdefault(r[KW], []).append(r[TR])

    deduped = []
    for kw, traits in by_kw.items():
        unique = list(dict.fromkeys(traits))
        tr = _owner_trait(kw, unique)
        intent = _classify_intent_row(kw, tr)
        deduped.append({KW: kw, TR: tr, INTENT: intent})

    out = pd.DataFrame(deduped).sort_values([TR, KW]).reset_index(drop=True)

    pd.read_excel(XLSX).to_excel(BACKUP, index=False)
    curated = ROOT / "keywords_traits_curated.xlsx"
    try:
        out.to_excel(XLSX, index=False)
        if APP.parent.exists():
            out[[KW, TR]].to_excel(APP, index=False)
        print(f"Written: {XLSX}")
    except PermissionError:
        out.to_excel(curated, index=False)
        print(f"File terkunci — simpan ke: {curated}")
        print("Tutup Excel lalu ganti keywords_traits.xlsx dengan file ini.")

    print(f"Sebelum: {len(df)} | Drop narasi/generik: {dropped} | Sesudah: {len(out)}")
    print("\nPer Intent:")
    print(out[INTENT].value_counts().to_string())
    print("\nPer Trait:")
    print(out[TR].value_counts().to_string())


if __name__ == "__main__":
    main()
