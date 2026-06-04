"""
Tambah kata/frasa slang kekinian untuk prediksi emosi hostile (append-only).

  python scripts/add_kekinian_slang_keywords.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- Slang inti per kategori ---

CRITICAL_SLANG = (
    "bacot", "bacot mulu", "bacot terus", "bacot doang", "bacot aja",
    "tolol", "tolol banget", "tolol parah", "tolol abis", "tolol amat",
    "bullshit", "bulshit", "bs banget", "omong kosong", "ngawur",
    "ngaco", "ngaco banget", "ngaco parah", "ngaco terus",
    "nyindir", "nyindir terus", "nyinyir", "nyinyir mulu", "sarkas",
    "nyolot", "nyolot banget", "ngegas", "ngegas banget", "sok tau",
    "sok pinter", "sok jagoan", "sok keras", "sok asik", "sok bijak",
    "toxic", "toxic banget", "toxic parah", "hater", "haters",
    "roast", "diroast", "sindiran", "sindiran tajam", "caci maki",
    "hujat", "hujat terus", "ngehina", "ngehina terus", "mempermalukan",
    "kritik kasar", "kritik pedas", "komen toxic", "komentar toxic",
    "sok ngatur", "sok suci", "sok benar", "sok polos",
    "kaga pernah inget dosa", "gak pernah inget dosa", "nggak pernah inget dosa",
    "kaga pernah sadar", "gak pernah sadar diri", "nggak pernah sadar diri",
    "sok suci padahal", "sok baik padahal", "pura pura baik",
    "munafik banget", "munafik parah", "two face", "palsu banget",
    "sok victim", "main victim", "playing victim", "sok tersakiti",
    "blok aja", "blok lu", "blok dia", "blok deh", "mending blok",
    "skip aja", "skip lu", "gak layak", "kaga layak", "nggak layak",
    "payah banget", "payah parah", "goblog", "goblok", "bodoh banget",
    "dongo", "bego banget", "tolol pisan", "bacot pisan",
)

HATRED_SLANG = (
    "bangsat", "bangsat lu", "bangsat dia", "bangsat banget", "bangsat emang",
    "keparat", "keparat lu", "keparat emang", "keparat banget",
    "brengsek", "brengsek banget", "brengsek lu", "brengsek parah",
    "biadab", "biadab banget", "jahanam", "jahanam lu",
    "sialan", "sialan lu", "sialan banget", "setan", "setan lu",
    "benci banget", "benci lu", "benci dia", "benci parah", "benci mati",
    "muak", "muak banget", "muak liat", "muak sama", "muak parah",
    "jijik", "jijik banget", "jijik liat", "jijik parah",
    "dendam", "dendam banget", "dendam parah", "balas dendam",
    "benci setengah mati", "benci berat", "membenci", "membenci lu",
    "kebencian", "penuh kebencian", "rasa benci", "benci total",
    "gak suka banget", "kaga suka banget", "nggak suka banget",
    "benci banget sama", "muak sama lu", "jijik sama lu",
    "benci lu banget", "benci dia banget", "benci banget lu",
    "kaga pernah inget dosa", "gak pernah inget dosa",
    "pantang ampun", "gak mau maafin", "kaga mau maafin",
    "benci sampe tulang", "benci sampai mati",
)

RAGE_SLANG = (
    "mengacau", "mengacau banget", "mengacau terus", "mengacau parah",
    "ngacau", "ngacau banget", "ngacau terus", "ngacau parah",
    "ngamuk", "ngamuk banget", "ngamuk terus", "ngamuk parah",
    "meledak", "meledak emosi", "emosi meledak", "marah meledak",
    "luapan emosi", "luapan kemarahan", "emosi meluap",
    "ngegas", "ngegas banget", "ngegas terus", "ngegas parah",
    "kesel banget", "kesel parah", "kesel terus", "kesel abis",
    "jengkel banget", "jengkel parah", "jengkel terus",
    "emosi banget", "emosi parah", "emosi terus", "emosi naik",
    "marah banget", "marah parah", "marah terus", "marah abis",
    "naik pitam", "pitam banget", "geram banget", "murka",
    "frustasi banget", "frustasi parah", "frustasi terus",
    "emosi tidak terkendali", "marah tak terkendali",
    "sulit menahan emosi", "sulit menahan marah",
    "emosi meledak keluar", "marah meledak keluar",
)

EMO_DARK_SLANG = (
    "suram", "suram banget", "suram parah", "hidup suram", "mood suram",
    "bobrok", "bobrok banget", "bobrok parah", "hidup bobrok", "mental bobrok",
    "hancur", "hancur banget", "hancur parah", "mental hancur", "hidup hancur",
    "ancur", "ancur banget", "ancur parah", "ancur total", "mental ancur",
    "down bad", "down banget", "down parah", "sangat down",
    "lelah hidup", "lelah banget", "lelah parah", "capek hidup",
    "kosong banget", "hampa banget", "suram terus", "bobrok terus",
    "hidup suram banget", "hidup bobrok banget", "mental suram",
    "perasaan suram", "perasaan bobrok", "emosi suram",
)

ANGER_SLANG = (
    "kesel", "kesel banget", "jengkel", "jengkel banget", "geram",
    "marah", "marah banget", "kesal", "kesal banget", "sebel",
    "sebel banget", "gemesin", "nyebelin", "nyebelin banget",
    "kesel lu", "jengkel lu", "marah lu", "kesal lu",
    "blok", "blok dia", "blok lu", "mending blok",
)

NEGATIVE_SOCIAL_SLANG = (
    "blok", "blok aja", "blok lu", "blok dia", "blok deh",
    "ghosting", "di ghosting", "skip", "skip aja", "skip lu",
    "jauhin", "jauhin aja", "jauhin lu", "menjauh", "menjauh aja",
    "gak mau deket", "kaga mau deket", "nggak mau deket",
    "gak mau ngobrol", "kaga mau ngobrol", "cut off", "cut off aja",
)

# Modifier & objek untuk kombinasi
MODIFIERS = (
    "banget", "parah", "terus", "mulu", "abis", "amat", "pisan",
    "sangat", "bener bener", "beneran", "sungguh", "total",
)
TARGETS = (
    "lu", "dia", "mereka", "orang itu", "temen gue", "temen gw",
    "dia banget", "lu banget", "orang ini", "si dia",
)
ACTIONS = (
    "bacot", "ngaco", "mengacau", "nyindir", "ngehina", "ngegas",
    "ngamuk", "kesel", "marah", "benci", "muak", "jijik",
)
CONTEXTS = (
    "di chat", "di grup", "di medsos", "di twitter", "di ig",
    "di kantor", "di sekolah", "di rumah", "online", "offline",
    "tiap hari", "terus menerus", "mulu tiap hari", "setiap hari",
    "pas marah", "pas emosi", "pas kesel", "pas ngamuk",
)


def _combine_slang(
    core: tuple[str, ...],
    extra_mod: tuple[str, ...] = MODIFIERS,
    extra_tgt: tuple[str, ...] = TARGETS,
    extra_ctx: tuple[str, ...] = CONTEXTS,
) -> list[str]:
    out: set[str] = set(core)
    singles = [w for w in core if " " not in w and len(w) >= 4]
    for s in singles[:20]:
        for m in extra_mod:
            out.add(f"{s} {m}")
        for t in extra_tgt[:8]:
            out.add(f"{s} {t}")
    for a in ACTIONS[:12]:
        for m in extra_mod[:8]:
            out.add(f"{a} {m}")
        for c in extra_ctx[:10]:
            out.add(f"{a} {c}")
    for a, t in itertools.product(ACTIONS[:10], extra_tgt[:6]):
        out.add(f"{a} {t}")
    for a, m, t in itertools.product(ACTIONS[:8], extra_mod[:5], extra_tgt[:4]):
        out.add(f"{a} {m} {t}")
    for phrase in core:
        if " " in phrase:
            for c in extra_ctx[:6]:
                out.add(f"{phrase} {c}")
    return sorted(p.lower().strip() for p in out if len(p.strip()) >= 3)


def _build_phrases() -> dict[str, list[str]]:
    return {
        "CRITICAL_HOSTILE": _combine_slang(CRITICAL_SLANG),
        "HATRED_EMO": _combine_slang(HATRED_SLANG),
        "RAGE_OVERFLOW": _combine_slang(RAGE_SLANG),
        "EMO_NEGATIVE": _combine_slang(EMO_DARK_SLANG),
        "ANGER_EMO": _combine_slang(ANGER_SLANG),
        "NEGATIVE_SOCIAL": _combine_slang(NEGATIVE_SOCIAL_SLANG),
    }


def main() -> int:
    paths = [ROOT / "keywords_traits.xlsx", ROOT / "app" / "keywords_traits.xlsx"]
    categories = _build_phrases()
    new_rows: list[dict[str, str]] = []

    for cat, phrases in categories.items():
        for p in phrases:
            new_rows.append({"Keyword / Phrase": p, "Trait / Kategori": cat})
        print(f"{cat}: +{len(phrases)} frasa slang")

    new_df = pd.DataFrame(new_rows)

    for path in paths:
        if not path.exists():
            print(f"skip {path}")
            continue
        df = pd.read_excel(path)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["Keyword / Phrase", "Trait / Kategori"]
        df["Keyword / Phrase"] = df["Keyword / Phrase"].astype(str).str.lower().str.strip()
        before = len(df)
        merged = pd.concat([df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Keyword / Phrase", "Trait / Kategori"])
        merged.to_excel(path, index=False)
        print(f"saved {path}: {before} -> {len(merged)} (+{len(merged) - before} baru)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
