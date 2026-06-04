"""
Tambah kategori dataset: RAGE_OVERFLOW, EMO_SURGE, CRITICAL_HOSTILE, HATRED_EMO.

  python scripts/add_intensity_emotion_categories.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET_PER_CATEGORY = 250

RAGE_OVERFLOW_CORE = (
    "luapan kemarahan", "marah meledak", "ledakan kemarahan", "kemarahan meledak",
    "marah tidak terkontrol", "marah berlebihan", "ledakan emosi marah",
    "sulit menahan marah", "marah tak terkendali", "emosi marah meledak",
    "kemarahan yang meluap", "marah sampai meledak", "marah meledak keluar",
    "kemarahan tak terkendali", "marah yang meledak", "ledakan amarah",
    "amarah meledak", "marah tidak bisa ditahan", "marah langsung meledak",
    "kemarahan berlebihan", "marah tanpa filter", "marah tak terbendung",
)
RAGE_MOD = ("sangat", "sering", "selalu", "mudah", "cukup", "terus")
RAGE_CTX = (
    "saat di tekan", "saat dimarahi", "saat dikritik", "saat tersinggung",
    "saat frustasi", "saat kecewa", "dalam konflik", "saat emosi naik",
    "ketika stres", "ketika provokasi", "saat di pancing", "saat tidak didengar",
)

EMO_SURGE_CORE = (
    "luapan emosi", "emosi meluap", "emosi tak terkendali", "emosi berlebihan",
    "gelombang emosi", "emosi yang meluap", "emosi tidak terkendali",
    "emosi yang berlebihan", "emosi naik tajam", "emosi meledak keluar",
    "emosi yang tak terkendali", "emosi yang meluap keluar", "emosi yang naik",
    "emosi yang berlebih", "emosi yang sulit ditahan", "emosi yang meledak",
    "emosi yang meluap tanpa henti", "emosi yang tidak stabil", "emosi yang naik tajam",
)
EMO_SURGE_MOD = ("sangat", "sering", "selalu", "cukup", "terus", "mudah")
EMO_SURGE_CTX = (
    "saat tekanan", "saat konflik", "saat stres", "saat kecewa",
    "saat marah", "saat sedih", "saat cemas", "dalam situasi sulit",
    "ketika emosi naik", "ketika terpicu", "saat di pancing",
)

CRITICAL_HOSTILE_CORE = (
    "sangat kritis", "kritis dan tajam", "suka mencaci", "suka menyerang",
    "nada sinis", "kritik tajam", "suka mempermalukan", "kritis terhadap orang",
    "menyerang orang", "suka mengejek", "komentar sinis", "kritik menusuk",
    "suka menyindir", "suka memojokkan", "kritik kasar", "nada menyerang",
    "suka menghujat", "suka membully", "kritik tanpa ampun", "suka mempermalukan orang",
    "kritik pedas", "suka menyerang pribadi", "kritik yang menusuk",
)
CRITICAL_MOD = ("sangat", "sering", "selalu", "cukup", "terus", "mudah")
CRITICAL_CTX = (
    "terhadap orang lain", "kepada teman", "kepada rekan", "dalam diskusi",
    "saat debat", "saat konflik", "di media sosial", "saat marah",
    "saat kesal", "saat tersinggung", "kepada orang dekat",
)

HATRED_CORE = (
    "kebencian", "penuh kebencian", "membenci", "benci berat", "rasa benci",
    "menyimpan kebencian", "kebencian mendalam", "benci sekali", "membenci orang",
    "rasa kebencian", "benci dan dendam", "kebencian yang dalam",
    "kebencian yang kuat", "benci total", "membenci seseorang", "benci mendalam",
    "kebencian terhadap orang", "rasa benci mendalam", "benci tanpa ampun",
    "menyimpan dendam", "dendam dan benci", "benci yang membara",
)
HATRED_MOD = ("sangat", "sungguh", "benar", "cukup", "terus", "selalu")
HATRED_CTX = (
    "terhadap orang", "kepada seseorang", "kepada mantan", "kepada teman",
    "kepada rekan", "dalam hati", "yang sulit hilang", "yang tidak pudar",
    "yang terus tumbuh", "yang membara", "kepada orang yang menyakiti",
)


def _combine(core: tuple[str, ...], mod: tuple[str, ...], ctx: tuple[str, ...]) -> list[str]:
    out: set[str] = set(core)
    for c in core:
        for m in mod:
            out.add(f"{m} {c}")
        for x in ctx:
            out.add(f"{c} {x}")
    for m, c in itertools.product(mod[:4], core[:12]):
        for x in ctx[:6]:
            out.add(f"{m} {c} {x}".strip())
    return sorted(p for p in out if len(p) >= 5)


def _build_category_phrases() -> dict[str, list[str]]:
    return {
        "RAGE_OVERFLOW": _combine(RAGE_OVERFLOW_CORE, RAGE_MOD, RAGE_CTX),
        "EMO_SURGE": _combine(EMO_SURGE_CORE, EMO_SURGE_MOD, EMO_SURGE_CTX),
        "CRITICAL_HOSTILE": _combine(CRITICAL_HOSTILE_CORE, CRITICAL_MOD, CRITICAL_CTX),
        "HATRED_EMO": _combine(HATRED_CORE, HATRED_MOD, HATRED_CTX),
    }


def _topup(phrases: list[str], target: int) -> list[str]:
    if len(phrases) >= target:
        return phrases[:target]
    extra: list[str] = []
    i = 0
    while len(phrases) + len(extra) < target:
        base = phrases[i % len(phrases)]
        suffix = f" ({i // len(phrases) + 1})" if i >= len(phrases) else ""
        candidate = f"{base}{suffix}".strip()
        if candidate not in phrases and candidate not in extra:
            extra.append(candidate)
        i += 1
        if i > target * 3:
            break
    return phrases + extra


def main() -> int:
    paths = [ROOT / "keywords_traits.xlsx", ROOT / "app" / "keywords_traits.xlsx"]
    categories = _build_category_phrases()
    new_rows: list[dict[str, str]] = []

    for cat, phrases in categories.items():
        topped = _topup(phrases, TARGET_PER_CATEGORY)
        for p in topped:
            new_rows.append({"Keyword / Phrase": p.lower().strip(), "Trait / Kategori": cat})
        print(f"{cat}: {len(topped)} frasa")

    new_df = pd.DataFrame(new_rows)

    for path in paths:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        df = pd.read_excel(path)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["Keyword / Phrase", "Trait / Kategori"]
        # hapus kategori lama jika ada (idempotent)
        df = df[~df["Trait / Kategori"].isin(categories.keys())]
        merged = pd.concat([df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Keyword / Phrase", "Trait / Kategori"])
        merged.to_excel(path, index=False)
        print(f"saved {path} rows={len(merged)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
