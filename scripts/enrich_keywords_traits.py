"""
Perkaya keywords_traits.xlsx dengan kata/frasa representatif Big Five.
Jalankan: python scripts/enrich_keywords_traits.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "keywords_traits.xlsx"
BACKUP_PATH = ROOT / "keywords_traits.backup.xlsx"

# Kata generik — jangan tambah sebagai keyword tunggal
GENERIC = frozenset({
    "saya", "aku", "gue", "kamu", "kita", "kami", "dia", "mereka",
    "mudah", "merasa", "rasanya", "rasa", "hal", "banget", "sangat",
    "jadi", "kalau", "kalo", "yang", "ini", "itu", "dan", "atau",
    "bisa", "akan", "sudah", "pernah", "selalu", "kadang", "juga",
    "hari", "besok", "nanti", "sekarang", "dalam", "untuk", "dengan",
    "the", "and", "or", "very", "really", "emosi",
})

# Penambahan kurasi (frasa multi-kata + kata tunggal bermakna psikologis)
ENRICHMENT: dict[str, list[str]] = {
    "ANXIETY_EMO": [
        "mudah terganggu",
        "mudah cemas",
        "mudah panik",
        "mudah khawatir",
        "cemas menghadapi",
        "cemas berat",
        "sangat cemas",
        "merasa cemas",
        "merasa khawatir",
        "merasa gelisah",
        "merasa kewalahan",
        "merasa tidak tenang",
        "pikiran kusut",
        "kepikiran terus",
        "sulit tenang",
        "sulit rileks",
        "stres berat",
        "stres berlebihan",
        "tekanan membebani",
        "overthinking",
        "khawatir berlebihan",
        "takut gagal",
        "takut salah",
        "gelisah",
        "gelisah berlebihan",
        "cemas",
        "khawatir",
        "kewalahan",
        "kacau pikiran",
        "dadaku sesak",
        "jantung berdebar",
    ],
    "SAD_EMO": [
        "merasa sedih",
        "merasa hampa",
        "merasa kosong",
        "merasa menyerah",
        "putus harapan",
        "tidak bersemangat",
        "kehilangan motivasi",
        "menangis sendiri",
        "murung",
        "terpuruk",
        "depresi",
        "melankolis",
        "kesedihan mendalam",
        "merasa tidak berharga",
    ],
    "ANGER_EMO": [
        "mudah marah",
        "mudah emosi",
        "sulit mengendalikan emosi",
        "meledak emosi",
        "naik pitam",
        "kesal berat",
        "jengkel berat",
        "frustrasi berat",
        "marah",
        "jengkel",
        "geram",
        "murka",
    ],
    "MENTAL_UNSTABLE_N": [
        "emosional tidak stabil",
        "mood swing",
        "naik turun emosi",
        "sulit stabil",
        "mental drop",
        "burnout",
        "kelelahan mental",
        "kelelahan emosional",
        "tidak kuat mentally",
        "crisis mental",
    ],
    "NEGATIVE_SOCIAL": [
        "menghindari orang",
        "menarik diri sosial",
        "tidak nyaman sosial",
        "sulit percaya orang",
        "merasa dijauhi",
        "merasa tidak diterima",
        "konflik terus",
        "suka debat",
        "suka konfrontasi",
        "komunikasi toxic",
    ],
    "EXTREME_NEGATIVE": [
        "bunuh diri",
        "ingin bunuh diri",
        "pengen bunuh diri",
        "mau bunuh diri",
        "ingin mati",
        "pengen mati",
        "tidak ingin hidup",
        "tidak mau hidup",
        "melukai diri",
        "menyakiti diri",
        "akhiri hidup",
        "putus asa total",
        "ingin menghilang",
        "hidup tidak berarti",
    ],
    "EMO_POSITIVE": [
        "merasa bahagia",
        "merasa senang",
        "merasa puas",
        "merasa optimis",
        "semangat tinggi",
        "bahagia",
        "senang",
        "optimis",
        "lega",
        "tenang hati",
        "syukur",
        "grateful",
    ],
    "POSITIVE_SOCIAL": [
        "suka bergaul",
        "suka bertemu orang",
        "nyaman di keramaian",
        "aktif bersosialisasi",
        "ramah tamah",
        "mudah akrab",
        "suka ngobrol",
        "hangat ke orang",
        "komunikatif",
        "percaya diri sosial",
    ],
    "EXTRAVERSION_E": [
        "mudah menyesuaikan diri",
        "menyesuaikan diri",
        "lingkungan baru",
        "suka tampil",
        "suka menjadi pusat perhatian",
        "energik di sosial",
        "percaya diri bicara",
        "suka memimpin",
        "suka presentasi",
        "ekstrovert",
        "outgoing",
        "talkative",
        "sosial",
        "energik",
    ],
    "E_SOCIAL_DEPENDENCY": [
        "butuh orang lain",
        "sulit sendirian",
        "nyaman punya teman",
        "suka diajak",
        "suka diajak bicara",
        "merasa kesepian",
        "takut sendiri",
    ],
    "COLLABORATION": [
        "suka kerja tim",
        "suka kolaborasi",
        "kooperatif",
        "gotong royong",
        "suka brainstorming",
        "suka diskusi tim",
        "membantu tim",
    ],
    "RELATIONSHIP_AFFECTION": [
        "sayang",
        "penuh kasih",
        "perhatian",
        "romantis",
        "hangat",
        "affectionate",
        "peduli pasangan",
        "setia",
    ],
    "EMPATHY_HARMONY_A": [
        "empati",
        "peka perasaan",
        "mengerti perasaan",
        "harmonis",
        "menjaga perasaan",
        "tidak suka konflik",
        "damai",
        "sabar",
        "pengertian",
        "mendengarkan",
    ],
    "TRUST": [
        "percaya",
        "saling percaya",
        "jujur",
        "transparan",
        "setia",
        "aman",
        "terbuka",
        "tidak curiga",
    ],
    "CREATIVE_DISCUSSION_A": [
        "suka ide baru",
        "imajinatif",
        "kreatif",
        "inovatif",
        "eksploratif",
        "penasaran",
        "suka bereksperimen",
        "suka diskusi ide",
        "visioner",
        "out of the box",
        "suka belajar hal baru",
        "terbuka ide",
    ],
    "INTROSPECTION": [
        "suka merenung",
        "reflektif",
        "introspeksi",
        "merenungkan diri",
        "mencari makna",
        "filosofis",
        "kontemplatif",
        "jurnal diri",
        "sadar diri",
    ],
    "DISCIPLINE_C": [
        "disiplin",
        "terorganisir",
        "rajin",
        "teliti",
        "sistematis",
        "terencana",
        "tepat waktu",
        "fokus menyelesaikan",
        "bertanggung jawab",
        "konsisten",
        "detail oriented",
        "rajin belajar",
        "target oriented",
    ],
    "ACHIEVEMENT": [
        "suka tantangan",
        "ambisius",
        "berprestasi",
        "suka menang",
        "goal oriented",
        "motivasi tinggi",
        "suka kompetisi",
        "ingin sukses",
        "berorientasi hasil",
    ],
}

# Batch 2 — frasa kontekstual Big Five (editor & klinis ringan)
ENRICHMENT_BATCH2: dict[str, list[str]] = {
    "ANXIETY_EMO": [
        "kepikiran",
        "terus kepikiran",
        "membuat saya kepikiran",
        "membuat saya stres",
        "terganggu hal kecil",
        "hal kecil membuat",
        "cemas menghadapi hari",
        "menghadapi hari esok",
        "hari esok",
        "merasa stres",
        "stress",
        "gugup",
        "was-was",
        "resah",
        "tidak bisa tidur",
        "insomnia ringan",
    ],
    "SAD_EMO": [
        "merasa down",
        "feeling down",
        "tidak ada semangat",
        "lesu",
        "apatis",
        "merasa lelah hidup",
    ],
    "NEGATIVE_SOCIAL": [
        "isolasi sosial",
        "menyendiri",
        "diam di kamar",
        "jarang keluar",
    ],
    "CREATIVE_DISCUSSION_A": [
        "berpikir kreatif",
        "pemikiran terbuka",
        "suka hal baru",
        "open minded",
        "fleksibel berpikir",
    ],
    "INTROSPECTION": [
        "merenung",
        "berpikir dalam",
        "introvert reflektif",
        "jiwa tenang",
    ],
    "DISCIPLINE_C": [
        "manajemen waktu",
        "to do list",
        "deadline",
        "produktif",
        "efisien",
        "terstruktur",
    ],
    "EXTRAVERSION_E": [
        "beradaptasi",
        "adaptif",
        "fleksibel",
        "terbuka lingkungan",
        "suka kenalan",
        "perkenalan baru",
    ],
    "EMPATHY_HARMONY_A": [
        "menjaga hubungan",
        "komunikasi baik",
        "menghargai perasaan",
        "tidak egois",
    ],
    "POSITIVE_SOCIAL": [
        "interaksi sosial",
        "networking",
        "kumpul teman",
        "arisan",
    ],
    "EMO_POSITIVE": [
        "ceria",
        "gembira",
        "antusias",
        "motivasi bangkit",
    ],
    "MENTAL_UNSTABLE_N": [
        "mental health struggle",
        "gangguan emosi",
        "emosional labil",
    ],
}

ENRICHMENT.update({
    trait: ENRICHMENT.get(trait, []) + words
    for trait, words in ENRICHMENT_BATCH2.items()
})


def _normalize_kw(kw: str) -> str:
    return " ".join(kw.strip().lower().split())


def _is_valid_keyword(kw: str) -> bool:
    if not kw or len(kw) < 2:
        return False
    if " " in kw:
        return len(kw) >= 5
    if kw in GENERIC:
        return False
    if len(kw) < 4:
        return False
    return True


def main():
    if not XLSX_PATH.exists():
        raise FileNotFoundError(XLSX_PATH)

    existing = pd.read_excel(XLSX_PATH)
    existing.columns = ["Keyword / Phrase", "Trait / Kategori"][: len(existing.columns)]
    if len(existing.columns) >= 2:
        existing = existing.iloc[:, :2]
        existing.columns = ["Keyword / Phrase", "Trait / Kategori"]

    existing["Keyword / Phrase"] = existing["Keyword / Phrase"].astype(str).map(_normalize_kw)
    existing["Trait / Kategori"] = existing["Trait / Kategori"].astype(str).str.strip()

    seen = set(zip(existing["Trait / Kategori"], existing["Keyword / Phrase"]))
    new_rows = []

    for trait, keywords in ENRICHMENT.items():
        for kw in keywords:
            kw = _normalize_kw(kw)
            if not _is_valid_keyword(kw):
                continue
            key = (trait, kw)
            if key in seen:
                continue
            seen.add(key)
            new_rows.append({"Keyword / Phrase": kw, "Trait / Kategori": trait})

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = existing

    combined = combined.drop_duplicates(subset=["Trait / Kategori", "Keyword / Phrase"])
    combined = combined.sort_values(["Trait / Kategori", "Keyword / Phrase"]).reset_index(drop=True)

    # Backup
    existing.to_excel(BACKUP_PATH, index=False)
    combined.to_excel(XLSX_PATH, index=False)

    print(f"Backup: {BACKUP_PATH}")
    print(f"Sebelum: {len(existing)} baris")
    print(f"Ditambah: {len(new_rows)} baris")
    print(f"Sesudah: {len(combined)} baris")
    print("\nPer kategori:")
    print(combined["Trait / Kategori"].value_counts().to_string())


if __name__ == "__main__":
    main()
