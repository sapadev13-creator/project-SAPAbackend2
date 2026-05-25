"""
Seimbangkan jumlah keyword per kategori di keywords_traits.xlsx.

Target default: 180 entri/kategori (krisis EXTREME_NEGATIVE: 42).
Prioritas saat trim: frasa 2–6 kata, hindari narasi panjang (>55 karakter).

  python scripts/balance_keywords_traits.py
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

TARGET_DEFAULT = 180
TARGET_BY_TRAIT = {
    "EXTREME_NEGATIVE": 42,
}

TRAIT_VARIANT_PREFIX = {
    "E_SOCIAL_DEPENDENCY": "butuh",
    "COLLABORATION": "tim",
    "ACHIEVEMENT": "prestasi",
    "ANGER_EMO": "emosi",
    "EMO_POSITIVE": "merasa",
    "EMO_NEGATIVE": "batin",
    "POSITIVE_SOCIAL": "sosial",
}

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

# Pengisian kategori yang kurang (frasa kurasi, bukan narasi)
BALANCE_FILL: dict[str, list[str]] = {
    "ACHIEVEMENT": [
        "berorientasi prestasi", "target jelas", "suka menyelesaikan tugas",
        "menyelesaikan tepat waktu", "giat bekerja", "fokus pada hasil",
        "kompetitif sehat", "ingin unggul", "berkomitmen penuh", "tekun",
        "gigih", "ulem", "pekerja keras", "produktif tinggi", "disiplin kerja",
        "mencapai milestone", "suka tantangan baru", "growth mindset",
        "berani mengambil risiko", "inisiatif tinggi", "mandiri menyelesaikan",
        "bertanggung jawab tugas", "reliable", "konsisten berprestasi",
        "suka diakui", "bangga pada capaian", "motivasi internal kuat",
        "tidak mudah menyerah", "fokus jangka panjang", "berani bersaing",
        "orientasi sukses", "rajin mengejar mimpi", "suka pencapaian",
        "menyelesaikan proyek", "deadline ketat", "manajemen prioritas",
        "bertanggung jawab hasil", "suka evaluasi diri", "ingin terus naik",
        "ambisi karier", "goal oriented", "high performer", "peak performance",
        "suka ukuran", "benchmark diri", "rajin latihan", "suka sertifikasi",
        "ingin juara", "berani presentasi hasil", "suka feedback kinerja",
        "menyelesaikan target bulanan", "fokus KPI", "suka angka positif",
        "berani memimpin proyek", "driver tim",         "hasil oriented",
        "suka checklist selesai", "progres harian", "habit produktif",
        "prestasi tinggi", "track record bagus", "rekor bagus",
    ],
    "ANGER_EMO": [
        "mudah kesal", "mudah jengkel", "emosi meledak", "sulit menahan amarah",
        "sulit mengendalikan amarah", "sering marah", "sering kesal", "naik darah",
        "emosi naik turun", "sakit hati", "kecewa berat", "frustrasi",
        "kesal berat", "jengkel", "geram", "murka", "emosi negatif",
        "sulit sabar", "reaksi berlebihan", "sulit tenang saat marah",
        "konflik emosional", "emosional meledak", "tidak terkontrol",
        "marah mendadak", "kesal mendadak", "emosi tidak stabil",
        "sulit menerima kritik", "defensif", "sensitif disentuh",
        "mudah tersinggung", "sulit memaafkan", "dendam", "kesal diam-diam",
        "emosi panas", "amarah", "naik emosi",
    ],
    "COLLABORATION": [
        "suka kerja sama", "suka tim", "kooperatif", "gotong royong",
        "suka brainstorming", "suka diskusi tim", "membantu rekan",
        "sharing tugas", "saling mendukung", "tim solid", "sinergi tim",
        "suka proyek bersama", "open collaboration", "team player",
        "suka koordinasi", "suka sinkronisasi", "suka alignment tim",
        "membagi beban", "saling cover", "suka rapat tim",
        "suka workshop tim", "suka agile", "suka scrum", "pair work",
        "suka mentoring tim", "suka knowledge sharing", "suka feedback tim",
        "saling percaya tim", "komunikasi tim baik", "suka kolaborasi lintas divisi",
        "suka proyek kelompok", "suka tim kecil", "suka tim besar",
        "suka dinamika tim", "harmoni tim", "suka membangun tim",
        "suka memfasilitasi", "mediator tim", "penghubung tim",
        "suka delegasi", "suka trust tim", "suka tim lintas fungsi",
        "suka co-create", "suka ide bersama", "suka voting tim",
        "suka konsensus", "suka musyawarah", "suka tim remote",
        "suka hybrid work tim", "suka standup", "suka retrospective",
        "suka OKR tim", "suka KPI tim", "suka celebrate tim",
        "suka tim win", "suka tim spirit", "suka bonding tim",
        "suka outing tim", "suka ice breaking", "suka energizer tim",
        "suka tim building", "suka support rekan", "suka backup rekan",
        "suka kerja bareng", "suka bareng tim", "suka tim kerja",
        "suka rekan kerja", "suka partner kerja", "suka squad",
        "suka unit kerja", "suka divisi", "suka departemen",
        "suka kolega", "suka rekan setim", "suka solid tim",
        "suka tim kompak", "suka tim rapi", "suka tim efektif",
    ],
    "E_SOCIAL_DEPENDENCY": [
        "butuh teman", "butuh orang", "sulit sendiri", "takut sendirian",
        "suka diajak", "suka diajak ngobrol", "suka diajak keluar",
        "merasa kesepian", "kesepian", "butuh pendengar", "butuh curhat",
        "butuh validasi sosial", "butuh dukungan", "butuh hadir orang",
        "suka komunitas", "suka grup", "suka circle", "suka geng",
        "suka tongkrongan", "suka kumpul", "suka nongkrong",
        "suka chat grup", "suka call teman", "suka video call",
        "butuh interaksi", "butuh sosialisasi", "butuh kehadiran",
        "sulit tanpa teman", "sulit tanpa support", "butuh teman dekat",
        "butuh sahabat", "butuh partner ngobrol", "butuh circle dekat",
        "suka hangout", "suka meetup", "suka event sosial",
        "suka pesta", "suka reuni", "suka gathering",
        "butuh energi sosial", "butuh vibe orang", "butuh suasana ramai",
        "sulit isolasi", "sulit karantina sosial", "butuh teman kerja",
        "butuh teman kuliah", "butuh teman main", "butuh support system",
        "butuh emotional support", "butuh social battery", "butuh recharge sosial",
        "suka teman banyak", "suka jaringan", "suka relasi dekat",
        "butuh orang percaya", "butuh orang dekat", "butuh orang mengerti",
        "butuh orang mendengar", "butuh orang hadir", "butuh orang setia",
        "butuh orang hangat", "butuh orang peduli", "butuh orang support",
        "butuh teman ngobrol", "butuh teman curhat", "butuh teman main",
        "butuh teman jalan", "butuh teman makan", "butuh teman nongkrong",
        "butuh teman online", "butuh teman offline", "butuh teman dekat",
        "butuh teman setia", "butuh teman hangat", "butuh teman support",
        "butuh circle teman", "butuh geng teman", "butuh komunitas teman",
    ],
    "EMO_POSITIVE": [
        "merasa bahagia", "merasa senang", "merasa puas", "merasa optimis",
        "semangat tinggi", "bahagia", "senang", "optimis", "lega", "syukur",
        "grateful", "ceria", "gembira", "antusias", "motivasi bangkit",
        "hati senang", "perasaan baik", "mood bagus", "energi positif",
        "positif thinking", "harapan tinggi", "percaya diri", "bangga diri",
        "puas hidup", "nikmati hidup", "menikmati momen", "suka tertawa",
        "suka tersenyum", "ringan hati", "damai batin", "tenang bahagia",
        "merasa diberkati", "merasa beruntung", "merasa cukup",
        "merasa damai", "merasa hangat", "merasa dihargai",
        "merasa dicintai", "merasa diterima", "merasa aman",
        "merasa nyaman", "merasa rileks", "merasa fresh",
        "merasa semangat", "merasa kuat", "merasa siap",
        "merasa berani", "merasa bebas", "merasa ringan",
        "merasa ceria", "merasa gembira", "merasa antusias",
        "merasa motivated", "merasa inspired", "merasa alive",
        "merasa grateful", "merasa blessed", "merasa happy",
        "merasa good", "merasa great", "merasa wonderful",
        "merasa amazing", "merasa fantastic", "merasa excellent",
        "merasa proud", "merasa satisfied", "merasa content",
    ],
    "NEGATIVE_SOCIAL": [
        "menghindari orang", "menarik diri", "isolasi sosial", "diam di kamar",
        "jarang keluar", "sulit bergaul", "sulit sosial", "antisosial ringan",
        "tidak nyaman keramaian", "sulit small talk", "sulit percaya",
        "curiga orang", "waswas sosial", "takut dinilai", "takut ditolak",
        "merasa dijauhi", "merasa tidak diterima", "merasa outcast",
        "konflik sosial", "suka debat", "suka konfrontasi", "komunikasi toxic",
        "sulit kerja sama", "sulit kompromi", "sulit mendengar",
        "sulit empati sosial", "dingin ke orang", "cuek", "jauh dari orang",
        "menjauh dari teman", "memutus kontak", "ghosting",
        "sulit maintain relasi", "sulit jaga teman", "sulit balas chat",
        "menolak undangan", "menolak kumpul", "menolak sosial",
        "sulit adaptasi sosial", "sulit lingkungan baru", "sulit perkenalan",
        "sulit eye contact", "sulit public speaking", "sulit presentasi",
        "sulit networking", "sulit event", "sulit party",
        "sulit keramaian", "sulit crowd", "sulit kerumunan",
        "sulit interaksi", "sulit komunikasi", "sulit dialog",
        "sulit obrolan", "sulit ngobrol", "sulit curhat",
        "sulit terbuka", "sulit trust", "sulit percaya teman",
        "sulit percaya rekan", "sulit percaya pasangan", "sulit percaya keluarga",
    ],
    "POSITIVE_SOCIAL": [
        "suka bergaul", "suka bertemu orang", "nyaman keramaian",
        "aktif bersosialisasi", "ramah tamah", "mudah akrab",
        "suka ngobrol", "hangat ke orang", "komunikatif", "percaya diri sosial",
        "interaksi sosial", "networking", "kumpul teman", "suka kenalan",
        "perkenalan baru", "suka event", "suka meetup", "suka hangout",
        "suka tongkrongan", "suka nongkrong", "suka pesta", "suka reuni",
        "suka gathering", "suka komunitas", "suka grup", "suka circle",
        "suka tim sosial", "suka bonding", "suka ice breaking",
        "suka energizer", "suka tim building", "suka outing",
        "suka arisan", "suka kopi bareng", "suka makan bareng",
        "suka jalan bareng", "suka main bareng", "suka nonton bareng",
        "suka diskusi santai", "suka obrol ringan", "suka humor sosial",
        "suka memecah kebekuan", "suka memulai obrolan", "suka host",
        "suka fasilitator", "suka mediator", "suka penghubung",
        "suka menghubungkan orang", "suka memperkenalkan", "suka connect",
        "suka relasi", "suka pertemanan", "suka persahabatan",
        "suka keluarga besar", "suka komunitas online", "suka forum",
        "suka grup chat", "suka live", "suka streaming sosial",
        "suka kolaborasi sosial", "suka proyek sosial", "suka volunteer",
        "suka sosial work", "suka membantu komunitas", "suka gotong royong sosial",
        "suka senyum", "suka sapa", "suka salam", "suka small talk",
        "suka basa-basi", "suka hangat", "suka ramah", "suka approachable",
    ],
    "INTROSPECTION": [
        "suka merenung", "reflektif", "introspeksi", "merenungkan diri",
        "mencari makna", "filosofis", "kontemplatif", "jurnal diri",
        "sadar diri", "merenung", "berpikir dalam", "jiwa tenang",
        "meditasi ringan", "mindfulness", "suka journaling",
        "suka refleksi", "suka evaluasi diri", "suka introspeksi",
        "suka makna hidup", "suka spiritual ringan", "suka filosofi",
        "suka renungan", "suka hening", "suka sunyi",
        "suka waktu sendiri reflektif", "suka diary", "suka catatan harian",
        "suka self review", "suka self awareness", "suka self growth",
        "suka belajar diri", "suka memahami diri", "suka mengenal diri",
        "suka pertanyaan dalam", "suka why", "suka makna",
        "suka values", "suka prinsip", "suka etika pribadi",
        "suka moral", "suka hati nurani", "suka batin",
        "suka jiwa", "suka rohani ringan", "suka kontemplasi",
        "suka meditasi", "suka breathing", "suka tenang batin",
        "suka damai batin", "suka inner peace", "suka self care mental",
        "suka pause", "suka slow down", "suka hening pikiran",
        "suka clarity", "suka insight", "suka wisdom",
        "suka pemahaman diri", "suka penerimaan diri", "suka pemaafan diri",
        "suka kompas batin", "suka arah hidup", "suka purpose",
        "suka calling", "suka passion dalam", "suka makna kerja",
    ],
    "EXTREME_NEGATIVE": [
        "ingin bunuh diri", "pengen bunuh diri", "mau bunuh diri",
        "ingin bunuh", "pengen bunuh", "mau bunuh", "ingin mati",
        "pengen mati", "mau mati", "tidak ingin hidup", "tidak mau hidup",
        "melukai diri", "menyakiti diri", "akhiri hidup", "akhiri hidupku",
        "ingin menghilang", "ingin hilang saja", "ingin berhenti hidup",
        "tak ingin lanjut hidup", "putus asa ingin mati", "hidup tidak berarti",
        "hidup tak berarti", "lepaskan nyawa", "bunuh diri", "membunuh diri",
        "akan bunuh diri", "rencana bunuh diri", "pikiran bunuh diri",
        "keinginan bunuh diri", "fantasi bunuh diri", "mimpi bunuh diri",
        "dorongan bunuh diri", "urge bunuh diri", "self harm",
        "menyakiti tubuh", "luka diri", "potong diri", "bunuh diri sendiri",
        "ingin mati saja", "pengen mati saja", "mau mati saja",
        "tidak ada alasan hidup", "tidak ada harapan hidup",
        "tidak sanggup hidup", "tak sanggup hidup", "ingin lenyap",
        "ingin berhenti semua", "ingin selesai hidup", "pikiran mati",
    ],
    "EMO_NEGATIVE": [
        "merasa hampa", "merasa kosong", "merasa sepi", "merasa sunyi",
        "merasa pilu", "merasa getir", "merasa perih", "merasa sakit hati",
        "merasa kecewa", "merasa menyesal", "merasa bersalah",
        "merasa tidak berharga", "merasa tidak dicintai", "merasa ditinggal",
        "merasa dikhianati", "merasa ditolak", "merasa diabaikan",
        "perasaan campur aduk", "perasaan berat", "perasaan gelap",
        "hati berat", "jiwa berat", "beban emosional", "beban batin",
        "luka batin", "luka emosional", "trauma ringan", "luka lama",
        "kenangan menyakitkan", "rindu menyakitkan", "kehilangan menyakitkan",
        "duka", "berkabung", "nestapa", "tangis", "isak",
        "merasa rapuh", "merasa rentan", "merasa lemah",
        "merasa tidak kuat", "merasa goyah", "merasa runtuh ringan",
        "merasa tidak ada harapan ringan", "merasa suram",
        "merasa kelam", "merasa muram", "merasa redup",
        "merasa pudar", "merasa hilang arah ringan",         "merasa tersesat",
    ],
}

# Frasa unik per kategori (hindari bentrok antar-sheet saat dedupe)
TOP_UP_UNIQUE: dict[str, list[str]] = {
    "E_SOCIAL_DEPENDENCY": [
        "ketergantungan sosial", "afiliasi sosial tinggi", "butuh validasi teman",
        "butuh teman setiap hari", "butuh interaksi harian", "butuh obrolan harian",
        "butuh teman dekat rutin", "butuh dukungan sosial rutin", "butuh hadir teman",
        "butuh teman saat down", "butuh teman saat stres", "butuh circle dekat",
        "butuh komunitas dekat", "butuh relasi dekat", "butuh bonding teman",
        "butuh hangout rutin", "butuh nongkrong rutin", "butuh kumpul rutin",
        "butuh sosialisasi rutin", "butuh energi dari teman",
        "butuh teman curhat malam", "butuh teman saat sendirian",
        "butuh teman saat galau", "butuh teman saat cemas",
        "butuh teman untuk pulih", "butuh teman untuk tenang",
        "butuh orang saat down", "butuh orang saat galau",
        "butuh orang saat sendirian", "butuh orang setiap minggu",
        "butuh jaringan dekat", "butuh komunitas hangat",
        "butuh geng dekat", "butuh squad dekat", "butuh circle hangat",
        "butuh relasi hangat", "butuh ikatan sosial kuat",
        "butuh kebersamaan", "butuh kehadiran teman",
    ],
    "EMO_NEGATIVE": [
        "batin merasa kosong", "batin merasa hampa", "batin merasa sunyi",
        "batin merasa pilu", "batin merasa perih", "batin duka mendalam",
        "batin luka lama", "batin trauma ringan", "batin beban berat",
        "batin gelap", "batin muram", "batin suram", "batin lesu",
        "batin rapuh", "batin rentan", "batin goyah", "batin lemah",
        "batin tidak stabil", "batin kacau", "batin kusut",
    ],
    "POSITIVE_SOCIAL": [
        "sosial aktif sekali", "sosial ramah sekali", "sosial hangat sekali",
        "sosial komunikatif", "sosial percaya diri", "sosial energik",
        "sosial suka kenalan", "sosial suka networking", "sosial suka event",
        "sosial suka meetup", "sosial suka hangout", "sosial suka kumpul",
        "sosial suka pesta", "sosial suka reuni", "sosial suka gathering",
        "sosial suka komunitas", "sosial suka grup", "sosial suka circle",
        "sosial suka bonding", "sosial suka ice breaking",
        "sosial suka ngobrol", "sosial suka bertemu", "sosial suka ramah",
        "sosial suka akrab", "sosial suka interaksi",
    ],
    "COLLABORATION": [
        "tim kolaboratif", "tim kooperatif", "tim sinergi", "tim solid",
        "tim kompak", "tim efektif", "tim produktif", "tim harmonis",
        "tim komunikatif", "tim saling dukung", "tim saling bantu",
        "tim brainstorming", "tim diskusi", "tim workshop", "tim proyek",
        "tim lintas fungsi", "tim agile", "tim scrum", "tim remote",
        "tim hybrid",
    ],
    "ACHIEVEMENT": [
        "prestasi konsisten", "prestasi meningkat", "prestasi unggul",
        "prestasi terbaik", "prestasi tinggi", "prestasi gemilang",
        "prestasi membanggakan", "prestasi memuaskan", "prestasi signifikan",
        "prestasi berkelanjutan", "prestasi kerja bagus", "prestasi akademik",
    ],
    "ANGER_EMO": [
        "emosi marah cepat", "emosi kesal cepat", "emosi meledak cepat",
        "emosi panas cepat", "emosi naik cepat",
    ],
    "EMO_POSITIVE": [
        "merasa ceria sekali", "merasa bahagia sekali", "merasa optimis sekali",
        "merasa puas sekali", "merasa lega sekali",
    ],
    "COLLABORATION": [
        "tim brainstorming rutin", "tim diskusi rutin", "tim proyek rutin",
    ],
}

GENERIC_DROP = frozenset({
    "tim", "iba", "kamu", "kita", "kami", "dia", "lo", "lu", "gue", "gw",
    "the", "and", "or", "yang", "ini", "itu", "dan", "atau", "bisa", "akan",
    "hari", "besok", "nanti", "sekarang", "emosi", "hal", "sesuatu",
})

CRISIS_EXTREME_PHRASES = frozenset({
    "bunuh diri", "ingin bunuh diri", "pengen bunuh diri", "mau bunuh diri",
    "ingin bunuh", "pengen bunuh", "mau bunuh", "ingin mati", "pengen mati",
    "mau mati", "tidak ingin hidup", "tidak mau hidup", "melukai diri",
    "menyakiti diri", "akhiri hidup", "akhiri hidupku", "lepaskan nyawa",
    "putus asa ingin mati", "hidup tidak berarti", "hidup tak berarti",
    "ingin berhenti hidup", "tak ingin lanjut hidup", "ingin menghilang",
    "ingin hilang saja", "membunuh diri", "akan bunuh diri",
})


def _normalize(kw: str) -> str:
    return " ".join(str(kw).strip().lower().split())


def _is_crisis_extreme(kw: str) -> bool:
    if kw in CRISIS_EXTREME_PHRASES:
        return True
    return any(x in kw for x in ("bunuh diri", "ingin bunuh", "pengen bunuh", "mau bunuh", "ingin mati", "pengen mati"))


def _reassign_extreme(kw: str) -> str | None:
    if kw in GENERIC_DROP:
        return None
    return None


def _quality_score(kw: str, trait: str) -> float:
    n_words = len(kw.split())
    length = len(kw)
    score = 0.0

    if kw in GENERIC_DROP:
        return -100.0
    if trait == "EXTREME_NEGATIVE" and not _is_crisis_extreme(kw):
        return -50.0

    if 2 <= n_words <= 6:
        score += 4.0
    elif n_words == 1 and 4 <= length <= 20:
        score += 1.5
    elif n_words == 1:
        score += 0.5

    if 8 <= length <= 48:
        score += 2.0
    if length > 55:
        score -= 6.0
    if length > 80:
        score -= 10.0

    if kw.startswith(("aku ", "ada ", "kenapa ", "baru ", "setiap ")):
        score -= 5.0
    if "timsenang" in kw or "  " in kw:
        score -= 8.0

    if trait in ("ANXIETY_EMO", "SAD_EMO", "MENTAL_UNSTABLE_N") and n_words >= 2:
        score += 0.5
    if trait in ("EXTRAVERSION_E", "POSITIVE_SOCIAL", "COLLABORATION") and n_words >= 2:
        score += 0.5

    return score


def _target_for(trait: str) -> int:
    return TARGET_BY_TRAIT.get(trait, TARGET_DEFAULT)


def _trim_group(sub: pd.DataFrame, target: int) -> pd.DataFrame:
    if len(sub) <= target:
        return sub
    sub = sub.copy()
    sub["_score"] = sub[KW_COL].map(lambda k: _quality_score(k, sub[TR_COL].iloc[0]))
    sub = sub.sort_values("_score", ascending=False)
    return sub.drop(columns="_score").head(target)


def _fill_group(sub: pd.DataFrame, trait: str, target: int) -> pd.DataFrame:
    need = target - len(sub)
    if need <= 0:
        return sub
    existing = set(sub[KW_COL])
    new_rows = []
    pools = list(TOP_UP_UNIQUE.get(trait, []))
    raw_fill = BALANCE_FILL.get(trait, [])
    if trait == "E_SOCIAL_DEPENDENCY":
        pools.extend(p for p in raw_fill if p.startswith("butuh") or "ketergantungan" in p)
    elif trait == "POSITIVE_SOCIAL":
        pools.extend(p for p in raw_fill if not p.startswith("butuh"))
    elif trait == "EMO_NEGATIVE":
        pools.extend(p for p in raw_fill if p.startswith(("batin ", "merasa ")))
    elif trait == "COLLABORATION":
        pools.extend(p for p in raw_fill if "tim" in p or "kolabor" in p or "kooper" in p)
    elif trait == "ACHIEVEMENT":
        pools.extend(p for p in raw_fill if "prestasi" in p or p in raw_fill[:20])
    elif trait == "ANGER_EMO":
        pools.extend(p for p in raw_fill if p.startswith("emosi ") or "marah" in p)
    elif trait == "EMO_POSITIVE":
        pools.extend(p for p in raw_fill if p.startswith("merasa "))
    else:
        pools.extend(raw_fill)

    for kw in pools:
        kw = _normalize(kw)
        if trait == "EXTREME_NEGATIVE":
            if not _is_crisis_extreme(kw):
                continue
        elif _reassign_extreme(kw):
            continue
        if kw in existing or kw in GENERIC_DROP or len(kw) < 3:
            continue
        new_rows.append({KW_COL: kw, TR_COL: trait})
        existing.add(kw)
        if len(sub) + len(new_rows) >= target:
            break
    if new_rows:
        sub = pd.concat([sub, pd.DataFrame(new_rows)], ignore_index=True)

    combined = sub
    if len(combined) < target:
        prefix = TRAIT_VARIANT_PREFIX.get(trait)
        if prefix:
            extra = []
            exist = set(combined[KW_COL])
            for kw in combined[KW_COL]:
                if len(combined) + len(extra) >= target:
                    break
                if len(kw.split()) >= 2 and not kw.startswith(f"{prefix} "):
                    variant = _normalize(f"{prefix} {kw}")
                    if variant not in exist and variant not in GENERIC_DROP and len(variant) <= 55:
                        extra.append({KW_COL: variant, TR_COL: trait})
                        exist.add(variant)
            if extra:
                combined = pd.concat([combined, pd.DataFrame(extra)], ignore_index=True)

    return combined.head(target)


def _owner_trait(kw: str, traits: list[str]) -> str:
    """Pilih kategori pemilik keyword jika bentrok (frasa prefix khas)."""
    unique = list(dict.fromkeys(traits))
    rules: list[tuple[str, str]] = [
        ("bunuh", "EXTREME_NEGATIVE"),
        ("ingin mati", "EXTREME_NEGATIVE"),
        ("pengen mati", "EXTREME_NEGATIVE"),
        ("melukai diri", "EXTREME_NEGATIVE"),
        ("ketergantungan sosial", "E_SOCIAL_DEPENDENCY"),
        ("afiliasi sosial", "E_SOCIAL_DEPENDENCY"),
        ("tim kolaboratif", "COLLABORATION"),
        ("tim kooperatif", "COLLABORATION"),
        ("tim ", "COLLABORATION"),
        ("prestasi ", "ACHIEVEMENT"),
        ("batin ", "EMO_NEGATIVE"),
        ("sosial ", "POSITIVE_SOCIAL"),
        ("emosi marah", "ANGER_EMO"),
        ("emosi kesal", "ANGER_EMO"),
        ("emosi meledak", "ANGER_EMO"),
    ]
    for needle, owner in rules:
        if needle in kw and owner in unique:
            return owner
    if kw.startswith("butuh") and "E_SOCIAL_DEPENDENCY" in unique:
        return "E_SOCIAL_DEPENDENCY"
    for t in TRAIT_PRIORITY:
        if t in unique:
            return t
    return unique[0]


def _dedupe_cross_trait(df: pd.DataFrame) -> pd.DataFrame:
    by_kw: dict[str, list[str]] = {}
    for kw, tr in zip(df[KW_COL], df[TR_COL]):
        by_kw.setdefault(kw, []).append(tr)
    rows = []
    for kw, traits in by_kw.items():
        rows.append({KW_COL: kw, TR_COL: _owner_trait(kw, traits)})
    return pd.DataFrame(rows)


def balance_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[KW_COL] = df[KW_COL].astype(str).map(_normalize)
    df[TR_COL] = df[TR_COL].astype(str).str.strip()
    df = df[~df[KW_COL].isin(GENERIC_DROP)]

    parts = []
    stats = {"before": {}, "after": {}}

    for trait in sorted(df[TR_COL].unique()):
        sub = df[df[TR_COL] == trait][[KW_COL, TR_COL]].drop_duplicates(subset=[KW_COL])
        stats["before"][trait] = len(sub)
        target = _target_for(trait)
        sub = _trim_group(sub, target)
        sub = _fill_group(sub, trait, target)
        stats["after"][trait] = len(sub)
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    out = _dedupe_cross_trait(out)

    # Isi ulang kategori yang turun di bawah target setelah dedupe (max 3 putaran)
    for _ in range(3):
        changed = False
        for trait in sorted(out[TR_COL].unique()):
            target = _target_for(trait)
            sub = out[out[TR_COL] == trait]
            if len(sub) < target:
                filled = _fill_group(sub[[KW_COL, TR_COL]], trait, target)
                out = pd.concat([
                    out[out[TR_COL] != trait],
                    filled,
                ], ignore_index=True)
                changed = True
        out = _dedupe_cross_trait(out)
        if not changed:
            break
    out = out.sort_values([TR_COL, KW_COL]).reset_index(drop=True)

    for trait in out[TR_COL].unique():
        stats["after"][trait] = int((out[TR_COL] == trait).sum())
    return out, stats


def main():
    if not XLSX.exists():
        raise FileNotFoundError(XLSX)

    df = pd.read_excel(XLSX)
    df = df.iloc[:, :2]
    df.columns = [KW_COL, TR_COL]

    pd.read_excel(XLSX).to_excel(BACKUP, index=False)
    balanced, stats = balance_dataframe(df)

    # Top-up akhir jika masih di bawah target
    for _ in range(2):
        for trait in sorted(balanced[TR_COL].unique()):
            target = _target_for(trait)
            sub = balanced[balanced[TR_COL] == trait]
            if len(sub) < target:
                filled = _fill_group(sub[[KW_COL, TR_COL]], trait, target)
                balanced = pd.concat([
                    balanced[balanced[TR_COL] != trait],
                    filled,
                ], ignore_index=True)
        balanced = _dedupe_cross_trait(balanced)

    balanced = balanced.sort_values([TR_COL, KW_COL]).reset_index(drop=True)
    balanced.to_excel(XLSX, index=False)
    if APP_COPY.parent.exists():
        balanced.to_excel(APP_COPY, index=False)

    print("=== balance_keywords_traits ===")
    print(f"Target umum: {TARGET_DEFAULT} | EXTREME_NEGATIVE: {TARGET_BY_TRAIT['EXTREME_NEGATIVE']}")
    print(f"Total: {len(df)} -> {len(balanced)}\n")
    print("Per kategori (sesudah):")
    for trait in sorted(stats["after"]):
        b, a = stats["before"][trait], stats["after"][trait]
        mark = "+" if a > b else ("-" if a < b else "=")
        print(f"  {trait}: {b} -> {a} ({mark}{abs(a-b)})")


if __name__ == "__main__":
    main()
