"""Generator frasa keyword per trait — untuk enrich_keywords_to_target.py."""

from __future__ import annotations

import itertools
import random

from sapa_api.text_utils import EMPATHY_VALIDATION_PHRASES, is_meaningful_token

GENERIC = frozenset({
    "saya", "aku", "gue", "kamu", "yang", "ini", "itu", "dan", "atau",
    "bisa", "akan", "sudah", "juga", "hari", "besok", "nanti", "hal",
    "the", "and", "or", "banget", "sangat", "sekali", "memang",
})

NARRATIVE_PREFIXES = ("aku ", "ada ", "kenapa ", "baru bangun ", "gimana kalau ")

# --- Vocabulary pools ---

ANXIETY_CORE = (
    "cemas", "khawatir", "gelisah", "panik", "stres", "stress", "takut",
    "resah", "was-was", "gugup", "deg-degan", "overthinking", "kewalahan",
    "tertekan", "gelisah berat", "cemas berat", "stres berat", "khawatir berat",
)
ANXIETY_MOD = ("mudah", "sering", "selalu", "terus", "sangat", "cukup", "agak")
ANXIETY_CTX = (
    "hari esok", "masa depan", "presentasi", "ujian", "wawancara", "pekerjaan",
    "keuangan", "kesehatan", "hubungan", "perubahan", "ketidakpastian",
    "hal kecil", "situasi baru", "keramaian", "deadline", "tugas menumpuk",
)
ANXIETY_VERB = (
    "menghadapi", "memikirkan", "mengkhawatirkan", "membayangkan",
)

SAD_CORE = (
    "sedih", "murung", "hampa", "kosong", "lesu", "apatis", "terpuruk",
    "putus harapan", "down", "melankolis", "menyerah", "kecewa berat",
    "kehilangan semangat", "tidak bersemangat", "merasa sedih", "merasa hampa",
)
SAD_CTX = (
    "hidup", "diri sendiri", "masa lalu", "hubungan", "pekerjaan",
    "hari-hari", "pagi hari", "malam hari", "sendirian",
)

ANGER_CORE = (
    "marah", "kesal", "jengkel", "geram", "murka", "frustasi", "naik pitam",
    "emosional", "tersinggung", "sakit hati", "dendam", "kesal berat",
)
ANGER_MOD = ("mudah", "sering", "selalu", "sangat", "cukup")

POSITIVE_EMO = (
    "bahagia", "senang", "gembira", "ceria", "optimis", "lega", "puas",
    "syukur", "grateful", "antusias", "semangat", "bangga", "tenang hati",
    "ringan hati", "merasa bahagia", "merasa senang", "merasa puas",
)

EMPATHY_PHRASES = list(EMPATHY_VALIDATION_PHRASES) + [
    "empati", "peka perasaan", "mengerti perasaan", "harmonis", "damai",
    "sabar", "pengertian", "mendengarkan", "validasi emosi", "menjaga perasaan",
    "tidak suka konflik", "komunikasi hangat", "perhatian tulus", "kasih sayang",
    "hangat", "lembut", "menerima perasaan", "menghargai perasaan",
]

EXTRAVERSION = (
    "sosial", "ekstrovert", "outgoing", "komunikatif", "ramah", "percaya diri",
    "aktif", "energik", "suka bergaul", "suka ngobrol", "suka kenalan",
    "mudah akrab", "mudah berinteraksi", "percaya diri sosial", "aktif bersosialisasi",
)

CREATIVE = (
    "kreatif", "imajinatif", "inovatif", "penasaran", "eksploratif", "visioner",
    "ide baru", "suka ide baru", "open minded", "out of the box", "berpikir kreatif",
    "suka bereksperimen", "suka diskusi ide", "fleksibel berpikir",
)

DISCIPLINE = (
    "disiplin", "terorganisir", "rajin", "teliti", "sistematis", "terencana",
    "tepat waktu", "konsisten", "produktif", "efisien", "terstruktur",
    "bertanggung jawab", "fokus", "detail oriented", "manajemen waktu",
)

ACHIEVEMENT = (
    "ambisius", "berprestasi", "bertekad", "gigih", "tekun", "motivasi tinggi",
    "goal oriented", "suka tantangan", "suka kompetisi", "ingin sukses",
    "berorientasi hasil", "prestasi", "target oriented", "high performer",
)

CRISIS_ONLY = (
    "bunuh diri", "ingin bunuh diri", "pengen bunuh diri", "mau bunuh diri",
    "ingin mati", "pengen mati", "mau mati", "tidak ingin hidup", "tidak mau hidup",
    "melukai diri", "menyakiti diri", "akhiri hidup", "ingin menghilang",
    "hidup tidak berarti", "putus asa ingin mati", "ingin berhenti hidup",
)


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _ok(kw: str) -> bool:
    if not kw or len(kw) < 3:
        return False
    if kw in GENERIC:
        return False
    if kw.startswith(NARRATIVE_PREFIXES):
        return False
    if len(kw) > 52:
        return False
    if " " not in kw and not is_meaningful_token(kw):
        return False
    return True


def _fill_from_iter(phrases: set[str], target: int, iterator, existing: set[str]):
    for p in iterator:
        if len(phrases) >= target:
            break
        p = _norm(p)
        if not _ok(p) or p in existing or p in phrases:
            continue
        phrases.add(p)


def generate_anxiety(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    templates = (
        "{m} {c}", "merasa {c}", "{c} saat {ctx}", "{c} karena {ctx}",
        "khawatir tentang {ctx}", "cemas menghadapi {ctx}", "stres saat {ctx}",
        "sulit tenang saat {ctx}", "tidak tenang saat {ctx}", "kepikiran {ctx}",
        "overthinking tentang {ctx}", "takut akan {ctx}", "panik saat {ctx}",
        "gelisah menghadapi {ctx}", "resah karena {ctx}", "was-was soal {ctx}",
    )
    calm_issues = ("tenang", "tidur", "fokus", "rileks", "berpikir jernih")
    it = []
    for tpl, m, c, ctx in itertools.product(
        templates, ANXIETY_MOD, ANXIETY_CORE, ANXIETY_CTX
    ):
        it.append(tpl.format(m=m, c=c, ctx=ctx))
    for c, ctx, v in itertools.product(ANXIETY_CORE, ANXIETY_CTX, calm_issues):
        it.append(f"sulit {v} saat {ctx}")
        it.append(f"tidak bisa {v} karena {c}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_sad(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    mods = ("merasa", "selalu", "sering", "sangat", "cukup", "terus")
    templates = (
        "{m} {c}", "{c} karena {ctx}", "{c} saat {ctx}", "kondisi {c}",
        "nuansa {c}", "fase {c}", "fase {c} mendalam",
    )
    it = []
    for tpl, m, c, ctx in itertools.product(templates, mods, SAD_CORE, SAD_CTX):
        it.append(tpl.format(m=m, c=c, ctx=ctx))
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_anger(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    it = list(ANGER_CORE)
    for c in ANGER_CORE:
        for m in ANGER_MOD:
            it.append(f"{m} {c}")
        it.append(f"sulit mengendalikan {c}")
        it.append(f"emosi {c}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_simple_pool(
    pool: tuple[str, ...],
    existing: set[str],
    target: int,
    prefixes: tuple[str, ...] = ("suka", "mudah", "sering", "sangat", "cukup"),
    suffixes: tuple[str, ...] = ("tinggi", "banget", "sekali", "dalam", "penuh"),
    trait_prefix: str = "",
) -> list[str]:
    out: set[str] = set()
    it = list(pool)
    for p in pool:
        for pre in prefixes:
            if pre not in p:
                it.append(f"{pre} {p}")
        for suf in suffixes:
            if suf not in p:
                it.append(f"{p} {suf}")
        if trait_prefix:
            it.append(f"{trait_prefix} {p}")
    for a, b in itertools.product(prefixes, pool):
        it.append(f"{a} {b}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_empathy(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    it = list(EMPATHY_PHRASES)
    objs = ("perasaan", "emosi", "hati", "pikiran", "diri")
    acts = ("didengarkan", "divalidasi", "diterima", "dipahami", "dihargai")
    for o, a in itertools.product(objs, acts):
        it.append(f"{o} yang {a}")
        it.append(f"merasa {a}")
        it.append(f"{o} {a} terasa lebih ringan")
    for pre in ("sangat", "merasa", "butuh"):
        for p in EMPATHY_PHRASES[:30]:
            it.append(f"{pre} {p}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_crisis(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    it = list(CRISIS_ONLY)
    intents = ("ingin", "pengen", "mau", "akan")
    for i in intents:
        it.append(f"{i} bunuh diri")
        it.append(f"{i} mati")
    random.shuffle(it)
    _fill_from_iter(out, min(target, 120), it, existing)
    return sorted(out)


def generate_collaboration(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    bases = ("kerja tim", "kolaborasi", "kooperatif", "gotong royong", "tim", "squad")
    acts = ("brainstorming", "diskusi", "proyek", "rapat", "sinkronisasi", "koordinasi")
    it = []
    for b in bases:
        it.append(b)
        for a in acts:
            it.append(f"suka {b} {a}")
            it.append(f"{b} {a}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_social_negative(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    it = []
    verbs = ("menghindari", "menarik diri", "menolak", "menjauh dari")
    objs = ("orang", "keramaian", "sosial", "pertemuan", "undangan", "interaksi")
    for v, o in itertools.product(verbs, objs):
        it.append(f"{v} {o}")
    extras = (
        "isolasi sosial", "antisosial", "sulit bergaul", "sulit percaya orang",
        "tidak nyaman sosial", "sulit small talk", "konflik sosial",
    )
    it.extend(extras)
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_mental_unstable(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    pool = (
        "mood swing", "emosional labil", "naik turun emosi", "burnout",
        "kelelahan mental", "kelelahan emosional", "mental drop", "crisis mental",
        "emosional tidak stabil", "sulit stabil", "gangguan emosi",
        "mental health struggle", "labil emosional", "emosional meledak",
    )
    it = list(pool)
    for p in pool:
        it.append(f"sering {p}")
        it.append(f"mengalami {p}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_emo_negative(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    cores = (
        "hampa", "kosong", "pilu", "perih", "duka", "nestapa", "rapuh", "rentan",
        "beban emosional", "luka batin", "trauma", "rindu", "kehilangan",
    )
    it = []
    for c in cores:
        it.append(f"merasa {c}")
        it.append(f"batin {c}")
        it.append(f"perasaan {c}")
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_relationship(existing: set[str], target: int) -> list[str]:
    pool = (
        "sayang", "kasih", "cinta", "romantis", "setia", "perhatian", "hangat",
        "affectionate", "peduli pasangan", "dekat", "mesra", "penyayang",
    )
    return generate_simple_pool(pool, existing, target, ("sangat", "penuh", "merasa"))


def generate_trust(existing: set[str], target: int) -> list[str]:
    pool = (
        "percaya", "jujur", "transparan", "setia", "aman", "terbuka", "loyal",
        "saling percaya", "tidak curiga", "integritas", "konsisten",
    )
    return generate_simple_pool(pool, existing, target, ("sangat", "saling", "mudah"))


def generate_esocial_dep(existing: set[str], target: int) -> list[str]:
    out: set[str] = set()
    it = []
    for o in ("teman", "orang", "dukungan", "interaksi", "komunitas", "circle"):
        it.append(f"butuh {o}")
        it.append(f"butuh {o} dekat")
        it.append(f"sulit tanpa {o}")
    extras = (
        "ketergantungan sosial", "kesepian", "butuh validasi", "butuh curhat",
        "butuh pendengar", "afiliasi sosial",
    )
    it.extend(extras)
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_introspection(existing: set[str], target: int) -> list[str]:
    pool = (
        "introspeksi", "reflektif", "merenung", "kontemplatif", "filosofis",
        "meditasi", "mindfulness", "jurnal diri", "sadar diri", "makna hidup",
        "refleksi diri", "berpikir dalam", "jiwa tenang",
    )
    return generate_simple_pool(pool, existing, target, ("suka", "sering", "cenderung"))


PLACES = (
    "kantor", "rumah", "sekolah", "kampus", "kota", "desa", "kafe",
    "tempat kerja", "lingkungan baru", "keramaian", "rapat", "kelas",
    "lab", "studio", "pasar", "jalan", "komunitas", "forum", "grup",
    "tim", "divisi", "unit", "proyek", "presentasi", "wawancara", "ujian",
)
TIMES = (
    "pagi", "siang", "sore", "malam", "senin", "jumat", "akhir pekan",
    "bulan ini", "minggu ini", "hari ini", "besok", "nanti", "sering",
    "kadang", "selalu", "jarang", "awal bulan", "akhir tahun",
)
MODIFIERS = (
    "cukup", "agak", "sedikit", "sangat", "terlalu", "cenderung",
    "sering", "jarang", "selalu", "kadang", "mudah", "sulit",
)


def generate_massive_marker(
    trait: str,
    markers: tuple[str, ...],
    existing: set[str],
    target: int,
) -> list[str]:
    """Kombinasi marker + tempat + waktu untuk mencapai ribuan frasa unik."""
    out: set[str] = set()
    templates = (
        "{m} saat {t}", "{m} di {p}", "{m} {t} di {p}",
        "pengalaman {m} {t}", "kondisi {m} di {p}",
        "{mod} {m} saat {t}", "{mod} {m} di {p}",
        "merasa {m} saat {t}", "situasi {m} di {p}",
    )
    it = []
    for tpl, m, p, t, mod in itertools.product(
        templates, markers[:6], PLACES, TIMES, MODIFIERS[:8]
    ):
        it.append(tpl.format(m=m, p=p, t=t, mod=mod))
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    return sorted(out)


def generate_bulk_exclusive(
    trait: str,
    existing: set[str],
    target: int,
) -> list[str]:
    """Kombinasi frasa dengan marker eksklusif per trait → minim bentrok global."""
    cfg = BULK_CONFIG.get(trait)
    if not cfg:
        return []
    out: set[str] = set()
    it = []
    markers = cfg["markers"]
    for tpl in cfg["templates"]:
        for m in markers:
            for x in cfg.get("x", ("")) or ("",):
                for y in cfg.get("y", ("")) or ("",):
                    try:
                        kw = _norm(tpl.format(m=m, x=x, y=y, a="", b=""))
                    except KeyError:
                        kw = _norm(tpl.format(m=m))
                    if m in kw and _ok(kw):
                        it.append(kw)
    for m, x, y in itertools.product(
        markers, cfg.get("x", ()), cfg.get("y", ())
    ):
        it.append(_norm(f"{m} {x} {y}"))
        it.append(_norm(f"{x} {m} {y}"))
        it.append(_norm(f"suka {m} {y}"))
    random.shuffle(it)
    _fill_from_iter(out, target, it, existing)
    if len(out) < target:
        extra = generate_massive_marker(trait, tuple(markers), existing | out, target)
        out.update(extra[: target - len(out)])
    return sorted(out)


# Marker wajib dalam frasa agar tidak rebut keyword antar-kategori
TRAIT_MARKERS: dict[str, tuple[str, ...]] = {
    "ACHIEVEMENT": ("prestasi", "ambisi", "target", "sukses", "kompetisi", "capaian"),
    "DISCIPLINE_C": ("disiplin", "terorganisir", "sistematis", "konsisten", "teliti"),
    "CREATIVE_DISCUSSION_A": ("kreatif", "imajinatif", "inovatif", "ide baru", "visioner"),
    "INTROSPECTION": ("refleksi", "introspeksi", "merenung", "kontemplasi", "makna"),
    "EXTRAVERSION_E": ("ekstrovert", "outgoing", "sosial aktif", "energik", "ramah"),
    "POSITIVE_SOCIAL": ("sosial", "interaksi", "jaringan", "komunitas", "networking"),
    "COLLABORATION": ("kolaborasi", "kooperatif", "gotong royong", "sinergi tim"),
    "TRUST": ("percaya", "kepercayaan", "transparan", "jujur", "loyal"),
    "RELATIONSHIP_AFFECTION": ("kasih", "sayang", "romantis", "mesra", "cinta"),
    "ANGER_EMO": ("marah", "kesal", "jengkel", "amarah", "frustasi"),
    "MENTAL_UNSTABLE_N": ("labil", "burnout", "mood swing", "kelelahan mental"),
    "NEGATIVE_SOCIAL": ("isolasi", "menghindar", "antisosial", "menarik diri"),
    "EMO_POSITIVE": ("bahagia", "senang", "gembira", "ceria", "optimis"),
    "EMO_NEGATIVE": ("hampa", "pilu", "duka", "nestapa", "perih"),
    "E_SOCIAL_DEPENDENCY": ("ketergantungan sosial", "butuh teman", "kesepian"),
    "EMPATHY_HARMONY_A": ("empati", "validasi", "harmonis", "peka", "mendengarkan"),
    "SAD_EMO": ("sedih", "murung", "hampa", "apatis", "terpuruk"),
}

BULK_CONFIG: dict[str, dict] = {
    "ACHIEVEMENT": {
        "markers": ("prestasi", "ambisi", "target", "sukses", "kompetisi", "karier", "capaian"),
        "x": ("tinggi", "besar", "utama", "bulanan", "tahunan", "profesional", "akademik"),
        "y": ("kerja", "proyek", "tim", "diri", "organisasi", "bisnis", "studi"),
        "templates": ("{m} {y}", "orientasi {m}", "fokus {m} {y}", "motivasi {m}"),
    },
    "DISCIPLINE_C": {
        "markers": ("disiplin", "terorganisir", "sistematis", "konsisten", "teliti", "rajin"),
        "x": ("kerja", "belajar", "harian", "jadwal", "proses", "kebiasaan"),
        "y": ("tinggi", "kuat", "baik", "rutin", "mapan", "terjaga"),
        "templates": ("{m} dalam {x}", "{m} saat {x}", "kebiasaan {m}"),
    },
    "CREATIVE_DISCUSSION_A": {
        "markers": ("kreatif", "imajinatif", "inovatif", "ide", "visioner", "eksploratif"),
        "x": ("baru", "segara", "beda", "unik", "radikal", "fresh"),
        "y": ("solusi", "masalah", "produk", "diskusi", "proyek", "konsep"),
        "templates": ("{m} {y}", "pendekatan {m}", "pikiran {m}"),
    },
    "INTROSPECTION": {
        "markers": ("refleksi", "introspeksi", "merenung", "kontemplasi", "makna", "jurnal"),
        "x": ("diri", "hati", "pikiran", "hidup", "pengalaman", "nilai"),
        "y": ("dalam", "tenang", "malam", "pagi", "rutin", "menyendiri"),
        "templates": ("{m} {x}", "waktu {m}", "latihan {m}"),
    },
    "EXTRAVERSION_E": {
        "markers": ("ekstrovert", "outgoing", "sosial aktif", "energik", "talkative", "ramah"),
        "x": ("orang", "keramaian", "acara", "panggung", "obrolan", "kenalan"),
        "y": ("suka", "senang", "nyaman", "percaya diri", "terbuka", "hangat"),
        "templates": ("{m} saat {x}", "gaya {m}", "cenderung {m}"),
    },
    "POSITIVE_SOCIAL": {
        "markers": ("sosial", "interaksi", "jaringan", "komunitas", "kenalan", "networking"),
        "x": ("aktif", "positif", "hangat", "luas", "dekat", "rutin"),
        "y": ("teman", "rekan", "orang", "kelompok", "lingkungan", "forum"),
        "templates": ("{m} {y}", "kebiasaan {m}", "minat {m}"),
    },
    "COLLABORATION": {
        "markers": ("kolaborasi", "kooperatif", "gotong royong", "sinergi", "tim solid"),
        "x": ("proyek", "kerja", "tim", "divisi", "unit", "remote"),
        "y": ("efektif", "erat", "produktif", "harmonis", "intens", "lintas"),
        "templates": ("{m} {x}", "budaya {m}", "spirit {m}"),
    },
    "TRUST": {
        "markers": ("percaya", "kepercayaan", "transparan", "jujur", "loyal", "integritas"),
        "x": ("rekan", "pasangan", "teman", "tim", "keluarga", "diri"),
        "y": ("penuh", "kuat", "saling", "bertahap", "konsisten", "tulus"),
        "templates": ("{m} {x}", "bangun {m}", "rasa {m}"),
    },
    "RELATIONSHIP_AFFECTION": {
        "markers": ("kasih", "sayang", "romantis", "mesra", "affection", "cinta"),
        "x": ("pasangan", "keluarga", "teman dekat", "hubungan", "orang tersayang"),
        "y": ("tulus", "dalam", "hangat", "setia", "lembut", "dekat"),
        "templates": ("{m} {x}", "ekspresi {m}", "bahasa {m}"),
    },
    "ANGER_EMO": {
        "markers": ("marah", "kesal", "jengkel", "amarah", "frustasi", "murka"),
        "x": ("cepat", "mudah", "dalam", "tiba", "berat", "pendam"),
        "y": ("situasi", "komentar", "tekanan", "konflik", "masalah", "orang"),
        "templates": ("{m} {x}", "emosional {m}", "reaksi {m}"),
    },
    "MENTAL_UNSTABLE_N": {
        "markers": ("labil", "burnout", "mood swing", "instabilitas", "kelelahan mental"),
        "x": ("emosional", "mental", "harian", "berat", "kronis", "akut"),
        "y": ("fase", "periode", "gejala", "tanda", "kondisi", "episode"),
        "templates": ("{m} {x}", "mengalami {m}", "fase {m}"),
    },
    "NEGATIVE_SOCIAL": {
        "markers": ("isolasi", "menghindar", "antisosial", "menarik diri", "konflik sosial"),
        "x": ("sosial", "orang", "keramaian", "pertemuan", "interaksi", "undangan"),
        "y": ("cenderung", "sering", "mudah", "cukup", "agak", "terus"),
        "templates": ("{m} {x}", "pola {m}", "kecenderungan {m}"),
    },
    "EMO_POSITIVE": {
        "markers": ("bahagia", "senang", "gembira", "ceria", "optimis", "lega"),
        "x": ("hari", "pagi", "kerja", "belajar", "liburan", "keluarga"),
        "y": ("terasa", "banget", "sekali", "dalam", "sungguh", "benar"),
        "templates": ("merasa {m}", "{m} saat {x}", "nuansa {m}"),
    },
    "EMO_NEGATIVE": {
        "markers": ("hampa", "pilu", "duka", "nestapa", "perih", "kosong batin"),
        "x": ("emosional", "batin", "hati", "jiwa", "perasaan", "diri"),
        "y": ("dalam", "berat", "sunyi", "lama", "tiba", "terus"),
        "templates": ("{m} {x}", "rasa {m}", "nuansa {m}"),
    },
    "E_SOCIAL_DEPENDENCY": {
        "markers": ("ketergantungan", "butuh orang", "butuh teman", "afiliasi", "kesepian"),
        "x": ("sosial", "dekat", "emotional", "validasi", "dukungan", "interaksi"),
        "y": ("kuat", "cukup", "sering", "terus", "dalam", "signifikan"),
        "templates": ("{m} {x}", "pola {m}", "kebutuhan {m}"),
    },
    "EMPATHY_HARMONY_A": {
        "markers": ("empati", "validasi", "harmonis", "peka", "mendengarkan", "pengertian"),
        "x": ("emosi", "perasaan", "teman", "pasangan", "rekan", "keluarga"),
        "y": ("tulus", "aktif", "dalam", "penuh", "hangat", "konsisten"),
        "templates": ("{m} {x}", "keterampilan {m}", "sikap {m}"),
    },
    "EXTREME_NEGATIVE": {
        "markers": ("bunuh diri", "ingin mati", "melukai diri", "akhiri hidup", "tidak ingin hidup"),
        "x": ("pikiran", "keinginan", "dorongan", "fantasi", "rencana"),
        "y": ("serius", "kuat", "terus", "malam", "sendiri"),
        "templates": ("{m}", "pikiran {m}", "keinginan {m}"),
    },
    "SAD_EMO": {
        "markers": ("sedih", "murung", "hampa", "apatis", "terpuruk", "kehilangan semangat"),
        "x": ("dalam", "berat", "sunyi", "lama", "mendalam", "terus"),
        "y": ("hidup", "diri", "hubungan", "pekerjaan", "malam", "pagi"),
        "templates": ("merasa {m}", "{m} {y}", "nuansa {m} {x}"),
    },
}


GENERATORS = {
    "ANXIETY_EMO": generate_anxiety,
    "SAD_EMO": generate_sad,
    "ANGER_EMO": generate_anger,
    "EMO_POSITIVE": lambda e, t: generate_simple_pool(POSITIVE_EMO, e, t),
    "EMPATHY_HARMONY_A": generate_empathy,
    "EXTRAVERSION_E": lambda e, t: generate_simple_pool(
        EXTRAVERSION, e, t, trait_prefix="ekstrovert"
    ),
    "POSITIVE_SOCIAL": lambda e, t: generate_simple_pool(
        EXTRAVERSION, e, t, trait_prefix="sosial"
    ),
    "CREATIVE_DISCUSSION_A": lambda e, t: generate_simple_pool(CREATIVE, e, t),
    "INTROSPECTION": generate_introspection,
    "DISCIPLINE_C": lambda e, t: generate_simple_pool(DISCIPLINE, e, t),
    "ACHIEVEMENT": lambda e, t: generate_simple_pool(ACHIEVEMENT, e, t),
    "COLLABORATION": generate_collaboration,
    "NEGATIVE_SOCIAL": generate_social_negative,
    "MENTAL_UNSTABLE_N": generate_mental_unstable,
    "EMO_NEGATIVE": generate_emo_negative,
    "RELATIONSHIP_AFFECTION": generate_relationship,
    "TRUST": generate_trust,
    "E_SOCIAL_DEPENDENCY": generate_esocial_dep,
    "EXTREME_NEGATIVE": generate_crisis,
}
