"""
slang_dict.py — Kamus normalisasi slang/singkatan chat Indonesia ke bentuk baku.

Digunakan oleh grooming_scorer untuk pre-process text sebelum matching.
Contoh: "kmu cantik bgt" → "kamu cantik banget"

Aturan:
- Match dengan word boundary (biar "kmu" tidak match "kmuas")
- Case-insensitive (input sudah lowercase)
- Multi-word slang di-handle terpisah (mis. "ga tau" → "tidak tahu")
- Kalau ada konflik, entry terpanjang menang (longest-match-first)

Format: {slang: baku}. Bebas ditambah/edit.
"""

SLANG_DICT = {
    # Kata ganti orang & artikel
    "kmu": "kamu",
    "kmi": "kami",
    "km": "kamu",
    "aq": "aku",
    "sy": "saya",
    "sya": "saya",
    "u": "kamu",
    "gw": "aku",
    "gue": "aku",
    "gua": "aku",
    "lu": "kamu",
    "lo": "kamu",
    "elu": "kamu",
    "loe": "kamu",

    # Konjungsi & partikel
    "yg": "yang",
    "dgn": "dengan",
    "sm": "sama",
    "sma": "sama",
    "utk": "untuk",
    "krn": "karena",
    "krna": "karena",
    "tp": "tapi",
    "tpi": "tapi",
    "klo": "kalau",
    "kalo": "kalau",
    "klu": "kalau",
    "kl": "kalau",
    "dr": "dari",
    "dri": "dari",
    "jg": "juga",
    "jga": "juga",
    "aja": "saja",
    "ajah": "saja",
    "ja": "saja",
    "sj": "saja",
    "spy": "supaya",
    "sblm": "sebelum",
    "stlh": "setelah",
    "krng": "kurang",
    "hrs": "harus",

    # Adverbia frekuensi & kuantitas
    "bgt": "banget",
    "bnget": "banget",
    "byk": "banyak",
    "bnyk": "banyak",
    "srg": "sering",
    "kdg": "kadang",
    "kdng": "kadang",
    "trs": "terus",
    "trus": "terus",

    # Waktu
    "skrg": "sekarang",
    "skg": "sekarang",
    "kpn": "kapan",
    "kmrn": "kemarin",
    "kmren": "kemarin",
    "bsk": "besok",
    "hri": "hari",
    "mlm": "malam",
    "sre": "sore",
    "sblum": "sebelum",

    # Pertanyaan & kata tanya
    "gmn": "gimana",
    "gmna": "gimana",
    "dmn": "dimana",
    "dmna": "dimana",
    "kmn": "kemana",
    "kmna": "kemana",
    "knp": "kenapa",
    "knpa": "kenapa",
    "mngkin": "mungkin",
    "mngkn": "mungkin",

    # Verba umum
    "udh": "udah",
    "uda": "udah",
    "sdh": "sudah",
    "blm": "belum",
    "blom": "belum",
    "belom": "belum",
    "msh": "masih",
    "bs": "bisa",
    "bsa": "bisa",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "engga": "tidak",
    "gpp": "gapapa",
    "gapapa": "tidak apa apa",
    "bkn": "bukan",
    "jgn": "jangan",
    "jangn": "jangan",
    "pgn": "pengen",
    "pgin": "pengen",
    "pengin": "pengen",
    "kyk": "kayak",
    "kya": "kayak",
    "kayak": "seperti",
    "kek": "kayak",
    "prnh": "pernah",
    "pnh": "pernah",
    "tdk": "tidak",
    "tdak": "tidak",
    "tau": "tahu",

    # Kata benda umum
    "tmn": "teman",
    "tmen": "teman",
    "temen": "teman",
    "rmh": "rumah",
    "sklh": "sekolah",
    "org": "orang",
    "ortu": "orang tua",
    "ortumu": "orang tuamu",
    "cwe": "cewek",
    "cwo": "cowok",
    "cewe": "cewek",
    "cowo": "cowok",
    "pcr": "pacar",
    "pacar": "pacar",

    # Chat/internet slang
    "pap": "kirim foto",
    "vc": "video call",
    "vn": "voice note",
    "dm": "direct message",
    "pm": "private message",
    "pict": "foto",
    "pic": "foto",
    "foto2": "foto",
    "foto-foto": "foto",
    "chat2": "chat",
    "chatan": "chat",
    "wa": "whatsapp",
    "ig": "instagram",
    "fb": "facebook",
    "yt": "youtube",
    "tt": "tiktok",
    "sosmed": "sosial media",
    "medsos": "sosial media",

    # Emosi/reaksi
    "mksh": "makasih",
    "mksih": "makasih",
    "makasi": "makasih",
    "trmksh": "terima kasih",
    "trmksih": "terima kasih",
    "maaf": "maaf",
    "srry": "maaf",
    "plis": "tolong",
    "plz": "tolong",
    "sayangku": "sayang",
    "sygku": "sayang",

    # Angka & waktu spesifik (biar match "jam 6.30" dll)
    "jm": "jam",

    # Partikel/interjeksi (kadang muncul di frasa lexicon)
    "loh": "loh",
    "lho": "loh",
    "sih": "sih",
    "dong": "dong",
    "donk": "dong",
    "dunk": "dong",
    "dh": "deh",
    "deh": "deh",
    "kok": "kok",
    "kek": "kayak",  # duplicate handled: longest first akan pilih ini
    "koq": "kok",
    "ya": "ya",
    "yaa": "ya",
    "yah": "ya",
}


# Bikin sorted list dari terpanjang → terpendek untuk longest-match-first
_SORTED_SLANG = sorted(SLANG_DICT.items(), key=lambda x: -len(x[0]))


def normalize_slang(text: str) -> str:
    """
    Normalize slang di text jadi bentuk baku.
    Word boundary awareness — 'kmu' tidak match 'akmu'.

    Contoh:
        "kmu cantik bgt sih" → "kamu cantik banget sih"
        "gpp kok, gw jg pgn" → "gapapa kok, aku juga pengen"
    """
    import re
    result = text
    for slang, baku in _SORTED_SLANG:
        # Word boundary regex
        pattern = r'\b' + re.escape(slang) + r'\b'
        result = re.sub(pattern, baku, result, flags=re.IGNORECASE)
    return result


if __name__ == "__main__":
    # Quick test
    tests = [
        "kmu cantik bgt sih",
        "gpp kok, gw jg pgn",
        "gmn kmren malam?",
        "aq udh blm makan",
        "jgn cerita ke ortumu ya",
        "pap dong yg selfie",
    ]
    for t in tests:
        print(f"  '{t}'\n  → '{normalize_slang(t)}'\n")
