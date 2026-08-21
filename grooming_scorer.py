"""
grooming_scorer.py — Core scoring engine untuk deteksi cyber grooming berbasis 7P.

Modul standalone, tidak tergantung notebook.
Dependency: pandas, openpyxl (untuk baca lexicon Excel).

Public API:
    load_lexicon(path) -> Lexicon
    detect_phrases(text, sender_role, lexicon) -> list[Match]
    accumulate_evidence(conversation, lexicon) -> Evidence
    classify_conv(evidence) -> Classification

Contoh usage:
    from grooming_scorer import load_lexicon, accumulate_evidence, classify_conv

    lex = load_lexicon("lexicon_7p_final_v3.xlsx")
    conv = [
        {"text": "hai, kamu cantik banget", "sender_role": "guru_dosen_asing"},
        {"text": "makasih kak hehe",         "sender_role": "teman_saudara"},
        {"text": "udah punya pacar belum?",  "sender_role": "guru_dosen_asing"},
    ]
    evidence = accumulate_evidence(conv, lex)
    result = classify_conv(evidence)
    print(result['classification'])  # 'HIJAU' | 'KUNING' | 'MERAH'
"""
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================
# Constants
# ============================================================

VALID_ROLES = {'ortu', 'teman_saudara', 'guru_dosen_asing', 'user'}

ROLE_MODIFIERS = {
    'ortu': -1,
    'teman_saudara': 0,
    'guru_dosen_asing': +1,
    'user': 0,  # user = pihak yang dianalisis (anak/korban), tidak di-score
}

# Role yang di-SKIP saat scoring — hanya untuk konteks percakapan,
# tidak dianalisis karena bukan sender yang di-evaluasi.
SKIP_SCORING_ROLES = {'user'}

P_CODES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']

P_NAMES = {
    'P1': 'Praise',
    'P2': 'Precocious',
    'P3': 'Photo sharing',
    'P4': 'Privacy',
    'P5': 'Pressure',
    'P6': 'Presents',
    'P7': 'Pulling away',
}

# Kombinasi kritis: pasangan P yang kalau muncul bersama = langsung MERAH
CRITICAL_COMBINATIONS = [
    ('P2', 'P4'),  # precocious + privacy
    ('P2', 'P5'),  # precocious + pressure
    ('P3', 'P4'),  # photo + privacy
    ('P3', 'P5'),  # photo + pressure
    ('P5', 'P7'),  # pressure + pulling away
]

# Kombinasi waspada: pasangan P yang kalau muncul dengan kondisi tertentu = KUNING
# Format: (Pa, Pb, kondisi, alasan)
# kondisi:
#   'both_ge_2'  = kedua P harus punya skor efektif >= 2
#   'a_ge_2'     = P pertama harus punya skor efektif >= 2
#   'either_ge_2'= salah satu P harus punya skor efektif >= 2
CAUTION_COMBINATIONS = [
    ('P1', 'P3', 'both_ge_2', 'Manipulasi awal untuk meminta foto lebih banyak (praise + photo)'),
    ('P2', 'P3', 'a_ge_2', 'Indikator fantasi seksual (precocious + photo)'),
    ('P5', 'P6', 'either_ge_2', 'Manipulasi emosional/psikologis (pressure + presents)'),
]

# Threshold klasifikasi (draft — akan di-tune psikolog)
THRESHOLD_MERAH = 8   # total score >= 8 = merah
THRESHOLD_KUNING = 4  # 4 <= total score < 8 = kuning


# ============================================================
# Data structures
# ============================================================

@dataclass
class LexiconEntry:
    kode_p: str
    frasa: str              # frasa yang dipakai untuk matching (lowercase)
    frasa_lang: str         # 'id' atau 'en'
    skor_dasar: int
    role_sensitive: bool
    frasa_display: str      # frasa asli untuk display (belum lowercase)
    concept_id: str = ''    # ID konsep — semua padanan dari row lexicon yang sama share concept_id


@dataclass
class Lexicon:
    entries: List[LexiconEntry] = field(default_factory=list)

    def __len__(self):
        return len(self.entries)


@dataclass
class Match:
    message_idx: int        # index pesan di conversation
    kode_p: str
    frasa_matched: str      # frasa lexicon yang match
    frasa_display: str      # untuk highlight UI
    skor_dasar: int
    skor_efektif: int       # setelah modifier role
    sender_role: str
    span: Tuple[int, int]   # posisi (start, end) di teks pesan (approx untuk semantic)
    source: str = 'literal' # 'literal' | 'semantic'
    similarity: float = 1.0 # untuk semantic match; literal selalu 1.0
    concept_id: str = ''    # inherit dari LexiconEntry — untuk dedup per-konsep


@dataclass
class PEvidence:
    total_score: int = 0          # sum skor efektif dari semua match untuk P ini
    max_score: int = 0            # max skor efektif single match
    count_matches: int = 0        # berapa kali P ini terdeteksi
    matched_phrases: List[Match] = field(default_factory=list)


# ============================================================
# 1. Load lexicon
# ============================================================

def _clean_phrase(text) -> str:
    """Bersihkan frasa: lowercase, strip, normalize whitespace."""
    if pd.isna(text):
        return ""
    s = str(text).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def _split_id_phrases(text) -> List[str]:
    """Split kolom Bahasa Indonesia yang berisi multi-frasa dengan '/'.
    Return list of cleaned phrases (non-empty)."""
    if pd.isna(text):
        return []
    parts = str(text).split('/')
    cleaned = [_clean_phrase(p) for p in parts]
    return [p for p in cleaned if p]


def load_lexicon(path: str, sheet_name: str = 'Lexicon v3') -> Lexicon:
    """Load lexicon dari Excel. Kembalikan Lexicon dengan entries siap match."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    required_cols = ['Kode P', 'frasa_en', 'Frasa (Bahasa Indonesia)',
                     'Skor (1-3)', 'role_sensitive']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang di lexicon: {missing}")

    entries: List[LexiconEntry] = []

    for row_idx, row in df.iterrows():
        kode_p = str(row['Kode P']).strip()
        skor = int(row['Skor (1-3)'])
        role_sens = bool(row['role_sensitive'])
        # concept_id: 1 baris Excel = 1 konsep, semua padanan share ini
        concept_id = f"{kode_p}_row{row_idx}"

        # English phrase (1 per baris)
        frasa_en = _clean_phrase(row['frasa_en'])
        if frasa_en:
            entries.append(LexiconEntry(
                kode_p=kode_p,
                frasa=frasa_en,
                frasa_lang='en',
                skor_dasar=skor,
                role_sensitive=role_sens,
                frasa_display=frasa_en,
                concept_id=concept_id,
            ))

        # Indonesian phrases (bisa multiple, dipisah /)
        for frasa_id in _split_id_phrases(row['Frasa (Bahasa Indonesia)']):
            entries.append(LexiconEntry(
                kode_p=kode_p,
                frasa=frasa_id,
                frasa_lang='id',
                skor_dasar=skor,
                role_sensitive=role_sens,
                frasa_display=frasa_id,
                concept_id=concept_id,
            ))

    # Dedup: kalau ada 2 entri dengan frasa+kode_p+lang identik, ambil yg skor tertinggi
    seen = {}
    for e in entries:
        key = (e.kode_p, e.frasa, e.frasa_lang)
        if key not in seen or e.skor_dasar > seen[key].skor_dasar:
            seen[key] = e

    return Lexicon(entries=list(seen.values()))


# ============================================================
# 2. Detect phrases in one message
# ============================================================

def _apply_role_modifier(skor_dasar: int, role_sensitive: bool, sender_role: str) -> int:
    """Hitung skor efektif setelah apply modifier role.
    Skor di-clip ke [0, 3]. Frasa dengan role_sensitive=False → skor tidak berubah."""
    if not role_sensitive:
        return skor_dasar
    if sender_role not in ROLE_MODIFIERS:
        return skor_dasar
    modifier = ROLE_MODIFIERS[sender_role]
    return max(0, min(3, skor_dasar + modifier))


def _build_pattern(frasa: str) -> str:
    """Regex word-boundary yang handle apostrofe & special chars."""
    escaped = re.escape(frasa)
    prefix = r'\b' if frasa[0].isalnum() else ''
    suffix = r'\b' if frasa[-1].isalnum() else ''
    return prefix + escaped + suffix


def detect_phrases(text: str, sender_role: str, lexicon: Lexicon,
                    message_idx: int = 0, use_slang_normalization: bool = True) -> List[Match]:
    """Deteksi semua frasa lexicon di pesan (literal match).
    Kalau use_slang_normalization=True, singkatan (kmu, bgt, dll) di-normalize
    ke bentuk baku dulu sebelum match."""
    if sender_role not in VALID_ROLES:
        raise ValueError(f"sender_role tidak valid: '{sender_role}'. "
                         f"Harus salah satu dari {VALID_ROLES}")

    text_lower = text.lower()

    # Normalize slang: "kmu bgt" -> "kamu banget"
    if use_slang_normalization:
        try:
            from slang_dict import normalize_slang
            text_lower = normalize_slang(text_lower)
        except ImportError:
            pass  # graceful fallback

    matches = []

    for entry in lexicon.entries:
        pattern = _build_pattern(entry.frasa)
        try:
            for m in re.finditer(pattern, text_lower):
                skor_efektif = _apply_role_modifier(
                    entry.skor_dasar, entry.role_sensitive, sender_role
                )
                if skor_efektif <= 0:
                    continue  # skor 0 tidak dihitung sebagai bukti
                matches.append(Match(
                    message_idx=message_idx,
                    kode_p=entry.kode_p,
                    frasa_matched=entry.frasa,
                    frasa_display=entry.frasa_display,
                    skor_dasar=entry.skor_dasar,
                    skor_efektif=skor_efektif,
                    sender_role=sender_role,
                    span=(m.start(), m.end()),
                    concept_id=entry.concept_id,
                ))
        except re.error:
            continue

    # Dedup: kalau ada 2 match dengan kode_p sama di posisi overlap, ambil skor tertinggi
    matches = _dedup_matches(matches)
    return matches


def _dedup_matches(matches: List[Match]) -> List[Match]:
    """Buang match duplikat (sama kode_p + span overlap). Prioritaskan skor efektif tertinggi."""
    matches_sorted = sorted(matches, key=lambda m: (-m.skor_efektif, m.span[0]))
    kept = []
    for m in matches_sorted:
        overlap = False
        for k in kept:
            if k.kode_p == m.kode_p and not (m.span[1] <= k.span[0] or m.span[0] >= k.span[1]):
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return sorted(kept, key=lambda m: m.span[0])


# ============================================================
# 3. Accumulate evidence across conversation
# ============================================================

def accumulate_evidence(conversation: List[Dict], lexicon: Lexicon,
                        semantic_matcher=None) -> Dict:
    """
    Input: list of {'text': str, 'sender_role': str}
    Output: dict berisi:
        - 'per_p': {P1: PEvidence, P2: PEvidence, ...}
        - 'critical_pairs_triggered': list of (Pa, Pb) tuples
        - 'has_score_3': bool
        - 'total_score': int (sum semua skor efektif)
        - 'total_matches': int (semua match)
        - 'n_p_detected': int (berapa P berbeda terdeteksi)
        - 'all_matches': list[Match] (untuk display detail)
    """
    per_p: Dict[str, PEvidence] = {p: PEvidence() for p in P_CODES}
    all_matches: List[Match] = []  # ALL raw matches (untuk display detail)

    # STEP 1: collect all raw matches (literal + semantic) per pesan
    for idx, msg in enumerate(conversation):
        text = msg.get('text', '')
        role = msg.get('sender_role', 'teman_saudara')
        if not text.strip():
            continue
        # Skip role yang bukan sender (user/korban) — tidak di-score,
        # hanya jadi konteks percakapan
        if role in SKIP_SCORING_ROLES:
            continue

        # Literal match (dengan slang normalization)
        literal_matches = detect_phrases(text, role, lexicon, message_idx=idx)
        all_matches.extend(literal_matches)

        # Semantic match (kalau matcher tersedia)
        if semantic_matcher is not None and semantic_matcher.is_available():
            sem_matches = semantic_matcher.match(
                text, role, message_idx=idx,
                apply_role_modifier_fn=_apply_role_modifier,
            )
            # Skip semantic match kalau frasa yang sama sudah di-match literal (di pesan ini)
            literal_keys = {(m.kode_p, m.frasa_matched) for m in literal_matches}
            # CAP: semantic tidak boleh keluar skor 3 (red flag reserved untuk literal)
            # Alasan: semantic MiniLM sering false positive dengan sim tinggi.
            # Konsisten dengan Fix B: semantic tidak trigger MERAH sendirian.
            SEMANTIC_MAX_SCORE = 2
            for sm in sem_matches:
                if (sm.kode_p, sm.frasa_matched) not in literal_keys:
                    if sm.skor_efektif > SEMANTIC_MAX_SCORE:
                        sm.skor_efektif = SEMANTIC_MAX_SCORE
                    all_matches.append(sm)

    # STEP 2: Multi-stage dedup untuk mencegah over-counting
    #
    # (2a) Per (msg_idx, kode_p) → kalau ada literal, drop semua semantic di P ini
    #      Alasan: literal match punya span exact, lebih trusted daripada semantic
    #      similarity yang mungkin false positive.
    #
    # (2b) Per pesan (across all P) → greedy overlap-dedup by span.
    #      1 potongan text max 1 bukti. Kalau "ganteng banget" dan "ganteng" sama-
    #      sama match di span overlap, keep salah satu:
    #      - skor tertinggi menang
    #      - kalau skor sama, frasa lebih panjang menang (lebih kaya konteks)
    #      - kalau masih sama, literal > semantic
    #
    # (2c) Per (kode_p, concept_id) global → dedup akhir konsep sama antar-pesan
    #      1 konsep dalam 1 percakapan = 1 bukti (skor tertinggi).

    # ---- (2a) literal-wins-over-semantic per (msg_idx, kode_p) ----
    literal_keys = {(m.message_idx, m.kode_p) for m in all_matches if m.source == 'literal'}
    matches_after_2a = [
        m for m in all_matches
        if m.source == 'literal' or (m.message_idx, m.kode_p) not in literal_keys
    ]

    # ---- (2b) per-pesan overlap-dedup by span ----
    # Semantic tidak punya span exact — skip dari span-overlap check (tapi tetap masuk hasil).
    from collections import defaultdict
    per_msg = defaultdict(list)
    for m in matches_after_2a:
        per_msg[m.message_idx].append(m)

    matches_after_2b = []
    for msg_idx, msg_matches in per_msg.items():
        # Semantic lolos otomatis (span tidak akurat)
        sem = [m for m in msg_matches if m.source == 'semantic']
        lit = [m for m in msg_matches if m.source == 'literal']
        matches_after_2b.extend(sem)

        # Literal: sort by (skor desc, len frasa desc), greedy skip overlap
        lit.sort(key=lambda x: (-x.skor_efektif, -len(x.frasa_matched)))
        consumed_spans = []  # list of (start, end)
        for m in lit:
            m_start, m_end = m.span
            # Cek overlap dengan span yang sudah di-keep
            overlap = any(
                m_start < cs_end and m_end > cs_start
                for cs_start, cs_end in consumed_spans
            )
            if not overlap:
                matches_after_2b.append(m)
                consumed_spans.append(m.span)

    # ---- (2d) per (msg_idx, frasa_matched) — frasa exact sama di msg sama = 1 bukti ----
    # Contoh: "ketemuan yuk" muncul di P2 dan P6 (frasa sama, P beda, msg sama).
    # Keep skor tertinggi; skor sama → sim tertinggi; masih sama → literal wins.
    key_2d = {}
    for m in matches_after_2b:
        key = (m.message_idx, m.frasa_matched)
        if key not in key_2d:
            key_2d[key] = m
        else:
            existing = key_2d[key]
            if m.skor_efektif > existing.skor_efektif:
                key_2d[key] = m
            elif m.skor_efektif == existing.skor_efektif:
                if m.similarity > existing.similarity:
                    key_2d[key] = m
                elif m.similarity == existing.similarity and m.source == 'literal' and existing.source == 'semantic':
                    key_2d[key] = m
    matches_after_2d = list(key_2d.values())

    # ---- (2e) cap semantic per pesan (max 1 semantic per msg, top similarity) ----
    # Alasan: semua semantic match di 1 pesan berasal dari embedding KALIMAT yg sama.
    # Jadi 1 pesan = 1 sinyal semantic saja, bukan puluhan.
    # 1 kalimat pendek yg vibe "praise" = 1 P1 bukti, bukan 15 padanan berbeda.
    # Literal tidak di-cap (mereka punya span exact, lebih trusted).
    MAX_SEMANTIC_PER_MSG = 1  # 1 pesan = 1 sinyal semantic saja (embedding kalimat)
    from collections import defaultdict
    per_msg_2e = defaultdict(list)
    for m in matches_after_2d:
        per_msg_2e[m.message_idx].append(m)

    matches_after_2e = []
    for msg_idx, msg_matches in per_msg_2e.items():
        lit = [m for m in msg_matches if m.source == 'literal']
        sem = [m for m in msg_matches if m.source == 'semantic']
        # Sort semantic by sim desc, keep top MAX_SEMANTIC_PER_MSG
        sem.sort(key=lambda x: -x.similarity)
        sem_capped = sem[:MAX_SEMANTIC_PER_MSG]
        matches_after_2e.extend(lit + sem_capped)

    # ---- (2c) per (kode_p, concept_id) global dedup ----
    unique_matches = {}
    for m in matches_after_2e:
        key = (m.kode_p, m.concept_id or m.frasa_matched)
        if key not in unique_matches:
            unique_matches[key] = m
        else:
            existing = unique_matches[key]
            if m.skor_efektif > existing.skor_efektif:
                unique_matches[key] = m
            elif m.skor_efektif == existing.skor_efektif and m.source == 'literal' and existing.source == 'semantic':
                unique_matches[key] = m

    # STEP 3: build per_p dari unique matches (bukan raw)
    # ATURAN: skor 1 (hijau/indikator lemah) TIDAK masuk akumulasi total_score.
    # Alasan: 10 frasa skor 1 tidak boleh terakumulasi jadi 10 (yang bakal trigger MERAH).
    # Skor 1 tetap dihitung count_matches dan max_score (biar tetap terlihat di P summary).
    MIN_SCORE_FOR_ACCUMULATION = 2
    for m in unique_matches.values():
        if m.kode_p not in per_p:
            continue
        if m.skor_efektif >= MIN_SCORE_FOR_ACCUMULATION:
            per_p[m.kode_p].total_score += m.skor_efektif
        per_p[m.kode_p].max_score = max(per_p[m.kode_p].max_score, m.skor_efektif)
        per_p[m.kode_p].count_matches += 1
        per_p[m.kode_p].matched_phrases.append(m)

    # Kombinasi kritis
    critical_triggered = []
    for pa, pb in CRITICAL_COMBINATIONS:
        if per_p[pa].count_matches > 0 and per_p[pb].count_matches > 0:
            critical_triggered.append((pa, pb))

    # Kombinasi waspada (KUNING)
    caution_triggered = []
    for pa, pb, kondisi, alasan in CAUTION_COMBINATIONS:
        eva, evb = per_p[pa], per_p[pb]
        if eva.count_matches == 0 or evb.count_matches == 0:
            continue
        cond_met = False
        if kondisi == 'both_ge_2':
            cond_met = eva.max_score >= 2 and evb.max_score >= 2
        elif kondisi == 'a_ge_2':
            cond_met = eva.max_score >= 2
        elif kondisi == 'either_ge_2':
            cond_met = eva.max_score >= 2 or evb.max_score >= 2
        if cond_met:
            caution_triggered.append((pa, pb, alasan))

    # Frasa skor 3? — HANYA dari LITERAL match yang bisa trigger MERAH sendirian
    # Semantic tidak boleh single-source decision-maker karena bisa false positive
    # (mis. sim=0.86 tapi konteks salah). Semantic red flag butuh konfirmasi bukti lain.
    has_score_3 = any(
        m.skor_efektif >= 3 and m.source == 'literal'
        for m in unique_matches.values()
    )
    # Track semantic red flags secara terpisah untuk display/reasoning
    has_score_3_semantic = any(
        m.skor_efektif >= 3 and m.source == 'semantic'
        for m in unique_matches.values()
    )

    total_score = sum(p.total_score for p in per_p.values())
    total_matches = sum(p.count_matches for p in per_p.values())
    n_p_detected = sum(1 for p in per_p.values() if p.count_matches > 0)

    return {
        'per_p': per_p,
        'critical_pairs_triggered': critical_triggered,
        'caution_pairs_triggered': caution_triggered,
        'has_score_3': has_score_3,                    # LITERAL only
        'has_score_3_semantic': has_score_3_semantic,  # semantic (untuk display, tidak trigger MERAH)
        'total_score': total_score,
        'total_matches': total_matches,
        'n_p_detected': n_p_detected,
        'all_matches': list(unique_matches.values()),
        'raw_matches': all_matches,
    }


# ============================================================
# 4. Classify conversation
# ============================================================

def classify_conv(evidence: Dict) -> Dict:
    """
    Klasifikasi konservatif berdasarkan akumulasi bukti.

    MERAH (minimal 1):
        - Ada kombinasi kritis terdeteksi
        - Ada minimal 1 frasa dengan skor efektif = 3
        - Total 3+ P berbeda dengan >= 2 match masing-masing (kecuali cuma P1+P6 saja)
        - Total akumulasi skor >= THRESHOLD_MERAH (8)

    KUNING (minimal 1, dan tidak merah):
        - 2+ P berbeda terdeteksi
        - Ada minimal 1 P dengan >= 3 match
        - Ada frasa dengan skor efektif = 2
        - Total akumulasi skor >= THRESHOLD_KUNING (4)

    HIJAU: sisanya.
    """
    per_p = evidence['per_p']
    reasoning = []

    # === MERAH checks ===
    merah = False

    if evidence['critical_pairs_triggered']:
        merah = True
        for pa, pb in evidence['critical_pairs_triggered']:
            reasoning.append(
                f"Kombinasi kritis {pa} ({P_NAMES[pa]}) + {pb} ({P_NAMES[pb]}) terdeteksi"
            )

    if evidence['has_score_3']:
        merah = True
        # Cari mana yang skor 3
        score3_ps = sorted(set(m.kode_p for m in evidence['all_matches'] if m.skor_efektif >= 3))
        reasoning.append(
            f"Frasa red flag skor 3 terdeteksi di variabel: {', '.join(score3_ps)}"
        )

    # 3+ P dengan >= 2 match, kecuali kombinasi ringan
    ps_with_2plus = [p for p, ev in per_p.items() if ev.count_matches >= 2]
    non_trivial_ps = [p for p in ps_with_2plus if p not in ('P1', 'P6')]
    if len(ps_with_2plus) >= 3 and len(non_trivial_ps) >= 1:
        merah = True
        reasoning.append(
            f"{len(ps_with_2plus)} variabel P terdeteksi dengan bukti berulang: "
            f"{', '.join(ps_with_2plus)}"
        )

    if evidence['total_score'] >= THRESHOLD_MERAH:
        merah = True
        reasoning.append(
            f"Total akumulasi skor {evidence['total_score']} "
            f"melewati threshold merah ({THRESHOLD_MERAH})"
        )

    if merah:
        return _build_result('MERAH', evidence, reasoning)

    # === KUNING checks ===
    kuning = False

    # Kombinasi waspada langsung trigger KUNING (sebelum rule C, karena multi-P)
    if evidence.get('caution_pairs_triggered'):
        kuning = True
        for pa, pb, alasan in evidence['caution_pairs_triggered']:
            reasoning.append(f"Kombinasi waspada {pa}+{pb}: {alasan}")

    # Rule C: kalau cuma 1 P terdeteksi, butuh minimal 3 match dan skor > 3
    # untuk trigger KUNING. Ini menghindari misfire dari repeated single word
    # (mis. 'sayang' 2x dari pacar → tetap HIJAU).
    single_p_only = evidence['n_p_detected'] == 1
    if single_p_only:
        the_p = next(p for p, ev in per_p.items() if ev.count_matches > 0)
        the_ev = per_p[the_p]
        # Rule C hanya bypass ke HIJAU kalau max_score=1 (warning saja, butuh repetisi 3+)
        # Kalau max_score >= 2 (frasa spesifik), lanjutkan ke KUNING checks di bawah.
        if the_ev.max_score <= 1 and the_ev.count_matches < 3:
            reasoning.append(
                f"Hanya 1 variabel P ({the_p}) terdeteksi dengan skor warning (max=1) "
                f"dan {the_ev.count_matches} match — belum cukup untuk kuning"
            )
            return _build_result('HIJAU', evidence, reasoning)

    # Aturan multi-P: hanya hitung P dengan max_score >= 2 (skor 1 = hijau/warning, tidak trigger kuning)
    # Konsisten dengan aturan akumulasi: skor 1 tidak dorong ke merah/kuning
    p_with_score_ge_2 = [p for p, ev in per_p.items() if ev.max_score >= 2]
    if len(p_with_score_ge_2) >= 2:
        kuning = True
        reasoning.append(f"{len(p_with_score_ge_2)} variabel P dengan bukti skor >=2 terdeteksi: {', '.join(sorted(p_with_score_ge_2))}")

    p_with_3plus = [p for p, ev in per_p.items() if ev.count_matches >= 3]
    if p_with_3plus:
        kuning = True
        reasoning.append(f"Bukti berulang (3+ match) di variabel: {', '.join(p_with_3plus)}")

    max_score_any = max((ev.max_score for ev in per_p.values()), default=0)
    if max_score_any >= 2:
        kuning = True
        p_with_score2 = sorted(set(
            m.kode_p for m in evidence['all_matches'] if m.skor_efektif >= 2
        ))
        reasoning.append(f"Frasa skor sedang (2) terdeteksi di: {', '.join(p_with_score2)}")

    if evidence['total_score'] >= THRESHOLD_KUNING:
        kuning = True
        reasoning.append(
            f"Total akumulasi skor {evidence['total_score']} "
            f"melewati threshold kuning ({THRESHOLD_KUNING})"
        )

    if kuning:
        return _build_result('KUNING', evidence, reasoning)

    # === HIJAU ===
    if evidence['total_matches'] == 0:
        reasoning.append("Tidak ada frasa lexicon terdeteksi")
    else:
        reasoning.append(
            f"Deteksi minim: {evidence['total_matches']} match, skor total {evidence['total_score']}, "
            f"di bawah threshold kuning ({THRESHOLD_KUNING})"
        )
    return _build_result('HIJAU', evidence, reasoning)


def _build_result(classification: str, evidence: Dict, reasoning: List[str]) -> Dict:
    """Format output final klasifikasi."""
    return {
        'classification': classification,
        'total_score': evidence['total_score'],
        'total_matches': evidence['total_matches'],
        'n_p_detected': evidence['n_p_detected'],
        'per_p_summary': {
            p: {
                'count': ev.count_matches,
                'total_score': ev.total_score,
                'max_score': ev.max_score,
            }
            for p, ev in evidence['per_p'].items()
        },
        'critical_pairs': evidence['critical_pairs_triggered'],
        'caution_pairs': evidence.get('caution_pairs_triggered', []),
        'has_score_3': evidence['has_score_3'],
        'reasoning': reasoning,
        'all_matches': evidence['all_matches'],
    }


# ============================================================
# CLI untuk quick test
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python grooming_scorer.py <lexicon_path.xlsx>")
        sys.exit(1)

    lex = load_lexicon(sys.argv[1])
    print(f"[OK] Loaded {len(lex)} lexicon entries")
    print(f"  English : {sum(1 for e in lex.entries if e.frasa_lang == 'en')}")
    print(f"  Indonesia: {sum(1 for e in lex.entries if e.frasa_lang == 'id')}")

    # Smoke test
    demo_conv = [
        {"text": "hai, kamu cantik banget di foto profilnya", "sender_role": "guru_dosen_asing"},
        {"text": "makasih kak hehe", "sender_role": "teman_saudara"},
        {"text": "udah punya pacar belum?", "sender_role": "guru_dosen_asing"},
        {"text": "belum kak", "sender_role": "teman_saudara"},
        {"text": "kamu dewasa banget loh buat umur segitu", "sender_role": "guru_dosen_asing"},
        {"text": "eh foto kamu dong yang lagi selfie", "sender_role": "guru_dosen_asing"},
        {"text": "tapi jangan bilang siapa-siapa ya, ini rahasia kita berdua aja", "sender_role": "guru_dosen_asing"},
        {"text": "cuma aku yang peduli sama kamu, temen-temenmu gak ngerti kamu", "sender_role": "guru_dosen_asing"},
    ]
    ev = accumulate_evidence(demo_conv, lex)
    result = classify_conv(ev)

    print(f"\n{'='*60}\nDEMO CLASSIFICATION\n{'='*60}")
    print(f"Klasifikasi : {result['classification']}")
    print(f"Total skor  : {result['total_score']}")
    print(f"Total match : {result['total_matches']}")
    print(f"P terdeteksi: {result['n_p_detected']} dari 7")
    print(f"\nPer P:")
    for p in P_CODES:
        s = result['per_p_summary'][p]
        if s['count'] > 0:
            print(f"  {p} ({P_NAMES[p]:15s}): {s['count']} match, total {s['total_score']}, max {s['max_score']}")
    print(f"\nAlasan:")
    for r in result['reasoning']:
        print(f"  - {r}")
