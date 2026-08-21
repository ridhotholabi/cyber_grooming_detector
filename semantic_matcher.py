"""
semantic_matcher.py — Semantic matching untuk padanan kata/frasa yang tidak literal.

Pakai sentence-transformers (model default: paraphrase-multilingual-MiniLM-L12-v2).
Semua frasa lexicon di-encode sekali di awal, lalu tiap pesan input di-encode
dan dibandingkan cosine similarity.

Dependency: pip install sentence-transformers
Model download otomatis pertama kali (~350MB), cached di ~/.cache/huggingface/

Fallback: kalau library atau model tidak tersedia, class tetap bisa di-instantiate
tapi match() selalu return []. Ini biar engine tetap jalan (lexicon literal only)
kalau semantic tidak aktif.
"""
import re
from typing import List, Optional


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.85

# Role-aware threshold: role dengan baseline trust rendah (asing) pakai threshold default.
# Role trust tinggi (pacar, ortu) butuh evidence lebih kuat untuk mencegah false positive
# dari context wajar (mis. "hehe aku pengen mie" != "aku tag ortumu" walau sim 0.87).
ROLE_THRESHOLD_OVERRIDES = {
    'teman_saudara': 0.92,   # pacar/teman — evidence sangat kuat (rare true positive di konteks wajar)
    'ortu': 0.92,             # ortu — mustahil predator, evidence sangat kuat
    'guru_dosen_asing': None, # pakai default (0.85)
    'user': None,             # tidak di-score
}
MIN_TEXT_LENGTH = 15        # skip kalimat < 15 char (mis. "hehe", "iya", ":)") — embedding tidak reliable
MIN_WORD_COUNT = 3          # atau minimal 3 kata untuk semantic match


def _split_sentences(text: str) -> List[str]:
    """Pecah text jadi sentence chunks berdasarkan tanda baca.
    Fallback: kalau tidak ada tanda baca, return whole text."""
    # Split by . ! ? ; atau newline (tapi jaga panjang minimal 3 kata)
    parts = re.split(r'[.!?;\n]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    return parts


class SemanticMatcher:
    """
    Semantic matcher untuk frasa lexicon.

    Usage:
        matcher = SemanticMatcher(threshold=0.65)
        if matcher.is_available():
            matcher.index_lexicon(lexicon.entries)
            matches = matcher.match(text, sender_role, message_idx=0)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 threshold: float = DEFAULT_THRESHOLD,
                 verbose: bool = True):
        self.model_name = model_name
        self.threshold = threshold
        self.verbose = verbose
        self.model = None
        self.lexicon_entries = None
        self.lexicon_embeddings = None
        self._load_error = None

        # Lazy load
        self._try_load_model()

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            if self.verbose:
                print(f"[SemanticMatcher] Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"[SemanticMatcher] Model loaded (threshold={self.threshold})")
        except ImportError:
            self._load_error = "sentence-transformers not installed. Run: pip install sentence-transformers"
            if self.verbose:
                print(f"[SemanticMatcher] DISABLED: {self._load_error}")
        except Exception as e:
            self._load_error = f"Model load failed: {e}"
            if self.verbose:
                print(f"[SemanticMatcher] DISABLED: {self._load_error}")

    def is_available(self) -> bool:
        return self.model is not None

    def index_lexicon(self, lexicon_entries: List):
        """Precompute embeddings untuk semua frasa lexicon. Panggil sekali di awal."""
        if not self.is_available():
            return
        self.lexicon_entries = lexicon_entries
        phrases = [e.frasa for e in lexicon_entries]
        if self.verbose:
            print(f"[SemanticMatcher] Encoding {len(phrases)} lexicon phrases...")
        self.lexicon_embeddings = self.model.encode(
            phrases, normalize_embeddings=True, show_progress_bar=False
        )
        if self.verbose:
            print(f"[SemanticMatcher] Indexed {len(phrases)} phrases")

    def match(self, text: str, sender_role: str, message_idx: int = 0,
              apply_role_modifier_fn=None, min_skor_after_modifier: int = 1) -> List:
        """
        Cari frasa lexicon yang semantically similar ke text.

        Returns list of dict (kalau perlu, di-convert ke Match di caller).
        Skip kalau text kosong atau matcher tidak available.
        """
        from grooming_scorer import Match  # avoid circular at module import time

        if not self.is_available() or self.lexicon_embeddings is None:
            return []
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        # Filter kalimat terlalu pendek — embedding text pendek tidak reliable
        # ("hehe" bisa cosine-similar ke "ganteng", "ngewe", "beliin", dst)
        sentences = [
            s for s in sentences
            if len(s.strip()) >= MIN_TEXT_LENGTH
            and len(s.strip().split()) >= MIN_WORD_COUNT
        ]
        if not sentences:
            return []

        # Encode all sentences at once
        sent_embs = self.model.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )

        # Track best match per kode_p untuk seluruh pesan ini
        # Semantic: max 1 match per P per pesan (yang tertinggi similarity)
        # Alasan: 1 kalimat mirip "praise" vibe → 1 sinyal P1, bukan 15 sinyal
        best_per_p = {}  # kode_p -> (sim_f, entry_index)

        for sent_emb in sent_embs:
            # Cosine similarity (embeddings sudah normalized, jadi dot product = cosine)
            sims = sent_emb @ self.lexicon_embeddings.T  # shape: (n_lexicon,)
            # Role-aware threshold
            effective_threshold = ROLE_THRESHOLD_OVERRIDES.get(sender_role) or self.threshold
            for i, sim in enumerate(sims):
                sim_f = float(sim)
                if sim_f < effective_threshold:
                    continue
                entry = self.lexicon_entries[i]
                # Keep hanya yang tertinggi per kode_p
                existing = best_per_p.get(entry.kode_p)
                if existing is None or sim_f > existing[0]:
                    best_per_p[entry.kode_p] = (sim_f, i)

        # Build matches dari best_per_p
        matches = []
        for kode_p, (sim_f, entry_idx) in best_per_p.items():
            entry = self.lexicon_entries[entry_idx]

            # Apply role modifier
            if apply_role_modifier_fn:
                skor_efektif = apply_role_modifier_fn(
                    entry.skor_dasar, entry.role_sensitive, sender_role
                )
            else:
                skor_efektif = entry.skor_dasar
            if skor_efektif < min_skor_after_modifier:
                continue

            matches.append(Match(
                message_idx=message_idx,
                kode_p=entry.kode_p,
                frasa_matched=entry.frasa,
                frasa_display=entry.frasa_display,
                skor_dasar=entry.skor_dasar,
                skor_efektif=skor_efektif,
                sender_role=sender_role,
                span=(0, min(len(text), 100)),
                source='semantic',
                similarity=sim_f,
                concept_id=entry.concept_id,
            ))

        return matches
