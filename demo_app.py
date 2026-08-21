"""
demo_app.py — Streamlit demo aplikasi deteksi cyber grooming 7P.

Menjalankan lokal:
    streamlit run demo_app.py

Prasyarat:
    - lexicon_7p_final_v5.xlsx di folder yang sama
    - grooming_scorer.py + semantic_matcher.py + slang_dict.py di folder yang sama
    - pip install -r requirements.txt
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from grooming_scorer import (
    load_lexicon, accumulate_evidence, classify_conv,
    P_CODES, P_NAMES, VALID_ROLES,
    CRITICAL_COMBINATIONS, CAUTION_COMBINATIONS,
    THRESHOLD_MERAH, THRESHOLD_KUNING, ROLE_MODIFIERS,
)

# Semantic matcher (optional)
try:
    from semantic_matcher import SemanticMatcher
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# ============================================================
# Config
# ============================================================
st.set_page_config(
    page_title="Deteksi Cyber Grooming 7P",
    page_icon="🛡️",
    layout="wide",
)

# --- CSS: chat bubble style (adaptif dark/light via Streamlit theme) ---
st.markdown("""
<style>
    /* Chat container — inherit background dari Streamlit theme */
    .chat-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px;
        max-height: 620px;
        overflow-y: auto;
    }

    /* Chat bubble base */
    .msg-row { display: flex; margin: 10px 0; }
    .msg-row.left { justify-content: flex-start; }
    .msg-row.right { justify-content: flex-end; }

    .bubble {
        max-width: 72%;
        padding: 10px 14px;
        border-radius: 14px;
        font-size: 0.95em;
        line-height: 1.4;
        word-wrap: break-word;
    }

    /* Sender bubble (left) — background transparan biar adapt ke theme */
    .bubble.sender {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-top-left-radius: 4px;
    }
    .bubble.sender-asing { border-left: 3px solid #DC2626; }
    .bubble.sender-ortu { border-left: 3px solid #3B82F6; }
    .bubble.sender-teman { border-left: 3px solid #9CA3AF; }

    /* User bubble (right) — tint biru yang jalan di dark & light */
    .bubble.user {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-top-right-radius: 4px;
    }

    /* Role label di atas bubble */
    .role-label {
        font-size: 0.72em;
        font-weight: 600;
        margin-bottom: 4px;
        opacity: 0.75;
    }

    /* Semantic badge */
    .sem-badges { margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

LEXICON_PATH = "lexicon_7p_final_v5.xlsx"

ROLE_LABELS = {
    'user': '🧒 User (anak/korban)',
    'ortu': '👨‍👩‍👧 Orang tua',
    'teman_saudara': '👥 Teman / saudara',
    'guru_dosen_asing': '⚠️ Guru / dosen / asing',
}

# Warna per P
P_COLORS = {
    'P1': '#E8A33D', 'P2': '#E85D5D', 'P3': '#D8659F', 'P4': '#8B7BC7',
    'P5': '#5DA5E8', 'P6': '#5DBF8E', 'P7': '#7BC77B',
}

CLASSIFICATION_COLORS = {'HIJAU': '#10B981', 'KUNING': '#F59E0B', 'MERAH': '#EF4444'}
CLASSIFICATION_EMOJI = {'HIJAU': '🟢', 'KUNING': '🟡', 'MERAH': '🔴'}


# ============================================================
# Cache
# ============================================================
@st.cache_resource
def get_lexicon():
    return load_lexicon(LEXICON_PATH)


@st.cache_resource
def get_semantic_matcher(_lexicon):
    """Load semantic matcher & index lexicon. Cached across reruns."""
    if not SEMANTIC_AVAILABLE:
        return None
    try:
        matcher = SemanticMatcher(threshold=0.85, verbose=False)
        if not matcher.is_available():
            return None
        matcher.index_lexicon(_lexicon.entries)
        return matcher
    except Exception:
        return None


# ============================================================
# Skenario predefined
# ============================================================
PREDEFINED_SCENARIOS = {
    "-- Kosong (input manual) --": [],
    "Grooming klasik (orang asing → remaja)": [
        ("guru_dosen_asing", "hai, kamu cantik banget di foto profilnya"),
        ("user", "makasih kak hehe"),
        ("guru_dosen_asing", "udah punya pacar belum?"),
        ("user", "belum kak"),
        ("guru_dosen_asing", "kamu dewasa banget loh buat umur segitu"),
        ("user", "ah kakak bisa aja"),
        ("guru_dosen_asing", "eh foto kamu dong yang lagi selfie"),
        ("user", "yang mana kak?"),
        ("guru_dosen_asing", "tapi jangan bilang siapa-siapa ya, ini rahasia kita berdua aja"),
        ("guru_dosen_asing", "cuma aku yang peduli sama kamu, temen-temenmu gak ngerti kamu"),
    ],
    "Pacaran remaja wajar (teman sebaya)": [
        ("teman_saudara", "sayang lagi ngapain?"),
        ("user", "lagi belajar matematika, kamu?"),
        ("teman_saudara", "aku lagi bosen di rumah"),
        ("teman_saudara", "kangen kamu deh"),
        ("user", "aku juga kangen kamu beb"),
        ("teman_saudara", "besok jadi ke sekolah bareng?"),
    ],
    "Percakapan orang tua-anak wajar": [
        ("ortu", "nak, udah makan belum?"),
        ("user", "udah ma, tadi di sekolah"),
        ("ortu", "kamu pinter deh, nilai matematikanya bagus"),
        ("user", "makasih ma"),
        ("ortu", "papa mau beliin kamu sepatu baru mau warna apa?"),
        ("user", "yang biru aja pa"),
        ("ortu", "kirim foto sepatunya dong biar papa liat model"),
    ],
    "Modus ketemuan berbayar": [
        ("guru_dosen_asing", "kamu kalo aku bayar 1 jt mau gk buat ikut aku malem ini"),
        ("user", "maksud lo?"),
        ("guru_dosen_asing", "buat main ajaa"),
        ("guru_dosen_asing", "kamu tinggal duduk manis"),
    ],
    "Diskredit orang tua (P7)": [
        ("guru_dosen_asing", "orang tua kamu memang strict banget ya"),
        ("guru_dosen_asing", "mereka emang gk tau cara jadi orang tua ya"),
        ("guru_dosen_asing", "masa anaknya mau di batesin terus sih"),
        ("user", "aku jadi sebel deh sama ortu ku"),
        ("user", "aku lebih nyaman sama kamu"),
    ],
}


# ============================================================
# Callback untuk reset (biar dropdown ikut ter-reset)
# ============================================================
def reset_all():
    st.session_state.messages = [
        {'role': 'guru_dosen_asing', 'text': ''},
        {'role': 'user', 'text': ''},
    ]
    st.session_state.scenario_select = "-- Kosong (input manual) --"
    st.session_state.last_scenario = "-- Kosong (input manual) --"


def apply_scenario():
    """Trigger saat selectbox berubah — replace messages dengan skenario terpilih."""
    scenario = st.session_state.scenario_select
    if scenario != "-- Kosong (input manual) --":
        st.session_state.messages = [
            {'role': role, 'text': text}
            for role, text in PREDEFINED_SCENARIOS[scenario]
        ]
    st.session_state.last_scenario = scenario


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🛡️ Deteksi Cyber Grooming 7P")
    st.markdown("---")

    st.subheader("📋 Skenario Contoh")

    # Init state
    if 'scenario_select' not in st.session_state:
        st.session_state.scenario_select = "-- Kosong (input manual) --"

    st.selectbox(
        "Pilih skenario predefined:",
        list(PREDEFINED_SCENARIOS.keys()),
        key='scenario_select',
        on_change=apply_scenario,
    )

    st.markdown("---")

    with st.expander("ℹ️ Tentang 7P Framework"):
        st.markdown("""
        **7P** (Child Rescue Coalition) — 7 variabel indikator grooming:
        - **P1 Praise** — pujian berlebihan, love-bombing
        - **P2 Precocious** — pembicaraan romantis/seksual prematur
        - **P3 Photo** — meminta/mengirim foto
        - **P4 Privacy** — minta merahasiakan
        - **P5 Pressure** — tekanan, ancaman, sextortion
        - **P6 Presents** — iming-iming hadiah/materi
        - **P7 Pulling away** — isolasi korban
        """)

    with st.expander("⚙️ Aturan Klasifikasi"):
        st.markdown(f"""
        **MERAH** jika minimal 1 dari:
        - Ada kombinasi kritis (P2+P4, P2+P5, P3+P4, P3+P5, P5+P7)
        - Ada frasa literal skor efektif = 3 (red flag)
        - 3+ variabel P dengan bukti berulang
        - Total skor ≥ {THRESHOLD_MERAH}

        **KUNING** jika minimal 1 dari:
        - Ada kombinasi waspada (P1+P3, P2+P3, P5+P6)
        - 2+ variabel P dengan skor ≥ 2
        - Ada frasa skor efektif ≥ 2
        - Total skor ≥ {THRESHOLD_KUNING}

        **HIJAU** — sisanya
        """)

    with st.expander("🎭 Modifier Role"):
        st.markdown("""
        Skor frasa disesuaikan berdasarkan sender:
        - **Orang tua**: −1
        - **Teman/saudara**: 0 (baseline)
        - **Guru/dosen/asing**: +1
        - **User**: (tidak di-score, hanya konteks)

        Berlaku hanya untuk frasa `role_sensitive=TRUE`.
        """)


# ============================================================
# Main
# ============================================================
st.title("Simulasi Deteksi Cyber Grooming 7P")
st.caption(
    "Aplikasi deteksi pola cyber grooming berdasarkan framework 7P dengan "
    "*context-aware scoring* berbasis role pengirim + semantic matching."
)

# Load lexicon
try:
    lex = get_lexicon()
except Exception as e:
    st.error(f"Gagal load lexicon: {e}")
    st.stop()

# Status bar + toggle semantic (top of main area, biar terlihat)
col_status, col_toggle = st.columns([3, 1])
with col_status:
    st.caption(f"✅ Lexicon loaded: {len(lex)} phrase entries")
with col_toggle:
    use_semantic = st.toggle(
        "Semantic",
        value=SEMANTIC_AVAILABLE,
        disabled=not SEMANTIC_AVAILABLE,
        help="Deteksi padanan makna (bukan hanya literal). "
             "First-run download model ~470MB.",
    )
    if not SEMANTIC_AVAILABLE:
        st.caption("⚠️ Install `sentence-transformers`")

st.markdown("---")

# ============================================================
# Input area
# ============================================================
st.subheader("📝 Input Percakapan")
st.caption("Pilih role & tulis pesan. Role `user` = anak/korban (tidak dianalisis, hanya konteks).")

# Init messages state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'guru_dosen_asing', 'text': ''},
        {'role': 'user', 'text': ''},
        {'role': 'guru_dosen_asing', 'text': ''},
        {'role': 'user', 'text': ''},
    ]

# Kontrol baris
col_add, col_clear, _ = st.columns([1, 1, 4])
with col_add:
    if st.button("➕ Tambah baris"):
        last_role = st.session_state.messages[-1]['role'] if st.session_state.messages else 'user'
        next_role = 'user' if last_role != 'user' else 'guru_dosen_asing'
        st.session_state.messages.append({'role': next_role, 'text': ''})
        st.rerun()
with col_clear:
    st.button("🗑️ Kosongkan semua", on_click=reset_all)

st.write("")

# Render input tiap pesan
for i, msg in enumerate(st.session_state.messages):
    col1, col2, col3 = st.columns([2, 8, 0.5])
    with col1:
        role = st.selectbox(
            f"Role {i+1}",
            options=list(ROLE_LABELS.keys()),
            format_func=lambda r: ROLE_LABELS[r],
            key=f"role_{i}",
            index=list(ROLE_LABELS.keys()).index(msg['role']),
            label_visibility="collapsed",
        )
        st.session_state.messages[i]['role'] = role
    with col2:
        text = st.text_input(
            f"Pesan {i+1}",
            value=msg['text'],
            key=f"text_{i}",
            label_visibility="collapsed",
            placeholder=f"Tulis pesan {i+1}...",
        )
        st.session_state.messages[i]['text'] = text
    with col3:
        if st.button("✕", key=f"del_{i}", help="Hapus baris"):
            st.session_state.messages.pop(i)
            st.rerun()

st.markdown("---")

# ============================================================
# Analisis
# ============================================================
analyze = st.button("🔍 Analisis Percakapan", type="primary", use_container_width=True)

if analyze:
    conversation = [
        {'text': m['text'].strip(), 'sender_role': m['role']}
        for m in st.session_state.messages
        if m['text'].strip()
    ]

    if not conversation:
        st.warning("Tidak ada pesan untuk dianalisis. Silakan isi minimal 1 pesan.")
    else:
        # Load semantic matcher kalau aktif
        sem_matcher = None
        if use_semantic:
            with st.spinner("Loading semantic model (first-run bisa 2-5 menit)..."):
                sem_matcher = get_semantic_matcher(lex)

        with st.spinner("Menganalisis percakapan..."):
            evidence = accumulate_evidence(conversation, lex, semantic_matcher=sem_matcher)
            result = classify_conv(evidence)

        # ---- BANNER KLASIFIKASI ----
        cls = result['classification']
        color = CLASSIFICATION_COLORS[cls]
        emoji = CLASSIFICATION_EMOJI[cls]

        st.markdown(
            f"""
            <div style="background-color: {color}22; border-left: 8px solid {color};
                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: {color}; margin: 0;">{emoji} Hasil: {cls}</h2>
                <p style="margin: 8px 0 0 0; font-size: 1.1em; color: #1F2937;">
                    Total skor: <b>{result['total_score']}</b> |
                    Match: <b>{result['total_matches']}</b> |
                    Variabel P: <b>{result['n_p_detected']} / 7</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 2 kolom: Chat bubble + Chart ----
        col_chat, col_chart = st.columns([1.3, 1])

        with col_chat:
            st.subheader("💬 Percakapan")

            matches_by_msg = {}
            for m in result['all_matches']:
                matches_by_msg.setdefault(m.message_idx, []).append(m)

            # Build chat bubbles
            bubbles_html = ['<div class="chat-container">']
            for i, msg in enumerate(conversation):
                role = msg['sender_role']
                role_label = ROLE_LABELS[role]

                # Determine side + bubble class
                if role == 'user':
                    side = 'right'
                    bubble_class = 'bubble user'
                    role_color = '#1E40AF'
                else:
                    side = 'left'
                    if role == 'guru_dosen_asing':
                        bubble_class = 'bubble sender sender-asing'
                        role_color = '#DC2626'
                    elif role == 'ortu':
                        bubble_class = 'bubble sender sender-ortu'
                        role_color = '#2563EB'
                    else:
                        bubble_class = 'bubble sender sender-teman'
                        role_color = '#6B7280'

                # Highlight text
                text_html = msg['text']
                msg_matches = matches_by_msg.get(i, [])
                literal_matches = [m for m in msg_matches if m.source == 'literal']
                semantic_matches = [m for m in msg_matches if m.source == 'semantic']

                # Literal highlight in-line
                if literal_matches:
                    sorted_lit = sorted(literal_matches, key=lambda m: -m.span[0])
                    for m in sorted_lit:
                        start, end = m.span
                        if start >= len(text_html) or end > len(text_html) or start >= end:
                            continue
                        p_color = P_COLORS[m.kode_p]
                        highlight = (
                            f'<span style="background-color: {p_color}33; '
                            f'border-bottom: 2px solid {p_color}; padding: 1px 3px; '
                            f'border-radius: 3px; font-weight: 500;">'
                            f'{text_html[start:end]}'
                            f'<sub style="color: {p_color}; font-weight: bold; margin-left: 3px; font-size: 0.7em;">'
                            f'{m.kode_p}:{m.skor_efektif}</sub>'
                            f'</span>'
                        )
                        text_html = text_html[:start] + highlight + text_html[end:]

                # Semantic badges di bawah bubble
                sem_html = ''
                if semantic_matches:
                    badges = []
                    for m in semantic_matches:
                        p_color = P_COLORS[m.kode_p]
                        sim_pct = int(m.similarity * 100)
                        badges.append(
                            f'<span style="display: inline-block; background: {p_color}22; '
                            f'border: 1px dashed {p_color}; padding: 1px 6px; '
                            f'border-radius: 4px; font-size: 0.78em; margin: 2px 4px 0 0; '
                            f'color: {p_color};">'
                            f'~{m.frasa_display} <b>{m.kode_p}:{m.skor_efektif}</b> '
                            f'({sim_pct}%)</span>'
                        )
                    sem_html = f'<div class="sem-badges">{"".join(badges)}</div>'

                bubble = (
                    f'<div class="msg-row {side}">'
                    f'<div class="{bubble_class}">'
                    f'<div class="role-label" style="color: {role_color};">{role_label}</div>'
                    f'<div>{text_html}</div>'
                    f'{sem_html}'
                    f'</div>'
                    f'</div>'
                )
                bubbles_html.append(bubble)
            bubbles_html.append('</div>')
            st.markdown('\n'.join(bubbles_html), unsafe_allow_html=True)

        with col_chart:
            st.subheader("📊 Skor per Variabel P")

            p_totals = [result['per_p_summary'][p]['total_score'] for p in P_CODES]
            p_labels_full = [f"{p} {P_NAMES[p]}" for p in P_CODES]
            colors = [P_COLORS[p] for p in P_CODES]

            fig = go.Figure(go.Bar(
                x=p_totals, y=p_labels_full, orientation='h',
                marker_color=colors, text=p_totals, textposition='outside',
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=20, t=20, b=20),
                xaxis_title="Total skor efektif",
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            fig.update_xaxes(gridcolor='#F3F4F6')
            st.plotly_chart(fig, use_container_width=True)

            st.metric(
                "Total Skor Akumulasi",
                result['total_score'],
                delta=f"Threshold merah: {THRESHOLD_MERAH}",
                delta_color="off",
            )

        st.markdown("---")

        # ---- Reasoning + Kombinasi ----
        col_reason, col_pairs = st.columns(2)

        with col_reason:
            st.subheader("📋 Alasan Klasifikasi")
            for r in result['reasoning']:
                st.markdown(f"- {r}")

        with col_pairs:
            st.subheader("🔗 Kombinasi Terdeteksi")
            if result['critical_pairs']:
                st.markdown("**🔴 Kombinasi Kritis (MERAH):**")
                for pa, pb in result['critical_pairs']:
                    st.markdown(f"- **{pa}+{pb}** — {P_NAMES[pa]} + {P_NAMES[pb]}")
            if result.get('caution_pairs'):
                st.markdown("**🟡 Kombinasi Waspada (KUNING):**")
                for pa, pb, alasan in result['caution_pairs']:
                    st.markdown(f"- **{pa}+{pb}** — {alasan}")
            if not result['critical_pairs'] and not result.get('caution_pairs'):
                st.info("Tidak ada kombinasi berbahaya terdeteksi.")

        # ---- Detail (expandable) ----
        with st.expander("🔍 Detail semua frasa terdeteksi"):
            if result['all_matches']:
                match_data = []
                for m in result['all_matches']:
                    match_data.append({
                        'Baris': m.message_idx + 1,
                        'Kode P': m.kode_p,
                        'Frasa': m.frasa_display,
                        'Skor dasar': m.skor_dasar,
                        'Skor efektif': m.skor_efektif,
                        'Sumber': m.source,
                        'Sim': f"{m.similarity:.2f}" if m.source == 'semantic' else '1.00',
                        'Role sender': ROLE_LABELS[m.sender_role].split(' ', 1)[1],
                    })
                df_matches = pd.DataFrame(match_data)
                st.dataframe(df_matches, use_container_width=True, hide_index=True)
                st.caption(
                    "**Sumber**: `literal` = match kata persis (dengan slang normalization); "
                    "`semantic` = match kemiripan makna (badge dashed di bubble)."
                )
            else:
                st.info("Tidak ada frasa lexicon yang terdeteksi.")

# Footer
st.markdown("---")
st.caption(
    "🎓 Sistem prototype deteksi cyber grooming — untuk validasi ahli & riset akademik. "
    "Bukan pengganti judgment psikolog profesional."
)
