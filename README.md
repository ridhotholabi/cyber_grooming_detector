# 🛡️ Deteksi Cyber Grooming 7P

Sistem deteksi pola cyber grooming berdasarkan framework 7P (Child Rescue Coalition)
dengan context-aware scoring berbasis role pengirim + semantic matching (sentence-transformer).

**Bahasa**: Indonesia (dengan slang normalization) + English
**Model semantic**: `paraphrase-multilingual-MiniLM-L12-v2`
**Lexicon**: ~750 phrase entries dari 7 variabel P

## 🚀 Cara Menjalankan

### 1. Lokal (rekomendasi untuk development)

```bash
# Clone repo
git clone <repo-url>
cd cyber-grooming-detection

# Install dependencies
pip install -r requirements.txt

# Jalankan
streamlit run demo_app.py
```

Buka `http://localhost:8501` di browser.

### 2. Deploy ke Streamlit Community Cloud (public, permanent URL)

**Prasyarat**: akun GitHub + akun Streamlit Cloud (gratis)

1. Push semua file (`demo_app.py`, `grooming_scorer.py`, `semantic_matcher.py`,
   `slang_dict.py`, `lexicon_7p_final_v5.xlsx`, `requirements.txt`,
   `.streamlit/config.toml`) ke repo GitHub public atau private.

2. Login ke [share.streamlit.io](https://share.streamlit.io), klik **"New app"**.

3. Isi form:
   - **Repository**: pilih repo kamu
   - **Branch**: `main` (atau branch yang berisi kode)
   - **Main file path**: `demo_app.py`
   - (opsional) **App URL**: custom subdomain, mis. `groomingdetect`

4. Klik **"Deploy"**. Streamlit Cloud akan:
   - Install dependencies dari `requirements.txt` (~3-5 menit first time)
   - Download semantic model saat aplikasi pertama diakses (~2-3 menit)
   - Kasih URL public: `https://<app-name>.streamlit.app`

5. **Share URL** ke psikolog / validator ahli.

**Batasan Free Tier:**
- 1GB RAM (model 470MB muat, tapi tidak banyak headroom)
- CPU limited
- Aplikasi sleep setelah tidak diakses beberapa jam (wake time ~30 detik)
- 1 aplikasi private, unlimited public

**Tips:**
- Kalau butuh lebih dari 1GB RAM, upgrade ke [Streamlit Cloud paid tier](https://streamlit.io/pricing) atau deploy sendiri (VPS).
- Model download hanya sekali (di-cache oleh Streamlit Cloud).

### 3. Deploy dengan ngrok (temporary public URL dari laptop)

Untuk demo cepat / live session tanpa upload ke Cloud.

**Prasyarat:**
- Aplikasi jalan di local (`streamlit run demo_app.py`)
- Install [ngrok](https://ngrok.com/download) (gratis)
- Sign up + dapat authtoken dari [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

**Langkah:**

```bash
# 1. Setup ngrok authtoken (sekali saja)
ngrok config add-authtoken YOUR_AUTHTOKEN

# 2. Terminal 1: jalankan Streamlit
streamlit run demo_app.py

# 3. Terminal 2: expose lewat ngrok
ngrok http 8501
```

Ngrok akan kasih URL seperti:
```
Forwarding    https://abc123.ngrok-free.app -> http://localhost:8501
```

Share URL ini ke validator. **URL berubah tiap restart ngrok** (kecuali pakai paid plan).

**Optional — password protection:**
```bash
ngrok http 8501 --basic-auth "user:password123"
```

**Batasan free tier ngrok:**
- 1 tunnel aktif
- 40 requests/menit
- URL berubah tiap restart
- Session timeout ~2 jam kalau tidak ada aktivitas

### Perbandingan Deploy

| Method | Setup | URL | Cost | Cocok untuk |
|---|---|---|---|---|
| Local | 2 menit | localhost | Free | Development, testing |
| Streamlit Cloud | 15 menit | Permanent public | Free | Share ke psikolog jangka panjang |
| ngrok | 5 menit | Temporary public | Free | Demo live, presentasi |

## 📁 Struktur File

```
├── demo_app.py                      # Streamlit UI (main)
├── grooming_scorer.py               # Core scoring engine
├── semantic_matcher.py              # Sentence-transformer wrapper
├── slang_dict.py                    # Kamus normalisasi slang Indonesia
├── lexicon_7p_final_v5.xlsx         # Lexicon 7P (~750 entries)
├── requirements.txt                 # Python dependencies
├── .streamlit/config.toml           # Theme config (white bg)
├── packages.txt                     # (empty — untuk system deps di Streamlit Cloud)
└── README.md
```

## 🧪 Testing & Reports

Untuk test batch skenario dari file `.txt`:

```bash
# Literal only
python test_from_txt.py lexicon_7p_final_v5.xlsx skenario_failing.txt

# Dengan semantic + generate report
python test_from_txt.py lexicon_7p_final_v5.xlsx skenario_failing.txt --semantic --report
```

Report otomatis di-generate sebagai `report_{skenario_stem}_DDMMYYYY.txt` untuk validasi ahli.

## ⚠️ Disclaimer

Sistem ini adalah **prototype riset akademik**. Bukan pengganti judgment psikolog profesional
atau alat forensik hukum. Hasil deteksi harus divalidasi oleh ahli sebelum tindakan lanjutan.
