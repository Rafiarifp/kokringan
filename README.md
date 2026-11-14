# KOKRINGAN — PUSSIBERAD Command Center

**Sistem Intelijen Siber Sosial & Komando Kognitif Terpadu** berbasis *Open-Source Intelligence* (OSINT) dan *Natural Language Processing* (IndoBERT) untuk pemantauan ruang informasi nasional, deteksi dini operasi informasi, serta pemetaan risiko narasi lintas platform.

---

## 1. Urgensi & Latar Belakang Strategis

### Dinamika Ancaman di Domain Kognitif
Ruang informasi digital Indonesia kini menjadi arena strategis tempat opini publik, stabilitas sosial, dan kepercayaan terhadap institusi dapat dipengaruhi secara masif dalam hitungan jam. Eskalasi isu strategis—mulai dari kebijakan ekonomi, hukum, keamanan, hingga isu sosial—sering kali disertai oleh:
1. **Amplifikasi Buatan (*Coordinated Inauthentic Behavior* / CIB):** Penyebaran narasi seragam secara terkoordinasi oleh jaringan akun untuk menciptakan ilusi konsensus publik.
2. **Distorsi Narasi & Sarkasme Kontekstual:** Kritik atau propaganda terselubung menggunakan bahasa gaul (*slang*), singkatan lokal, dan sindiran sarkastik yang sering salah diklasifikasikan sebagai sentimen positif oleh perangkat analisis konvensional.
3. **Kesenjangan antara Gema Media Sosial dan Minat Publik Nyata:** Lonjakan percakapan di media sosial belum tentu mencerminkan perhatian masyarakat luas, sehingga diperlukan pembanding objektif dari indeks pencarian publik (Google Trends).

### Mengapa KOKRINGAN Dibutuhkan?
Perangkat *social listening* komersial umumnya berbiaya tinggi, berbasis *cloud* eksternal (berisiko membocorkan kata kunci pemantauan intelijen), kurang akurat pada bahasa Indonesia non-baku, serta tidak memetakan temuan ke dalam kerangka kerja pertahanan informasi standar.

**KOKRINGAN** hadir menjawab kebutuhan tersebut dengan menyediakan:
* **Kedaulatan Data (*Local-First / On-Premise*):** Seluruh proses penarikan data, inferensi AI, dan penyimpanan hasil berjalan di infrastruktur lokal tanpa mengirim data mentah ke pihak ketiga.
* **Konteks Bahasa Indonesia Mendalam:** Menggabungkan kamus normalisasi *slang* lokal dengan model *deep learning* **IndoBERT** (Sentimen & Sarkasme) serta **Multilingual MiniLM** untuk pengelompokan semantik.
* **Standar Evaluasi Operasi Informasi (DISARM):** Mengonversi ribuan data tak terstruktur menjadi indikator risiko terukur, deteksi duplikasi lintas akun, peringatan anomali lonjakan volume/sentimen, dan ringkasan intelijen eksekutif yang siap ditindaklanjuti.

---

## 2. Peta Lengkap Struktur Direktori & File Proyek

Sistem ini mengintegrasikan modul akuisisi OSINT, fondasi data, mesin inferensi AI, analitik tren/anomali, orkestrator pipeline, dan antarmuka *command center* dalam satu repositori terpadu:

```text
kokringan-cyberlens/
├── README.md                          # Dokumentasi arsitektur, urgensi, & panduan operasional
├── .env                               # Kredensial API YouTube & Telegram (jangan di-commit)
│
├── config/                            # Konfigurasi target pemantauan & kamus bahasa
│   ├── targets.json                   # Daftar topik, query YouTube, RSS berita, Telegram, & Google Trends
│   └── slang_dict.json                # Kamus normalisasi kata tidak baku / slang Bahasa Indonesia
│
├── ingestion/                         # Lapisan Akuisisi OSINT Multi-Platform
│   ├── __init__.py
│   ├── main.py                        # Orkestrator scraping paralel (asyncio), CLI, & incremental merge
│   ├── scrape.py                      # Scraper YouTube API v3, News RSS, Telegram, Google Trends, & X
│   └── data_preprocessing.py          # Normalisasi waktu WIB, pembersihan teks, & konversi slang
│
├── core/                              # Lapisan Fondasi & Standardisasi Data
│   ├── __init__.py
│   └── data_foundation.py             # Adapter skema kanonik (25 kolom), waktu UTC, & ID SHA-256
│
├── nlp/                               # Lapisan Inferensi Deep Learning
│   ├── __init__.py
│   └── inference_engine.py            # IndoBERT Sentiment, IndoBERT Sarcasm, & MiniLM Embeddings
│
├── analytics/                         # Lapisan Analitik Narasi, Tren, Anomali, & DISARM
│   ├── __init__.py
│   ├── narrative_and_trends.py        # Agregasi sentimen, tren waktu, DBSCAN clustering, & CIB/DISARM
│   └── anomaly_engine.py              # Deteksi anomali volume, lonjakan lintas platform, & pergeseran sentimen
│
├── pipeline/                          # Lapisan Orkestrasi Backend End-to-End
│   ├── __init__.py
│   └── orchestrator.py                # Eksekusi DAG satu-lintasan & sintesis ringkasan intelijen
│
├── app/                               # Lapisan Antarmuka Komando (Streamlit UI)
│   ├── dashboard.py                   # Command Center 4 Tab (Tema Terang/Gelap WCAG AA)
│   └── ingestion_control.py           # Panel kontrol sidebar untuk scraping & analisis 1-klik
│
├── tools/                             # Lapisan Pengujian & Quality Assurance
│   └── evaluate.py                    # Evaluasi CLI akurasi, F1-score, & confusion matrix model
│
└── data/                              # Penyimpanan Artefak Dataset & Laporan
    ├── master_osint_data.csv          # Dataset hasil scraping terstandarisasi
    ├── master_osint_enriched.csv      # Dataset hasil pengayaan NLP & klaster narasi
    ├── cyberlens_report.json          # Laporan terstruktur (ringkasan, klaster, anomali, DISARM)
    ├── trends_interest.csv            # Deret waktu skor minat pencarian Google Trends (0-100)
    ├── latest_run.log                 # Log eksekusi scraping & analisis terakhir
    └── osint_session.session          # File sesi autentikasi lokal Telethon (Telegram)
```

---

## 3. Rincian Fungsi Setiap File & Modul

### A. Lapisan Konfigurasi (`config/`)
* **`config/targets.json`**: Pusat kendali target pemantauan yang memuat pengaturan global (`timeframe`, batas artikel/video/komentar) serta daftar kata kunci terstruktur per domain isu nasional (Ekonomi, Politik, Hukum, Keamanan, Sosial), daftar URL RSS portal berita nasional, daftar channel publik Telegram, dan daftar kata kunci Google Trends.
* **`config/slang_dict.json`**: Kamus pemetaan ribuan singkatan dan bahasa gaul Indonesia (misalnya `"yg"` $\rightarrow$ `"yang"`, `"ga"` $\rightarrow$ `"tidak"`, `"judol"` $\rightarrow$ `"judi online"`) untuk meningkatkan akurasi pembacaan konteks oleh model NLP.

### B. Lapisan Akuisisi OSINT (`ingestion/`)
* **`ingestion/scrape.py`**: Modul penarik data multi-sumber dengan manajemen *timeout* dan *rate-limit*:
  * `scrape_youtube_by_keywords()`: Mencari video terkini via YouTube Data API v3 dan menarik komentar teratas secara efisien.
  * `scrape_news()`: Mengambil *feed* RSS dari portal berita nasional (Detik, Tempo, ANTARA, CNN Indonesia, CNBC Indonesia, Liputan6, Republika, Okezone, dll.) dan mengekstrak isi artikel secara paralel menggunakan `ThreadPoolExecutor` + `trafilatura`.
  * `scrape_telegram_channels()`: Membaca pesan dan jumlah *views* dari channel Telegram menggunakan satu sesi bersama (`Telethon`) dengan mekanisme *early-break* saat melewati batas waktu.
  * `scrape_google_trends_series()`: Menarik indeks minat pencarian Google Trends (`0-100`) melalui `pytrends` dengan dukungan *batching* otomatis (menggunakan kata kunci jangkar agar skala antar-batch tetap konsisten), dukungan operator gabungan (`polisi/polri` $\rightarrow$ `polisi + polri`), serta *patch* kompatibilitas `urllib3`.
  * `scrape_x()`: Dukungan opsional pengambilan cuitan X/Twitter berbasis *cookie* (`twikit`).
* **`ingestion/data_preprocessing.py`**: Membersihkan artefak HTML/URL/mention, menormalisasi bahasa gaul dengan pola *regex* terkompilasi berkecepatan tinggi, serta menyeragamkan zona waktu ke `Asia/Jakarta` (WIB).
* **`ingestion/main.py`**: Orkestrator pengumpulan data berbasis `asyncio` yang menjalankan seluruh scraper secara konkuren, menggabungkan hasil baru dengan `data/master_osint_data.csv` (*incremental merge* tanpa menduplikasi komentar unik), serta menyediakan argumen CLI (`--sources`, `--timeframe`, `--replace-existing`).

### C. Lapisan Fondasi Data (`core/`)
* **`core/data_foundation.py`**: Gerbang standardisasi data (`normalize_posts_df`) yang menyelaraskan variasi nama kolom, mengonversi seluruh *timestamp* ke format kanonik UTC, mengelompokkan outlet berita ke kategori induk `platform = "News"` (sambil menyimpan nama medianya di `platform_detail`), membersihkan teks untuk tokenizer dan *embedding*, serta membangkitkan ID unik deterministik berbasis hash SHA-256 (`make_post_id`).

### D. Lapisan Inferensi Neural (`nlp/`)
* **`nlp/inference_engine.py`**: Mesin inferensi *deep learning* dengan manajemen memori aktif (*lazy loading* dan pembebasan VRAM GPU otomatis antar-tahap agar dapat berjalan pada GPU berkapasitas terbatas maupun CPU):
  * `SentimentRunner`: Klasifikasi sentimen Bahasa Indonesia (`Positif`, `Netral`, `Negatif`) menggunakan `mdhugol/indonesia-bert-sentiment-classification`.
  * `SarcasmRunner`: Deteksi sarkasme menggunakan `w11wo/indobert-base-p1-twitter-indonesia-sarcastic` dengan ambang keyakinan terkalibrasi.
  * `interpret_combined_result()`: Menggabungkan output sentimen dan sarkasme ke dalam 8 kelas interpretasi kontekstual (seperti `possible_sarcastic_criticism` dan `negative_with_sarcasm`) serta menandai unggahan yang memerlukan tinjauan analis (`needs_review`).
  * `EmbeddingRunner`: Menghasilkan vektor representasi semantik ternormalisasi L2 berdimensi $N \times 384$ menggunakan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### E. Lapisan Analitik & Deteksi Ancaman (`analytics/`)
* **`analytics/narrative_and_trends.py`**:
  * Agregasi statistik sentimen global, per platform, per topik, dan tren harian (`aggregate_sentiment`).
  * Konstruksi deret waktu kontinu beserta *moving average* dan arah pergerakan tren (`build_time_series`, `analyze_keyword_trend`).
  * Pengelompokan klaster narasi semantik berbasis **Cosine DBSCAN** (`detect_narratives`) dan pemilihan judul klaster perwakilan (*medoid*).
  * Evaluasi *Coordinated Inauthentic Behavior* (CIB) dan perhitungan skor risiko kerangka **DISARM** ($0.0\text{--}10.0$) beserta pemetaan TTP (`TA08 · T0015`, `TA11 · T0049`, `TA12 · T0050`).
* **`analytics/anomaly_engine.py`**: Mesin deteksi peringatan dini (`detect_all_anomalies`) yang memantau lonjakan volume tidak wajar pada satu platform (*Z-score* & lonjakan persentase), lonjakan serentak lintas platform (`cross_platform_surge`), lonjakan rasio sentimen negatif (`negative_sentiment_spike`), serta pembalikan polaritas sentimen (`sentiment_polarity_reversal`).

### F. Lapisan Orkestrator Pipeline (`pipeline/`)
* **`pipeline/orchestrator.py`**: Menjalankan seluruh rangkaian mulai dari pembacaan `data/master_osint_data.csv`, inferensi NLP, analitik narasi/anomali/DISARM, hingga penyusunan ringkasan intelijen eksekutif berbahasa Indonesia (`generate_intelligence_summary`), lalu menyimpan hasilnya ke `data/master_osint_enriched.csv` dan `data/cyberlens_report.json`.

### G. Lapisan Antarmuka & Evaluasi (`app/` & `tools/`)
* **`app/dashboard.py`**: Aplikasi web **KOKRINGAN Command Center** berbasis Streamlit dengan desain *Bento Kinetic*, dukungan mode Terang/Gelap kontras tinggi, *caching* otomatis berbasis perubahan file, serta **4 Tab Analitik Utama**.
* **`app/ingestion_control.py`**: Komponen panel kendali di *sidebar* dashboard yang memungkinkan operator memilih sumber data, mode pembaruan (*Update Data* vs *Delete Old Data*), serta rentang waktu scraping, lalu menjalankan *scraping* dan analisis NLP secara otomatis dalam satu klik.
* **`tools/evaluate.py`**: Skrip evaluasi mandiri berbasis CLI untuk menguji akurasi, *precision*, *recall*, *F1-score*, dan *confusion matrix* model sentimen terhadap dataset uji berlabel manual (`ground_truth.csv`).

---

## 4. Arsitektur & Alur Kerja End-to-End

```text
+-----------------------------------------------------------------------------------+
| 1. KONFIGURASI & AKUISISI OSINT (config/targets.json + ingestion/)                |
|    [YouTube API v3]  [News RSS + Trafilatura]  [Telegram]  [Google Trends]  [X]   |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     v                                         v
      [ data/master_osint_data.csv ]            [ data/trends_interest.csv ]
                     |                                         |
                     v                                         |
+---------------------------------------------------+          |
| 2. DATA FOUNDATION (core/data_foundation.py)      |          |
|    - Normalisasi Skema Kanonik & Timestamp UTC    |          |
|    - Pemetaan Outlet Berita -> platform="News"    |          |
|    - ID Deterministik SHA-256 & Pembersihan Teks  |          |
+--------------------+------------------------------+          |
                     |                                         |
                     v                                         |
+---------------------------------------------------+          |
| 3. NEURAL INFERENCE ENGINE (nlp/inference_engine) |          |
|    - IndoBERT Sentiment  -> Unload VRAM           |          |
|    - IndoBERT Sarcasm    -> Unload VRAM           |          |
|    - Gabungan 8 Kelas Interpretasi & Flag Review  |          |
|    - MiniLM Embeddings (N x 384) -> Unload VRAM   |          |
+--------------------+------------------------------+          |
                     |                                         |
                     v                                         |
+---------------------------------------------------+          |
| 4. ANALYTICS & ORCHESTRATOR (analytics + pipeline)|          |
|    - Klastering Narasi Semantik (Cosine DBSCAN)   |          |
|    - Deteksi Anomali Volume, Lintas Platform, &   |          |
|      Pergeseran Sentimen                          |          |
|    - Evaluasi CIB & Skor Risiko DISARM (0-10)     |          |
|    - Sintesis Ringkasan Intelijen Eksekutif (ID)  |          |
+---------+---------------------------+-------------+          |
          |                           |                        |
          v                           v                        |
[ master_osint_enriched.csv ] [ cyberlens_report.json ]        |
          |                           |                        |
          +-------------+-------------+------------------------+
                        |
                        v
+-----------------------------------------------------------------------------------+
| 5. COMMAND CENTER DASHBOARD (app/dashboard.py & app/ingestion_control.py)         |
|    - Sidebar: Filter Global, Unduh CSV, & Panel Kontrol Scraping + NLP 1-Klik     |
|    - Tab 01: Ringkasan Eksekutif & Tren Lintas Platform                           |
|    - Tab 02: Semantik, WordCloud, Klaster Narasi, & Sampel Sarkasme               |
|    - Tab 03: Deteksi Anomali, Skor Ancaman Kognitif DISARM, & Bukti CIB           |
|    - Tab 04: Pemantauan Minat Publik Google Trends & Korelasi vs. Media Sosial    |
+-----------------------------------------------------------------------------------+
```

---

## 5. Fitur Utama Command Center (4 Tab Operasional)

1. **Tab 01 · Ringkasan Eksekutif**
   * Menampilkan kartu KPI utama (Total Unggahan, Total Engagement, Persentase Sentimen Negatif, Indikasi Sarkasme, dan Aktor Paling Aktif).
   * Grafik tren interaktif (*Volume Postingan* atau *Total Engagement*) dengan agregasi per 3 jam, per 6 jam, atau harian, baik secara gabungan maupun per platform.
   * Distribusi sentimen per platform, komposisi sentimen keseluruhan, peringkat 10 topik teratas, dan daftar konten teratas tiap platform.
2. **Tab 02 · Semantik dan Kluster Narasi**
   * Menyajikan ringkasan situasi otomatis, temuan utama, dan rekomendasi tindakan operasional hasil sintesis sistem.
   * *WordCloud* terpisah untuk narasi **Positif** dan **Negatif** guna melihat kata kunci dominan secara visual.
   * Tabel **Klaster Narasi Dominan** hasil pengelompokan semantik DBSCAN beserta sampel unggahan yang terindikasi mengandung sarkasme.
3. **Tab 03 · Deteksi Anomali dan DISARM**
   * **Indeks Ancaman Kognitif ($0.0\text{--}10.0$)** dengan status kesiapsiagaan (*Normal*, *Waspada*, *Kritis*) dan matriks teknik DISARM TTP (`TA08 · T0015`, `TA11 · T0049`, `TA12 · T0050`).
   * Indikator dan peringkat akun suspek **CIB** (*Coordinated Inauthentic Behavior*), daftar peringatan anomali lonjakan volume/sentimen, kotak bukti duplikasi teks identik lintas akun, serta tabel eksplorasi 300 rekaman terbaru.
4. **Tab 04 · Google Trends & Minat Publik**
   * Memantau indeks minat pencarian relatif ($0\text{--}100$) masyarakat Indonesia, lengkap dengan panel penarikan/pembaruan kata kunci langsung dari antarmuka.
   * Menampilkan grafik multi-garis tren minat (dengan opsi penghalusan kurva *Moving Average* 3 titik), peta panas (*heatmap*) intensitas pencarian per tanggal, grafik **korelasi silang antara Google Trends dan volume percakapan media sosial**, serta tabel deteksi lonjakan (*Z-score* & momentum).

---

## 6. Instruksi Instalasi & Konfigurasi

### Prasyarat Sistem
* **Python:** Versi `3.10` atau `3.11` direkomendasikan.
* **Perangkat Keras:** RAM minimal $8\text{ GB}$ (disarankan $16\text{ GB}$). Mendukung eksekusi pada CPU maupun GPU NVIDIA CUDA (manajemen memori otomatis membebaskan VRAM setelah setiap model selesai dijalankan).
* **Koneksi Internet:** Diperlukan untuk mengambil data OSINT dan mengunduh bobot model HuggingFace pada eksekusi pertama (selanjutnya bobot model tersimpan lokal di *cache*).

### Langkah 1: Setup Virtual Environment & Dependensi

```bash
# 1. Masuk ke direktori proyek
cd kokringan-cyberlens

# 2. Buat dan aktifkan virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux / macOS / WSL:
source .venv/bin/activate

# 3. Instal seluruh dependensi proyek
pip install --upgrade pip
pip install pandas numpy scikit-learn torch transformers sentence-transformers \
    streamlit plotly wordcloud python-dotenv feedparser trafilatura \
    google-api-python-client telethon pytrends twikit certifi
```

### Langkah 2: Konfigurasi Kredensial API (`.env`)

Buat file `.env` di direktori root proyek untuk menyimpan kunci akses API:

```env
# YouTube Data API v3 (diperoleh dari Google Cloud Console)
YOUTUBE_API_KEY=isi_api_key_youtube_anda

# Telegram API (diperoleh dari https://my.telegram.org)
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=isi_api_hash_telegram_anda
```

> **Catatan Login Telegram Pertama Kali:** Jika menggunakan sumber `Telegram`, jalankan `python -m ingestion.main --sources telegram` satu kali di terminal untuk memasukkan nomor telepon dan kode OTP. Setelah berhasil, file sesi `data/osint_session.session` akan tersimpan sehingga penarikan berikutnya berjalan otomatis tanpa perlu OTP ulang.

### Langkah 3: Penyesuaian Target Pemantauan (`config/targets.json`)

Buka `config/targets.json` untuk mengatur topik dan kata kunci yang ingin dipantau:
* **Optimasi Kuota YouTube:** Gunakan operator `|` (*OR*) dalam satu string query (contoh: `"judol | judi online | pinjol ilegal"`) agar dihitung sebagai 1 panggilan pencarian API.
* **Penggabungan Sinonim Google Trends:** Gunakan pemisah `/` pada kata kunci Google Trends (contoh: `"polisi/polri"` atau `"tentara/tni"`). Sistem otomatis menerjemahkannya menjadi operator `+` (*OR*) di Google Trends sehingga pencarian yang memuat kedua istilah tidak terhitung ganda dan menghemat kuota *batch*.

---

## 7. Panduan Penggunaan (Operasional)

Terdapat dua cara mengoperasikan sistem: **Melalui Antarmuka Web (Direkomendasikan)** atau **Melalui Terminal (CLI)**.

### Cara A: Operasional Terpadu via Dashboard Streamlit (1-Klik)

1. Jalankan server dashboard dari direktori root proyek:
   ```bash
   streamlit run app/dashboard.py
   ```
2. Buka peramban pada alamat `http://localhost:8501`.
3. **Menarik Data Baru & Analisis Otomatis:**
   * Buka bagian **OSINT Ingestion & Pipeline** di bagian bawah *sidebar* kiri.
   * Pilih sumber data yang diinginkan (`YouTube`, `News RSS`, `Telegram`, `Google Trends`).
   * Pilih **Mode Pengelolaan Dataset**:
     * *Update Data*: Mempertahankan data historis dan menambahkan rekaman terbaru tanpa duplikasi.
     * *Delete Old Data*: Menghapus dataset lama dan memulai pemantauan bersih dari awal.
   * Pilih **Timeframe scraping** (misalnya *7 hari terakhir*, *1 bulan terakhir*, atau *3 bulan terakhir*).
   * Klik **Update Data + Analisis**. Sistem akan menjalankan *scraping*, memicu pipeline NLP secara otomatis setelah *scraping* selesai, dan memperbarui seluruh grafik di dashboard.
4. **Eksplorasi & Ekspor Laporan:**
   * Gunakan filter **Sumber Platform**, **Kategori Sentimen**, **Filter Topik / Kata Kunci**, serta kotak centang **Hanya indikasi sarkasme atau perlu tinjauan** di *sidebar* untuk menyaring data secara *real-time*.
   * Klik **Unduh data terfilter (CSV)** di *sidebar* (atau di bawah tabel Tab 04 untuk data Google Trends) guna mengekspor data siap olah.

### Cara B: Operasional via Command Line (CLI)

Gunakan mode terminal jika ingin menjalankan tahapan tertentu secara terpisah atau menjadwalkannya melalui *cron job* / *Task Scheduler*:

```bash
# 1. Jalankan scraping semua sumber aktif (timeframe: 1w, 1m, atau 1q)
python -m ingestion.main --timeframe 1m

# 2. Jalankan scraping untuk sumber tertentu saja (contoh: Google Trends & News)
python -m ingestion.main --sources google_trends news --timeframe 1q --replace-existing

# 3. Jalankan pipeline NLP, klastering narasi, deteksi anomali, & evaluasi DISARM
python -m pipeline.orchestrator --input data/master_osint_data.csv \
    --output-csv data/master_osint_enriched.csv \
    --output-json data/cyberlens_report.json \
    --interval 3h

# 4. (Opsional) Evaluasi akurasi model sentimen terhadap dataset ground truth
python -m tools.evaluate --data data/ground_truth.csv --batch-size 16
```

---

## 8. Pemecahan Masalah Umum (*Troubleshooting*)

* **`ModuleNotFoundError: No module named 'core'` atau `'ingestion'`:** Pastikan seluruh perintah dijalankan dari direktori root proyek (`kokringan-cyberlens/`) dengan flag `-m` (contoh: `python -m pipeline.orchestrator`), bukan dari dalam subdirektori.
* **Kuota YouTube Habis (`HttpError 429 / quotaExceeded`):** YouTube Data API v3 memiliki kuota harian per proyek Google Cloud. Jika kuota harian tercapai, sistem otomatis mencatat peringatan di `data/latest_run.log` dan tetap melanjutkan penarikan dari sumber lain (*News RSS*, *Telegram*, *Google Trends*) tanpa menghentikan pipeline.
* **Google Trends Terkena Pembatasan Sementara (`HTTP 429`):** Tunggu beberapa menit sebelum menarik ulang, atau kurangi jumlah kata kunci dalam satu kali penarikan di `config/targets.json`.
* **Memori GPU Penuh (*CUDA Out of Memory*):** Perkecil parameter `sentiment_batch_size`, `sarcasm_batch_size`, atau `embedding_batch_size` saat memanggil `InferenceEngine.enrich_dataframe()` di `pipeline/orchestrator.py`.