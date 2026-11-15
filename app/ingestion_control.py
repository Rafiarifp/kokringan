"""Panel Streamlit untuk Delete Old Data / Update Data, konfigurasi API, dan pipeline analitik."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import streamlit as st


def _find_project_root() -> Path:
    """Cari root folder proyek berdasarkan struktur folder proyek."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "ingestion" / "main.py").is_file() and (candidate / "config" / "targets.json").is_file():
            return candidate
    return here.parent


ROOT_DIR = _find_project_root()
MAIN_SCRIPT = ROOT_DIR / "ingestion" / "main.py"
TARGETS_FILE = ROOT_DIR / "config" / "targets.json"
ENV_FILE = ROOT_DIR / ".env"
LOG_FILE = ROOT_DIR / "data" / "latest_run.log"

TIMEFRAME_LABELS = {
    "1w": "7 hari terakhir",
    "1m": "1 bulan terakhir",
    "1q": "3 bulan terakhir",
}
SOURCE_LABELS = {
    "youtube": "YouTube",
    "news": "News RSS",
    "telegram": "Telegram",
    "google_trends": "Google Trends",
    "instagram": "Instagram (API)",
    "threads": "Threads (API)",
    "x": "X (Twitter)",
}
DATASET_SOURCES = {"youtube", "news", "telegram", "x", "instagram", "threads"}

API_KEY_SPECS = [
    ("INSTAGRAM_API_KEY", "Instagram Scraper API Key", "Token API berbayar (Apify/RapidAPI/Graph) untuk Instagram"),
    ("THREADS_API_KEY", "Threads Scraper API Key", "Token API berbayar (Apify/RapidAPI/Graph) untuk Meta Threads"),
    ("YOUTUBE_API_KEY", "YouTube Data API v3 Key", "Opsional: timpa key YouTube di .env"),
    ("TELEGRAM_API_ID", "Telegram API ID", "Opsional: angka ID dari my.telegram.org"),
    ("TELEGRAM_API_HASH", "Telegram API Hash", "Opsional: hash dari my.telegram.org"),
]

NLP_INTERVAL = "3h"
ENRICHED_FILE = "data/master_osint_enriched.csv"
REPORT_FILE = "data/cyberlens_report.json"


def _load_targets() -> dict:
    if not TARGETS_FILE.exists():
        return {}
    try:
        return json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_default_sources() -> list[str]:
    cfg = _load_targets()
    enabled = [s for s in SOURCE_LABELS if cfg.get(s, {}).get("enabled", False)]
    return enabled or ["youtube", "news", "telegram"]


def _load_default_timeframe() -> str:
    value = _load_targets().get("settings", {}).get("timeframe", "1m")
    return value if value in TIMEFRAME_LABELS else "1m"


def _tail_log(max_lines: int = 18) -> str:
    if not LOG_FILE.exists():
        return "Menunggu output log..."
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else "Log masih kosong..."
    except Exception as exc:
        return f"Gagal membaca log: {exc}"


def _invalidate_previous_analysis() -> None:
    """Hapus artefak analisis lama sebelum NLP baru dimulai."""
    for filename in (ENRICHED_FILE, REPORT_FILE):
        stale_file = ROOT_DIR / filename
        try:
            stale_file.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Tidak dapat menghapus hasil NLP lama {filename}: {exc}"
            ) from exc


def _upsert_env_file(updates: dict[str, str]) -> None:
    """Simpan atau perbarui key di .env tanpa menghapus variabel lain yang sudah ada."""
    existing_lines: list[str] = []
    if ENV_FILE.exists():
        existing_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue

        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            val = remaining.pop(key)
            new_lines.append(f"{key}={val}")
        else:
            new_lines.append(line)

    for key, val in remaining.items():
        new_lines.append(f"{key}={val}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def render_api_config_panel() -> None:
    """Panel konfigurasi API key langsung dari sidebar Streamlit (aman untuk demo)."""
    try:
        from dotenv import load_dotenv
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
    except ImportError:
        pass

    ig_ready = bool(os.getenv("INSTAGRAM_API_KEY", "").strip())
    th_ready = bool(os.getenv("THREADS_API_KEY", "").strip())

    with st.sidebar.expander("Konfigurasi API & Kredensial", expanded=False):
        st.caption(
            f"Status Konektor Berbayar: "
            f"Instagram **{'Aktif' if ig_ready else 'Standby'}** · "
            f"Threads **{'Aktif' if th_ready else 'Standby'}**"
        )

        with st.form("api_credentials_form", clear_on_submit=False):
            inputs: dict[str, str] = {}
            for env_key, label, help_text in API_KEY_SPECS:
                current_val = os.getenv(env_key, "")
                inputs[env_key] = st.text_input(
                    label,
                    value=current_val,
                    type="password",
                    placeholder="Kosongkan jika belum ada API key",
                    help=help_text,
                )

            col_save, col_clear = st.columns([2, 1])
            with col_save:
                save_keys = st.form_submit_button("Simpan Kredensial", type="primary", use_container_width=True)
            with col_clear:
                clear_paid = st.form_submit_button("Reset IG/Threads", use_container_width=True)

        if save_keys:
            cleaned = {k: v.strip() for k, v in inputs.items()}
            for k, v in cleaned.items():
                os.environ[k] = v
            _upsert_env_file(cleaned)
            st.success("Kredensial berhasil disimpan ke memori sesi & file .env.")
            st.rerun()

        if clear_paid:
            reset_map = {"INSTAGRAM_API_KEY": "", "THREADS_API_KEY": ""}
            for k in reset_map:
                os.environ[k] = ""
            _upsert_env_file(reset_map)
            st.info("Kredensial Instagram & Threads dikembalikan ke mode Standby.")
            st.rerun()


def render_ingestion_panel() -> None:
    """Scraping -> NLP wajib -> dashboard hanya menampilkan hasil analisis terbaru."""
    for obsolete_key in (
        "osint_auto_analyze",
        "osint_interval",
        "osint_auto_analyze_widget",
        "osint_analysis_interval_widget",
    ):
        st.session_state.pop(obsolete_key, None)

    st.session_state.setdefault("osint_proc", None)
    st.session_state.setdefault("osint_stage", "idle")
    st.session_state.setdefault("osint_data_mode", "update")
    st.session_state.setdefault("osint_analysis_failed", False)
    st.session_state.setdefault("osint_error_message", "")
    st.session_state.setdefault("osint_log_handle", None)

    with st.sidebar.expander("OSINT Ingestion & Pipeline", expanded=True):
        selected_sources = st.multiselect(
            "Sumber data untuk rerun",
            options=list(SOURCE_LABELS),
            default=_load_default_sources(),
            format_func=lambda key: SOURCE_LABELS.get(key, key),
            key="osint_selected_sources",
            disabled=(st.session_state.osint_stage in ("scraping", "analyzing")),
            help="Sumber yang dipilih akan di-scrape sebelum analisis NLP dijalankan otomatis.",
        )

        missing_paid_keys = []
        if "instagram" in selected_sources and not os.getenv("INSTAGRAM_API_KEY", "").strip():
            missing_paid_keys.append("INSTAGRAM_API_KEY")
        if "threads" in selected_sources and not os.getenv("THREADS_API_KEY", "").strip():
            missing_paid_keys.append("THREADS_API_KEY")
        if missing_paid_keys:
            st.caption(
                f"Mode Standby API: isi `{', '.join(missing_paid_keys)}` pada panel **Konfigurasi API & Kredensial** di bawah untuk mengaktifkan."
            )

        data_mode = st.selectbox(
            "Mode Pengelolaan Dataset",
            options=["update", "delete_old"],
            format_func=lambda mode: {
                "update": "Update Data",
                "delete_old": "Delete Old Data",
            }[mode],
            key="osint_data_mode",
            disabled=(st.session_state.osint_stage in ("scraping", "analyzing")),
            help=(
                "Update Data mempertahankan histori dan menambahkan data baru. "
                "Delete Old Data mengganti dataset dengan hasil scraping terbaru."
            ),
        )
        replace_existing = data_mode == "delete_old"

        if replace_existing:
            st.caption("Delete Old Data: dataset lama diganti setelah scraping berhasil.")
        else:
            st.caption("Update Data: seluruh histori dipertahankan; data baru ditambahkan dan duplikasi dihapus.")

        timeframe = st.selectbox(
            "Timeframe scraping",
            options=list(TIMEFRAME_LABELS),
            format_func=lambda key: TIMEFRAME_LABELS[key],
            index=list(TIMEFRAME_LABELS).index(_load_default_timeframe()),
            key="osint_timeframe",
            disabled=(st.session_state.osint_stage in ("scraping", "analyzing")),
            help="1w = 7 hari, 1m = 1 bulan, 1q = 3 bulan terakhir.",
        )

        st.info("Analisis NLP, klaster narasi, dan DISARM wajib dijalankan otomatis setelah scraping berhasil.")

        proc = st.session_state.osint_proc
        is_running = proc is not None and proc.poll() is None
        has_dataset_source = bool(set(selected_sources).intersection(DATASET_SOURCES))

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            start_clicked = st.button(
                ("Delete Old Data · Scrape + Analisis" if replace_existing else "Update Data + Analisis")
                if not is_running else "Sedang Berjalan...",
                type="primary",
                use_container_width=True,
                disabled=is_running or not has_dataset_source,
                help="Pilih minimal satu sumber selain Google Trends untuk membangun master_osint_data.csv.",
            )
        with btn_col2:
            stop_clicked = st.button("Stop", use_container_width=True, disabled=not is_running)

        if selected_sources and not has_dataset_source:
            st.warning(
                "Google Trends menyimpan data di `trends_interest.csv`; pilih minimal satu "
                "dari YouTube, News RSS, Telegram, Instagram, Threads, atau X."
            )
        elif not selected_sources:
            st.info("Pilih minimal satu sumber data.")

        if stop_clicked and is_running:
            was_analyzing = st.session_state.osint_stage == "analyzing"
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            st.session_state.osint_proc = None
            st.session_state.osint_stage = "error"
            st.session_state.osint_analysis_failed = was_analyzing
            st.session_state.osint_error_message = "Proses dihentikan oleh operator."
            log_handle = st.session_state.get("osint_log_handle")
            if log_handle:
                try:
                    log_handle.close()
                except Exception:
                    pass
                st.session_state.osint_log_handle = None
            st.warning("Proses dihentikan. Dashboard tidak akan menampilkan CSV mentah sebagai pengganti hasil NLP.")
            st.rerun()

        if start_clicked and not is_running:
            if not MAIN_SCRIPT.exists():
                st.error(f"File {MAIN_SCRIPT.name} tidak ditemukan di {ROOT_DIR}.")
            else:
                st.session_state.osint_stage = "scraping"
                st.session_state.osint_analysis_failed = False
                st.session_state.osint_error_message = ""
                cmd = [
                    sys.executable,
                    "-m",
                    "ingestion.main",
                    "--sources",
                    *selected_sources,
                    "--timeframe",
                    timeframe,
                ]
                if replace_existing:
                    cmd.append("--replace-existing")
                st.session_state.osint_proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
                    env=os.environ.copy(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                st.rerun()

        proc = st.session_state.osint_proc
        if proc is not None:
            ret_code = proc.poll()
            if ret_code is None:
                stage_label = (
                    "Tahap 1/2: Scraping data OSINT..."
                    if st.session_state.osint_stage == "scraping"
                    else "Tahap 2/2: Analisis NLP, klaster narasi, dan DISARM..."
                )
                st.info(stage_label)
                st.code(_tail_log(16), language="text")
                time.sleep(2.0)
                st.rerun()

            elif st.session_state.osint_stage == "scraping":
                st.session_state.osint_proc = None
                if ret_code != 0:
                    st.session_state.osint_stage = "error"
                    st.session_state.osint_analysis_failed = False
                    st.session_state.osint_error_message = (
                        f"Scraping gagal (exit code {ret_code}). Hasil analisis sebelumnya tetap dipakai."
                    )
                    st.cache_data.clear()
                    st.rerun()

                try:
                    _invalidate_previous_analysis()
                except Exception as exc:
                    st.session_state.osint_stage = "error"
                    st.session_state.osint_analysis_failed = True
                    st.session_state.osint_error_message = str(exc)
                    st.cache_data.clear()
                    st.rerun()

                raw_file = ROOT_DIR / "data" / "master_osint_data.csv"
                if not raw_file.exists() or raw_file.stat().st_size <= 10:
                    st.session_state.osint_stage = "error"
                    st.session_state.osint_analysis_failed = True
                    st.session_state.osint_error_message = (
                        "Scraping selesai tetapi master_osint_data.csv kosong/tidak ditemukan. NLP tidak dijalankan."
                    )
                    st.cache_data.clear()
                    st.rerun()

                st.session_state.osint_stage = "analyzing"
                orch_cmd = [
                    sys.executable,
                    "-m",
                    "pipeline.orchestrator",
                    "--input",
                    "data/master_osint_data.csv",
                    "--interval",
                    NLP_INTERVAL,
                ]
                try:
                    log_handle = LOG_FILE.open("a", encoding="utf-8")
                    st.session_state.osint_log_handle = log_handle
                    log_handle.write("\n=== NLP WAJIB: MEMULAI ANALISIS SETELAH SCRAPING ===\n")
                    log_handle.flush()
                    st.session_state.osint_proc = subprocess.Popen(
                        orch_cmd,
                        cwd=str(ROOT_DIR),
                        env=os.environ.copy(),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                except Exception as exc:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    st.session_state.osint_log_handle = None
                    st.session_state.osint_stage = "error"
                    st.session_state.osint_analysis_failed = True
                    st.session_state.osint_error_message = f"Gagal menjalankan orchestrator NLP: {exc}"
                    st.cache_data.clear()
                    st.rerun()
                st.rerun()

            elif st.session_state.osint_stage == "analyzing":
                st.session_state.osint_proc = None
                log_handle = st.session_state.get("osint_log_handle")
                if log_handle:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    st.session_state.osint_log_handle = None

                enriched = ROOT_DIR / ENRICHED_FILE
                report = ROOT_DIR / REPORT_FILE
                outputs_ready = (
                    enriched.exists() and enriched.stat().st_size > 10
                    and report.exists() and report.stat().st_size > 2
                )
                if ret_code == 0 and outputs_ready:
                    st.session_state.osint_stage = "done"
                    st.session_state.osint_analysis_failed = False
                    st.session_state.osint_error_message = ""
                else:
                    st.session_state.osint_stage = "error"
                    st.session_state.osint_analysis_failed = True
                    missing = []
                    if not enriched.exists() or enriched.stat().st_size <= 10:
                        missing.append(ENRICHED_FILE)
                    if not report.exists() or report.stat().st_size <= 2:
                        missing.append(REPORT_FILE)
                    st.session_state.osint_error_message = (
                        f"Analisis NLP gagal (exit code {ret_code}) atau output belum lengkap. "
                        f"Periksa: {', '.join(missing) if missing else 'latest_run.log'}."
                    )
                st.cache_data.clear()
                st.rerun()

        if st.session_state.osint_stage == "done":
            st.success("Scraping dan analisis NLP selesai. Dashboard menampilkan hasil analisis terbaru.")
            with st.expander("Lihat Log Terakhir", expanded=False):
                st.code(_tail_log(28), language="text")
        elif st.session_state.osint_stage == "error":
            st.error(st.session_state.osint_error_message or "Pipeline berhenti dengan error.")
            with st.expander("Lihat Log Terakhir", expanded=True):
                st.code(_tail_log(28), language="text")

    render_api_config_panel()
