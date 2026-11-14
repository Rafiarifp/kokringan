"""Panel Streamlit untuk Delete Old Data / Update Data dan pipeline analitik."""
import json
import subprocess
import sys
import time
from pathlib import Path
import streamlit as st


def _find_project_root() -> Path:
    """Cari root folder proyek berdasarkan struktur folder baru."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "ingestion" / "main.py").is_file() and (candidate / "config" / "targets.json").is_file():
            return candidate
    return here.parent


ROOT_DIR = _find_project_root()
MAIN_SCRIPT = ROOT_DIR / "ingestion" / "main.py"
TARGETS_FILE = ROOT_DIR / "config" / "targets.json"
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
    "x": "X (Twitter)",
}
DATASET_SOURCES = {"youtube", "news", "telegram", "x"}

# Nilai tetap untuk argumen --interval pada orchestrator lama.
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


def render_ingestion_panel() -> None:
    """Scraping -> NLP wajib -> dashboard hanya menampilkan hasil analisis terbaru."""
    # Hapus state widget lama yang sudah tidak dipakai (checkbox/interval opsional).
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
        trends_only = bool(selected_sources) and set(selected_sources) == {"google_trends"}

        can_run = (
            (has_dataset_source or (trends_only and not replace_existing))
            and not is_running
        )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            btn_label = (
                "Sedang Berjalan..."
                if is_running
                else (
                    "Ambil Google Trends Saja"
                    if trends_only
                    else ("Delete Old Data · Scrape + Analisis" if replace_existing else "Update Data + Analisis")
                )
            )
            start_clicked = st.button(
                btn_label,
                type="primary",
                use_container_width=True,
                disabled=not can_run,
            )
        with btn_col2:
            stop_clicked = st.button("Stop", use_container_width=True, disabled=not is_running)

        if selected_sources and not has_dataset_source:
            st.warning(
                "Google Trends menyimpan data di `trends_interest.csv`; pilih minimal satu "
                "dari YouTube, News RSS, Telegram, atau X untuk membangun dataset utama."
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
                cmd.extend(["--sources", *selected_sources])
                st.session_state.osint_proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
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

            # Proses scraping selesai.
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

                # Scraping sukses: artefak analisis lama wajib dibuang agar tidak stale.
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

            # Proses NLP selesai. Dashboard hanya boleh dibuka jika output terbaru lengkap.
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
