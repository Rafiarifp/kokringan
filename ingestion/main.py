import os
import json
import asyncio
import logging
import argparse
import threading
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from ingestion.data_preprocessing import (
    TIMEZONE,
    COLUMNS,
    calculate_cutoff_datetime,
    load_slang_dictionary,
    standardize_records,
    clean_text,
)

from ingestion.scrape import (
    safe_sync_scrape,
    safe_async_scrape,
    scrape_youtube_by_keywords,
    scrape_news,
    scrape_google_trends_series,
    save_trends_csv,
    scrape_telegram_channels,
    get_or_refresh_x_client,
    scrape_x,
    timeframe_to_trends_timeframe,
    detect_matching_topics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("osint_main")
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("telethon").setLevel(logging.WARNING)

load_dotenv()

# ==========================================
# GLOBAL SWITCHES (DEFAULT ON / OFF)
# ==========================================
ENABLE_YOUTUBE = True
ENABLE_NEWS = True
ENABLE_GOOGLE_TRENDS = True
ENABLE_TELEGRAM = True
ENABLE_X = False


def _load_existing_dataset(output_path: Path, slang_dict: dict, cutoff_dt=None) -> pd.DataFrame:
    """Memuat CSV lama tanpa re-cleaning baris yang sudah bersih."""
    if not output_path.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        old_df = pd.read_csv(output_path)
        for col in COLUMNS:
            if col not in old_df.columns:
                old_df[col] = None

        old_df["timestamp"] = pd.to_datetime(old_df["timestamp"], utc=True, errors="coerce")
        old_df = old_df.dropna(subset=["timestamp"])
        old_df["timestamp"] = old_df["timestamp"].dt.tz_convert(TIMEZONE)

        # Update Data mempertahankan seluruh histori. Cutoff hanya diterapkan
        # bila caller memang meminta pembatasan data lama.
        if cutoff_dt is not None:
            cutoff_tz = pd.to_datetime(cutoff_dt, utc=True).tz_convert(TIMEZONE)
            old_df = old_df[old_df["timestamp"] >= cutoff_tz].copy()

        if old_df.empty:
            return pd.DataFrame(columns=COLUMNS)

        old_df["platform"] = old_df["platform"].fillna("Unknown").astype(str)
        old_df["author"] = old_df["author"].fillna("Unknown").astype(str)
        old_df["text"] = old_df["text"].fillna("").astype(str)
        old_df["cleaned_text"] = old_df["cleaned_text"].fillna("").astype(str)

        missing_clean = old_df["cleaned_text"].str.strip().str.len() == 0
        if missing_clean.any():
            old_df.loc[missing_clean, "cleaned_text"] = old_df.loc[missing_clean, "text"].apply(
                lambda v: clean_text(v, slang_dict)
            )

        old_df["engagement"] = pd.to_numeric(old_df["engagement"], errors="coerce").fillna(0)
        old_df["topic"] = old_df["topic"].fillna("").astype(str)
        old_df["url"] = old_df["url"].fillna("").astype(str)

        old_df = old_df[old_df["cleaned_text"].str.len() > 0]
        return old_df[COLUMNS].reset_index(drop=True)
    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logger.error("Dataset lama gagal dibaca; Update Data dibatalkan agar CSV lama tidak tertimpa | ERR: %s", err_msg)
        raise RuntimeError(f"Dataset lama tidak dapat dibaca: {err_msg}") from exc


def _deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus duplikat eksak namun pertahankan komentar berbeda pada satu video."""
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["platform", "author", "timestamp", "text"],
        keep="last",
    ).reset_index(drop=True)


def _apply_topic_filter_to_sources(
    df: pd.DataFrame,
    topic_filters: dict | None,
    news_sources: set[str],
    label: str,
) -> pd.DataFrame:
    """Filter record Telegram dan portal berita berdasarkan isi teks.

    Fungsi ini juga diterapkan ke CSV lama agar data tidak relevan yang pernah
    tersimpan sebelum filter diaktifkan tidak ikut dipertahankan. Sumber lain
    (YouTube, X, dan lainnya) tidak diubah.
    """
    if df.empty or topic_filters is None:
        return df

    result = df.copy()
    platform = result["platform"].fillna("").astype(str)
    target_mask = platform.eq("Telegram") | platform.isin(news_sources)
    if not target_mask.any():
        return result

    target_text = result.loc[target_mask, "text"].fillna("").astype(str)
    matched = target_text.apply(lambda value: detect_matching_topics(value, topic_filters))
    keep_mask = matched.apply(bool)
    rejected = int((~keep_mask).sum())

    kept_indices = matched.index[keep_mask]
    result.loc[kept_indices, "topic"] = matched.loc[kept_indices].apply(
        lambda topics: "; ".join(topics)
    )

    result = result.loc[~target_mask | keep_mask].copy()
    logger.info(
        "[Topic Filter:%s] %d record diperiksa, %d relevan, %d diabaikan.",
        label, int(target_mask.sum()), int(keep_mask.sum()), rejected,
    )
    return result.reset_index(drop=True)


async def main(
    sources: list[str] | None = None,
    timeframe: str | None = None,
    replace_existing: bool = False,
):
    default_switches = {
        "youtube": ENABLE_YOUTUBE,
        "news": ENABLE_NEWS,
        "google_trends": ENABLE_GOOGLE_TRENDS,
        "telegram": ENABLE_TELEGRAM,
        "x": ENABLE_X,
    }

    if sources is not None:
        normalized = {s.strip().lower() for s in sources if s.strip()}
        active = {k: (k in normalized) for k in default_switches}
    else:
        normalized = None
        active = default_switches

    logger.info("Memuat konfigurasi targets.json.")
    targets_path = Path("config/targets.json")
    if not targets_path.exists():
        logger.error("File targets.json tidak ditemukan!")
        return

    with open(targets_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    settings = targets.get("settings", {})
    timeframe = (timeframe or settings.get("timeframe", "1m")).strip().lower()
    supported_timeframes = {"1w", "1m", "1q"}
    if timeframe not in supported_timeframes:
        logger.error(
            "Timeframe tidak valid: %s. Pilihan: %s",
            timeframe, ", ".join(sorted(supported_timeframes))
        )
        return False

    if replace_existing and sources is not None:
        data_sources = {"youtube", "news", "telegram", "x"}
        if not (normalized or set()).intersection(data_sources):
            logger.error(
                "Delete Old Data membutuhkan minimal satu sumber dataset "
                "(YouTube, News, Telegram, atau X). Google Trends menyimpan file terpisah."
            )
            return False

    output_csv = settings.get("output_csv", "master_osint_data.csv")
    slang_file = settings.get("slang_file", "slang_dict.json")

    # Filter topik hanya diterapkan pada News RSS dan Telegram.
    filter_cfg = targets.get("topic_filters", {})
    topic_filters = None
    if filter_cfg.get("enabled", False):
        configured_topics = filter_cfg.get("topics", {})
        if not isinstance(configured_topics, dict) or not configured_topics:
            logger.error(
                "[Topic Filter] Diaktifkan tetapi 'topic_filters.topics' kosong/tidak valid. "
                "Scraping dihentikan agar data tidak tersimpan tanpa filter."
            )
            return
        topic_filters = configured_topics
        logger.info(
            "[Topic Filter] Aktif | %d kategori topik.",
            len(topic_filters),
        )
    else:
        logger.warning("[Topic Filter] Nonaktif | News dan Telegram tidak difilter.")

    news_sources = {
        str(feed.get("name", ""))
        for feed in targets.get("news", {}).get("feeds", [])
        if feed.get("name")
    }

    cutoff_dt = calculate_cutoff_datetime(timeframe)
    active_names = [k for k, v in active.items() if v]
    logger.info(
        "OSINT Ingestion Dimulai | Sumber: %s | Timeframe: %s (Cutoff: >= %s)",
        ", ".join(active_names) if active_names else "Tidak ada",
        timeframe,
        cutoff_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    slang_dict = load_slang_dictionary(slang_file)
    sync_tasks = []
    sync_task_labels = []
    scrape_errors = []
    scrape_errors_lock = threading.Lock()

    def _record_scrape_error(platform, exc):
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        with scrape_errors_lock:
            scrape_errors.append((platform, message))

    # 1. YOUTUBE
    cfg_yt = targets.get("youtube", {})
    if active["youtube"] and (sources is not None or cfg_yt.get("enabled", False)):
        api_key = os.getenv("YOUTUBE_API_KEY")
        if api_key and cfg_yt.get("queries"):
            sync_tasks.append(asyncio.to_thread(
                safe_sync_scrape,
                "YouTube",
                scrape_youtube_by_keywords,
                api_key,
                cfg_yt["queries"],
                cfg_yt.get("max_videos_per_query", 15),
                cfg_yt.get("max_comments_per_video", 100),
                cutoff_dt,
                on_error=_record_scrape_error,
            ))
            sync_task_labels.append("YouTube")
        else:
            message = "YOUTUBE_API_KEY tidak ada di .env atau query YouTube kosong."
            logger.warning("[YouTube] Dilewati | ERR: %s", message)
            _record_scrape_error("YouTube", RuntimeError(message))

    # 2. NEWS RSS
    cfg_news = targets.get("news", {})
    if active["news"] and (sources is not None or cfg_news.get("enabled", False)):
        news_sem = threading.Semaphore(4)
        limit_per_feed = cfg_news.get("limit_per_feed", 200)

        def _bounded_news_scrape(feed_cfg):
            with news_sem:
                return safe_sync_scrape(
                    feed_cfg["name"],
                    scrape_news,
                    feed_cfg["url"],
                    limit_per_feed,
                    feed_cfg.get("topic", feed_cfg["name"]),
                    feed_cfg["name"],
                    cutoff_dt,
                    topic_filters=topic_filters,
                    on_error=_record_scrape_error,
                )

        news_feeds = cfg_news.get("feeds", [])
        if not news_feeds:
            _record_scrape_error("News RSS", RuntimeError("Daftar RSS feed kosong."))
        for feed in news_feeds:
            sync_tasks.append(asyncio.to_thread(_bounded_news_scrape, feed))
            sync_task_labels.append(f"News RSS:{feed.get('name', 'Unknown')}")

    # 3. GOOGLE TRENDS
    cfg_trends = targets.get("google_trends", {})
    if active["google_trends"] and (sources is not None or cfg_trends.get("enabled", False)):
        keywords = cfg_trends.get("keywords", [])
        if keywords:
            trends_csv = settings.get("trends_csv", "trends_interest.csv")
            trends_tf = timeframe_to_trends_timeframe(timeframe)

            def _run_trends():
                trends = safe_sync_scrape(
                    "Google Trends",
                    scrape_google_trends_series,
                    keywords,
                    trends_tf,
                    on_error=_record_scrape_error,
                )
                trends_failed = any(name == "Google Trends" for name, _ in scrape_errors)
                if replace_existing and trends_failed and not trends:
                    logger.warning(
                        "[Google Trends] Refresh gagal; trends CSV lama dipertahankan."
                    )
                else:
                    total = save_trends_csv(
                        trends, trends_csv, replace=replace_existing
                    )
                    logger.info("[Google Trends] Tersimpan: %s baris (file terpisah).", total)
                return []

            sync_tasks.append(asyncio.to_thread(_run_trends))
            sync_task_labels.append("Google Trends")

    # 4. TELEGRAM
    async_tasks = []
    async_task_labels = []
    cfg_tele = targets.get("telegram", {})
    if active["telegram"] and (sources is not None or cfg_tele.get("enabled", False)):
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        if api_id and api_hash and cfg_tele.get("channels", []):
            try:
                tele_id = int(api_id)
                async_tasks.append(safe_async_scrape(
                    "Telegram",
                    scrape_telegram_channels,
                    tele_id,
                    api_hash,
                    cfg_tele.get("channels", []),
                    cfg_tele.get("limit_per_channel", 400),
                    cutoff_dt,
                    topic_filters=topic_filters,
                    on_error=_record_scrape_error,
                ))
                async_task_labels.append("Telegram")
            except ValueError as exc:
                logger.error("[Telegram] Dilewati | ERR: TELEGRAM_API_ID harus berupa angka integer.")
                _record_scrape_error("Telegram", exc)
        else:
            message = (
                "TELEGRAM_API_ID/TELEGRAM_API_HASH tidak ada di .env."
                if not (api_id and api_hash)
                else "Daftar channel Telegram kosong."
            )
            logger.warning("[Telegram] Dilewati | ERR: %s", message)
            _record_scrape_error("Telegram", RuntimeError(message))

    # 5. X (TWITTER)
    cfg_x = targets.get("x", {})
    if active["x"] and (sources is not None or cfg_x.get("enabled", False)):
        cookies_path = cfg_x.get("cookies_path", "cookies.json")
        x_queries = cfg_x.get("queries", [])
        if Path(cookies_path).exists() and x_queries:
            async def _run_x_batch():
                recs = []
                try:
                    x_client = await get_or_refresh_x_client(cookies_path)
                except Exception as exc:
                    err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                    logger.warning("[X/Twitter] Dilewati | ERR: %s", err_msg)
                    _record_scrape_error("X client", exc)
                    return recs

                for query in x_queries:
                    recs += await safe_async_scrape(
                        f"X:{query}",
                        scrape_x,
                        cookies_path,
                        query,
                        cutoff_dt,
                        cfg_x.get("limit_per_query", 50),
                        x_client,
                        on_error=_record_scrape_error,
                    )
                return recs

            async_tasks.append(_run_x_batch())
            async_task_labels.append("X/Twitter")
        elif not Path(cookies_path).exists():
            message = f"File cookie '{cookies_path}' tidak ditemukan."
            logger.warning("[X/Twitter] Dilewati | ERR: %s", message)
            _record_scrape_error("X/Twitter", FileNotFoundError(message))
        elif not x_queries:
            message = "Daftar query X kosong."
            logger.warning("[X/Twitter] Dilewati | ERR: %s", message)
            _record_scrape_error("X/Twitter", RuntimeError(message))

    # Pantau penyelesaian setiap scraper. Master CSV baru ditulis setelah batch
    # selesai; progress ini membantu menemukan sumber yang masih tertahan.
    pending_scrapers = list(zip(
        sync_tasks + async_tasks,
        sync_task_labels + async_task_labels,
    ))
    total_scrapers = len(pending_scrapers)
    logger.info("[Progress] Menunggu %d tugas scraper selesai.", total_scrapers)

    async def _run_tracked_scraper(index, awaitable, label):
        logger.info("[Progress] Mulai tugas %d/%d: %s", index, total_scrapers, label)
        try:
            result = await awaitable
            logger.info(
                "[Progress] Selesai %d/%d: %s (%d record).",
                index, total_scrapers, label, len(result or [])
            )
            return result or []
        except Exception as exc:
            err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.exception(
                "[Progress] Gagal %d/%d: %s | ERR: %s",
                index, total_scrapers, label, err_msg
            )
            _record_scrape_error(label, exc)
            return []

    tracked_tasks = [
        asyncio.create_task(_run_tracked_scraper(i, task, label))
        for i, (task, label) in enumerate(pending_scrapers, start=1)
    ]
    all_batches = []
    for completed_task in asyncio.as_completed(tracked_tasks):
        all_batches.append(await completed_task)

    logger.info("[Progress] Semua tugas scraper selesai. Mulai standardisasi dan penyimpanan CSV.")
    raw_records = [rec for batch in all_batches for rec in batch]

    if replace_existing and not raw_records and scrape_errors:
        logger.error(
            "[DELETE OLD DATA] Tidak ada data yang berhasil diambil dan terdapat %d kegagalan. "
            "CSV lama dipertahankan; tidak ada file yang diganti.",
            len(scrape_errors),
        )
        for platform, message in scrape_errors:
            logger.error("[DELETE OLD DATA] Sumber gagal: %s | %s", platform, message)
        return False

    if replace_existing and scrape_errors:
        logger.warning(
            "[DELETE OLD DATA] Ada %d kegagalan sumber; hasil yang berhasil tetap akan disimpan.",
            len(scrape_errors),
        )

    new_df = standardize_records(raw_records, slang_dict=slang_dict, cutoff_dt=cutoff_dt)
    new_df = _apply_topic_filter_to_sources(
        new_df, topic_filters, news_sources, label="data baru"
    )

    output_path = Path(output_csv)

    if replace_existing:
        # DELETE OLD DATA: bangun dataset hanya dari hasil scraping saat ini.
        # CSV lama baru diganti setelah file baru berhasil ditulis secara atomik.
        logger.info(
            "[DELETE OLD DATA] CSV lama tidak akan digabungkan ke dataset baru."
        )
        combined = (
            new_df.copy()
            if not new_df.empty
            else pd.DataFrame(columns=COLUMNS)
        )
    else:
        # UPDATE DATA: pertahankan semua histori dan tambahkan hasil scraping baru.
        old_df = _load_existing_dataset(
            output_path,
            slang_dict,
            cutoff_dt=None,
        )
        frames = [df for df in (old_df, new_df) if not df.empty]
        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=COLUMNS)
        )

    should_write = replace_existing or not combined.empty or topic_filters is not None
    if should_write:
        if not combined.empty:
            combined = _deduplicate_records(combined)
            combined = combined.sort_values("timestamp", ascending=False).reset_index(drop=True)
            timestamp_series = pd.to_datetime(
                combined["timestamp"], utc=True, errors="coerce"
            ).dt.tz_convert(TIMEZONE)
            combined["timestamp"] = timestamp_series.apply(
                lambda value: value.isoformat() if pd.notna(value) else ""
            )
        else:
            combined = pd.DataFrame(columns=COLUMNS)

        # Tulis ke file sementara terlebih dahulu. CSV lama baru diganti setelah
        # proses scraping selesai dan file baru berhasil ditulis.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            combined.to_csv(temp_path, index=False, encoding="utf-8-sig")
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        logger.info(
            "%sDataset tersimpan: %s (%d baris).",
            "[DELETE OLD DATA] " if replace_existing else "[UPDATE DATA] ",
            output_path.resolve(), len(combined)
        )

    summary_lines = [
        "",
        "--- RINGKASAN PIPELINE ---",
        f"Mode             : {'DELETE OLD DATA' if replace_existing else 'UPDATE DATA'}",
        f"Timeframe        : {timeframe} (cutoff >= {cutoff_dt.strftime('%Y-%m-%d')})",
        f"Sources          : {', '.join(normalized) if normalized is not None else 'config aktif'}",
        f"Data baru        : {len(new_df)}",
        f"Total dataset    : {len(combined)}",
        f"Target 2.000     : {'TERCAPAI' if len(combined) >= 2000 else 'BELUM TERCAPAI'}",
    ]
    if not combined.empty:
        summary_lines.append("\nJumlah data per platform:")
        summary_lines.append(combined["platform"].value_counts().to_string())

    print("\n".join(summary_lines))
    return True


def run_ingestion(
    sources: list[str] | None = None,
    timeframe: str | None = None,
    replace_existing: bool = False,
):
    """Helper sinkronus untuk dipanggil langsung dari Streamlit atau script lain."""
    return asyncio.run(
        main(sources=sources, timeframe=timeframe, replace_existing=replace_existing)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSINT Data Ingestion Pipeline")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["youtube", "news", "google_trends", "telegram", "x"],
        help="Pilih sumber spesifik untuk di-scrape (contoh: --sources telegram news)",
    )
    parser.add_argument(
        "--timeframe", choices=["1w", "1m", "1q"],
        help="Rentang pengambilan data: 1w (7 hari), 1m (1 bulan), 1q (3 bulan)."
    )
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="Ganti dataset lama dengan hasil scraping baru setelah penyimpanan berhasil."
    )
    args = parser.parse_args()
    success = run_ingestion(
        sources=args.sources,
        timeframe=args.timeframe,
        replace_existing=args.replace_existing,
    )
    raise SystemExit(0 if success else 1)
