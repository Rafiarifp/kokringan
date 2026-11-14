import os
import json
import time
import asyncio
import logging
import re
import html
import unicodedata
import ssl
import certifi
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from twikit import Client as TwikitClient

SSL_HANDLER = urllib.request.HTTPSHandler(
    context=ssl.create_default_context(cafile=certifi.where())
)

logger = logging.getLogger("osint_ingestion")


# -----------------------------
# HELPER KONVERSI TIMEFRAME
# -----------------------------
def timeframe_to_trends_timeframe(tf: str) -> str:
    tf = tf.lower().strip()
    if tf in ["1w", "week", "last_week"]:
        return "now 7-d"
    if tf in ["1q", "quarter", "last_quarter"]:
        return "today 3-m"
    return "today 1-m"


# -----------------------------
# ERROR HANDLING WRAPPERS
# -----------------------------
def safe_sync_scrape(platform, function, *args, on_error=None, **kwargs):
    """Jalankan scraper sinkronus; callback error tidak diteruskan ke scraper."""
    try:
        records = function(*args, **kwargs)
        logger.info("[%s] Selesai: %d record.", platform, len(records))
        return records
    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logger.error("[%s] Gagal eksekusi | ERR: %s", platform, err_msg)
        if callable(on_error):
            on_error(platform, exc)
        return []


async def safe_async_scrape(platform, function, *args, on_error=None, **kwargs):
    """Jalankan scraper asinkronus; callback error tidak diteruskan ke scraper."""
    try:
        records = await function(*args, **kwargs)
        logger.info("[%s] Selesai: %d record.", platform, len(records))
        return records
    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logger.error("[%s] Gagal eksekusi | ERR: %s", platform, err_msg)
        if callable(on_error):
            on_error(platform, exc)
        return []


# -----------------------------
# TOPIC FILTER HELPERS
# -----------------------------
def normalize_topic_text(value):
    """Normalisasi teks agar pencocokan keyword tidak sensitif kapital/tanda baca."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)  # hapus tag HTML dari RSS summary
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_matching_topics(text, topic_filters):
    """Kembalikan semua nama topik yang keyword-nya ditemukan di dalam teks.

    topic_filters format: {"Nama Topik": ["keyword 1", "keyword 2", ...]}
    Pencocokan menggunakan batas kata/frasa, bukan substring bebas.
    """
    if not topic_filters:
        return []

    normalized_text = f" {normalize_topic_text(text)} "
    matched_topics = []

    for topic_name, keywords in topic_filters.items():
        if isinstance(keywords, str):
            keywords = [keywords]

        for keyword in keywords or []:
            normalized_keyword = normalize_topic_text(keyword)
            if normalized_keyword and f" {normalized_keyword} " in normalized_text:
                matched_topics.append(topic_name)
                break

    return matched_topics


# -----------------------------
# YOUTUBE SCRAPER (REUSE CLIENT)
# -----------------------------
def search_youtube_videos(youtube, query, max_videos=5, cutoff_dt=None):
    published_after = cutoff_dt.isoformat(timespec="seconds").replace("+00:00", "Z")

    response = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        order="date",
        regionCode="ID",
        relevanceLanguage="id",
        publishedAfter=published_after,
        maxResults=min(max_videos, 50),
        safeSearch="none"
    ).execute()

    videos = []
    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if video_id:
            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", "Unknown"),
                "published_at": snippet.get("publishedAt"),
                "query": query
            })
    return videos


def scrape_youtube_comments(youtube, video_id, max_results=100, topic=""):
    data = []
    page_token = None
    remaining = max_results

    while remaining > 0:
        page_size = min(100, remaining)
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=page_size,
            textFormat="plainText",
            pageToken=page_token
        ).execute()

        items = response.get("items", [])
        for item in items:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            data.append({
                "timestamp": comment.get("publishedAt"),
                "platform": "YouTube",
                "author": comment.get("authorDisplayName", "Unknown"),
                "text": comment.get("textOriginal", comment.get("textDisplay", "")),
                "engagement": comment.get("likeCount", 0),
                "topic": topic,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })

        remaining -= len(items)
        page_token = response.get("nextPageToken")
        if not page_token or not items:
            break

    return data


def scrape_youtube_by_keywords(api_key, queries, max_videos_per_query=5,
                               max_comments_per_video=100, cutoff_dt=None):
    from googleapiclient.discovery import build
    import httplib2

    # Tanpa timeout, satu request search/komentar dapat menahan gather()
    # sehingga master_osint_data.csv tidak pernah mencapai tahap penyimpanan.
    http = httplib2.Http(timeout=25)
    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key,
        http=http,
        cache_discovery=False,
    )
    all_records = []
    seen_video_ids = set()

    for query in queries:
        logger.info("[YouTube] Cari query: '%s'", query)
        try:
            videos = search_youtube_videos(
                youtube, query, max_videos=max_videos_per_query, cutoff_dt=cutoff_dt
            )
            logger.info("[YouTube] Ditemukan %d video untuk query '%s'", len(videos), query)
        except Exception as exc:
            err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.error("[YouTube] Search gagal [query: '%s'] | ERR: %s", query, err_msg)
            continue

        for video in videos:
            video_id = video["video_id"]
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            try:
                comments = scrape_youtube_comments(youtube, video_id, max_comments_per_video, query)
                all_records.extend(comments)
            except Exception as exc:
                err_str = str(exc)
                if "commentsDisabled" in err_str:
                    logger.warning("[YouTube] Komentar dinonaktifkan [video: %s]", video_id)
                else:
                    err_msg = err_str.splitlines()[0] if err_str else type(exc).__name__
                    logger.warning("[YouTube] Gagal ambil komentar [video: %s] | ERR: %s", video_id, err_msg)

    return all_records


# -----------------------------
# NEWS RSS SCRAPER (PARALLEL ARTICLE FETCH)
# -----------------------------
def scrape_news(rss_url, limit=200, topic="", source_name="News", cutoff_dt=None,
                topic_filters=None):
    """Ambil RSS dengan timeout jaringan dan fallback ke ringkasan RSS.

    Fetch full article dibatasi agar satu portal/artikel yang lambat tidak
    menahan penyelesaian seluruh pipeline.
    """
    import feedparser
    import requests
    import trafilatura
    from concurrent.futures import ThreadPoolExecutor

    headers = {"User-Agent": "Mozilla/5.0 (compatible; KOKRINGAN-OSINT/1.0)"}
    logger.info("[%s] Mulai mengambil RSS: %s", source_name, rss_url)
    try:
        response = requests.get(
            rss_url,
            headers=headers,
            timeout=(8, 20),
            allow_redirects=True,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except requests.RequestException as exc:
        raise RuntimeError(f"RSS timeout/koneksi gagal: {exc}") from exc

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"RSS Feed invalid atau tidak dapat diakses ({rss_url})")

    logger.info(
        "[%s] RSS berhasil dibaca: %d entri. Memproses konten...",
        source_name, len(feed.entries)
    )
    cutoff_ts = pd.to_datetime(cutoff_dt, utc=True) if cutoff_dt is not None else None
    valid_entries = []
    scanned_count = 0

    for entry in feed.entries[:limit]:
        pub_raw = entry.get("published") or entry.get("updated")
        if cutoff_ts is not None and pub_raw:
            pub_ts = pd.to_datetime(pub_raw, utc=True, errors="coerce")
            if pd.notna(pub_ts) and pub_ts < cutoff_ts:
                continue

        scanned_count += 1
        title = entry.get("title", "")
        rss_summary = entry.get("summary") or entry.get("description") or ""

        if topic_filters is not None:
            initial_topics = detect_matching_topics(
                f"{title}\n{rss_summary}", topic_filters
            )
            if not initial_topics:
                continue
        else:
            initial_topics = []

        valid_entries.append((entry, pub_raw, initial_topics))

    # RSS summary tetap dipakai bila fetch artikel gagal/timeout.
    def _process_entry(idx_item):
        idx, (entry, pub_raw, initial_topics) = idx_item
        title = entry.get("title", "")
        rss_summary = entry.get("summary") or entry.get("description") or ""
        article_text = ""
        article_url = entry.get("link", "")

        # Batasi full-text fetch hanya ke 8 artikel per portal.
        # Connect timeout 5s + read timeout 12s mencegah request menggantung.
        if idx < 8 and source_name not in ["Tempo", "Tempo Bisnis"] and article_url:
            try:
                article_response = requests.get(
                    article_url,
                    headers=headers,
                    timeout=(5, 12),
                    allow_redirects=True,
                )
                article_response.raise_for_status()
                html_content = article_response.content[:3_000_000]
                article_text = trafilatura.extract(html_content) or ""
            except requests.RequestException as exc:
                logger.debug(
                    "[%s] Full article dilewati (timeout/koneksi): %s | %s",
                    source_name, article_url, str(exc).splitlines()[0]
                )
            except Exception as exc:
                logger.debug(
                    "[%s] Gagal ekstrak artikel %s: %s",
                    source_name, article_url, str(exc).splitlines()[0]
                )

        content = article_text or rss_summary or title
        topic_value = (
            "; ".join(initial_topics)
            if topic_filters is not None
            else topic
        )

        return {
            "timestamp": pub_raw,
            "platform": source_name,
            "author": entry.get("author", "Redaksi"),
            "text": f"{title}\n{content}",
            "engagement": 0,
            "topic": topic_value,
            "url": article_url,
        }

    # Sedikit worker mengurangi ledakan koneksi saat beberapa RSS diproses serentak.
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed = list(executor.map(_process_entry, enumerate(valid_entries)))

    data = [record for record in processed if record is not None]
    if topic_filters is not None:
        logger.info(
            "[%s] %d artikel diperiksa, %d relevan disimpan, %d diabaikan.",
            source_name, scanned_count, len(data), scanned_count - len(data)
        )
    return data


# -----------------------------
# GOOGLE TRENDS SCRAPER
# -----------------------------
def scrape_google_trends_series(keywords, timeframe="today 1-m", geo="ID"):
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("[Google Trends] Package 'pytrends' belum terinstall.")
        return []

    # Kompatibilitas urllib3 2.0+
    try:
        import inspect
        from urllib3.util.retry import Retry

        if "method_whitelist" not in inspect.signature(Retry.__init__).parameters:
            _orig_retry_init = Retry.__init__

            def _patched_retry_init(self, *args, **kwargs):
                if "method_whitelist" in kwargs:
                    kwargs.setdefault("allowed_methods", kwargs.pop("method_whitelist"))
                return _orig_retry_init(self, *args, **kwargs)

            Retry.__init__ = _patched_retry_init
    except Exception:
        pass

    # Bersihkan & dedup keyword tanpa membatasi 5
    clean_kws = []
    for k in keywords or []:
        k_clean = str(k).strip()
        if k_clean and k_clean not in clean_kws:
            clean_kws.append(k_clean)
    if not clean_kws:
        return []

    logger.info("[Google Trends] Mulai mengambil %d keyword (%s).", len(clean_kws), timeframe)
    pytrends = TrendReq(hl="id-ID", tz=420, timeout=(8, 20), retries=1, backoff_factor=0.2)

    # Bagi menjadi batch maks. 5 keyword dengan 1 anchor bersama agar skala 0-100 konsisten
    anchor = clean_kws[0]
    batches = [clean_kws[:5]]
    for i in range(5, len(clean_kws), 4):
        batches.append([anchor] + clean_kws[i : i + 4])

    combined_series: dict[str, pd.Series] = {}
    anchor_ref: pd.Series | None = None

    for idx, batch in enumerate(batches):
        if idx > 0:
            time.sleep(2.5)  # Jeda antar-batch untuk menghindari rate-limit HTTP 429

        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            df_interest = pytrends.interest_over_time()
        except Exception as exc:
            logger.warning("[Google Trends] Batch %d/%d (%s) gagal: %s", idx + 1, len(batches), batch, exc)
            continue

        if df_interest.empty:
            continue
        if "isPartial" in df_interest.columns:
            df_interest = df_interest.drop(columns=["isPartial"])

        if idx == 0:
            anchor_ref = df_interest[anchor].astype(float)
            for kw in batch:
                if kw in df_interest.columns:
                    combined_series[kw] = df_interest[kw].astype(float)
        else:
            # Kalibrasi skor batch terhadap anchor di batch pertama
            batch_anchor = df_interest[anchor].astype(float)
            denom = float(batch_anchor.sum())
            scale = float(anchor_ref.sum()) / denom if (anchor_ref is not None and denom > 0) else 1.0
            for kw in batch:
                if kw != anchor and kw in df_interest.columns:
                    combined_series[kw] = df_interest[kw].astype(float) * scale

    if not combined_series:
        return []

    merged_df = pd.DataFrame(combined_series).fillna(0.0)
    max_val = float(merged_df.max().max())
    if max_val > 0:
        merged_df = (merged_df / max_val) * 100.0

    data = []
    for timestamp, row in merged_df.iterrows():
        for keyword, val in row.items():
            data.append({
                "timestamp": timestamp.isoformat(),
                "keyword": keyword,
                "interest_score": int(round(val)),
                "geo": geo,
            })

    logger.info("[Google Trends] Selesai mengambil %d titik data (%d keyword).", len(data), len(combined_series))
    return data


def save_trends_csv(records, filepath="trends_interest.csv", replace=False):
    if not records:
        return 0
    df = pd.DataFrame(records)
    out = Path(filepath)
    if out.exists() and not replace:
        try:
            old_df = pd.read_csv(out)
            df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(subset=["timestamp", "keyword"])
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return len(df)


# -----------------------------
# TELEGRAM SCRAPER (SHARED SESSION + EARLY BREAK)
# -----------------------------
async def scrape_telegram_channels(api_id, api_hash, channels, limit_per_channel=120,
                                  cutoff_dt=None, topic_filters=None):
    from telethon import TelegramClient

    if not channels:
        return []

    client = TelegramClient(
        "data/osint_session", api_id, api_hash,
        timeout=20,
        connection_retries=1,
        request_retries=1,
        retry_delay=1,
        auto_reconnect=False,
    )
    all_records = []

    async def _read_one_channel(channel):
        channel_records = []
        scanned_count = 0
        saved_count = 0

        async for message in client.iter_messages(channel, limit=limit_per_channel):
            if cutoff_dt and message.date and message.date < cutoff_dt:
                break
            if not message.text:
                continue

            scanned_count += 1
            if topic_filters is not None:
                matched_topics = detect_matching_topics(message.text, topic_filters)
                if not matched_topics:
                    continue
                topic_value = "; ".join(matched_topics)
            else:
                topic_value = channel

            channel_records.append({
                "timestamp": message.date,
                "platform": "Telegram",
                "author": channel,
                "text": message.text,
                "engagement": message.views or 0,
                "topic": topic_value,
                "url": f"https://t.me/{channel}/{message.id}",
            })
            saved_count += 1

        return channel_records, scanned_count, saved_count

    try:
        await asyncio.wait_for(client.start(), timeout=30)

        for channel in channels:
            logger.info("[Telegram:%s] Mulai mengambil pesan...", channel)
            try:
                records, scanned_count, saved_count = await asyncio.wait_for(
                    _read_one_channel(channel), timeout=90
                )
                all_records.extend(records)
                logger.info(
                    "[Telegram:%s] Selesai: %d pesan diperiksa, %d record disimpan.",
                    channel, scanned_count, saved_count,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[Telegram:%s] Timeout 90 detik; channel dilewati.", channel
                )
            except Exception as exc:
                err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                logger.error("[Telegram:%s] Gagal | ERR: %s", channel, err_msg)

    except asyncio.TimeoutError:
        logger.error("[Telegram] Timeout saat membuat koneksi/login (30 detik).")
    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logger.error("[Telegram] Koneksi gagal | ERR: %s", err_msg)
    finally:
        if client.is_connected():
            await client.disconnect()

    return all_records


# -----------------------------
# X (TWITTER) SCRAPER
# -----------------------------
async def get_or_refresh_x_client(cookies_path: str):
    client = TwikitClient("en-US")
    cookie_file = Path(cookies_path)

    if cookie_file.exists():
        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                formatted_data = {c["name"]: c["value"] for c in raw_data if "name" in c and "value" in c}
                with open(cookie_file, "w", encoding="utf-8") as f:
                    json.dump(formatted_data, f, indent=4)
                logger.info("[X/Twitter] Konversi cookies.json (List -> Dict) sukses.")

            client.load_cookies(str(cookie_file))
            logger.info("[X/Twitter] Cookie dimuat dari %s", cookies_path)
            return client
        except Exception as exc:
            err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.warning("[X/Twitter] Cookie gagal dimuat | ERR: %s", err_msg)

    raise FileNotFoundError(f"File cookie '{cookies_path}' tidak ditemukan atau invalid.")


async def scrape_x(cookies_path, raw_query, cutoff_dt, limit=100, client=None):
    if client is None:
        client = await get_or_refresh_x_client(cookies_path)

    since_date = cutoff_dt.strftime("%Y-%m-%d")
    formatted_query = f"({raw_query}) since:{since_date} lang:id" if " OR " in raw_query else f"{raw_query} since:{since_date} lang:id"

    data = []
    try:
        tweets = await client.search_tweet(formatted_query, product="Top", count=min(limit, 20))

        while tweets and len(data) < limit:
            for tweet in tweets:
                data.append({
                    "timestamp": tweet.created_at,
                    "platform": "X",
                    "author": tweet.user.screen_name,
                    "text": tweet.text,
                    "engagement": (tweet.favorite_count or 0) + (tweet.retweet_count or 0),
                    "topic": raw_query,
                    "url": f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}"
                })
                if len(data) >= limit:
                    break

            await asyncio.sleep(2)
            tweets = await tweets.next()

    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if "NoneType" in err_msg or "group" in err_msg:
            logger.error("[X/Twitter] Query '%s' terhenti | ERR: Format API/JS internal X berubah (Twikit butuh update).", raw_query)
        else:
            logger.warning("[X/Twitter] Query '%s' terhenti | ERR: %s", raw_query, err_msg)

    return data
