import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import pandas as pd

TIMEZONE = "Asia/Jakarta"
COLUMNS = [
    "timestamp", "platform", "author", "text",
    "cleaned_text", "engagement", "topic", "url"
]

logger = logging.getLogger("osint_ingestion")

DEFAULT_SLANG = {
    "bgt": "banget", "dgn": "dengan", "tdk": "tidak",
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "yg": "yang",
    "aja": "saja", "krn": "karena", "karna": "karena",
    "dr": "dari", "dlm": "dalam", "utk": "untuk",
    "klo": "kalau", "kalo": "kalau", "udah": "sudah", "udh": "sudah",
    "blm": "belum", "sm": "sama", "sbg": "sebagai", "tp": "tetapi",
    "tapi": "tetapi", "skrg": "sekarang",
    "org": "orang", "dl": "dulu"
}

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"(?<!\w)@\w+")
_RE_NON_ALLOWED = re.compile(r"[^a-z0-9\s.,!?;:'\"()\-]")
_RE_WHITESPACE = re.compile(r"\s+")


def _prepare_slang_items(slang_dict):
    """Konversi dictionary slang ke tuple terurut satu kali agar cache optimal."""
    if not slang_dict:
        return None
    if isinstance(slang_dict, tuple):
        return slang_dict
    return tuple(sorted(
        (str(k).strip().lower(), str(v).strip().lower())
        for k, v in slang_dict.items()
    ))


@lru_cache(maxsize=8)
def _compile_slang_matcher(slang_items):
    """Compile a single longest-first word-boundary regex for O(N) slang replacement."""
    mapping = {
        k: v for k, v in slang_items
        if k and v and k != v
    }
    if not mapping:
        return None, {}

    sorted_keys = sorted(mapping.keys(), key=lambda s: (-len(s), s))
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
    )
    return pattern, mapping


def calculate_cutoff_datetime(timeframe_str: str) -> datetime:
    """Calculate UTC cutoff datetime from timeframe string."""
    now = datetime.now(timezone.utc)
    tf = str(timeframe_str).lower().strip()

    if tf in ("1w", "week", "last_week"):
        days = 7
    elif tf in ("2w", "2weeks"):
        days = 14
    elif tf in ("1m", "month", "last_month"):
        days = 30
    elif tf in ("1q", "quarter", "last_quarter"):
        days = 90
    elif tf.endswith("d") and tf[:-1].isdigit():
        days = int(tf[:-1])
    else:
        days = 30

    return now - timedelta(days=days)


def load_slang_dictionary(path="slang_dict.json"):
    """Load slang dictionary from JSON file; fall back to DEFAULT_SLANG on error."""
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Kamus slang '%s' tidak ditemukan | Memakai dictionary bawaan.", path)
        return DEFAULT_SLANG.copy()

    try:
        with file_path.open("r", encoding="utf-8") as file:
            slang = json.load(file)
        if not isinstance(slang, dict):
            raise ValueError("Isi slang_dict.json harus berupa JSON object.")

        cleaned = {}
        for k, v in slang.items():
            norm_k = str(k).strip().lower()
            norm_v = str(v).strip().lower()
            if norm_k and norm_v and norm_k != norm_v:
                cleaned[norm_k] = norm_v
        return cleaned
    except Exception as exc:
        err_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logger.warning("Gagal membaca kamus slang (%s) | ERR: %s", path, err_msg)
        return DEFAULT_SLANG.copy()


def normalize_timestamp(value):
    """Convert datetime, string, or Unix epoch to Asia/Jakarta pandas Timestamp."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            unit = "ms" if abs(value) >= 1e11 else "s"
            timestamp = pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
        else:
            timestamp = pd.to_datetime(value, utc=True, errors="coerce")

        if pd.isna(timestamp):
            return pd.NaT
        return timestamp.tz_convert(TIMEZONE)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


_ID_TZ_PATTERN = re.compile(r"\b(WIB|WITA|WIT)\b", re.IGNORECASE)
_ID_TZ_OFFSETS = {"WIB": "+0700", "WITA": "+0800", "WIT": "+0900"}


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """Vectorized conversion of mixed datetime/string/epoch Series to Asia/Jakarta."""
    if series.empty:
        return pd.Series(dtype=f"datetime64[ns, {TIMEZONE}]")

    is_num = series.apply(
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v)
    )
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    if is_num.any():
        num_vals = pd.to_numeric(series[is_num], errors="coerce")
        ms_mask = num_vals.abs() >= 1e11
        if (~ms_mask).any():
            result.loc[num_vals[~ms_mask].index] = pd.to_datetime(
                num_vals[~ms_mask], unit="s", utc=True, errors="coerce"
            )
        if ms_mask.any():
            result.loc[num_vals[ms_mask].index] = pd.to_datetime(
                num_vals[ms_mask], unit="ms", utc=True, errors="coerce"
            )

    if (~is_num).any():
        non_num = (
            series[~is_num]
            .replace("", pd.NA)
            .astype("string")
            .str.replace(
                _ID_TZ_PATTERN,
                lambda m: _ID_TZ_OFFSETS[m.group(1).upper()],
                regex=True,
            )
        )
        result.loc[non_num.index] = pd.to_datetime(
            non_num, format="mixed", utc=True, errors="coerce"
        )

    return result.dt.tz_convert(TIMEZONE)


def clean_text(text, slang_dict=None):
    """Strip HTML, URLs, mentions, non-ASCII chars, and normalize slang in O(N)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    value = html.unescape(html.unescape(str(text)))
    value = _RE_HTML_TAG.sub(" ", value)
    value = _RE_URL.sub(" ", value)
    value = _RE_MENTION.sub(" ", value)
    value = value.replace("#", "").lower()

    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = _RE_NON_ALLOWED.sub(" ", value)
    value = _RE_WHITESPACE.sub(" ", value).strip()

    slang_items = _prepare_slang_items(slang_dict)
    if slang_items and value:
        pattern, mapping = _compile_slang_matcher(slang_items)
        if pattern is not None:
            value = pattern.sub(lambda m: mapping[m.group(0)], value)
            value = _RE_WHITESPACE.sub(" ", value).strip()

    return value


def standardize_records(records, topic="", slang_dict=None, cutoff_dt=None):
    """Standardize schema and filter records by cutoff timestamp."""
    if not records:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(records)
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = None

    prepared_slang = _prepare_slang_items(slang_dict)

    df["timestamp"] = _normalize_timestamp_series(df["timestamp"])
    df["platform"] = df["platform"].replace("", pd.NA).fillna("Unknown").astype(str)
    df["author"] = df["author"].replace("", pd.NA).fillna("Unknown").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["cleaned_text"] = df["text"].apply(lambda v: clean_text(v, prepared_slang))
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0)
    df["topic"] = df["topic"].replace("", pd.NA).fillna(topic).astype(str)
    df["url"] = df["url"].fillna("").astype(str)

    df = df.dropna(subset=["timestamp"])
    df = df[df["cleaned_text"].str.len() > 0]

    if cutoff_dt is not None:
        cutoff_tz = pd.to_datetime(cutoff_dt, utc=True).tz_convert(TIMEZONE)
        df = df[df["timestamp"] >= cutoff_tz]

    return df[COLUMNS].reset_index(drop=True)
