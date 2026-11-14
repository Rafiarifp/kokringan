### Subprogram 1: `core/data_foundation.py`
"""
Subprogram 1: Data Foundation & Canonical Schema Adapter.
Consolidates text_preprocessing_2.py and schema normalization from run_closed_loop.py,
app_fin_2.py, trend_analysis_2.py, sentiment_aggregation_3.py, and anomaly modules.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import pandas as pd

NEWS_OUTLETS = {
    "CNN Indonesia", "CNBC Indonesia", "Detik", "ANTARA",
    "Liputan6", "Okezone", "Republika", "Tempo", "Tempo Bisnis",
    "Kompas", "Kumparan",
}
NEWS_OUTLETS_LOWER = {o.lower() for o in NEWS_OUTLETS}

SENTIMENT_EN_TO_ID = {
    "positive": "Positif",
    "negative": "Negatif",
    "neutral": "Netral",
    "positif": "Positif",
    "negatif": "Negatif",
    "netral": "Netral",
}

SENTIMENT_ID_TO_EN = {
    "positif": "positive",
    "negatif": "negative",
    "netral": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}

VALID_SENTIMENTS_EN = {"positive", "neutral", "negative"}
TRUE_BOOL_STRINGS = {"true", "1", "yes", "ya"}

REQUIRED_STANDARD_COLS = [
    "id", "timestamp", "platform", "platform_detail", "author",
    "text", "content", "cleaned_text", "engagement",
    "topic", "keyword", "url",
    "sentiment", "sentiment_en", "sentiment_confidence",
    "positive_score", "neutral_score", "negative_score",
    "sarcasm", "is_sarcastic", "sarcasm_confidence", "sarcasm_status",
    "interpretation", "needs_review", "narrative_cluster_id",
]

CATEGORICAL_COLS = ("platform", "platform_detail", "sentiment", "sarcasm", "topic")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"\brt\b", flags=re.IGNORECASE)
_KEYWORD_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_\s]{2,30}:\s*")
_CONTROL_RE = re.compile(r"[\r\n\t]+")
_MULTISPACE_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[a-zA-Z0-9]")


def make_post_id(row: dict[str, Any] | pd.Series) -> str:
    """Generate a deterministic 16-char SHA-256 post ID."""
    raw_key = (
        f"{row.get('platform', '')}|{row.get('timestamp', '')}|"
        f"{row.get('author', '')}|{str(row.get('text', row.get('content', '')))[:80]}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def clean_text(text: Any) -> str:
    """Normalize unicode, strip URLs/mentions/control chars, and keep hashtag words."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _URL_RE.sub("", value)
    value = _MENTION_RE.sub("", value)
    value = _HASHTAG_RE.sub(r"\1", value)
    value = _CONTROL_RE.sub(" ", value)
    value = _MULTISPACE_RE.sub(" ", value).strip()
    return value


def validate_text(text: str, min_length: int = 3) -> bool:
    """Return True if text meets minimum length and contains alphanumeric characters."""
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < min_length:
        return False
    return bool(_ALNUM_RE.search(stripped))


def preprocess_text(text: Any, min_length: int = 3) -> str:
    """Clean and validate text for transformer classification; return '' if invalid."""
    cleaned = clean_text(text)
    return cleaned if validate_text(cleaned, min_length=min_length) else ""


def clean_text_for_embedding(text: Any) -> str:
    """Prepare text for semantic sentence embeddings (strips RT and keyword prefixes)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text).strip()
    if not value:
        return ""
    value = _URL_RE.sub("", value)
    value = _MENTION_RE.sub("", value)
    value = _HASHTAG_RE.sub(r"\1", value)
    value = _RT_RE.sub("", value)
    value = _KEYWORD_PREFIX_RE.sub("", value)
    value = _MULTISPACE_RE.sub(" ", value).strip()
    return value


def empty_standardized_df() -> pd.DataFrame:
    """Return an empty DataFrame matching REQUIRED_STANDARD_COLS."""
    return pd.DataFrame(columns=REQUIRED_STANDARD_COLS)


def to_bool_series(series: pd.Series) -> pd.Series:
    """Coerce mixed boolean/string/numeric Series into strict boolean dtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(TRUE_BOOL_STRINGS)


def normalize_posts_df(
    data: pd.DataFrame | list[dict[str, Any]] | None,
    default_keyword: str = "general",
    categorize: bool = False,
) -> pd.DataFrame:
    """
    Convert raw scraper CSVs, database records, or enriched CSVs into one canonical schema.
    Synchronizes text<->content, topic<->keyword, sentiment (ID)<->sentiment_en, and UTC timestamps.
    """
    if data is None:
        return empty_standardized_df()

    df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
    if df.empty:
        return empty_standardized_df()

    # Synchronize text <-> content aliases
    if "text" not in df.columns and "content" in df.columns:
        df["text"] = df["content"]
    elif "content" not in df.columns and "text" in df.columns:
        df["content"] = df["text"]
    elif "text" in df.columns and "content" in df.columns:
        df["text"] = df["text"].fillna(df["content"])
        df["content"] = df["content"].fillna(df["text"])

    # Synchronize topic <-> keyword aliases
    if "topic" not in df.columns and "keyword" in df.columns:
        df["topic"] = df["keyword"]
    elif "keyword" not in df.columns and "topic" in df.columns:
        df["keyword"] = df["topic"]

    defaults: dict[str, Any] = {
        "timestamp": pd.NA,
        "platform": "UNKNOWN",
        "author": "unknown",
        "text": "",
        "content": "",
        "cleaned_text": pd.NA,
        "engagement": 0,
        "topic": default_keyword,
        "keyword": default_keyword,
        "url": "",
        "sentiment": "Netral",
        "sentiment_en": "neutral",
        "sentiment_confidence": 0.0,
        "positive_score": 0.0,
        "neutral_score": 0.0,
        "negative_score": 0.0,
        "sarcasm": "not_sarcastic",
        "is_sarcastic": False,
        "sarcasm_confidence": 0.0,
        "sarcasm_status": "invalid",
        "interpretation": "ordinary_neutral",
        "needs_review": False,
        "narrative_cluster_id": -1,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = df["text"]

    raw_cleaned = df["cleaned_text"].fillna("").astype(str)
    computed_cleaned = df["text"].map(clean_text)
    df["cleaned_text"] = raw_cleaned.where(raw_cleaned.str.strip() != "", computed_cleaned)

    df["author"] = df["author"].fillna("unknown").astype(str).str.strip()
    df["topic"] = df["topic"].fillna(default_keyword).astype(str).str.strip()
    df["topic"] = df["topic"].where(df["topic"] != "", default_keyword)
    df["keyword"] = df["topic"]
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0).astype(int)
    for score_col in ("sentiment_confidence", "positive_score", "neutral_score", "negative_score", "sarcasm_confidence"):
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df["narrative_cluster_id"] = pd.to_numeric(df["narrative_cluster_id"], errors="coerce").fillna(-1).astype(int)

    # Platform & news outlet normalization
    raw_detail = df["platform_detail"] if "platform_detail" in df.columns else df["platform"]
    df["platform_detail"] = raw_detail.fillna("UNKNOWN").astype(str).str.strip()
    df["platform"] = df["platform_detail"].where(
        ~df["platform_detail"].str.lower().isin(NEWS_OUTLETS_LOWER), "News"
    )

    # Sentiment dual-language normalization
    raw_sent = df["sentiment"].fillna("Netral").astype(str).str.strip().str.lower()
    raw_sent_en = df["sentiment_en"].fillna("").astype(str).str.strip().str.lower()
    resolved_en = raw_sent_en.where(raw_sent_en.isin(VALID_SENTIMENTS_EN), raw_sent.map(SENTIMENT_ID_TO_EN).fillna("neutral"))
    df["sentiment_en"] = resolved_en
    df["sentiment"] = resolved_en.map(SENTIMENT_EN_TO_ID).fillna("Netral")

    # Sarcasm normalization
    sarc_norm = df["sarcasm"].fillna("not_sarcastic").astype(str).str.strip().str.lower()
    df["sarcasm"] = sarc_norm.where(sarc_norm.isin({"sarcastic", "not_sarcastic"}), "not_sarcastic")
    df["is_sarcastic"] = to_bool_series(df["is_sarcastic"]) | (df["sarcasm"] == "sarcastic")
    df["needs_review"] = to_bool_series(df["needs_review"])

    # UTC timestamp enforcement
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Deterministic ID
    if "id" not in df.columns or df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        generated_ids = df.apply(make_post_id, axis=1)
        if "id" not in df.columns:
            df["id"] = generated_ids
        else:
            existing = df["id"].fillna("").astype(str).str.strip()
            df["id"] = existing.where(existing != "", generated_ids)
    else:
        df["id"] = df["id"].astype(str).str.strip()

    if categorize:
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category")

    return df
