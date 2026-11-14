"""
KOKRINGAN | PUSSIBERAD Command Center
Dashboard intelijen siber sosial (Streamlit) - Bento Kinetic (Spatial Modular Grid).
"""
import html
import inspect
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

from app.ingestion_control import render_ingestion_panel

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & UTILITAS INTI
# ---------------------------------------------------------
st.set_page_config(
    page_title="KOKRINGAN | PUSSIBERAD Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT_SANS = "'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, 'Courier New', monospace"


def _stretch(fn) -> dict:
    """Kompatibel lintas versi Streamlit (width='stretch' vs use_container_width)."""
    try:
        param = inspect.signature(fn).parameters.get("width")
        if param is not None and ("stretch" in str(param.annotation) or param.default == "stretch"):
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


PLOTLY_KW = _stretch(st.plotly_chart)
BUTTON_KW = _stretch(st.button)
DOWNLOAD_KW = _stretch(st.download_button)

esc = html.escape


def H(markup: str) -> str:
    """Rapatkan HTML sambil mempertahankan spasi antar-baris agar tidak dibaca sebagai code-block."""
    return " ".join(s for line in markup.splitlines() if (s := line.strip()))


def html_block(markup: str) -> None:
    st.markdown(H(markup), unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. PEMUATAN DATA & STANDARDISASI SKEMA (TANPA DATA FIKTIF)
# ---------------------------------------------------------
NEWS_OUTLETS = {
    "CNN Indonesia", "CNBC Indonesia", "Detik", "ANTARA",
    "Liputan6", "Okezone", "Republika", "Tempo", "Tempo Bisnis",
    "Kompas", "Kumparan",
}
NEWS_OUTLETS_LOWER = {o.lower() for o in NEWS_OUTLETS}
TRUE_BOOL_STRINGS = {"true", "1", "yes", "ya"}

REQUIRED_STANDARD_COLS = [
    "timestamp", "platform", "platform_detail", "author", "text",
    "cleaned_text", "engagement", "topic", "url", "sentiment",
    "sentiment_confidence", "sarcasm", "is_sarcastic",
    "sarcasm_confidence", "sarcasm_status", "interpretation",
    "needs_review", "narrative_cluster_id",
]

TRENDS_COLS = ["timestamp", "keyword", "interest_score", "geo"]
CATEGORICAL_COLS = ("platform", "platform_detail", "sentiment", "sarcasm", "topic")


def _empty_standardized_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_STANDARD_COLS)


def _empty_trends_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRENDS_COLS)


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(TRUE_BOOL_STRINGS)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_standardized_df()

    defaults = {
        "timestamp": pd.NA,
        "platform": "UNKNOWN",
        "author": "unknown",
        "text": "",
        "cleaned_text": pd.NA,
        "engagement": 0,
        "topic": "general",
        "url": "",
        "sentiment": "Netral",
        "sentiment_confidence": 0.0,
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
    cleaned = df["cleaned_text"].fillna("").astype(str)
    df["cleaned_text"] = cleaned.where(cleaned.str.strip() != "", df["text"])

    df["author"] = df["author"].fillna("unknown").astype(str).str.strip()
    df["topic"] = df["topic"].fillna("general").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str)
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0).astype(int)
    df["sentiment_confidence"] = pd.to_numeric(df["sentiment_confidence"], errors="coerce").fillna(0.0)
    df["sarcasm_confidence"] = pd.to_numeric(df["sarcasm_confidence"], errors="coerce").fillna(0.0)
    df["narrative_cluster_id"] = pd.to_numeric(df["narrative_cluster_id"], errors="coerce").fillna(-1).astype(int)

    df["needs_review"] = _to_bool_series(df["needs_review"])
    sarcasm_norm = df["sarcasm"].fillna("not_sarcastic").astype(str).str.strip().str.lower()
    df["sarcasm"] = sarcasm_norm.where(sarcasm_norm.isin({"sarcastic", "not_sarcastic"}), "not_sarcastic")
    df["is_sarcastic"] = _to_bool_series(df["is_sarcastic"]) | (df["sarcasm"] == "sarcastic")

    raw_detail = df["platform_detail"] if "platform_detail" in df.columns else df["platform"]
    df["platform_detail"] = raw_detail.fillna("UNKNOWN").astype(str).str.strip()
    df["platform"] = df["platform_detail"].where(
        ~df["platform_detail"].str.lower().isin(NEWS_OUTLETS_LOWER), "News"
    )

    sent_map = {
        "positive": "Positif",
        "negative": "Negatif",
        "neutral": "Netral",
        "positif": "Positif",
        "negatif": "Negatif",
        "netral": "Netral",
    }
    df["sentiment"] = (
        df["sentiment"]
        .fillna("Netral")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(sent_map)
        .fillna("Netral")
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def _standardize_trends_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_trends_df()

    out = df.copy()
    for col in TRENDS_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["keyword"] = out["keyword"].fillna("").astype(str).str.strip()
    out["geo"] = out["geo"].fillna("ID").astype(str).str.strip().replace("", "ID")
    out["interest_score"] = (
        pd.to_numeric(out["interest_score"], errors="coerce")
        .fillna(0)
        .clip(0, 100)
        .astype(int)
    )
    out = out.dropna(subset=["timestamp"])
    out = out[out["keyword"] != ""].sort_values("timestamp").reset_index(drop=True)
    return out[TRENDS_COLS]


def _file_signature(path: Path) -> tuple:
    if not path.exists():
        return (0, 0)
    st_info = path.stat()
    return (st_info.st_mtime_ns, st_info.st_size)


@st.cache_data(show_spinner=False)
def load_data(signatures: tuple):
    """Muat hasil NLP terbaru saja; CSV mentah tidak pernah menjadi fallback."""
    enriched_path = DATA_DIR / "master_osint_enriched.csv"
    raw_path = DATA_DIR / "master_osint_data.csv"
    report_path = DATA_DIR / "cyberlens_report.json"
    empty = _empty_standardized_df()

    if not enriched_path.exists() or not report_path.exists():
        return {
            "df": empty,
            "report": {},
            "status": "pending_analysis" if raw_path.exists() else "missing",
            "error": None,
            "warnings": [],
        }

    if raw_path.exists():
        raw_mtime = raw_path.stat().st_mtime_ns
        if (
            enriched_path.stat().st_mtime_ns < raw_mtime
            or report_path.stat().st_mtime_ns < raw_mtime
        ):
            return {
                "df": empty,
                "report": {},
                "status": "pending_analysis",
                "error": None,
                "warnings": [
                    "Dataset scraping lebih baru daripada hasil NLP. Dashboard menunggu analisis terbaru selesai."
                ],
            }

    try:
        with report_path.open("r", encoding="utf-8") as f:
            report_data = json.load(f)
        if not isinstance(report_data, dict):
            raise ValueError("Format cyberlens_report.json harus berupa object/dictionary.")

        allowed_cols = set(REQUIRED_STANDARD_COLS)
        enriched_df = pd.read_csv(
            enriched_path,
            usecols=lambda col: col in allowed_cols,
        )
        standardized = _standardize_dataframe(enriched_df)
        return {
            "df": standardized,
            "report": report_data,
            "status": "ok" if not standardized.empty else "empty_file",
            "error": None,
            "warnings": [],
        }
    except Exception as exc:
        return {
            "df": empty,
            "report": {},
            "status": "error",
            "error": f"Gagal memuat hasil NLP terbaru: {exc}",
            "warnings": [],
        }


@st.cache_data(show_spinner=False)
def load_trends_data(signature: tuple) -> dict:
    """Muat deret waktu minat pencarian Google Trends dari data/trends_interest.csv."""
    trends_path = DATA_DIR / "trends_interest.csv"
    if not trends_path.exists():
        return {"df": _empty_trends_df(), "status": "missing", "error": None}
    try:
        raw_trends = pd.read_csv(trends_path)
        standardized = _standardize_trends_df(raw_trends)
        return {
            "df": standardized,
            "status": "ok" if not standardized.empty else "empty_file",
            "error": None,
        }
    except Exception as exc:
        return {
            "df": _empty_trends_df(),
            "status": "error",
            "error": f"Gagal membaca trends_interest.csv: {exc}",
        }


@st.cache_data(show_spinner=False, max_entries=4)
def export_filtered_csv(_filter_sig: tuple, _df: pd.DataFrame) -> bytes:
    """Cache serialisasi CSV agar rerun widget grafik tidak mengulang encode seluruh tabel."""
    return _df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------
# 3. TEMA & DESIGN TOKENS (TERANG & GELAP, WCAG AA)
# ---------------------------------------------------------
def _reset_filters(platforms, sentiments, topics):
    st.session_state["filter_platforms"] = platforms
    st.session_state["filter_sentiments"] = sentiments
    st.session_state["filter_topics"] = topics
    st.session_state["filter_sarcasm"] = False


with st.sidebar:
    html_block("""
    <div class="side-brand"><div class="logo">K</div>
    <div><strong>KOKRINGAN</strong><span>Analisis Lingkungan Siber</span></div></div>
    <div class="side-title">Mode Tampilan</div>
    """)
    theme_choice = st.radio(
        "Mode Tampilan:",
        options=["Terang", "Gelap"],
        horizontal=True,
        key="ui_theme_mode",
        label_visibility="collapsed",
    )

is_dark = theme_choice == "Gelap"

LIGHT = {
    "scheme": "light",
    "bg": "#f4f6f9",
    "mesh": "radial-gradient(circle at 12% 10%, rgba(4,120,87,0.08), transparent 42%), radial-gradient(circle at 88% 12%, rgba(79,70,229,0.06), transparent 44%)",
    "surface": "#ffffff",
    "sidebar_glass": "linear-gradient(180deg, rgba(255, 255, 255, 0.86) 0%, rgba(241, 245, 249, 0.82) 100%)",
    "surface_alt": "#eef2f6",
    "zebra": "#f8fafc",
    "border": "rgba(15, 23, 42, 0.09)",
    "border_strong": "rgba(15, 23, 42, 0.18)",
    "rim": "rgba(255, 255, 255, 0.95)",
    "shadow": "0 2px 6px rgba(15, 23, 42, 0.04), 0 10px 24px rgba(15, 23, 42, 0.05)",
    "shadow_hover": "0 6px 18px rgba(15, 23, 42, 0.08), 0 14px 28px rgba(4, 120, 87, 0.08)",
    "text": "#0f172a",
    "text_2": "#334155",
    "muted": "#4b5563",
    "brand": "#047857",
    "brand_soft": "rgba(4, 120, 87, 0.10)",
    "brand_fill": "rgba(4, 120, 87, 0.14)",
    "accent": "#b45309",
    "pos": "#047857",
    "neg": "#be123c",
    "neu": "#475569",
    "danger_bg": "#fff1f2",
    "danger_fg": "#881337",
    "danger_bd": "rgba(190, 18, 60, 0.32)",
    "warn_bg": "#fffbeb",
    "warn_fg": "#78350f",
    "warn_bd": "rgba(180, 83, 9, 0.32)",
    "ok_bg": "#ecfdf5",
    "ok_fg": "#064e3b",
    "ok_bd": "rgba(4, 120, 87, 0.32)",
    "hero": "linear-gradient(135deg, #064e3b 0%, #0f766e 55%, #312e81 100%)",
    "grid": "#e2e8f0",
    "plotly_template": "plotly_white",
    "palette": ["#047857", "#4f46e5", "#b45309", "#0369a1", "#be123c", "#0f766e"],
    "heatmap_scale": ["#f1f5f9", "#a7f3d0", "#34d399", "#059669", "#064e3b"],
    "wc_pos": ("#065f46", "#047857", "#0f766e", "#4338ca", "#064e3b"),
    "wc_neg": ("#9f1239", "#be123c", "#881337", "#b45309", "#7f1d1d"),
}

DARK = {
    "scheme": "dark",
    "bg": "#0b0f17",
    "mesh": "radial-gradient(circle at 12% 10%, rgba(52,211,153,0.12), transparent 42%), radial-gradient(circle at 88% 12%, rgba(129,140,248,0.09), transparent 44%)",
    "surface": "#121824",
    "sidebar_glass": "linear-gradient(180deg, rgba(22, 30, 46, 0.78) 0%, rgba(13, 18, 28, 0.74) 100%)",
    "surface_alt": "#1c2536",
    "zebra": "#0f141f",
    "border": "rgba(255, 255, 255, 0.09)",
    "border_strong": "rgba(255, 255, 255, 0.18)",
    "rim": "rgba(255, 255, 255, 0.08)",
    "shadow": "0 4px 12px rgba(0, 0, 0, 0.28), 0 12px 28px rgba(0, 0, 0, 0.24)",
    "shadow_hover": "0 8px 22px rgba(0, 0, 0, 0.38), 0 14px 30px rgba(52, 211, 153, 0.10)",
    "text": "#f1f5f9",
    "text_2": "#cbd5e1",
    "muted": "#94a3b8",
    "brand": "#34d399",
    "brand_soft": "rgba(52, 211, 153, 0.14)",
    "brand_fill": "rgba(52, 211, 153, 0.18)",
    "accent": "#fbbf24",
    "pos": "#34d399",
    "neg": "#fb7185",
    "neu": "#94a3b8",
    "danger_bg": "#2c111b",
    "danger_fg": "#fecdd3",
    "danger_bd": "rgba(251, 113, 133, 0.36)",
    "warn_bg": "#2b1f0e",
    "warn_fg": "#fef08a",
    "warn_bd": "rgba(251, 191, 36, 0.36)",
    "ok_bg": "#0e2920",
    "ok_fg": "#bbf7d0",
    "ok_bd": "rgba(52, 211, 153, 0.36)",
    "hero": "linear-gradient(135deg, #064e3b 0%, #115e59 55%, #312e81 100%)",
    "grid": "#1e2738",
    "plotly_template": "plotly_dark",
    "palette": ["#34d399", "#818cf8", "#fbbf24", "#38bdf8", "#fb7185", "#2dd4bf"],
    "heatmap_scale": ["#162032", "#065f46", "#059669", "#10b981", "#6ee7b7"],
    "wc_pos": ("#34d399", "#6ee7b7", "#818cf8", "#2dd4bf", "#a7f3d0"),
    "wc_neg": ("#fb7185", "#fda4af", "#f43f5e", "#fb923c", "#fecdd3"),
}

T = DARK if is_dark else LIGHT
SENT_COLORS = {"Positif": T["pos"], "Netral": T["neu"], "Negatif": T["neg"]}
PALETTE = T["palette"]

CSS_VARS = f"""
:root, #root .stApp {{
  color-scheme: {T["scheme"]};
  --background-color: {T["bg"]};
  --secondary-background-color: {T["surface"]};
  --text-color: {T["text"]};
  --primary-color: {T["brand"]};
  --bg:{T["bg"]}; --mesh:{T["mesh"]};
  --surface:{T["surface"]}; --sidebar-glass:{T["sidebar_glass"]};
  --surface-alt:{T["surface_alt"]}; --zebra:{T["zebra"]};
  --border:{T["border"]}; --border-strong:{T["border_strong"]}; --rim:{T["rim"]};
  --shadow:{T["shadow"]}; --shadow-hover:{T["shadow_hover"]}; --hero:{T["hero"]};
  --text:{T["text"]}; --text-2:{T["text_2"]}; --muted:{T["muted"]};
  --brand:{T["brand"]}; --brand-soft:{T["brand_soft"]}; --accent:{T["accent"]};
  --pos:{T["pos"]}; --neg:{T["neg"]}; --neu:{T["neu"]};
  --danger-bg:{T["danger_bg"]}; --danger-fg:{T["danger_fg"]}; --danger-bd:{T["danger_bd"]};
  --warn-bg:{T["warn_bg"]}; --warn-fg:{T["warn_fg"]}; --warn-bd:{T["warn_bd"]};
  --ok-bg:{T["ok_bg"]}; --ok-fg:{T["ok_fg"]}; --ok-bd:{T["ok_bd"]};
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --font:{FONT_SANS}; --mono:{FONT_MONO};
}}
"""

CSS_RULES = """
/* ===== Kanvas Utama ===== */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background-color: var(--bg) !important;
  background-image: var(--mesh) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
.block-container { max-width: 1460px; padding: 1.4rem 2.1rem 3.2rem !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"] *, [data-testid="stToolbar"] * { color: var(--text-2) !important; }
[data-testid="stAppDeployButton"], .stDeployButton, footer { display: none !important; }

.stApp :where(p, li, label, span, small, h1, h2, h3, h4, h5, h6, legend, summary) { color: var(--text) !important; }
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stWidgetLabel"] label {
  color: var(--text-2) !important; font-weight: 600; font-size: .85rem;
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
.stApp hr { border-color: var(--border) !important; }
.stApp a { color: var(--brand) !important; font-weight: 600; }
.stApp code {
  background: var(--surface-alt) !important; color: var(--text) !important;
  border-radius: 6px; padding: 2px 7px; border: 1px solid var(--border);
}

/* ===== Sidebar Spatial Drawer (Satu-satunya Permukaan Frosted Glass) ===== */
section[data-testid="stSidebar"] {
  background: var(--sidebar-glass) !important;
  backdrop-filter: blur(18px) saturate(140%) !important;
  -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
  border-right: 1px solid var(--border-strong) !important;
  box-shadow: inset -1px 0 0 var(--rim), 6px 0 24px rgba(0, 0, 0, 0.10);
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  background: transparent !important;
}
.stApp .side-brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 18px; }
.stApp .side-brand .logo {
  width: 42px; height: 42px; border-radius: 12px; background: var(--hero);
  color: #ffffff !important; display: grid; place-items: center; font-weight: 800; font-size: 1.2rem;
}
.stApp .side-brand strong { display: block; font-size: 1rem; letter-spacing: .01em; color: var(--text) !important; }
.stApp .side-brand span { display: block; font-size: .72rem; color: var(--muted) !important; letter-spacing: .04em; }
.stApp .side-title { font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted) !important; margin: 16px 0 6px; }
.stApp .side-note {
  font-size: .8rem; line-height: 1.65; color: var(--text-2) !important;
  background: var(--surface);
  border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px;
}
.stApp .side-note b { color: var(--text) !important; }

/* ===== Input, Selector & Pill Tags (Terang & Gelap) ===== */
#root .stApp [data-testid="stSelectbox"] > div,
#root .stApp [data-testid="stMultiSelect"] > div,
#root .stApp [data-baseweb="select"],
#root .stApp [data-baseweb="input"],
#root .stApp [data-baseweb="base-input"] {
  background: transparent !important;
  border: none !important;
}

#root .stApp [data-testid="stSelectbox"] > div > div,
#root .stApp [data-testid="stMultiSelect"] > div > div,
#root .stApp [data-baseweb="select"] > div,
#root .stApp [data-baseweb="input"] > div,
#root .stApp [data-baseweb="base-input"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 12px !important;
  min-height: 44px !important;
  color: var(--text) !important;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
}

#root .stApp [data-testid="stSelectbox"] > div > div div:not([data-baseweb="tag"]):not([data-baseweb="tag"] *),
#root .stApp [data-testid="stMultiSelect"] > div > div div:not([data-baseweb="tag"]):not([data-baseweb="tag"] *),
#root .stApp [data-baseweb="select"] > div div:not([data-baseweb="tag"]):not([data-baseweb="tag"] *),
#root .stApp [data-baseweb="input"] > div div,
#root .stApp [data-baseweb="base-input"] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--text) !important;
}

#root .stApp [data-testid="stSelectbox"] > div > div:hover,
#root .stApp [data-testid="stMultiSelect"] > div > div:hover,
#root .stApp [data-baseweb="select"]:hover > div {
  border-color: var(--brand) !important;
}

#root .stApp [data-testid="stSelectbox"] > div > div:focus-within,
#root .stApp [data-testid="stMultiSelect"] > div > div:focus-within,
#root .stApp [data-baseweb="select"]:focus-within > div {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 3px var(--brand-soft) !important;
}

#root .stApp [data-testid="stSelectbox"] span,
#root .stApp [data-testid="stSelectbox"] input,
#root .stApp [data-testid="stMultiSelect"] span,
#root .stApp [data-testid="stMultiSelect"] input,
#root .stApp [data-baseweb="select"] span,
#root .stApp [data-baseweb="select"] input {
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}

#root .stApp [data-testid="stSelectbox"] svg,
#root .stApp [data-testid="stMultiSelect"] svg,
#root .stApp [data-baseweb="select"] svg {
  fill: var(--text-2) !important;
  color: var(--text-2) !important;
}

/* Pill Tag di dalam Multiselect */
#root .stApp [data-testid="stMultiSelect"] [data-baseweb="tag"],
#root .stApp [data-testid="stMultiSelectTag"],
#root .stApp [data-baseweb="tag"] {
  background: var(--brand-soft) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 999px !important;
  padding: 2px 10px !important;
}
#root .stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
#root .stApp [data-testid="stMultiSelectTag"] span,
#root .stApp [data-baseweb="tag"] span {
  color: var(--brand) !important;
  -webkit-text-fill-color: var(--brand) !important;
  font-weight: 600 !important;
}
#root .stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
#root .stApp [data-testid="stMultiSelectTag"] svg,
#root .stApp [data-baseweb="tag"] svg {
  fill: var(--brand) !important;
  color: var(--brand) !important;
}

/* Dropdown Popover */
html body [data-baseweb="popover"],
html body [data-baseweb="popover"] > div,
html body [data-baseweb="popover"] [data-baseweb="menu"],
html body [data-baseweb="popover"] ul[role="listbox"],
html body [data-testid="stSelectboxVirtualDropdown"],
html body [role="listbox"] {
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 14px !important;
  box-shadow: var(--shadow-hover) !important;
}
html body [data-baseweb="popover"] li,
html body [data-baseweb="popover"] li div,
html body [data-baseweb="popover"] li span,
html body [role="listbox"] [role="option"],
html body [role="listbox"] [role="option"] span {
  background: transparent !important;
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}
html body [data-baseweb="popover"] li:hover,
html body [data-baseweb="popover"] li[aria-selected="true"],
html body [role="listbox"] [role="option"]:hover,
html body [role="listbox"] [role="option"][aria-selected="true"] {
  background: var(--brand-soft) !important;
}

/* Radio & Checkbox */
#root .stApp [data-testid="stRadio"] label,
#root .stApp [data-testid="stCheckbox"] label {
  min-height: 44px;
  display: inline-flex;
  align-items: center;
}
#root .stApp [data-testid="stRadio"] label > div:first-child,
#root .stApp [data-testid="stCheckbox"] label > span:first-child {
  background-color: var(--surface) !important;
  border-color: var(--border-strong) !important;
}
#root .stApp [data-testid="stRadio"] input:checked + div,
#root .stApp [data-testid="stCheckbox"] input:checked + span {
  background-color: var(--brand) !important;
  border-color: var(--brand) !important;
}

/* Tombol Sidebar (Tap Target >= 44px) */
#root .stApp .stButton > button,
#root .stApp .stDownloadButton > button,
#root .stApp [data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 12px !important;
  min-height: 44px !important;
  font-weight: 600 !important;
  transition: border-color .18s var(--ease), background-color .18s var(--ease);
}
#root .stApp .stButton > button:hover,
#root .stApp .stDownloadButton > button:hover,
#root .stApp [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--brand) !important;
  background: var(--brand-soft) !important;
  color: var(--brand) !important;
}
#root .stApp .stButton > button p,
#root .stApp .stButton > button span,
#root .stApp .stDownloadButton > button p,
#root .stApp .stDownloadButton > button span,
#root .stApp [data-testid="stBaseButton-secondary"] p,
#root .stApp [data-testid="stBaseButton-secondary"] span {
  color: inherit !important;
  -webkit-text-fill-color: currentColor !important;
}

/* ===== Tabs Navigasi (Tap Target >= 44px) ===== */
#root .stApp [data-testid="stTabs"] [role="tablist"] {
  gap: 8px !important;
  background: transparent !important;
  padding: 4px 2px !important;
  border: none !important;
  flex-wrap: wrap !important;
  width: fit-content !important;
  max-width: 100% !important;
  margin-bottom: 12px !important;
}
#root .stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"],
#root .stApp [data-testid="stTabs"] [data-baseweb="tab-border"] {
  display: none !important;
}
#root .stApp [data-testid="stTabs"] [role="tab"] {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: var(--surface) !important;
  border-radius: 999px !important;
  padding: 10px 18px !important;
  margin: 0 !important;
  min-height: 44px !important;
  border: 1px solid var(--border) !important;
  transition: border-color .18s var(--ease), background-color .18s var(--ease);
}
#root .stApp [data-testid="stTabs"] [role="tab"] > div,
#root .stApp [data-testid="stTabs"] [role="tab"] p {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1.25 !important;
  white-space: nowrap !important;
  color: var(--text-2) !important;
  -webkit-text-fill-color: var(--text-2) !important;
  font-weight: 600 !important;
  font-size: .86rem !important;
  background: transparent !important;
}
#root .stApp [data-testid="stTabs"] [role="tab"]:hover {
  border-color: var(--brand) !important;
  background: var(--brand-soft) !important;
}
#root .stApp [data-testid="stTabs"] [role="tab"]:hover p {
  color: var(--brand) !important;
  -webkit-text-fill-color: var(--brand) !important;
}
#root .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--hero) !important;
  border-color: transparent !important;
}
#root .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div,
#root .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
#root .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-weight: 700 !important;
}

/* ===== Bento Chart Containers ===== */
.stApp [data-testid="stPlotlyChart"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
  padding: 10px 10px 6px; box-shadow: var(--shadow);
  box-sizing: border-box !important;
  width: 100% !important; max-width: 100% !important;
  overflow: hidden !important;
}
.stApp [data-testid="stPlotlyChart"] > div,
.stApp [data-testid="stPlotlyChart"] .js-plotly-plot,
.stApp [data-testid="stPlotlyChart"] .plot-container,
.stApp [data-testid="stPlotlyChart"] .svg-container {
  width: 100% !important;
  max-width: 100% !important;
  overflow: hidden !important;
}

/* ===== Hero Focal Banner ===== */
.stApp .hero {
  background: var(--hero);
  border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 20px; padding: 28px 32px; margin-bottom: 24px;
  box-shadow: var(--shadow); position: relative; overflow: hidden;
}
#root .stApp .hero h1 {
  font-size: clamp(1.55rem, 2.6vw, 2.2rem);
  font-weight: 800;
  letter-spacing: -.025em;
  margin: 12px 0 6px;
  line-height: 1.15;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
.stApp .hero p { margin: 0 0 16px; font-size: .95rem; color: rgba(255, 255, 255, 0.90) !important; max-width: 820px; line-height: 1.55; }
.stApp .hero .badge {
  display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .07em;
  padding: 5px 12px; border-radius: 999px; background: rgba(255, 255, 255, 0.18); color: #ffffff !important; margin-right: 6px;
}
.stApp .hero .chip {
  display: inline-flex; align-items: center; gap: 7px; font-size: .78rem;
  padding: 6px 14px; border-radius: 999px; background: rgba(0, 0, 0, 0.28); color: #ffffff !important; margin: 0 8px 6px 0; font-weight: 500;
}
.stApp .hero .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: #34d399; }
.stApp .hero .dot.warn { background: #fbbf24; }
.stApp .hero .dot.bad { background: #fb7185; }

/* ===== Judul Seksi ===== */
.stApp .sec { margin: 8px 0 16px; }
.stApp .sec h2 { font-size: 1.15rem; font-weight: 700; letter-spacing: -.015em; margin: 0; color: var(--text) !important; }
.stApp .sec p { margin: 4px 0 0; font-size: .86rem; color: var(--muted) !important; }
.stApp .sub { font-size: .98rem; font-weight: 700; letter-spacing: -.01em; margin: 24px 0 12px; color: var(--text) !important; }

/* ===== Bento KPI Cards ===== */
.stApp .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 16px; margin: 4px 0 22px; }
.stApp .kpi {
  --tone: var(--brand);
  background: var(--surface);
  border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px 16px;
  transition: border-color .18s var(--ease);
}
.stApp .kpi:hover { border-color: var(--border-strong); }
.stApp .kpi.tone-neg { --tone: var(--neg); }
.stApp .kpi.tone-warn { --tone: var(--accent); }
.stApp .kpi.tone-neu { --tone: var(--neu); }
.stApp .kpi-label {
  display: flex; align-items: center; gap: 7px; font-size: .75rem; font-weight: 600;
  letter-spacing: .04em; text-transform: uppercase; color: var(--muted) !important;
}
.stApp .kpi-label::before {
  content: ""; width: 8px; height: 8px; border-radius: 999px; background: var(--tone); flex-shrink: 0;
}
.stApp .kpi-value { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: clamp(1.42rem, 2vw, 1.88rem); font-weight: 700; margin: 8px 0 3px; color: var(--text) !important; line-height: 1.15; }
.stApp .kpi-value.txt { font-family: var(--font); font-size: 1.2rem; word-break: break-word; }
.stApp .kpi-hint { font-size: .8rem; color: var(--text-2) !important; }

/* ===== Kartu & Callout ===== */
.stApp .card {
  background: var(--surface);
  border: 1px solid var(--border); border-radius: 16px; padding: 20px 22px;
  color: var(--text) !important; line-height: 1.68; font-size: .93rem;
}
.stApp .card h4 { margin: 0 0 10px; font-size: .76rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted) !important; }
.stApp .card ul { margin: 0; padding-left: 1.15rem; }
.stApp .card li { margin-bottom: 7px; color: var(--text) !important; }
.stApp .empty {
  background: var(--surface); border: 1px dashed var(--border-strong); border-radius: 16px;
  padding: 28px; margin: 12px 0; color: var(--text) !important; line-height: 1.7;
}
.stApp .callout { border-radius: 12px; padding: 13px 18px; margin: 8px 0; font-size: .9rem; border: 1px solid; }
.stApp .callout.warn { background: var(--warn-bg); border-color: var(--warn-bd); color: var(--warn-fg) !important; }
.stApp .callout.bad { background: var(--danger-bg); border-color: var(--danger-bd); color: var(--danger-fg) !important; }
.stApp .callout.ok { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok-fg) !important; }
.stApp .callout.info { background: var(--brand-soft); border-color: var(--border-strong); color: var(--text) !important; }
.stApp .callout.warn *, .stApp .callout.bad *, .stApp .callout.ok * { color: inherit !important; }

/* ===== Pill Status ===== */
.stApp .pill { display: inline-block; font-size: .72rem; font-weight: 700; padding: 3px 11px; border-radius: 999px; border: 1px solid; white-space: nowrap; }
.stApp .pill.pos { color: var(--ok-fg) !important; background: var(--ok-bg); border-color: var(--ok-bd); }
.stApp .pill.neg { color: var(--danger-fg) !important; background: var(--danger-bg); border-color: var(--danger-bd); }
.stApp .pill.warn { color: var(--warn-fg) !important; background: var(--warn-bg); border-color: var(--warn-bd); }
.stApp .pill.neu { color: var(--text-2) !important; background: var(--surface-alt); border-color: var(--border-strong); }

/* ===== Tabel Bento ===== */
.stApp .tbl-wrap {
  overflow: auto; border: 1px solid var(--border); border-radius: 16px;
  background: var(--surface);
}
.stApp table.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .84rem; }
.stApp table.tbl th {
  position: sticky; top: 0; z-index: 1; background: var(--surface-alt); color: var(--text-2) !important;
  text-align: left; padding: 12px 15px; font-size: .71rem; text-transform: uppercase; letter-spacing: .05em;
  border-bottom: 1px solid var(--border-strong); white-space: nowrap;
}
.stApp table.tbl td {
  color: var(--text) !important; padding: 11px 15px; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.stApp table.tbl tr:nth-child(even) td { background: var(--zebra); }
.stApp table.tbl tr:hover td { background: var(--brand-soft); }
.stApp table.tbl td.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.stApp table.tbl td.wrap { min-width: 260px; max-width: 540px; line-height: 1.5; }
.stApp table.tbl td.nw { white-space: nowrap; }

/* ===== Indeks Ancaman & DISARM ===== */
.stApp .threat {
  border-radius: 16px; padding: 24px 26px; border: 1px solid;
  box-shadow: var(--shadow);
}
.stApp .threat.high { background: var(--danger-bg); border-color: var(--danger-bd); color: var(--danger-fg) !important; }
.stApp .threat.medium { background: var(--warn-bg); border-color: var(--warn-bd); color: var(--warn-fg) !important; }
.stApp .threat.low { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok-fg) !important; }
.stApp .threat h4 { margin: 0; font-size: .76rem; letter-spacing: .06em; text-transform: uppercase; color: inherit !important; }
.stApp .threat .score { font-family: var(--mono); font-size: clamp(2rem, 3.6vw, 2.75rem); font-weight: 800; margin: 8px 0; color: inherit !important; }
.stApp .threat .score small { font-size: 1rem; opacity: .85; color: inherit !important; }
.stApp .threat .status { font-family: var(--mono); font-weight: 700; font-size: .84rem; margin: 12px 0 0; color: inherit !important; }
.stApp .meter { height: 9px; border-radius: 999px; background: rgba(127, 127, 127, 0.22); overflow: hidden; }
.stApp .meter > i { display: block; height: 100%; background: currentColor; border-radius: 999px; }
.stApp .ttp {
  display: flex; gap: 12px; align-items: flex-start; background: var(--surface);
  border: 1px solid var(--border); border-radius: 14px;
  padding: 13px 16px; margin-bottom: 10px;
}
.stApp .ttp b { display: block; color: var(--text) !important; font-size: .88rem; }
.stApp .ttp span.d { color: var(--text-2) !important; font-size: .82rem; }
.stApp .evidence {
  background: var(--danger-bg);
  border: 1px solid var(--danger-bd); border-radius: 14px; padding: 15px 18px; margin-bottom: 12px;
}
.stApp .evidence .q { color: var(--danger-fg) !important; font-size: .9rem; line-height: 1.55; }
.stApp .evidence .a { color: var(--danger-fg) !important; font-size: .8rem; margin-top: 8px; }

/* ===== Responsif & Aksesibilitas ===== */
@media (max-width: 768px) {
  .block-container { padding: 1rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .75rem !important; }
  [data-testid="stColumn"], [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
  .stApp .hero { padding: 20px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
.stApp button:focus-visible,
.stApp [role="tab"]:focus-visible,
.stApp a:focus-visible,
.stApp input:focus-visible {
  outline: 3px solid var(--accent) !important;
  outline-offset: 2px !important;
}
"""

st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@500;700&display=swap');"
    + CSS_VARS + CSS_RULES + "</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 4. KOMPONEN UI & ANALITIK TER-CACHE
# ---------------------------------------------------------
COL_LABELS = {
    "timestamp": "Waktu (UTC)", "platform": "Platform", "platform_detail": "Sumber",
    "author": "Akun", "topic": "Topik", "text": "Teks", "sentiment": "Sentimen",
    "sarcasm": "Sarkasme", "engagement": "Engagement", "url": "Tautan",
    "sarcasm_confidence": "Konf. Sarkasme", "interpretation": "Interpretasi",
    "narrative_cluster_id": "Klaster", "jumlah_post": "Jumlah Post",
    "klaster_terlibat": "Klaster Terlibat", "total_engagement": "Total Engagement",
    "keyword": "Kata Kunci", "geo": "Wilayah", "interest_score": "Skor Minat",
    "skor_terbaru": "Skor Terbaru", "rata_rata": "Rata-rata", "puncak": "Puncak",
    "tanggal_puncak": "Tgl. Puncak", "momentum": "Momentum (%)",
    "z_score": "Z-Score", "status_tren": "Status Tren",
}
PILL_CLASS = {
    "positif": "pos", "negatif": "neg", "netral": "neu", "sarcastic": "warn",
    "not_sarcastic": "neu", "high": "neg", "critical": "neg", "medium": "warn", "low": "neu",
    "lonjakan": "neg", "meningkat": "warn", "stabil": "pos", "mereda": "neu",
}


def section_head(title: str, desc: str = "") -> None:
    d = f"<p>{esc(desc)}</p>" if desc else ""
    html_block(f'<div class="sec"><h2>{esc(title)}</h2>{d}</div>')


def sub_head(title: str) -> None:
    html_block(f'<div class="sub">{esc(title)}</div>')


def callout(kind: str, message: str) -> None:
    html_block(f'<div class="callout {kind}">{esc(message)}</div>')


def kpi_grid(items: list) -> None:
    cards = []
    for it in items:
        cls = "kpi-value txt" if it.get("text") else "kpi-value"
        cards.append(
            f'<div class="kpi tone-{it.get("tone", "brand")}">'
            f'<div class="kpi-label">{esc(str(it["label"]))}</div>'
            f'<div class="{cls}">{esc(str(it["value"]))}</div>'
            f'<div class="kpi-hint">{esc(str(it.get("hint", "")))}</div></div>'
        )
    html_block(f'<div class="kpi-grid">{"".join(cards)}</div>')


def _fmt_cell(col: str, val, badge, num, trunc: dict) -> str:
    if val is None or (not isinstance(val, (list, dict)) and pd.isna(val)):
        return '<td class="nw">-</td>'
    if isinstance(val, pd.Timestamp):
        return f'<td class="nw">{esc(val.strftime("%d-%m-%Y %H:%M"))}</td>'
    if col in badge:
        key = str(val).strip().lower()
        label = {"sarcastic": "Sarkastik", "not_sarcastic": "Tidak"}.get(key, str(val))
        return f'<td class="nw"><span class="pill {PILL_CLASS.get(key, "neu")}">{esc(label)}</span></td>'
    if col == "url":
        u = str(val)
        if u.startswith(("http://", "https://")):
            return f'<td class="nw"><a href="{esc(u, quote=True)}" target="_blank" rel="noopener noreferrer">Buka</a></td>'
        return '<td class="nw">-</td>'
    if col in num:
        try:
            return f'<td class="num">{int(val):,}</td>'
        except (ValueError, TypeError):
            pass
    if isinstance(val, float):
        return f'<td class="num">{val:.2f}</td>'
    s = str(val)
    limit = trunc.get(col)
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + "…"
    cls = "wrap" if limit else "nw"
    return f'<td class="{cls}">{esc(s)}</td>'


def build_table_html(df: pd.DataFrame, badge=(), num=(), trunc=None, max_height: int = 420) -> str:
    trunc = trunc or {}
    cols = list(df.columns)
    head = "".join(f"<th>{esc(COL_LABELS.get(c, str(c)))}</th>" for c in cols)
    rows = [
        "<tr>" + "".join(_fmt_cell(c, rec[c], badge, num, trunc) for c in cols) + "</tr>"
        for rec in df.to_dict("records")
    ]
    return H(
        f'<div class="tbl-wrap" style="max-height:{max_height}px"><table class="tbl">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_table(df: pd.DataFrame, badge=(), num=(), trunc=None, max_height: int = 420) -> None:
    st.markdown(build_table_html(df, badge=badge, num=num, trunc=trunc, max_height=max_height), unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str, height: int = 360, legend_top: bool = True) -> go.Figure:
    fig.update_layout(
        autosize=True,
        template=T["plotly_template"],
        title=dict(text=title, x=0.015, xanchor="left", font=dict(size=14, color=T["text"], family=FONT_SANS)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, color=T["text_2"], size=11),
        margin=dict(l=10, r=14, t=68, b=14),
        height=height,
        colorway=PALETTE,
        hoverlabel=dict(bgcolor=T["surface"], font=dict(color=T["text"]), bordercolor=T["border_strong"]),
        legend=dict(
            title_text="",
            font=dict(color=T["text_2"], size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02 if legend_top else -0.22,
            xanchor="right" if legend_top else "center",
            x=0.99 if legend_top else 0.5,
            itemwidth=30,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    axis = dict(
        automargin=True,
        gridcolor=T["grid"],
        linecolor=T["border_strong"],
        zerolinecolor=T["grid"],
        tickfont=dict(color=T["text_2"]),
        title_font=dict(color=T["text_2"]),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show_fig(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **PLOTLY_KW)


def build_trend(df: pd.DataFrame, freq: str, metric: str, by=None) -> pd.DataFrame:
    cols = ["timestamp"] + ([by] if by else []) + ([] if metric == "Volume Postingan" else ["engagement"])
    d = df[cols].copy()
    d["bucket"] = d["timestamp"].dt.floor(freq)
    keys = ["bucket"] + ([by] if by else [])
    agg = (d.groupby(keys, observed=True).size() if metric == "Volume Postingan" else d.groupby(keys, observed=True)["engagement"].sum()).rename("nilai")
    full = pd.date_range(d["bucket"].min(), d["bucket"].max(), freq=freq)
    if 0 < len(full) <= 4000:
        idx = (
            pd.MultiIndex.from_product([full, sorted(d[by].unique())], names=keys)
            if by else pd.Index(full, name="bucket")
        )
        agg = agg.reindex(idx, fill_value=0)
    return agg.reset_index()


@st.cache_data(show_spinner=False, max_entries=6)
def compute_trends_metrics(_sig: tuple, _tdf: pd.DataFrame) -> pd.DataFrame:
    """Hitung metrik puncak, rata-rata, momentum terbaru, dan z-score lonjakan per kata kunci."""
    rows = []
    for kw, grp in _tdf.groupby("keyword", observed=True):
        grp = grp.sort_values("timestamp")
        scores = grp["interest_score"].to_numpy(dtype=float)
        latest = int(scores[-1])
        peak_idx = int(np.argmax(scores))
        peak_val = int(scores[peak_idx])
        peak_date = grp["timestamp"].iloc[peak_idx].strftime("%d-%m-%Y")
        mean_val = float(np.mean(scores))
        std_val = float(np.std(scores))
        z_score = (latest - mean_val) / std_val if std_val > 1e-6 else 0.0

        window = max(1, min(3, len(scores) // 3))
        recent_mean = float(np.mean(scores[-window:]))
        prev_slice = scores[:-window] if len(scores) > window else scores
        prev_mean = float(np.mean(prev_slice))
        if prev_mean > 0:
            momentum_pct = (recent_mean - prev_mean) / prev_mean * 100.0
        else:
            momentum_pct = 100.0 if recent_mean > 0 else 0.0

        if z_score >= 1.5 and latest >= 50:
            status_tren = "Lonjakan"
        elif momentum_pct >= 15.0:
            status_tren = "Meningkat"
        elif momentum_pct <= -15.0:
            status_tren = "Mereda"
        else:
            status_tren = "Stabil"

        rows.append({
            "keyword": str(kw),
            "skor_terbaru": latest,
            "rata_rata": round(mean_val, 1),
            "puncak": peak_val,
            "tanggal_puncak": peak_date,
            "momentum": round(momentum_pct, 1),
            "z_score": round(z_score, 2),
            "status_tren": status_tren,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["skor_terbaru", "puncak"], ascending=False).reset_index(drop=True)


ID_STOPWORDS = set(STOPWORDS) | {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "pada", "adalah", "akan",
    "juga", "atau", "tidak", "ada", "dalam", "oleh", "sebagai", "karena", "bisa", "sudah", "saya",
    "kita", "kami", "mereka", "kata", "https", "http", "www", "com", "co", "id", "rt", "amp",
}


def _collect_sentiment_text(df: pd.DataFrame, sentiment: str, max_chars: int = 300_000) -> str:
    """Akumulasi teks hingga batas karakter tanpa menggabungkan seluruh korpus."""
    sub = df.loc[df["sentiment"] == sentiment, "cleaned_text"]
    chunks, total = [], 0
    for val in sub:
        s = str(val)
        if not s:
            continue
        chunks.append(s)
        total += len(s) + 1
        if total >= max_chars:
            break
    return " ".join(chunks)[:max_chars]


@st.cache_data(show_spinner=False, max_entries=8)
def get_wordcloud_png(_filter_sig: tuple, sentiment: str, colors: tuple, _df: pd.DataFrame) -> bytes | None:
    text_blob = _collect_sentiment_text(_df, sentiment)
    if not text_blob.strip():
        return None
    rng = random.Random(7)

    def color_func(*_a, **_k):
        return rng.choice(colors)

    wc = WordCloud(
        width=900, height=420, mode="RGBA", background_color=None, max_words=120,
        stopwords=ID_STOPWORDS, color_func=color_func, collocations=False,
        prefer_horizontal=0.9, min_font_size=9, random_state=7,
    ).generate(text_blob)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=4)
def compute_disarm_analysis(_filter_sig: tuple, _df: pd.DataFrame, anomaly_alerts: list) -> dict:
    """Hitung metrik CIB, skor risiko DISARM, bukti duplikasi, dan tabel eksplorasi sekali per filter."""
    txt_df = _df[_df["text"].str.strip() != ""]
    dup_series = txt_df.groupby("text")["author"].unique()
    dup_series = dup_series[dup_series.str.len() > 1]
    duplicated_texts = dup_series.index.tolist()

    clustered_df = _df[_df["narrative_cluster_id"] >= 0]
    cluster_authors = clustered_df.groupby("narrative_cluster_id")["author"].nunique()
    coordinated_clusters = cluster_authors[cluster_authors > 1].index.tolist()

    cib_df = _df[
        _df["text"].isin(duplicated_texts) | _df["narrative_cluster_id"].isin(coordinated_clusters)
    ]
    suspect_count = int(cib_df["author"].nunique())
    bot_post_count = len(cib_df)
    cib_ratio = bot_post_count / len(_df)

    neg_ratio = (_df["sentiment"] == "Negatif").sum() / len(_df)
    total_risk_score = min(10.0, round(
        neg_ratio * 4.0 + min(3.5, cib_ratio * 5.0) + min(2.5, len(anomaly_alerts) * 0.4), 1
    ))

    if total_risk_score >= 7.0:
        threat_class, status_text = "high", "[KRITIS] ANCAMAN TINGGI"
    elif total_risk_score >= 4.0:
        threat_class, status_text = "medium", "[WASPADA] ANCAMAN SEDANG"
    else:
        threat_class, status_text = "low", "[NORMAL] ANCAMAN RENDAH"

    ttps = []
    if duplicated_texts or coordinated_clusters:
        ttps.append(("TA08 · T0015", "Develop Content: Inauthentic Text or Narrative Duplication"))
    if cib_ratio > 0.10 or anomaly_alerts:
        ttps.append(("TA11 · T0049", "Maximize Exposure: Information Flooding or Volume Spike"))
    if total_risk_score >= 6.0:
        ttps.append(("TA12 · T0050", "Microtargeting: Coordinated Amplification"))

    suspect_table_html = ""
    if suspect_count:
        suspect_summary = (
            cib_df.groupby(["author", "platform_detail"], observed=True)
            .agg(
                jumlah_post=("text", "count"),
                klaster_terlibat=("narrative_cluster_id", "nunique"),
                total_engagement=("engagement", "sum"),
            )
            .reset_index()
            .sort_values(by="jumlah_post", ascending=False)
            .head(50)
        )
        suspect_table_html = build_table_html(
            suspect_summary, num=("jumlah_post", "klaster_terlibat", "total_engagement"), max_height=380
        )

    evidence_items = []
    for text_item, authors in sorted(dup_series.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]:
        shown = ", ".join(map(str, authors[:12])) + (" …" if len(authors) > 12 else "")
        evidence_items.append((text_item[:300], len(authors), shown))

    explorer_cols = [
        "timestamp", "platform", "platform_detail", "author", "topic",
        "text", "sentiment", "sarcasm", "narrative_cluster_id", "engagement", "url",
    ]
    explorer_html = build_table_html(
        _df.sort_values("timestamp", ascending=False)[explorer_cols].head(300),
        badge=("sentiment", "sarcasm"),
        num=("engagement",),
        trunc={"text": 220},
        max_height=520,
    )

    return {
        "total_risk_score": total_risk_score,
        "threat_class": threat_class,
        "status_text": status_text,
        "ttps": ttps,
        "suspect_count": suspect_count,
        "bot_post_count": bot_post_count,
        "cib_ratio": cib_ratio,
        "suspect_table_html": suspect_table_html,
        "evidence_items": evidence_items,
        "explorer_html": explorer_html,
    }


# ---------------------------------------------------------
# 5. MUAT DATA & HEADER
# ---------------------------------------------------------
data_sig = tuple(
    _file_signature(DATA_DIR / name) for name in (
        "master_osint_enriched.csv",
        "master_osint_data.csv",
        "cyberlens_report.json",
    )
)
trends_sig = _file_signature(DATA_DIR / "trends_interest.csv")

pipeline_stage = st.session_state.get("osint_stage", "idle")
analysis_failed = bool(st.session_state.get("osint_analysis_failed", False))

if pipeline_stage in ("scraping", "analyzing"):
    payload = {
        "df": _empty_standardized_df(),
        "report": {},
        "status": "processing",
        "error": None,
        "warnings": [],
    }
elif analysis_failed:
    payload = {
        "df": _empty_standardized_df(),
        "report": {},
        "status": "analysis_error",
        "error": st.session_state.get("osint_error_message") or "Analisis NLP belum berhasil.",
        "warnings": [],
    }
else:
    with st.spinner("Memuat hasil scraping yang sudah dianalisis NLP..."):
        payload = load_data(data_sig)

trends_payload = load_trends_data(trends_sig)
trends_df = trends_payload["df"]
trends_status = trends_payload["status"]

df = payload["df"]
cyberlens_report = payload["report"]
load_status = payload["status"]

STATUS_DOT = {
    "ok": ("", "Data NLP siap"),
    "empty_file": ("warn", "Hasil NLP kosong"),
    "missing": ("warn", "Dataset tidak ditemukan"),
    "pending_analysis": ("warn", "Menunggu analisis NLP"),
    "processing": ("warn", "Scraping / NLP berjalan"),
    "analysis_error": ("bad", "Analisis NLP gagal"),
    "error": ("bad", "Gagal memuat"),
}
dot_cls, dot_label = STATUS_DOT.get(load_status, ("warn", load_status))
range_chip = ""
if not df.empty:
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    range_chip = f'<span class="chip">Rentang: {esc(t0.strftime("%d-%m-%Y"))} s.d. {esc(t1.strftime("%d-%m-%Y"))} (UTC)</span>'
trends_chip = (
    f'<span class="chip">Google Trends: {len(trends_df):,} titik</span>'
    if not trends_df.empty
    else ""
)

html_block(f"""
<div class="hero">
<span class="badge">PUSSIBERAD TNI AD</span><span class="badge">INTELIJEN SIBER SOSIAL</span>
<h1>KOKRINGAN Command Center</h1>
<p>Komando Kognitif: Ringkasan Aktivitas Narasi, pemantauan sentimen, klaster narasi, dan indikasi operasi informasi lintas platform.</p>
<span class="chip"><i class="dot {dot_cls}"></i>{esc(dot_label)}</span>
<span class="chip">{len(df):,} rekaman</span>{trends_chip}{range_chip}
</div>
""")

for warn_msg in payload["warnings"]:
    callout("warn", warn_msg)
if load_status in ("error", "analysis_error"):
    callout("bad", payload["error"])

# ---------------------------------------------------------
# 6. FILTER SIDEBAR
# ---------------------------------------------------------
all_platforms = sorted(df["platform"].cat.categories.tolist()) if not df.empty else []
all_sentiments = ["Positif", "Negatif", "Netral"]
all_topics = sorted(df["topic"].cat.categories.tolist()) if not df.empty else []

filter_options = (
    ("filter_platforms", all_platforms),
    ("filter_sentiments", all_sentiments),
    ("filter_topics", all_topics),
)

if not df.empty:
    if st.session_state.get("_filter_data_sig") != data_sig:
        for key, options in filter_options:
            st.session_state[key] = list(options)
        st.session_state["_filter_data_sig"] = data_sig
    else:
        for key, options in filter_options:
            if key not in st.session_state:
                st.session_state[key] = list(options)
            else:
                st.session_state[key] = [v for v in st.session_state[key] if v in options]
else:
    for key, options in filter_options:
        st.session_state.setdefault(key, list(options))

st.session_state.setdefault("filter_sarcasm", False)

with st.sidebar:
    html_block('<div class="side-title">Parameter Pemantauan</div>')
    selected_platforms = st.multiselect("Sumber Platform:", options=all_platforms, key="filter_platforms")
    selected_sentiments = st.multiselect("Kategori Sentimen:", options=all_sentiments, key="filter_sentiments")
    selected_topics = st.multiselect("Filter Topik / Kata Kunci:", options=all_topics, key="filter_topics")
    sarcasm_only = st.checkbox("Hanya indikasi sarkasme atau perlu tinjauan", key="filter_sarcasm")
    st.button(
        "Reset Semua Filter",
        on_click=_reset_filters,
        args=(all_platforms, all_sentiments, all_topics),
        **BUTTON_KW,
    )

if not df.empty:
    mask = (
        df["platform"].isin(selected_platforms)
        & df["sentiment"].isin(selected_sentiments)
        & df["topic"].isin(selected_topics)
    )
    if sarcasm_only:
        mask = mask & (df["is_sarcastic"] | df["needs_review"])
    filtered_df = df.loc[mask]
else:
    filtered_df = _empty_standardized_df()

filter_sig = (
    data_sig,
    tuple(selected_platforms),
    tuple(selected_sentiments),
    tuple(selected_topics),
    bool(sarcasm_only),
)

with st.sidebar:
    if not filtered_df.empty:
        st.download_button(
            "Unduh data terfilter (CSV)",
            data=export_filtered_csv(filter_sig, filtered_df),
            file_name="kokringan_data_terfilter.csv",
            mime="text/csv",
            **DOWNLOAD_KW,
        )
    html_block(f"""
    <div class="side-title">Ringkasan Dataset</div>
    <div class="side-note"><b>{len(filtered_df):,}</b> dari <b>{len(df):,}</b> rekaman ditampilkan<br>
    Google Trends: <b>{len(trends_df):,}</b> titik<br>
    Status: <b>{esc(load_status.upper())}</b><br>Unit: PUSSIBERAD TNI AD</div>
    """)

if render_ingestion_panel is not None:
    render_ingestion_panel()


def _render_empty_state(context_name: str) -> None:
    if load_status == "processing":
        message = (
            "Scraping atau analisis NLP sedang berlangsung. Dashboard akan "
            "menampilkan data setelah seluruh analisis selesai."
        )
    elif load_status == "pending_analysis":
        message = (
            "Dataset scraping tersedia, tetapi hasil NLP terbaru belum lengkap. "
            "Data mentah tidak ditampilkan."
        )
    elif load_status == "analysis_error":
        message = (
            "Analisis NLP gagal. Dashboard tidak menampilkan data mentah agar "
            "indikator tidak menggunakan hasil yang belum dianalisis. Periksa latest_run.log."
        )
    elif load_status == "missing":
        message = (
            "Dataset hasil analisis belum tersedia. Jalankan scraping melalui "
            "panel OSINT Ingestion & Pipeline; NLP akan dijalankan setelah scraping berhasil."
        )
    elif load_status == "empty_file":
        message = "Hasil analisis NLP tersedia, tetapi tidak berisi rekaman yang dapat ditampilkan."
    else:
        message = (
            f"Tidak ada data yang cocok pada tampilan {context_name}. "
            "Sesuaikan filter platform, sentimen, atau topik."
        )
    html_block(f'<div class="empty"><strong>{esc(message)}</strong></div>')


# ---------------------------------------------------------
# 7. EMPAT TAB UTAMA
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Ringkasan Eksekutif",
    "02 · Semantik dan Kluster Narasi",
    "03 · Deteksi Anomali dan DISARM",
    "04 · Google Trends & Minat Publik",
])

# =========================================================
# TAB 1: RINGKASAN EKSEKUTIF
# =========================================================
with tab1:
    section_head("Ringkasan Intelijen Siber Lintas Platform", "Indikator utama berdasarkan filter yang aktif.")

    if filtered_df.empty:
        _render_empty_state("Ringkasan Eksekutif")
    else:
        total_posts = len(filtered_df)
        total_engagement = int(filtered_df["engagement"].sum())
        neg_count = int((filtered_df["sentiment"] == "Negatif").sum())
        neg_pct = neg_count / total_posts * 100
        sarcastic_count = int(filtered_df["is_sarcastic"].sum())
        author_counts = filtered_df["author"].value_counts()
        top_author, top_author_n = author_counts.index[0], int(author_counts.iloc[0])
        present_platforms = set(filtered_df["platform"].unique())
        active_platforms = [p for p in all_platforms if p in present_platforms]

        kpi_grid([
            {"label": "Total Unggahan", "value": f"{total_posts:,}", "hint": f"{len(active_platforms)} platform · {filtered_df['topic'].nunique()} topik"},
            {"label": "Total Engagement", "value": f"{total_engagement:,}", "hint": f"Rata-rata {total_engagement / total_posts:,.1f} per unggahan"},
            {"label": "Sentimen Negatif", "value": f"{neg_pct:.1f}%", "hint": f"{neg_count:,} unggahan negatif", "tone": "neg"},
            {"label": "Indikasi Sarkasme", "value": f"{sarcastic_count:,}", "hint": f"{sarcastic_count / total_posts * 100:.1f}% dari total", "tone": "warn"},
            {"label": "Aktor Paling Aktif", "value": top_author, "hint": f"{top_author_n:,} unggahan", "tone": "neu", "text": True},
        ])

        sub_head("Kontrol Agregasi Tren Waktu")
        ctrl_c1, ctrl_c2 = st.columns(2)
        with ctrl_c1:
            trend_metric = st.radio("Metrik Grafik Tren:", ["Volume Postingan", "Total Engagement"], horizontal=True)
        with ctrl_c2:
            freq_options = {"Per 3 Jam": "3h", "Per 6 Jam": "6h", "Harian (1D)": "1D"}
            resample_freq = freq_options[st.selectbox("Interval Waktu:", list(freq_options), index=1)]

        sub_tabs = st.tabs(["Gabungan Lintas Platform"] + active_platforms)

        with sub_tabs[0]:
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                agg_ts = build_trend(filtered_df, resample_freq, trend_metric, by="platform")
                y_title = "Jumlah Postingan" if trend_metric == "Volume Postingan" else "Total Engagement"
                fig_all = px.line(
                    agg_ts, x="bucket", y="nilai", color="platform",
                    markers=agg_ts["bucket"].nunique() <= 60,
                    labels={"bucket": "Waktu (UTC)", "nilai": y_title, "platform": "Platform"},
                )
                show_fig(style_fig(fig_all, f"Tren {trend_metric} Lintas Platform"), "trend_all")
            with col_g2:
                sent_bar_df = filtered_df.groupby(["platform", "sentiment"], observed=True).size().reset_index(name="jumlah")
                fig_bar = px.bar(
                    sent_bar_df, x="platform", y="jumlah", color="sentiment", barmode="group",
                    labels={"platform": "Platform", "jumlah": "Jumlah", "sentiment": "Sentimen"},
                    color_discrete_map=SENT_COLORS,
                    category_orders={"sentiment": ["Positif", "Netral", "Negatif"]},
                )
                show_fig(style_fig(fig_bar, "Distribusi Sentimen per Platform"), "bar_sent")

            col_h1, col_h2 = st.columns([1, 2])
            with col_h1:
                sent_tot = filtered_df["sentiment"].value_counts().reindex(["Positif", "Netral", "Negatif"]).fillna(0).reset_index()
                sent_tot.columns = ["sentiment", "jumlah"]
                fig_pie = px.pie(sent_tot, names="sentiment", values="jumlah", hole=0.64, color="sentiment", color_discrete_map=SENT_COLORS)
                fig_pie.update_traces(textinfo="none", marker=dict(line=dict(color=T["surface"], width=2)))
                fig_pie.add_annotation(text=f"<b>{total_posts:,}</b><br>unggahan", showarrow=False, font=dict(size=14, color=T["text"]))
                show_fig(style_fig(fig_pie, "Komposisi Sentimen", legend_top=False), "pie_sent")
            with col_h2:
                top_topics = filtered_df["topic"].value_counts().head(10).iloc[::-1].reset_index()
                top_topics.columns = ["topic", "jumlah"]
                fig_topic = px.bar(top_topics, x="jumlah", y="topic", orientation="h", labels={"topic": "Topik", "jumlah": "Jumlah"})
                fig_topic.update_traces(marker_color=T["brand"])
                show_fig(style_fig(fig_topic, "10 Topik Teratas", legend_top=False), "bar_topics")

        for idx, plat in enumerate(active_platforms):
            with sub_tabs[idx + 1]:
                plat_df = filtered_df.loc[filtered_df["platform"] == plat]
                p_total = len(plat_df)
                p_eng = int(plat_df["engagement"].sum())
                p_neg = (plat_df["sentiment"] == "Negatif").sum() / p_total * 100
                p_counts = plat_df["author"].value_counts()

                kpi_grid([
                    {"label": f"Unggahan ({plat})", "value": f"{p_total:,}", "hint": f"{p_total / total_posts * 100:.1f}% dari total"},
                    {"label": "Engagement", "value": f"{p_eng:,}", "hint": f"{p_eng / total_engagement * 100:.1f}% dari total" if total_engagement else "-"},
                    {"label": "Sentimen Negatif", "value": f"{p_neg:.1f}%", "tone": "neg", "hint": "pada platform ini"},
                    {"label": "Aktor Teratas", "value": p_counts.index[0], "tone": "neu", "text": True, "hint": f"{int(p_counts.iloc[0]):,} unggahan"},
                ])

                use_volume = trend_metric == "Volume Postingan" or p_eng == 0
                plat_label = "Jumlah Postingan" if use_volume else "Total Engagement"
                plat_ts = build_trend(plat_df, resample_freq, "Volume Postingan" if use_volume else "Total Engagement")
                fig_plat = px.area(plat_ts, x="bucket", y="nilai", labels={"bucket": "Waktu (UTC)", "nilai": plat_label})
                fig_plat.update_traces(line=dict(color=T["brand"], width=2.4), fillcolor=T["brand_fill"])
                show_fig(style_fig(fig_plat, f"Dinamika Aktivitas {plat} ({plat_label})"), f"plat_{idx}")

                sub_head(f"5 Konten Teratas di {plat}")
                sort_col = "engagement" if p_eng > 0 else "timestamp"
                render_table(
                    plat_df.sort_values(by=sort_col, ascending=False)[
                        ["timestamp", "platform_detail", "author", "topic", "text", "sentiment", "sarcasm", "engagement"]
                    ].head(5),
                    badge=("sentiment", "sarcasm"),
                    num=("engagement",),
                    trunc={"text": 220},
                    max_height=380,
                )

# =========================================================
# TAB 2: SEMANTIK DAN KLASTER NARASI
# =========================================================
with tab2:
    section_head("Pemetaan Kluster Narasi, Sarkasme, dan Ringkasan Intelijen", "Hasil analisis semantik otomatis berbasis kecerdasan artifisial (AI) buatan Indonesia, yaitu IndoBERT.")

    intel = cyberlens_report.get("intelligence_summary", {})
    if intel:
        sub_head("Ringkasan Otomatis CYBERLENS")
        html_block(f'<div class="card">{esc(str(intel.get("summary", "")))}</div>')
        findings, recs = intel.get("key_findings", []), intel.get("recommendations", [])

        def _list_card(title: str, items: list, empty: str) -> str:
            body = "".join(f"<li>{esc(str(i))}</li>" for i in items) if items else f"<li>{esc(empty)}</li>"
            return f'<div class="card"><h4>{esc(title)}</h4><ul>{body}</ul></div>'

        kf_col, rec_col = st.columns(2)
        with kf_col:
            html_block(_list_card("Temuan Utama", findings, "Belum ada temuan utama tercatat."))
        with rec_col:
            html_block(_list_card("Rekomendasi Tindakan", recs, "Belum ada rekomendasi tercatat."))

    if filtered_df.empty:
        _render_empty_state("Semantik dan Kluster Narasi")
    else:
        col_w1, col_w2 = st.columns(2)
        for col, sentiment, title, colors, empty_msg in (
            (col_w1, "Positif", "WordCloud Narasi Positif (Dukungan dan Stabilitas)", T["wc_pos"], "Tidak ditemukan teks berlabel sentimen Positif pada filter aktif."),
            (col_w2, "Negatif", "WordCloud Narasi Negatif (Potensi Propaganda atau Hoaks)", T["wc_neg"], "Tidak ditemukan teks berlabel sentimen Negatif pada filter aktif."),
        ):
            with col:
                sub_head(title)
                png_bytes = get_wordcloud_png(filter_sig, sentiment, tuple(colors), filtered_df)
                if png_bytes:
                    st.image(png_bytes, caption=title, use_container_width=True)
                else:
                    callout("info", empty_msg)

        narratives = cyberlens_report.get("narratives", [])
        if narratives:
            sub_head(f"Klaster Narasi Dominan ({len(narratives)} Klaster Semantik)")
            nar_df = pd.DataFrame([
                {
                    "Cluster ID": n.get("cluster_id"),
                    "Perwakilan Narasi": str(n.get("narrative_title", "")),
                    "Jumlah Post": n.get("post_count", 0),
                    "Platform Terlibat": ", ".join(n.get("platforms", [])),
                }
                for n in sorted(narratives, key=lambda x: x.get("post_count", 0), reverse=True)
            ])
            render_table(nar_df, num=("Jumlah Post",), trunc={"Perwakilan Narasi": 160}, max_height=420)

        sarcastic_df = filtered_df.loc[filtered_df["is_sarcastic"]]
        if not sarcastic_df.empty:
            sub_head(f"Sampel Unggahan Terindikasi Sarkasme ({len(sarcastic_df):,} Unggahan)")
            render_table(
                sarcastic_df[["timestamp", "platform_detail", "author", "text", "sentiment", "sarcasm_confidence", "interpretation"]].head(15),
                badge=("sentiment",),
                trunc={"text": 240},
                max_height=460,
            )

# =========================================================
# TAB 3: DETEKSI ANOMALI, CIB, DAN DISARM
# =========================================================
with tab3:
    section_head("Penilaian Risiko DISARM, Anomali, dan Aktivitas Terkoordinasi (CIB)", "Skor risiko gabungan dari sentimen negatif, CIB, dan alert anomali.")

    if filtered_df.empty:
        _render_empty_state("Deteksi Anomali dan DISARM")
    else:
        anomaly_alerts = cyberlens_report.get("anomalies", {}).get("alerts", [])
        disarm = compute_disarm_analysis(filter_sig, filtered_df, anomaly_alerts)

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            html_block(f"""
            <div class="threat {disarm['threat_class']}"><h4>Indeks Ancaman Kognitif</h4>
            <div class="score">{disarm['total_risk_score']}<small> / 10.0</small></div>
            <div class="meter"><i style="width:{disarm['total_risk_score'] * 10:.0f}%"></i></div>
            <p class="status">STATUS: {esc(disarm['status_text'])}</p></div>
            """)

            sub_head("Matriks TTPs DISARM Teridentifikasi")
            if disarm["ttps"]:
                html_block("".join(
                    f'<div class="ttp"><span class="pill neg">AKTIF</span><div><b>{esc(code)}</b><span class="d">{esc(desc)}</span></div></div>'
                    for code, desc in disarm["ttps"]
                ))
            else:
                html_block('<div class="ttp"><span class="pill pos">NORMAL</span><div><span class="d">Tidak ditemukan indikator TTPs DISARM di atas ambang batas.</span></div></div>')

        with col_r2:
            sub_head("Indikator Aktivitas Terkoordinasi (CIB) dan Anomali")
            kpi_grid([
                {"label": "Akun Suspek CIB", "value": f"{disarm['suspect_count']:,}", "tone": "warn"},
                {"label": "Post Terkoordinasi", "value": f"{disarm['bot_post_count']:,}", "tone": "warn"},
                {"label": "Rasio CIB", "value": f"{disarm['cib_ratio'] * 100:.1f}%", "tone": "neg"},
                {"label": "Alert Anomali", "value": f"{len(anomaly_alerts):,}", "tone": "neg"},
            ])
            sub_head("Peringkat Akun Terindikasi Penggerak Narasi")
            if disarm["suspect_table_html"]:
                st.markdown(disarm["suspect_table_html"], unsafe_allow_html=True)
            else:
                callout("info", "Tidak terdeteksi jaringan akun terkoordinasi pada kombinasi filter aktif.")

        if anomaly_alerts:
            sub_head("Peringatan Anomali Volume, Lintas Platform, dan Sentimen")
            anom_df = pd.DataFrame([
                {
                    "Waktu": a.get("time_bucket") or a.get("detected_at"),
                    "Tipe Anomali": a.get("anomaly_type"),
                    "Severity": str(a.get("severity", "")).lower(),
                    "Platform": a.get("platform") or ", ".join(a.get("affected_platforms", [])),
                    "Deskripsi": a.get("description"),
                }
                for a in anomaly_alerts
            ])
            render_table(anom_df, badge=("Severity",), trunc={"Deskripsi": 220}, max_height=420)

        sub_head("Bukti Duplikasi Teks Identik Lintas Akun")
        if disarm["evidence_items"]:
            for snippet, author_cnt, shown_authors in disarm["evidence_items"]:
                html_block(
                    f'<div class="evidence"><div class="q">“{esc(snippet)}”</div>'
                    f'<div class="a"><b>Aktor penyebar ({author_cnt} akun):</b> {esc(shown_authors)}</div></div>'
                )
        else:
            callout("ok", "Tidak ditemukan duplikasi teks identik lintas akun pada filter aktif.")

        sub_head("Tabel Eksplorasi Data OSINT dan Hasil NLP")
        st.markdown(disarm["explorer_html"], unsafe_allow_html=True)
        st.caption(f"Menampilkan 300 rekaman terbaru dari {len(filtered_df):,}. Gunakan tombol unduh CSV di panel kiri untuk data lengkap.")

# =========================================================
# TAB 4: GOOGLE TRENDS & MINAT PUBLIK
# =========================================================
with tab4:
    section_head(
        "Pemantauan Indeks Minat Pencarian Publik (Google Trends)",
        "Skor indeks relatif (0–100) dari data/trends_interest.csv dan perbandingan terhadap dinamika percakapan media sosial.",
    )

    default_gt_kws = [
      "IHSG",
      "kurs rupiah",
      "PPN 12 persen",
      "PHK",
      "pemerintah",
      "presiden",
      "boikot",
      "polisi/polri",
      "tentara/tni",
      "judol",
      "pinjol",
      "kebocoran data/peretasan",
      "PDNS",
      "KPK",
      "korupsi/kolusi/gratifikasi",
      "RUU perampasan aset",
      "DPR/DPRD",
      "demo mahasiswa/aksi mahasiswa/tuntutan mahasiswa",
      "demo buruh/aksi buruh/tuntutan buruh",
      "aksi ojol/keluhan ojol",
      "umr/ump"
    ]
    for cfg_candidate in (ROOT_DIR / "config" / "targets.json", ROOT_DIR / "targets.json"):
        if cfg_candidate.exists():
            try:
                cfg_data = json.loads(cfg_candidate.read_text(encoding="utf-8"))
                cfg_kws = cfg_data.get("google_trends", {}).get("keywords", [])
                if cfg_kws:
                    default_gt_kws = cfg_kws
            except Exception:
                pass
            break

    with st.expander("Tarik / Perbarui Data Google Trends", expanded=trends_df.empty):
        with st.form("quick_fetch_google_trends"):
            f_c1, f_c2 = st.columns([3, 1])
            with f_c1:
                kw_input = st.text_input(
                    "Kata Kunci Google Trends (pisahkan dengan koma):",
                    value=", ".join(default_gt_kws),
                )
            with f_c2:
                tf_choice = st.selectbox(
                    "Rentang Waktu:",
                    options=["today 3-m", "today 1-m", "now 7-d"],
                    format_func=lambda x: {
                        "now 7-d": "7 Hari Terakhir",
                        "today 1-m": "1 Bulan Terakhir",
                        "today 3-m": "3 Bulan Terakhir",
                    }[x],
                )
            fetch_now = st.form_submit_button("Ambil & Perbarui Data Google Trends", type="primary")

        if fetch_now:
            kws = [k.strip() for k in kw_input.split(",") if k.strip()]
            if not kws:
                st.warning("Masukkan minimal 1 kata kunci.")
            else:
                with st.spinner("Mengambil indeks minat pencarian dari Google Trends..."):
                    try:
                        try:
                            from ingestion.scrape import scrape_google_trends_series, save_trends_csv
                        except ImportError:
                            from scrape import scrape_google_trends_series, save_trends_csv

                        records = scrape_google_trends_series(kws, timeframe=tf_choice, geo="ID")
                        if not records:
                            st.error("Google Trends mengembalikan data kosong (kena rate-limit HTTP 429). Coba lagi beberapa saat.")
                        else:
                            DATA_DIR.mkdir(parents=True, exist_ok=True)
                            saved_rows = save_trends_csv(records, str(DATA_DIR / "trends_interest.csv"), replace=True)
                            st.cache_data.clear()
                            st.success(f"Berhasil menyimpan {saved_rows:,} titik data Google Trends.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Gagal mengambil data Google Trends: {exc}")

    if trends_status == "error":
        callout("bad", trends_payload["error"])
    elif trends_df.empty:
        callout("info", "File data/trends_interest.csv masih kosong (0 titik). Gunakan panel di atas untuk menarik data.")
    else:
        geo_options = sorted(trends_df["geo"].unique().tolist())
        kw_options = sorted(trends_df["keyword"].unique().tolist())

        gt_f1, gt_f2, gt_f3 = st.columns([1, 2, 1])
        with gt_f1:
            selected_geos = st.multiselect(
                "Wilayah (Geo):",
                options=geo_options,
                default=geo_options,
                key="gt_filter_geo",
            )
        with gt_f2:
            selected_kws = st.multiselect(
                "Kata Kunci Google Trends:",
                options=kw_options,
                default=kw_options,
                key="gt_filter_kw",
            )
        with gt_f3:
            smooth_ma = st.checkbox(
                "Haluskan kurva (MA-3)",
                value=False,
                key="gt_smooth_ma",
                help="Menampilkan rata-rata bergerak 3 titik waktu untuk meredam fluktuasi harian.",
            )

        gt_view = trends_df[
            trends_df["geo"].isin(selected_geos) & trends_df["keyword"].isin(selected_kws)
        ].copy()

        if gt_view.empty:
            html_block(
                '<div class="empty"><strong>Tidak ada data Google Trends untuk kombinasi wilayah dan kata kunci yang dipilih.</strong></div>'
            )
        else:
            gt_sub_sig = (trends_sig, tuple(selected_geos), tuple(selected_kws))
            metrics_df = compute_trends_metrics(gt_sub_sig, gt_view)

            top_kw_row = metrics_df.sort_values(by=["puncak", "skor_terbaru"], ascending=False).iloc[0]
            top_mom_row = metrics_df.sort_values(by="momentum", ascending=False).iloc[0]
            avg_interest = float(gt_view["interest_score"].mean())
            gt_t0, gt_t1 = gt_view["timestamp"].min(), gt_view["timestamp"].max()

            kpi_grid([
                {
                    "label": "Kata Kunci Puncak",
                    "value": str(top_kw_row["keyword"]),
                    "hint": f"Skor {int(top_kw_row['puncak'])}/100 pada {top_kw_row['tanggal_puncak']}",
                    "text": True,
                },
                {
                    "label": "Momentum Tertinggi",
                    "value": f"{top_mom_row['momentum']:+.1f}%",
                    "hint": f"{top_mom_row['keyword']} (skor terbaru {int(top_mom_row['skor_terbaru'])})",
                    "tone": "warn" if top_mom_row["momentum"] >= 15 else "brand",
                },
                {
                    "label": "Rata-rata Indeks Minat",
                    "value": f"{avg_interest:.1f}",
                    "hint": "Skala relatif 0–100 lintas titik waktu",
                    "tone": "neu",
                },
                {
                    "label": "Cakupan Pemantauan",
                    "value": f"{gt_view['keyword'].nunique()} Keyword",
                    "hint": f"{', '.join(selected_geos)} · {gt_t0.strftime('%d-%m-%Y')} s.d. {gt_t1.strftime('%d-%m-%Y')}",
                    "tone": "neu",
                    "text": True,
                },
            ])

            # 1. Grafik Multi-Line Tren Minat Pencarian (0-100)
            plot_gt = (
                gt_view.groupby(["timestamp", "keyword"], observed=True)["interest_score"]
                .mean()
                .reset_index()
                .sort_values("timestamp")
            )
            if smooth_ma:
                plot_gt["skor_tampil"] = plot_gt.groupby("keyword")["interest_score"].transform(
                    lambda s: s.rolling(3, min_periods=1).mean()
                )
            else:
                plot_gt["skor_tampil"] = plot_gt["interest_score"]

            fig_gt_line = px.line(
                plot_gt,
                x="timestamp",
                y="skor_tampil",
                color="keyword",
                markers=plot_gt["timestamp"].nunique() <= 60,
                labels={
                    "timestamp": "Waktu (UTC)",
                    "skor_tampil": "Indeks Minat Pencarian (0–100)",
                    "keyword": "Kata Kunci",
                },
            )
            fig_gt_line.update_yaxes(range=[0, 105])
            show_fig(
                style_fig(fig_gt_line, "Dinamika Minat Pencarian Publik (Google Trends Index 0–100)"),
                "gt_main_line",
            )

            # 2. Heatmap Matriks Keyword x Tanggal & Korelasi Silang vs Media Sosial
            col_gt_left, col_gt_right = st.columns(2)
            with col_gt_left:
                heat_df = plot_gt.copy()
                heat_df["tanggal"] = heat_df["timestamp"].dt.strftime("%d-%m-%Y")
                piv = heat_df.pivot_table(
                    index="keyword",
                    columns="tanggal",
                    values="interest_score",
                    aggfunc="mean",
                    fill_value=0,
                )
                # Pertahankan urutan kronologis kolom tanggal
                ordered_cols = (
                    heat_df[["timestamp", "tanggal"]]
                    .drop_duplicates()
                    .sort_values("timestamp")["tanggal"]
                    .tolist()
                )
                piv = piv.reindex(columns=ordered_cols)

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=piv.values,
                        x=piv.columns.tolist(),
                        y=piv.index.tolist(),
                        colorscale=T["heatmap_scale"],
                        zmin=0,
                        zmax=100,
                        colorbar=dict(title="Skor", thickness=12),
                        hovertemplate="Kata Kunci: %{y}<br>Tanggal: %{x}<br>Skor Minat: %{z:.0f}/100<extra></extra>",
                    )
                )
                show_fig(
                    style_fig(fig_heat, "Peta Panas (Heatmap) Intensitas Pencarian per Tanggal", legend_top=False),
                    "gt_heatmap",
                )

            with col_gt_right:
                if filtered_df.empty:
                    sub_head("Korelasi Google Trends vs. Volume Medsos")
                    callout(
                        "info",
                        "Grafik korelasi silang akan aktif otomatis setelah hasil analisis NLP media sosial tersedia.",
                    )
                else:
                    gt_daily = (
                        gt_view.assign(hari=gt_view["timestamp"].dt.floor("1D"))
                        .groupby("hari", observed=True)["interest_score"]
                        .mean()
                        .rename("rerata_trends")
                    )
                    soc_daily = (
                        filtered_df.assign(hari=filtered_df["timestamp"].dt.floor("1D"))
                        .groupby("hari", observed=True)
                        .size()
                        .rename("volume_medsos")
                    )
                    comp = pd.concat([gt_daily, soc_daily], axis=1).sort_index()
                    comp["rerata_trends"] = comp["rerata_trends"].interpolate(limit_direction="both")
                    comp["volume_medsos"] = comp["volume_medsos"].fillna(0)

                    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_dual.add_trace(
                        go.Scatter(
                            x=comp.index,
                            y=comp["rerata_trends"],
                            name="Google Trends (Rata-rata)",
                            mode="lines+markers",
                            line=dict(color=T["brand"], width=2.5),
                        ),
                        secondary_y=False,
                    )
                    fig_dual.add_trace(
                        go.Bar(
                            x=comp.index,
                            y=comp["volume_medsos"],
                            name="Volume Unggahan Medsos",
                            marker_color=T["accent"],
                            opacity=0.45,
                        ),
                        secondary_y=True,
                    )
                    style_fig(fig_dual, "Korelasi Minat Google Trends vs. Volume Percakapan Medsos")
                    fig_dual.update_yaxes(
                        title_text="Indeks Google Trends (0–100)",
                        range=[0, 105],
                        secondary_y=False,
                    )
                    fig_dual.update_yaxes(
                        title_text="Jumlah Unggahan Medsos",
                        showgrid=False,
                        secondary_y=True,
                    )
                    show_fig(fig_dual, "gt_vs_social")

            # 3. Tabel Statistik & Momentum + Unduh CSV
            sub_head("Ringkasan Statistik, Momentum, dan Deteksi Lonjakan per Kata Kunci")
            render_table(
                metrics_df,
                badge=("status_tren",),
                num=("skor_terbaru", "puncak"),
                max_height=380,
            )

            st.download_button(
                "Unduh data Google Trends terfilter (CSV)",
                data=export_filtered_csv(gt_sub_sig, gt_view),
                file_name="kokringan_google_trends.csv",
                mime="text/csv",
                **DOWNLOAD_KW,
            )

html_block('<p style="text-align:center;font-size:.76rem;margin-top:28px;color:var(--muted)!important">KOKRINGAN · 2026</p>')
