import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from core.data_foundation import clean_text_for_embedding, normalize_posts_df


# =====================================================================
# 1. AGREGASI SENTIMEN (GLOBAL, PLATFORM, TOPIK, HARIAN)
# =====================================================================
def aggregate_sentiment(df: pd.DataFrame, freq: str = "D") -> dict[str, Any]:
    """Menghitung distribusi sentimen keseluruhan, per platform, per topik, dan deret waktu."""
    norm_df = normalize_posts_df(df)
    total = int(len(norm_df))
    if total == 0:
        return {
            "total_posts": 0,
            "overall": {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "positive_pct": 0.0,
                "neutral_pct": 0.0,
                "negative_pct": 0.0,
            },
            "by_platform": [],
            "by_keyword": [],
            "daily_trend": [],
        }

    counts = norm_df["sentiment_en"].value_counts()
    pos = int(counts.get("positive", 0))
    neu = int(counts.get("neutral", 0))
    neg = int(counts.get("negative", 0))

    overall = {
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "positive_pct": round(pos / total * 100, 2),
        "neutral_pct": round(neu / total * 100, 2),
        "negative_pct": round(neg / total * 100, 2),
    }

    def _group_breakdown(group_col: str) -> list[dict[str, Any]]:
        if group_col not in norm_df.columns:
            return []
        piv = (
            norm_df.groupby([group_col, "sentiment_en"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        for col in ("positive", "neutral", "negative"):
            if col not in piv.columns:
                piv[col] = 0
        piv["total"] = piv["positive"] + piv["neutral"] + piv["negative"]
        piv = piv.sort_values("total", ascending=False)

        records = []
        for idx_val, row in piv.iterrows():
            tot_g = int(row["total"])
            if tot_g == 0:
                continue
            p_g = int(row["positive"])
            u_g = int(row["neutral"])
            n_g = int(row["negative"])
            records.append(
                {
                    group_col: str(idx_val),
                    "total": tot_g,
                    "positive": p_g,
                    "neutral": u_g,
                    "negative": n_g,
                    "positive_pct": round(p_g / tot_g * 100, 2),
                    "neutral_pct": round(u_g / tot_g * 100, 2),
                    "negative_pct": round(n_g / tot_g * 100, 2),
                }
            )
        return records

    # Tren harian / periodik
    trend_df = norm_df.copy()
    trend_df["time_bucket"] = trend_df["timestamp"].dt.floor(freq)
    t_piv = (
        trend_df.groupby(["time_bucket", "sentiment_en"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for col in ("positive", "neutral", "negative"):
        if col not in t_piv.columns:
            t_piv[col] = 0
    t_piv["total"] = t_piv["positive"] + t_piv["neutral"] + t_piv["negative"]

    daily_trend = []
    for ts, row in t_piv.iterrows():
        tot_t = int(row["total"])
        daily_trend.append(
            {
                "time_bucket": ts.isoformat(),
                "total": tot_t,
                "positive": int(row["positive"]),
                "neutral": int(row["neutral"]),
                "negative": int(row["negative"]),
                "negative_pct": round(int(row["negative"]) / tot_t * 100, 2) if tot_t else 0.0,
            }
        )

    return {
        "total_posts": total,
        "overall": overall,
        "by_platform": _group_breakdown("platform"),
        "by_keyword": _group_breakdown("topic"),
        "daily_trend": daily_trend,
    }


# =====================================================================
# 2. DERET WAKTU & ANALISIS TREN
# =====================================================================
def build_time_series(
    df: pd.DataFrame,
    freq: str = "3h",
    metric: str = "Volume Postingan",
    by: str | None = None,
) -> pd.DataFrame:
    """Membangun deret waktu kontinu (interval kosong diisi 0) untuk grafik & anomali."""
    norm_df = normalize_posts_df(df)
    if norm_df.empty:
        cols = ["time_bucket", "value"] + ([by] if by else [])
        return pd.DataFrame(columns=cols)

    work = norm_df.copy()
    work["time_bucket"] = work["timestamp"].dt.floor(freq)

    min_ts = work["time_bucket"].min()
    max_ts = work["time_bucket"].max()
    full_grid = pd.date_range(start=min_ts, end=max_ts, freq=freq, tz="UTC")

    group_cols = ["time_bucket"] + ([by] if by and by in work.columns else [])

    if metric == "Total Engagement":
        agg = work.groupby(group_cols, observed=True)["engagement"].sum()
    elif metric == "Sentimen Negatif":
        neg_only = work[work["sentiment_en"] == "negative"]
        agg = neg_only.groupby(group_cols, observed=True).size()
    else:
        agg = work.groupby(group_cols, observed=True).size()

    if by and by in work.columns:
        categories = sorted(work[by].dropna().astype(str).unique().tolist())
        multi_idx = pd.MultiIndex.from_product(
            [full_grid, categories], names=["time_bucket", by]
        )
        res = agg.reindex(multi_idx, fill_value=0).reset_index(name="value")
    else:
        res = agg.reindex(full_grid, fill_value=0).reset_index()
        res.columns = ["time_bucket", "value"]

    return res.sort_values("time_bucket").reset_index(drop=True)


def analyze_keyword_trend(
    df: pd.DataFrame,
    keyword: str | None = None,
    interval: str = "3h",
    ma_window: int = 3,
) -> list[dict[str, Any]]:
    """Menghitung volume unggahan per interval, moving average, % perubahan, dan arah tren."""
    norm_df = normalize_posts_df(df)
    if norm_df.empty:
        return []

    work = norm_df.copy()
    if keyword:
        work = work[work["topic"].astype(str).str.lower() == str(keyword).lower()]
    if work.empty:
        return []

    work["time_bucket"] = work["timestamp"].dt.floor(interval)
    min_ts = work["time_bucket"].min()
    max_ts = work["time_bucket"].max()
    full_grid = pd.date_range(start=min_ts, end=max_ts, freq=interval, tz="UTC")

    records: list[dict[str, Any]] = []
    grouped = work.groupby(["topic", "platform"], observed=True)

    for (kw_val, plat_val), grp in grouped:
        counts = grp.groupby("time_bucket").size().reindex(full_grid, fill_value=0)
        ts_df = pd.DataFrame({"time_bucket": full_grid, "post_count": counts.values})
        ts_df["moving_average"] = (
            ts_df["post_count"].rolling(window=ma_window, min_periods=1).mean().round(2)
        )

        prev_counts = ts_df["post_count"].shift(1)
        change_pcts = []
        directions = []

        for curr, prev in zip(ts_df["post_count"], prev_counts):
            if pd.isna(prev):
                pct = 0.0
            elif prev == 0:
                pct = 100.0 if curr > 0 else 0.0
            else:
                pct = round((curr - prev) / prev * 100.0, 2)

            change_pcts.append(pct)
            if pct >= 20.0 and curr >= 2:
                directions.append("increasing")
            elif pct <= -20.0:
                directions.append("decreasing")
            else:
                directions.append("stable")

        ts_df["change_percentage"] = change_pcts
        ts_df["trend_direction"] = directions

        for row in ts_df.itertuples(index=False):
            records.append(
                {
                    "keyword": str(kw_val),
                    "platform": str(plat_val),
                    "time_bucket": row.time_bucket.isoformat(),
                    "post_count": int(row.post_count),
                    "moving_average": float(row.moving_average),
                    "change_percentage": float(row.change_percentage),
                    "trend_direction": row.trend_direction,
                }
            )

    return records


# =====================================================================
# 3. KLASTERING NARASI SEMANTIK (COSINE DBSCAN + NOISE FILTER)
# =====================================================================
_RE_ALPHA_WORDS = re.compile(r"[a-zA-ZÀ-ÿ0-9]{2,}")


def _is_substantive_for_clustering(
    text: str,
    min_words: int = 5,
    min_chars: int = 30,
) -> bool:
    """
    Menyaring komentar pendek / receh (mis. 'ndabrus aja lo', 'hadir bang', 'wkwk')
    agar tidak menggumpal menjadi klaster sampah di ruang embedding.
    """
    cleaned = clean_text_for_embedding(text)
    if len(cleaned) < min_chars:
        return False
    words = _RE_ALPHA_WORDS.findall(cleaned)
    if len(words) < min_words:
        return False
    # Pastikan kosakata tidak cuma pengulangan 1-2 kata yang sama (mis. 'mantap mantap mantap')
    if len(set(w.lower() for w in words)) < max(3, min_words - 1):
        return False
    return True


def _pick_representative_text(
    texts: list[str],
    cluster_embeddings: np.ndarray,
    centroid: np.ndarray,
) -> str:
    """Memilih medoid paling dekat dengan centroid sekaligus bersih & mudah dibaca."""
    sims = cluster_embeddings @ centroid
    # Ambil top-3 kandidat terdekat dengan centroid, lalu pilih yang paling rapi
    top_k = np.argsort(sims)[::-1][: min(3, len(texts))]
    best_idx = int(top_k[0])
    best_score = -1e9

    for idx in top_k:
        cand = str(texts[int(idx)]).strip()
        sim_score = float(sims[int(idx)])
        # Penalti ringan jika terlalu pendek atau terlalu panjang (> 240 karakter)
        len_bonus = 0.05 if 35 <= len(cand) <= 220 else 0.0
        total_score = sim_score + len_bonus
        if total_score > best_score:
            best_score = total_score
            best_idx = int(idx)

    rep = " ".join(str(texts[best_idx]).split())
    return rep[:240] + ("..." if len(rep) > 240 else "")


def detect_narratives(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    eps: float = 0.19,
    min_samples: int = 3,
    keyword: str | None = None,
    min_words: int = 5,
    min_chars: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Mengelompokkan narasi berdasarkan kemiripan semantik vektor embedding
    menggunakan Cosine DBSCAN dengan penyaringan komentar pendek/noise.
    """
    norm_df = normalize_posts_df(df)
    norm_df["narrative_cluster_id"] = -1

    empty_summary: dict[str, Any] = {
        "total_clusters": 0,
        "clustered_posts": 0,
        "noise_posts": int(len(norm_df)),
        "clusters": [],
    }

    if norm_df.empty or embeddings is None or len(embeddings) != len(norm_df):
        return norm_df, empty_summary

    # Mask kandidat substansial (lolos filter panjang + filter topik jika diminta)
    substantive_mask = norm_df["text"].apply(
        lambda t: _is_substantive_for_clustering(t, min_words=min_words, min_chars=min_chars)
    ).to_numpy(dtype=bool)

    if keyword:
        kw_mask = (
            norm_df["topic"].astype(str).str.lower() == str(keyword).lower()
        ).to_numpy(dtype=bool)
        substantive_mask = substantive_mask & kw_mask

    valid_indices = np.where(substantive_mask)[0]
    if len(valid_indices) < min_samples:
        return norm_df, empty_summary

    sub_emb = embeddings[valid_indices]
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    sub_labels = clustering.fit_predict(sub_emb)

    # Remap cluster ID secara terurut berdasarkan ukuran klaster terbesar -> terkecil (0, 1, 2, ...)
    unique_raw = [c for c in set(sub_labels.tolist()) if c != -1]
    if not unique_raw:
        return norm_df, empty_summary

    raw_sizes = {c: int(np.sum(sub_labels == c)) for c in unique_raw}
    sorted_raw_ids = sorted(unique_raw, key=lambda c: (-raw_sizes[c], c))
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted_raw_ids)}

    remapped_labels = np.array(
        [remap[lbl] if lbl in remap else -1 for lbl in sub_labels],
        dtype=int,
    )
    norm_df.iloc[valid_indices, norm_df.columns.get_loc("narrative_cluster_id")] = remapped_labels

    clusters_meta: list[dict[str, Any]] = []
    for new_id in range(len(sorted_raw_ids)):
        local_pos = np.where(remapped_labels == new_id)[0]
        global_pos = valid_indices[local_pos]

        c_emb = sub_emb[local_pos]
        centroid = c_emb.mean(axis=0)
        norm_c = np.linalg.norm(centroid)
        if norm_c > 0:
            centroid = centroid / norm_c

        c_df = norm_df.iloc[global_pos]
        rep_title = _pick_representative_text(
            c_df["text"].tolist(),
            c_emb,
            centroid,
        )

        platforms = sorted(c_df["platform"].dropna().astype(str).unique().tolist())
        topics = sorted(c_df["topic"].dropna().astype(str).unique().tolist())
        sent_counts = c_df["sentiment_en"].value_counts().to_dict()
        dominant_sent = max(sent_counts, key=sent_counts.get) if sent_counts else "neutral"

        clusters_meta.append(
            {
                "cluster_id": int(new_id),
                "narrative_title": rep_title,
                "post_count": int(len(global_pos)),
                "unique_authors": int(c_df["author"].nunique()),
                "platforms": platforms,
                "topics": topics,
                "dominant_sentiment": dominant_sent,
                "negative_pct": round(
                    int(sent_counts.get("negative", 0)) / len(global_pos) * 100, 2
                ),
            }
        )

    clustered_total = int((norm_df["narrative_cluster_id"] >= 0).sum())
    summary = {
        "total_clusters": len(clusters_meta),
        "clustered_posts": clustered_total,
        "noise_posts": int(len(norm_df) - clustered_total),
        "clusters": clusters_meta,
    }
    return norm_df, summary


# =====================================================================
# 4. EVALUASI CIB & SKOR RISIKO DISARM
# =====================================================================
def analyze_cib_and_disarm(
    df: pd.DataFrame,
    anomaly_alerts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Mendeteksi indikator Coordinated Inauthentic Behavior (CIB), memetakan TTP DISARM,
    dan menghitung skor risiko operasional (0.0 - 10.0).
    """
    norm_df = normalize_posts_df(df)
    total = int(len(norm_df))
    if total == 0:
        return {
            "risk_score": 0.0,
            "risk_level": "low",
            "cib_post_count": 0,
            "cib_ratio": 0.0,
            "coordinated_authors": [],
            "ttps": [],
        }

    # Cabang 1: Duplikasi teks identik (minimal 25 karakter) oleh > 1 akun berbeda
    work = norm_df.copy()
    work["_norm_dup"] = work["text"].apply(clean_text_for_embedding)
    substantive_dup = work[work["_norm_dup"].str.len() >= 25]
    dup_groups = substantive_dup.groupby("_norm_dup")["author"].nunique()
    dup_texts = set(dup_groups[dup_groups > 1].index.tolist())
    mask_dup = work["_norm_dup"].isin(dup_texts)

    # Cabang 2: Keterlibatan dalam klaster narasi substansial multi-aktor (>= 3 akun & >= 4 post)
    mask_cluster = pd.Series(False, index=work.index)
    if "narrative_cluster_id" in work.columns:
        valid_c = work[work["narrative_cluster_id"] >= 0]
        if not valid_c.empty:
            c_stats = valid_c.groupby("narrative_cluster_id").agg(
                posts=("id", "count"),
                authors=("author", "nunique"),
            )
            coord_clusters = c_stats[
                (c_stats["posts"] >= 4) & (c_stats["authors"] >= 3)
            ].index.tolist()
            mask_cluster = work["narrative_cluster_id"].isin(coord_clusters)

    cib_df = work[mask_dup | mask_cluster]
    cib_count = int(len(cib_df))
    cib_ratio = cib_count / total

    neg_ratio = float((norm_df["sentiment_en"] == "negative").sum()) / total
    num_alerts = len(anomaly_alerts) if anomaly_alerts else 0

    risk_score = min(
        10.0,
        round(
            (4.0 * neg_ratio)
            + min(3.5, 5.0 * cib_ratio)
            + min(2.5, 0.4 * num_alerts),
            2,
        ),
    )

    if risk_score >= 7.0:
        risk_level = "high"
    elif risk_score >= 4.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    ttps: list[dict[str, Any]] = []
    if mask_dup.any():
        ttps.append(
            {
                "tactic_id": "TA08",
                "technique_id": "T0015",
                "name": "Create Hashtags and Search Artifacts / Copypasta Amplification",
                "evidence_count": int(mask_dup.sum()),
                "description": "Penyebaran teks identik secara berulang lintas akun untuk memanipulasi visibilitas.",
            }
        )
    if mask_cluster.any():
        ttps.append(
            {
                "tactic_id": "TA11",
                "technique_id": "T0049",
                "name": "Flood Information Space / Coordinated Narrative Clustering",
                "evidence_count": int(mask_cluster.sum()),
                "description": "Penggiringan narasi semantik seragam oleh banyak aktor dalam jendela waktu berdekatan.",
            }
        )
    if num_alerts > 0:
        ttps.append(
            {
                "tactic_id": "TA12",
                "technique_id": "T0050",
                "name": "Cross-Platform Cross-Posting & Volume Surge",
                "evidence_count": int(num_alerts),
                "description": "Terdeteksi lonjakan volume atau pergeseran polaritas sentimen signifikan.",
            }
        )

    coord_authors = (
        cib_df["author"].value_counts().head(10).to_dict() if not cib_df.empty else {}
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "cib_post_count": cib_count,
        "cib_ratio": round(cib_ratio, 4),
        "coordinated_authors": coord_authors,
        "ttps": ttps,
    }
