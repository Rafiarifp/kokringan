"""
Subprogram 4: Unified Anomaly Detection Engine.
Consolidates anomaly_detection_3.py, cross_platform_anomaly_2.py, and sentiment_anomaly_3.py
under one standardized alert contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _classify_severity(zscore: float, pct_change: float) -> str:
    if zscore >= 3.0 or pct_change >= 300.0:
        return "high"
    if zscore >= 2.0 or pct_change >= 150.0:
        return "medium"
    return "low"


# =========================================================
# 1. SINGLE-PLATFORM VOLUME SPIKE DETECTOR
# =========================================================
def detect_volume_anomalies(
    trend_records: list[dict[str, Any]],
    z_threshold: float = 2.0,
    pct_threshold: float = 100.0,
    min_baseline_points: int = 3,
    min_spike_count: int = 5,
) -> list[dict[str, Any]]:
    if not trend_records:
        return []

    df = pd.DataFrame(trend_records)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_bucket"]).sort_values(["keyword", "platform", "time_bucket"]).reset_index(drop=True)

    alerts: list[dict[str, Any]] = []
    for (kw, plat), group in df.groupby(["keyword", "platform"], observed=True):
        counts = group["post_count"].astype(float).tolist()
        buckets = group["time_bucket"].tolist()

        for idx in range(min_baseline_points, len(counts)):
            curr = counts[idx]
            if curr < min_spike_count:
                continue

            history = np.array(counts[:idx], dtype=float)
            mean_val = float(np.mean(history))
            std_val = float(np.std(history, ddof=0))
            prev_val = counts[idx - 1]

            zscore = (curr - mean_val) / std_val if std_val > 0 else (3.0 if curr > mean_val else 0.0)
            pct_change = ((curr - prev_val) / prev_val * 100.0) if prev_val > 0 else (100.0 if curr > 0 else 0.0)

            if zscore >= z_threshold or pct_change >= pct_threshold:
                severity = _classify_severity(zscore, pct_change)
                alerts.append({
                    "anomaly_type": "volume_spike",
                    "keyword": kw,
                    "platform": plat,
                    "affected_platforms": [plat],
                    "time_bucket": buckets[idx].isoformat(),
                    "severity": severity,
                    "anomaly_score": round(max(zscore, pct_change / 100.0), 2),
                    "description": (
                        f"Lonjakan volume pada {plat} untuk '{kw}': {int(curr)} post "
                        f"(Z-score={zscore:.2f}, perubahan={pct_change:.1f}%)."
                    ),
                    "metrics": {
                        "post_count": int(curr),
                        "baseline_mean": round(mean_val, 2),
                        "baseline_std": round(std_val, 2),
                        "zscore": round(zscore, 2),
                        "pct_change": round(pct_change, 2),
                    },
                    "model_version": "anomaly-detection-v1",
                })
    return alerts


# =========================================================
# 2. CROSS-PLATFORM SURGE DETECTOR
# =========================================================
def detect_cross_platform_anomalies(
    trend_records: list[dict[str, Any]],
    z_threshold: float = 2.0,
    pct_threshold: float = 100.0,
    min_platforms: int = 2,
    min_total_posts: int = 5,
    min_baseline_points: int = 3,
) -> list[dict[str, Any]]:
    if not trend_records:
        return []

    df = pd.DataFrame(trend_records)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_bucket"]).sort_values(["keyword", "platform", "time_bucket"]).reset_index(drop=True)

    spike_rows: list[dict[str, Any]] = []
    for (kw, plat), group in df.groupby(["keyword", "platform"], observed=True):
        counts = group["post_count"].astype(float).tolist()
        buckets = group["time_bucket"].tolist()

        for idx in range(min_baseline_points, len(counts)):
            curr = counts[idx]
            if curr <= 0:
                continue
            history = np.array(counts[:idx], dtype=float)
            mean_val = float(np.mean(history))
            std_val = float(np.std(history, ddof=0))
            prev_val = counts[idx - 1]

            zscore = (curr - mean_val) / std_val if std_val > 0 else (3.0 if curr > mean_val else 0.0)
            pct_change = ((curr - prev_val) / prev_val * 100.0) if prev_val > 0 else (100.0 if curr > 0 else 0.0)

            if zscore >= z_threshold or pct_change >= pct_threshold:
                spike_rows.append({
                    "keyword": kw,
                    "platform": plat,
                    "time_bucket": buckets[idx],
                    "post_count": int(curr),
                    "zscore": zscore,
                    "pct_change": pct_change,
                })

    if not spike_rows:
        return []

    spike_df = pd.DataFrame(spike_rows)
    alerts: list[dict[str, Any]] = []

    for (kw, bucket), grp in spike_df.groupby(["keyword", "time_bucket"], observed=True):
        platforms = sorted(grp["platform"].unique().tolist())
        total_posts = int(grp["post_count"].sum())
        if len(platforms) < min_platforms or total_posts < min_total_posts:
            continue

        max_z = float(grp["zscore"].max())
        max_pct = float(grp["pct_change"].max())
        severity = "high" if len(platforms) >= 3 or max_z >= 3.0 else "medium"
        plat_str = ", ".join(platforms)

        alerts.append({
            "anomaly_type": "cross_platform_surge",
            "keyword": kw,
            "platform": plat_str,
            "affected_platforms": platforms,
            "time_bucket": bucket.isoformat(),
            "severity": severity,
            "anomaly_score": round(max_z, 2),
            "description": (
                f"Lonjakan serentak lintas {len(platforms)} platform ({plat_str}) "
                f"untuk '{kw}' pada {bucket.strftime('%Y-%m-%d %H:%M UTC')} ({total_posts} post)."
            ),
            "metrics": {
                "platform_count": len(platforms),
                "total_posts": total_posts,
                "max_zscore": round(max_z, 2),
                "max_pct_change": round(max_pct, 2),
            },
            "model_version": "cross-platform-anomaly-v1",
        })
    return alerts


# =========================================================
# 3. SENTIMENT SHIFT & POLARITY REVERSAL DETECTOR
# =========================================================
def detect_sentiment_anomalies(
    df: pd.DataFrame,
    freq: str = "3h",
    z_threshold: float = 2.0,
    shift_threshold: float = 0.25,
    min_baseline_buckets: int = 3,
    min_bucket_posts: int = 5,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    work = df.copy()
    work["time_bucket"] = work["timestamp"].dt.floor(freq)
    work["_sent"] = work["sentiment_en"] if "sentiment_en" in work.columns else work["sentiment"].str.lower()
    if "keyword" not in work.columns:
        work["keyword"] = work["topic"] if "topic" in work.columns else "general"

    grouped = work.groupby(["keyword", "platform", "time_bucket", "_sent"], observed=True).size().unstack(fill_value=0)
    for col in ("positive", "neutral", "negative"):
        if col not in grouped.columns:
            grouped[col] = 0
    dist_df = grouped[["positive", "neutral", "negative"]].reset_index()
    dist_df["total_posts"] = dist_df["positive"] + dist_df["neutral"] + dist_df["negative"]
    dist_df = dist_df[dist_df["total_posts"] >= min_bucket_posts].copy()
    if dist_df.empty:
        return []

    dist_df["positive_ratio"] = dist_df["positive"] / dist_df["total_posts"]
    dist_df["negative_ratio"] = dist_df["negative"] / dist_df["total_posts"]
    dist_df["polarity"] = dist_df["positive_ratio"] - dist_df["negative_ratio"]
    dist_df = dist_df.sort_values(["keyword", "platform", "time_bucket"]).reset_index(drop=True)

    alerts: list[dict[str, Any]] = []
    for (kw, plat), grp in dist_df.groupby(["keyword", "platform"], observed=True):
        if len(grp) <= min_baseline_buckets:
            continue

        neg_vals = grp["negative_ratio"].tolist()
        pos_vals = grp["positive_ratio"].tolist()
        pol_vals = grp["polarity"].tolist()
        buckets = grp["time_bucket"].tolist()
        totals = grp["total_posts"].tolist()

        for idx in range(min_baseline_buckets, len(grp)):
            hist_neg = np.array(neg_vals[:idx], dtype=float)
            neg_mean, neg_std = float(np.mean(hist_neg)), float(np.std(hist_neg, ddof=0))
            curr_neg = neg_vals[idx]
            neg_z = (curr_neg - neg_mean) / neg_std if neg_std > 0 else 0.0
            neg_shift = curr_neg - neg_mean

            if neg_z >= z_threshold or neg_shift >= shift_threshold:
                sev = "high" if (neg_z >= 3.0 or neg_shift >= 0.40) else "medium"
                alerts.append({
                    "anomaly_type": "negative_sentiment_spike",
                    "keyword": kw,
                    "platform": plat,
                    "affected_platforms": [plat],
                    "time_bucket": buckets[idx].isoformat(),
                    "severity": sev,
                    "anomaly_score": round(max(neg_z, neg_shift * 10), 2),
                    "description": (
                        f"Lonjakan sentimen negatif di {plat} untuk '{kw}': "
                        f"{curr_neg * 100:.1f}% (baseline {neg_mean * 100:.1f}%, {int(totals[idx])} post)."
                    ),
                    "metrics": {
                        "negative_ratio": round(curr_neg, 4),
                        "baseline_negative_ratio": round(neg_mean, 4),
                        "zscore": round(neg_z, 2),
                        "shift": round(neg_shift, 4),
                    },
                    "model_version": "sentiment-anomaly-v1",
                })

            pol_delta = abs(pol_vals[idx] - pol_vals[idx - 1])
            if pol_delta >= shift_threshold * 1.5 and (pol_vals[idx] * pol_vals[idx - 1] < 0):
                alerts.append({
                    "anomaly_type": "sentiment_polarity_reversal",
                    "keyword": kw,
                    "platform": plat,
                    "affected_platforms": [plat],
                    "time_bucket": buckets[idx].isoformat(),
                    "severity": "medium",
                    "anomaly_score": round(pol_delta, 2),
                    "description": (
                        f"Pembalikan polaritas sentimen di {plat} untuk '{kw}': "
                        f"{pol_vals[idx - 1]:+.2f} menjadi {pol_vals[idx]:+.2f}."
                    ),
                    "metrics": {
                        "previous_polarity": round(pol_vals[idx - 1], 4),
                        "current_polarity": round(pol_vals[idx], 4),
                        "positive_ratio": round(pos_vals[idx], 4),
                    },
                    "model_version": "sentiment-anomaly-v1",
                })
    return alerts


# =========================================================
# 4. UNIFIED ENTRYPOINT
# =========================================================
def detect_all_anomalies(
    df: pd.DataFrame,
    trend_records: list[dict[str, Any]],
    freq: str = "3h",
    z_threshold: float = 2.0,
    pct_threshold: float = 100.0,
    shift_threshold: float = 0.25,
) -> dict[str, Any]:
    """Run volume, cross-platform, and sentiment anomaly detectors and return unified alert summary."""
    vol_alerts = detect_volume_anomalies(trend_records, z_threshold=z_threshold, pct_threshold=pct_threshold)
    cross_alerts = detect_cross_platform_anomalies(trend_records, z_threshold=z_threshold, pct_threshold=pct_threshold)
    sent_alerts = detect_sentiment_anomalies(df, freq=freq, z_threshold=z_threshold, shift_threshold=shift_threshold)

    all_alerts = vol_alerts + cross_alerts + sent_alerts
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for alert in all_alerts:
        sev = str(alert.get("severity", "low")).lower()
        if sev in severity_counts:
            severity_counts[sev] += 1

    return {
        "total_alerts": len(all_alerts),
        "volume_alert_count": len(vol_alerts),
        "cross_platform_alert_count": len(cross_alerts),
        "sentiment_alert_count": len(sent_alerts),
        "high_alerts": severity_counts["high"],
        "medium_alerts": severity_counts["medium"],
        "low_alerts": severity_counts["low"],
        "alerts": all_alerts,
        "model_version": "unified-anomaly-engine-v1",
    }
