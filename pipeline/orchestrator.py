from __future__ import annotations
import logging
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.anomaly_engine import detect_all_anomalies
from analytics.narrative_and_trends import (
    aggregate_sentiment,
    analyze_cib_and_disarm,
    analyze_keyword_trend,
    detect_narratives,
)
from core.data_foundation import normalize_posts_df
from nlp.inference_engine import InferenceEngine

TREND_ID_LABEL = {
    "increasing": "meningkat",
    "decreasing": "menurun",
    "stable": "stabil",
}

SENTIMENT_PROSE_LABEL = {
    "positive": "positif",
    "negative": "negatif",
    "neutral": "netral",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "latest_run.log"

def setup_pipeline_logger(append: bool = True) -> logging.Logger:
    file_mode = "a" if append else "w"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, mode=file_mode, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("cyberlens_pipeline")

# =========================================================
# 1. EXECUTIVE INTELLIGENCE SYNTHESIZER (INDONESIAN)
# =========================================================
def generate_intelligence_summary(
    keyword: str,
    sentiment_result: dict[str, Any],
    trend_records: list[dict[str, Any]],
    narrative_result: dict[str, Any],
    anomaly_result: dict[str, Any],
    combined_sentiment_result: dict[str, Any],
    disarm_result: dict[str, Any],
) -> dict[str, Any]:
    overall = sentiment_result.get("overall", {})
    total_posts = int(overall.get("total_posts", 0))
    pos_pct = float(overall.get("positive_pct", 0.0))
    neg_pct = float(overall.get("negative_pct", 0.0))
    neu_pct = float(overall.get("neutral_pct", 0.0))

    # Trend snapshot
    if trend_records:
        sorted_tr = sorted(trend_records, key=lambda r: str(r.get("time_bucket", "")))
        latest = sorted_tr[-1]
        trend_dir = str(latest.get("trend_direction", "stable"))
        change_pct = float(latest.get("change_percentage", 0.0))
        latest_count = int(latest.get("post_count", 0))
    else:
        trend_dir, change_pct, latest_count = "stable", 0.0, 0

    # Narrative snapshot
    narratives = narrative_result.get("narratives", [])
    cluster_count = int(narrative_result.get("cluster_count", len(narratives)))
    top_narrative = narratives[0] if narratives else None

    # Anomaly & combined sarcasm snapshot
    total_alerts = int(anomaly_result.get("total_alerts", 0))
    high_alerts = int(anomaly_result.get("high_alerts", 0))
    medium_alerts = int(anomaly_result.get("medium_alerts", 0))
    comb_stats = combined_sentiment_result.get("statistics", {})
    sarc_count = int(comb_stats.get("sarcasm_distribution", {}).get("sarcastic", 0))
    review_count = int(comb_stats.get("needs_review", 0))
    risk_score = float(disarm_result.get("total_risk_score", 0.0))

    # Risk level determination
    if high_alerts >= 1 or risk_score >= 7.0 or (neg_pct >= 50.0 and trend_dir == "increasing"):
        risk_level = "high"
    elif medium_alerts >= 1 or risk_score >= 4.0 or neg_pct >= 35.0 or sarc_count >= 5:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Dominant sentiment
    scores = {"positive": pos_pct, "negative": neg_pct, "neutral": neu_pct}
    dominant_en = max(scores, key=scores.get) if total_posts > 0 else "neutral"
    dominant_id = SENTIMENT_PROSE_LABEL[dominant_en]
    trend_id = TREND_ID_LABEL.get(trend_dir, "stabil")

    summary = (
        f"Pemantauan kata kunci '{keyword}' mencatat {total_posts:,} unggahan dengan "
        f"tren percakapan {trend_id} (perubahan {change_pct:+.1f}%). "
        f"Sentimen publik didominasi kategori {dominant_id} "
        f"(positif {pos_pct:.1f}%, netral {neu_pct:.1f}%, negatif {neg_pct:.1f}%), "
        f"dengan skor risiko DISARM {risk_score:.1f}/10.0."
    )
    if top_narrative:
        summary += (
            f" Narasi utama terdeteksi pada klaster '{top_narrative.get('narrative_title', '')[:120]}' "
            f"({top_narrative.get('post_count', 0)} unggahan)."
        )
    if total_alerts > 0:
        summary += f" Sistem mendeteksi {total_alerts} peringatan anomali ({high_alerts} prioritas tinggi)."

    findings: list[str] = [
        f"Total volume pemantauan mencapai {total_posts:,} unggahan dengan distribusi sentimen "
        f"positif {pos_pct:.1f}%, netral {neu_pct:.1f}%, dan negatif {neg_pct:.1f}%.",
        f"Arah tren terbaru berada pada status {trend_id} dengan {latest_count} unggahan pada interval terakhir.",
    ]
    if cluster_count > 0 and top_narrative:
        findings.append(
            f"Teridentifikasi {cluster_count} klaster narasi semantik; klaster terpadat memuat "
            f"{top_narrative.get('post_count', 0)} unggahan lintas {', '.join(top_narrative.get('platforms', []))}."
        )
    if sarc_count > 0 or review_count > 0:
        findings.append(
            f"Terdeteksi {sarc_count:,} unggahan terindikasi sarkastik dan {review_count:,} unggahan memerlukan verifikasi analis."
        )
    if disarm_result.get("suspect_count", 0) > 0:
        findings.append(
            f"Indikasi aktivitas terkoordinasi (CIB) melibatkan {disarm_result['suspect_count']:,} akun "
            f"({disarm_result['cib_ratio'] * 100:.1f}% dari total unggahan)."
        )
    if total_alerts > 0:
        findings.append(f"Tercatat {total_alerts} alert anomali ({high_alerts} tinggi, {medium_alerts} sedang).")

    recommendations: list[str] = []
    if risk_level == "high":
        recommendations.append("Lakukan pemantauan intensif dan verifikasi sumber utama atas lonjakan sentimen negatif atau anomali tinggi.")
    if neg_pct >= 35.0:
        recommendations.append("Petakan pemicu utama narasi negatif dan siapkan klarifikasi komunikasi berbasis fakta.")
    if sarc_count > 0 or review_count > 0:
        recommendations.append("Tinjau unggahan berlabel sarkasme/kritik terselubung agar tidak terjadi bias pembacaan sentimen.")
    if disarm_result.get("suspect_count", 0) > 0:
        recommendations.append("Audit jaringan akun dengan duplikasi teks identik dan klaster terkoordinasi menggunakan matriks TTP DISARM.")
    if not recommendations:
        recommendations.append("Pertahankan pemantauan berkala karena distribusi sentimen dan volume masih berada pada batas normal.")

    return {
        "keyword": keyword,
        "risk_level": risk_level,
        "summary": summary,
        "key_findings": findings,
        "recommendations": recommendations,
        "sentiment": overall,
        "trend": {
            "latest_direction": trend_dir,
            "latest_change_percentage": change_pct,
            "latest_post_count": latest_count,
        },
        "narrative": {
            "cluster_count": cluster_count,
            "top_narrative": top_narrative,
        },
        "anomaly": anomaly_result,
        "disarm": {
            "total_risk_score": disarm_result.get("total_risk_score", 0.0),
            "threat_class": disarm_result.get("threat_class", "low"),
            "suspect_count": disarm_result.get("suspect_count", 0),
            "cib_ratio": disarm_result.get("cib_ratio", 0.0),
            "ttps": disarm_result.get("ttps", []),
        },
        "combined_sentiment": comb_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "intelligence-summary-v2",
    }


# =========================================================
# 2. SINGLE-PASS PIPELINE ORCHESTRATOR
# =========================================================
def run_cyberlens_pipeline(
    input_csv: str = "data/master_osint_data.csv",
    output_csv: str = "data/master_osint_enriched.csv",
    output_report_json: str = "data/cyberlens_report.json",
    topic_filter: str | None = None,
    interval: str = "3h",
    sarcasm_threshold: float = 0.70,
    narrative_eps: float = 0.25,
    min_samples: int = 2,
    max_rows: int | None = None,
    save_to_database: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute the complete KOKRINGAN x CYBERLENS analytical DAG in a single in-memory pass.
    Emits master_osint_enriched.csv and cyberlens_report.json for the Streamlit dashboard.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw_df = pd.read_csv(input_path)
    df = normalize_posts_df(raw_df, default_keyword=topic_filter or "general", categorize=False)

    if topic_filter:
        df = df[df["topic"].str.contains(topic_filter, case=False, na=False)].reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty after normalization and filtering.")

    target_keyword = topic_filter if topic_filter else "Semua Topik OSINT"
    print(f"[0/6] Dataset normalized: {len(df):,} posts | Target: {target_keyword}")

    # Stage 1: NLP Inference (Sentiment + Sarcasm + Multilingual Embeddings)
    print("[1/6] Running IndoBERT Sentiment, Sarcasm & SentenceTransformer Embeddings...")
    engine = InferenceEngine()
    df, combined_sentiment_result, embeddings = engine.enrich_dataframe(
        df=df,
        sarcasm_threshold=sarcasm_threshold,
        compute_embeddings=True,
        unload_after_stage=True,
    )

    # Stage 2: Sentiment Aggregation
    print("[2/6] Aggregating sentiment statistics...")
    sentiment_result = aggregate_sentiment(df, freq="D")

    # Stage 3: Temporal Trend Analysis
    print("[3/6] Analyzing volume trends...")
    trend_records = analyze_keyword_trend(df, keyword=target_keyword, interval=interval)

    # Stage 4: Semantic Narrative Clustering
    print("[4/6] Clustering semantic narratives (DBSCAN)...")
    df, narrative_result = detect_narratives(
        df=df,
        embeddings=embeddings,
        eps=narrative_eps,
        min_samples=min_samples,
        keyword=target_keyword,
    )

    # Stage 5: Unified Anomaly Detection + CIB/DISARM Scoring
    print("[5/6] Detecting anomalies and evaluating DISARM risk...")
    anomaly_result = detect_all_anomalies(df=df, trend_records=trend_records, freq=interval)
    disarm_result = analyze_cib_and_disarm(df=df, anomaly_alerts=anomaly_result["alerts"])

    # Stage 6: Indonesian Intelligence Summary
    print("[6/6] Synthesizing executive intelligence report...")
    intelligence_summary = generate_intelligence_summary(
        keyword=target_keyword,
        sentiment_result=sentiment_result,
        trend_records=trend_records,
        narrative_result=narrative_result,
        anomaly_result=anomaly_result,
        combined_sentiment_result=combined_sentiment_result,
        disarm_result=disarm_result,
    )

    # Persist local artifacts for Streamlit dashboard
    export_cols = [
        "id", "timestamp", "platform", "platform_detail", "author",
        "text", "cleaned_text", "engagement", "topic", "url",
        "sentiment", "sentiment_en", "sentiment_confidence",
        "positive_score", "neutral_score", "negative_score",
        "sarcasm", "is_sarcastic", "sarcasm_confidence",
        "sarcasm_status", "interpretation", "needs_review",
        "narrative_cluster_id",
    ]
    export_df = df[export_cols].copy()
    export_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(export_df),
        "sentiment_summary": sentiment_result,
        "combined_sentiment_stats": combined_sentiment_result["statistics"],
        "trend_records": trend_records,
        "narratives": narrative_result.get("narratives", []),
        "narrative_cluster_count": narrative_result.get("cluster_count", 0),
        "anomalies": anomaly_result,
        "disarm_analysis": disarm_result,
        "intelligence_summary": intelligence_summary,
    }
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    # Optional Supabase persistence (if database integration modules are configured)
    if save_to_database:
        from integrations.cyberlens_db import (
            save_anomaly_alerts,
            save_intelligence_summary,
            save_narrative_clusters,
            save_sentiment_predictions,
        )

        save_sentiment_predictions(combined_sentiment_result["results"])
        save_narrative_clusters(narrative_result["narratives"])
        save_anomaly_alerts(anomaly_result["alerts"])
        save_intelligence_summary(intelligence_summary)

    print(f"Completed: {output_csv} ({len(export_df):,} rows) & {output_report_json}")
    return export_df, full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KOKRINGAN x CYBERLENS unified pipeline.")
    parser.add_argument("--input", default="data/master_osint_data.csv", help="Path to raw scraper CSV")
    parser.add_argument("--output-csv", default="data/master_osint_enriched.csv", help="Path to enriched output CSV")
    parser.add_argument("--output-json", default="data/cyberlens_report.json", help="Path to analytical JSON report")
    parser.add_argument("--topic", default=None, help="Optional keyword/topic filter")
    parser.add_argument("--interval", default="3h", choices=["1h", "3h", "6h", "1D"], help="Aggregation interval")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to process")
    parser.add_argument("--save-db", action="store_true", help="Persist results to Supabase")
    args = parser.parse_args()

    run_cyberlens_pipeline(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_report_json=args.output_json,
        topic_filter=args.topic,
        interval=args.interval,
        max_rows=args.max_rows,
        save_to_database=args.save_db,
    )
