"""
Subprogram 6B: CLI Ground-Truth Sentiment Model Evaluation.
Refactored from sentiment_evaluation_2.py to use nlp.inference_engine.SentimentRunner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.data_foundation import SENTIMENT_ID_TO_EN, VALID_SENTIMENTS_EN
from nlp.inference_engine import SentimentRunner

LABELS = ["positive", "neutral", "negative"]


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth dataset not found: {csv_path}")

    df = pd.read_csv(path)
    if "text" not in df.columns and "content" in df.columns:
        df["text"] = df["content"]
    if "label" not in df.columns and "sentiment" in df.columns:
        df["label"] = df["sentiment"]

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SENTIMENT_ID_TO_EN)
        .fillna(df["label"].astype(str).str.strip().str.lower())
    )
    df = df[(df["text"] != "") & (df["label"].isin(VALID_SENTIMENTS_EN))].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid evaluation rows after normalization.")
    return df


def evaluate_sentiment_model(csv_path: str, batch_size: int = 16) -> dict[str, Any]:
    df = load_ground_truth(csv_path)
    runner = SentimentRunner()
    predictions = runner.predict_batch(df["text"].tolist(), batch_size=batch_size)
    runner.unload()

    y_true = df["label"].tolist()
    y_pred = [p["sentiment"] for p in predictions]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report_dict = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    return {
        "total_samples": len(df),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            lbl: {
                "precision": round(float(report_dict[lbl]["precision"]), 4),
                "recall": round(float(report_dict[lbl]["recall"]), 4),
                "f1_score": round(float(report_dict[lbl]["f1-score"]), 4),
                "support": int(report_dict[lbl]["support"]),
            }
            for lbl in LABELS
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IndoBERT sentiment classifier against ground truth.")
    parser.add_argument("--data", default="ground_truth.csv", help="Path to ground-truth CSV (text, label)")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    metrics = evaluate_sentiment_model(args.data, batch_size=args.batch_size)
    print(f"Samples     : {metrics['total_samples']}")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 : {metrics['weighted_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for cls_name, vals in metrics["per_class"].items():
        print(
            f"  {cls_name:<10} | P={vals['precision']:.4f} | R={vals['recall']:.4f} "
            f"| F1={vals['f1_score']:.4f} | n={vals['support']}"
        )
