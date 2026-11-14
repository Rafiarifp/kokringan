"""
Subprogram 2: Unified Neural Inference Engine.
Consolidates sentiment_analysis_2.py, sarcasm_detection_2.py,
combined_sentiment_analysis_2.py, and embedding generation from narrative_detection_2.py.
Implements lazy model loading and explicit VRAM release between stages.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.data_foundation import (
    SENTIMENT_EN_TO_ID,
    clean_text_for_embedding,
    preprocess_text,
    validate_text,
)

logger = logging.getLogger(__name__)

SENTIMENT_MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
SENTIMENT_MODEL_VERSION = "indobert-sentiment-v1"
SENTIMENT_INDEX_MAP = {0: "positive", 1: "neutral", 2: "negative"}

SARCASM_MODEL_NAME = "w11wo/indobert-base-p1-twitter-indonesia-sarcastic"
SARCASM_MODEL_VERSION = "indobert-sarcasm-v1"
SARCASM_INDEX_MAP = {0: "not_sarcastic", 1: "sarcastic"}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COMBINED_MODEL_VERSION = "combined-sentiment-sarcasm-v1"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SequenceClassifierRunner:
    """Shared HuggingFace sequence classification runner with batched inference and VRAM cleanup."""

    def __init__(self, model_name: str, index_map: dict[int, str], max_length: int = 128) -> None:
        self.model_name = model_name
        self.index_map = index_map
        self.max_length = max_length
        self.device: torch.device | None = None
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.device = get_device()
        logger.info("Loading model %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        if self.model is not None or self.tokenizer is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            release_memory()

    def predict_probs(self, valid_texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return an (N, num_labels) numpy array of softmax probabilities."""
        if not valid_texts:
            return np.empty((0, len(self.index_map)), dtype=np.float32)

        self.load()
        all_probs: list[np.ndarray] = []

        for start in range(0, len(valid_texts), batch_size):
            batch = valid_texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)


class SentimentRunner(SequenceClassifierRunner):
    """IndoBERT sentiment classifier wrapper."""

    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME, max_length: int = 128) -> None:
        super().__init__(model_name=model_name, index_map=SENTIMENT_INDEX_MAP, max_length=max_length)

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict[str, Any]]:
        cleaned_list = [preprocess_text(t) for t in texts]
        valid_indices = [i for i, t in enumerate(cleaned_list) if t]
        valid_texts = [cleaned_list[i] for i in valid_indices]

        probs_matrix = self.predict_probs(valid_texts, batch_size=batch_size)
        valid_lookup = {idx: probs_matrix[pos] for pos, idx in enumerate(valid_indices)}

        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            if i not in valid_lookup:
                results.append({
                    "sentiment": "neutral",
                    "sentiment_confidence": 0.0,
                    "positive_score": 0.0,
                    "neutral_score": 0.0,
                    "negative_score": 0.0,
                    "model_version": SENTIMENT_MODEL_VERSION,
                })
                continue

            probs = valid_lookup[i]
            pred_idx = int(np.argmax(probs))
            results.append({
                "sentiment": self.index_map[pred_idx],
                "sentiment_confidence": round(float(probs[pred_idx]), 4),
                "positive_score": round(float(probs[0]), 4),
                "neutral_score": round(float(probs[1]), 4),
                "negative_score": round(float(probs[2]), 4),
                "model_version": SENTIMENT_MODEL_VERSION,
            })
        return results


class SarcasmRunner(SequenceClassifierRunner):
    """IndoBERT sarcasm classifier wrapper."""

    def __init__(self, model_name: str = SARCASM_MODEL_NAME, max_length: int = 128) -> None:
        super().__init__(model_name=model_name, index_map=SARCASM_INDEX_MAP, max_length=max_length)

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
        confidence_threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        stripped_list = [str(t).strip() if t is not None else "" for t in texts]
        valid_indices = [i for i, t in enumerate(stripped_list) if validate_text(t)]
        valid_texts = [stripped_list[i] for i in valid_indices]

        probs_matrix = self.predict_probs(valid_texts, batch_size=batch_size)
        valid_lookup = {idx: probs_matrix[pos] for pos, idx in enumerate(valid_indices)}

        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            if i not in valid_lookup:
                results.append({
                    "sarcasm": None,
                    "is_sarcastic": False,
                    "sarcasm_confidence": 0.0,
                    "not_sarcastic_score": 0.0,
                    "sarcastic_score": 0.0,
                    "sarcasm_status": "invalid",
                    "confidence_threshold": confidence_threshold,
                    "model_version": SARCASM_MODEL_VERSION,
                })
                continue

            probs = valid_lookup[i]
            not_sarc_score = float(probs[0])
            sarc_score = float(probs[1])

            if sarc_score >= confidence_threshold:
                label, is_sarc, status, conf = "sarcastic", True, "confident", sarc_score
            elif not_sarc_score >= confidence_threshold:
                label, is_sarc, status, conf = "not_sarcastic", False, "confident", not_sarc_score
            else:
                pred_idx = int(np.argmax(probs))
                label = self.index_map[pred_idx]
                is_sarc = label == "sarcastic"
                status = "low_confidence"
                conf = float(probs[pred_idx])

            results.append({
                "sarcasm": label,
                "is_sarcastic": is_sarc,
                "sarcasm_confidence": round(conf, 4),
                "not_sarcastic_score": round(not_sarc_score, 4),
                "sarcastic_score": round(sarc_score, 4),
                "sarcasm_status": status,
                "confidence_threshold": confidence_threshold,
                "model_version": SARCASM_MODEL_VERSION,
            })
        return results


class EmbeddingRunner:
    """SentenceTransformer wrapper for multilingual semantic embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            release_memory()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        self.load()
        cleaned = [clean_text_for_embedding(t) or "empty" for t in texts]
        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def interpret_combined_result(
    sentiment: str,
    is_sarcastic: bool,
    sarcasm_status: str,
) -> tuple[str, bool]:
    """Combine sentiment label with sarcasm flag and confidence status."""
    if sarcasm_status == "invalid":
        return "invalid_text", True
    if sarcasm_status == "low_confidence":
        return "uncertain_sarcasm", True
    if is_sarcastic and sentiment == "positive":
        return "possible_sarcastic_criticism", True
    if is_sarcastic and sentiment == "negative":
        return "negative_with_sarcasm", False
    if is_sarcastic and sentiment == "neutral":
        return "neutral_with_sarcasm", True
    if not is_sarcastic and sentiment == "positive":
        return "ordinary_positive", False
    if not is_sarcastic and sentiment == "negative":
        return "ordinary_negative", False
    return "ordinary_neutral", False


def calculate_combined_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute distribution summaries across combined sentiment + sarcasm predictions."""
    total = len(results)
    if total == 0:
        return {
            "total_posts": 0,
            "sentiment_distribution": {},
            "sarcasm_distribution": {},
            "sarcasm_status_distribution": {},
            "interpretation_distribution": {},
            "confident_sarcastic_posts": 0,
            "needs_review": 0,
        }

    sentiment_dist: dict[str, int] = {}
    sarcasm_dist: dict[str, int] = {}
    status_dist: dict[str, int] = {}
    interp_dist: dict[str, int] = {}
    confident_sarcastic = 0
    needs_review_count = 0

    for item in results:
        sent = item.get("sentiment", "neutral")
        sentiment_dist[sent] = sentiment_dist.get(sent, 0) + 1

        sarc = item.get("sarcasm") or "invalid"
        sarcasm_dist[sarc] = sarcasm_dist.get(sarc, 0) + 1

        status = item.get("sarcasm_status", "invalid")
        status_dist[status] = status_dist.get(status, 0) + 1

        interp = item.get("interpretation", "ordinary_neutral")
        interp_dist[interp] = interp_dist.get(interp, 0) + 1

        if item.get("is_sarcastic") and status == "confident":
            confident_sarcastic += 1
        if item.get("needs_review"):
            needs_review_count += 1

    return {
        "total_posts": total,
        "sentiment_distribution": sentiment_dist,
        "sarcasm_distribution": sarcasm_dist,
        "sarcasm_status_distribution": status_dist,
        "interpretation_distribution": interp_dist,
        "confident_sarcastic_posts": confident_sarcastic,
        "needs_review": needs_review_count,
    }


class InferenceEngine:
    """Coordinates sentiment, sarcasm, and embedding stages sequentially to protect GPU VRAM."""

    def __init__(
        self,
        sentiment_model: str = SENTIMENT_MODEL_NAME,
        sarcasm_model: str = SARCASM_MODEL_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.sentiment_runner = SentimentRunner(model_name=sentiment_model)
        self.sarcasm_runner = SarcasmRunner(model_name=sarcasm_model)
        self.embedding_runner = EmbeddingRunner(model_name=embedding_model)

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        sentiment_batch_size: int = 16,
        sarcasm_batch_size: int = 16,
        sarcasm_threshold: float = 0.70,
        compute_embeddings: bool = True,
        embedding_batch_size: int = 32,
        unload_after_stage: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray | None]:
        """Run NLP inference on a normalized DataFrame and return enriched DataFrame, stats, and embeddings."""
        out = df.copy()
        texts = out["text"].fillna("").astype(str).tolist()

        # 1. Sentiment classification
        sent_results = self.sentiment_runner.predict_batch(texts, batch_size=sentiment_batch_size)
        if unload_after_stage:
            self.sentiment_runner.unload()

        # 2. Sarcasm classification
        sarc_results = self.sarcasm_runner.predict_batch(
            texts,
            batch_size=sarcasm_batch_size,
            confidence_threshold=sarcasm_threshold,
        )
        if unload_after_stage:
            self.sarcasm_runner.unload()

        # 3. Combined interpretation
        combined_results: list[dict[str, Any]] = []
        for idx, (s_res, c_res) in enumerate(zip(sent_results, sarc_results)):
            interp, review = interpret_combined_result(
                sentiment=s_res["sentiment"],
                is_sarcastic=c_res["is_sarcastic"],
                sarcasm_status=c_res["sarcasm_status"],
            )
            combined_results.append({
                "post_id": out.loc[idx, "id"] if "id" in out.columns else str(idx),
                "platform": out.loc[idx, "platform"] if "platform" in out.columns else "UNKNOWN",
                "timestamp": out.loc[idx, "timestamp"].isoformat() if "timestamp" in out.columns else "",
                **s_res,
                **c_res,
                "interpretation": interp,
                "needs_review": review,
                "model_version": COMBINED_MODEL_VERSION,
            })

        comb_df = pd.DataFrame(combined_results)
        for col in (
            "sentiment", "sentiment_confidence",
            "positive_score", "neutral_score", "negative_score",
            "sarcasm", "is_sarcastic", "sarcasm_confidence",
            "sarcasm_status", "interpretation", "needs_review",
        ):
            out[col] = comb_df[col]

        out["sentiment_en"] = out["sentiment"].fillna("neutral")
        out["sentiment"] = out["sentiment_en"].map(SENTIMENT_EN_TO_ID).fillna("Netral")
        out["sarcasm"] = out["sarcasm"].fillna("not_sarcastic")

        # 4. Semantic embeddings for DBSCAN clustering
        embeddings: np.ndarray | None = None
        if compute_embeddings:
            embeddings = self.embedding_runner.encode(texts, batch_size=embedding_batch_size)
            if unload_after_stage:
                self.embedding_runner.unload()

        combined_summary = {
            "total_posts": len(combined_results),
            "results": combined_results,
            "statistics": calculate_combined_statistics(combined_results),
        }
        return out, combined_summary, embeddings
