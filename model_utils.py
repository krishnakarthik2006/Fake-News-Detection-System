from __future__ import annotations

import csv
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_CACHE = ROOT / "trained_model.pkl"
TRAINING_FILES = ("train.csv", "valid.csv")
TEST_FILE = "test.csv"
LABELS = ("TRUE", "FALSE")
TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")
ALPHA = 1.0
CACHE_VERSION = 2


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class DatasetRow:
    statement: str
    label: str


@dataclass(frozen=True)
class TrainedModel:
    signature: tuple[tuple[str, int, int], ...]
    labels: tuple[str, ...]
    alpha: float
    doc_counts: dict[str, int]
    token_counts: dict[str, dict[str, int]]
    vocabulary_size: int
    log_priors: dict[str, float]
    log_denominators: dict[str, float]

    def predict_probabilities(self, feature_counts: Counter[str]) -> dict[str, float]:
        log_scores: dict[str, float] = {}

        for label in self.labels:
            label_token_counts = self.token_counts[label]
            log_probability = self.log_priors[label]
            log_denominator = self.log_denominators[label]

            for feature, count in feature_counts.items():
                numerator = label_token_counts.get(feature, 0) + self.alpha
                log_probability += count * math.log(numerator)
                log_probability -= count * log_denominator

            log_scores[label] = log_probability

        return _normalize_log_scores(log_scores)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def featurize(text: str) -> list[str]:
    tokens = tokenize(text)
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def read_dataset(*filenames: str) -> list[tuple[str, str]]:
    signature = build_signature(*filenames)
    return [(row.statement, row.label) for row in _read_dataset_cached(signature)]


@lru_cache(maxsize=8)
def _read_dataset_cached(signature: tuple[tuple[str, int, int], ...]) -> tuple[DatasetRow, ...]:
    rows: list[DatasetRow] = []

    for filename, _, _ in signature:
        path = ROOT / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                statement = (row.get("Statement") or "").strip()
                label = (row.get("Label") or "").strip().upper()
                if statement and label in LABELS:
                    rows.append(DatasetRow(statement=statement, label=label))

    if not rows:
        raise RuntimeError("No training rows were loaded from the dataset files.")

    return tuple(rows)


def build_signature(*filenames: str) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for filename in filenames:
        path = ROOT / filename
        stat = path.stat()
        signature.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def train_model(signature: tuple[tuple[str, int, int], ...] | None = None) -> TrainedModel:
    signature = signature or build_signature(*TRAINING_FILES)
    samples = read_dataset(*TRAINING_FILES)
    doc_counts = Counter()
    token_counts = {label: Counter() for label in LABELS}
    vocabulary: set[str] = set()

    for statement, label in samples:
        features = featurize(statement)
        if not features:
            continue

        doc_counts[label] += 1
        token_counts[label].update(features)
        vocabulary.update(features)

    total_docs = sum(doc_counts.values())
    if total_docs == 0 or not vocabulary:
        raise RuntimeError("Training data did not produce a usable model.")

    vocabulary_size = len(vocabulary)
    token_counts_dict = {label: dict(counts) for label, counts in token_counts.items()}
    log_priors = {label: math.log(doc_counts[label] / total_docs) for label in LABELS}
    log_denominators = {
        label: math.log(sum(token_counts[label].values()) + ALPHA * vocabulary_size)
        for label in LABELS
    }

    model = TrainedModel(
        signature=signature,
        labels=LABELS,
        alpha=ALPHA,
        doc_counts=dict(doc_counts),
        token_counts=token_counts_dict,
        vocabulary_size=vocabulary_size,
        log_priors=log_priors,
        log_denominators=log_denominators,
    )

    with MODEL_CACHE.open("wb") as handle:
        pickle.dump(_serialize_model(model), handle)

    return model


def _serialize_model(model: TrainedModel) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "signature": model.signature,
        "labels": model.labels,
        "alpha": model.alpha,
        "doc_counts": model.doc_counts,
        "token_counts": model.token_counts,
        "vocabulary_size": model.vocabulary_size,
        "log_priors": model.log_priors,
        "log_denominators": model.log_denominators,
    }


def _coerce_cached_model(payload: Any) -> TrainedModel:
    if isinstance(payload, TrainedModel):
        return payload

    if not isinstance(payload, dict):
        raise TypeError("Unsupported model cache payload.")

    signature = tuple(tuple(item) for item in payload["signature"])
    labels = tuple(payload["labels"])
    alpha = float(payload["alpha"])
    doc_counts = {label: int(count) for label, count in payload["doc_counts"].items()}
    token_counts = {
        label: {feature: int(count) for feature, count in counts.items()}
        for label, counts in payload["token_counts"].items()
    }

    if "vocabulary_size" in payload and "log_priors" in payload and "log_denominators" in payload:
        return TrainedModel(
            signature=signature,
            labels=labels,
            alpha=alpha,
            doc_counts=doc_counts,
            token_counts=token_counts,
            vocabulary_size=int(payload["vocabulary_size"]),
            log_priors={label: float(score) for label, score in payload["log_priors"].items()},
            log_denominators={label: float(score) for label, score in payload["log_denominators"].items()},
        )

    # Backward compatibility for the older plain-dict cache shape.
    if "vocabulary" in payload and "total_docs" in payload and "total_token_counts" in payload:
        vocabulary_size = len(payload["vocabulary"])
        total_docs = int(payload["total_docs"])
        log_priors = {label: math.log(doc_counts[label] / total_docs) for label in labels}
        log_denominators = {
            label: math.log(int(payload["total_token_counts"][label]) + alpha * vocabulary_size)
            for label in labels
        }
        return TrainedModel(
            signature=signature,
            labels=labels,
            alpha=alpha,
            doc_counts=doc_counts,
            token_counts=token_counts,
            vocabulary_size=vocabulary_size,
            log_priors=log_priors,
            log_denominators=log_denominators,
        )

    raise KeyError("Model cache payload is missing required fields.")


def _read_cached_model() -> TrainedModel | None:
    if not MODEL_CACHE.is_file():
        return None

    try:
        with MODEL_CACHE.open("rb") as handle:
            payload = pickle.load(handle)
        return _coerce_cached_model(payload)
    except (AttributeError, EOFError, KeyError, ModuleNotFoundError, TypeError, ValueError, pickle.PickleError):
        return None


@lru_cache(maxsize=4)
def _load_model_for_signature(signature: tuple[tuple[str, int, int], ...]) -> TrainedModel:
    model = _read_cached_model()
    if model is not None:
        if model.signature == signature:
            return model

    return train_model(signature)


def load_or_train_model(force_retrain: bool = False) -> TrainedModel:
    signature = build_signature(*TRAINING_FILES)

    if force_retrain:
        _load_model_for_signature.cache_clear()
        return train_model(signature)

    return _load_model_for_signature(signature)


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    highest = max(log_scores.values())
    scaled = {label: math.exp(score - highest) for label, score in log_scores.items()}
    total = sum(scaled.values()) or 1.0
    return {label: value / total for label, value in scaled.items()}


def predict_text(text: str, model: TrainedModel | None = None) -> PredictionResult:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Please enter some news text first.")

    model = model or load_or_train_model()
    features = featurize(cleaned)
    if not features:
        raise ValueError("Please enter a longer sentence containing words.")

    feature_counts = Counter(features)
    probabilities = model.predict_probabilities(feature_counts)
    predicted_label = max(probabilities, key=probabilities.get)
    return PredictionResult(
        label=predicted_label,
        confidence=probabilities[predicted_label],
        probabilities=probabilities,
    )


def evaluate_model(model: TrainedModel | None = None) -> dict[str, float]:
    model = model or load_or_train_model()
    samples = read_dataset(TEST_FILE)
    correct = sum(1 for statement, label in samples if predict_text(statement, model).label == label)

    accuracy = correct / len(samples)
    return {"samples": len(samples), "accuracy": accuracy}


def format_label(label: str) -> str:
    return "Real" if label == "TRUE" else "Fake"
