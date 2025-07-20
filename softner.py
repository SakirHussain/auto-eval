"""
softener_model.py
─────────────────
• Trains / (re‑)trains a Gradient‑Boosting Regressor that
  “softens” ProCoT scores so they align better with human marks.
• Persists the model to disk (joblib) and exposes a
  `predict_softened_score` helper you can import from Streamlit.
"""

import json
import math
import pathlib
from typing import List

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Local imports
import config

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
DATA_PATH   = config.DATA_PATH
MODEL_PATH  = pathlib.Path(config.MODEL_PATH)
EMBED_MODEL = config.EMBEDDING_MODEL

# ────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────
_sbert = SentenceTransformer(EMBED_MODEL)


def _sbert_sim(a: str, b: str) -> float:
    emb1, emb2 = _sbert.encode([a, b])
    return float(cosine_similarity([emb1], [emb2])[0][0])


def _tfidf_sim(a: str, b: str) -> float:
    tfidf = TfidfVectorizer().fit_transform([a, b])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


def _extract_features(
    procot_score: float,
    student_answer: str,
    ideal_answer: str,
) -> np.ndarray:
    return np.array(
        [
            procot_score,
            _sbert_sim(student_answer, ideal_answer),
            _tfidf_sim(student_answer, ideal_answer),
            len(student_answer.split()),
        ],
        dtype=float,
    )


# ────────────────────────────────────────────────────────────
# Training routine (run once or when you have new data)
# ────────────────────────────────────────────────────────────
def train_and_save():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["human_avg_score"] = df["human_scores"].apply(lambda xs: sum(xs) / len(xs))

    X = np.vstack(
        [
            _extract_features(r["procot_score"], r["student_answer"], r["ideal_answer"])
            for r in data
        ]
    )
    y = df["human_avg_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✔ Softener model saved → {MODEL_PATH.resolve()}")


# ────────────────────────────────────────────────────────────
# Public helper for inference
# ────────────────────────────────────────────────────────────
def predict_softened_score(
    procot_score: float,
    student_answer: str,
    ideal_answer: str,
) -> float:
    """
    Load the trained model (if not already in memory) and
    return the rounded‑up softened score.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_and_save() first."
        )

    model: GradientBoostingRegressor = joblib.load(MODEL_PATH)
    feats = _extract_features(procot_score, student_answer, ideal_answer).reshape(1, -1)
    softened = math.ceil(model.predict(feats)[0])
    return softened


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
