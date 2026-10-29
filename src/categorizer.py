"""The categorization pipeline the app uses.

Order for every transaction, later steps win:
1. the model (embeddings + LightGBM) gives a category and how sure it is,
2. keyword hints fix plain words and Australian shops,
3. the user's own rules (learned from corrections) always win.
"""
import os

import joblib
import numpy as np

import category_hints
import corrections
from clean_data import add_features
from train_improved_v2 import NUMERIC_V2, add_ambiguity_feature

MODEL_PATH = "data/improved_model_v2.joblib"
ENCODER_PATH = "data/label_encoder_v2.joblib"
EMBED_MODEL = "all-MiniLM-L6-v2"
FEATURE_COLS = NUMERIC_V2


def models_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)


def load_models():
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    return embedder, joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def categorize(df, embedder, clf, le, rules=None):
    df = add_ambiguity_feature(add_features(df))
    if df.empty:
        for col in ["model_category", "predicted_category", "hint_word"]:
            df[col] = []
        df["confidence"] = []
        df["model_confidence"] = []
        df["rule_applied"] = []
        return df
    emb = np.asarray(embedder.encode(df["merchant_clean"].tolist(), show_progress_bar=False))
    x = np.hstack([emb, df[FEATURE_COLS].to_numpy(dtype=float)])
    proba = clf.predict_proba(x)
    labels = le.inverse_transform(proba.argmax(axis=1))
    df["model_category"] = labels
    df["predicted_category"] = labels
    df["confidence"] = proba.max(axis=1).astype(float)
    df["model_confidence"] = df["confidence"]
    df = category_hints.apply_hints(df, known_categories=le.classes_)
    return corrections.apply_rules(df, rules or {})
