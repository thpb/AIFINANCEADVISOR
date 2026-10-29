"""Improved model v1: MiniLM sentence embeddings + numeric features + LightGBM.

The embedding gives the model the *meaning* of a merchant name (384 numbers),
so it can make a sensible guess for names it has never seen. Amount and date
features help with shops that sell many kinds of things. Settings are close to
LightGBM's defaults; train_improved_v2.py tunes them.
"""
import _root  # noqa: F401

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from clean_data import load_and_clean
from train_baseline import time_split

EMBED_MODEL = "all-MiniLM-L6-v2"
NUMERIC = ["amount", "day_of_week", "day_of_month", "month", "is_recurring"]


def build_matrix(df, embedder, numeric):
    emb = embedder.encode(df["merchant_clean"].tolist(), show_progress_bar=False)
    return np.hstack([emb, df[numeric].to_numpy(dtype=float)])


def main():
    df = load_and_clean("data/transactions.csv")
    train, test = time_split(df)
    embedder = SentenceTransformer(EMBED_MODEL)

    le = LabelEncoder().fit(df["category"])
    clf = LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(build_matrix(train, embedder, NUMERIC), le.transform(train["category"]))
    pred = le.inverse_transform(clf.predict(build_matrix(test, embedder, NUMERIC)))

    print(f"Improved v1 accuracy: {accuracy_score(test['category'], pred):.1%}\n")
    print(classification_report(test["category"], pred, zero_division=0))
    joblib.dump(clf, "data/improved_model.joblib")
    joblib.dump(le, "data/label_encoder.joblib")
    print("Saved data/improved_model.joblib and data/label_encoder.joblib")


if __name__ == "__main__":
    main()
