"""Improved model v2 (the one the app uses): embeddings + LightGBM, tuned.

Three improvements over v1:
1. class_weight="balanced": rare categories (Other, Entertainment) count as
   much as common ones, so the model stops ignoring them.
2. Smaller trees for a small dataset (num_leaves=15, max_depth=6,
   min_child_samples=5): fewer leaves overfit less, and a lower minimum lets
   small categories still form their own leaves.
3. is_ambiguous_merchant: a flag for shops known to sell everything (Amazon,
   Target ...). It tells the model "lean on amount and date here, not the name".
"""
import _root  # noqa: F401

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from clean_data import load_and_clean
from train_baseline import data_path, time_split
from train_improved import EMBED_MODEL, build_matrix

AMBIGUOUS_KEYWORDS = ["AMAZON", "TARGET", "WALMART", "COSTCO", "VENMO", "PAYPAL"]
NUMERIC_V2 = ["amount", "day_of_week", "day_of_month", "month", "is_recurring", "is_ambiguous_merchant"]


def add_ambiguity_feature(df):
    """1 if the cleaned merchant contains any ambiguous keyword, else 0."""
    df = df.copy()
    pattern = "|".join(AMBIGUOUS_KEYWORDS)
    df["is_ambiguous_merchant"] = df["merchant_clean"].str.contains(pattern, regex=True).astype(int)
    return df


def make_classifier():
    return LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=15, min_child_samples=5,
                          class_weight="balanced", random_state=42, verbose=-1)


def main():
    path = data_path()
    print(f"Data: {path}")
    df = add_ambiguity_feature(load_and_clean(path))
    train, test = time_split(df)
    embedder = SentenceTransformer(EMBED_MODEL)

    le = LabelEncoder().fit(df["category"])
    clf = make_classifier()
    clf.fit(build_matrix(train, embedder, NUMERIC_V2), le.transform(train["category"]))
    pred = le.inverse_transform(clf.predict(build_matrix(test, embedder, NUMERIC_V2)))

    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Improved v2 accuracy: {accuracy_score(test['category'], pred):.1%}\n")
    print(classification_report(test["category"], pred, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(le.classes_.tolist())
    print(confusion_matrix(test["category"], pred, labels=le.classes_))

    names = [f"emb_{i}" for i in range(384)] + NUMERIC_V2
    imp = pd.Series(clf.feature_importances_, index=names)
    print("\nFeature importance (numeric features + all embedding values together):")
    print(imp[NUMERIC_V2].sort_values(ascending=False).to_string())
    print(f"embedding (384 values, total) {int(np.sum(imp.iloc[:384]))}")

    joblib.dump(clf, "data/improved_model_v2.joblib")
    joblib.dump(le, "data/label_encoder_v2.joblib")
    print("\nSaved data/improved_model_v2.joblib and data/label_encoder_v2.joblib")


if __name__ == "__main__":
    main()
