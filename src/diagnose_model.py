"""Per-category report and confusion matrix for the v2 model (for the report).

Shows where the model goes wrong: mostly the ambiguous shops, which is expected.
"""
import _root  # noqa: F401

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix

from clean_data import load_and_clean
from train_baseline import time_split
from train_improved import EMBED_MODEL, build_matrix
from train_improved_v2 import NUMERIC_V2, add_ambiguity_feature


def main():
    df = add_ambiguity_feature(load_and_clean("data/transactions.csv"))
    _, test = time_split(df)
    clf = joblib.load("data/improved_model_v2.joblib")
    le = joblib.load("data/label_encoder_v2.joblib")
    embedder = SentenceTransformer(EMBED_MODEL)

    proba = clf.predict_proba(build_matrix(test, embedder, NUMERIC_V2))
    test = test.assign(pred=le.inverse_transform(proba.argmax(axis=1)), confidence=proba.max(axis=1))

    print(classification_report(test["category"], test["pred"], zero_division=0))
    cm = confusion_matrix(test["category"], test["pred"], labels=le.classes_)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    test["correct"] = test["category"] == test["pred"]
    print("\nAccuracy: ambiguous shops vs clean names")
    print(test.groupby("is_ambiguous_merchant")["correct"].mean().rename({0: "clean", 1: "ambiguous"}).to_string())
    print("\nAverage confidence when right vs wrong")
    print(test.groupby("correct")["confidence"].mean().to_string())
    wrong = test.loc[~test["correct"], ["merchant", "amount", "category", "pred", "confidence"]]
    print(f"\nMistakes ({len(wrong)}):")
    print(wrong.to_string(index=False))


if __name__ == "__main__":
    main()
