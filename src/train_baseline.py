"""Baseline model: TF-IDF on the cleaned merchant name + Logistic Regression.

Uses a time-based split (first 80% of dates to train, newest 20% to test),
which is more realistic than a random split: the app predicts new spending
from old spending.
"""
import _root  # noqa: F401

import argparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from clean_data import load_and_clean


def data_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/transactions.csv",
                        help="Training CSV (default data/transactions.csv; data/transactions_v2.csv for the larger set)")
    return parser.parse_args().data


def time_split(df, train_share=0.8):
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    cut = int(len(df) * train_share)
    return df.iloc[:cut], df.iloc[cut:]


def main():
    path = data_path()
    print(f"Data: {path}")
    df = load_and_clean(path)
    train, test = time_split(df)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform(train["merchant_clean"])
    x_test = vectorizer.transform(test["merchant_clean"])

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train["category"])
    pred = model.predict(x_test)

    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Baseline accuracy: {accuracy_score(test['category'], pred):.1%}\n")
    print(classification_report(test["category"], pred, zero_division=0))
    labels = sorted(df["category"].unique())
    print("Confusion matrix (rows = true, columns = predicted):")
    print(labels)
    print(confusion_matrix(test["category"], pred, labels=labels))

    joblib.dump(vectorizer, "data/baseline_vectorizer.joblib")
    joblib.dump(model, "data/baseline_model.joblib")
    print("\nSaved data/baseline_vectorizer.joblib and data/baseline_model.joblib")


if __name__ == "__main__":
    main()
