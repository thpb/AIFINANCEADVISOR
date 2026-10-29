"""Cleaning merchant names and adding the features the models use."""
import re

import pandas as pd


def clean_merchant(raw: str) -> str:
    s = str(raw).upper()
    s = re.sub(r"^(SQ \*|SP \*|TST\*|DOORDASH\*)", "", s)   # processor prefixes
    s = re.sub(r"#\d+", "", s)                               # store numbers
    s = re.sub(r"\b\d{4,}\b", "", s)                         # long numeric codes
    s = re.sub(r"[^A-Z0-9&' ]", " ", s)                      # punctuation noise
    return re.sub(r"\s+", " ", s).strip()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["merchant_clean"] = df["merchant"].map(clean_merchant)
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    counts = df.groupby(["merchant_clean", "amount"])["amount"].transform("size")
    df["is_recurring"] = (counts >= 2).astype(int)
    return df


def load_and_clean(path="data/transactions.csv") -> pd.DataFrame:
    return add_features(pd.read_csv(path))
