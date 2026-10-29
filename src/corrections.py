"""Learning from corrections (merchant rules) and finding low-confidence predictions."""
import pandas as pd

from clean_data import clean_merchant

LOW_CONFIDENCE = 0.60
CERTAIN = 1.0


def merchant_key(raw):
    """The same shop written differently gives the same key ("SQ *BLUE BOTTLE #12" == "blue bottle")."""
    return clean_merchant(raw)


def apply_rules(df, rules):
    """User rules win over the model and the hints."""
    df = df.copy()
    df["rule_applied"] = False
    if not rules:
        return df
    for i, merchant in df["merchant"].items():
        cat = rules.get(merchant_key(merchant))
        if cat:
            df.at[i, "predicted_category"] = cat
            df.at[i, "confidence"] = CERTAIN
            df.at[i, "rule_applied"] = True
    return df


def needs_review(df):
    """Rows the model was unsure about, least sure first. NULL (old, unscored) rows are skipped."""
    if df.empty or "confidence" not in df:
        return df.iloc[0:0]
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    out = df[conf.notna() & (conf < LOW_CONFIDENCE)]
    return out.assign(confidence=conf[out.index]).sort_values("confidence", kind="stable")


def ids_for_merchant(df, merchant):
    key = merchant_key(merchant)
    if df.empty:
        return []
    return [int(i) for i in df.loc[df["merchant"].map(merchant_key) == key, "id"]]


def describe_uncertainty(df):
    n = len(needs_review(df))
    if n == 0:
        return ""
    noun = "transaction" if n == 1 else "transactions"
    return (f"Note: {n} {noun} had a low-confidence category prediction, "
            "so category totals may be slightly off.")
