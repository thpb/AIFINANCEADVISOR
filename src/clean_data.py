"""Cleaning merchant names and adding the features the models use."""
import re

import pandas as pd


def clean_merchant(raw: str) -> str:
    s = str(raw).upper()
    s = re.sub(r"^(SQ \*|SP \*|TST\*|DOORDASH\*|EFTPOS |VISA PURCHASE )", "", s)   # processor prefixes
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


MAX_AMOUNT = 1_000_000


def parse_user_dates(values):
    """CSV dates: ISO (2026-10-05) or day-first (05/10/2026). Unreadable dates become NaT."""
    text = values.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(text[iso], format="mixed", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(text[~iso], dayfirst=True, format="mixed", errors="coerce")
    return out


def read_bank_csv(source):
    """Read a bank CSV with date, merchant and amount columns.

    Returns (rows, skipped): rows that could be read, and how many couldn't (unreadable date or amount,
    no merchant, or over $1,000,000). Raises ValueError with a plain message when nothing can be used.
    """
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise ValueError("That file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise ValueError("That file doesn't look like a CSV. Save it from your bank or spreadsheet as CSV.") from None
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = {"date", "merchant", "amount"} - set(raw.columns)
    if missing:
        raise ValueError(f"The file is missing these columns: {', '.join(sorted(missing))}. "
                         "It needs date, merchant and amount.")
    df = raw[["date", "merchant", "amount"]].copy()
    df["merchant"] = df["merchant"].fillna("").astype(str).str.strip()
    df["date"] = parse_user_dates(df["date"])
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    ok = df["date"].notna() & df["amount"].notna() & (df["merchant"] != "") & (df["amount"].abs() <= MAX_AMOUNT)
    rows = df[ok].reset_index(drop=True)
    if rows.empty:
        raise ValueError("No rows could be read. Check that dates look like 05/10/2026 and amounts are numbers.")
    return rows, int((~ok).sum())


def load_and_clean(path="data/transactions.csv") -> pd.DataFrame:
    return add_features(pd.read_csv(path))
