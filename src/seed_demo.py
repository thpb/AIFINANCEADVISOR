"""Create a demo account filled with the generated data, for the live demo.

Login: demo / demo123 (a test account on this computer only).
Run after generate_realistic_data.py and train_improved_v2.py. Safe to run again:
it wipes and refills only the demo account.
"""
import _root  # noqa: F401

import pandas as pd

import auth
import categorizer
import db
from table_edits import as_date_text

USERNAME, PASSWORD = "demo", "demo123"
BUDGETS = {"Groceries": 1000, "Dining": 400, "Shopping": 1100, "Transport": 450, "Entertainment": 450}


def main():
    db.init_db()
    if not db.get_user_by_username(USERNAME):
        auth.signup(USERNAME, PASSWORD)
    uid = db.get_user_by_username(USERNAME)["id"]
    db.delete_all_transactions(uid)
    db.delete_all_income(uid)

    df = pd.read_csv("data/transactions.csv")[["date", "merchant", "amount"]]
    embedder, clf, le = categorizer.load_models()
    out = categorizer.categorize(df, embedder, clf, le, db.get_rules(uid))
    out["date"] = out["date"].map(as_date_text)
    n = db.insert_transactions(uid, out, source="upload")

    # A fortnightly salary across the same months.
    start = pd.to_datetime(df["date"]).min()
    for d in pd.date_range(start, pd.Timestamp.today(), freq="14D"):
        db.add_income(uid, as_date_text(d), "salary", 2400.0)
    for cat, limit in BUDGETS.items():
        db.set_budget(uid, cat, limit)
    print(f"Demo account ready: {n} transactions, income and budgets. Log in as {USERNAME} / {PASSWORD}")


if __name__ == "__main__":
    main()
