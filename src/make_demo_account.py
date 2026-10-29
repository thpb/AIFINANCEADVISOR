"""Create the demo account for the live demo: log in as demo / demo1234.

- Loads data/transactions_v2.csv (or data/transactions.csv if it doesn't exist) through the real
  categorization pipeline: the model, the keyword hints and the user's rules.
- Moves the dates forward so the data ends today, whatever day the demo is.
- The v2 file has about 4,000 rows, far more than one person spends. The demo keeps every rent, bill
  and subscription payment and a fixed 15% sample of everyday spending, so the totals look like a
  real person's.
- Sets budgets for Groceries, Dining, Shopping and Transport, with Dining close to its limit this month
  (about 90% used) and the others comfortably on track.
- Adds 6 months of fortnightly salary, sized so the demo person saves a little each month.

Safe to run again: it deletes the old demo account first, so nothing is duplicated.
This is a test account that only exists in data/app.db on this computer.
"""
import _root  # noqa: F401

import math
import os
import random
from datetime import date

import pandas as pd

import auth
import categorizer
import db
from generate_more_data import MONTHLY
from table_edits import as_date_text

USERNAME, PASSWORD = "demo", "demo1234"
DATA_FILES = ["data/transactions_v2.csv", "data/transactions.csv"]
EVERYDAY_SHARE = 0.15
NEAR_LIMIT = "Dining"
BUDGET_CATEGORIES = ["Groceries", "Dining", "Shopping", "Transport"]
SALARY_MONTHS = 6
SAVE_SHARE = 0.12          # salary is set so about 12% of income is saved


def load_data(today):
    path = next((p for p in DATA_FILES if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("No data yet. Run python src/generate_more_data.py (or generate_realistic_data.py) first.")
    df = pd.read_csv(path)[["date", "merchant", "amount"]]
    if path.endswith("_v2.csv"):
        fixed = df["merchant"].isin({name for name, *_ in MONTHLY})
        rng = random.Random(7)
        keep = [f or rng.random() < EVERYDAY_SHARE for f in fixed]
        df = df[keep]
    dates = pd.to_datetime(df["date"])
    df = df.assign(date=dates + (pd.Timestamp(today) - dates.max()))      # the data ends today
    return path, df.reset_index(drop=True)


def round_up(value, step):
    return int(math.ceil(value / step) * step)


def pick_budgets(spend, today):
    """Dining about 90% used this month; the others at 125% of a normal month, never close to the limit."""
    month = spend["date"].dt.strftime("%Y-%m")
    this_month = spend[month == today.strftime("%Y-%m")].groupby("predicted_category")["amount"].sum()
    full_months = spend[month < today.strftime("%Y-%m")]
    n_months = max(full_months["date"].dt.strftime("%Y-%m").nunique(), 1)
    average = full_months.groupby("predicted_category")["amount"].sum() / n_months
    budgets = {}
    for cat in BUDGET_CATEGORIES:
        spent = float(this_month.get(cat, 0.0))
        if cat == NEAR_LIMIT and spent > 0:
            budgets[cat] = round_up(spent / 0.90, 10)
        else:
            budgets[cat] = max(round_up(float(average.get(cat, 0.0)) * 1.25, 50), round_up(spent / 0.6, 50), 50)
    return budgets


def make_demo_account(today=None, models=None):
    today = today or date.today()
    db.init_db()
    old = db.get_user_by_username(USERNAME)
    if old:
        db.delete_user(old["id"])
    ok, message = auth.signup(USERNAME, PASSWORD)
    if not ok:
        raise SystemExit(message)
    uid = db.get_user_by_username(USERNAME)["id"]

    path, df = load_data(today)
    embedder, clf, le = models or categorizer.load_models()
    out = categorizer.categorize(df, embedder, clf, le, db.get_rules(uid))
    out["date"] = out["date"].map(as_date_text)
    n = db.insert_transactions(uid, out, source="upload")

    spend = db.get_transactions(uid)
    spend["date"] = pd.to_datetime(spend["date"], format="mixed")
    for cat, limit in pick_budgets(spend, today).items():
        db.set_budget(uid, cat, limit)

    recent = spend[spend["date"] >= pd.Timestamp(today) - pd.DateOffset(months=SALARY_MONTHS)]
    monthly_spend = recent["amount"].sum() / SALARY_MONTHS
    pay = round_up(monthly_spend / (1 - SAVE_SHARE) * 12 / 26, 10)        # fortnightly
    start = pd.Timestamp(today) - pd.DateOffset(months=SALARY_MONTHS)
    paydays = pd.date_range(start, pd.Timestamp(today), freq="14D")
    for d in paydays:
        db.add_income(uid, as_date_text(d), "salary", pay)
    return {"user_id": uid, "file": path, "transactions": n, "paydays": len(paydays), "pay": pay,
            "budgets": db.get_budgets(uid)}


def main():
    info = make_demo_account()
    print(f"Demo account ready: log in as {USERNAME} / {PASSWORD}")
    print(f"- {info['transactions']} transactions from {info['file']}, ending today")
    print(f"- {info['paydays']} fortnightly salary payments of ${info['pay']:,}")
    print(f"- Budgets: {', '.join(f'{c} ${v:,.0f}' for c, v in info['budgets'].items())} "
          f"({NEAR_LIMIT} is close to its limit this month)")


if __name__ == "__main__":
    main()
