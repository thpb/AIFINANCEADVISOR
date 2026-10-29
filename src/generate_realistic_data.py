"""Generate a realistic, deliberately messy set of card transactions.

Why the data is messy on purpose:
- Real bank exports are not tidy. The same shop (Amazon, Target, Venmo ...) is
  used for groceries one day and a present the next, so the merchant name alone
  cannot tell the category. These "ambiguous" rows are what make the problem hard.
- About 4% of rows are deliberately mislabelled, like a tired person tagging
  their own spending. No model can learn these, so 100% accuracy is impossible.
- Refunds are negative amounts and subscriptions repeat every month.

Because of this, the baseline (merchant words only) and the improved model
(meaning of the name + amount + date + recurring + ambiguity) give honestly
different scores, both below 100%. That comparison is the point of the report.

Output: data/transactions.csv with columns date, merchant, amount, category.
"""
import _root  # noqa: F401

import argparse
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

DAYS = 270
OUT_PATH = "data/transactions.csv"

CLEAN_MERCHANTS = {
    "Groceries": ["TRADER JOE'S", "SAFEWAY", "WHOLE FOODS MARKET"],
    "Dining": ["BLUE BOTTLE COFFEE", "CHIPOTLE", "STARBUCKS"],
    "Transport": ["UBER", "LYFT", "SHELL OIL"],
    "Subscriptions": ["NETFLIX.COM", "SPOTIFY USA", "APPLE.COM/BILL"],
    "Entertainment": ["AMC THEATRES", "STEAM GAMES", "TICKETMASTER"],
    "Bills": ["PG&E ONLINE PMT", "COMCAST CABLE", "STATE FARM INSURANCE"],
}
AMBIGUOUS_MERCHANTS = {   # same merchant string, genuinely different categories
    "AMAZON.COM": ["Shopping", "Groceries", "Entertainment", "Other"],
    "TARGET": ["Shopping", "Groceries", "Other"],
    "WALMART.COM": ["Shopping", "Groceries"],
    "COSTCO WHOLESALE": ["Groceries", "Shopping"],
    "VENMO PAYMENT": ["Other", "Dining", "Shopping"],
    "PAYPAL": ["Shopping", "Subscriptions", "Other"],
}
AMOUNT_RANGES = {"Groceries": (15, 140), "Dining": (6, 60), "Transport": (8, 70),
                 "Subscriptions": (5, 20), "Entertainment": (10, 90), "Bills": (40, 220),
                 "Shopping": (10, 250), "Other": (10, 300)}
RECURRING = [("NETFLIX.COM", 15.49), ("SPOTIFY USA", 11.99), ("APPLE.COM/BILL", 2.99)]
CATEGORIES = sorted(AMOUNT_RANGES)

# Small decorations real bank feeds add to merchant names.
PREFIXES = ["", "", "", "SQ *", "TST* "]


def _random_day(start):
    return start + timedelta(days=random.randint(0, DAYS - 1))


def _amount(category):
    low, high = AMOUNT_RANGES[category]
    return round(random.uniform(low, high), 2)


def _decorate(name):
    """Sometimes add a store number, used for ambiguous shops only."""
    if random.random() < 0.3:
        return f"{name} #{random.randint(100, 9999)}"
    return name


def generate(end=None):
    today = end or date.today()
    start = today - timedelta(days=DAYS - 1)
    rows = []

    # Monthly subscriptions: fixed amount, every 30 days.
    for name, price in RECURRING:
        d = start + timedelta(days=random.randint(0, 29))
        while d <= today:
            rows.append((d, name, price, "Subscriptions"))
            d += timedelta(days=30)

    # Clean rows: merchant names written exactly the same each time.
    clean_cats = list(CLEAN_MERCHANTS)
    for _ in range(380):
        cat = random.choice(clean_cats)
        rows.append((_random_day(start), random.choice(CLEAN_MERCHANTS[cat]), _amount(cat), cat))

    # Ambiguous rows: the same merchant in different categories.
    for _ in range(180):
        name = random.choice(list(AMBIGUOUS_MERCHANTS))
        cat = random.choice(AMBIGUOUS_MERCHANTS[name])
        prefix = "" if name.startswith("VENMO") else random.choice(PREFIXES)
        rows.append((_random_day(start), prefix + _decorate(name), _amount(cat), cat))

    # Refunds: negative amounts from shops people return things to.
    for _ in range(25):
        name = random.choice(["AMAZON.COM", "TARGET", "WALMART.COM", "COSTCO WHOLESALE"])
        cat = random.choice(AMBIGUOUS_MERCHANTS[name])
        rows.append((_random_day(start), name, -_amount(cat), cat))

    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "category"])

    # Label noise: about 4% of rows get a wrong category on purpose.
    noisy = np.random.choice(df.index, size=20, replace=False)
    for i in noisy:
        others = [c for c in CATEGORIES if c != df.at[i, "category"]]
        df.at[i, "category"] = random.choice(others)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="Last day of the data (YYYY-MM-DD). Default: today. "
                             "Use 2026-10-29 to reproduce the reported results.")
    df = generate(parser.parse_args().end)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} transactions to {OUT_PATH}")
    print(f"From {df['date'].min()} to {df['date'].max()}")
    print(f"Refunds: {(df['amount'] < 0).sum()}  Mislabelled on purpose: 20")
    print("\nRows per category:")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
