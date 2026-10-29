"""Generate a larger, more Australian set of transactions: data/transactions_v2.csv.

The original generator (generate_realistic_data.py) stays untouched because the
report's first results come from it. This one adds what the original lacks:

- Australian merchants for every category, written the way Australian bank
  statements show them: store numbers ("WOOLWORTHS 1234"), suburbs and state
  codes ("COLES 0456 PARRAMATTA NS"), processor prefixes ("SQ *", "EFTPOS ",
  "VISA PURCHASE ") and mixed casing. About half the rows stay clean.
- About 5% of rows typed the way quick add receives them ("coffee", "uber home").
- Realistic rhythm: rent and bills monthly, subscriptions monthly at a fixed
  price, groceries weekly, more dining on weekends.

The hard parts that keep the results honest stay in: shops that sell everything
(Amazon, Target, Costco, PayPal, bank transfers), about 4% wrong labels on
purpose, and refunds (negative amounts). All US merchants from the original
generator are kept too.

Same columns (date, merchant, amount, category) and the same 8 categories.
Fixed seed, so the same --end date always gives the same file.
"""
import _root  # noqa: F401

import argparse
import random
from datetime import date, timedelta

import pandas as pd

from generate_realistic_data import AMBIGUOUS_MERCHANTS as US_AMBIGUOUS
from generate_realistic_data import AMOUNT_RANGES, CATEGORIES
from generate_realistic_data import CLEAN_MERCHANTS as US_MERCHANTS

SEED = 2026
DAYS = 365
TARGET_ROWS = 4000
OUT_PATH = "data/transactions_v2.csv"

AU_MERCHANTS = {
    "Groceries": ["WOOLWORTHS", "COLES", "ALDI", "IGA", "HARRIS FARM MARKETS", "WOOLWORTHS METRO", "COLES EXPRESS"],
    "Dining": ["MCDONALDS", "HUNGRY JACKS", "GUZMAN Y GOMEZ", "MENULOG", "UBER EATS", "KFC", "OPORTO",
               "BETTYS BURGERS", "THE GROUNDS CAFE", "BAY LEAF CAFE", "BEAN THERE ESPRESSO", "SUSHI HUB"],
    "Transport": ["TRANSPORTNSW OPAL", "MYKI", "UBER *TRIP", "DIDI", "BP", "AMPOL", "SHELL", "LINKT TOLL",
                  "WILSON PARKING", "13CABS"],
    "Bills": ["AGL ENERGY", "ORIGIN ENERGY", "TELSTRA", "OPTUS", "SYDNEY WATER", "ALLIANZ INSURANCE"],
    "Subscriptions": ["NETFLIX.COM", "SPOTIFY", "STAN", "BINGE", "DISNEY PLUS", "APPLE.COM/BILL", "ICLOUD"],
    "Entertainment": ["HOYTS", "EVENT CINEMAS", "TICKETEK", "STEAM PURCHASE", "PLAYSTATION NETWORK",
                      "TICKETMASTER AU", "TIMEZONE"],
    "Shopping": ["KMART", "BIG W", "JB HI-FI", "OFFICEWORKS", "BUNNINGS", "MYER", "UNIQLO", "DAVID JONES",
                 "REBEL SPORT"],
    "Other": ["CHEMIST WAREHOUSE", "AUSTRALIA POST", "JUST CUTS", "SPECSAVERS", "PRICELINE PHARMACY",
              "GREENCROSS VETS"],
}
AU_AMBIGUOUS = {
    "AMAZON AU": ["Shopping", "Groceries", "Entertainment", "Other"],
    "AMAZON MKTPLACE AU": ["Shopping", "Other"],
    "TARGET AUSTRALIA": ["Shopping", "Groceries", "Other"],
    "COSTCO WHOLESALE AU": ["Groceries", "Shopping"],
    "PAYPAL *": ["Shopping", "Subscriptions", "Entertainment", "Other"],
    "TRANSFER TO J SMITH": ["Other", "Dining", "Bills"],
    "OSKO PAYMENT": ["Other", "Dining", "Bills", "Shopping"],
}
TYPED = {
    "Groceries": ["groceries", "grocery shopping", "supermarket", "fruit and veg"],
    "Dining": ["coffee", "lunch", "dinner", "brunch", "takeaway", "pizza", "coffee with sam", "sushi"],
    "Transport": ["petrol", "fuel", "uber home", "uber to work", "train", "bus fare", "parking", "taxi"],
    "Bills": ["rent", "electricity bill", "phone bill", "internet", "car insurance"],
    "Subscriptions": ["netflix", "spotify", "gym membership", "icloud storage"],
    "Entertainment": ["movie tickets", "concert tickets", "cinema", "new game", "bowling"],
    "Shopping": ["clothes", "shoes", "new headphones", "birthday present", "phone case"],
    "Other": ["haircut", "doctor", "pharmacy", "gift for mum", "vet"],
}
SUBURBS = [("PARRAMATTA", "NS"), ("BONDI JUNCTION", "NS"), ("CHATSWOOD", "NS"), ("NEWTOWN", "NS"),
           ("MELBOURNE", "VI"), ("RICHMOND", "VI"), ("FITZROY", "VI"), ("BRISBANE", "QL"),
           ("FORTITUDE VALLEY", "QL"), ("ADELAIDE", "SA"), ("PERTH", "WA"), ("CANBERRA", "AC"), ("HOBART", "TA")]
PREFIXES = ["SQ *", "EFTPOS ", "VISA PURCHASE ", "SQ *", "EFTPOS "]

# Fixed monthly payments: (merchant, category, amount, day of month).
MONTHLY = [
    ("RENT PAYMENT RAY WHITE", "Bills", 2150.00, 1), ("AGL ENERGY", "Bills", None, 12),
    ("TELSTRA", "Bills", 89.00, 18), ("ALLIANZ INSURANCE", "Bills", 64.50, 22),
    ("PG&E ONLINE PMT", "Bills", None, 25), ("NETFLIX.COM", "Subscriptions", 18.99, 3),
    ("SPOTIFY", "Subscriptions", 13.99, 9), ("STAN", "Subscriptions", 12.00, 14),
    ("DISNEY PLUS", "Subscriptions", 15.99, 20), ("APPLE.COM/BILL", "Subscriptions", 4.49, 27),
    ("ICLOUD", "Subscriptions", 1.49, 6),
]
# How often each category appears among the everyday random rows.
WEIGHTS = {"Dining": 26, "Groceries": 18, "Transport": 20, "Shopping": 13, "Entertainment": 9, "Other": 7,
           "Bills": 3, "Subscriptions": 4}
AMOUNTS = {**AMOUNT_RANGES, "Groceries": (8, 220), "Dining": (4, 85), "Transport": (3, 95)}


def decorate(name, rng):
    """Write a merchant the way a bank statement might. About half the rows stay clean."""
    if rng.random() < 0.5:
        return name
    out = name
    roll = rng.random()
    if roll < 0.35:
        out = f"{out} {rng.randint(1, 9999):04d}"
    elif roll < 0.65:
        suburb, state = rng.choice(SUBURBS)
        out = f"{out} {rng.randint(1, 999):04d} {suburb} {state}"
    elif roll < 0.85:
        out = rng.choice(PREFIXES) + out
    else:
        suburb, state = rng.choice(SUBURBS)
        out = f"{rng.choice(PREFIXES)}{out} {suburb} {state}"
    if rng.random() < 0.15:
        out = out.title()
    return out


def _amount(cat, rng):
    low, high = AMOUNTS[cat]
    return round(rng.uniform(low, high), 2)


def _day(start, rng, weekend_boost=1.0):
    """A random day; weekend_boost > 1 makes Saturdays and Sundays more likely."""
    while True:
        d = start + timedelta(days=rng.randint(0, DAYS - 1))
        if d.weekday() >= 5 or rng.random() < 1 / weekend_boost:
            return d


def generate(seed=SEED, end=None, target=TARGET_ROWS):
    rng = random.Random(seed)
    end = end or date.today()
    start = end - timedelta(days=DAYS - 1)
    rows = []

    # Rent, bills and subscriptions: every month, same day, same price (bills vary a little).
    month = date(start.year, start.month, 1)
    while month <= end:
        for name, cat, price, dom in MONTHLY:
            d = month.replace(day=dom)
            if start <= d <= end:
                amount = price if price else round(rng.uniform(70, 210), 2)
                rows.append((d, name, amount, cat))
        month = (month + timedelta(days=32)).replace(day=1)

    # Groceries every week at an Australian supermarket.
    d = start + timedelta(days=rng.randint(0, 6))
    while d <= end:
        name = rng.choice(["WOOLWORTHS", "COLES", "ALDI"])
        rows.append((d, decorate(name, rng), round(rng.uniform(60, 190), 2), "Groceries"))
        d += timedelta(days=7)

    fixed = len(rows)
    n_typed = int(target * 0.05)
    n_ambiguous = int(target * 0.15)
    n_refunds = int(target * 0.015)
    n_everyday = target - fixed - n_typed - n_ambiguous - n_refunds

    # Everyday spending: 70% Australian merchants, 30% the original US ones.
    cats, weights = list(WEIGHTS), list(WEIGHTS.values())
    for _ in range(n_everyday):
        cat = rng.choices(cats, weights)[0]
        pool = US_MERCHANTS.get(cat) if rng.random() < 0.3 and cat in US_MERCHANTS else AU_MERCHANTS[cat]
        boost = 2.5 if cat in ("Dining", "Entertainment") else 1.0
        rows.append((_day(start, rng, boost), decorate(rng.choice(pool), rng), _amount(cat, rng), cat))

    # Shops and transfers that genuinely span several categories.
    ambiguous = {**US_AMBIGUOUS, **AU_AMBIGUOUS}
    for _ in range(n_ambiguous):
        name = rng.choice(list(ambiguous))
        cat = rng.choice(ambiguous[name])
        if name == "PAYPAL *":
            name += rng.choice(["EBAY", "STEAMGAMES", "DEPOP", "UBEREATS", "KOGAN"])
        rows.append((_day(start, rng), decorate(name, rng), _amount(cat, rng), cat))

    # Typed the way quick add receives them.
    for _ in range(n_typed):
        cat = rng.choice(list(TYPED))
        text = rng.choice(TYPED[cat])
        amount = round(rng.uniform(450, 2200), 2) if text == "rent" else _amount(cat, rng)   # weekly or monthly rent
        rows.append((_day(start, rng), text, amount, cat))

    # Refunds: negative amounts from shops people return things to.
    for _ in range(n_refunds):
        name = rng.choice(["AMAZON AU", "KMART", "BIG W", "JB HI-FI", "MYER", "TARGET AUSTRALIA", "UNIQLO"])
        cat = "Shopping" if name not in AU_AMBIGUOUS else rng.choice(AU_AMBIGUOUS[name])
        rows.append((_day(start, rng), decorate(name, rng), -_amount(cat, rng), cat))

    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "category"])

    # About 4% wrong labels on purpose, like a person tagging in a hurry.
    for i in rng.sample(range(len(df)), int(len(df) * 0.04)):
        df.at[i, "category"] = rng.choice([c for c in CATEGORIES if c != df.at[i, "category"]])

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate data/transactions_v2.csv")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="Last day of the data (YYYY-MM-DD). Default: today.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    df = generate(args.seed, args.end)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} transactions to {args.out} ({df['date'].min()} to {df['date'].max()})")
    print(f"Refunds: {(df['amount'] < 0).sum()}")
    print("\nRows per category:")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
