"""Older, simpler generator: clean merchant names only, no ambiguity or noise.

Kept for comparison. Every model scores near 100% on this data, which is why
the project moved to generate_realistic_data.py.
"""
import _root  # noqa: F401

import random
from datetime import date, timedelta

import pandas as pd

from generate_realistic_data import AMOUNT_RANGES, CLEAN_MERCHANTS

random.seed(1)


def main(n=300, out="data/sample_transactions.csv"):
    today = date.today()
    rows = []
    for _ in range(n):
        cat = random.choice(list(CLEAN_MERCHANTS))
        low, high = AMOUNT_RANGES[cat]
        d = today - timedelta(days=random.randint(0, 179))
        rows.append((d.isoformat(), random.choice(CLEAN_MERCHANTS[cat]), round(random.uniform(low, high), 2), cat))
    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "category"]).sort_values("date")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
