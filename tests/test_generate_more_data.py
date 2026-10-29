import unittest
from datetime import date

from helpers import TODAY  # noqa: F401  (puts src on the path)

import generate_more_data as g
from clean_data import clean_merchant

END = date(2026, 10, 29)
CATEGORIES = {"Bills", "Dining", "Entertainment", "Groceries", "Other", "Shopping", "Subscriptions", "Transport"}


class MoreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = g.generate(end=END)

    def test_columns_size_and_dates(self):
        self.assertEqual(list(self.df.columns), ["date", "merchant", "amount", "category"])
        self.assertAlmostEqual(len(self.df), 4000, delta=100)
        self.assertEqual(self.df["date"].max(), "2026-10-29")
        self.assertEqual(self.df["date"].min(), "2025-10-30")
        self.assertTrue(self.df["date"].is_monotonic_increasing)

    def test_only_valid_categories(self):
        self.assertEqual(set(self.df["category"]), CATEGORIES)

    def test_reproducible_with_same_seed(self):
        self.assertTrue(g.generate(end=END).equals(self.df))
        self.assertFalse(g.generate(seed=1, end=END).equals(self.df))

    def test_australian_merchants(self):
        names = " ".join(self.df["merchant"]).upper()
        for shop in ["WOOLWORTHS", "COLES", "TRANSPORTNSW OPAL", "AGL ENERGY", "STAN", "HOYTS", "BUNNINGS"]:
            self.assertIn(shop, names)

    def test_us_merchants_kept(self):
        names = " ".join(self.df["merchant"]).upper()
        for shop in ["TRADER JOE'S", "CHIPOTLE", "LYFT", "STEAM GAMES"]:
            self.assertIn(shop, names)

    def test_bank_statement_noise(self):
        m = self.df["merchant"]
        self.assertTrue(m.str.contains(r" \d{4}( |$)").any())                       # store numbers
        self.assertTrue(m.str.contains(r" (NS|VI|QL|SA|WA|AC|TA)$").any())         # suburb + state
        self.assertTrue(m.str.upper().str.startswith(("SQ *", "EFTPOS ", "VISA PURCHASE ")).any())
        clean = (m == m.str.upper()) & ~m.str.contains(r"\d|\*|EFTPOS|VISA PURCHASE")
        self.assertGreater(clean.mean(), 0.3)                                       # about half stay clean

    def test_typed_entries(self):
        typed = self.df[self.df["merchant"].isin([t for ts in g.TYPED.values() for t in ts])]
        self.assertAlmostEqual(len(typed) / len(self.df), 0.05, delta=0.015)
        self.assertIn("uber home", set(typed["merchant"]))

    def test_refunds_and_ambiguous(self):
        self.assertGreater((self.df["amount"] < 0).sum(), 30)
        amazon = self.df[self.df["merchant"].map(clean_merchant).str.startswith("AMAZON")]
        self.assertGreater(amazon["category"].nunique(), 2)

    def test_monthly_patterns(self):
        rent = self.df[self.df["merchant"] == "RENT PAYMENT RAY WHITE"]
        self.assertEqual(len(rent), 12)
        self.assertEqual(set(rent["amount"]), {2150.0})
        netflix = self.df[self.df["merchant"] == "NETFLIX.COM"]
        self.assertGreaterEqual((netflix["amount"] == 18.99).sum(), 12)

    def test_more_dining_on_weekends(self):
        import pandas as pd
        dining = self.df[self.df["category"] == "Dining"]
        weekend = pd.to_datetime(dining["date"]).dt.day_of_week >= 5
        self.assertGreater(weekend.mean(), 2 / 7 + 0.1)

    def test_clean_merchant_strips_bank_prefixes(self):
        self.assertEqual(clean_merchant("EFTPOS WOOLWORTHS 1234"), "WOOLWORTHS")
        self.assertEqual(clean_merchant("VISA PURCHASE COLES 0456 PARRAMATTA NS"), "COLES PARRAMATTA NS")


if __name__ == "__main__":
    unittest.main()
