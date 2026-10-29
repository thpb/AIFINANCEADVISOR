import os
from datetime import date, timedelta
from unittest import mock

import numpy as np
import pandas as pd
from helpers import DBTestCase

import auth
import budgets
import db
import make_demo_account as demo

TODAY = date(2026, 11, 20)      # a different day from the data's last day, to check the date shift


class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 384))


class FakeEncoder:
    classes_ = np.array(["Bills", "Dining", "Groceries", "Other", "Shopping", "Transport"])

    def inverse_transform(self, idx):
        return self.classes_[idx]


class FakeModel:
    def predict_proba(self, x):
        return np.tile([0.1, 0.1, 0.1, 0.5, 0.1, 0.1], (len(x), 1))    # "Other"; keyword hints do the rest


MODELS = (FakeEmbedder(), FakeModel(), FakeEncoder())


def sample_csv(path, end=date(2026, 10, 29)):
    rows = []
    for back in range(120):
        d = end - timedelta(days=back)
        rows.append((d, "coffee", 6.0))
        if back % 7 == 0:
            rows += [(d, "COLES 0456 PARRAMATTA NS", 120.0), (d, "uber home", 25.0), (d, "kmart", 40.0)]
        if d.day == 1:
            rows.append((d, "RENT PAYMENT RAY WHITE", 2150.0))
    pd.DataFrame(rows, columns=["date", "merchant", "amount"]).to_csv(path, index=False)


class DemoAccountTests(DBTestCase):
    def setUp(self):
        super().setUp()
        self.csv = os.path.join(self._tmp.name, "transactions.csv")
        sample_csv(self.csv)
        self.patch = mock.patch.object(demo, "DATA_FILES", [os.path.join(self._tmp.name, "missing.csv"), self.csv])
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        super().tearDown()

    def test_creates_account_that_logs_in(self):
        info = demo.make_demo_account(today=TODAY, models=MODELS)
        uid, _ = auth.login("demo", "demo1234")
        self.assertEqual(uid, info["user_id"])
        self.assertEqual(info["file"], self.csv)          # falls back to transactions.csv

    def test_safe_to_run_twice(self):
        demo.make_demo_account(today=TODAY, models=MODELS)
        info = demo.make_demo_account(today=TODAY, models=MODELS)
        uid = info["user_id"]
        self.assertEqual(len(db.get_transactions(uid)), info["transactions"])
        self.assertEqual(len(db.get_income(uid)), info["paydays"])
        self.assertEqual(len(db.get_budgets(uid)), 4)
        conn = db.get_connection()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM users WHERE username = 'demo'").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], info["transactions"])
        conn.close()

    def test_data_ends_today_through_the_pipeline(self):
        uid = demo.make_demo_account(today=TODAY, models=MODELS)["user_id"]
        t = db.get_transactions(uid)
        self.assertEqual(pd.to_datetime(t["date"]).max().date(), TODAY)
        cats = dict(zip(t["merchant"], t["predicted_category"]))
        self.assertEqual((cats["coffee"], cats["COLES 0456 PARRAMATTA NS"], cats["uber home"]),
                         ("Dining", "Groceries", "Transport"))         # keyword hints were applied

    def test_budgets_one_close_to_limit(self):
        uid = demo.make_demo_account(today=TODAY, models=MODELS)["user_id"]
        status = budgets.budget_status(db.get_transactions(uid), db.get_budgets(uid), TODAY.strftime("%Y-%m"))
        states = {r["category"]: r["state"] for r in status}
        self.assertEqual(set(db.get_budgets(uid)), {"Groceries", "Dining", "Shopping", "Transport"})
        self.assertEqual(states["Dining"], "warning")
        self.assertEqual([c for c, s in states.items() if s != "ok"], ["Dining"])

    def test_six_months_of_salary(self):
        uid = demo.make_demo_account(today=TODAY, models=MODELS)["user_id"]
        inc = db.get_income(uid)
        dates = pd.to_datetime(inc["date"])
        self.assertEqual(set(inc["source"]), {"salary"})
        self.assertGreaterEqual((dates.max() - dates.min()).days, 5 * 30)
        self.assertLessEqual(dates.max().date(), TODAY)

    def test_v2_file_keeps_every_fixed_payment(self):
        v2 = os.path.join(self._tmp.name, "transactions_v2.csv")
        sample_csv(v2)
        with mock.patch.object(demo, "DATA_FILES", [v2]):
            _, df = demo.load_data(TODAY)
        source = pd.read_csv(v2)
        self.assertEqual((df["merchant"] == "RENT PAYMENT RAY WHITE").sum(),
                         (source["merchant"] == "RENT PAYMENT RAY WHITE").sum())    # all kept
        self.assertLess((df["merchant"] == "coffee").sum(), 60)          # everyday rows thinned (120 -> ~18)


if __name__ == "__main__":
    import unittest
    unittest.main()
