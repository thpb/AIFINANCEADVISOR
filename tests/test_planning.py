import unittest
from datetime import date

from helpers import TODAY, txns

import planning as p


class ForecastTests(unittest.TestCase):
    def setUp(self):
        # October 2026 has 31 days; TODAY is the 29th.
        self.df = txns([
            ("2026-10-01", "CAFE", 290.0, "Dining"),
            ("2026-10-10 00:00:00", "COLES", 100.0, "Groceries"),
            ("2026-09-15", "COLES", 300.0, "Groceries"),
        ])

    def test_run_rate(self):
        fc = p.forecast_month(self.df, "2026-10", date(2026, 10, 10), {"Dining": 400})
        self.assertEqual(fc["days_elapsed"], 10)
        self.assertAlmostEqual(fc["projected"], 390 * 31 / 10, places=2)
        dining = next(r for r in fc["categories"] if r["category"] == "Dining")
        self.assertEqual(dining["projected"], 899.0)
        self.assertEqual(dining["state"], "on_pace_over")
        self.assertEqual(p.forecast_messages(fc),
                         ["Dining is on pace for $899 by month end, $499 over its $400 budget."])

    def test_already_over_and_ok(self):
        fc = p.forecast_month(self.df, "2026-10", date(2026, 10, 10), {"Dining": 200, "Groceries": 1000})
        st = {r["category"]: r["state"] for r in fc["categories"]}
        self.assertEqual(st, {"Dining": "already_over", "Groceries": "ok"})

    def test_past_month_is_complete(self):
        fc = p.forecast_month(self.df, "2026-09", TODAY)
        self.assertTrue(fc["complete"])
        self.assertEqual(fc["projected"], 300.0)

    def test_future_month_empty(self):
        fc = p.forecast_month(self.df, "2026-12", TODAY)
        self.assertTrue(fc["complete"])
        self.assertEqual(fc["projected"], 0)

    def test_too_early(self):
        fc = p.forecast_month(self.df, "2026-10", date(2026, 10, 4), {"Dining": 100})
        self.assertTrue(fc["too_early"])
        self.assertEqual(p.forecast_messages(fc), [])

    def test_empty(self):
        fc = p.forecast_month(txns([]), "2026-10", TODAY)
        self.assertEqual(fc["projected"], 0)


class AverageTests(unittest.TestCase):
    def setUp(self):
        self.df = txns([
            ("2026-07-03", "a", 100.0, "Dining"),
            ("2026-08-03", "a", 200.0, "Dining"),
            ("2026-09-03", "a", 300.0, "Dining"),
            ("2026-09-03", "b", 60.0, "Groceries"),
            ("2026-10-03", "a", 5000.0, "Dining"),     # current month: ignored
        ])

    def test_complete_months_only(self):
        avg = p.monthly_average(self.df, TODAY)
        self.assertEqual(avg["Dining"], 200.0)
        self.assertEqual(avg["Groceries"], 20.0)
        self.assertEqual(p.months_used(self.df, TODAY), ["2026-07", "2026-08", "2026-09"])

    def test_falls_back_to_all_months(self):
        df = txns([("2026-10-03", "a", 50.0, "Dining")])
        self.assertEqual(p.monthly_average(df, TODAY)["Dining"], 50.0)

    def test_what_if(self):
        res = p.what_if(self.df, {"Dining": 10, "Groceries": 150}, TODAY)
        rows = {r["category"]: r for r in res["rows"]}
        self.assertEqual(rows["Dining"]["monthly_saving"], 20.0)
        self.assertEqual(rows["Groceries"]["percent"], 100.0)       # clamped
        self.assertEqual(rows["Groceries"]["monthly_saving"], 20.0)
        self.assertEqual(res["monthly_total"], 40.0)
        self.assertEqual(res["yearly_total"], 480.0)

    def test_what_if_negative_clamped_and_empty(self):
        self.assertEqual(p.what_if(self.df, {"Dining": -10}, TODAY)["monthly_total"], 0)
        self.assertEqual(p.what_if(txns([]), {"Dining": 10}, TODAY)["monthly_total"], 0)


if __name__ == "__main__":
    unittest.main()
