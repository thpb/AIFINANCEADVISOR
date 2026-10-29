import sqlite3
import unittest

from helpers import DBTestCase, txns

import budgets as b
import db


class BudgetMathsTests(unittest.TestCase):
    def setUp(self):
        self.df = txns([
            ("2026-10-02", "COLES", 80.0, "Groceries"),
            ("2026-10-05 00:00:00", "WOOLWORTHS", 0.0, "Groceries"),   # mixed date formats
            ("2026-10-03", "CAFE", 90.0, "Dining"),
            ("2026-10-04", "UBER", 50.0, "Transport"),
            ("2026-09-04", "UBER", 500.0, "Transport"),               # other month
        ])

    def test_states_and_boundaries(self):
        self.assertEqual(b.state_for(0.79), "ok")
        self.assertEqual(b.state_for(0.80), "warning")
        self.assertEqual(b.state_for(1.00), "warning")
        self.assertEqual(b.state_for(1.0001), "over")

    def test_exact_80_and_100_percent(self):
        st = {r["category"]: r for r in b.budget_status(self.df, {"Groceries": 100, "Dining": 90}, "2026-10")}
        self.assertEqual(st["Groceries"]["state"], "warning")
        self.assertEqual(st["Dining"]["state"], "warning")

    def test_sorted_most_used_first_and_zero_skipped(self):
        st = b.budget_status(self.df, {"Groceries": 400, "Transport": 40, "Dining": 0, "Bills": None}, "2026-10")
        self.assertEqual([r["category"] for r in st], ["Transport", "Groceries"])
        self.assertEqual(st[0]["state"], "over")
        self.assertEqual(st[1]["remaining"], 320.0)

    def test_alert_messages(self):
        st = b.budget_status(txns([("2026-10-01", "x", 410.5, "Groceries"), ("2026-10-01", "y", 84.71, "Dining")]),
                             {"Groceries": 400, "Dining": 90}, "2026-10")
        self.assertEqual(b.alert_messages(st), [
            "Groceries is $10.50 over its $400 budget.",
            "Dining has used 94% of its $90 budget ($5.29 left).",
        ])

    def test_advisor_text(self):
        st = b.budget_status(self.df, {"Groceries": 400}, "2026-10")
        text = b.describe_for_advisor(st, "2026-10")
        self.assertIn("October 2026", text)
        self.assertIn("spent $80 of $400 (20%), $320 left, on track", text)
        self.assertIn("No budgets", b.describe_for_advisor([], "2026-10"))

    def test_money(self):
        self.assertEqual(b.money(5), "$5")
        self.assertEqual(b.money(5.5), "$5.50")
        self.assertEqual(b.money(1234), "$1,234")
        self.assertEqual(b.money(1234.567), "$1,234.57")
        self.assertEqual(b.money(-5), "-$5")

    def test_spent_by_category(self):
        s = b.spent_by_category(self.df, "2026-10")
        self.assertEqual(s["Transport"], 50.0)
        self.assertTrue(b.spent_by_category(txns([]), "2026-10").empty)


class BudgetDBTests(DBTestCase):
    def test_set_update_delete(self):
        u = self.make_user()
        db.set_budget(u, "Groceries", 400)
        db.set_budget(u, "Groceries", 450)
        db.set_budget(u, "Dining", 90)
        self.assertEqual(db.get_budgets(u), {"Groceries": 450.0, "Dining": 90.0})
        db.delete_budget(u, "Dining")
        self.assertEqual(db.get_budgets(u), {"Groceries": 450.0})

    def test_per_user(self):
        a, c = self.make_user("a"), self.make_user("c")
        db.set_budget(a, "Groceries", 400)
        self.assertEqual(db.get_budgets(c), {})

    def test_upgrade_old_database_keeps_data(self):
        # Build a database the way the first version did: no confidence column, no income table.
        conn = sqlite3.connect(db.DB_PATH)
        conn.executescript("DROP TABLE income; DROP TABLE transactions;")
        conn.executescript("""CREATE TABLE transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
          date TEXT NOT NULL, merchant TEXT NOT NULL, amount REAL NOT NULL, predicted_category TEXT,
          is_recurring INTEGER DEFAULT 0, is_ambiguous_merchant INTEGER DEFAULT 0, source TEXT DEFAULT 'upload',
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")
        conn.execute("INSERT INTO users (username, password_hash, salt) VALUES ('old', 'h', 's')")
        conn.execute("INSERT INTO transactions (user_id, date, merchant, amount, predicted_category)"
                     " VALUES (1, '2026-01-01', 'COLES', 10, 'Groceries')")
        conn.commit()
        conn.close()

        db.init_db()
        db.init_db()   # safe twice
        self.assertIsNotNone(db.get_user_by_username("old"))
        t = db.get_transactions(1)
        self.assertEqual(len(t), 1)
        self.assertTrue(t["confidence"].isna().all())
        self.assertTrue(db.get_income(1).empty)


if __name__ == "__main__":
    unittest.main()
