import sqlite3
import unittest

import pandas as pd
from helpers import TODAY, DBTestCase, txns

import db
import savings
from nl_entry import parse_entry
from table_edits import as_date_text, editor_changes


def income(rows):
    return pd.DataFrame(rows, columns=["date", "source", "amount"])


class SavingsTests(unittest.TestCase):
    def test_income_sentences(self):
        e = parse_entry("got paid 2500 salary", today=TODAY, use_llm=False)
        self.assertEqual((e["kind"], e["amount"]), ("income", 2500.0))

    def test_monthly_table_and_rate(self):
        spend = txns([("2026-09-02", "a", 1000.0, "Dining"), ("2026-10-02", "a", 1499.0, "Dining")])
        inc = income([("2026-10-01", "salary", 4300.0)])
        t = savings.monthly_table(spend, inc).set_index("month")
        self.assertIsNone(t.loc["2026-09", "rate"])
        self.assertEqual(t.loc["2026-10", "saved"], 2801.0)
        self.assertAlmostEqual(t.loc["2026-10", "rate"], 2801 / 4300)
        s = savings.month_summary(spend, inc, "2026-10")
        self.assertEqual((s["income"], s["spent"], s["saved"]), (4300.0, 1499.0, 2801.0))

    def test_no_income(self):
        spend = txns([("2026-10-02", "a", 10.0, "Dining")])
        self.assertIsNone(savings.month_summary(spend, income([]), "2026-10")["rate"])
        self.assertIn("No income", savings.describe_for_advisor(spend, income([])))

    def test_average_income(self):
        inc = income([("2026-07-01", "s", 3000.0), ("2026-08-01", "s", 4000.0), ("2026-09-01", "s", 5000.0),
                      ("2026-10-01", "s", 9999.0)])
        self.assertEqual(savings.average_monthly_income(inc, TODAY), 4000.0)
        self.assertEqual(savings.average_monthly_income(income([]), TODAY), 0.0)


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.shown = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"]),
                                   "merchant": ["A", "B", "C"], "amount": [1.0, 2.0, 3.0],
                                   "category": ["Dining", "Dining", "Other"]}, index=[11, 22, 33])

    def test_positions_map_to_ids(self):
        upd, dele = editor_changes(self.shown, {"edited_rows": {1: {"category": "Groceries"}}, "deleted_rows": [2]})
        self.assertEqual(list(upd), [22])
        self.assertEqual(upd[22]["category"], "Groceries")
        self.assertEqual(upd[22]["merchant"], "B")
        self.assertEqual(dele, [33])

    def test_edit_then_delete_counts_as_delete(self):
        upd, dele = editor_changes(self.shown, {"edited_rows": {0: {"amount": 9}}, "deleted_rows": [0]})
        self.assertEqual((upd, dele), ({}, [11]))

    def test_no_op_edits_ignored(self):
        state = {"edited_rows": {0: {"amount": 1.001, "date": "2026-10-01", "merchant": "A"}}}
        self.assertEqual(editor_changes(self.shown, state), ({}, []))

    def test_date_change(self):
        upd, _ = editor_changes(self.shown, {"edited_rows": {0: {"date": "2026-10-09"}}})
        self.assertEqual(as_date_text(upd[11]["date"]), "2026-10-09 00:00:00")

    def test_empty_state(self):
        self.assertEqual(editor_changes(self.shown, {}), ({}, []))

    def test_as_date_text(self):
        self.assertEqual(as_date_text("2026-10-05"), "2026-10-05 00:00:00")
        self.assertEqual(as_date_text(pd.Timestamp("2026-10-05 00:00:00")), "2026-10-05 00:00:00")


class EditDBTests(DBTestCase):
    def test_update_and_delete_transactions(self):
        u, other = self.make_user("u"), self.make_user("o")
        df = txns([("2026-10-01 00:00:00", "A", 1.0, "Other"), ("2026-10-02 00:00:00", "B", 2.0, "Other")])
        db.insert_transactions(u, df)
        ids = list(db.get_transactions(u)["id"])
        self.assertEqual(db.update_transaction(other, ids[0], "2026-10-05", "Z", 9, "Dining"), 0)
        db.update_transaction(u, ids[0], "2026-10-05 00:00:00", "Cafe", 9.5, "Dining")
        row = db.get_transactions(u).set_index("id").loc[ids[0]]
        self.assertEqual((row["merchant"], row["amount"], row["predicted_category"], row["confidence"]),
                         ("Cafe", 9.5, "Dining", 1.0))
        self.assertEqual(db.delete_transactions(other, ids), 0)
        db.delete_transactions(u, [ids[1]])
        self.assertEqual(len(db.get_transactions(u)), 1)

    def test_income_crud_and_isolation(self):
        u, other = self.make_user("u"), self.make_user("o")
        i = db.add_income(u, "2026-10-01", "salary", 2000)
        db.add_income(u, "2026-10-15", "bonus", 100)
        self.assertTrue(db.get_income(other).empty)
        self.assertEqual(db.update_income(other, i, "2026-10-01", "x", 1), 0)
        db.update_income(u, i, "2026-10-02", "wages", 2100)
        inc = db.get_income(u).set_index("id")
        self.assertEqual((inc.loc[i, "source"], inc.loc[i, "amount"]), ("wages", 2100.0))
        db.delete_income(u, [i])
        self.assertEqual(len(db.get_income(u)), 1)
        db.delete_all_income(u)
        self.assertTrue(db.get_income(u).empty)

    def test_delete_all_keeps_budgets(self):
        u = self.make_user()
        db.insert_transactions(u, txns([("2026-10-01", "A", 1.0, "Other")]))
        db.set_budget(u, "Other", 50)
        db.delete_all_transactions(u)
        self.assertTrue(db.get_transactions(u).empty)
        self.assertEqual(db.get_budgets(u), {"Other": 50.0})

    def test_only_new_skips_stored_rows(self):
        u = self.make_user()
        db.insert_transactions(u, txns([("2026-10-01 00:00:00", "ALDI", 30.0, "Groceries")]))
        incoming = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-10-02"]),
                                 "merchant": ["aldi", "ALDI", "ALDI"], "amount": [30.0, 30.0, 30.0]})
        new = db.only_new(u, incoming)
        self.assertEqual(len(new), 2)                     # the second identical coffee is kept
        self.assertEqual(list(new.index), [1, 2])
        self.assertEqual(len(db.only_new(self.make_user("other"), incoming)), 3)

    def test_old_db_gets_income_table(self):
        conn = sqlite3.connect(db.DB_PATH)
        conn.execute("DROP TABLE income")
        conn.commit()
        conn.close()
        db.init_db()
        u = self.make_user()
        db.add_income(u, "2026-10-01", "salary", 1)
        self.assertEqual(len(db.get_income(u)), 1)


if __name__ == "__main__":
    unittest.main()
