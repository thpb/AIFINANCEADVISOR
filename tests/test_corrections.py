import sqlite3
import unittest

import pandas as pd
from helpers import DBTestCase

import auth
import corrections as c
import db


class RuleTests(unittest.TestCase):
    def test_merchant_key_matches_variants(self):
        self.assertEqual(c.merchant_key("SQ *BLUE BOTTLE #12"), c.merchant_key("blue bottle"))
        self.assertEqual(c.merchant_key("TST* Chipotle 123456"), "CHIPOTLE")

    def test_rules_override_with_full_confidence(self):
        df = pd.DataFrame({"merchant": ["SQ *BLUE BOTTLE #12", "UBER"], "predicted_category": ["Other", "Transport"],
                           "confidence": [0.4, 0.9]})
        out = c.apply_rules(df, {"BLUE BOTTLE": "Dining"})
        self.assertEqual(out.loc[0, "predicted_category"], "Dining")
        self.assertEqual(out.loc[0, "confidence"], 1.0)
        self.assertTrue(out.loc[0, "rule_applied"])
        self.assertFalse(out.loc[1, "rule_applied"])
        self.assertEqual(out.loc[1, "confidence"], 0.9)

    def test_needs_review_order_threshold_and_null(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "merchant": list("abcde"),
                           "confidence": [0.59, 0.60, None, 0.2, 0.9]})
        out = c.needs_review(df)
        self.assertEqual(list(out["id"]), [4, 1])

    def test_ids_for_merchant(self):
        df = pd.DataFrame({"id": [1, 2, 3], "merchant": ["Blue Bottle", "SQ *BLUE BOTTLE #4", "UBER"]})
        self.assertEqual(c.ids_for_merchant(df, "blue bottle"), [1, 2])

    def test_describe_uncertainty(self):
        df = pd.DataFrame({"merchant": list("abc"), "confidence": [0.1, 0.2, 0.3]})
        self.assertEqual(c.describe_uncertainty(df), "Note: 3 transactions had a low-confidence category "
                                                     "prediction, so category totals may be slightly off.")
        self.assertEqual(c.describe_uncertainty(df.assign(confidence=1.0)), "")


class RuleDBTests(DBTestCase):
    def test_rules_crud(self):
        u = self.make_user()
        db.set_rule(u, "BLUE BOTTLE", "Dining")
        db.set_rule(u, "BLUE BOTTLE", "Groceries")
        self.assertEqual(db.get_rules(u), {"BLUE BOTTLE": "Groceries"})
        db.delete_rule(u, "BLUE BOTTLE")
        self.assertEqual(db.get_rules(u), {})

    def test_update_categories_is_per_user(self):
        a, b = self.make_user("a"), self.make_user("b")
        df = pd.DataFrame({"date": ["2026-10-01"], "merchant": ["X"], "amount": [5.0],
                           "predicted_category": ["Other"], "confidence": [0.3]})
        db.insert_transactions(a, df)
        db.insert_transactions(b, df)
        ids_b = list(db.get_transactions(b)["id"])
        self.assertEqual(db.update_categories(a, ids_b, "Dining"), 0)
        self.assertEqual(db.get_transactions(b).loc[0, "predicted_category"], "Other")
        ids_a = list(db.get_transactions(a)["id"])
        db.update_categories(a, ids_a, "Dining")
        row = db.get_transactions(a).iloc[0]
        self.assertEqual((row["predicted_category"], row["confidence"]), ("Dining", 1.0))

    def test_confidence_migration_twice_is_safe(self):
        conn = sqlite3.connect(db.DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(transactions)")]
        conn.close()
        self.assertIn("confidence", cols)
        db.init_db()
        db.init_db()

    def test_auth(self):
        self.assertEqual(auth.signup("", "secret1")[0], False)
        self.assertEqual(auth.signup("sam", "short")[0], False)
        self.assertTrue(auth.signup("sam", "secret1")[0])
        self.assertFalse(auth.signup("sam", "secret2")[0])
        uid, _ = auth.login("sam", "secret1")
        self.assertIsNotNone(uid)
        self.assertIsNone(auth.login("sam", "wrong!!")[0])
        self.assertIsNone(auth.login("nobody", "secret1")[0])
        self.assertNotIn("secret1", db.get_user_by_username("sam")["password_hash"])


if __name__ == "__main__":
    unittest.main()
