import os
import unittest
from unittest import mock

from helpers import TODAY, txns

import advisor
import money_facts as mf


def reference_data():
    """Three complete months whose averages are the spec's reference numbers."""
    avgs = {"Shopping": 1274.87, "Groceries": 921.34, "Transport": 419.58, "Dining": 145.71, "Bills": 600.0}
    rows = []
    for month in ["2026-07", "2026-08", "2026-09"]:
        for cat, avg in avgs.items():
            rows.append((f"{month}-10", cat.upper(), avg, cat))
    return txns(rows)


class GoalDetectionTests(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual(mf.detect_goal("How can I save $200 this month?"), 200.0)
        self.assertEqual(mf.detect_goal("I want to cut 150 from my spending"), 150.0)
        self.assertEqual(mf.detect_goal("help me put aside $1,000"), 1000.0)
        self.assertEqual(mf.detect_goal("reduce spending by $75.50"), 75.5)

    def test_percentages_years_and_no_verb(self):
        self.assertIsNone(mf.detect_goal("cut dining by 20%"))
        self.assertIsNone(mf.detect_goal("save in 2026"))
        self.assertIsNone(mf.detect_goal("I spent $200 on dining"))
        self.assertIsNone(mf.detect_goal("how do I save money?"))
        self.assertEqual(mf.detect_goal("save 20% or $300 in 2026"), 300.0)


class SavingsGoalTests(unittest.TestCase):
    def test_reference_example_exactly(self):
        plan = mf.savings_goal(reference_data(), 300, TODAY)
        cuts = {r["category"]: r["cut"] for r in plan["rows"]}
        self.assertEqual(cuts, {"Shopping": 138.50, "Groceries": 100.09, "Transport": 45.58, "Dining": 15.83})
        self.assertEqual(round(sum(cuts.values()), 2), 300.00)
        self.assertEqual(plan["total_cut"], 300.0)
        self.assertAlmostEqual(plan["percent"], 0.1086, places=4)
        self.assertTrue(plan["feasible"])

    def test_bills_excluded(self):
        plan = mf.savings_goal(reference_data(), 300, TODAY)
        self.assertNotIn("Bills", [r["category"] for r in plan["rows"]])
        self.assertEqual(plan["excluded"], ["Bills"])

    def test_cap_and_feasibility(self):
        plan = mf.savings_goal(reference_data(), 5000, TODAY)
        self.assertFalse(plan["feasible"])
        self.assertEqual(plan["max_possible"], round(2761.50 * 0.30, 2))
        self.assertEqual(plan["total_cut"], plan["max_possible"])
        for r in plan["rows"]:
            self.assertLessEqual(r["cut"], r["average"] * 0.30 + 0.01)
        self.assertIn("only reaches", mf.describe_goal(plan))

    def test_describe_goal(self):
        text = mf.describe_goal(mf.savings_goal(reference_data(), 300, TODAY))
        self.assertIn("Shopping: usually $1,274.87 a month, spend $138.50 less", text)
        self.assertIn("Total: $300 less.", text)


class FactsTests(unittest.TestCase):
    def test_facts_text(self):
        df = txns([("2026-08-05", "c", 145.71, "Dining"), ("2026-09-05", "c", 190.10, "Dining")])
        text = mf.facts_for_advisor(df, TODAY)
        self.assertIn("$145.71 -> $190.10, up $44.39 (+30%)", text)
        self.assertIn("calculated by the app", text)
        self.assertIn("10% less saves", text)

    def test_facts_reach_the_advisor(self):
        convo = advisor.start_conversation(reference_data(), "Budget text here", today=TODAY)
        self.assertEqual([m["role"] for m in convo], ["system", "user", "assistant"])
        self.assertIn("Average spending per month", convo[1]["content"])
        self.assertIn("Budget text here", convo[1]["content"])

    def test_number_check(self):
        refs = ["Dining: $145.71, Shopping $1,274.87", "How can I save $300?"]
        self.assertEqual(mf.unverified_amounts("Cut dining from $145.71 and shopping ($1,275).", refs), [])
        self.assertEqual(mf.unverified_amounts("You could save $818.64 or $21.40, even $300.", refs),
                         ["$818.64", "$21.40"])

    def test_local_temperature(self):
        self.assertEqual(advisor.ADVICE_TEMPERATURE, 0.2)
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"message": {"content": "ok"}}
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "ollama"}), \
                mock.patch("advisor.requests.post", return_value=reply) as post:
            advisor.chat([{"role": "user", "content": "x"}])
        self.assertEqual(post.call_args.kwargs["json"]["options"]["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
