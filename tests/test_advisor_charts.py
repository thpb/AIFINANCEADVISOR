import os
import unittest
from unittest import mock

import pandas as pd
from helpers import txns

import advisor
import advisor_charts as ac

CATS = ["Dining", "Groceries", "Shopping"]
MONTHS = ["2026-08", "2026-09", "2026-10"]


class TriggerAndValidationTests(unittest.TestCase):
    def test_trigger_words(self):
        for q in ["Show my spending by category as a chart", "plot dining", "monthly trend?", "top 5 merchants",
                  "what were my biggest purchases", "Give me a breakdown", "compare months", "list my budgets"]:
            self.assertTrue(ac.wants_visual(q), q)
        for q in ["How can I save money?", "Am I doing ok?"]:
            self.assertFalse(ac.wants_visual(q), q)

    def test_invalid_replies(self):
        for raw in [None, "chart", [], {"kind": "pie"}, {"kind": "none"}, {}]:
            self.assertIsNone(ac.validate_spec(raw, CATS, MONTHS), raw)

    def test_categories_and_months_cleaned(self):
        s = ac.validate_spec({"kind": "monthly_spending", "categories": ["dining", "Pets", "DINING"],
                              "month": "2030-01", "months": 99, "limit": 1}, CATS, MONTHS)
        self.assertEqual(s["categories"], ["Dining"])
        self.assertIsNone(s["month"])
        self.assertEqual((s["months"], s["limit"]), (24, 3))

    def test_single_string_and_defaults(self):
        s = ac.validate_spec({"kind": "category_totals", "categories": "groceries", "months": "x"}, CATS, MONTHS)
        self.assertEqual(s["categories"], ["Groceries"])
        self.assertEqual((s["months"], s["limit"], s["format"]), (6, 10, "chart"))

    def test_table_only_kinds(self):
        s = ac.validate_spec({"kind": "top_purchases", "format": "chart"}, CATS, MONTHS)
        self.assertEqual(s["format"], "table")
        self.assertEqual(ac.validate_spec({"kind": "budget_status"}, CATS, MONTHS)["format"], "table")

    def test_router_prompt(self):
        text = ac.router_prompt(CATS, MONTHS)
        self.assertTrue(text.startswith("Reply with JSON only"))
        self.assertIn("Dining, Groceries, Shopping", text)
        self.assertIn("2026-10", text)
        self.assertIn("Never invent categories or months.", text)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.spend = txns([
            ("2026-09-02", "SQ *BLUE BOTTLE #12", 10.0, "Dining"),
            ("2026-09-05", "blue bottle", 15.0, "Dining"),
            ("2026-09-07", "COLES", 100.0, "Groceries"),
            ("2026-10-02 00:00:00", "BLUE BOTTLE", 20.0, "Dining"),
            ("2026-10-03", "MYER", 300.0, "Shopping"),
        ])
        self.income = pd.DataFrame({"date": ["2026-09-01", "2026-10-01"], "source": ["s", "s"],
                                    "amount": [1000.0, 1200.0]})

    def spec(self, **kw):
        return ac.validate_spec(kw, CATS, MONTHS)

    def test_category_totals(self):
        r = ac.build(self.spec(kind="category_totals"), self.spend, self.income, {})
        self.assertEqual(r["series"].to_dict(), {"Shopping": 300.0, "Groceries": 100.0, "Dining": 45.0})
        self.assertIn("$300", r["text"])

    def test_category_totals_one_month(self):
        r = ac.build(self.spec(kind="category_totals", month="2026-09"), self.spend, self.income, {})
        self.assertEqual(r["series"].to_dict(), {"Groceries": 100.0, "Dining": 25.0})
        self.assertIn("September 2026", r["title"])

    def test_monthly_spending_filtered(self):
        r = ac.build(self.spec(kind="monthly_spending", categories=["Dining"]), self.spend, self.income, {})
        self.assertEqual(list(r["chart_df"]["amount"]), [25.0, 20.0])
        self.assertEqual(set(r["chart_df"]["category"]), {"Dining"})
        self.assertEqual(list(r["chart_df"]["month_str"]), ["Sep 2026", "Oct 2026"])

    def test_income_vs_spending(self):
        r = ac.build(self.spec(kind="income_vs_spending"), self.spend, self.income, {})
        t = r["table"].set_index("Month")
        self.assertEqual(t.loc["2026-09", "Saved"], 875.0)
        self.assertEqual(t.loc["2026-10", "Saved"], 880.0)
        self.assertEqual(list(r["income_by_month"]), [1000.0, 1200.0])

    def test_top_purchases(self):
        r = ac.build(self.spec(kind="top_purchases", limit=3), self.spend, self.income, {})
        self.assertEqual(list(r["table"]["Amount"]), [300.0, 100.0, 20.0])

    def test_top_merchants_grouped(self):
        r = ac.build(self.spec(kind="top_merchants"), self.spend, self.income, {})
        self.assertEqual(r["series"]["Blue Bottle"], 45.0)

    def test_budget_status(self):
        r = ac.build(self.spec(kind="budget_status"), self.spend, self.income, {"Shopping": 250})
        self.assertEqual(r["table"].loc[0, "State"], "over budget")
        self.assertIsNone(ac.build(self.spec(kind="budget_status"), self.spend, self.income, {}))

    def test_no_data(self):
        self.assertIsNone(ac.build(self.spec(kind="category_totals"), txns([]), self.income, {}))
        self.assertIsNone(ac.build(None, self.spend, self.income, {}))
        r = ac.build(self.spec(kind="category_totals", month="2026-08"), self.spend, self.income, {})
        self.assertIsNone(r)

    def test_note_sent_but_hidden(self):
        r = ac.build(self.spec(kind="category_totals"), self.spend, self.income, {})
        note = ac.chart_note(r)
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"message": {"content": "Shopping is the biggest at $300."}}
        history = [{"role": "system", "content": "s"}, {"role": "user", "content": "d"},
                   {"role": "assistant", "content": "ok"}]
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "ollama"}), \
                mock.patch("advisor.requests.post", return_value=reply) as post:
            answer, hist = advisor.ask_advisor("show a chart", history, note=note)
        sent = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertIn("Its exact numbers", sent)
        shown = hist[-2]["content"].split(advisor.NOTE_MARK)[0]
        self.assertEqual(shown, "show a chart")

    def test_pick_visual_failure_gives_none(self):
        with mock.patch("advisor._call_llm_json", side_effect=ValueError("bad json")):
            self.assertIsNone(advisor.pick_visual("chart", CATS, MONTHS))
        with mock.patch("advisor._call_llm_json", return_value={"kind": "category_totals"}):
            self.assertEqual(advisor.pick_visual("chart", CATS, MONTHS)["kind"], "category_totals")

    def test_named_category_added_when_model_drops_it(self):
        with mock.patch("advisor._call_llm_json", return_value={"kind": "monthly_spending"}):
            spec = advisor.pick_visual("show my dining spending by month as a chart", CATS, MONTHS)
        self.assertEqual(spec["categories"], ["Dining"])
        with mock.patch("advisor._call_llm_json", return_value={"kind": "monthly_spending", "categories": ["Shopping"]}):
            spec = advisor.pick_visual("show my dining spending by month", CATS, MONTHS)
        self.assertEqual(spec["categories"], ["Shopping"])      # the model's valid choice is kept
        self.assertEqual(ac.mentioned_categories("my spending by category", CATS), [])


if __name__ == "__main__":
    unittest.main()
