"""Click through the real Streamlit app headlessly with AppTest.

Run from the project root:  python -m unittest discover -s apptests
Uses the real trained model (train it first) and a throwaway database.
The Ollama call is mocked, so no AI model needs to be running.
AppTest can't type into st.data_editor; that logic is covered by tests/test_income_and_editing.py.
"""
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import db  # noqa: E402

APP = os.path.join(SRC, "dashboard.py")


def seed(uid):
    today = date.today()
    rows = []
    for back in range(1, 4):
        d = (pd.Timestamp(today.replace(day=1)) - pd.DateOffset(months=back)).date()
        rows += [(d + timedelta(days=2), "COLES", 600.0, "Groceries"), (d + timedelta(days=3), "BLUE BOTTLE", 40.0, "Dining"),
                 (d + timedelta(days=4), "MYER", 200.0, "Shopping"), (d + timedelta(days=5), "AGL", 150.0, "Bills"),
                 (d + timedelta(days=6), "NETFLIX.COM", 15.49, "Subscriptions")]
    rows.append((today, "AMAZON.COM", 39.99, "Other"))
    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "predicted_category"])
    df["date"] = df["date"].map(lambda d: f"{d:%Y-%m-%d} 00:00:00")
    df["confidence"] = [0.9] * (len(df) - 1) + [0.35]
    db.insert_transactions(uid, df, source="upload")


def ollama_reply(text):
    r = mock.Mock(status_code=200)
    r.json.return_value = {"message": {"content": text}}
    return r


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmp.name, "app.db")
        db.init_db()
        self.uid = db.create_user("tester", "h", "s")
        seed(self.uid)
        os.environ["ADVISOR_BACKEND"] = "ollama"
        self.at = AppTest.from_file(APP, default_timeout=120)
        self.at.session_state["user_id"] = self.uid
        self.at.session_state["username"] = "tester"
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)

    def tearDown(self):
        db.DB_PATH = self.old
        self.tmp.cleanup()

    def go(self, page):
        self.at.radio(key="page").set_value(page).run()
        self.assertFalse(self.at.exception, self.at.exception)

    def quick_add(self, text):
        box = self.at.container(key="quickadd")
        box.text_input[0].set_value(text)
        box.button[0].click().run()
        self.assertFalse(self.at.exception, self.at.exception)

    def texts(self, kind):
        return " ".join(str(e.value) for e in getattr(self.at, kind))

    def htmls(self):
        return " ".join(str(e.value) for e in self.at.get("html"))

    def test_every_page_renders(self):
        for page in ["Overview", "Plan ahead", "Transactions", "AI advisor"]:
            self.go(page)
        self.go("Transactions")
        labels = [t.label for t in self.at.tabs]
        self.assertIn("Worth a second look (1)", labels)

    def test_login_gate(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertEqual(at.title[0].value, "Expense Categorizer")
        self.assertEqual([t.label for t in at.tabs], ["Log in", "Sign up"])
        self.assertFalse(at.sidebar.radio)

    def test_sentence_hint_then_correction_learns_rule(self):
        self.quick_add("spent 50 on groceries yesterday")
        self.assertIn('The category comes from the word "groceries"', self.texts("caption"))
        form = self.at.get_by_key("confirm_box")
        self.assertEqual(form.date_input[0].value, date.today() - timedelta(days=1))
        self.assertEqual(form.selectbox[0].value, "Groceries")
        form.selectbox[0].set_value("Dining")
        form.button[0].click().run()          # Save
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual(db.get_rules(self.uid), {"GROCERIES": "Dining"})
        txns = db.get_transactions(self.uid)
        row = txns[txns["merchant"] == "groceries"].iloc[0]
        self.assertEqual((row["predicted_category"], row["amount"], row["confidence"]), ("Dining", 50.0, 1.0))

    def test_income_sentence_and_saved_stat(self):
        self.quick_add("got paid 2500 salary")
        self.assertEqual(self.at.radio(key="pending_kind").value, "Income")
        self.at.get_by_key("confirm_box").button[0].click().run()
        inc = db.get_income(self.uid)
        self.assertEqual((inc.loc[0, "source"], inc.loc[0, "amount"]), ("salary", 2500.0))
        self.assertIn("of $2,500 income", self.htmls())

    def test_switch_expense_to_income(self):
        self.quick_add("bonus 300")
        self.at.radio(key="pending_kind").set_value("Expense").run()
        self.at.radio(key="pending_kind").set_value("Income").run()
        self.at.get_by_key("confirm_box").button[0].click().run()
        self.assertEqual(db.get_income(self.uid).loc[0, "amount"], 300.0)
        self.quick_add("coffee 5")
        self.at.radio(key="pending_kind").set_value("Income").run()
        self.at.get_by_key("confirm_box").button[0].click().run()
        self.assertEqual(len(db.get_income(self.uid)), 2)

    def test_budget_alert_after_saving(self):
        db.set_budget(self.uid, "Dining", 50)
        self.at.run()
        self.quick_add("lunch 45")
        self.at.get_by_key("confirm_box").button[0].click().run()
        self.assertTrue(any("Dining has used 90%" in str(t.value) for t in self.at.toast))

    def test_forecast_page(self):
        self.go("Plan ahead")
        page = self.htmls() + self.texts("info")
        self.assertTrue("On pace for" in page or "forecast starts" in page)
        self.assertIn("a year", page)

    def test_advisor_savings_plan(self):
        self.go("AI advisor")
        answer = "Try cooking at home. Shopping cut is in the plan."
        with mock.patch("advisor.requests.post", return_value=ollama_reply(answer)) as post:
            self.at.chat_input[0].set_value("How can I save $200 this month?").run()
        self.assertFalse(self.at.exception, self.at.exception)
        sent = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertIn("[App note: Savings plan", sent)
        self.assertIn("Plan to save", self.texts("markdown"))
        plan = self.at.dataframe[0].value
        self.assertEqual(plan.iloc[-1]["Spend less by"], 200.0)
        self.assertNotIn("Bills", list(plan["Category"]))
        self.assertIn("Answered by Qwen 2.5 3B on this computer (8-bit)", self.texts("caption"))

    def test_advisor_chart_and_number_check(self):
        self.go("AI advisor")
        replies = [ollama_reply('{"kind": "category_totals", "format": "chart"}'),
                   ollama_reply("Groceries is your biggest category, but you also spent $818.64 somewhere.")]
        with mock.patch("advisor.requests.post", side_effect=replies):
            self.at.chat_input[0].set_value("Show my spending by category as a chart").run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertIn("Spending by category", self.texts("markdown"))
        self.assertIn("$818.64, which doesn't come from your data", self.texts("warning"))

    def test_follow_up_may_quote_earlier_plan(self):
        self.go("AI advisor")
        with mock.patch("advisor.requests.post", return_value=ollama_reply("Here is the plan.")):
            self.at.chat_input[0].set_value("How can I save $200 this month?").run()
        cut = self.at.dataframe[0].value.iloc[0]["Spend less by"]
        follow_up = f"As planned, spend ${cut:.2f} less on that category."
        with mock.patch("advisor.requests.post", return_value=ollama_reply(follow_up)):
            self.at.chat_input[0].set_value("Which category matters most?").run()
        self.assertFalse(self.at.warning, [w.value for w in self.at.warning])

    def test_correction_fixes_past_and_future(self):
        self.go("Transactions")
        self.assertIn("Worth a second look (1)", [t.label for t in self.at.tabs])
        self.quick_add("amazon.com 20")
        box = self.at.get_by_key("confirm_box")
        box.selectbox[0].set_value("Shopping")
        box.button[0].click().run()
        txns = db.get_transactions(self.uid)
        amazon = txns[txns["merchant"].str.upper() == "AMAZON.COM"]
        self.assertEqual(len(amazon), 2)
        self.assertEqual(set(amazon["predicted_category"]), {"Shopping"})      # the earlier one was fixed
        self.assertTrue((amazon["confidence"] == 1.0).all())
        self.assertTrue(any("updated 1 earlier one" in str(t.value) for t in self.at.toast))
        self.go("Transactions")
        self.assertIn("Worth a second look (0)", [t.label for t in self.at.tabs])  # disappears when corrected
        self.quick_add("amazon.com 15")                                          # future ones follow the rule
        self.assertEqual(self.at.get_by_key("confirm_box").selectbox[0].value, "Shopping")
        self.assertIn("comes from a correction you made earlier", self.texts("caption"))

    def test_over_budget_alert_after_saving(self):
        db.set_budget(self.uid, "Dining", 40)
        self.at.run()
        self.quick_add("lunch 45")
        self.at.get_by_key("confirm_box").button[0].click().run()
        toasts = [str(t.value).replace("\\$", "$") for t in self.at.toast]
        self.assertIn("Dining is $5 over its $40 budget.", toasts)

    def test_local_model_box_switches_model(self):
        self.go("AI advisor")
        self.at.text_input(key="local_model").set_value("llama3.2:3b").run()
        with mock.patch("advisor.requests.post", return_value=ollama_reply("Fine.")) as post:
            self.at.chat_input[0].set_value("Am I on track with my budgets?").run()
        self.assertEqual(post.call_args.kwargs["json"]["model"], "llama3.2:3b")
        self.assertIn("Answered by Llama 3.2 3B on this computer (4-bit)", self.texts("caption"))
        self.at.text_input(key="local_model").set_value("qwen2.5:3b-instruct-q8_0").run()

    def test_groq_without_key_explains(self):
        self.go("AI advisor")
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            self.at.radio(key="engine").set_value("Groq (free cloud)").run()
            self.at.chat_input[0].set_value("Am I on track with my budgets?").run()
        self.assertIn("Add your Groq API key in the sidebar", self.texts("error"))
        self.at.radio(key="engine").set_value("Local (Ollama)").run()

    def test_reuploading_csv_adds_nothing(self):
        csv = b"date,merchant,amount\n05/10/2026,BUNNINGS,45.50\n06/10/2026,ALDI,30\n06/10/2026,ALDI,30\n"
        before = len(db.get_transactions(self.uid))
        self.at.file_uploader[0].upload("bank.csv", csv, "text/csv").run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual(len(db.get_transactions(self.uid)), before + 3)
        again = AppTest.from_file(APP, default_timeout=120)   # a new browser session, same account
        again.session_state["user_id"] = self.uid
        again.run()
        again.file_uploader[0].upload("bank.csv", csv, "text/csv").run()
        self.assertEqual(len(db.get_transactions(self.uid)), before + 3)
        self.assertTrue(any("Skipped 3 already in your account" in str(t.value) for t in again.toast))

    def test_advisor_unreachable(self):
        import requests
        self.go("AI advisor")
        with mock.patch("advisor.requests.post", side_effect=requests.ConnectionError()):
            self.at.chat_input[0].set_value("What changed most since last month?").run()
        self.assertIn("Make sure Ollama is running", self.texts("error"))


if __name__ == "__main__":
    unittest.main()
