import unittest
from datetime import date, timedelta

from helpers import TODAY

import nl_entry
from nl_entry import EXAMPLE_HINT, looks_like_income, parse_entry


def p(text, **kw):
    kw.setdefault("use_llm", False)
    return parse_entry(text, today=TODAY, **kw)


class AmountTests(unittest.TestCase):
    def test_plain_number(self):
        self.assertEqual(p("coffee 5")["amount"], 5.0)

    def test_dollar_sign_and_cents(self):
        self.assertEqual(p("paid $12.50 at Starbucks")["amount"], 12.5)

    def test_decimal_without_sign(self):
        self.assertEqual(p("coles 82.40")["amount"], 82.4)

    def test_bucks(self):
        e = p("50 bucks petrol")
        self.assertEqual(e["amount"], 50.0)
        self.assertEqual(e["merchant"], "petrol")

    def test_thousands_comma(self):
        self.assertEqual(p("rent 1,200")["amount"], 1200.0)

    def test_aud_prefix(self):
        e = p("aud 30 taxi")
        self.assertEqual(e["amount"], 30.0)
        self.assertEqual(e["merchant"], "taxi")

    def test_quantity_kept_in_merchant(self):
        e = p("2 coffees 9")
        self.assertEqual(e["amount"], 9.0)
        self.assertEqual(e["merchant"], "2 coffees")

    def test_zero_refused(self):
        with self.assertRaises(ValueError):
            p("coffee 0")

    def test_negative_refused(self):
        with self.assertRaises(ValueError):
            p("coffee -5")

    def test_max_amount(self):
        self.assertEqual(p("house 1,000,000")["amount"], 1_000_000)
        with self.assertRaises(ValueError):
            p("house 1,000,001")


class DateTests(unittest.TestCase):
    def test_default_today(self):
        self.assertEqual(p("coffee 5")["date"], TODAY)

    def test_today_tonight(self):
        self.assertEqual(p("lunch 18 today")["date"], TODAY)
        self.assertEqual(p("dinner 40 tonight")["date"], TODAY)

    def test_yesterday_and_last_night(self):
        y = TODAY - timedelta(days=1)
        self.assertEqual(p("spent 50 on groceries yesterday")["date"], y)
        self.assertEqual(p("pizza 20 last night")["date"], y)

    def test_day_before_yesterday(self):
        self.assertEqual(p("the day before yesterday taxi 30")["date"], TODAY - timedelta(days=2))

    def test_days_and_weeks_ago(self):
        self.assertEqual(p("coffee 5 3 days ago")["date"], TODAY - timedelta(days=3))
        self.assertEqual(p("shoes 80 a week ago")["date"], TODAY - timedelta(days=7))
        self.assertEqual(p("shoes 80 two weeks ago")["date"], TODAY - timedelta(days=14))

    def test_last_weekday_strictly_before(self):
        self.assertEqual(p("uber 23 last friday")["date"], date(2026, 10, 23))
        self.assertEqual(p("uber 23 last thursday")["date"], date(2026, 10, 22))   # today is Thursday

    def test_on_weekday_includes_today(self):
        self.assertEqual(p("uber 23 on thursday")["date"], TODAY)
        self.assertEqual(p("uber 23 friday")["date"], date(2026, 10, 23))

    def test_abbreviation_needs_prefix(self):
        self.assertEqual(p("movie 20 on fri")["date"], date(2026, 10, 23))
        self.assertEqual(p("movie 20 last sat")["date"], date(2026, 10, 24))
        e = p("sun hat 20")
        self.assertEqual(e["date"], TODAY)
        self.assertEqual(e["merchant"], "sun hat")

    def test_iso(self):
        self.assertEqual(p("coffee 5 2026-10-05")["date"], date(2026, 10, 5))

    def test_day_first_numeric(self):
        self.assertEqual(p("coffee 5 12/10")["date"], date(2026, 10, 12))
        self.assertEqual(p("coffee 5 12/10/2026")["date"], date(2026, 10, 12))
        self.assertEqual(p("coffee 5 12-10-26")["date"], date(2026, 10, 12))

    def test_no_year_future_rolls_back(self):
        self.assertEqual(p("coffee 5 25/12")["date"], date(2025, 12, 25))

    def test_month_names(self):
        self.assertEqual(p("coffee 5 5 oct")["date"], date(2026, 10, 5))
        self.assertEqual(p("coffee 5 october 5th")["date"], date(2026, 10, 5))
        self.assertEqual(p("coffee 5 on 3 november")["date"], date(2025, 11, 3))

    def test_future_date_is_warning(self):
        e = p("concert 90 2026-12-01")
        self.assertEqual(e["date"], date(2026, 12, 1))
        self.assertTrue(any("future" in w for w in e["warnings"]))


class MerchantTests(unittest.TestCase):
    def test_spec_example(self):
        e = p("spent 50 on groceries yesterday")
        self.assertEqual((e["merchant"], e["amount"], e["kind"], e["method"]), ("groceries", 50.0, "expense", "rules"))

    def test_paid_at(self):
        e = p("paid $12.50 at Starbucks yesterday")
        self.assertEqual(e["merchant"], "Starbucks")
        self.assertEqual(e["kind"], "expense")

    def test_multi_word(self):
        self.assertEqual(p("movie tickets 32")["merchant"], "movie tickets")

    def test_empty_and_too_long(self):
        with self.assertRaises(ValueError) as ctx:
            p("   ")
        self.assertTrue(str(ctx.exception).endswith(EXAMPLE_HINT))
        with self.assertRaises(ValueError):
            p("coffee 5 " + "x" * 300)

    def test_no_amount_without_llm(self):
        with self.assertRaises(ValueError) as ctx:
            p("coffee yesterday")
        self.assertIn("amount", str(ctx.exception))


class LLMFallbackTests(unittest.TestCase):
    def test_word_amount_trusts_model(self):
        fake = lambda text: {"amount": 30, "merchant": "taxi", "date_phrase": "yesterday"}  # noqa: E731
        e = p("thirty bucks on a taxi yesterday", use_llm=True, llm_fn=fake)
        self.assertEqual((e["amount"], e["merchant"], e["method"]), (30.0, "taxi", "llm"))
        self.assertEqual(e["date"], TODAY - timedelta(days=1))

    def test_model_date_phrase_goes_through_rules(self):
        fake = lambda text: {"amount": 12, "merchant": "lunch", "date_phrase": "last friday"}  # noqa: E731
        e = p("twelve dollars lunch the friday before", use_llm=True, llm_fn=fake)
        self.assertEqual(e["date"], date(2026, 10, 23))

    def test_model_only_fills_merchant_when_rules_found_amount(self):
        fake = lambda text: {"amount": 999, "merchant": "bakery", "date_phrase": ""}  # noqa: E731
        e = p("spent 7 at the", use_llm=True, llm_fn=fake)
        self.assertEqual((e["amount"], e["merchant"]), (7.0, "bakery"))

    def test_model_returns_json_text(self):
        e = p("forty for books", use_llm=True, llm_fn=lambda t: '{"amount": "40", "merchant": "books"}')
        self.assertEqual(e["amount"], 40.0)

    def test_unreachable_model_message(self):
        def down(text):
            raise ConnectionError("no ollama")
        with self.assertRaises(ValueError) as ctx:
            p("thirty bucks on a taxi", use_llm=True, llm_fn=down)
        msg = str(ctx.exception)
        self.assertIn("isn't running", msg)
        self.assertTrue(msg.endswith(EXAMPLE_HINT))

    def test_model_without_amount(self):
        with self.assertRaises(ValueError):
            p("something nice", use_llm=True, llm_fn=lambda t: {"amount": None, "merchant": "x"})

    def test_default_llm_is_used(self):
        orig = nl_entry.llm_extract
        nl_entry.llm_extract = lambda t: {"amount": 3, "merchant": "gum"}
        try:
            self.assertEqual(p("three dollars gum", use_llm=True)["merchant"], "gum")
        finally:
            nl_entry.llm_extract = orig


class IncomeTests(unittest.TestCase):
    def test_income_words(self):
        for s in ["got paid 2000", "salary 3000", "bonus 500", "sold my bike 200", "dividends 40",
                  "freelance job 300", "interest earned 12", "payday 1500", "received 50 from mum"]:
            self.assertTrue(looks_like_income(s), s)

    def test_not_income(self):
        for s in ["paid 12.50 at Starbucks", "refund 20 from target", "coffee 5"]:
            self.assertFalse(looks_like_income(s), s)

    def test_got_paid_salary(self):
        e = p("got paid 2000 salary")
        self.assertEqual((e["kind"], e["merchant"], e["amount"]), ("income", "salary", 2000.0))

    def test_got_paid_without_source(self):
        self.assertEqual(p("got paid 2000")["merchant"], "salary")
        self.assertEqual(p("received 50")["merchant"], "income")

    def test_income_source_kept(self):
        self.assertEqual(p("received 50 from mum")["merchant"], "mum")


if __name__ == "__main__":
    unittest.main()
