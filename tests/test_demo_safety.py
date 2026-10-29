"""Things that must show a friendly message instead of an error during a live demo."""
import io
import unittest

import requests
from helpers import TODAY

import advisor
from clean_data import read_bank_csv
from nl_entry import EXAMPLE_HINT, parse_entry


def csv(text):
    return io.StringIO(text)


class BankCsvTests(unittest.TestCase):
    def test_good_file_with_money_formats(self):
        rows, skipped = read_bank_csv(csv("Date,Merchant,Amount\n05/10/2026,COLES,\"$1,200.50\"\n"
                                          "2026-10-06,ALDI,-12\n"))
        self.assertEqual(skipped, 0)
        self.assertEqual(list(rows["amount"]), [1200.5, -12.0])
        self.assertEqual(rows.loc[0, "date"].strftime("%Y-%m-%d"), "2026-10-05")     # day-first

    def test_bad_rows_are_skipped_not_fatal(self):
        rows, skipped = read_bank_csv(csv("date,merchant,amount\n05/10/2026,COLES,20\nnot a date,ALDI,5\n"
                                          "06/10/2026,,5\n07/10/2026,IGA,abc\n08/10/2026,HOUSE,2000000\n"))
        self.assertEqual((len(rows), skipped), (1, 4))

    def test_missing_columns(self):
        with self.assertRaises(ValueError) as ctx:
            read_bank_csv(csv("when,what\n05/10/2026,COLES\n"))
        self.assertIn("missing these columns: amount, date, merchant", str(ctx.exception))

    def test_empty_and_unreadable(self):
        with self.assertRaises(ValueError) as ctx:
            read_bank_csv(csv(""))
        self.assertEqual(str(ctx.exception), "That file is empty.")
        with self.assertRaises(ValueError) as ctx:
            read_bank_csv(csv("date,merchant,amount\nyesterday,COLES,lots\n"))
        self.assertIn("No rows could be read", str(ctx.exception))


class QuickAddMessageTests(unittest.TestCase):
    def test_model_not_running(self):
        def down(text):
            raise advisor.AdvisorUnreachable("Can't reach Ollama on this computer.")
        with self.assertRaises(ValueError) as ctx:
            parse_entry("thirty bucks on a taxi", today=TODAY, llm_fn=down)
        self.assertIn("isn't running", str(ctx.exception))

    def test_wrong_model_name_says_so(self):
        def missing(text):
            raise advisor.AdvisorError('The model "qwen9" isn\'t installed. Run `ollama pull qwen9`.')
        with self.assertRaises(ValueError) as ctx:
            parse_entry("thirty bucks on a taxi", today=TODAY, llm_fn=missing)
        msg = str(ctx.exception)
        self.assertIn('The model "qwen9" isn\'t installed', msg)
        self.assertNotIn("isn't running", msg)
        self.assertTrue(msg.endswith(EXAMPLE_HINT))

    def test_large_amounts(self):
        self.assertEqual(parse_entry("house deposit 999,999", today=TODAY, use_llm=False)["amount"], 999999.0)
        with self.assertRaises(ValueError) as ctx:
            parse_entry("house 1,000,001", today=TODAY, use_llm=False)
        self.assertIn("too large", str(ctx.exception))

    def test_unreachable_engines_raise_the_right_type(self):
        from unittest import mock
        with mock.patch("advisor.requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(advisor.AdvisorUnreachable):
                advisor.ollama_chat([{"role": "user", "content": "x"}])
            with mock.patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
                with self.assertRaises(advisor.AdvisorUnreachable):
                    advisor.groq_chat([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
