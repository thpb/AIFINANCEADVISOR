import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from helpers import TODAY  # noqa: F401  (puts src on the path)

import advisor
import categorizer
from category_hints import apply_hints, find_hint


class HintTests(unittest.TestCase):
    def test_plain_words_and_australian_merchants(self):
        cases = {"groceries": "Groceries", "coles": "Groceries", "woolies": "Groceries", "coffee": "Dining",
                 "opal": "Transport", "petrol": "Transport", "rent": "Bills", "telstra": "Bills",
                 "netflix": "Subscriptions", "hoyts": "Entertainment", "kmart": "Shopping", "big w": "Shopping"}
        for text, cat in cases.items():
            self.assertEqual(find_hint(text)[0], cat, text)

    def test_australian_slang_and_health(self):
        cases = {"maccas": "Dining", "brekkie": "Dining", "servo": "Transport", "rego": "Transport",
                 "dentist": "Other", "chemist": "Other", "vet": "Other"}
        for text, cat in cases.items():
            self.assertEqual(find_hint(text)[0], cat, text)
        self.assertEqual(find_hint("velvet cushion"), (None, ""))

    def test_longest_phrase_wins(self):
        self.assertEqual(find_hint("uber eats"), ("Dining", "uber eats"))
        self.assertEqual(find_hint("uber"), ("Transport", "uber"))
        self.assertEqual(find_hint("gas bill"), ("Bills", "gas bill"))

    def test_whole_words_only(self):
        self.assertEqual(find_hint("busybee"), (None, ""))
        self.assertEqual(find_hint("stanley tools"), (None, ""))
        self.assertEqual(find_hint("NETFLIX.COM")[0], "Subscriptions")

    def test_ambiguous_shops_left_to_model(self):
        for name in ["AMAZON.COM", "costco", "target"]:
            self.assertEqual(find_hint(name), (None, ""))

    def test_apply_hints_sets_confidence_and_empty_string(self):
        df = pd.DataFrame({"merchant": ["coffee", "zorbo"], "predicted_category": ["Other", "Other"],
                           "confidence": [0.3, 0.4]})
        out = apply_hints(df)
        self.assertEqual(out.loc[0, "predicted_category"], "Dining")
        self.assertEqual(out.loc[0, "confidence"], 0.95)
        self.assertEqual(out.loc[1, "hint_word"], "")
        self.assertFalse(out.loc[1, "hint_word"])

    def test_unknown_category_not_assigned(self):
        df = pd.DataFrame({"merchant": ["coffee"], "predicted_category": ["Other"], "confidence": [0.3]})
        out = apply_hints(df, known_categories=["Other", "Groceries"])
        self.assertEqual(out.loc[0, "predicted_category"], "Other")
        self.assertEqual(out.loc[0, "hint_word"], "")


class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 384))


class FakeEncoder:
    classes_ = np.array(["Dining", "Other", "Transport"])

    def inverse_transform(self, idx):
        return self.classes_[idx]


class FakeModel:
    def predict_proba(self, x):
        return np.tile([0.2, 0.7, 0.1], (len(x), 1))      # always "Other", 70% sure


class PipelineTests(unittest.TestCase):
    def frame(self, merchants):
        return pd.DataFrame({"date": ["2026-10-01"] * len(merchants), "merchant": merchants,
                             "amount": [5.0] * len(merchants)})

    def test_order_model_hint_rule(self):
        df = categorizer.categorize(self.frame(["zorbo", "coffee", "SQ *COFFEE #12"]), FakeEmbedder(), FakeModel(),
                                    FakeEncoder(), rules={"COFFEE": "Transport"})
        self.assertEqual(df.loc[0, "predicted_category"], "Other")
        self.assertAlmostEqual(df.loc[0, "confidence"], 0.7)
        self.assertEqual(df.loc[0, "hint_word"], "")
        # "coffee" has a hint, but the user's rule wins for both spellings.
        self.assertEqual(list(df["predicted_category"][1:]), ["Transport", "Transport"])
        self.assertTrue(df.loc[1, "rule_applied"])
        self.assertEqual(df.loc[1, "confidence"], 1.0)

    def test_hint_without_rule(self):
        df = categorizer.categorize(self.frame(["coffee"]), FakeEmbedder(), FakeModel(), FakeEncoder())
        self.assertEqual((df.loc[0, "model_category"], df.loc[0, "predicted_category"]), ("Other", "Dining"))
        self.assertEqual(df.loc[0, "hint_word"], "coffee")
        self.assertEqual(df.loc[0, "confidence"], 0.95)

    def test_feature_columns(self):
        self.assertIn("is_ambiguous_merchant", categorizer.FEATURE_COLS)
        self.assertEqual(len(categorizer.FEATURE_COLS), 6)


class ModelNameAndContextTests(unittest.TestCase):
    def test_answered_by_labels(self):
        f = advisor.friendly_name
        self.assertEqual(f("ollama", "qwen2.5:3b-instruct-q8_0"), "Qwen 2.5 3B on this computer (8-bit)")
        self.assertEqual(f("ollama", "qwen2.5:3b-instruct"), "Qwen 2.5 3B on this computer (4-bit)")
        self.assertEqual(f("ollama", "QWEN2.5:3B-INSTRUCT-Q5_K_M"), "Qwen 2.5 3B on this computer (5-bit)")
        self.assertEqual(f("ollama", "llama3.2:3b"), "Llama 3.2 3B on this computer (4-bit)")
        self.assertEqual(f("ollama", "mistral"), "mistral on this computer")
        self.assertEqual(f("groq", "llama-3.3-70b-versatile"), "Llama 3.3 70B via Groq")
        self.assertEqual(f("groq", "openai/gpt-oss-120b"), "GPT-OSS 120B via Groq")
        self.assertEqual(f("groq", "new-model"), "new-model via Groq")

    def test_default_local_model_and_context(self):
        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": ""}):
            self.assertEqual(advisor.local_model(), "qwen2.5:3b-instruct-q8_0")
        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.2:3b"}):
            self.assertEqual(advisor.local_model(), "llama3.2:3b")
        self.assertEqual(advisor.LOCAL_CONTEXT, 8192)

    def test_num_ctx_sent_on_every_call(self):
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"message": {"content": "{}"}}
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "ollama"}), \
                mock.patch("advisor.requests.post", return_value=reply) as post:
            advisor.chat([{"role": "user", "content": "hi"}])
            advisor._call_llm_json([{"role": "user", "content": "hi"}])
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["json"]["options"]["num_ctx"], 8192)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["format"], "json")
        self.assertEqual(post.call_args_list[1].kwargs["json"]["options"]["temperature"], 0)

    def test_trim_keeps_data_and_drops_oldest(self):
        head = [{"role": "system", "content": "S"}, {"role": "user", "content": "DATA"},
                {"role": "assistant", "content": "ok"}]
        pairs = []
        for i in range(20):
            pairs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        out = advisor._trim_history(head + pairs + [{"role": "user", "content": "new"}])
        self.assertEqual(out[:3], head)
        self.assertEqual(len(out), 3 + 24 + 1)
        self.assertEqual(out[3]["content"], "q8")
        self.assertEqual(out[-1]["content"], "new")

    def test_trim_by_size(self):
        head = [{"role": "system", "content": "S"}, {"role": "user", "content": "D" * 20000},
                {"role": "assistant", "content": "ok"}]
        big = [{"role": "user", "content": "q" * 3000}, {"role": "assistant", "content": "a" * 3000}] * 3
        out = advisor._trim_history(head + big + [{"role": "user", "content": "new"}])
        self.assertEqual(out[1]["content"], "D" * 20000)
        self.assertLessEqual(sum(len(m["content"]) for m in out), 8192 * 4 * 0.75)
        self.assertEqual(out[-1]["content"], "new")


if __name__ == "__main__":
    unittest.main()
