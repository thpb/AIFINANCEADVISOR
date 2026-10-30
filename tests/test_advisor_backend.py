import os
import unittest
from unittest import mock

from helpers import TODAY  # noqa: F401

import advisor


def response(status, body):
    r = mock.Mock(status_code=status, text=str(body))
    r.json.return_value = body
    return r


def ok(text):
    return response(200, {"choices": [{"message": {"content": text}}]})


MSG = [{"role": "user", "content": "hi"}]


class BackendTests(unittest.TestCase):
    def setUp(self):
        advisor._groq_models.clear()
        self.env = mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test", "GROQ_MODEL": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_backend_switches_at_runtime(self):
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "groq"}), \
                mock.patch("advisor.requests.post", return_value=ok("from groq")) as post:
            self.assertEqual(advisor.chat(MSG), "from groq")
            self.assertIn("groq.com", post.call_args.args[0])
        local = response(200, {"message": {"content": "from ollama"}})
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "ollama"}), \
                mock.patch("advisor.requests.post", return_value=local) as post:
            self.assertEqual(advisor.chat(MSG), "from ollama")
            self.assertIn("11434", post.call_args.args[0])
            self.assertIn("on this computer", advisor.last_model())

    def test_key_header(self):
        with mock.patch("advisor.requests.post", return_value=ok("x")) as post:
            advisor.groq_chat(MSG)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer gsk_test")
        self.assertEqual(advisor.last_model(), "Llama 3.3 70B via Groq")

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaises(advisor.AdvisorError) as ctx:
                advisor.groq_chat(MSG)
        self.assertEqual(str(ctx.exception), "Add your Groq API key in Settings (or set GROQ_API_KEY).")

    def test_401(self):
        with mock.patch("advisor.requests.post", return_value=response(401, {})):
            with self.assertRaises(advisor.AdvisorError) as ctx:
                advisor.groq_chat(MSG)
        self.assertEqual(str(ctx.exception), "Groq rejected the API key. Check that you copied the whole key.")

    def test_429(self):
        with mock.patch("advisor.requests.post", return_value=response(429, {})):
            with self.assertRaises(advisor.AdvisorError) as ctx:
                advisor.groq_chat(MSG)
        self.assertIn("free limit", str(ctx.exception))

    def test_404_falls_back_to_available_model(self):
        retired = response(404, {"error": {"message": "model decommissioned"}})
        models = response(200, {"data": [{"id": "whisper-large-v3"}, {"id": "openai/gpt-oss-120b"},
                                         {"id": "llama-3.1-8b-instant"}]})
        with mock.patch("advisor.requests.post", side_effect=[retired, ok("hello")]) as post, \
                mock.patch("advisor.requests.get", return_value=models):
            self.assertEqual(advisor.groq_chat(MSG), "hello")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(advisor.last_model(), "GPT-OSS 120B via Groq")
        # Cached per key: the next call goes straight to the working model.
        with mock.patch("advisor.requests.post", return_value=ok("again")) as post:
            advisor.groq_chat(MSG)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "openai/gpt-oss-120b")

    def test_fallback_skips_non_chat_models(self):
        models = response(200, {"data": [{"id": "whisper-large-v3"}, {"id": "llama-guard-4"}, {"id": "qwen-x"}]})
        with mock.patch("advisor.requests.get", return_value=models):
            self.assertEqual(advisor._pick_groq_model("k"), "qwen-x")

    def test_other_errors_show_groq_message(self):
        with mock.patch("advisor.requests.post", return_value=response(503, {"error": {"message": "over capacity"}})):
            with self.assertRaises(advisor.AdvisorError) as ctx:
                advisor.groq_chat(MSG)
        self.assertEqual(str(ctx.exception), "Groq error 503: over capacity")

    def test_json_mode_for_groq(self):
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "groq"}), \
                mock.patch("advisor.requests.post", return_value=ok('{"kind": "none"}')) as post:
            self.assertEqual(advisor._call_llm_json(MSG), {"kind": "none"})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0)

    def test_ollama_unreachable(self):
        import requests
        with mock.patch.dict(os.environ, {"ADVISOR_BACKEND": "ollama"}), \
                mock.patch("advisor.requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(advisor.AdvisorError):
                advisor.chat(MSG)


if __name__ == "__main__":
    unittest.main()
