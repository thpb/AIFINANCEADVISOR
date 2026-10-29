"""The AI advisor: calls a local model (Ollama) or Groq, keeps the conversation.

The model only gives ADVICE. Every number it may quote is calculated by the
app and handed over as text; money_facts.unverified_amounts checks the answer.
"""
import json
import os

import requests

import money_facts
from budgets import category_column, money, with_month

OLLAMA_URL = "http://localhost:11434/api/chat"
GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5:3b-instruct-q8_0"
LOCAL_CONTEXT = int(os.environ.get("OLLAMA_NUM_CTX") or 8192)
ADVICE_TEMPERATURE = 0.2
MAX_PAIRS = 12
GROQ_PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-70b-versatile",
                  "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
NOT_CHAT = ("whisper", "guard", "tts", "embed")

NOTE_MARK = "\n\n[App note: "

SYSTEM_PROMPT = """You are a friendly, careful personal finance adviser in a budgeting app. Your job is ADVICE:
practical habits, where to look, what to try, what to watch. The app does all calculations; you do not.
Rules:
1. Only quote numbers that appear in the data summary or in an app note. Copy them exactly. Never add,
   subtract, divide, average or estimate a number yourself, and never invent one. If an answer would need
   a new calculation, say what the app shows instead (for example the Plan ahead page's what-if sliders).
2. Prefer the facts the app already calculated: monthly averages, changes from last month, and what a
   10% or 20% cut would save.
3. When the app note gives a plan or a chart, explain it and give 2-3 practical tips. Do not change its numbers.
4. If the data does not answer the question, say so. If the question is unclear, ask one short question.
5. Keep it short: 2-5 sentences or a short bullet list. A small Markdown table is fine for numbers copied
   from the summary. No headings, no LaTeX, no code blocks. Write money like $150 or $1,250.50.
You may refer back to earlier questions in this conversation."""

ADVICE_PROMPT = ("Give me 2-3 practical pieces of advice based on my data: categories that went up since last "
                 "month, my biggest categories, or subscriptions worth reviewing. For each, quote the app's "
                 "numbers exactly as written in the summary (do not calculate anything) and suggest one "
                 "concrete thing to try. No generic advice.")

FRIENDLY_LOCAL = {
    "qwen2.5:3b-instruct-q8_0": "Qwen 2.5 3B on this computer (8-bit)",
    "qwen2.5:3b-instruct": "Qwen 2.5 3B on this computer (4-bit)",
    "qwen2.5:3b-instruct-q5_k_m": "Qwen 2.5 3B on this computer (5-bit)",
    "llama3.2:3b": "Llama 3.2 3B on this computer (4-bit)",
    "llama3.2": "Llama 3.2 3B on this computer (4-bit)",
}
FRIENDLY_GROQ = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B via Groq",
    "openai/gpt-oss-120b": "GPT-OSS 120B via Groq",
}

_groq_models = {}          # api key -> model that works for it
_last = {"engine": None, "model": None}


class AdvisorError(Exception):
    """A problem the user can fix; the message says how."""


def backend():
    return (os.environ.get("ADVISOR_BACKEND") or "ollama").strip().lower()


def local_model():
    return (os.environ.get("OLLAMA_MODEL") or "").strip() or DEFAULT_LOCAL_MODEL


def friendly_name(engine, model_id):
    if engine == "groq":
        return FRIENDLY_GROQ.get(model_id, f"{model_id} via Groq")
    return FRIENDLY_LOCAL.get(model_id.lower(), f"{model_id} on this computer")


def last_model():
    if not _last["model"]:
        return ""
    return friendly_name(_last["engine"], _last["model"])


# ---------- engines ----------

def ollama_chat(messages, temperature=ADVICE_TEMPERATURE, json_mode=False):
    model = local_model()
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_ctx": LOCAL_CONTEXT}}
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.RequestException:
        raise AdvisorError("Can't reach Ollama on this computer.") from None
    if r.status_code == 404:
        raise AdvisorError(f'The model "{model}" isn\'t installed. Run `ollama pull {model}` '
                           "or change the Local model box.")
    if r.status_code != 200:
        raise AdvisorError(f"Ollama error {r.status_code}: {r.text[:200]}")
    _last.update(engine="ollama", model=model)
    return r.json()["message"]["content"]


def _groq_key():
    return (os.environ.get("GROQ_API_KEY") or "").strip()


def _groq_post(key, model, messages, extra):
    payload = {"model": model, "messages": messages, "temperature": ADVICE_TEMPERATURE, **extra}
    return requests.post(f"{GROQ_URL}/chat/completions", json=payload, timeout=120,
                         headers={"Authorization": f"Bearer {key}"})


def _pick_groq_model(key):
    """Groq renames and retires models; ask which ones exist now."""
    try:
        r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return None
    for p in GROQ_PREFERRED:
        if p in ids:
            return p
    chat_ids = [i for i in ids if not any(w in i.lower() for w in NOT_CHAT)]
    return chat_ids[0] if chat_ids else None


def _groq_error(r):
    if r.status_code == 401:
        return AdvisorError("Groq rejected the API key. Check that you copied the whole key.")
    if r.status_code == 429:
        return AdvisorError("Groq's free limit was reached. Wait a minute and try again.")
    try:
        message = r.json()["error"]["message"]
    except Exception:
        message = r.text[:200]
    return AdvisorError(f"Groq error {r.status_code}: {message}")


def groq_chat(messages, **extra):
    key = _groq_key()
    if not key:
        raise AdvisorError("Add your Groq API key in the sidebar (or set GROQ_API_KEY).")
    model = os.environ.get("GROQ_MODEL") or _groq_models.get(key) or GROQ_PREFERRED[0]
    try:
        r = _groq_post(key, model, messages, extra)
        if r.status_code in (400, 404):
            fallback = _pick_groq_model(key)
            if fallback and fallback != model:
                model = fallback
                r = _groq_post(key, model, messages, extra)
    except requests.RequestException:
        raise AdvisorError("Can't reach Groq. Check your internet connection.") from None
    if r.status_code != 200:
        raise _groq_error(r)
    _groq_models[key] = model
    _last.update(engine="groq", model=model)
    return r.json()["choices"][0]["message"]["content"]


def chat(messages, temperature=ADVICE_TEMPERATURE):
    if backend() == "groq":
        return groq_chat(messages, temperature=temperature)
    return ollama_chat(messages, temperature=temperature)


def _call_llm_json(messages):
    """A JSON-only reply at temperature 0, parsed into a dict."""
    if backend() == "groq":
        text = groq_chat(messages, response_format={"type": "json_object"}, temperature=0)
    else:
        text = ollama_chat(messages, temperature=0, json_mode=True)
    return json.loads(text)


# ---------- conversation ----------

def summarize_spending(df):
    if df.empty:
        return "No transactions yet."
    d = with_month(df)
    cat = category_column(d)
    lines = [f"{len(d)} transactions from {d['date'].min():%d/%m/%Y} to {d['date'].max():%d/%m/%Y}.",
             "", "Total spending by category (all months):"]
    for c, v in d.groupby(cat)["amount"].sum().sort_values(ascending=False).items():
        lines.append(f"- {c}: {money(v)}")
    lines += ["", "Spending by month and category:"]
    for mo, rows in d.groupby("month"):
        parts = ", ".join(f"{c} {money(v)}" for c, v in rows.groupby(cat)["amount"].sum().items())
        lines.append(f"- {mo}: {parts} (total {money(rows['amount'].sum())})")
    if "is_recurring" in d.columns:
        rec = d[d["is_recurring"].fillna(0).astype(int) == 1]
        if not rec.empty:
            lines += ["", "Recurring payments (same merchant and amount more than once):"]
            for (m, a), rows in rec.groupby(["merchant", "amount"]):
                lines.append(f"- {m}: {money(a)} ({len(rows)} times)")
    return "\n".join(lines)


def data_summary(df, extra_context="", today=None):
    from datetime import date
    parts = [summarize_spending(df), money_facts.facts_for_advisor(df, today or date.today())]
    if extra_context:
        parts.append(extra_context)
    return "\n\n".join(parts)


def start_conversation(df, extra_context="", today=None):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Here is my categorized spending data:\n" + data_summary(df, extra_context, today)},
        {"role": "assistant", "content": "Thanks, I can see your spending data. Ask me anything about it."},
    ]


def _chars(messages):
    return sum(len(m["content"]) for m in messages)


def _trim_history(messages):
    """Always keep system + data; keep at most 12 recent pairs; drop the oldest until it fits.

    Ollama silently cuts the OLDEST text when a request is too long, which would remove the spending data.
    """
    head, rest = messages[:3], messages[3:]
    keep = MAX_PAIRS * 2 + (len(rest) % 2)
    rest = rest[-keep:] if len(rest) > keep else rest
    budget = LOCAL_CONTEXT * 4 * 0.75
    while len(rest) > 1 and _chars(head) + _chars(rest) > budget:
        rest = rest[2:]
    return head + rest


def ask_advisor(question, history, note=""):
    content = question + (NOTE_MARK + note + "]" if note else "")
    messages = history + [{"role": "user", "content": content}]
    answer = chat(_trim_history(messages)).strip()
    return answer, messages + [{"role": "assistant", "content": answer}]


def get_advice(history):
    return ask_advisor(ADVICE_PROMPT, history)


def pick_visual(question, categories, months):
    """Let the model fill a small chart request; code validates it. None on any failure."""
    import advisor_charts
    messages = [{"role": "system", "content": advisor_charts.router_prompt(categories, months)},
                {"role": "user", "content": question}]
    try:
        spec = advisor_charts.validate_spec(_call_llm_json(messages), categories, months)
    except Exception:
        return None
    if spec and not spec["categories"] and spec["kind"] in ("monthly_spending", "category_totals"):
        # Small models often drop the filter; the question itself names the categories.
        spec["categories"] = advisor_charts.mentioned_categories(question, categories)
    return spec
