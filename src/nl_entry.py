"""Turn a typed sentence ("spent 50 on groceries yesterday") into a transaction.

Rules first, AI model second. Dates are ALWAYS calculated by this code, never
by the model: the model may only copy the date words it saw ("last friday"),
which then go back through the same rule-based date parser.
Idea credit: Spritan/expense_tracker (MIT) for typed-sentence entry and income.
"""
import calendar
import json
import re
from datetime import date, timedelta

EXAMPLE_HINT = 'Try something like "spent 50 on groceries yesterday".'
MAX_AMOUNT = 1_000_000
MAX_LENGTH = 200

FILLER = {
    "spent", "spend", "spending", "paid", "pay", "on", "at", "for", "bought", "buy", "got", "get", "dollar",
    "dollars", "bucks", "aud", "usd", "ago", "today", "yesterday", "tonight", "the", "a", "an", "my", "some",
    "i", "i've", "ive", "just", "from", "to", "in", "of", "and", "was", "it", "with", "me", "this", "that",
    "$", "-", "received", "earned",
}
INCOME_RE = re.compile(r"\b(salary|wages?|pay ?check|payday|got paid|income|earned|received|bonus|dividends?|"
                       r"interest earned|sold|freelance|allowance)\b", re.I)
SALARY_RE = re.compile(r"\b(salary|wages?|pay|paid|payday|pay ?check)\b", re.I)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_ABBR = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5,
                "sun": 6}
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}

_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_DAY_FULL = "|".join(WEEKDAYS)
_DAY_ANY = "|".join(WEEKDAYS + sorted(WEEKDAY_ABBR, key=len, reverse=True))
_NUM_WORD = "|".join(NUMBER_WORDS)

ISO_RE = re.compile(r"(?<![\d/-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d/-])")
NUMERIC_RE = re.compile(r"(?<![\d/.,$-])(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}|\d{2}))?(?![\d/-])")
DAY_MONTH_RE = re.compile(rf"(?<![a-z0-9$.])(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})(?:\s*,?\s*(\d{{4}}))?(?![a-z])", re.I)
MONTH_DAY_RE = re.compile(rf"(?<![a-z])({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?(?![a-z0-9.])", re.I)
AGO_RE = re.compile(rf"\b(\d+|{_NUM_WORD})\s+(days?|weeks?)\s+ago\b", re.I)
LAST_DAY_RE = re.compile(rf"\b(last|on|this past)\s+({_DAY_ANY})\b", re.I)
BARE_DAY_RE = re.compile(rf"\b({_DAY_FULL})\b", re.I)

AMOUNT_RE = re.compile(
    r"(?<![\w.,])(?P<pre>(?:aud|usd|a)\s*\$|(?:aud|usd)\s+|\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"(?:\s*(?P<suf>bucks|dollars?|aud|usd)\b)?(?![a-z0-9])", re.I)
NEGATIVE_RE = re.compile(r"(?<![\w])-\s*\$?\s*\d")

LLM_PROMPT = """Read a short note about money someone spent or received. Reply with JSON only:
{"amount": number or null, "merchant": "short name of the shop or thing", "date_phrase": "date words as written, or empty"}
Rules: write the amount as a number (thirty bucks -> 30). Do not calculate any date: copy date words such as
"yesterday" or "last friday" exactly as they appear. If there are no date words, use "".
"""


def looks_like_income(text):
    return bool(INCOME_RE.search(text or ""))


def _safe_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _roll_back(d, today):
    """A date with no year that would land in the future means last year."""
    if d and d > today:
        return _safe_date(d.year - 1, d.month, d.day)
    return d


def _year(text):
    y = int(text)
    return y + 2000 if y < 100 else y


def find_date(text, today):
    """Return (date or None, (start, end) or None). All date maths happens here."""
    m = ISO_RE.search(text)
    if m:
        d = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        if d:
            return d, m.span()
    for m in NUMERIC_RE.finditer(text):
        day, month = int(m[1]), int(m[2])
        if m[3]:
            d = _safe_date(_year(m[3]), month, day)
        else:
            d = _roll_back(_safe_date(today.year, month, day), today)
        if d:
            return d, m.span()
    m = re.search(r"\b(the\s+)?day\s+before\s+yesterday\b", text, re.I)
    if m:
        return today - timedelta(days=2), m.span()
    m = AGO_RE.search(text)
    if m:
        n = NUMBER_WORDS.get(m[1].lower()) or int(m[1])
        days = n * 7 if m[2].lower().startswith("week") else n
        return today - timedelta(days=days), m.span()
    for regex, day_group, month_group in ((MONTH_DAY_RE, 2, 1), (DAY_MONTH_RE, 1, 2)):
        m = regex.search(text)
        if m:
            month = MONTHS[m[month_group].lower()]
            if m[3]:
                d = _safe_date(int(m[3]), month, int(m[day_group]))
            else:
                d = _roll_back(_safe_date(today.year, month, int(m[day_group])), today)
            if d:
                return d, m.span()
    m = LAST_DAY_RE.search(text)
    if m:
        word = m[2].lower()
        wd = WEEKDAYS.index(word) if word in WEEKDAYS else WEEKDAY_ABBR[word]
        back = (today.weekday() - wd) % 7
        if m[1].lower() != "on" and back == 0:
            back = 7          # "last friday" is strictly before today
        return today - timedelta(days=back), m.span()
    m = BARE_DAY_RE.search(text)
    if m:
        back = (today.weekday() - WEEKDAYS.index(m[1].lower())) % 7
        return today - timedelta(days=back), m.span()
    m = re.search(r"\b(yesterday|last\s+night)\b", text, re.I)
    if m:
        return today - timedelta(days=1), m.span()
    m = re.search(r"\b(today|tonight|this\s+morning|this\s+afternoon)\b", text, re.I)
    if m:
        return today, m.span()
    return None, None


def _blank(text, span):
    return text[:span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _to_amount(num_text):
    return float(str(num_text).replace(",", "").replace("$", "").strip())


def find_amount(text):
    """Return (amount or None, (start, end) or None). Prefers $, decimals or 'bucks' over quantities."""
    cands = []
    for m in AMOUNT_RE.finditer(text):
        strong = bool(m["pre"] or m["suf"] or "." in m["num"] or "," in m["num"])
        nxt = re.match(r"\s+([a-z'+]+)", text[m.end():], re.I)
        quantity = bool(nxt and nxt[1].lower() not in FILLER)   # "2 coffees"
        cands.append((strong, quantity, m))
    if not cands:
        return None, None
    strong = [c for c in cands if c[0]]
    plain = [c for c in cands if not c[1]]
    _, _, m = (strong[0] if strong else plain[-1] if plain else cands[-1])
    return _to_amount(m["num"]), m.span()


def _merchant_from(text):
    words = []
    for tok in re.findall(r"[A-Za-z0-9&'+.\-$]+", text):
        tok = tok.strip(".,!?-")
        if tok and tok.lower() not in FILLER:
            words.append(tok)
    return " ".join(words)


def _check_amount(amount):
    if amount is None:
        return
    if amount <= 0:
        raise ValueError("The amount must be more than zero. " + EXAMPLE_HINT)
    if amount > MAX_AMOUNT:
        raise ValueError("That amount is too large (over $1,000,000). Check the number. " + EXAMPLE_HINT)


def llm_extract(text):
    """Ask the AI model (Ollama or Groq, via advisor) to read the sentence. Returns a dict."""
    import advisor
    messages = [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": text}]
    return advisor._call_llm_json(messages)


def _no_amount_message(err):
    """Why the AI couldn't help: not running at all, or a problem the user can fix (wrong model, missing key)."""
    import advisor
    if isinstance(err, advisor.AdvisorError) and not isinstance(err, advisor.AdvisorUnreachable):
        return (f"I couldn't find an amount in that sentence, and the AI model couldn't help: {err} "
                "Until then only simple phrasing works. " + EXAMPLE_HINT)
    return ("I couldn't find an amount in that sentence. The local AI model isn't running, "
            "so only simple phrasing works right now. " + EXAMPLE_HINT)


def parse_entry(text, today=None, use_llm=True, llm_fn=None):
    text = (text or "").strip()
    if not text:
        raise ValueError("Type what you spent or earned. " + EXAMPLE_HINT)
    if len(text) > MAX_LENGTH:
        raise ValueError(f"That sentence is too long (over {MAX_LENGTH} characters). " + EXAMPLE_HINT)
    today = today or date.today()
    kind = "income" if looks_like_income(text) else "expense"
    warnings, method = [], "rules"

    when, span = find_date(text, today)
    rest = _blank(text, span) if span else text
    if NEGATIVE_RE.search(rest):
        raise ValueError("The amount must be more than zero. " + EXAMPLE_HINT)
    amount, span = find_amount(rest)
    if span:
        rest = _blank(rest, span)
    _check_amount(amount)
    merchant = _merchant_from(rest)
    if kind == "income":
        merchant = re.sub(r"\b(got\s+)?paid\b", "", merchant, flags=re.I).strip()

    need_merchant = not merchant and kind == "expense"
    if amount is None or need_merchant:
        if not use_llm:
            if amount is None:
                raise ValueError("I couldn't find an amount in that sentence. " + EXAMPLE_HINT)
            raise ValueError("I couldn't tell what the money was spent on. " + EXAMPLE_HINT)
        try:
            reply = (llm_fn or llm_extract)(text)
            if isinstance(reply, str):
                reply = json.loads(reply)
        except Exception as err:
            if amount is None:
                raise ValueError(_no_amount_message(err)) from None
            raise ValueError("I couldn't tell what the money was spent on. " + EXAMPLE_HINT) from None
        method = "llm"
        llm_merchant = str(reply.get("merchant") or "").strip()
        if amount is None:
            # The leftover text still holds the amount words ("thirty bucks"), so trust the model for both.
            try:
                amount = _to_amount(reply.get("amount"))
            except (TypeError, ValueError):
                raise ValueError("I couldn't find an amount in that sentence. " + EXAMPLE_HINT) from None
            _check_amount(amount)
            merchant = llm_merchant or merchant
        elif not merchant:
            merchant = llm_merchant
        phrase = str(reply.get("date_phrase") or "").strip()
        if when is None and phrase:
            when, _ = find_date(phrase, today)

    if not merchant:
        if kind == "income":
            merchant = "salary" if SALARY_RE.search(text) else "income"
        else:
            raise ValueError("I couldn't tell what the money was spent on. " + EXAMPLE_HINT)

    when = when or today
    if when > today:
        warnings.append(f"That date ({when:%d/%m/%Y}) is in the future. Check it before saving.")
    return {"date": when, "merchant": merchant, "amount": round(amount, 2), "kind": kind,
            "method": method, "warnings": warnings}
