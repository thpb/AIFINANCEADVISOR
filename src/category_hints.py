"""Keyword hints for plain words and Australian merchants.

The model learned from US bank-style names ("TRADER JOE'S", "SHELL OIL"), so
words typed into quick add ("coffee", "rent") and Australian shops it never saw
("Coles", "Opal") are only guesses for it. This short, readable list fixes the
obvious cases. Amazon, Costco and Target are deliberately NOT here: the model's
uncertainty about them is the honest answer.
"""
import re

HINTS = {
    "Groceries": ["groceries", "grocery", "supermarket", "coles", "woolworths", "woolies", "aldi", "iga",
                  "harris farm"],
    "Dining": ["coffee", "cafe", "lunch", "dinner", "breakfast", "brunch", "restaurant", "takeaway", "pizza",
               "burger", "sushi", "kebab", "uber eats", "menulog", "doordash", "mcdonalds", "mcdonald's",
               "hungry jacks", "kfc", "guzman", "starbucks"],
    "Transport": ["uber", "taxi", "didi", "bus", "train", "tram", "ferry", "opal", "myki", "petrol", "fuel",
                  "parking", "toll", "linkt", "bp", "ampol", "shell"],
    "Bills": ["rent", "electricity", "power bill", "gas bill", "water bill", "internet", "phone bill",
              "insurance", "telstra", "optus", "vodafone", "agl", "origin energy"],
    "Subscriptions": ["netflix", "spotify", "stan", "binge", "disney+", "disney plus", "youtube premium",
                      "subscription", "membership", "icloud"],
    "Entertainment": ["movie", "movies", "cinema", "concert", "tickets", "ticketek", "hoyts", "event cinemas",
                      "steam", "playstation", "xbox"],
    "Shopping": ["clothes", "shoes", "kmart", "big w", "jb hi-fi", "jb hifi", "officeworks", "bunnings", "myer",
                 "david jones", "uniqlo"],
}

HINT_CONFIDENCE = 0.95

# (phrase, category, compiled pattern), longest phrase first.
_PATTERNS = sorted(
    ((w, cat, re.compile(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])")) for cat, ws in HINTS.items() for w in ws),
    key=lambda t: -len(t[0]),
)


def find_hint(text):
    """Return (category, word) for the longest whole-word hint in text, or (None, "")."""
    low = str(text).lower()
    for word, cat, pattern in _PATTERNS:
        if pattern.search(low):
            return cat, word
    return None, ""


def apply_hints(df, known_categories=None):
    """Override the model's category where a hint word matches.

    hint_word is "" (never None) when there is no hint: pandas turns None into
    NaN, NaN is truthy, and the UI once said "the category comes from the word nan".
    """
    df = df.copy()
    known = set(known_categories) if known_categories is not None else None
    words = []
    for i, merchant in df["merchant"].items():
        cat, word = find_hint(merchant)
        if cat and (known is None or cat in known):
            df.at[i, "predicted_category"] = cat
            df.at[i, "confidence"] = max(float(df.at[i, "confidence"] or 0), HINT_CONFIDENCE)
            words.append(word)
        else:
            words.append("")
    df["hint_word"] = words
    return df
