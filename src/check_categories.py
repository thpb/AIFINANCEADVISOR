"""Show how the demo sentences are categorized by the real trained model.

Run this before presenting so there are no surprises on stage.
"""
import _root  # noqa: F401

from datetime import date

import pandas as pd

import categorizer
import nl_entry

SENTENCES = [
    "spent 50 on groceries yesterday", "coles 82.40", "woolworths 61", "coffee 5.50", "lunch 18 today",
    "uber 23 last friday", "opal 4.80", "petrol 70", "netflix 15.99", "rent 450", "movie tickets 32",
    "kmart 45", "starbucks 7.20", "amazon 39.99", "paid $12.50 at Starbucks yesterday", "zorbo widgets 20",
]


def main():
    if not categorizer.models_exist():
        print("Run `python src/train_improved_v2.py` first.")
        return
    embedder, clf, le = categorizer.load_models()
    rows = []
    for s in SENTENCES:
        e = nl_entry.parse_entry(s, today=date.today(), use_llm=False)
        df = pd.DataFrame([{"date": pd.Timestamp(e["date"]), "merchant": e["merchant"], "amount": e["amount"]}])
        out = categorizer.categorize(df, embedder, clf, le).iloc[0]
        rows.append({
            "sentence": s, "merchant": e["merchant"], "model guess": out["model_category"],
            "sure": f"{out['model_confidence']:.0%}", "final": out["predicted_category"],
            "why": f'word "{out["hint_word"]}"' if out["hint_word"] else "model",
        })
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
