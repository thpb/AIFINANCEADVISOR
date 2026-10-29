"""Compare the baseline and the v2 model honestly, and write report_results.md.

Three tests, all with the same model settings as the training scripts:
1. Original data (data/transactions.csv), time-based 80/20 split.
2. New data (data/transactions_v2.csv), time-based 80/20 split.
3. Unseen merchants (new data): about 10% of shops are held out completely, so
   the model has never seen their names. This is the real-world weakness: a new
   café or a shop from a holiday.

The last column adds the keyword hints, which is what the app actually does
before your own corrections. This script trains its own models in memory and
never touches the models the app uses.
"""
import _root  # noqa: F401

import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import category_hints
from clean_data import load_and_clean
from corrections import LOW_CONFIDENCE
from generate_more_data import TYPED
from train_baseline import time_split
from train_improved import EMBED_MODEL
from train_improved_v2 import NUMERIC_V2, add_ambiguity_feature, make_classifier

ORIGINAL, NEW = "data/transactions.csv", "data/transactions_v2.csv"
HOLDOUT_SHARE = 0.10
REPORT = "report_results.md"


def brand(merchant_clean):
    """The shop behind a statement line: 'WOOLWORTHS METRO PARRAMATTA NS' and 'WOOLWORTHS' are both WOOLWORTHS."""
    words = merchant_clean.split()
    return words[0] if words else ""


class Embeddings:
    """Embed each distinct merchant name once."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.cache = {}

    def matrix(self, df):
        todo = [t for t in df["merchant_clean"].unique() if t not in self.cache]
        if todo:
            self.cache.update(zip(todo, self.model.encode(todo, show_progress_bar=False)))
        emb = np.vstack([self.cache[t] for t in df["merchant_clean"]])
        return np.hstack([emb, df[NUMERIC_V2].to_numpy(dtype=float)])


def score(train, test, emb, labels):
    """Accuracy of baseline, v2, and v2 + keyword hints on one split."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    base = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train["merchant_clean"]), train["category"])
    base_pred = base.predict(vec.transform(test["merchant_clean"]))

    le = LabelEncoder().fit(labels)
    clf = make_classifier().fit(emb.matrix(train), le.transform(train["category"]))
    proba = clf.predict_proba(emb.matrix(test))
    v2_pred = le.inverse_transform(proba.argmax(axis=1))

    hinted = test.assign(predicted_category=v2_pred, confidence=proba.max(axis=1))
    hinted = category_hints.apply_hints(hinted, known_categories=le.classes_)
    truth = test["category"]
    return {"rows": len(test), "baseline": accuracy_score(truth, base_pred), "v2": accuracy_score(truth, v2_pred),
            "v2_hints": accuracy_score(truth, hinted["predicted_category"]),
            "clf": clf, "le": le, "test": test.assign(pred=v2_pred, confidence=proba.max(axis=1))}


def md_table(df, index_name=""):
    """A Markdown table without extra libraries."""
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def details(r):
    """Per-category scores, confusion matrix, feature importance and where the mistakes are, for v2."""
    test, le, clf = r["test"], r["le"], r["clf"]
    report = classification_report(test["category"], test["pred"], output_dict=True, zero_division=0)
    per_cat = pd.DataFrame({c: report[c] for c in le.classes_}).T[["precision", "recall", "f1-score", "support"]]
    per_cat = per_cat.sort_values("f1-score")
    shown = per_cat.assign(**{c: per_cat[c].map("{:.2f}".format) for c in ["precision", "recall", "f1-score"]},
                           support=per_cat["support"].astype(int))

    cm = pd.DataFrame(confusion_matrix(test["category"], test["pred"], labels=le.classes_),
                      index=le.classes_, columns=le.classes_)

    names = [f"embedding value {i}" for i in range(384)] + NUMERIC_V2
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    top10 = imp.head(10).to_frame("times used in a split")
    embedding_share = imp[names[:384]].sum() / imp.sum()

    typed = {t for ts in TYPED.values() for t in ts}
    groups = np.where(test["merchant"].isin(typed), "typed by a person",
                      np.where(test["is_ambiguous_merchant"] == 1, "shops that sell everything", "other shops"))
    correct = test["category"] == test["pred"]
    by_group = pd.DataFrame({"rows": test.groupby(groups).size(),
                             "v2 accuracy": correct.groupby(groups).mean().map("{:.1%}".format)})
    wrong_sure = test.loc[~correct, "confidence"]
    return {"per_cat": per_cat, "per_cat_md": md_table(shown, "Category"), "cm_md": md_table(cm, "True \\ Predicted"),
            "top10_md": md_table(top10, "Feature"), "embedding_share": embedding_share,
            "groups_md": md_table(by_group, "Kind of transaction"),
            "conf_right": test.loc[correct, "confidence"].mean(), "conf_wrong": wrong_sure.mean(),
            "wrong_flagged": (wrong_sure < LOW_CONFIDENCE).mean()}


def unseen_split(df, share=HOLDOUT_SHARE, seed=42):
    brands = sorted(df["merchant_clean"].map(brand).unique())
    held = set(random.Random(seed).sample(brands, max(1, round(len(brands) * share))))
    is_held = df["merchant_clean"].map(brand).isin(held)
    return df[~is_held], df[is_held], sorted(held)


def main():
    emb = Embeddings()
    original = add_ambiguity_feature(load_and_clean(ORIGINAL))
    new = add_ambiguity_feature(load_and_clean(NEW))
    results = [("Original data, time split", score(*time_split(original), emb, original["category"])),
               ("New data, time split", score(*time_split(new), emb, new["category"]))]
    train, test, held = unseen_split(new)
    results.append(("New data, unseen merchants", score(train, test, emb, new["category"])))

    lines = ["| Test | Test rows | Baseline (TF-IDF + LogReg) | v2 (embeddings + LightGBM) | v2 + keyword hints |",
             "|------|-----------|----------------------------|-----------------------------|--------------------|"]
    for name, r in results:
        lines.append(f"| {name} | {r['rows']} | {r['baseline']:.1%} | {r['v2']:.1%} | {r['v2_hints']:.1%} |")
    table = "\n".join(lines)
    print(table)
    print(f"\nHeld-out shops ({len(held)}): {', '.join(held)}")

    o, n, u = (r for _, r in results)
    d = details(n)
    hard = d["per_cat"].head(2)
    groups = d["groups_md"]
    detail_text = f"""
## The v2 model in detail (new data, time split)

This is the setup the app uses: v2 trained on the larger data. Test set: the newest {n['rows']} transactions.

### Precision, recall and F1 per category (hardest first)

Precision: of the rows the model put in a category, how many belong there. Recall: of the rows that
belong in a category, how many the model found. F1 combines the two.

{d['per_cat_md']}

### Confusion matrix

Rows are the true category, columns what v2 predicted. The diagonal is correct.

{d['cm_md']}

### Where the mistakes are

{groups}

### Top 10 features

How many times LightGBM used each feature to split a tree. The 384 "embedding values" are the meaning of
the merchant name as numbers; together they account for {d['embedding_share']:.0%} of all splits.

{d['top10_md']}

### How sure the model is

When v2 is right it is on average {d['conf_right']:.0%} sure; when it is wrong, {d['conf_wrong']:.0%}.
{d['wrong_flagged']:.0%} of its mistakes are below the app's 60% line, so they appear in "Worth a second look".

## Which categories are hardest, and why

The hardest categories are {hard.index[0]} (F1 {hard['f1-score'].iloc[0]:.2f}) and {hard.index[1]}
(F1 {hard['f1-score'].iloc[1]:.2f}). Three things cause most mistakes. First, shops that sell everything: an
Amazon, Target or PayPal line can be groceries one day and a present the next, so the name alone can't
decide (see "Where the mistakes are"). Second, label noise: about 4% of the labels are wrong on purpose,
like a person tagging in a hurry, so some "mistakes" are really the label being wrong. Third, cold start:
a shop the model has never seen only has the meaning of its name to go on, which is why accuracy on unseen
merchants drops to {u['v2']:.1%}.

## How the app deals with it

The app works in layers. Every prediction has a confidence score, and rows under 60% land in the "Worth a
second look" list. On this data that catches only {d['wrong_flagged']:.0%} of v2's mistakes: when v2 is wrong it is
still {d['conf_wrong']:.0%} sure on average, so LightGBM is over-confident here. The score is a useful signal but
not a safety net on its own; calibrating the probabilities (for example with scikit-learn's
`CalibratedClassifierCV`) is a possible improvement. The bigger help comes from the other two layers.
Keyword hints fix everyday words and Australian shops the model can't know: on unseen merchants they lift
accuracy from {u['v2']:.1%} to {u['v2_hints']:.1%}. And every correction the user makes becomes a rule for that
shop: earlier transactions are fixed at once and future ones are categorized with full confidence, so the
app gets better the more it is used.
"""
    text = f"""# Model results

Generated by `python src/evaluate_models.py`. Data: `transactions.csv` from
`generate_realistic_data.py --end 2026-10-29` (612 rows) and `transactions_v2.csv` from
`generate_more_data.py --end 2026-10-29` (4,000 rows).

{table}

Held-out shops for the unseen-merchant test ({len(held)} of {new['merchant_clean'].map(brand).nunique()}):
{', '.join(held)}.

## What this means

- **Time split** trains on the oldest 80% of transactions and tests on the newest 20%, the way the app is
  used: it learns from the past and categorizes what comes next.
- On the original data the v2 model scores {o['v2']:.1%} against the baseline's {o['baseline']:.1%}. On the
  larger, more Australian data it scores {n['v2']:.1%} against {n['baseline']:.1%}.
- On the new data the two are close, and the baseline can come out slightly ahead. Most shops in the newest
  20% already appear in training, and word matching handles a known shop's spelling variants well. An
  experiment without the `month` feature (not applied) changed v2 by less than one point, so the dates
  are not the cause. The real difference shows when a shop is new.
- **Unseen merchants** is the honest test for real life: these shops never appear in training. The baseline
  only knows the words it has seen, so it drops to {u['baseline']:.1%}. The v2 model reads the *meaning* of the
  name, the amount and the date, and gets {u['v2']:.1%}. With the app's keyword hints it reaches
  {u['v2_hints']:.1%}. Corrections from the user then fix a shop permanently.
- No model reaches 100%: about 4% of labels are wrong on purpose, and shops like Amazon, Target and
  PayPal genuinely sell things from several categories.
{detail_text}
## Warning

All data here is synthetic. Real bank data is messier (more shop names, odd abbreviations, shared
accounts, spending habits that change), so accuracy on a real statement is likely to be **lower** than
these numbers. The app is built for that: it shows how sure it is, lists low-confidence rows for review,
and learns from every correction.
"""
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
