"""Charts and tables the advisor can ask for.

The AI never draws or calculates. It only fills a small JSON request; this code
validates it, calculates the numbers with pandas and the app draws them.
"""
import re

import pandas as pd

import budgets as budget_mod
from budgets import category_column, money, month_label, with_month
from clean_data import clean_merchant

VISUAL_RE = re.compile(
    r"\b(chart|graph|plot|visuali[sz]e|visual|table|breakdown|compare|comparison|trend|over time|by month|"
    r"per month|monthly|top \d+|biggest|largest|show me|list)\b", re.I)

KINDS = {
    "category_totals": "Total spending per category",
    "monthly_spending": "Spending per month, optionally only some categories",
    "income_vs_spending": "Income compared with spending per month",
    "top_purchases": "The largest single purchases (table only)",
    "top_merchants": "The merchants with the most spending",
    "budget_status": "Budgets for one month: limit, spent, left (table only)",
}
TABLE_ONLY = {"top_purchases", "budget_status"}


def wants_visual(question):
    return bool(VISUAL_RE.search(question or ""))


def mentioned_categories(question, categories):
    """Category names the question spells out ("my dining spending" -> ["Dining"])."""
    return [c for c in categories if re.search(rf"\b{re.escape(c)}\b", question or "", re.I)]


def router_prompt(categories, months):
    kinds = "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    return (
        "Reply with JSON only, in this format:\n"
        '{"kind": "...", "format": "chart" or "table", "categories": [], "months": 1-24, '
        '"month": "YYYY-MM" or null, "limit": 3-20}\n'
        f"Kinds:\n{kinds}\n"
        f"Categories: {', '.join(categories)}\n"
        f"Months with data: {', '.join(months)}\n"
        'Use "none" as the kind if no chart/table is asked for. Never invent categories or months.'
    )


def _clamp_int(value, low, high, default):
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError):
        return default


def validate_spec(raw, categories, months):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in KINDS:
        return None
    fmt = raw.get("format") if raw.get("format") in ("chart", "table") else "chart"
    if kind in TABLE_ONLY:
        fmt = "table"
    wanted = raw.get("categories") or []
    if isinstance(wanted, str):
        wanted = [wanted]
    lookup = {c.lower(): c for c in categories}
    cats = []
    for c in wanted if isinstance(wanted, list) else []:
        real = lookup.get(str(c).strip().lower())
        if real and real not in cats:
            cats.append(real)
    month = raw.get("month")
    month = month if isinstance(month, str) and month in months else None
    return {"kind": kind, "format": fmt, "categories": cats, "month": month,
            "months": _clamp_int(raw.get("months"), 1, 24, 6), "limit": _clamp_int(raw.get("limit"), 3, 20, 10)}


def _period(spend, spec):
    """Rows for the requested month, or the last N months with data. Returns (rows, label)."""
    d = with_month(spend)
    if spec["month"]:
        return d[d["month"] == spec["month"]], month_label(spec["month"])
    use = sorted(d["month"].unique())[-spec["months"]:]
    label = month_label(use[0]) if len(use) == 1 else f"{month_label(use[0])} to {month_label(use[-1])}"
    return d[d["month"].isin(use)], label


def _text(table):
    shown = table.head(15).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(money)
    return shown.to_string(index=False)


def build(spec, spend, income, budgets):
    if spec is None or spend is None or spend.empty:
        return None
    kind = spec["kind"]
    cat = category_column(spend)
    rows, label = _period(spend, spec)
    if spec["categories"] and kind not in ("income_vs_spending", "budget_status"):
        rows = rows[rows[cat].isin(spec["categories"])]
    out = {"spec": spec, "series": None, "chart_df": None, "income_by_month": None}

    if kind == "category_totals":
        s = rows.groupby(cat)["amount"].sum().round(2).sort_values(ascending=False)
        if s.empty:
            return None
        out["title"] = f"Spending by category, {label}"
        out["series"] = s
        out["table"] = s.rename("Spent").rename_axis("Category").reset_index()

    elif kind == "monthly_spending":
        if rows.empty:
            return None
        g = rows.groupby(["month", cat])["amount"].sum().round(2).reset_index()
        g.columns = ["month", "category", "amount"]
        g["month_str"] = g["month"].map(lambda m: pd.Period(m, freq="M").strftime("%b %Y"))
        which = ", ".join(spec["categories"]) if spec["categories"] else "All"
        out["title"] = f"{which} spending by month, {label}"
        out["chart_df"] = g
        table = g.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0)
        if len(table.columns) > 1:
            table["Total"] = table.sum(axis=1)
        out["table"] = table.round(2).rename_axis(None, axis=1).rename_axis("Month").reset_index()

    elif kind == "income_vs_spending":
        if rows.empty:
            return None
        months = sorted(rows["month"].unique())
        spent = rows.groupby("month")["amount"].sum()
        inc = with_month(income).groupby("month")["amount"].sum() if income is not None and not income.empty \
            else pd.Series(dtype=float)
        table = pd.DataFrame({"Month": months,
                              "Income": [round(float(inc.get(m, 0.0)), 2) for m in months],
                              "Spent": [round(float(spent.get(m, 0.0)), 2) for m in months]})
        table["Saved"] = (table["Income"] - table["Spent"]).round(2)
        g = rows.groupby(["month", cat])["amount"].sum().round(2).reset_index()
        g.columns = ["month", "category", "amount"]
        g["month_str"] = g["month"].map(lambda m: pd.Period(m, freq="M").strftime("%b %Y"))
        out["title"] = f"Income and spending, {label}"
        out["chart_df"] = g
        out["income_by_month"] = pd.Series(table["Income"].values, index=months)
        out["table"] = table

    elif kind == "top_purchases":
        top = rows.nlargest(spec["limit"], "amount")
        if top.empty:
            return None
        out["title"] = f"Top {len(top)} purchases, {label}"
        out["table"] = pd.DataFrame({"Date": top["date"].dt.strftime("%d/%m/%Y"), "Merchant": top["merchant"],
                                     "Category": top[cat], "Amount": top["amount"].round(2)}).reset_index(drop=True)

    elif kind == "top_merchants":
        names = rows["merchant"].map(clean_merchant).str.title()
        s = rows.groupby(names)["amount"].sum().round(2).nlargest(spec["limit"])
        if s.empty:
            return None
        out["title"] = f"Top {len(s)} merchants, {label}"
        out["series"] = s
        out["table"] = s.rename("Spent").rename_axis("Merchant").reset_index()

    elif kind == "budget_status":
        month = spec["month"] or sorted(with_month(spend)["month"].unique())[-1]
        status = budget_mod.budget_status(spend, budgets, month)
        if not status:
            return None
        out["title"] = f"Budgets, {month_label(month)}"
        out["table"] = pd.DataFrame([{"Category": r["category"], "Budget": float(r["limit"]), "Spent": r["spent"],
                                      "Left": r["remaining"], "Used": f"{r['pct']:.0%}",
                                      "State": budget_mod.STATE_WORDS[r["state"]]} for r in status])
    else:
        return None

    out["text"] = _text(out["table"])
    return out


def chart_note(result):
    spec = result["spec"]
    return (f"The app is showing the user a {spec['format']} titled '{result['title']}' right under your answer. "
            f"Its exact numbers:\n{result['text']}\nPoint out what stands out in 2-4 sentences, quoting these "
            "numbers exactly. Do not draw the table or chart yourself.")
