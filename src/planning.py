"""Month-end forecast (simple run-rate) and the what-if saver."""
import calendar
from datetime import date

import pandas as pd

from budgets import category_column, money, spent_by_category, with_month

TOO_EARLY_DAYS = 5


def forecast_month(df, month, today, budgets=None):
    budgets = {c: v for c, v in (budgets or {}).items() if v and v > 0}
    y, m = (int(p) for p in month.split("-"))
    days_in = calendar.monthrange(y, m)[1]
    first, last = date(y, m, 1), date(y, m, days_in)

    if today > last:
        status, elapsed = "past", days_in
    elif today < first:
        status, elapsed = "future", 0
    else:
        status, elapsed = "current", today.day
    complete = status != "current"
    too_early = status == "current" and elapsed < TOO_EARLY_DAYS
    factor = 1.0 if complete else days_in / elapsed

    spent = spent_by_category(df, month) if status != "future" else pd.Series(dtype=float)
    rows = []
    for cat in sorted(set(spent.index) | set(budgets)):
        s = float(spent.get(cat, 0.0))
        projected = s * factor
        limit = budgets.get(cat)
        state = None
        if limit:
            state = "already_over" if s > limit else "on_pace_over" if projected > limit else "ok"
        rows.append({"category": cat, "spent": round(s, 2), "projected": round(projected, 2),
                     "limit": limit, "state": state})
    rows.sort(key=lambda r: -r["projected"])
    total = float(spent.sum()) if len(spent) else 0.0
    return {"month": month, "status": status, "complete": complete, "too_early": too_early,
            "days_elapsed": elapsed, "days_in_month": days_in, "spent": round(total, 2),
            "projected": round(total * factor, 2), "categories": rows}


def forecast_messages(fc):
    if fc["too_early"]:
        return []
    out = []
    for r in fc["categories"]:
        if r["state"] == "on_pace_over":
            out.append(f"{r['category']} is on pace for {money(round(r['projected']))} by month end, "
                       f"{money(round(r['projected'] - r['limit']))} over its {money(r['limit'])} budget.")
    return out


def months_used(df, today, months_back=3):
    """The last complete months with data (falls back to all months if none are complete)."""
    if df.empty:
        return []
    months = sorted(with_month(df)["month"].unique())
    current = today.strftime("%Y-%m")
    complete = [mo for mo in months if mo < current]
    return complete[-months_back:] if complete else months


def monthly_average(df, today, months_back=3):
    """Average spend per category per month over the last complete months, largest first."""
    used = months_used(df, today, months_back)
    if not used:
        return pd.Series(dtype=float)
    df = with_month(df)
    rows = df[df["month"].isin(used)]
    avg = rows.groupby(category_column(rows))["amount"].sum() / len(used)
    return avg.round(2).sort_values(ascending=False)


def what_if(df, cuts, today):
    avg = monthly_average(df, today)
    rows = []
    for cat, pct in cuts.items():
        pct = min(max(float(pct), 0.0), 100.0)
        a = float(avg.get(cat, 0.0))
        monthly = round(a * pct / 100, 2)
        rows.append({"category": cat, "average": a, "percent": pct, "monthly_saving": monthly,
                     "yearly_saving": round(monthly * 12, 2)})
    monthly_total = round(sum(r["monthly_saving"] for r in rows), 2)
    return {"rows": rows, "monthly_total": monthly_total, "yearly_total": round(monthly_total * 12, 2)}
