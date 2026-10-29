"""Budget maths. Idea credit: nirban256/expense_tracker (MIT) for category budgets."""
import pandas as pd

WARN_AT = 0.80


def money(v):
    """$5, $5.50, $1,234 (cents only when there are cents)."""
    v = float(v)
    sign = "-" if v < -0.004 else ""
    v = abs(round(v, 2))
    if v == round(v):
        return f"{sign}${v:,.0f}"
    return f"{sign}${v:,.2f}"


def month_label(month):
    """'2026-10' -> 'October 2026'."""
    return pd.Period(month, freq="M").strftime("%B %Y")


def category_column(df):
    return "predicted_category" if "predicted_category" in df.columns else "category"


def with_month(df):
    """Add a 'month' column ('YYYY-MM'). Dates may be '2026-10-05' or '2026-10-05 00:00:00'."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df


def spent_by_category(df, month):
    if df.empty:
        return pd.Series(dtype=float)
    df = with_month(df)
    rows = df[df["month"] == month]
    return rows.groupby(category_column(rows))["amount"].sum().sort_values(ascending=False)


def state_for(pct):
    pct = round(pct, 6)
    if pct > 1:
        return "over"
    if pct >= WARN_AT:
        return "warning"
    return "ok"


def budget_status(df, budgets, month):
    spent = spent_by_category(df, month)
    rows = []
    for cat, limit in (budgets or {}).items():
        if not limit or limit <= 0:
            continue
        s = float(spent.get(cat, 0.0))
        pct = s / limit
        rows.append({"category": cat, "limit": float(limit), "spent": round(s, 2),
                     "remaining": round(limit - s, 2), "pct": pct, "state": state_for(pct)})
    return sorted(rows, key=lambda r: -r["pct"])


def alert_messages(statuses):
    out = []
    for r in statuses:
        if r["state"] == "over":
            out.append(f"{r['category']} is {money(r['spent'] - r['limit'])} over its {money(r['limit'])} budget.")
        elif r["state"] == "warning":
            out.append(f"{r['category']} has used {r['pct']:.0%} of its {money(r['limit'])} budget "
                       f"({money(r['remaining'])} left).")
    return out


STATE_WORDS = {"ok": "on track", "warning": "close to the limit", "over": "over budget"}


def describe_for_advisor(statuses, month):
    if not statuses:
        return f"No budgets are set for {month_label(month)}."
    lines = [f"Budget status for {month_label(month)} (calculated by the app):"]
    for r in statuses:
        lines.append(f"- {r['category']}: spent {money(r['spent'])} of {money(r['limit'])} ({r['pct']:.0%}), "
                     f"{money(r['remaining'])} left, {STATE_WORDS[r['state']]}")
    return "\n".join(lines)
