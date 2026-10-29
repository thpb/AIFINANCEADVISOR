"""Income, savings and savings rate. Saved = income - spent; rate = saved / income."""
import pandas as pd

from budgets import money, with_month


def income_by_month(income):
    if income is None or income.empty:
        return pd.Series(dtype=float)
    return with_month(income).groupby("month")["amount"].sum()


def spend_by_month(spend):
    if spend is None or spend.empty:
        return pd.Series(dtype=float)
    return with_month(spend).groupby("month")["amount"].sum()


def monthly_table(spend, income):
    inc, out = income_by_month(income), spend_by_month(spend)
    months = sorted(set(inc.index) | set(out.index))
    rows = []
    for mo in months:
        i, s = float(inc.get(mo, 0.0)), float(out.get(mo, 0.0))
        saved = round(i - s, 2)
        rows.append({"month": mo, "income": round(i, 2), "spent": round(s, 2), "saved": saved,
                     "rate": (saved / i) if i > 0 else None})
    table = pd.DataFrame(rows, columns=["month", "income", "spent", "saved", "rate"])
    table["rate"] = pd.Series([r["rate"] for r in rows], dtype=object)   # None, not NaN, when no income
    return table


def month_summary(spend, income, month):
    i = float(income_by_month(income).get(month, 0.0))
    s = float(spend_by_month(spend).get(month, 0.0))
    saved = round(i - s, 2)
    return {"month": month, "income": round(i, 2), "spent": round(s, 2), "saved": saved,
            "rate": (saved / i) if i > 0 else None}


def average_monthly_income(income, today, months_back=3):
    inc = income_by_month(income)
    if inc.empty:
        return 0.0
    current = today.strftime("%Y-%m")
    complete = inc[inc.index < current]
    use = complete.iloc[-months_back:] if len(complete) else inc
    return round(float(use.mean()), 2)


def describe_for_advisor(spend, income):
    table = monthly_table(spend, income)
    if table.empty or table["income"].sum() == 0:
        return "No income has been recorded, so savings can't be calculated."
    lines = ["Income and savings by month (calculated by the app):"]
    for r in table.itertuples():
        rate = f" ({r.rate:.0%} of income)" if r.rate is not None else ""
        lines.append(f"- {r.month}: income {money(r.income)}, spent {money(r.spent)}, saved {money(r.saved)}{rate}")
    return "\n".join(lines)
