"""Numbers for the advisor, calculated by code.

Small language models are bad at arithmetic (a 3B model once "saved" $818.64
from a $300 goal). So the app calculates every number the advisor may need,
hands it over as text, and checks the answer's dollar amounts afterwards.
"""
import calendar
import re

from budgets import category_column, money, with_month
from planning import monthly_average, months_used

FIXED_CATEGORIES = {"Bills"}
MAX_CUT = 0.30

GOAL_RE = re.compile(r"\b(save|saving|cut|cutting|reduce|reducing|spend less|put aside|set aside)\b", re.I)
NUMBER_RE = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|percent\b)?", re.I)
ANSWER_AMOUNT_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
REF_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"[+-]?(\d+(?:\.\d+)?)\s?%")


def _pct(x):
    return f"{x:+.0%}"


def _changes(by, prev, last, title):
    """Per-category change between two months, biggest first. by: amounts indexed by (month, category)."""
    changes = []
    for cat in {c for (mo, c) in by.index if mo in (prev, last)}:
        a, b = float(by.get((prev, cat), 0.0)), float(by.get((last, cat), 0.0))
        diff = b - a
        word = "up" if diff >= 0 else "down"
        change = f" ({_pct(diff / a)})" if a else ""
        changes.append((abs(diff), cat, f"{cat}: {money(a)} -> {money(b)}, {word} {money(abs(diff))}{change}"))
    # Ranking is maths too, so the app sorts and names the biggest change instead of the model.
    changes.sort(key=lambda c: (-c[0], c[1]))
    lines = [f"\n{title} (calculated by the app, biggest change first):"] + [f"- {t}" for _, _, t in changes]
    if changes:
        lines.append(f"Biggest change: {changes[0][2]}")
    return lines


def recent_change(df, today):
    """The one comparison a "what changed since last month?" question needs, as text ("" if there is none).

    This month so far against all of last month when this month has spending, otherwise the last two full months.
    """
    if df.empty:
        return ""
    used = months_used(df, today)
    d = with_month(df)
    by = d.groupby(["month", category_column(d)])["amount"].sum()
    current = today.strftime("%Y-%m")
    if used and used[-1] < current and current in set(d["month"]):
        days = calendar.monthrange(today.year, today.month)[1]
        lines = _changes(by, used[-1], current, f"This month so far ({current}, day {today.day} of {days}) "
                                                f"compared with all of {used[-1]}")
    elif len(used) >= 2:
        lines = _changes(by, used[-2], used[-1], f"Change from {used[-2]} to {used[-1]}")
    else:
        return ""
    return "\n".join(lines).strip()


def facts_for_advisor(df, today):
    if df.empty:
        return "No spending recorded yet."
    avg = monthly_average(df, today)
    used = months_used(df, today)
    lines = [f"Average spending per month over {', '.join(used)} (calculated by the app):"]
    lines += [f"- {cat}: {money(v)}" for cat, v in avg.items()]

    d = with_month(df)
    by = d.groupby(["month", category_column(d)])["amount"].sum()
    if len(used) >= 2:
        lines += _changes(by, used[-2], used[-1], f"Change from {used[-2]} to {used[-1]}")
    current = today.strftime("%Y-%m")
    if used and used[-1] < current and current in set(d["month"]):
        days = calendar.monthrange(today.year, today.month)[1]
        lines += _changes(by, used[-1], current, f"This month so far ({current}, day {today.day} of {days}) "
                                                 f"compared with all of {used[-1]}")

    lines.append("\nWhat cutting the top 5 categories would save (calculated by the app):")
    for cat, v in avg.head(5).items():
        if v <= 0:
            continue
        c10, c20 = round(v * 0.10, 2), round(v * 0.20, 2)
        lines.append(f"- {cat}: 10% less saves {money(c10)} a month ({money(c10 * 12)} a year); "
                     f"20% less saves {money(c20)} a month ({money(c20 * 12)} a year)")
    return "\n".join(lines)


def detect_goal(question):
    """'How can I save $200 this month?' -> 200.0. Percentages and years are not goals."""
    m = GOAL_RE.search(question or "")
    if not m:
        return None
    for n in NUMBER_RE.finditer(question, m.end()):
        if n.group(4):
            continue                      # "cut dining by 20%"
        value = float(n.group(2).replace(",", "") + (n.group(3) or ""))
        if not n.group(1) and not n.group(3) and 1900 <= value <= 2100:
            continue                      # "save in 2026"
        if value > 0:
            return value
    return None


def savings_goal(df, goal, today):
    avg = monthly_average(df, today)
    flex = avg[(~avg.index.isin(FIXED_CATEGORIES)) & (avg > 0)]
    total_flex = round(float(flex.sum()), 2)
    max_possible = round(total_flex * MAX_CUT, 2)
    feasible = total_flex > 0 and goal <= max_possible + 1e-9
    target = round(goal if feasible else max_possible, 2)
    share = target / total_flex if total_flex else 0.0

    rows = [{"category": c, "average": round(float(v), 2), "cut": round(float(v) * share, 2)} for c, v in flex.items()]
    if rows:
        # Fix rounding on the largest row so the cuts add up exactly to the target.
        diff = round(target - sum(r["cut"] for r in rows), 2)
        biggest = max(rows, key=lambda r: r["average"])
        biggest["cut"] = round(biggest["cut"] + diff, 2)
    for r in rows:
        r["new_amount"] = round(r["average"] - r["cut"], 2)
    return {"goal": float(goal), "target": target, "feasible": feasible, "max_possible": max_possible,
            "percent": share, "months": months_used(df, today), "rows": rows,
            "total_cut": round(sum(r["cut"] for r in rows), 2), "excluded": sorted(FIXED_CATEGORIES & set(avg.index))}


def describe_goal(plan):
    lines = [f"Savings plan for a goal of {money(plan['goal'])} in one month (calculated by the app, "
             f"based on the average of {', '.join(plan['months'])}):"]
    if not plan["rows"]:
        return lines[0] + "\nThere is no flexible spending to cut."
    for r in plan["rows"]:
        # One amount per line: with three, a small model quoted the new spending level as the cut.
        lines.append(f"- {r['category']}: spend {money(r['cut'])} less this month")
    lines.append(f"Total: {money(plan['total_cut'])} less. Every category is cut by the same share "
                 f"({plan['percent']:.1%}).")
    if plan["excluded"]:
        lines.append(f"Left out as mostly fixed costs: {', '.join(plan['excluded'])}.")
    if not plan["feasible"]:
        lines.append(f"The goal is more than {MAX_CUT:.0%} of all flexible spending, so the plan only reaches "
                     f"{money(plan['max_possible'])}.")
    return "\n".join(lines)


def unverified_amounts(answer, references):
    """Numbers in the answer that the app never gave the model.

    Dollar amounts must match a number in the references within max($1, 1%). Percentages must match a
    percentage in the references within one point (sign ignored: "down 43%" and "-43%" are the same).
    """
    refs, ref_pcts = [], []
    for text in references:
        refs += [float(x.replace(",", "")) for x in REF_NUMBER_RE.findall(text or "")]
        ref_pcts += [float(x) for x in PERCENT_RE.findall(text or "")]
    out = []
    for m in ANSWER_AMOUNT_RE.finditer(answer or ""):
        value = float(m.group(1).replace(",", ""))
        if not any(abs(value - r) <= max(1.0, 0.01 * abs(r)) for r in refs):
            shown = "$" + m.group(1)
            if shown not in out:
                out.append(shown)
    for m in PERCENT_RE.finditer(answer or ""):
        if not any(abs(float(m.group(1)) - r) <= 1.0 for r in ref_pcts):
            shown = m.group(0).replace(" ", "")
            if shown not in out:
                out.append(shown)
    return out
