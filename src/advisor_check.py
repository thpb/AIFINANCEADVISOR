"""Ask the advisor real questions without the browser, using the same code path as the app.

Prints each answer, the savings plan or chart the app calculated, and any
number-check warning. Use it to check the local model before a demo:

    python src/advisor_check.py                 # the demo account, Ollama
    python src/advisor_check.py --user alice
"""
import _root  # noqa: F401

import argparse
import time
from datetime import date

import pandas as pd

import advisor
import advisor_charts
import db
from budgets import money

QUESTIONS = [
    "How can I save $300 this month?",
    "What changed most since last month?",
    "Show my dining spending by month as a chart",
    "Am I on track with my budgets?",
    "Give me a table of my top 5 purchases",
]
CATEGORIES = ["Bills", "Dining", "Entertainment", "Groceries", "Other", "Shopping", "Subscriptions", "Transport"]


def show_info(info, spend, income, budget_map):
    plan = info.get("plan")
    if plan:
        print(f"  [savings plan, calculated by the app] goal {money(plan['goal'])}, total {money(plan['total_cut'])}, "
              f"cut {plan['percent']:.1%} each, feasible={plan['feasible']}, left out: {plan['excluded']}")
        for r in plan["rows"]:
            print(f"    {r['category']:<14} usually {money(r['average']):>10}  spend less by {money(r['cut']):>9}")
    spec = info.get("visual")
    if spec:
        result = advisor_charts.build(spec, spend, income, budget_map)
        print(f"  [{spec['format']}] {result['title']}  spec={spec}")
        print("    " + result["text"].replace("\n", "\n    "))
    for amount in info.get("unverified", []):
        print(f"  WARNING: this answer mentions {amount}, which doesn't come from your data.")
    print(f"  Answered by {info.get('model')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo")
    args = parser.parse_args()
    user = db.get_user_by_username(args.user)
    if not user:
        print(f'No user "{args.user}". Run python src/make_demo_account.py first.')
        return
    spend, income = db.get_transactions(user["id"]), db.get_income(user["id"])
    spend["date"] = pd.to_datetime(spend["date"], format="mixed")
    income["date"] = pd.to_datetime(income["date"], format="mixed")
    budget_map = db.get_budgets(user["id"])
    today = date.today()

    history = advisor.start_conversation(spend, advisor.app_context(spend, income, budget_map), today=today)
    print(f"Engine: {advisor.backend()}, model: {advisor.local_model()}\n")
    for q in QUESTIONS:
        print("=" * 100 + f"\nQ: {q}")
        started = time.time()
        try:
            answer, history, info = advisor.answer_question(q, history, spend, income, budget_map, CATEGORIES, today)
        except Exception as err:
            print(f"  ERROR: {err}")
            continue
        print(f"A ({time.time() - started:.0f}s):\n{answer}\n")
        show_info(info, spend, income, budget_map)


if __name__ == "__main__":
    main()
