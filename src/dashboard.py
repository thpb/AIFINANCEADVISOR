"""Expense Categorizer: the Streamlit app.

Run with:  streamlit run src/dashboard.py
Code does the maths, the AI only explains.
"""
import _root  # noqa: F401

import os
import re
from datetime import date

import pandas as pd
import streamlit as st

import advisor
import advisor_charts
import auth
import budgets as bud
import categorizer
import corrections
import db
import money_facts
import nl_entry
import planning
import savings
import table_edits
import ui
from budgets import money, month_label

st.set_page_config(page_title="Expense Categorizer", page_icon="💳", layout="wide")
db.init_db()
ui.inject_css()

PAGES = ["Overview", "Plan ahead", "Transactions", "AI advisor"]
ss = st.session_state
for key, value in {"user_id": None, "username": "", "history": None, "pending": None, "flash": [],
                   "answers": {}, "imported_files": [], "advisor_key": None, "queued": None}.items():
    ss.setdefault(key, value)


def flash(icon, text):
    ss.flash.append((icon, text))


for icon, text in ss.flash:
    st.toast(ui.esc(text), icon=icon)
ss.flash = []

if not categorizer.models_exist():
    st.error("The categorization model hasn't been trained yet. Run `python src/train_improved_v2.py` first, "
             "then reload this page.")
    st.stop()


@st.cache_resource(show_spinner="Loading the categorization model...")
def get_models():
    return categorizer.load_models()


# ---------------------------------------------------------------- login gate

if ss.user_id is None:
    st.title("Expense Categorizer")
    st.caption("A private, local finance assistant that measures its accuracy, knows when it is unsure, "
               "and learns from your corrections.")
    tab_in, tab_up = st.tabs(["Log in", "Sign up"])
    with tab_in, st.form("login"):
        name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            uid, msg = auth.login(name, password)
            if uid:
                ss.user_id, ss.username = uid, name.strip()
                st.rerun()
            st.error(msg)
    with tab_up, st.form("signup"):
        name = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password", help="At least 6 characters.")
        if st.form_submit_button("Create account", type="primary"):
            ok, msg = auth.signup(name, password)
            (st.success if ok else st.error)(msg)
    st.stop()

# ---------------------------------------------------------------- data

uid = ss.user_id
embedder, clf, le = get_models()
CATEGORIES = list(le.classes_)
today = date.today()

spend = db.get_transactions(uid)
income = db.get_income(uid)
budget_map = db.get_budgets(uid)
rules = db.get_rules(uid)
if not spend.empty:
    spend["date"] = pd.to_datetime(spend["date"], format="mixed")
if not income.empty:
    income["date"] = pd.to_datetime(income["date"], format="mixed")


def run_categorizer(df):
    return categorizer.categorize(df, embedder, clf, le, rules)


def pretty(merchant):
    name = corrections.merchant_key(merchant).title()
    return name or merchant


def learn(merchant, category):
    """Remember a correction and fix every earlier transaction from the same merchant."""
    db.set_rule(uid, corrections.merchant_key(merchant), category)
    current = db.get_transactions(uid)
    ids = corrections.ids_for_merchant(current, merchant)
    changed = current[current["id"].isin(ids) & (current["predicted_category"] != category)]
    db.update_categories(uid, ids, category)
    n = len(changed)
    tail = f" and updated {n} earlier one{'s' if n != 1 else ''}." if n else "."
    flash("🧠", f"Got it. {pretty(merchant)} is now {category}{tail}")


def budget_toasts(category, when):
    month = pd.Timestamp(when).strftime("%Y-%m")
    statuses = [r for r in bud.budget_status(db.get_transactions(uid), budget_map, month) if r["category"] == category]
    for msg in bud.alert_messages(statuses):
        flash("⚠️", msg)


def parse_user_dates(values):
    """CSV dates: ISO (2026-10-05) or day-first (05/10/2026)."""
    text = values.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(text[iso], format="mixed")
    if (~iso).any():
        out[~iso] = pd.to_datetime(text[~iso], dayfirst=True, format="mixed")
    return out


def save_new(df, source):
    """Categorize and store new expense rows (date, merchant, amount)."""
    out = run_categorizer(df)
    out["date"] = out["date"].map(table_edits.as_date_text)
    return db.insert_transactions(uid, out, source=source)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    page = st.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown(f"Signed in as **{ss.username or 'you'}**")
    if st.button("Log out", icon=":material/logout:"):
        for key in ["user_id", "history", "pending", "advisor_key"]:
            ss[key] = None
        ss.answers = {}
        st.rerun()

    st.subheader("AI engine")
    engine = st.radio("AI engine", ["Local (Ollama)", "Groq (free cloud)"], key="engine",
                      label_visibility="collapsed")
    if engine.startswith("Local"):
        os.environ["ADVISOR_BACKEND"] = "ollama"
        model = st.text_input("Local model", value=advisor.local_model(), key="local_model",
                              help="Any model you have pulled with Ollama, for example qwen2.5:3b-instruct-q8_0. "
                                   "Run `ollama list` to see yours.")
        os.environ["OLLAMA_MODEL"] = model.strip()
        st.caption("Runs on this computer. Nothing leaves it.")
    else:
        os.environ["ADVISOR_BACKEND"] = "groq"
        key = st.text_input("Groq API key", type="password", key="groq_key",
                            help="Free from console.groq.com. Kept in memory only, never saved.")
        if key.strip():
            os.environ["GROQ_API_KEY"] = key.strip()
        st.caption("Groq gives smarter answers, but your spending summary is sent to Groq's servers when you "
                   "ask the advisor.")

    st.divider()
    with st.expander("Danger zone"):
        sure = st.checkbox("I want to delete all my transactions and income")
        if st.button("Delete everything", type="primary", disabled=not sure):
            n = db.delete_all_transactions(uid)
            db.delete_all_income(uid)
            ss.history = None
            flash("🗑️", f"Deleted {n} transactions and all income. Your budgets are kept.")
            st.rerun()

# ---------------------------------------------------------------- quick add (every page)

ui.app_title()
with st.container(key="quickadd"):
    with st.form("quick_add", clear_on_submit=True, border=False):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        typed = c1.text_input("Add an expense in plain English", placeholder="spent 50 on groceries yesterday",
                              label_visibility="collapsed")
        added = c2.form_submit_button("Add", type="primary", width="stretch")
    st.caption('Try "Netflix 15.99", "paid \\$12.50 at Starbucks yesterday" or "got paid 2000 salary".')

if added and typed.strip():
    try:
        with st.spinner("Reading that..."):
            entry = nl_entry.parse_entry(typed, today=today)
    except ValueError as err:
        st.error(ui.esc(str(err)))
    else:
        row = pd.DataFrame([{"date": pd.Timestamp(entry["date"]), "merchant": entry["merchant"],
                             "amount": entry["amount"]}])
        out = run_categorizer(row).iloc[0]
        entry.update(category=out["predicted_category"], confidence=float(out["confidence"]),
                     hint_word=out["hint_word"] or "", rule_applied=bool(out["rule_applied"]),
                     ambiguous=int(out["is_ambiguous_merchant"]))
        ss.pending = entry
        ss.pending_kind = "Income" if entry["kind"] == "income" else "Expense"
elif added:
    st.error(ui.esc("Type what you spent or earned. " + nl_entry.EXAMPLE_HINT))

if ss.pending:
    e = ss.pending
    with st.container(border=True, key="confirm_box"):
        st.markdown("**Check this before it's saved**")
        kind = st.radio("This is", ["Expense", "Income"], key="pending_kind", horizontal=True)
        how = "simple rules" if e["method"] == "rules" else (
            "the local AI model" if advisor.backend() == "ollama" else "the AI model on Groq")
        notes = [f"I read this with {how}."]
        if kind == "Expense":
            if e["rule_applied"]:
                notes.append("The category comes from a correction you made earlier.")
            elif e["hint_word"]:
                notes.append(f'The category comes from the word "{e["hint_word"]}".')
            elif e["confidence"] < corrections.LOW_CONFIDENCE:
                notes.append(f"The model is only {e['confidence']:.0%} sure of the category, so please check it.")
            else:
                notes.append(f"The model is {e['confidence']:.0%} sure of the category.")
            notes.append("Change the category if it's wrong and I'll remember it.")
        st.caption(ui.esc(" ".join(notes)))
        for w in e["warnings"]:
            st.warning(ui.esc(w))
        with st.form("confirm", border=False):
            c1, c2, c3, c4 = st.columns(4)
            when = c1.date_input("Date", value=e["date"], format="DD/MM/YYYY")
            who = c2.text_input("Merchant" if kind == "Expense" else "From", value=e["merchant"])
            amount = c3.number_input("Amount", value=float(e["amount"]), min_value=0.01, step=1.0, format="%.2f")
            category = None
            if kind == "Expense":
                cats = CATEGORIES if e["category"] in CATEGORIES else CATEGORIES + [e["category"]]
                category = c4.selectbox("Category", cats, index=cats.index(e["category"]))
            b1, b2, _ = st.columns([1, 1, 6])
            save = b1.form_submit_button("Save", type="primary", width="stretch")
            cancel = b2.form_submit_button("Cancel", width="stretch")
        if save:
            if not who.strip():
                st.error("Add a merchant name before saving.")
            elif kind == "Expense":
                row = pd.DataFrame([{"date": table_edits.as_date_text(when), "merchant": who.strip(), "amount": amount,
                                     "predicted_category": category, "confidence": corrections.CERTAIN,
                                     "is_recurring": 0, "is_ambiguous_merchant": e["ambiguous"]}])
                db.insert_transactions(uid, row, source="quick add")
                flash("✅", f"Saved {money(amount)} at {who.strip()} as {category}.")
                if category != e["category"]:
                    learn(who.strip(), category)
                budget_toasts(category, when)
                ss.pending = None
                st.rerun()
            else:
                db.add_income(uid, table_edits.as_date_text(when), who.strip(), amount)
                flash("💰", f"Saved {money(amount)} of income from {who.strip()}.")
                ss.pending = None
                st.rerun()
        if cancel:
            ss.pending = None
            st.rerun()


# ---------------------------------------------------------------- shared numbers

def months_with_data():
    months = set()
    if not spend.empty:
        months |= set(spend["date"].dt.strftime("%Y-%m"))
    if not income.empty:
        months |= set(income["date"].dt.strftime("%Y-%m"))
    months.add(today.strftime("%Y-%m"))
    return sorted(months, reverse=True)


def month_picker():
    months = months_with_data()
    if ss.get("month") not in months:
        ss.month = months[0]
    return st.selectbox("Month", months, key="month", format_func=month_label)


def prev_month(month):
    return (pd.Period(month, freq="M") - 1).strftime("%Y-%m")


# ---------------------------------------------------------------- Overview

def page_overview():
    with st.expander("More ways to add: CSV import, a form, or income"):
        t_csv, t_hand, t_income = st.tabs(["Import a CSV", "Enter one by hand", "Add income"])
        with t_csv:
            st.caption("Columns: date, merchant, amount. Dates can be day-first (05/10/2026) or 2026-10-05. "
                       "Any category column is ignored: the app categorizes every row itself.")
            up = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
            if up is not None:
                file_id = getattr(up, "file_id", None) or f"{up.name}-{up.size}"
                if file_id in ss.imported_files:
                    st.caption(f"{up.name} is already imported.")
                else:
                    try:
                        raw = pd.read_csv(up)
                        raw.columns = [c.strip().lower() for c in raw.columns]
                        missing = {"date", "merchant", "amount"} - set(raw.columns)
                        if missing:
                            raise ValueError(f"The file is missing these columns: {', '.join(sorted(missing))}.")
                        df = raw[["date", "merchant", "amount"]].dropna().copy()
                        df["date"] = parse_user_dates(df["date"])
                        df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[$,]", "", regex=True))
                        with st.spinner(f"Categorizing {len(df)} transactions..."):
                            n = save_new(df.reset_index(drop=True), "upload")
                    except Exception as err:
                        st.error(ui.esc(f"Couldn't import that file: {err} Check that it has date, merchant and "
                                        "amount columns."))
                    else:
                        ss.imported_files.append(file_id)   # a rerun must not import it again
                        ss.history = None
                        flash("📥", f"Imported and categorized {n} transactions from {up.name}.")
                        st.rerun()
        with t_hand, st.form("by_hand", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            when = c1.date_input("Date", value=today, format="DD/MM/YYYY")
            who = c2.text_input("Merchant")
            amount = c3.number_input("Amount", min_value=0.0, step=1.0, format="%.2f")
            choice = c4.selectbox("Category", ["Let the app decide"] + CATEGORIES)
            if st.form_submit_button("Add expense", type="primary"):
                if not who.strip() or amount <= 0:
                    st.error("Enter a merchant and an amount above zero.")
                else:
                    df = pd.DataFrame([{"date": pd.Timestamp(when), "merchant": who.strip(), "amount": amount}])
                    out = run_categorizer(df)
                    if choice != "Let the app decide":
                        out["predicted_category"], out["confidence"] = choice, corrections.CERTAIN
                    out["date"] = out["date"].map(table_edits.as_date_text)
                    db.insert_transactions(uid, out, source="form")
                    cat = out["predicted_category"].iloc[0]
                    flash("✅", f"Saved {money(amount)} at {who.strip()} as {cat}.")
                    budget_toasts(cat, when)
                    st.rerun()
        with t_income, st.form("income_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            when = c1.date_input("Date", value=today, format="DD/MM/YYYY", key="inc_date")
            who = c2.text_input("From", placeholder="salary")
            amount = c3.number_input("Amount", min_value=0.0, step=50.0, format="%.2f", key="inc_amount")
            if st.form_submit_button("Add income", type="primary"):
                if amount <= 0:
                    st.error("Enter an amount above zero.")
                else:
                    db.add_income(uid, table_edits.as_date_text(when), who.strip() or "income", amount)
                    flash("💰", f"Saved {money(amount)} of income.")
                    st.rerun()

    month = month_picker()
    if spend.empty and income.empty:
        st.info("No spending yet. Type a purchase in the box above, or import a CSV under "
                "\"More ways to add\".", icon=":material/add_circle:")
        return

    rows = spend[spend["date"].dt.strftime("%Y-%m") == month] if not spend.empty else spend
    total = float(rows["amount"].sum()) if not rows.empty else 0.0
    summary = savings.month_summary(spend, income, month)
    statuses = bud.budget_status(spend, budget_map, month) if not spend.empty else []
    alerts = bud.alert_messages(statuses)
    review = corrections.needs_review(spend)

    notes = []
    prev = prev_month(month)
    prev_total = float(spend.loc[spend["date"].dt.strftime("%Y-%m") == prev, "amount"].sum()) if not spend.empty else 0
    if prev_total > 0:
        diff = total - prev_total
        word = "more" if diff >= 0 else "less"
        notes.append(f"That is {money(abs(diff))} {word} than {month_label(prev)} ({money(prev_total)}).")
    if summary["income"] > 0:
        if summary["saved"] >= 0:
            notes.append(f"You earned {money(summary['income'])} and saved {money(summary['saved'])}.")
        else:
            notes.append(f"You spent {money(-summary['saved'])} more than you earned.")
    if budget_map:
        notes.append(f"{len(alerts)} budget{'s' if len(alerts) != 1 else ''} need attention."
                     if alerts else "All budgets are on track.")
    if len(review):
        notes.append(f"{len(review)} categories are worth checking.")
    ui.hero(f"You spent {money(total)} in {month_label(month)}", " ".join(notes))

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        by_cat = rows.groupby("predicted_category")["amount"].sum() if not rows.empty else pd.Series(dtype=float)
        if len(by_cat):
            ui.stat("Top category", by_cat.idxmax(), money(by_cat.max()))
        else:
            ui.stat("Top category", "None yet", "")
    with s2:
        if not rows.empty:
            big = rows.loc[rows["amount"].idxmax()]
            ui.stat("Biggest purchase", money(big["amount"]), f"{big['merchant']}, {big['date']:%d/%m}")
        else:
            ui.stat("Biggest purchase", "None yet", "")
    with s3:
        ui.stat("Transactions", str(len(rows)), "this month")
    with s4:
        if summary["income"] > 0:
            ui.stat("Saved", money(summary["saved"]), f"{summary['rate']:.0%} of {money(summary['income'])} income")
        else:
            ui.stat("Saved", "Add income", 'Try "got paid 2000 salary"')

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Budgets")
        for msg in alerts:
            st.warning(ui.esc(msg))
        if not statuses:
            st.caption("No budgets yet. Set one below and the app will warn you at 80%.")
        for r in statuses:
            ui.budget_row(r)
        with st.expander("Set or change budgets"):
            with st.form("budgets_form", border=False):
                values = {}
                for cat in CATEGORIES:
                    values[cat] = st.number_input(cat, min_value=0.0, step=50.0, value=float(budget_map.get(cat, 0)),
                                                  key=f"budget_{cat}", help="0 means no budget.")
                if st.form_submit_button("Save budgets", type="primary"):
                    for cat, v in values.items():
                        if v > 0:
                            db.set_budget(uid, cat, v)
                        else:
                            db.delete_budget(uid, cat)
                    flash("✅", "Budgets saved.")
                    st.rerun()
    with right:
        st.subheader(f"Where it went in {month_label(month)}")
        positive = by_cat[by_cat > 0] if len(by_cat) else by_cat
        if len(positive):
            ui.show(ui.hbar(positive), alt=f"Spending by category in {month_label(month)}")
        else:
            st.caption("Nothing spent this month yet.")

    if not spend.empty:
        st.subheader("Month by month")
        inc = savings.income_by_month(income)
        ui.show(ui.month_chart(ui.month_rows(spend), inc if len(inc) else None, highlight=month),
                alt="Monthly spending by category with income")
        if len(inc):
            st.caption("Bars are spending per category. The line is income: the gap above the bars is what you saved.")


# ---------------------------------------------------------------- Plan ahead

def page_plan():
    month = month_picker()
    st.subheader("Month-end forecast")
    fc = planning.forecast_month(spend, month, today, budget_map)
    if fc["status"] == "past":
        ui.hero(f"{month_label(month)} is finished: you spent {money(fc['spent'])}")
    elif fc["status"] == "future":
        ui.hero(f"{month_label(month)} hasn't started yet")
    elif fc["too_early"]:
        st.info(ui.esc(f"It's only day {fc['days_elapsed']} of {month_label(month)}. The forecast starts after "
                       f"{planning.TOO_EARLY_DAYS} days, when there is enough spending to go on. "
                       f"So far you have spent {money(fc['spent'])}."), icon=":material/hourglass_top:")
    else:
        ui.hero(f"On pace for {money(round(fc['projected']))} by the end of {month_label(month)}",
                f"You have spent {money(fc['spent'])} in {fc['days_elapsed']} of {fc['days_in_month']} days. "
                "This is a simple run-rate, so it assumes you keep spending at the same pace. "
                "It can't know about bills that haven't been paid yet.")
        inc = savings.month_summary(spend, income, month)["income"]
        if inc > 0:
            left = inc - fc["projected"]
            if left >= 0:
                st.markdown(ui.esc(f"With {money(inc)} income this month, you're on pace to save about "
                                   f"{money(round(left))} ({left / inc:.0%})."))
            else:
                st.warning(ui.esc(f"At this pace you'll spend about {money(round(-left))} more than your "
                                  f"{money(inc)} income this month."))
        for msg in planning.forecast_messages(fc):
            st.warning(ui.esc(msg))
    if fc["categories"] and fc["status"] != "future":
        ui.show(ui.forecast_chart(fc), alt="Forecast of month-end spending per category")
        if not fc["complete"]:
            st.caption("Solid bar: spent so far. Faded bar: what the same pace adds by month end. "
                       "Red mark: your budget. The label is the month-end estimate.")

    st.subheader("What if I spent less?")
    avg = planning.monthly_average(spend, today) if not spend.empty else pd.Series(dtype=float)
    avg = avg[avg > 0]
    if avg.empty:
        st.caption("Add some spending first and the app will show what cutting back would save.")
        return
    used = planning.months_used(spend, today)
    st.caption(f"Based on your average month ({', '.join(month_label(m) for m in used)}). Move a slider to see "
               "what spending less in that category would save.")
    cuts = {}
    cols = st.columns(3)
    for i, cat in enumerate(avg.head(6).index):
        with cols[i % 3]:
            cuts[cat] = st.slider(f"{cat} (usually {money(avg[cat])})".replace("$", "\\$"), 0, 50, 0, 5,
                                  format="%d%% less", key=f"cut_{cat}")
    res = planning.what_if(spend, cuts, today)
    ui.hero(f"You would save about {money(res['yearly_total'])} a year")
    inc = savings.average_monthly_income(income, today)
    msg = f"That is {money(res['monthly_total'])} a month."
    if inc > 0:
        spent = float(avg.sum())
        before, after = (inc - spent) / inc, (inc - spent + res["monthly_total"]) / inc
        msg += f" Your savings rate would go from {before:.0%} to {after:.0%} of your income."
    st.markdown(ui.esc(msg))
    table = pd.DataFrame([{"Category": r["category"], "Usually a month": r["average"], "Spend less by": f"{r['percent']:.0f}%",
                           "Saves a month": r["monthly_saving"], "Saves a year": r["yearly_saving"]}
                          for r in res["rows"] if r["percent"] > 0])
    if not table.empty:
        st.dataframe(table, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="dollar") for c in ["Usually a month", "Saves a month", "Saves a year"]})


# ---------------------------------------------------------------- Transactions

def pending_text(updates, deleted):
    parts = []
    if updates:
        parts.append(f"{len(updates)} edit{'s' if len(updates) != 1 else ''}")
    if deleted:
        parts.append(f"{len(deleted)} deletion{'s' if len(deleted) != 1 else ''}")
    return " and ".join(parts) + " not saved yet." if parts else "No unsaved changes."


def page_transactions():
    n_review = len(corrections.needs_review(spend))
    t_spend, t_income, t_review = st.tabs(["Spending", "Income and savings", f"Worth a second look ({n_review})"])

    with t_spend:
        if spend.empty:
            st.info("No transactions yet. Add one with the box at the top of the page.")
        else:
            st.caption("Click a cell to edit it. Select rows and press Delete to remove them. Nothing is saved "
                       "until you press Save changes. Changing a category teaches the app that merchant.")
            shown = spend.sort_values(["date", "id"], ascending=False).set_index("id")[
                ["date", "merchant", "amount", "predicted_category", "source"]].rename(
                columns={"predicted_category": "category"})
            st.data_editor(shown, key="txn_editor", num_rows="delete", hide_index=True, alt="Your transactions",
                           column_config={
                               "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY", required=True),
                               "merchant": st.column_config.TextColumn("Merchant", required=True),
                               "amount": st.column_config.NumberColumn("Amount", format="dollar", required=True),
                               "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES,
                                                                            required=True),
                               "source": st.column_config.TextColumn("Added via", disabled=True)})
            updates, deleted = table_edits.editor_changes(shown, ss.get("txn_editor"))
            b1, b2, info = st.columns([1, 1, 4], vertical_alignment="center")
            if b1.button("Save changes", type="primary", disabled=not (updates or deleted), key="txn_save"):
                for tid, row in updates.items():
                    db.update_transaction(uid, tid, table_edits.as_date_text(row["date"]), row["merchant"],
                                          row["amount"], row["category"])
                    if row["category"] != shown.loc[tid, "category"]:
                        learn(row["merchant"], row["category"])
                db.delete_transactions(uid, deleted)
                flash("✅", f"Saved {pending_text(updates, deleted).replace(' not saved yet.', '')}.")
                ss.pop("txn_editor", None)
                st.rerun()
            if b2.button("Discard", disabled=not (updates or deleted), key="txn_discard"):
                ss.pop("txn_editor", None)
                st.rerun()
            info.caption(pending_text(updates, deleted))
            out = spend.assign(date=spend["date"].dt.strftime("%Y-%m-%d"))[
                ["date", "merchant", "amount", "predicted_category"]].rename(columns={"predicted_category": "category"})
            st.download_button("Download as CSV", out.to_csv(index=False), file_name="transactions.csv",
                               mime="text/csv", on_click="ignore", icon=":material/download:")

    with t_income:
        table = savings.monthly_table(spend, income)
        if income.empty:
            st.info('No income yet. Type "got paid 2000 salary" in the box at the top, or use "Add income" '
                    "under More ways to add on the Overview page.")
        if not table.empty and not income.empty:
            st.subheader("Savings by month")
            show = pd.DataFrame({"Month": table["month"].map(month_label), "Income": table["income"],
                                 "Spent": table["spent"], "Saved": table["saved"],
                                 "Savings rate": [f"{r:.0%}" if r is not None else "No income" for r in table["rate"]]})
            st.dataframe(show, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="dollar") for c in ["Income", "Spent", "Saved"]})
        if not income.empty:
            st.subheader("Income")
            shown = income.sort_values(["date", "id"], ascending=False).set_index("id")[["date", "source", "amount"]]
            st.data_editor(shown, key="income_editor", num_rows="delete", hide_index=True, alt="Your income",
                           column_config={
                               "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY", required=True),
                               "source": st.column_config.TextColumn("From", required=True),
                               "amount": st.column_config.NumberColumn("Amount", format="dollar", required=True)})
            updates, deleted = table_edits.editor_changes(shown, ss.get("income_editor"))
            b1, b2, info = st.columns([1, 1, 4], vertical_alignment="center")
            if b1.button("Save changes", type="primary", disabled=not (updates or deleted), key="inc_save"):
                for iid, row in updates.items():
                    db.update_income(uid, iid, table_edits.as_date_text(row["date"]), row["source"], row["amount"])
                db.delete_income(uid, deleted)
                flash("✅", "Income saved.")
                ss.pop("income_editor", None)
                st.rerun()
            if b2.button("Discard", disabled=not (updates or deleted), key="inc_discard"):
                ss.pop("income_editor", None)
                st.rerun()
            info.caption(pending_text(updates, deleted))

    with t_review:
        review = corrections.needs_review(spend).head(15)
        st.subheader("The model wasn't sure about these")
        if review.empty:
            st.success("Nothing to check. Every category is either confident or confirmed by you.")
        else:
            st.caption("Fix any wrong category and press Save corrections. The app will remember each merchant.")
            shown = review.set_index("id")[["date", "merchant", "amount", "confidence", "predicted_category"]].rename(
                columns={"predicted_category": "category"})
            st.data_editor(shown, key="review_editor", hide_index=True, disabled=["date", "merchant", "amount",
                                                                                  "confidence"],
                           alt="Transactions the model was unsure about",
                           column_config={
                               "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                               "merchant": "Merchant",
                               "amount": st.column_config.NumberColumn("Amount", format="dollar"),
                               "confidence": st.column_config.ProgressColumn("Model is sure", format="percent",
                                                                             min_value=0, max_value=1),
                               "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES,
                                                                            required=True)})
            updates, _ = table_edits.editor_changes(shown, ss.get("review_editor"))
            if st.button("Save corrections", type="primary", disabled=not updates):
                for tid, row in updates.items():
                    db.update_categories(uid, [tid], row["category"])
                    learn(row["merchant"], row["category"])
                ss.pop("review_editor", None)
                st.rerun()

        with st.expander(f"Merchants I've learned from you ({len(rules)})"):
            if not rules:
                st.caption("None yet. Change a category anywhere and the app will remember that merchant.")
            for mkey, cat in sorted(rules.items()):
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                c1.markdown(f"**{mkey.title()}** is always {cat}")
                if c2.button("Forget", key=f"forget_{mkey}"):
                    db.delete_rule(uid, mkey)
                    flash("🧹", f"Forgot the rule for {mkey.title()}.")
                    st.rerun()

        if not spend.empty:
            st.subheader("Unusually large for their category")
            stats = spend.groupby("predicted_category")["amount"].agg(["mean", "std"])
            joined = spend.join(stats, on="predicted_category")
            joined["z"] = (joined["amount"] - joined["mean"]) / joined["std"]
            big = joined[joined["z"] > 2].sort_values("z", ascending=False)
            if big.empty:
                st.caption("Nothing stands out.")
            else:
                st.dataframe(pd.DataFrame({"Date": big["date"].dt.strftime("%d/%m/%Y"), "Merchant": big["merchant"],
                                           "Category": big["predicted_category"], "Amount": big["amount"],
                                           "Usual": big["mean"].round(2)}),
                             hide_index=True, column_config={c: st.column_config.NumberColumn(format="dollar")
                                                             for c in ["Amount", "Usual"]})
            st.subheader("Detected recurring payments")
            keyed = spend.assign(key=spend["merchant"].map(corrections.merchant_key))
            rec = keyed.groupby(["key", "amount"]).agg(times=("id", "size"), last=("date", "max")).reset_index()
            rec = rec[rec["times"] >= 2].sort_values("times", ascending=False)
            if rec.empty:
                st.caption("No repeating payments found yet.")
            else:
                st.dataframe(pd.DataFrame({"Merchant": rec["key"].str.title(), "Amount": rec["amount"],
                                           "Times": rec["times"], "Last paid": rec["last"].dt.strftime("%d/%m/%Y")}),
                             hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="dollar")})


# ---------------------------------------------------------------- AI advisor

GOAL_NOTE = ("Do not repeat the table or change any number. Give 2-3 practical, specific tips for spending less "
             "in those categories this month.")
TRY_ASKING = ["How can I save $200 this month?", "What changed most since last month?",
              "Am I on track with my budgets?", "Show my spending by category as a chart"]


def advisor_context():
    parts = []
    if not spend.empty:
        latest = spend["date"].max().strftime("%Y-%m")
        parts.append(bud.describe_for_advisor(bud.budget_status(spend, budget_map, latest), latest))
    parts.append(savings.describe_for_advisor(spend, income))
    note = corrections.describe_uncertainty(spend)
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def data_key():
    fingerprint = int(pd.util.hash_pandas_object(spend[["id", "predicted_category", "amount"]], index=False).sum()) \
        if not spend.empty else 0
    return (len(spend), round(float(spend["amount"].sum()), 2) if not spend.empty else 0, fingerprint,
            len(income), round(float(income["amount"].sum()), 2) if not income.empty else 0,
            tuple(sorted(budget_map.items())))


def show_plan(plan):
    rows = [{"Category": r["category"], "Usually spend a month": r["average"], "Spend less by": r["cut"],
             "Cut": f"{plan['percent']:.1%}"} for r in plan["rows"]]
    rows.append({"Category": "Total", "Usually spend a month": round(sum(r["average"] for r in plan["rows"]), 2),
                 "Spend less by": plan["total_cut"], "Cut": ""})
    st.markdown(ui.esc(f"**Plan to save {money(plan['goal'])} in one month (calculated by the app)**"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="dollar") for c in ["Usually spend a month", "Spend less by"]})
    st.caption(f"Based on your average of {', '.join(month_label(m) for m in plan['months'])}. Each category is cut by "
               f"the same share, never more than {money_facts.MAX_CUT:.0%}. "
               + (f"{', '.join(plan['excluded'])} left out as mostly fixed costs." if plan["excluded"] else ""))
    if not plan["feasible"]:
        st.warning(ui.esc(f"{money(plan['goal'])} is more than {money_facts.MAX_CUT:.0%} of your flexible spending. "
                          f"This plan reaches {money(plan['max_possible'])}, the most that's realistic in one month."))


def show_visual(spec):
    result = advisor_charts.build(spec, spend, income, budget_map)
    if result is None:
        return
    st.markdown(f"**{ui.esc(result['title'])}**")
    if spec["format"] == "chart" and result["series"] is not None:
        ui.show(ui.hbar(result["series"], by_category=spec["kind"] == "category_totals"), alt=result["title"])
    elif spec["format"] == "chart" and result["chart_df"] is not None:
        ui.show(ui.month_chart(result["chart_df"], result["income_by_month"]), alt=result["title"])
    else:
        st.dataframe(result["table"], hide_index=True)
        return
    with st.expander("See the numbers"):
        st.dataframe(result["table"], hide_index=True)


def handle(question, history):
    note, info = "", {}
    goal = money_facts.detect_goal(question)
    if goal and not spend.empty:
        plan = money_facts.savings_goal(spend, goal, today)
        note = money_facts.describe_goal(plan) + "\n" + GOAL_NOTE
        info["plan"] = plan
    elif advisor_charts.wants_visual(question) and not spend.empty:
        months = sorted(spend["date"].dt.strftime("%Y-%m").unique())
        spec = advisor.pick_visual(question, CATEGORIES, months)
        result = advisor_charts.build(spec, spend, income, budget_map) if spec else None
        if result:
            note = advisor_charts.chart_note(result)
            info["visual"] = spec
    answer, new_history = advisor.ask_advisor(question, history, note)
    sent = question + (advisor.NOTE_MARK + note + "]" if note else "")
    info["model"] = advisor.last_model()
    info["unverified"] = money_facts.unverified_amounts(answer, [history[1]["content"], sent])
    ss.answers[hash(answer)] = info
    return new_history


def page_advisor():
    st.subheader("AI advisor")
    st.caption("Ask about your spending. The app calculates every number; the AI only explains and gives tips.")
    key = data_key()
    if ss.history is None or ss.advisor_key != key:
        ss.history = advisor.start_conversation(spend, advisor_context(), today=today)
        ss.advisor_key = key
        ss.answers = {}

    with st.expander("What the advisor can see"):
        st.caption("The advisor answers only from this summary of your own data. With the local engine it never "
                   "leaves this computer.")
        st.text(ss.history[1]["content"])

    for msg in ss.history[3:]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(ui.plain_dollars(msg["content"].split(advisor.NOTE_MARK)[0]))
        else:
            with st.chat_message("assistant"):
                st.markdown(ui.plain_dollars(msg["content"]))
                info = ss.answers.get(hash(msg["content"]), {})
                if info.get("plan"):
                    show_plan(info["plan"])
                if info.get("visual"):
                    show_visual(info["visual"])
                for amount in info.get("unverified", []):
                    st.warning(ui.esc(f"This answer mentions {amount}, which doesn't come from your data. "
                                      "Check it before relying on it."))
                if info.get("model"):
                    st.caption(f"Answered by {info['model']}")

    if spend.empty:
        st.info("Add some spending first, then the advisor has something to talk about.")
        return

    c1, c2 = st.columns([1, 1])
    insights = c1.button("Get spending insights", icon=":material/lightbulb:", width="stretch")
    if c2.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        ss.history, ss.answers = None, {}
        st.rerun()
    st.caption("Try asking")
    cols = st.columns(len(TRY_ASKING))
    for col, q in zip(cols, TRY_ASKING):
        if col.button(q.replace("$", "\\$"), key=f"try_{q}", width="stretch"):
            ss.queued = q
    st.caption('For a chart or table, ask things like "show my dining spending by month as a chart", '
               '"top 10 purchases" or "income vs spending by month".')
    typed = st.chat_input("Ask a question about your spending...")

    question = typed or ss.queued
    ss.queued = None
    if not (question or insights):
        return
    try:
        with st.spinner("Thinking..."):
            if insights:
                answer, new_history = advisor.get_advice(ss.history)
                ss.answers[hash(answer)] = {
                    "model": advisor.last_model(),
                    "unverified": money_facts.unverified_amounts(answer, [ss.history[1]["content"]])}
                ss.history = new_history
            else:
                ss.history = handle(question, ss.history)
    except Exception as err:
        st.error(ui.esc(f"The advisor couldn't answer: {err} Make sure Ollama is running (`ollama serve`), "
                        "or switch the AI engine in the sidebar to Groq."))
        return
    st.rerun()


# ---------------------------------------------------------------- router

if page == "Overview":
    page_overview()
elif page == "Plan ahead":
    page_plan()
elif page == "Transactions":
    page_transactions()
else:
    page_advisor()
