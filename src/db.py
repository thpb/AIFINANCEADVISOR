"""SQLite storage. Every query filters by user_id, so users never see each other's rows.

init_db() is safe to run on every start and upgrades older databases in place.
"""
import os
import sqlite3
from collections import Counter

import pandas as pd

DB_PATH = "data/app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, salt TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
  date TEXT NOT NULL, merchant TEXT NOT NULL, amount REAL NOT NULL, predicted_category TEXT,
  is_recurring INTEGER DEFAULT 0, is_ambiguous_merchant INTEGER DEFAULT 0, source TEXT DEFAULT 'upload',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS budgets (user_id INTEGER NOT NULL, category TEXT NOT NULL, monthly_limit REAL NOT NULL,
  PRIMARY KEY (user_id, category), FOREIGN KEY (user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS merchant_rules (user_id INTEGER NOT NULL, merchant_key TEXT NOT NULL, category TEXT NOT NULL,
  PRIMARY KEY (user_id, merchant_key), FOREIGN KEY (user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS income (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, date TEXT NOT NULL,
  source TEXT NOT NULL, amount REAL NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id));
"""

TXN_COLUMNS = ["id", "date", "merchant", "amount", "predicted_category", "is_recurring",
               "is_ambiguous_merchant", "source", "confidence", "created_at"]


def get_connection():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(transactions)")]
        if "confidence" not in cols:
            # Old rows stay NULL, meaning "not scored".
            conn.execute("ALTER TABLE transactions ADD COLUMN confidence REAL")
    conn.close()


def _run(sql, params=()):
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(sql, params)
            return cur.rowcount, cur.lastrowid
    finally:
        conn.close()


def _query(sql, params=()):
    conn = get_connection()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ---------- users ----------

def create_user(username, password_hash, salt):
    _, new_id = _run("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                     (username, password_hash, salt))
    return new_id


def delete_user(user_id):
    """Remove a user and everything they own (used to rebuild the demo account)."""
    conn = get_connection()
    try:
        with conn:
            for table in ["transactions", "income", "budgets", "merchant_rules"]:
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    finally:
        conn.close()


def get_user_by_username(username):
    rows = _query("SELECT id, username, password_hash, salt FROM users WHERE username = ?", (username,))
    if not rows:
        return None
    return dict(zip(["id", "username", "password_hash", "salt"], rows[0]))


# ---------- transactions ----------

def _none_if_nan(value):
    return None if value is None or pd.isna(value) else float(value)


def insert_transactions(user_id, df, source="upload"):
    rows = []
    for _, row in df.iterrows():
        rows.append((
            user_id, str(row["date"]), str(row["merchant"]), float(row["amount"]),
            row.get("predicted_category"), int(row.get("is_recurring", 0) or 0),
            int(row.get("is_ambiguous_merchant", 0) or 0), source, _none_if_nan(row.get("confidence")),
        ))
    conn = get_connection()
    try:
        with conn:
            conn.executemany(
                "INSERT INTO transactions (user_id, date, merchant, amount, predicted_category, is_recurring,"
                " is_ambiguous_merchant, source, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    finally:
        conn.close()
    return len(rows)


def _txn_key(date, merchant, amount):
    return (pd.Timestamp(date).strftime("%Y-%m-%d"), str(merchant).strip().upper(), round(float(amount), 2))


def only_new(user_id, df):
    """Rows of df that aren't stored yet (same day, merchant and amount), so a re-imported CSV adds nothing.

    Repeats still count: a file with two identical coffees adds the second one only if the database has one.
    """
    stored = get_transactions(user_id)
    left = Counter(_txn_key(d, m, a) for d, m, a in zip(stored["date"], stored["merchant"], stored["amount"]))
    keep = []
    for d, m, a in zip(df["date"], df["merchant"], df["amount"]):
        key = _txn_key(d, m, a)
        keep.append(left[key] == 0)
        if left[key]:
            left[key] -= 1
    return df[keep]


def get_transactions(user_id):
    rows = _query(f"SELECT {', '.join(TXN_COLUMNS)} FROM transactions WHERE user_id = ? ORDER BY date, id",
                  (user_id,))
    return pd.DataFrame(rows, columns=TXN_COLUMNS)


def delete_all_transactions(user_id):
    return _run("DELETE FROM transactions WHERE user_id = ?", (user_id,))[0]


def update_transaction(user_id, txn_id, date, merchant, amount, category):
    return _run("UPDATE transactions SET date = ?, merchant = ?, amount = ?, predicted_category = ?, confidence = 1.0"
                " WHERE user_id = ? AND id = ?",
                (str(date), str(merchant), float(amount), category, user_id, int(txn_id)))[0]


def delete_transactions(user_id, ids):
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    return _run(f"DELETE FROM transactions WHERE user_id = ? AND id IN ({marks})", (user_id, *ids))[0]


def update_categories(user_id, ids, category):
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    return _run(f"UPDATE transactions SET predicted_category = ?, confidence = 1.0"
                f" WHERE user_id = ? AND id IN ({marks})", (category, user_id, *ids))[0]


# ---------- budgets ----------

def set_budget(user_id, category, monthly_limit):
    _run("INSERT INTO budgets (user_id, category, monthly_limit) VALUES (?, ?, ?)"
         " ON CONFLICT(user_id, category) DO UPDATE SET monthly_limit = excluded.monthly_limit",
         (user_id, category, float(monthly_limit)))


def delete_budget(user_id, category):
    _run("DELETE FROM budgets WHERE user_id = ? AND category = ?", (user_id, category))


def get_budgets(user_id):
    return {c: lim for c, lim in _query("SELECT category, monthly_limit FROM budgets WHERE user_id = ?", (user_id,))}


# ---------- merchant rules ----------

def set_rule(user_id, merchant_key, category):
    _run("INSERT INTO merchant_rules (user_id, merchant_key, category) VALUES (?, ?, ?)"
         " ON CONFLICT(user_id, merchant_key) DO UPDATE SET category = excluded.category",
         (user_id, merchant_key, category))


def delete_rule(user_id, merchant_key):
    _run("DELETE FROM merchant_rules WHERE user_id = ? AND merchant_key = ?", (user_id, merchant_key))


def get_rules(user_id):
    return {k: c for k, c in _query("SELECT merchant_key, category FROM merchant_rules WHERE user_id = ?", (user_id,))}


# ---------- income ----------

INCOME_COLUMNS = ["id", "date", "source", "amount"]


def add_income(user_id, date, source, amount):
    return _run("INSERT INTO income (user_id, date, source, amount) VALUES (?, ?, ?, ?)",
                (user_id, str(date), str(source), float(amount)))[1]


def get_income(user_id):
    rows = _query("SELECT id, date, source, amount FROM income WHERE user_id = ? ORDER BY date, id", (user_id,))
    return pd.DataFrame(rows, columns=INCOME_COLUMNS)


def update_income(user_id, income_id, date, source, amount):
    return _run("UPDATE income SET date = ?, source = ?, amount = ? WHERE user_id = ? AND id = ?",
                (str(date), str(source), float(amount), user_id, int(income_id)))[0]


def delete_income(user_id, ids):
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    return _run(f"DELETE FROM income WHERE user_id = ? AND id IN ({marks})", (user_id, *ids))[0]


def delete_all_income(user_id):
    return _run("DELETE FROM income WHERE user_id = ?", (user_id,))[0]
