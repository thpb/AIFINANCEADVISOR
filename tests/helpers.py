"""Shared test setup: put src/ on the path and give each test a throwaway database."""
import os
import sys
import tempfile
import unittest
from datetime import date

import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import db  # noqa: E402


class DBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp.name, "test.db")
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self._old_path
        self._tmp.cleanup()

    def make_user(self, name="alice"):
        return db.create_user(name, "hash", "salt")


def txns(rows):
    """rows: (date, merchant, amount, category) tuples -> DataFrame like db.get_transactions."""
    return pd.DataFrame(rows, columns=["date", "merchant", "amount", "predicted_category"])


TODAY = date(2026, 10, 29)   # a Thursday
