"""Sign up and log in. Passwords are stored as salted PBKDF2 hashes, never as text."""
import hashlib
import hmac
import secrets
import sqlite3

import db

ITERATIONS = 100_000


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS).hex()


def signup(username, password):
    username = (username or "").strip()
    if not username:
        return False, "Enter a username."
    if len(password or "") < 6:
        return False, "Use a password with at least 6 characters."
    if db.get_user_by_username(username):
        return False, "That username is taken. Pick another one."
    salt = secrets.token_hex(16)
    try:
        db.create_user(username, hash_password(password, salt), salt)
    except sqlite3.IntegrityError:
        return False, "That username is taken. Pick another one."
    return True, "Account created. You can log in now."


def login(username, password):
    user = db.get_user_by_username((username or "").strip())
    if not user:
        return None, "Wrong username or password."
    if not hmac.compare_digest(hash_password(password or "", user["salt"]), user["password_hash"]):
        return None, "Wrong username or password."
    return user["id"], f"Welcome back, {user['username']}."
