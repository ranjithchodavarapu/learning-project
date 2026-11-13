"""
auth.py — Authentication: password hashing, signed tokens, brute-force lockout.

Uses only the Python standard library (hashlib, hmac, secrets, base64, json),
so there is nothing extra to install and every step is visible.

LINE BY LINE (security-relevant lines):

  hash_password(password, salt)
      hashlib.pbkdf2_hmac("sha256", password, salt, iterations)
      PBKDF2 = a deliberately SLOW hash. A fast hash like plain SHA-256 lets an
      attacker try billions of guesses per second. Repeating it 100,000 times
      slows each guess down. The SALT is random per user, so two users with the
      same password get different hashes, and precomputed "rainbow tables" are
      useless.

  hmac.compare_digest(a, b)
      Compares two strings in CONSTANT TIME. A normal `a == b` stops at the
      first differing character, so the time it takes leaks how many leading
      characters were correct (a timing attack). compare_digest always takes
      the same time.

  _dummy_salt + "always hash" in login()
      If the username does not exist we STILL run the expensive hash against a
      dummy salt. Otherwise "unknown user" returns instantly while "wrong
      password" takes 50 ms, and an attacker can enumerate valid usernames by
      timing. Same work in both cases = same response time.

  The error message is identical for unknown user and wrong password:
      "Invalid username or password". Never tell the attacker which was wrong.

  Lockout (_failures, _locked)
      After MAX_FAILED_LOGINS wrong attempts the account is locked until
      `now + LOCKOUT_SECONDS`. Failure counters reset on success.

  Token format:  base64url(payload_json) + "." + base64url(HMAC-SHA256 signature)
      This is the same idea as a JWT, built by hand so you can see it.
      The server signs the payload with SECRET_KEY. Any change to the payload
      (e.g. editing "role":"guest" to "role":"admin") invalidates the
      signature, because the attacker does not have SECRET_KEY.

  verify_token()
      1. split into body and signature
      2. recompute the signature and compare in constant time
      3. check expiry
      4. RE-LOAD the user from the store and use THEIR CURRENT role,
         not the role written in the token. If an admin is demoted or deleted,
         their old token stops working at once instead of at expiry.

  _b64e / _b64d
      base64url without "=" padding. _b64d re-adds the padding:
      `"=" * (-len(s) % 4)` pads to a multiple of 4 characters.

  build_demo_users()
      Creates 4 demo accounts, hashed at start-up. DEMO ONLY: a real system
      reads users from a database or an SSO provider (Okta, Azure AD) and
      never keeps passwords in source code.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Dict, Optional

from config import cfg
from models import Session, User


class AuthError(Exception):
    """Raised for any authentication failure. The message is safe to show."""


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        cfg.PBKDF2_ITERATIONS,
    )
    return digest.hex()


def new_salt() -> str:
    return secrets.token_hex(16)   # 16 random bytes = 32 hex characters


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class AuthService:
    """Logs users in, issues tokens, and verifies them."""

    def __init__(self, users: Dict[str, User]):
        self.users = users
        self._failures: Dict[str, int] = {}
        self._locked: Dict[str, float] = {}
        self._dummy_salt = new_salt()

    # ── login ─────────────────────────────────────────────────────────────
    def login(self, username: str, password: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now

        if now < self._locked.get(username, 0.0):
            raise AuthError("Account temporarily locked. Try again later.")

        user = self.users.get(username)

        # Always do the slow hash, even for unknown users (timing defence).
        salt = user.salt if user else self._dummy_salt
        candidate = hash_password(password, salt)
        ok = user is not None and hmac.compare_digest(candidate, user.password_hash)

        if not ok:
            count = self._failures.get(username, 0) + 1
            self._failures[username] = count
            if count >= cfg.MAX_FAILED_LOGINS:
                self._locked[username] = now + cfg.LOCKOUT_SECONDS
                self._failures[username] = 0
            raise AuthError("Invalid username or password")

        self._failures.pop(username, None)
        return self.create_token(user, now)

    # ── tokens ────────────────────────────────────────────────────────────
    def _sign(self, body: str) -> str:
        mac = hmac.new(cfg.SECRET_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
        return _b64e(mac.digest())

    def create_token(self, user: User, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        payload = {
            "sub": user.username,
            "role": user.role,
            "dept": user.department,
            "exp": int(now + cfg.TOKEN_TTL_SECONDS),
        }
        body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        return f"{body}.{self._sign(body)}"

    def verify_token(self, token: str, now: Optional[float] = None) -> Session:
        now = time.time() if now is None else now

        try:
            body, signature = token.split(".")
        except (ValueError, AttributeError):
            raise AuthError("Malformed token")

        if not hmac.compare_digest(signature, self._sign(body)):
            raise AuthError("Invalid token signature")

        try:
            payload = json.loads(_b64d(body))
            username, expires = payload["sub"], payload["exp"]
        except (ValueError, KeyError, TypeError):
            raise AuthError("Malformed token payload")

        if now >= expires:
            raise AuthError("Token expired")

        # Re-load the user: honour role changes and deletions immediately.
        user = self.users.get(username)
        if user is None:
            raise AuthError("User no longer exists")

        return Session(
            username=user.username,
            role=user.role,
            department=user.department,
            expires_at=float(expires),
        )


# ── demo accounts (DEMO ONLY) ─────────────────────────────────────────────
DEMO_PASSWORDS = {
    "alice": ("Admin#2024",   "admin",    "it"),
    "bob":   ("Manager#2024", "manager",  "finance"),
    "carol": ("Employee#2024", "employee", "engineering"),
    "dave":  ("Guest#2024",   "guest",    "none"),
}


def build_demo_users() -> Dict[str, User]:
    users = {}
    for username, (password, role, department) in DEMO_PASSWORDS.items():
        salt = new_salt()
        users[username] = User(
            username=username,
            role=role,
            department=department,
            password_hash=hash_password(password, salt),
            salt=salt,
        )
    return users
