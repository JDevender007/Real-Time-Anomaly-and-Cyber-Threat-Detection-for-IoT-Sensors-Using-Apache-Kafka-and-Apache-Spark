"""Security hardening utilities."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path


def generate_secret_key(length: int = 64) -> str:
    return secrets.token_urlsafe(length)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return key.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return hmac.compare_digest(key.hex(), hashed)


def generate_device_token(device_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), device_id.encode(), hashlib.sha256).hexdigest()


def validate_device_token(device_id: str, token: str, secret: str) -> bool:
    expected = generate_device_token(device_id, secret)
    return hmac.compare_digest(token, expected)


def sanitize_filename(filename: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)


def create_secure_headers() -> dict[str, str]:
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }


def rotate_api_keys(key_file: Path | None = None) -> list[str]:
    keys = [secrets.token_urlsafe(32) for _ in range(3)]
    if key_file:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, "w") as f:
            f.write("\n".join(keys))
    return keys
