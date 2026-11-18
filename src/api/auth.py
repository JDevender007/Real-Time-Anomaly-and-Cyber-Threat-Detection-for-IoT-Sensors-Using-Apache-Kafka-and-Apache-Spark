"""API key authentication middleware."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Callable

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_default_api_key: str | None = None


def _get_api_key() -> str:
    global _default_api_key
    if _default_api_key is None:
        import os
        _default_api_key = os.environ.get("IOT_SEC_API_KEY") or secrets.token_urlsafe(32)
    return _default_api_key


def get_api_keys() -> list[str]:
    return [_get_api_key()]


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    valid_keys = get_api_keys()
    if not any(hmac.compare_digest(api_key, key) for key in valid_keys):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )
    return api_key


def require_api_key(request: Request) -> str | None:
    api_key = request.headers.get("X-API-Key")
    if api_key is None:
        return None
    valid_keys = get_api_keys()
    if any(hmac.compare_digest(api_key, key) for key in valid_keys):
        return api_key
    return None
