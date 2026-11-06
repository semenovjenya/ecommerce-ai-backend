from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from agi_ecom_app.services.memory import (
    AppUserRecord,
    get_app_user,
    get_app_user_by_username,
    record_auth_audit,
    upsert_app_user,
)

try:
    import jwt
except ImportError:  # pragma: no cover - exercised only in a broken deployment
    jwt = None  # type: ignore[assignment]

try:
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
    from argon2.low_level import Type
except ImportError:  # pragma: no cover - exercised only in a broken deployment
    PasswordHasher = None  # type: ignore[assignment,misc]
    InvalidHashError = VerificationError = VerifyMismatchError = Exception  # type: ignore[misc]
    Type = None  # type: ignore[assignment]


ManagerRole = Literal["viewer", "manager", "admin"]
_ROLE_RANK = {"viewer": 0, "manager": 1, "admin": 2}


class AuthConfigurationError(RuntimeError):
    pass


class AuthenticationError(RuntimeError):
    pass


class AuthorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManagerActor:
    user_id: int
    username: str
    role: ManagerRole


@dataclass(frozen=True)
class LoginResult:
    access_token: str
    token_type: str
    expires_in: int
    actor: ManagerActor


def _password_hasher() -> PasswordHasher:
    if PasswordHasher is None or Type is None:
        raise AuthConfigurationError(
            "Argon2id support is unavailable; install argon2-cffi"
        )
    return PasswordHasher(
        time_cost=3,
        memory_cost=65_536,
        parallelism=2,
        hash_len=32,
        salt_len=16,
        type=Type.ID,
    )


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    return _password_hasher().hash(password)


def create_or_update_user(
    *,
    username: str,
    password: str,
    role: ManagerRole,
    is_active: bool = True,
) -> AppUserRecord:
    normalized = username.strip()
    if not normalized:
        raise ValueError("username is required")
    if role not in _ROLE_RANK:
        raise ValueError("unsupported role")
    return upsert_app_user(
        username=normalized,
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
    )


def authenticate(username: str, password: str) -> LoginResult:
    _jwt_secret()
    normalized = username.strip()
    user = get_app_user_by_username(normalized)
    if user is None:
        user = _maybe_create_bootstrap_admin(normalized, password)
    if user is None or not user.is_active:
        record_auth_audit(
            username=normalized,
            user_id=user.id if user else None,
            event="login",
            success=False,
            details={"reason": "invalid_credentials"},
        )
        raise AuthenticationError("invalid username or password")

    try:
        verified = _password_hasher().verify(user.password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        verified = False
    if not verified:
        record_auth_audit(
            username=normalized,
            user_id=user.id,
            event="login",
            success=False,
            details={"reason": "invalid_credentials"},
        )
        raise AuthenticationError("invalid username or password")

    actor = _actor_from_user(user)
    token, ttl = _encode_access_token(actor)
    record_auth_audit(
        username=user.username,
        user_id=user.id,
        event="login",
        success=True,
    )
    return LoginResult(
        access_token=token,
        token_type="bearer",
        expires_in=ttl,
        actor=actor,
    )


def decode_access_token(token: str) -> ManagerActor:
    if jwt is None:
        raise AuthConfigurationError("JWT support is unavailable; install PyJWT")
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=_jwt_issuer(),
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
        user_id = int(payload["sub"])
    except Exception as exc:
        raise AuthenticationError("invalid or expired access token") from exc
    user = get_app_user(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("user is inactive or no longer exists")
    return _actor_from_user(user)


def require_role(actor: ManagerActor, minimum_role: ManagerRole) -> None:
    if _ROLE_RANK[actor.role] < _ROLE_RANK[minimum_role]:
        raise AuthorizationError(f"{minimum_role} role is required")


def _encode_access_token(actor: ManagerActor) -> tuple[str, int]:
    if jwt is None:
        raise AuthConfigurationError("JWT support is unavailable; install PyJWT")
    ttl = _jwt_ttl_seconds()
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": str(actor.user_id),
            "username": actor.username,
            "role": actor.role,
            "iat": now,
            "exp": now + timedelta(seconds=ttl),
            "iss": _jwt_issuer(),
        },
        _jwt_secret(),
        algorithm="HS256",
    )
    return token, ttl


def _jwt_secret() -> str:
    secret = (os.getenv("MANAGER_JWT_SECRET") or "").strip()
    if len(secret.encode("utf-8")) < 32:
        raise AuthConfigurationError(
            "MANAGER_JWT_SECRET must be configured with at least 32 bytes"
        )
    return secret


def _jwt_issuer() -> str:
    return (os.getenv("MANAGER_JWT_ISSUER") or "ai-ecom-manager").strip()


def _jwt_ttl_seconds() -> int:
    raw = os.getenv("MANAGER_JWT_TTL_SECONDS", "900")
    try:
        value = int(raw)
    except ValueError as exc:
        raise AuthConfigurationError("MANAGER_JWT_TTL_SECONDS must be an integer") from exc
    if not 60 <= value <= 3600:
        raise AuthConfigurationError("MANAGER_JWT_TTL_SECONDS must be between 60 and 3600")
    return value


def _maybe_create_bootstrap_admin(username: str, password: str) -> AppUserRecord | None:
    bootstrap_username = (os.getenv("MANAGER_BOOTSTRAP_ADMIN_USERNAME") or "").strip()
    bootstrap_password = os.getenv("MANAGER_BOOTSTRAP_ADMIN_PASSWORD") or ""
    if not bootstrap_username or not bootstrap_password:
        return None
    if username.casefold() != bootstrap_username.casefold() or password != bootstrap_password:
        return None
    return create_or_update_user(
        username=bootstrap_username,
        password=bootstrap_password,
        role="admin",
    )


def _actor_from_user(user: AppUserRecord) -> ManagerActor:
    if user.role not in _ROLE_RANK:
        raise AuthenticationError("user has an unsupported role")
    return ManagerActor(
        user_id=user.id,
        username=user.username,
        role=user.role,  # type: ignore[arg-type]
    )


__all__ = [
    "AuthConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "LoginResult",
    "ManagerActor",
    "ManagerRole",
    "authenticate",
    "create_or_update_user",
    "decode_access_token",
    "hash_password",
    "require_role",
]
