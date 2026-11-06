from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agi_ecom_app.manager_lifecycle.auth import (
    AuthConfigurationError,
    AuthenticationError,
    authenticate,
    decode_access_token,
    hash_password,
)
from agi_ecom_app.services.memory import AppUserRecord


def _user(password_hash: str, *, active: bool = True) -> AppUserRecord:
    return AppUserRecord(
        id=7,
        username="manager@example.test",
        password_hash=password_hash,
        role="manager",
        is_active=active,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_password_hash_is_argon2id() -> None:
    hashed = hash_password("long-manager-password")
    assert hashed.startswith("$argon2id$")
    assert "long-manager-password" not in hashed


def test_login_issues_short_lived_jwt_and_resolves_current_user(monkeypatch) -> None:
    monkeypatch.setenv("MANAGER_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("MANAGER_JWT_TTL_SECONDS", "300")
    user = _user(hash_password("long-manager-password"))
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.auth.get_app_user_by_username",
            return_value=user,
        ),
        patch("agi_ecom_app.manager_lifecycle.auth.get_app_user", return_value=user),
        patch("agi_ecom_app.manager_lifecycle.auth.record_auth_audit") as audit,
    ):
        result = authenticate("manager@example.test", "long-manager-password")
        actor = decode_access_token(result.access_token)

    assert result.expires_in == 300
    assert actor.user_id == 7
    assert actor.role == "manager"
    assert audit.call_args_list[-1].kwargs["success"] is True


def test_auth_fails_closed_without_jwt_secret(monkeypatch) -> None:
    monkeypatch.delenv("MANAGER_JWT_SECRET", raising=False)
    with pytest.raises(AuthConfigurationError, match="MANAGER_JWT_SECRET"):
        authenticate("manager", "long-manager-password")


def test_inactive_user_cannot_authenticate(monkeypatch) -> None:
    monkeypatch.setenv("MANAGER_JWT_SECRET", "x" * 40)
    user = _user(hash_password("long-manager-password"), active=False)
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.auth.get_app_user_by_username",
            return_value=user,
        ),
        patch("agi_ecom_app.manager_lifecycle.auth.record_auth_audit"),
    ):
        with pytest.raises(AuthenticationError):
            authenticate("manager@example.test", "long-manager-password")
