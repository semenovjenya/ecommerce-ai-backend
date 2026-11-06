from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agi_ecom_app.services import db


class FakePool:
    instances: list["FakePool"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = 0
        self.connection_context = nullcontext(object())
        self.instances.append(self)

    def connection(self):
        return self.connection_context

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def reset_pool_state(monkeypatch: pytest.MonkeyPatch):
    db.close_postgres_connection_pool()
    FakePool.instances.clear()
    monkeypatch.setattr(db, "ConnectionPool", FakePool)
    monkeypatch.setattr(
        db,
        "get_settings",
        lambda: SimpleNamespace(
            postgres_dsn="postgresql://analytics/test",
            postgres_pool_min_size=2,
            postgres_pool_max_size=5,
            postgres_pool_timeout_seconds=3.5,
        ),
    )
    yield
    db.close_postgres_connection_pool()


def test_get_connection_lazily_builds_and_reuses_process_pool() -> None:
    assert FakePool.instances == []

    first_context = db.get_connection()
    second_context = db.get_connection()

    assert first_context is second_context
    assert len(FakePool.instances) == 1
    assert FakePool.instances[0].kwargs == {
        "conninfo": "postgresql://analytics/test",
        "min_size": 2,
        "max_size": 5,
        "timeout": 3.5,
        "kwargs": {"row_factory": db.dict_row},
        "open": True,
        "name": "agi-ecom-analytics",
    }


def test_get_connection_preserves_context_manager_contract() -> None:
    with db.get_connection() as connection:
        assert connection is not None


def test_changed_configuration_closes_and_replaces_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = db._get_postgres_pool()
    monkeypatch.setattr(
        db,
        "get_settings",
        lambda: SimpleNamespace(
            postgres_dsn="postgresql://analytics/other",
            postgres_pool_min_size=1,
            postgres_pool_max_size=1,
            postgres_pool_timeout_seconds=1,
        ),
    )

    second = db._get_postgres_pool()

    assert second is not first
    assert first.closed == 1
    assert len(FakePool.instances) == 2


def test_pool_limits_are_normalized_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        db,
        "get_settings",
        lambda: SimpleNamespace(
            postgres_dsn="postgresql://analytics/test",
            postgres_pool_min_size=4,
            postgres_pool_max_size=2,
            postgres_pool_timeout_seconds=0.25,
        ),
    )

    pool = db._get_postgres_pool()

    assert pool.kwargs["min_size"] == 4
    assert pool.kwargs["max_size"] == 4
    assert pool.kwargs["timeout"] == 1.0


def test_after_fork_discards_pool_without_closing_parent_sockets() -> None:
    pool = db._get_postgres_pool()
    old_lock = db._POSTGRES_POOL_LOCK

    db._discard_inherited_postgres_pool()

    assert pool.closed == 0
    assert db._POSTGRES_POOL is None
    assert db._POSTGRES_POOL_CONFIG is None
    assert db._POSTGRES_POOL_PID is None
    assert db._POSTGRES_POOL_LOCK is not old_lock


def test_close_is_idempotent_and_closes_only_owned_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = db._get_postgres_pool()
    db.close_postgres_connection_pool()
    db.close_postgres_connection_pool()
    assert pool.closed == 1

    inherited = db._get_postgres_pool()
    monkeypatch.setattr(db.os, "getpid", lambda: -1)
    db.close_postgres_connection_pool()
    assert inherited.closed == 0
