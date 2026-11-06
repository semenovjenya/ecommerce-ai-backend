from __future__ import annotations

import json
from pathlib import Path
import tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_deployment_contract() -> None:
    config = json.loads((PACKAGE_ROOT / "langgraph.json").read_text(encoding="utf-8"))

    assert config["python_version"] == "3.11"
    assert config["dependencies"] == ["."]
    assert config["graphs"] == {
        "commerce_agent": "./src/commerce_agent/graph.py:graph"
    }
    assert config["checkpointer"]["ttl"] == {
        "strategy": "keep_latest",
        "sweep_interval_minutes": 60,
        "default_ttl": 43_200,
    }


def test_self_managed_postgres_dependencies_are_pinned() -> None:
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(pyproject["project"]["dependencies"])

    assert "langgraph-checkpoint-postgres==3.1.0" in dependencies
    assert "psycopg[binary,pool]==3.3.4" in dependencies


def test_local_persistence_compose_uses_a_dedicated_database_and_volume() -> None:
    compose = (PACKAGE_ROOT / "compose.persistence.yml").read_text(encoding="utf-8")
    env_example = (PACKAGE_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "agent-postgres:" in compose
    assert "POSTGRES_DB: commerce_agent" in compose
    assert "agent-postgres-data:/var/lib/postgresql/data" in compose
    assert "COMMERCE_AGENT_DATABASE_URI=" in env_example
    assert "COMMERCE_AGENT_DB_POOL_MIN_SIZE=" in env_example
    assert "COMMERCE_AGENT_DB_POOL_MAX_SIZE=" in env_example
