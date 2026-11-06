"""Explicit schema migration for self-managed Agent Protocol records."""

from __future__ import annotations

import argparse
import asyncio

from commerce_agent.persistence import PersistenceSettings
from commerce_agent.protocol_store import initialize_protocol_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup",))
    args = parser.parse_args()
    if args.command == "setup":
        asyncio.run(initialize_protocol_schema(PersistenceSettings.from_environment()))
        print("protocol_schema_ready")


if __name__ == "__main__":
    main()
