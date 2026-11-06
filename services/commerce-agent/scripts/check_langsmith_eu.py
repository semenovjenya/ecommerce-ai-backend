"""Read-only LangSmith EU access check; never print credential material."""

from __future__ import annotations

import os
import re

from langsmith import Client

from commerce_agent.observability import EU_LANGSMITH_API_ENDPOINT


def main() -> int:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY is not configured")
        return 2

    try:
        client = Client(api_url=EU_LANGSMITH_API_ENDPOINT, api_key=api_key)
        next(client.list_projects(limit=1), None)
    except Exception as exc:
        causes = (exc, exc.__cause__, exc.__context__)
        status_code = next(
            (
                status
                for cause in causes
                if cause is not None
                for status in (
                    getattr(getattr(cause, "response", None), "status_code", None),
                )
                if isinstance(status, int)
            ),
            None,
        )
        if status_code is None:
            status_match = re.search(r"\b(401|403|404|429|5\d\d)\b", str(exc))
            status_code = int(status_match.group(1)) if status_match else None
        if isinstance(status_code, int):
            print(f"LangSmith EU access failed: HTTP {status_code}")
        else:
            cause_name = type(exc.__cause__).__name__ if exc.__cause__ else "none"
            print(
                "LangSmith EU access failed: "
                f"{type(exc).__name__} (cause: {cause_name})"
            )
        return 1

    print("LangSmith EU API access confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
