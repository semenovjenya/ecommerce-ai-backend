from __future__ import annotations

import pytest

from commerce_agent.safety import SensitiveStateError, assert_secret_free


@pytest.mark.parametrize(
    "candidate",
    [
        {"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
        {"nested": {"api_key": "value"}},
        {"messages": ["use sk-abcdefghijklmnopqrstuv"]},
        {"messages": ["ghp_" + "a" * 36]},
        {"messages": ["xoxb-" + "0" * 10 + "-" + "a" * 26]},
        {"messages": ["AKIA" + "A" * 16]},
        {"messages": ["-----BEGIN PRIVATE KEY-----\nnot-a-real-key"]},
        {"messages": ["AIza" + "a" * 35]},
        {"messages": ["eyJabcdefghijk.eyJabcdefghijkl.abcdefghijklmnop"]},
        {"messages": [{"content": "lsv2_" + "a" * 32}]},
        {"messages": ["sk-lf-" + "a" * 32]},
        {"dsn": "postgresql://user:password@example.invalid/database"},
        {"note": "password=definitely-not-durable"},
    ],
)
def test_secret_canaries_are_rejected(candidate: object) -> None:
    with pytest.raises(SensitiveStateError):
        assert_secret_free(candidate)


def test_normal_business_state_is_allowed() -> None:
    assert_secret_free(
        {
            "tenant_id": "tenant-1",
            "messages": ["compare sales and finance"],
            "evidence_matrix": [],
        }
    )
