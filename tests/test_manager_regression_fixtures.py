from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manager_decision"


def test_required_historical_manager_regression_fixtures_are_present() -> None:
    fixtures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(FIXTURE_ROOT.glob("run_*.json"))]

    assert [item["run_id"] for item in fixtures] == [149, 156, 160, 162, 191, 192, 196, 197]
    for fixture in fixtures:
        assert fixture["case"]
        assert fixture["hazard"]
        reference = Path(fixture["test_reference"].split("::", 1)[0])
        assert reference.exists(), fixture["test_reference"]
