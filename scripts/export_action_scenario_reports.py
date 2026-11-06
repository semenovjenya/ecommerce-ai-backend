from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agi_ecom_app.reporting.action_scenarios import (
    ScenarioReportRequest,
    export_action_scenario_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export two historical-only action scenario reports.",
    )
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--server-message-id", required=True, type=int)
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument(
        "--source-task-id",
        action="append",
        dest="source_task_ids",
        required=True,
        help="Repeat for each source task id that should be included in the reports.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = ScenarioReportRequest(
        chat_id=args.chat_id,
        server_message_id=args.server_message_id,
        source_run_id=args.source_run_id,
        source_task_ids=args.source_task_ids,
        output_dir=Path(args.output_dir),
    )
    paths = export_action_scenario_reports(request)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
