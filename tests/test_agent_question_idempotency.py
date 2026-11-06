import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agi_ecom_app import agent_queue
from agi_ecom_app.services import memory


class AgentQuestionIdempotencyTests(unittest.TestCase):
    def test_concurrent_enqueue_recovers_deterministic_existing_job(self):
        existing_job = SimpleNamespace(id="agent-question-42")
        queue = MagicMock()
        queue.enqueue_call.side_effect = RuntimeError("job already exists")
        queue.fetch_job.side_effect = [None, existing_job]
        settings = SimpleNamespace(
            redis_url="redis://test",
            agent_queue_name="agent-runs",
            agent_job_timeout_seconds=900,
        )
        with (
            patch(
                "agi_ecom_app.agent_queue.get_agent_run_question",
                return_value=SimpleNamespace(status="pending", job_id=None),
            ),
            patch("agi_ecom_app.agent_queue.get_settings", return_value=settings),
            patch("agi_ecom_app.agent_queue._build_rq_queue", return_value=queue),
            patch("agi_ecom_app.agent_queue._build_rq_retry", return_value=None),
            patch("agi_ecom_app.agent_queue.mark_agent_run_question_queued") as mark,
        ):
            job_id = agent_queue.enqueue_agent_question(42)

        self.assertEqual(job_id, "agent-question-42")
        self.assertEqual(queue.fetch_job.call_count, 2)
        queue.fetch_job.assert_called_with("agent-question-42")
        mark.assert_called_once()

    def test_enqueue_terminal_question_is_a_noop(self):
        with (
            patch(
                "agi_ecom_app.agent_queue.get_agent_run_question",
                return_value=SimpleNamespace(
                    status="completed",
                    job_id="agent-question-42",
                ),
            ),
            patch("agi_ecom_app.agent_queue._build_rq_queue") as build_queue,
            patch("agi_ecom_app.agent_queue.mark_agent_run_question_queued") as mark_queued,
        ):
            job_id = agent_queue.enqueue_agent_question(42)

        self.assertEqual(job_id, "agent-question-42")
        build_queue.assert_not_called()
        mark_queued.assert_not_called()

    def test_duplicate_delivery_does_not_execute_question_or_advance_run(self):
        current = SimpleNamespace(
            id=42,
            run_id=7,
            status="completed",
            job_id="agent-question-42",
        )
        with (
            patch(
                "agi_ecom_app.agent_queue.claim_agent_run_question_running",
                return_value=None,
            ),
            patch(
                "agi_ecom_app.agent_queue.get_agent_run_question",
                return_value=current,
            ),
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id") as get_trace,
            patch("agi_ecom_app.agent_queue._execute_question") as execute_question,
            patch("agi_ecom_app.agent_queue.advance_agent_run") as advance_run,
        ):
            agent_queue.run_agent_question_job(42)

        get_trace.assert_not_called()
        execute_question.assert_not_called()
        advance_run.assert_not_called()

    def test_claimed_delivery_executes_exactly_once(self):
        claimed = SimpleNamespace(
            id=42,
            run_id=7,
            status="running",
            job_id="agent-question-42",
            attempt_count=1,
        )
        trace = SimpleNamespace(id=7, status="running")
        with (
            patch(
                "agi_ecom_app.agent_queue.claim_agent_run_question_running",
                return_value=claimed,
            ),
            patch(
                "agi_ecom_app.agent_queue.get_agent_run_trace_by_id",
                return_value=trace,
            ),
            patch("agi_ecom_app.agent_queue._execute_question") as execute_question,
            patch("agi_ecom_app.agent_queue.advance_agent_run") as advance_run,
        ):
            agent_queue.run_agent_question_job(42)

        execute_question.assert_called_once_with(claimed)
        advance_run.assert_called_once_with(7)

    def test_database_claim_is_compare_and_set(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = None
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value = cursor

        with patch(
            "agi_ecom_app.services.memory.get_memory_connection",
            return_value=connection,
        ):
            claimed = memory.claim_agent_run_question_running(42)

        self.assertIsNone(claimed)
        sql = " ".join(cursor.execute.call_args.args[0].split()).lower()
        self.assertIn("where id = %s and status in ('pending', 'queued')", sql)

    def test_late_queued_write_cannot_regress_terminal_status(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = None
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value = cursor
        terminal = SimpleNamespace(id=42, status="completed")

        with (
            patch(
                "agi_ecom_app.services.memory.get_memory_connection",
                return_value=connection,
            ),
            patch(
                "agi_ecom_app.services.memory.get_agent_run_question",
                return_value=terminal,
            ),
        ):
            result = memory.mark_agent_run_question_queued(
                question_id=42,
                job_id="agent-question-42",
                timeout_seconds=120,
            )

        self.assertIs(result, terminal)
        sql = " ".join(cursor.execute.call_args.args[0].split()).lower()
        self.assertIn("where id = %s and status in ('pending', 'queued')", sql)

    def test_stale_reconciler_fails_killed_question_and_advances_run(self):
        now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        question = SimpleNamespace(id=42, run_id=7)
        settings = SimpleNamespace(agent_job_timeout_seconds=900)
        with (
            patch("agi_ecom_app.agent_queue.get_settings", return_value=settings),
            patch(
                "agi_ecom_app.agent_queue.claim_timed_out_agent_run_questions",
                return_value=[question],
            ) as claim,
            patch(
                "agi_ecom_app.agent_queue.claim_stale_agent_questions_for_delivery",
                return_value=[],
            ),
            patch("agi_ecom_app.agent_queue.advance_agent_run") as advance,
        ):
            result = agent_queue.reconcile_stale_agent_questions(now=now)

        self.assertEqual(
            result,
            {
                "requeued_question_ids": [],
                "timed_out_question_ids": [42],
                "advanced_run_ids": [7],
            },
        )
        claim.assert_called_once_with(
            now=now,
            default_timeout_seconds=900,
            grace_seconds=60,
            limit=100,
        )
        advance.assert_called_once_with(7)

    def test_stale_claim_uses_skip_locked_and_retries_timeout_marker(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value = cursor
        now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

        with (
            patch("agi_ecom_app.services.memory.ensure_memory_schema"),
            patch(
                "agi_ecom_app.services.memory.get_memory_connection",
                return_value=connection,
            ),
        ):
            result = memory.claim_timed_out_agent_run_questions(
                now=now,
                default_timeout_seconds=900,
            )

        self.assertEqual(result, [])
        sql = " ".join(cursor.execute.call_args.args[0].split()).lower()
        self.assertIn("for update skip locked", sql)
        self.assertIn("agent_question_timeout_reconciled", sql)
        self.assertIn("questions.status = 'running'", sql)

    def test_stale_queued_delivery_is_reenqueued(self):
        now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        question = SimpleNamespace(id=42, run_id=7)
        settings = SimpleNamespace(agent_job_timeout_seconds=900)
        with (
            patch("agi_ecom_app.agent_queue.get_settings", return_value=settings),
            patch(
                "agi_ecom_app.agent_queue.claim_stale_agent_questions_for_delivery",
                return_value=[question],
            ) as claim_delivery,
            patch(
                "agi_ecom_app.agent_queue.claim_timed_out_agent_run_questions",
                return_value=[],
            ),
            patch("agi_ecom_app.agent_queue.enqueue_agent_question") as enqueue,
            patch("agi_ecom_app.agent_queue.advance_agent_run") as advance,
        ):
            result = agent_queue.reconcile_stale_agent_questions(now=now)

        self.assertEqual(result["requeued_question_ids"], [42])
        claim_delivery.assert_called_once()
        enqueue.assert_called_once_with(42)
        advance.assert_not_called()

    def test_stale_delivery_claim_covers_queued_and_pending_with_skip_locked(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value = cursor
        now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

        with (
            patch("agi_ecom_app.services.memory.ensure_memory_schema"),
            patch(
                "agi_ecom_app.services.memory.get_memory_connection",
                return_value=connection,
            ),
        ):
            result = memory.claim_stale_agent_questions_for_delivery(
                now=now,
                default_timeout_seconds=900,
            )

        self.assertEqual(result, [])
        sql = " ".join(cursor.execute.call_args.args[0].split()).lower()
        self.assertIn("questions.status = 'queued'", sql)
        self.assertIn("questions.status = 'pending'", sql)
        self.assertIn("for update skip locked", sql)

    def test_reconciler_schedule_keeps_five_future_slots(self):
        now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        queue = MagicMock()
        settings = SimpleNamespace(
            redis_url="redis://test",
            agent_queue_name="agent-runs",
        )
        with patch("agi_ecom_app.agent_queue.get_settings", return_value=settings):
            first = agent_queue.schedule_agent_question_reconciliation(
                queue=queue,
                now=now,
            )

        self.assertEqual(queue.enqueue_at.call_count, 5)
        self.assertEqual(first, "agent-question-reconcile-29736721")
        job_ids = [call.kwargs["job_id"] for call in queue.enqueue_at.call_args_list]
        self.assertEqual(len(set(job_ids)), 5)


if __name__ == "__main__":
    unittest.main()
