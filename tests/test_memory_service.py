import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from agi_ecom_app.schemas import ChatRole
from agi_ecom_app.services import memory


class FakeMemoryStore:
    def __init__(self):
        self.sessions = {}
        self.messages = []
        self.summaries = {}
        self.traces = []
        self.rounds = []
        self.questions = []
        self.responses = []
        self.gaps = []
        self.action_tasks = []
        self.action_executions = []
        self.now_counter = 0
        self.trace_id_counter = 0
        self.round_id_counter = 0
        self.question_id_counter = 0
        self.response_id_counter = 0
        self.gap_id_counter = 0
        self.action_execution_id_counter = 0

    def now(self):
        self.now_counter += 1
        return datetime(2026, 3, 25, 12, 0, self.now_counter, tzinfo=timezone.utc)


class FakeMemoryCursor:
    def __init__(self, store: FakeMemoryStore):
        self.store = store
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        params = params or []
        normalized = " ".join(query.split()).lower()

        if normalized.startswith("select pg_advisory_xact_lock"):
            self._result = [{"ok": 1}]
            return

        if "create table if not exists chat_sessions" in normalized:
            self._result = None
            return

        if normalized.startswith("insert into chat_sessions"):
            chat_id, model_name, status = params
            existing = self.store.sessions.get(chat_id, {})
            self.store.sessions[chat_id] = {
                "chat_id": chat_id,
                "created_at": existing.get("created_at", self.store.now()),
                "updated_at": self.store.now(),
                "model_name": model_name,
                "status": status,
            }
            self._result = None
            return

        if normalized.startswith("update chat_sessions set updated_at = now()"):
            chat_id = params[0]
            if chat_id in self.store.sessions:
                self.store.sessions[chat_id]["updated_at"] = self.store.now()
            self._result = None
            return

        if normalized.startswith("select 1 as ok from chat_sessions where chat_id = %s limit 1"):
            chat_id = params[0]
            self._result = [{"ok": 1}] if chat_id in self.store.sessions else []
            return

        if normalized.startswith("select coalesce(max(seq), 0) + 1 as next_seq from chat_messages"):
            chat_id = params[0]
            next_seq = 1 + max(
                (message["seq"] for message in self.store.messages if message["chat_id"] == chat_id),
                default=0,
            )
            self._result = [{"next_seq": next_seq}]
            return

        if normalized.startswith("insert into chat_messages"):
            chat_id, seq, role, content, _metadata = params
            metadata = getattr(_metadata, "obj", _metadata) or {}
            row = {
                "chat_id": chat_id,
                "seq": int(seq),
                "role": role,
                "content": content,
                "created_at": self.store.now(),
                "metadata": metadata,
            }
            self.store.messages.append(row)
            self._result = [row]
            return

        if normalized.startswith("insert into agent_run_traces"):
            if len(params) == 6:
                chat_id, client_message_id, intent, primary_agent, _events, _summary = params
                trace_status = "running"
            else:
                chat_id, client_message_id, intent, primary_agent, trace_status, _events, _summary = params
            self.store.trace_id_counter += 1
            row = {
                "id": self.store.trace_id_counter,
                "chat_id": chat_id,
                "assistant_message_seq": None,
                "client_message_id": client_message_id,
                "intent": intent,
                "primary_agent": primary_agent,
                "status": trace_status,
                "started_at": self.store.now(),
                "completed_at": None,
                "events": getattr(_events, "obj", _events) or [],
                "summary": getattr(_summary, "obj", _summary) or {},
                "created_at": self.store.now(),
            }
            self.store.traces.append(row)
            self._result = [row]
            return

        if normalized.startswith("update agent_run_traces"):
            assistant_message_seq, status, primary_agent, _events, _summary, trace_id = params
            row = next(
                (trace for trace in self.store.traces if trace["id"] == int(trace_id)),
                None,
            )
            if row is None:
                self._result = []
                return
            row.update(
                {
                    "assistant_message_seq": assistant_message_seq,
                    "status": status,
                    "primary_agent": primary_agent or row.get("primary_agent"),
                    "completed_at": self.store.now(),
                    "events": getattr(_events, "obj", _events) or [],
                    "summary": getattr(_summary, "obj", _summary) or {},
                }
            )
            self._result = [row]
            return

        if normalized.startswith("select id, chat_id, assistant_message_seq, client_message_id, intent, primary_agent, status, started_at, completed_at, events, summary, created_at from agent_run_traces where id = %s"):
            run_id = int(params[0])
            rows = [trace for trace in self.store.traces if trace["id"] == run_id]
            self._result = rows[:1]
            return

        if normalized.startswith("insert into agent_run_rounds"):
            run_id, round_number, round_type, status = params
            self.store.round_id_counter += 1
            row = {
                "id": self.store.round_id_counter,
                "run_id": int(run_id),
                "round_number": int(round_number),
                "round_type": round_type,
                "status": status,
                "created_at": self.store.now(),
                "updated_at": self.store.now(),
                "completed_at": None,
            }
            self.store.rounds.append(row)
            self._result = [row]
            return

        if normalized.startswith("update agent_run_rounds"):
            status, _status_for_completed, round_id = params
            row = next((item for item in self.store.rounds if item["id"] == int(round_id)), None)
            if row is None:
                self._result = []
                return
            row["status"] = status
            row["updated_at"] = self.store.now()
            if status in {"completed", "failed", "skipped"}:
                row["completed_at"] = self.store.now()
            self._result = [row]
            return

        if normalized.startswith("insert into agent_run_questions"):
            (
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                parent_question_id,
            ) = params
            self.store.question_id_counter += 1
            row = {
                "id": self.store.question_id_counter,
                "run_id": int(run_id),
                "round_id": int(round_id),
                "source_agent": source_agent,
                "target_agent": target_agent,
                "question": question,
                "reason": reason,
                "verification_goal": getattr(verification_goal, "obj", verification_goal) or {},
                "priority": priority,
                "status": status,
                "job_id": None,
                "queued_at": None,
                "attempt_count": 0,
                "last_error": None,
                "timeout_seconds": None,
                "parent_question_id": parent_question_id,
                "created_at": self.store.now(),
                "updated_at": self.store.now(),
                "started_at": None,
                "completed_at": None,
            }
            self.store.questions.append(row)
            self._result = [row]
            return

        if normalized.startswith("update agent_run_questions"):
            status, status_for_started, status_for_completed, question_id = params
            row = next((item for item in self.store.questions if item["id"] == int(question_id)), None)
            if row is None:
                self._result = []
                return
            row["status"] = status
            row["updated_at"] = self.store.now()
            if status_for_started == "running" and row.get("started_at") is None:
                row["started_at"] = self.store.now()
            if status_for_completed in {"completed", "failed", "skipped"}:
                row["completed_at"] = self.store.now()
            self._result = [row]
            return

        if normalized.startswith("insert into agent_run_responses"):
            (
                question_id,
                agent_name,
                status,
                summary,
                raw_output,
                has_business_evidence,
                runtime_failure,
            ) = params
            self.store.response_id_counter += 1
            row = {
                "id": self.store.response_id_counter,
                "question_id": int(question_id),
                "agent_name": agent_name,
                "status": status,
                "summary": summary,
                "raw_output": getattr(raw_output, "obj", raw_output) or {},
                "has_business_evidence": bool(has_business_evidence),
                "runtime_failure": getattr(runtime_failure, "obj", runtime_failure),
                "created_at": self.store.now(),
                "completed_at": self.store.now(),
            }
            self.store.responses.append(row)
            self._result = [row]
            return

        if normalized.startswith("insert into agent_run_gaps"):
            (
                run_id,
                response_id,
                sku,
                source_agent,
                target_agent,
                metric,
                reason,
                impact,
                blocking,
                status,
                evidence_status,
            ) = params
            self.store.gap_id_counter += 1
            row = {
                "id": self.store.gap_id_counter,
                "run_id": int(run_id),
                "response_id": int(response_id) if response_id is not None else None,
                "sku": sku,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "metric": metric,
                "reason": reason,
                "impact": impact,
                "blocking": bool(blocking),
                "status": status,
                "evidence_status": evidence_status,
                "created_at": self.store.now(),
                "updated_at": self.store.now(),
            }
            self.store.gaps.append(row)
            self._result = [row]
            return

        if normalized.startswith("update agent_run_gaps"):
            status, gap_id = params
            row = next((item for item in self.store.gaps if item["id"] == int(gap_id)), None)
            if row is None:
                self._result = []
                return
            row["status"] = status
            row["updated_at"] = self.store.now()
            self._result = [row]
            return

        if normalized.startswith("delete from agent_action_tasks"):
            chat_id, assistant_message_seq = params
            self.store.action_tasks = [
                task
                for task in self.store.action_tasks
                if not (
                    task["chat_id"] == chat_id
                    and task["assistant_message_seq"] == int(assistant_message_seq)
                )
            ]
            self._result = None
            return

        if normalized.startswith("insert into agent_action_tasks"):
            (
                chat_id,
                task_id,
                action_id,
                assistant_message_seq,
                source,
                factor,
                observation,
                impact,
                cause,
                action,
                title,
                action_type,
                priority,
                targets,
                success_criteria,
                agent_prompt,
                action_kind,
                business_goal,
                decision_type,
                required_metrics,
                forecast_spec,
                execution_plan,
                evidence,
                confidence,
                driver_class,
                intent,
                raw_action,
            ) = params
            row = {
                "chat_id": chat_id,
                "task_id": task_id,
                "action_id": action_id,
                "assistant_message_seq": int(assistant_message_seq),
                "source": source,
                "factor": factor,
                "observation": observation,
                "impact": impact,
                "cause": cause,
                "action": action,
                "title": title,
                "action_type": action_type,
                "priority": priority,
                "targets": getattr(targets, "obj", targets) or [],
                "success_criteria": success_criteria,
                "agent_prompt": agent_prompt,
                "action_kind": action_kind,
                "business_goal": business_goal,
                "decision_type": decision_type,
                "required_metrics": getattr(required_metrics, "obj", required_metrics) or [],
                "forecast_spec": getattr(forecast_spec, "obj", forecast_spec) or {},
                "execution_plan": execution_plan,
                "evidence": getattr(evidence, "obj", evidence) or [],
                "confidence": confidence,
                "driver_class": driver_class,
                "intent": intent,
                "raw_action": getattr(raw_action, "obj", raw_action) or {},
                "created_at": self.store.now(),
            }
            existing_index = next(
                (
                    index
                    for index, task in enumerate(self.store.action_tasks)
                    if task["chat_id"] == chat_id and task["task_id"] == task_id
                ),
                None,
            )
            if existing_index is None:
                self.store.action_tasks.append(row)
            else:
                row["created_at"] = self.store.action_tasks[existing_index]["created_at"]
                self.store.action_tasks[existing_index] = row
            self._result = [row]
            return

        if normalized.startswith("insert into agent_action_executions"):
            (
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
            ) = params
            self.store.action_execution_id_counter += 1
            row = {
                "id": self.store.action_execution_id_counter,
                "source_chat_id": source_chat_id,
                "source_task_id": source_task_id,
                "task_id": task_id,
                "object_id": object_id,
                "object_label": object_label,
                "object_type": object_type,
                "object_external_id": object_external_id,
                "object_payload": getattr(object_payload, "obj", object_payload) or {},
                "execution_chat_id": execution_chat_id,
                "run_id": int(run_id) if run_id is not None else None,
                "client_message_id": client_message_id,
                "status": status,
                "execution_mode": execution_mode,
                "prompt": prompt,
                "task_snapshot": getattr(task_snapshot, "obj", task_snapshot) or {},
                "result_json": getattr(result_json, "obj", result_json) or {},
                "decision_status": decision_status,
                "review_summary": review_summary,
                "answer": None,
                "server_message_id": None,
                "error": None,
                "created_at": self.store.now(),
                "updated_at": self.store.now(),
                "completed_at": None,
            }
            self.store.action_executions.append(row)
            self._result = [row]
            return

        if normalized.startswith("update agent_action_executions"):
            (
                status,
                run_id,
                answer,
                server_message_id,
                error,
                result_json,
                decision_status,
                review_summary,
                _status_for_completed,
                execution_id,
            ) = params
            row = next(
                (
                    execution
                    for execution in self.store.action_executions
                    if execution["id"] == int(execution_id)
                ),
                None,
            )
            if row is None:
                self._result = []
                return
            if status is not None:
                row["status"] = status
            if run_id is not None:
                row["run_id"] = int(run_id)
            if answer is not None:
                row["answer"] = answer
            if server_message_id is not None:
                row["server_message_id"] = int(server_message_id)
            if error is not None:
                row["error"] = error
            if result_json is not None:
                row["result_json"] = getattr(result_json, "obj", result_json) or {}
            if decision_status is not None:
                row["decision_status"] = decision_status
            if review_summary is not None:
                row["review_summary"] = review_summary
            row["updated_at"] = self.store.now()
            if row["status"] in {"completed", "failed", "cancelled", "done"} and row["completed_at"] is None:
                row["completed_at"] = self.store.now()
            self._result = [row]
            return

        if normalized.startswith("select seq, role, content, created_at, metadata from chat_messages where chat_id = %s order by seq asc"):
            chat_id = params[0]
            rows = [message for message in self.store.messages if message["chat_id"] == chat_id]
            self._result = list(sorted(rows, key=lambda item: item["seq"]))
            return

        if normalized.startswith("select chat_id, task_id, action_id, assistant_message_seq, source, factor, observation, impact, cause, action, title, action_type, priority, targets, success_criteria, agent_prompt, action_kind, business_goal, decision_type, required_metrics, forecast_spec, execution_plan, evidence, confidence, driver_class, intent, raw_action, created_at from agent_action_tasks"):
            chat_id = params[0]
            rows = [
                task
                for task in self.store.action_tasks
                if task["chat_id"] == chat_id
            ]
            if "and assistant_message_seq = %s" in normalized:
                assistant_message_seq = int(params[1])
                rows = [
                    task
                    for task in rows
                    if task["assistant_message_seq"] == assistant_message_seq
                ]
            self._result = list(
                sorted(
                    rows,
                    key=lambda item: (
                        item["assistant_message_seq"],
                        item["created_at"],
                        item["task_id"],
                    ),
                )
            )
            return

        if "from agent_action_executions" in normalized:
            rows = list(self.store.action_executions)
            if "where id = %s" in normalized:
                execution_id = int(params[0])
                rows = [row for row in rows if row["id"] == execution_id]
            elif "where source_chat_id = %s and source_task_id = %s and object_id = %s" in normalized:
                source_chat_id, source_task_id, object_id = params
                rows = [
                    row
                    for row in rows
                    if row["source_chat_id"] == source_chat_id
                    and row["source_task_id"] == source_task_id
                    and row["object_id"] == object_id
                ]
            elif "where source_chat_id = %s and task_id = %s" in normalized:
                source_chat_id, task_id = params
                rows = [
                    row
                    for row in rows
                    if row["source_chat_id"] == source_chat_id and row["task_id"] == task_id
                ]
            elif "where source_chat_id = %s" in normalized:
                source_chat_id = params[0]
                rows = [row for row in rows if row["source_chat_id"] == source_chat_id]
            self._result = list(sorted(rows, key=lambda item: (item["created_at"], item["id"]), reverse=True))
            return

        if normalized.startswith("select id, run_id, round_number, round_type, status, created_at, updated_at, completed_at from agent_run_rounds"):
            run_id = int(params[0])
            rows = [row for row in self.store.rounds if row["run_id"] == run_id]
            self._result = list(sorted(rows, key=lambda item: (item["round_number"], item["id"])))
            return

        if normalized.startswith("select id, run_id, round_id, source_agent, target_agent, question, reason, verification_goal, priority, status") and "from agent_run_questions" in normalized:
            run_id = int(params[0])
            rows = [row for row in self.store.questions if row["run_id"] == run_id]
            self._result = list(
                sorted(
                    rows,
                    key=lambda item: (
                        item["round_id"],
                        -(item["priority"] or -1),
                        item["id"],
                    ),
                )
            )
            return

        if normalized.startswith("select id, question_id, agent_name, status, summary, raw_output, has_business_evidence, runtime_failure, created_at, completed_at from agent_run_responses"):
            question_ids = set(int(item) for item in params[0])
            rows = [
                row
                for row in self.store.responses
                if row["question_id"] in question_ids
            ]
            self._result = list(sorted(rows, key=lambda item: item["id"]))
            return

        if normalized.startswith("select id, run_id, response_id, sku, source_agent, target_agent, metric, reason, impact, blocking, status, evidence_status, created_at, updated_at from agent_run_gaps"):
            run_id = int(params[0])
            rows = [row for row in self.store.gaps if row["run_id"] == run_id]
            if "status = %s" in normalized:
                status = params[1]
                rows = [row for row in rows if row["status"] == status]
            if "blocking = %s" in normalized:
                blocking_param_index = 2 if "status = %s" in normalized else 1
                blocking = bool(params[blocking_param_index])
                rows = [row for row in rows if row["blocking"] == blocking]
            if "sku = %s" in normalized:
                sku_value = params[-1]
                if isinstance(sku_value, list):
                    sku_value = params[-2]
                rows = [row for row in rows if row["sku"] == sku_value]
            if "evidence_status = any(%s)" in normalized:
                evidence_statuses = set(params[-1])
                rows = [row for row in rows if row["evidence_status"] in evidence_statuses]
            self._result = list(
                sorted(
                    rows,
                    key=lambda item: (
                        not item["blocking"],
                        item["created_at"],
                        item["id"],
                    ),
                )
            )
            return

        if normalized.startswith("select q.round_id, r.round_number, r.round_type, q.target_agent, q.question, q.reason, q.verification_goal, resp.id"):
            run_id = int(params[0])
            rows = []
            for response in self.store.responses:
                question = next(
                    (
                        item
                        for item in self.store.questions
                        if item["id"] == response["question_id"] and item["run_id"] == run_id
                    ),
                    None,
                )
                if question is None:
                    continue
                round_row = next(
                    item for item in self.store.rounds if item["id"] == question["round_id"]
                )
                rows.append(
                    {
                        **response,
                        "round_id": question["round_id"],
                        "round_number": round_row["round_number"],
                        "round_type": round_row["round_type"],
                        "target_agent": question["target_agent"],
                        "question": question["question"],
                        "reason": question["reason"],
                        "verification_goal": question["verification_goal"],
                    }
                )
            self._result = list(
                sorted(
                    rows,
                    key=lambda item: (
                        item["round_number"],
                        item["question_id"],
                        item["id"],
                    ),
                )
            )
            return

        if normalized.startswith("select id, chat_id, assistant_message_seq, client_message_id, intent, primary_agent, status, started_at, completed_at, events, summary, created_at from agent_run_traces"):
            chat_id = params[0]
            rows = [trace for trace in self.store.traces if trace["chat_id"] == chat_id]
            if "assistant_message_seq = %s" in normalized:
                assistant_message_seq = int(params[1])
                rows = [
                    trace
                    for trace in rows
                    if trace["assistant_message_seq"] == assistant_message_seq
                ]
                self._result = list(sorted(rows, key=lambda item: item["created_at"], reverse=True)[:1])
                return
            limit = int(params[1])
            self._result = list(sorted(rows, key=lambda item: item["created_at"], reverse=True)[:limit])
            return

        if normalized.startswith("select s.chat_id, s.created_at, s.updated_at, s.model_name, s.status"):
            limit = int(params[0])
            rows = []
            for session in self.store.sessions.values():
                chat_id = session["chat_id"]
                messages = [
                    message
                    for message in self.store.messages
                    if message["chat_id"] == chat_id
                ]
                traces = [
                    trace
                    for trace in self.store.traces
                    if trace["chat_id"] == chat_id
                ]
                user_messages = [message for message in messages if message["role"] == "user"]
                assistant_messages = [
                    message for message in messages if message["role"] == "assistant"
                ]
                latest_run = next(
                    iter(
                        sorted(
                            traces,
                            key=lambda item: (item["created_at"], item["id"]),
                            reverse=True,
                        )
                    ),
                    None,
                )
                last_user = next(
                    iter(sorted(user_messages, key=lambda item: item["seq"], reverse=True)),
                    None,
                )
                last_assistant = next(
                    iter(sorted(assistant_messages, key=lambda item: item["seq"], reverse=True)),
                    None,
                )
                rows.append(
                    {
                        "chat_id": chat_id,
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "model_name": session["model_name"],
                        "status": session["status"],
                        "message_count": len(messages),
                        "run_count": len(traces),
                        "latest_run_id": latest_run["id"] if latest_run else None,
                        "latest_run_status": latest_run["status"] if latest_run else None,
                        "latest_run_intent": latest_run["intent"] if latest_run else None,
                        "latest_run_created_at": latest_run["created_at"] if latest_run else None,
                        "latest_server_message_id": (
                            latest_run["assistant_message_seq"] if latest_run else None
                        ),
                        "last_user_message": last_user["content"] if last_user else None,
                        "last_user_message_at": last_user["created_at"] if last_user else None,
                        "last_assistant_message": (
                            last_assistant["content"] if last_assistant else None
                        ),
                        "last_assistant_message_at": (
                            last_assistant["created_at"] if last_assistant else None
                        ),
                    }
                )
            self._result = list(
                sorted(
                    rows,
                    key=lambda item: (item["updated_at"], item["created_at"]),
                    reverse=True,
                )[:limit]
            )
            return

        if "select seq, role, content, created_at, metadata from chat_messages" in normalized:
            chat_id, role, metadata_key, metadata_value = params
            rows = [
                message
                for message in self.store.messages
                if message["chat_id"] == chat_id
                and message["role"] == role
                and message.get("metadata", {}).get(metadata_key) == metadata_value
            ]
            self._result = list(sorted(rows, key=lambda item: item["seq"], reverse=True)[:1])
            return

        if "from ( select seq, role, content, created_at from chat_messages" in normalized:
            chat_id, limit = params
            rows = [
                message
                for message in self.store.messages
                if message["chat_id"] == chat_id
            ]
            rows = sorted(rows, key=lambda item: item["seq"], reverse=True)[: int(limit)]
            self._result = list(sorted(rows, key=lambda item: item["seq"]))
            return

        if normalized.startswith("select seq, role, content, created_at from chat_messages where chat_id = %s and seq > %s"):
            chat_id, after_seq = params
            rows = [
                message
                for message in self.store.messages
                if message["chat_id"] == chat_id and message["seq"] > int(after_seq)
            ]
            self._result = list(sorted(rows, key=lambda item: item["seq"]))
            return

        if normalized.startswith("select summary_text, summarized_through_seq, updated_at from chat_summaries"):
            chat_id = params[0]
            row = self.store.summaries.get(chat_id)
            self._result = [row] if row else []
            return

        if normalized.startswith("insert into chat_summaries"):
            chat_id, summary_text, summarized_through_seq = params
            row = {
                "summary_text": summary_text,
                "summarized_through_seq": int(summarized_through_seq),
                "updated_at": self.store.now(),
            }
            self.store.summaries[chat_id] = row
            self._result = [row]
            return

        raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        if not self._result:
            return None
        return self._result[0]

    def fetchall(self):
        return list(self._result or [])


class FakeMemoryConnection:
    def __init__(self, store: FakeMemoryStore):
        self.store = store
        self.cursor_obj = FakeMemoryCursor(store)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeMemoryStore()
        patcher = patch(
            "agi_ecom_app.services.memory.get_memory_connection",
            side_effect=lambda: FakeMemoryConnection(self.store),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repository_creates_session_appends_messages_and_reads_window(self):
        memory.ensure_chat_session("chat-1", model_name="gpt-test")
        memory.append_message("chat-1", ChatRole.USER, "Привет")
        memory.append_message("chat-1", ChatRole.ASSISTANT, "Здравствуйте")
        memory.append_message("chat-1", ChatRole.USER, "Продолжим")

        window = memory.get_context_window("chat-1", limit=2)

        self.assertEqual(window.chat_id, "chat-1")
        self.assertIsNone(window.summary_text)
        self.assertEqual([message.seq for message in window.recent_messages], [2, 3])
        self.assertEqual(
            [message.role for message in window.recent_messages],
            [ChatRole.ASSISTANT, ChatRole.USER],
        )

    def test_save_and_load_summary_round_trip(self):
        memory.ensure_chat_session("chat-2", model_name="gpt-test")

        saved = memory.save_summary(
            chat_id="chat-2",
            summary_text="Пользователь хочет краткие отчеты.",
            summarized_through_seq=4,
        )
        loaded = memory.load_summary("chat-2")

        self.assertEqual(saved.summary_text, "Пользователь хочет краткие отчеты.")
        self.assertIsNotNone(saved.updated_at)
        self.assertEqual(loaded.summary_text, "Пользователь хочет краткие отчеты.")
        self.assertEqual(loaded.summarized_through_seq, 4)

    @patch("agi_ecom_app.services.memory.generate_summary_text")
    @patch("agi_ecom_app.services.memory.get_settings")
    def test_refresh_chat_summary_updates_only_unsummarized_messages(
        self,
        mock_get_settings,
        mock_generate_summary_text,
    ):
        mock_get_settings.return_value = type(
            "Settings",
            (),
            {"openai_api_key": "test-key", "openai_memory_model": "gpt-memory"},
        )()
        mock_generate_summary_text.return_value = "Обновленное summary."

        memory.ensure_chat_session("chat-3", model_name="gpt-test")
        memory.append_message("chat-3", ChatRole.USER, "Первый вопрос")
        memory.append_message("chat-3", ChatRole.ASSISTANT, "Первый ответ")
        memory.save_summary("chat-3", "Старое summary.", summarized_through_seq=2)
        memory.append_message("chat-3", ChatRole.USER, "Новый вопрос")
        memory.append_message("chat-3", ChatRole.ASSISTANT, "Новый ответ")

        refreshed = memory.refresh_chat_summary("chat-3")

        self.assertEqual(refreshed.summary_text, "Обновленное summary.")
        self.assertEqual(refreshed.summarized_through_seq, 4)
        pending_messages = mock_generate_summary_text.call_args.kwargs["pending_messages"]
        self.assertEqual([message.seq for message in pending_messages], [3, 4])
        self.assertEqual(mock_generate_summary_text.call_args.kwargs["existing_summary"], "Старое summary.")

    def test_append_user_message_idempotent_reuses_existing_message_by_client_id(self):
        memory.ensure_chat_session("chat-4", model_name="gpt-test")

        first, first_created = memory.append_user_message_idempotent(
            chat_id="chat-4",
            content="Найди остатки",
            metadata={"intent": "factual_qa"},
            client_message_id="client-1",
        )
        second, second_created = memory.append_user_message_idempotent(
            chat_id="chat-4",
            content="Найди остатки",
            metadata={"intent": "factual_qa"},
            client_message_id="client-1",
        )

        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first.seq, second.seq)
        self.assertEqual(len(self.store.messages), 1)

    def test_list_all_stored_messages_returns_metadata_for_history_restore(self):
        memory.ensure_chat_session("chat-5", model_name="gpt-test")
        memory.append_user_message_idempotent(
            chat_id="chat-5",
            content="Собери отчет",
            metadata={"intent": "report"},
            client_message_id="client-restore-1",
        )
        memory.append_message(
            "chat-5",
            ChatRole.ASSISTANT,
            "Готово",
            metadata={"intent": "report", "reply_to_client_message_id": "client-restore-1"},
        )

        messages = memory.list_all_stored_messages("chat-5")

        self.assertEqual([message.seq for message in messages], [1, 2])
        self.assertEqual(messages[0].metadata["client_message_id"], "client-restore-1")
        self.assertEqual(messages[1].metadata["reply_to_client_message_id"], "client-restore-1")

    def test_list_chat_session_summaries_returns_latest_dialogs_for_ui(self):
        memory.ensure_chat_session("chat-old", model_name="gpt-test")
        memory.append_message("chat-old", ChatRole.USER, "Старый вопрос")
        memory.append_message("chat-old", ChatRole.ASSISTANT, "Старый ответ")

        memory.ensure_chat_session("chat-new", model_name="gpt-test")
        memory.append_user_message_idempotent(
            chat_id="chat-new",
            content="Посчитай оборачиваемость склада",
            metadata={"intent": "supply"},
            client_message_id="client-ui-1",
        )
        memory.append_message(
            "chat-new",
            ChatRole.ASSISTANT,
            "Недостаточно квоты OpenAI API.",
            metadata={"intent": "supply", "reply_to_client_message_id": "client-ui-1"},
        )
        run = memory.create_agent_run_trace(
            chat_id="chat-new",
            intent="supply",
            client_message_id="client-ui-1",
            status="failed",
        )
        memory.complete_agent_run_trace(
            trace_id=run.id,
            assistant_message_seq=2,
            status="failed",
            events=[],
            summary={"error": "insufficient_quota"},
        )

        sessions = memory.list_chat_session_summaries(limit=10)

        self.assertEqual([session.chat_id for session in sessions], ["chat-new", "chat-old"])
        self.assertEqual(sessions[0].message_count, 2)
        self.assertEqual(sessions[0].run_count, 1)
        self.assertEqual(sessions[0].latest_run_id, run.id)
        self.assertEqual(sessions[0].latest_run_status, "failed")
        self.assertEqual(sessions[0].latest_server_message_id, 2)
        self.assertEqual(sessions[0].last_user_message, "Посчитай оборачиваемость склада")
        self.assertEqual(sessions[0].last_assistant_message, "Недостаточно квоты OpenAI API.")

    def test_agent_run_trace_round_trip(self):
        memory.ensure_chat_session("chat-trace-1", model_name="gpt-test")

        created = memory.create_agent_run_trace(
            chat_id="chat-trace-1",
            intent="sales",
            client_message_id="client-trace-1",
            primary_agent="sales_agent",
            events=[{"event": "run_started", "status": "started"}],
            summary={"status": "running"},
        )
        completed = memory.complete_agent_run_trace(
            trace_id=created.id,
            assistant_message_seq=2,
            status="completed",
            primary_agent="sales_agent",
            events=[
                {"event": "run_started", "status": "started"},
                {"event": "run_completed", "status": "completed"},
            ],
            summary={
                "primary_agent": "sales_agent",
                "secondary_agents": ["supply_agent"],
                "runtime_failure_count": 0,
            },
        )
        loaded = memory.get_agent_run_trace("chat-trace-1", assistant_message_seq=2)

        self.assertEqual(created.status, "running")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.assistant_message_seq, 2)
        self.assertIsNotNone(completed.completed_at)
        self.assertEqual(loaded.id, created.id)
        self.assertEqual(loaded.summary["primary_agent"], "sales_agent")
        self.assertEqual([event["event"] for event in loaded.events], ["run_started", "run_completed"])

    def test_list_agent_run_traces_returns_latest_first(self):
        memory.ensure_chat_session("chat-trace-2", model_name="gpt-test")

        first = memory.create_agent_run_trace(chat_id="chat-trace-2", intent="sales")
        second = memory.create_agent_run_trace(chat_id="chat-trace-2", intent="finance")
        memory.complete_agent_run_trace(
            trace_id=first.id,
            assistant_message_seq=2,
            status="completed",
            events=[],
            summary={"primary_agent": "sales_agent"},
        )
        memory.complete_agent_run_trace(
            trace_id=second.id,
            assistant_message_seq=4,
            status="failed",
            events=[],
            summary={"primary_agent": "finance_agent"},
        )

        traces = memory.list_agent_run_traces("chat-trace-2", limit=10)

        self.assertEqual([trace.id for trace in traces], [second.id, first.id])
        self.assertEqual([trace.assistant_message_seq for trace in traces], [4, 2])
        self.assertEqual(traces[0].status, "failed")

    def test_agent_run_trace_missing_message_returns_none(self):
        memory.ensure_chat_session("chat-trace-3", model_name="gpt-test")
        memory.create_agent_run_trace(chat_id="chat-trace-3", intent="sales")

        self.assertIsNone(
            memory.get_agent_run_trace("chat-trace-3", assistant_message_seq=999)
        )

    def test_agent_run_round_question_response_gap_round_trip(self):
        memory.ensure_chat_session("chat-rounds-1", model_name="gpt-test")
        run = memory.create_agent_run_trace(chat_id="chat-rounds-1", intent="finance")

        round_record = memory.create_agent_run_round(
            run_id=run.id,
            round_number=1,
            round_type="cross_check",
        )
        question = memory.create_agent_run_question(
            run_id=run.id,
            round_id=round_record.id,
            source_agent="finance",
            target_agent="supply",
            question="Проверь остатки SKU 46300.",
            reason="Нужно закрыть риск OOS.",
            verification_goal={"sku": "46300"},
            priority=90,
        )
        memory.update_agent_run_question_status(question_id=question.id, status="running")
        response = memory.save_agent_run_response(
            question_id=question.id,
            agent_name="supply",
            status="completed",
            summary="Остаток ограничен.",
            raw_output={"summary": "Остаток ограничен.", "data_gaps": []},
            has_business_evidence=True,
        )
        gaps = memory.save_agent_run_gaps(
            run_id=run.id,
            response_id=response.id,
            gaps=[
                {
                    "sku": "46300",
                    "source_agent": "supply",
                    "target_agent": "supply",
                    "metric": "inbound supply",
                    "reason": "Поставка не подтверждена.",
                    "impact": "Нельзя усиливать рекламу.",
                    "blocking": True,
                    "evidence_status": "unconfirmed",
                }
            ],
        )
        memory.update_agent_run_round_status(round_id=round_record.id, status="completed")

        rounds = memory.list_agent_run_rounds(run.id)

        self.assertEqual(rounds[0]["round"].round_type, "cross_check")
        self.assertEqual(rounds[0]["questions"][0]["question"].target_agent, "supply")
        self.assertEqual(rounds[0]["questions"][0]["responses"][0].summary, "Остаток ограничен.")
        self.assertEqual(rounds[0]["questions"][0]["gaps"][0].id, gaps[0].id)

    def test_agent_run_gaps_filter_by_blocking_status_and_sku(self):
        memory.ensure_chat_session("chat-gaps-1", model_name="gpt-test")
        run = memory.create_agent_run_trace(chat_id="chat-gaps-1", intent="finance")
        memory.save_agent_run_gaps(
            run_id=run.id,
            response_id=None,
            gaps=[
                {
                    "sku": "46300",
                    "source_agent": "supply",
                    "target_agent": "supply",
                    "metric": "stock",
                    "reason": "Нет данных.",
                    "blocking": True,
                    "evidence_status": "no_data",
                },
                {
                    "sku": "46227",
                    "source_agent": "marketing",
                    "target_agent": "marketing",
                    "metric": "bid",
                    "reason": "Ставки отсутствуют.",
                    "blocking": False,
                    "evidence_status": "unconfirmed",
                },
            ],
        )

        gaps = memory.list_agent_run_gaps(
            run_id=run.id,
            status="open",
            blocking=True,
            sku="46300",
            evidence_statuses=("no_data",),
        )

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].sku, "46300")
        self.assertTrue(gaps[0].blocking)

    def test_replace_and_list_agent_action_tasks_round_trip(self):
        memory.ensure_chat_session("chat-actions-1", model_name="gpt-test")

        saved = memory.replace_agent_action_tasks(
            chat_id="chat-actions-1",
            assistant_message_seq=2,
            tasks=[
                {
                    "task_id": "action_chat-actions-1_2_1",
                    "action_id": "action_chat-actions-1_2_1",
                    "source": "final_table_action",
                    "factor": "Что делать",
                    "observation": "полка пустая",
                    "impact": "Задача подготовлена после диагностики.",
                    "cause": "полка пустая",
                    "action": "Проверить доступный остаток",
                    "title": "Проверить доступный остаток",
                    "action_type": "check",
                    "priority": "high",
                    "targets": ["46230"],
                    "success_criteria": "Статус получен.",
                    "agent_prompt": "Проверить доступный остаток.",
                    "evidence": ["полка пустая"],
                    "confidence": "high",
                    "driver_class": "primary_driver",
                    "intent": "sales",
                    "raw_action": {"action": "Проверить доступный остаток"},
                }
            ],
        )
        loaded = memory.list_stored_agent_action_tasks("chat-actions-1")

        self.assertEqual(len(saved), 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].task_id, "action_chat-actions-1_2_1")
        self.assertEqual(loaded[0].assistant_message_seq, 2)
        self.assertEqual(loaded[0].targets, ["46230"])
        self.assertEqual(loaded[0].raw_action["action"], "Проверить доступный остаток")

    def test_replace_agent_action_tasks_clears_old_rows_for_message(self):
        memory.ensure_chat_session("chat-actions-2", model_name="gpt-test")
        memory.replace_agent_action_tasks(
            chat_id="chat-actions-2",
            assistant_message_seq=2,
            tasks=[
                {
                    "task_id": "action_chat-actions-2_2_1",
                    "factor": "Что делать",
                    "action": "Проверить остаток",
                }
            ],
        )
        memory.replace_agent_action_tasks(
            chat_id="chat-actions-2",
            assistant_message_seq=2,
            tasks=[],
        )

        self.assertEqual(memory.list_stored_agent_action_tasks("chat-actions-2"), [])

    def test_agent_action_execution_round_trip_and_status_update(self):
        memory.ensure_chat_session("chat-actions-3", model_name="gpt-test")
        memory.ensure_chat_session("chat-actions-exec-3", model_name="gpt-test")

        created = memory.create_agent_action_execution(
            source_chat_id="chat-actions-3",
            source_task_id="action_chat-actions-3_2_1",
            task_id="action_chat-actions-3_2_1",
            object_id="46230",
            object_label="SKU 46230",
            object_type="sku",
            object_external_id="46230",
            object_payload={"nm_id": 46230},
            execution_chat_id="chat-actions-exec-3",
            client_message_id="client-exec-1",
            prompt="Проверить остаток.",
            task_snapshot={"taskId": "action_chat-actions-3_2_1", "action": "Проверить остаток"},
            review_summary="Нужен дозаказ",
        )
        updated = memory.update_agent_action_execution(
            execution_id=created.id,
            status="completed",
            run_id=42,
            answer="Готово",
            server_message_id=3,
            result_json={"decisionStatus": "completed", "summary": "Нужен дозаказ"},
            decision_status="completed",
            review_summary="Нужен дозаказ",
        )
        loaded = memory.get_agent_action_execution(created.id)
        listed = memory.list_agent_action_executions("chat-actions-3")

        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.run_id, 42)
        self.assertEqual(updated.answer, "Готово")
        self.assertEqual(updated.review_summary, "Нужен дозаказ")
        self.assertEqual(updated.result_json["decisionStatus"], "completed")
        self.assertEqual(updated.source_task_id, "action_chat-actions-3_2_1")
        self.assertEqual(updated.object_id, "46230")
        self.assertEqual(updated.object_payload["nm_id"], 46230)
        self.assertIsNotNone(updated.completed_at)
        self.assertEqual(loaded.task_snapshot["action"], "Проверить остаток")
        self.assertEqual(listed[0].id, created.id)


if __name__ == "__main__":
    unittest.main()
