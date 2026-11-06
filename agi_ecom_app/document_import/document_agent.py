from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import get_active_llm_runtime
from agi_ecom_app.observability import generation_scope, response_payload, update_observation
from agi_ecom_app.document_import.models import Feature2AgentDocument, Feature2AgentResponse

MAX_CONTEXT_CHUNKS = 80
MAX_CHUNK_CHARS = 1_800

FEATURE2_AGENT_INSTRUCTIONS = """
Ты backend-агент для анализа договоров из БД Далкос.
Критически важно: ты не видишь и не читаешь загруженные пользователем файлы.
Единственный источник текста договоров — read-only SELECT-контекст из PostgreSQL БД Далкос.
Отвечай на русском языке, кратко и по делу.
Если в контексте БД нет нужных данных, прямо скажи, что данных в БД недостаточно.
Не выдумывай факты, реквизиты, суммы, сроки или условия, которых нет в DB_CONTEXT.
""".strip()


@dataclass(frozen=True)
class ContractReference:
    slot: int
    filename: str


def run_feature2_document_agent(
    *,
    message: str,
    document_refs: list[ContractReference],
) -> Feature2AgentResponse:
    cleaned_message = message.strip()
    if not cleaned_message and not document_refs:
        raise ValueError("message or at least one contract reference is required")

    settings = get_settings()
    runtime = get_active_llm_runtime(settings=settings)

    db_context = get_dalkos_contract_context(document_refs)
    input_text = _build_agent_input(message=cleaned_message, db_context=db_context)

    with generation_scope(
        name="feature2-document-answer",
        model=runtime.model_name,
        input={
            "instructions": FEATURE2_AGENT_INSTRUCTIONS,
            "input": input_text,
        },
        metadata={
            "model_profile": getattr(runtime, "profile", None),
            "model_provider": getattr(runtime, "provider", None),
            "model_name": runtime.model_name,
            "document_count": len(document_refs),
        },
        settings=settings,
    ) as observation:
        try:
            response = runtime.sync_client.responses.create(
                model=runtime.model_name,
                instructions=FEATURE2_AGENT_INSTRUCTIONS,
                input=input_text,
                max_output_tokens=1200,
            )
        except Exception as exc:
            update_observation(observation, error=exc)
            raise
        update_observation(
            observation,
            output=response_payload(response),
            response=response,
        )

    answer = (response.output_text or "").strip() or "Агент не вернул текстовый ответ."
    return Feature2AgentResponse(
        answer=answer,
        model=runtime.model_name,
        documents=[
            Feature2AgentDocument(
                slot=document["slot"],
                filename=document["source_file_basename"] or document["document_id"],
                sizeBytes=0,
                documentId=document["document_id"],
                source="dalkos_db",
            )
            for document in db_context["documents"]
        ],
    )


def get_dalkos_contract_context(document_refs: list[ContractReference]) -> dict[str, Any]:
    with _connect_dalkos_readonly() as conn:
        with conn.cursor() as cur:
            _configure_readonly_session(cur)
            documents = _resolve_contract_documents(cur, document_refs)
            if not documents:
                raise ValueError("contracts were not found in Dalkos DB")

            document_ids = [document["document_id"] for document in documents]
            chunks = _fetch_readable_chunks(cur, document_ids)
            risks = _fetch_risk_chunks(cur, document_ids)
            tables = _fetch_table_chunks(cur, document_ids)

    return {
        "documents": documents,
        "chunks": chunks,
        "risks": risks,
        "tables": tables,
    }


def _connect_dalkos_readonly():
    settings = get_settings()
    db = settings.dalkos_db
    return psycopg.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password,
        sslmode=db.sslmode,
        row_factory=dict_row,
    )


def _configure_readonly_session(cur) -> None:
    cur.execute("SET default_transaction_read_only = on")
    cur.execute("SET transaction_read_only = on")
    cur.execute("SET statement_timeout = '5000ms'")


def _resolve_contract_documents(cur, refs: list[ContractReference]) -> list[dict[str, Any]]:
    if not refs:
        cur.execute(
            """
            SELECT document_id, version, source_file, title, revision_text, revision_date, chunks_count
            FROM public.contract_documents
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT 5
            """
        )
        return [_document_payload(row, slot=index + 1) for index, row in enumerate(cur.fetchall())]

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        filename = Path(ref.filename).name
        stem = Path(filename).stem
        cur.execute(
            """
            SELECT document_id, version, source_file, title, revision_text, revision_date, chunks_count
            FROM public.contract_documents
            WHERE document_id = %(stem)s
               OR source_file = %(filename)s
               OR source_file ILIKE %(filename_like)s
               OR document_id ILIKE %(stem_like)s
            ORDER BY
              CASE
                WHEN document_id = %(stem)s THEN 0
                WHEN source_file ILIKE %(filename_like)s THEN 1
                ELSE 2
              END,
              version DESC
            LIMIT 1
            """,
            {
                "stem": stem,
                "filename": filename,
                "filename_like": f"%{filename}",
                "stem_like": f"%{stem}%",
            },
        )
        row = cur.fetchone()
        if row and row["document_id"] not in seen:
            seen.add(row["document_id"])
            resolved.append(_document_payload(row, slot=ref.slot))
    return resolved


def _fetch_readable_chunks(cur, document_ids: list[str]) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT document_id, version, chunk_id, type, section, clause, source_order, content,
               risk_hint, extracted_values, extracted_financial_terms
        FROM public.contract_chunks_readable
        WHERE document_id = ANY(%(document_ids)s)
        ORDER BY document_id, version, source_order
        LIMIT %(limit)s
        """,
        {"document_ids": document_ids, "limit": MAX_CONTEXT_CHUNKS},
    )
    return [_chunk_payload(row) for row in cur.fetchall()]


def _fetch_risk_chunks(cur, document_ids: list[str]) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT document_id, version, chunk_id, section, clause, source_order,
               clause_text, risk_hint, extracted_financial_terms
        FROM public.contract_risk_chunks
        WHERE document_id = ANY(%(document_ids)s)
        ORDER BY document_id, version, source_order
        LIMIT 40
        """,
        {"document_ids": document_ids},
    )
    return [_risk_payload(row) for row in cur.fetchall()]


def _fetch_table_chunks(cur, document_ids: list[str]) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT document_id, version, chunk_id, section, source_order, rows, extracted_values
        FROM public.contract_table_chunks
        WHERE document_id = ANY(%(document_ids)s)
        ORDER BY document_id, version, source_order
        LIMIT 20
        """,
        {"document_ids": document_ids},
    )
    return [_table_payload(row) for row in cur.fetchall()]


def _document_payload(row: dict[str, Any], *, slot: int) -> dict[str, Any]:
    source_file = row.get("source_file") or ""
    return {
        "slot": slot,
        "document_id": row["document_id"],
        "version": row["version"],
        "source_file": source_file,
        "source_file_basename": Path(source_file).name if source_file else None,
        "title": row.get("title"),
        "revision_text": row.get("revision_text"),
        "revision_date": row.get("revision_date").isoformat() if row.get("revision_date") else None,
        "chunks_count": row.get("chunks_count"),
    }


def _chunk_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": row["document_id"],
        "version": row["version"],
        "chunk_id": row["chunk_id"],
        "type": row["type"],
        "section": row["section"],
        "clause": row["clause"],
        "source_order": row["source_order"],
        "content": _truncate(row.get("content") or ""),
        "risk_hint": row.get("risk_hint"),
        "extracted_values": row.get("extracted_values"),
        "extracted_financial_terms": row.get("extracted_financial_terms"),
    }


def _risk_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": row["document_id"],
        "version": row["version"],
        "chunk_id": row["chunk_id"],
        "section": row["section"],
        "clause": row["clause"],
        "source_order": row["source_order"],
        "clause_text": _truncate(row.get("clause_text") or ""),
        "risk_hint": row.get("risk_hint"),
        "extracted_financial_terms": row.get("extracted_financial_terms"),
    }


def _table_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": row["document_id"],
        "version": row["version"],
        "chunk_id": row["chunk_id"],
        "section": row["section"],
        "source_order": row["source_order"],
        "rows": row.get("rows"),
        "extracted_values": row.get("extracted_values"),
    }


def _truncate(value: str) -> str:
    return value if len(value) <= MAX_CHUNK_CHARS else f"{value[:MAX_CHUNK_CHARS]}..."


def _build_agent_input(*, message: str, db_context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Задача пользователя:",
            message or "Пользователь выбрал договоры без отдельного вопроса.",
            "",
            "DB_CONTEXT_FROM_DALKOS_READONLY:",
            json.dumps(db_context, ensure_ascii=False, default=str, indent=2),
        ]
    )
