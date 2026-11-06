from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from agi_ecom_app.config import get_settings
from agi_ecom_app.document_import.models import ContractAirflowWebhookResponse

MAX_CONTRACT_UPLOAD_BYTES = 25 * 1024 * 1024


def trigger_contract_airflow_worker(*, slot: int, filename: str, content: bytes) -> ContractAirflowWebhookResponse:
    if slot not in {1, 2}:
        raise ValueError("slot must be 1 or 2")
    source_filename = _validate_contract_upload(filename=filename, content=content)
    settings = get_settings()
    dag_id = settings.airflow_contract_dag_id
    dag_run_id = _build_dag_run_id(source_filename)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    content_sha256 = hashlib.sha256(content).hexdigest()

    conf = {
        "slot": slot,
        "filename": source_filename,
        "document_id": Path(source_filename).stem,
        "size_bytes": len(content),
        "content_sha256": content_sha256,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "uploaded_at": uploaded_at,
    }
    payload = {"dag_run_id": dag_run_id, "conf": conf}

    url = f"{settings.airflow_base_url.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns"
    auth = None
    if settings.airflow_username and settings.airflow_password:
        auth = (settings.airflow_username, settings.airflow_password)

    try:
        response = httpx.post(url, json=payload, auth=auth, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _airflow_error_detail(exc.response)
        raise RuntimeError(f"Airflow rejected contract worker request: {detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Airflow is unavailable: {exc}") from exc

    airflow_payload = response.json()
    return ContractAirflowWebhookResponse(
        slot=slot,
        filename=source_filename,
        sizeBytes=len(content),
        dagId=dag_id,
        dagRunId=airflow_payload.get("dag_run_id") or dag_run_id,
        documentId=Path(source_filename).stem,
        contentSha256=content_sha256,
    )


def _validate_contract_upload(*, filename: str, content: bytes) -> str:
    source_filename = Path(filename or "").name.strip()
    if not source_filename:
        raise ValueError("file name must not be empty")
    if Path(source_filename).suffix.lower() != ".docx":
        raise ValueError("contract worker currently supports only .docx files")
    if not content:
        raise ValueError("uploaded contract file is empty")
    if len(content) > MAX_CONTRACT_UPLOAD_BYTES:
        raise ValueError("uploaded contract file is too large")
    return source_filename


def _build_dag_run_id(filename: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(filename).stem).strip("._-") or "contract"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"contract_ingest__{stem[:80]}__{timestamp}__{uuid4().hex[:8]}"


def _airflow_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"{response.status_code} {response.text[:300]}"

    detail = payload.get("detail") or payload.get("title") or payload
    return f"{response.status_code} {detail}"
