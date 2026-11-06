from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agi_ecom_app.document_import.contract_airflow import trigger_contract_airflow_worker
from agi_ecom_app.document_import.document_agent import ContractReference, run_feature2_document_agent
from agi_ecom_app.document_import.models import (
    ContractAirflowWebhookResponse,
    Feature2AgentResponse,
    ProductCoastImportResponse,
)
from agi_ecom_app.document_import.service import import_product_coast_file

router = APIRouter()


@router.post("/v1/finance/product-coast/import", response_model=ProductCoastImportResponse)
async def import_product_coast(file: UploadFile = File(...)) -> ProductCoastImportResponse:
    try:
        content = await file.read()
        return import_product_coast_file(
            content=content,
            filename=file.filename or "upload.xlsx",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    finally:
        await file.close()


@router.post("/v1/feature-2/contracts/airflow-webhook", response_model=ContractAirflowWebhookResponse)
async def trigger_contract_airflow_webhook(
    slot: int = Form(...),
    file: UploadFile = File(...),
) -> ContractAirflowWebhookResponse:
    try:
        content = await file.read()
        return trigger_contract_airflow_worker(
            slot=slot,
            filename=file.filename or "contract.docx",
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    finally:
        await file.close()


@router.post("/v1/feature-2/agent", response_model=Feature2AgentResponse)
async def run_document_agent(
    message: str = Form(""),
    document_1_name: str | None = Form(None),
    document_2_name: str | None = Form(None),
    document_1: UploadFile | None = File(None),
    document_2: UploadFile | None = File(None),
) -> Feature2AgentResponse:
    refs: list[ContractReference] = []
    for slot, name, file in ((1, document_1_name, document_1), (2, document_2_name, document_2)):
        filename = name or (file.filename if file else None)
        if not filename:
            continue
        refs.append(ContractReference(slot=slot, filename=filename))
        if file is not None:
            await file.close()

    try:
        return run_feature2_document_agent(
            message=message,
            document_refs=refs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
