from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ProductCoastImportResponse(BaseModel):
    status: str = "completed"
    uploadDate: date
    rowsParsed: int
    rowsSaved: int
    rowsSkipped: int
    sourceFilename: str


class Feature2AgentDocument(BaseModel):
    slot: int
    filename: str
    sizeBytes: int
    documentId: str | None = None
    source: str = "dalkos_db"


class Feature2AgentResponse(BaseModel):
    status: str = "completed"
    answer: str
    model: str
    documents: list[Feature2AgentDocument]


class ContractAirflowWebhookResponse(BaseModel):
    status: str = "queued"
    slot: int
    filename: str
    sizeBytes: int
    dagId: str
    dagRunId: str
    documentId: str
    contentSha256: str
