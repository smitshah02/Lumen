"""Request/response models for the Lumen HTTP API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# subject_id is INTEGER in every Lumen table.
SubjectId = Field(..., ge=1, le=2_147_483_647, description="Patient subject_id")


class _QueryModel(BaseModel):
    model_config = {"extra": "forbid"}
    subject_id: int = SubjectId
    query: str = Field(..., min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class RetrieveRequest(_QueryModel):
    top_k: int = Field(5, ge=1, le=20)
    temporal_filter: Literal["auto", "all", "latest", "trend", "recent"] = "auto"


class AskRequest(_QueryModel):
    pass


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: int
    note_id: int
    subject_id: int
    hadm_id: Optional[int]
    chunk_index: int
    note_type: Optional[str]
    charttime: Optional[str]
    score: float
    sources: list[str]
    text: str


class RetrieveResponse(BaseModel):
    request_id: str
    data_plane: str
    subject_id: int
    query: str
    temporal_mode: str
    results: list[RetrievedChunk]
    latency_ms: float


class Citation(BaseModel):
    label: str
    chunk_id: int
    claim: str
    verified: bool


class Source(BaseModel):
    label: str
    chunk_id: int
    source_type: str
    note_id: Optional[int] = None
    subject_id: Optional[int] = None
    hadm_id: Optional[int] = None
    chunk_index: Optional[int] = None
    note_type: Optional[str]
    charttime: Optional[str]


class AskResponse(BaseModel):
    request_id: str
    thread_id: str
    data_plane: str
    status: Literal["completed", "human_review_required", "refused", "failed"]
    review_status: Optional[str]
    answer: str
    answer_is_draft: bool
    citations: list[Citation]
    sources: list[Source]
    flagged_claims: int
    needs_human_review: bool
    query_type: Optional[str]
    temporal_mode: Optional[str]
    node_trail: list[str]
    models: dict
    latency_ms: float
    timings: dict


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None
    detail: Optional[object] = None
