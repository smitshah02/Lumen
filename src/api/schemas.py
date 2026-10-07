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
    # Optional admission scope stated by the caller. It must be one of this
    # subject's admissions (checked in the handler) and wins over anything the
    # question text says.
    hadm_id: Optional[int] = Field(None, ge=1, le=2_147_483_647, description="Admission to scope the answer to")


class SourceProvenance(BaseModel):
    """Stable identity of a v2 chunk. `chunk_id` is a handle within one table
    only; compare sources across profiles or builds by `source_id`, or by the
    MIMIC note and offsets."""
    source_id: str
    data_profile: str
    table: str
    build_id: str
    mimic_note_id: str
    section_ord: int
    chunk_ord: int
    start_offset: int
    end_offset: int
    chunk_id: int


# Present for v2 sources; left out of the response entirely for every other profile.
_Provenance = Field(None, exclude_if=lambda v: v is None)


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
    provenance: Optional[SourceProvenance] = _Provenance


class RetrieveResponse(BaseModel):
    request_id: str
    data_plane: str
    data_profile: str
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
    provenance: Optional[SourceProvenance] = _Provenance


class AdmissionScope(BaseModel):
    """Whether the answer was scoped to one admission, and why or why not."""
    applied: bool                 # retrieval searched within one admission
    requested: bool               # an admission was stated on the request or named in the question
    status: Literal["resolved", "unresolved", "ambiguous", "none", "not_enabled"]
    hadm_id: Optional[int]        # the resolved admission, when there is one
    source: Optional[str]         # "request" or the rule that resolved it
    reason: Optional[str]         # why scope was not applied


class AskResponse(BaseModel):
    request_id: str
    thread_id: str
    data_plane: str
    data_profile: str
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
    admission_scope: AdmissionScope
    models: dict
    latency_ms: float
    timings: dict


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None
    detail: Optional[object] = None


class ReviewDecision(BaseModel):
    """A reviewer's decision on a whole paused draft."""
    model_config = {"extra": "forbid"}
    decision: Literal["approve", "reject"]
    reviewer_note: str = Field(default="", max_length=1000)
