"""
V1 contract models shared by:
- FastAPI REST layer
- internal function-call integration (Django -> AI module)

If backend calls functions instead of HTTP, keep the same schema here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DecisionV1 = Literal["allow", "review", "block"]
NextActionV1 = Literal["none", "start_vote"]


class InputItemV1(BaseModel):
    url: str | None = None
    s3_uri: str | None = None
    s3_key: str | None = None
    local_path: str | None = None
    filename: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "InputItemV1":
        if not (self.url or self.s3_uri or self.s3_key or self.local_path):
            raise ValueError("input item requires one of: local_path, url, s3_uri, s3_key")
        return self


class SearchOptionsV1(BaseModel):
    top_k: int | None = None
    top_phash: int | None = None


class WatermarkOptionsV1(BaseModel):
    apply_on_allow: bool | None = None
    model: str | None = None
    nbits: int | None = None
    scaling_w: float | None = None
    proportion_masked: float | None = None


class GuardOptionsV1(BaseModel):
    search: SearchOptionsV1 | None = None
    watermark: WatermarkOptionsV1 | None = None


class GuardRequestV1(BaseModel):
    job_id: str
    mode: str = "register"
    content_type: str = "image"
    input: list[InputItemV1] = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)
    options: GuardOptionsV1 | None = None

    @field_validator("mode")
    @classmethod
    def normalize_mode(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("content_type")
    @classmethod
    def normalize_content_type(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("input", mode="before")
    @classmethod
    def normalize_input_list(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [v]
        return v


class MatchItemV1(BaseModel):
    db_key: str | None = None
    db_file: str
    cosine: float
    phash_dist: int | None = None


class ScoreV1(BaseModel):
    top_cosine: float | None = None
    top_phash_dist: int | None = None
    policy_version: str = "v1"


class WatermarkResultV1(BaseModel):
    requested: bool = False
    applied: bool = False
    output_url: str | None = None
    output_key: str | None = None
    model: str | None = None
    model_version: str | None = None
    nbits: int | None = None
    scaling_w: float | None = None
    proportion_masked: float | None = None
    payload_id: str | None = None


class TimingV1(BaseModel):
    download: int
    embed: int
    ann_search: int
    phash: int
    total: int


class GuardResponseV1(BaseModel):
    job_id: str
    mode: str
    content_type: str
    success: bool
    decision: DecisionV1
    reason: str
    next_action: NextActionV1
    scores: ScoreV1
    top_match: MatchItemV1 | None
    candidates: list[MatchItemV1]
    watermark: WatermarkResultV1
    timing_ms: TimingV1


AssetKindV1 = Literal[
    "watermark_request_original",
    "watermark_result",
    "verify_request",
    "register_request",
    "rejected_request",
]


class ArchiveImageRequestV1(BaseModel):
    job_id: str
    kind: AssetKindV1
    input: InputItemV1
    meta: dict[str, Any] = Field(default_factory=dict)
    bucket: str | None = None
    output_filename: str | None = None


class ArchiveImageResponseV1(BaseModel):
    job_id: str
    kind: AssetKindV1
    success: bool
    reason: str | None = None
    bucket: str | None = None
    s3_key: str | None = None
    s3_uri: str | None = None
    file_name: str | None = None


class VectorUpsertRequestV1(BaseModel):
    job_id: str
    input: InputItemV1
    s3_key: str | None = None
    asset_url: str | None = None
    file_name: str | None = None


class VectorUpsertResponseV1(BaseModel):
    job_id: str
    success: bool
    reason: str | None = None
    table: str | None = None
    record_id: int | None = None
    file_name: str | None = None
    s3_key: str | None = None
    phash: int | None = None


class RegisterWorkflowOptionsV1(BaseModel):
    archive_register_request: bool = True
    archive_rejected_request: bool = True
    archive_wm_request_original: bool = True
    archive_wm_result: bool = True
    upsert_vector_on_allow: bool = True
    require_token_issued_for_upsert: bool = True
    token_issued_meta_key: str = "token_issued"


class RegisterWorkflowRequestV1(BaseModel):
    job_id: str
    input: InputItemV1
    meta: dict[str, Any] = Field(default_factory=dict)
    bucket: str | None = None
    guard_options: GuardOptionsV1 | None = None
    watermark_options: WatermarkOptionsV1 | None = None
    options: RegisterWorkflowOptionsV1 = Field(default_factory=RegisterWorkflowOptionsV1)


class RegisterWorkflowAssetsV1(BaseModel):
    register_request_s3_key: str | None = None
    rejected_request_s3_key: str | None = None
    wm_request_original_s3_key: str | None = None
    wm_result_s3_key: str | None = None


class RegisterWorkflowResponseV1(BaseModel):
    job_id: str
    success: bool
    decision: DecisionV1 | None = None
    next_action: NextActionV1 | None = None
    reason: str | None = None
    guard: GuardResponseV1 | None = None
    assets: RegisterWorkflowAssetsV1 = Field(default_factory=RegisterWorkflowAssetsV1)
    watermark_embed_success: bool | None = None
    watermark_output_key: str | None = None
    vector_upsert: VectorUpsertResponseV1 | None = None
    pending_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
