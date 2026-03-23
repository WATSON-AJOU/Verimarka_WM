from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field


class MediaInput(BaseModel):
    url: Optional[str] = None
    s3_uri: Optional[str] = None
    s3_key: Optional[str] = None
    local_path: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None


class WatermarkEmbedOptions(BaseModel):
    model: str = "wam"
    nbits: int = 32
    scaling_w: float = 2.0
    proportion_masked: float = 0.65
    seed: Optional[int] = None


class WatermarkDetectOptions(BaseModel):
    model: str = "wam"
    threshold: float = 0.5


class WatermarkEmbedRequest(BaseModel):
    job_id: str
    input: MediaInput
    meta: dict[str, Any] = Field(default_factory=dict)
    options: WatermarkEmbedOptions = Field(default_factory=WatermarkEmbedOptions)
    payload_id: Optional[str] = None


class WatermarkDetectRequest(BaseModel):
    job_id: str
    input: MediaInput
    options: WatermarkDetectOptions = Field(default_factory=WatermarkDetectOptions)


@dataclass
class EmbedArtifact:
    applied: bool
    output_path: str | None
    output_url: str | None = None
    output_key: str | None = None
    payload_id: str | None = None
    model_version: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectArtifact:
    detected: bool
    confidence: float | None = None
    bit_accuracy: float | None = None
    payload_id: str | None = None
    model_version: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class WatermarkEmbedResult(BaseModel):
    applied: bool
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    output_key: Optional[str] = None
    payload_id: Optional[str] = None
    model: str
    model_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class WatermarkDetectResult(BaseModel):
    detected: bool
    confidence: Optional[float] = None
    bit_accuracy: Optional[float] = None
    payload_id: Optional[str] = None
    model: str
    model_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class WatermarkEmbedResponse(BaseModel):
    job_id: str
    success: bool
    reason: Optional[str] = None
    result: WatermarkEmbedResult


class WatermarkDetectResponse(BaseModel):
    job_id: str
    success: bool
    reason: Optional[str] = None
    result: WatermarkDetectResult
