from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.watermark.models import (
    WatermarkDetectRequest,
    WatermarkDetectResponse,
    WatermarkEmbedRequest,
    WatermarkEmbedResponse,
)
from app.watermark.service import WatermarkService


router = APIRouter(prefix="/v1/watermark", tags=["watermark"])

_SERVICE: dict[str, Any] = {}


def _get_service() -> WatermarkService:
    if "service" not in _SERVICE:
        _SERVICE["service"] = WatermarkService.create()
    return _SERVICE["service"]


@router.post("/embed", response_model=WatermarkEmbedResponse)
def watermark_embed(req: WatermarkEmbedRequest):
    return _get_service().embed(req)


@router.post("/detect", response_model=WatermarkDetectResponse)
def watermark_detect(req: WatermarkDetectRequest):
    return _get_service().detect(req)
