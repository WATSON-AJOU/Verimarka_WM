"""
FastAPI layer for img_guard.

Run:
  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.contracts_v1 import (
    ArchiveImageRequestV1,
    ArchiveImageResponseV1,
    GuardRequestV1,
    GuardResponseV1,
    RegisterWorkflowRequestV1,
    RegisterWorkflowResponseV1,
    VectorUpsertRequestV1,
    VectorUpsertResponseV1,
)
from app.guard_service import run_guard_v1
from app.persist_service import archive_image_v1, upsert_vector_embedding_v1
from app.register_workflow_service import run_register_workflow_v1
from app.watermark.router import router as watermark_router

app = FastAPI(title="img_guard API", version="v1")
app.include_router(watermark_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/guard/image", response_model=GuardResponseV1)
def guard_image(req: GuardRequestV1):
    try:
        return run_guard_v1(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/assets/archive", response_model=ArchiveImageResponseV1)
def archive_image(req: ArchiveImageRequestV1):
    try:
        return archive_image_v1(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/vector/upsert", response_model=VectorUpsertResponseV1)
def vector_upsert(req: VectorUpsertRequestV1):
    try:
        return upsert_vector_embedding_v1(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/workflow/register", response_model=RegisterWorkflowResponseV1)
def register_workflow(req: RegisterWorkflowRequestV1):
    try:
        return run_register_workflow_v1(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
