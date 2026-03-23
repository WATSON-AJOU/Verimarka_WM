from __future__ import annotations

from pathlib import Path

from app.config import (
    WAM_CHECKPOINT_PATH,
    WAM_PARAMS_PATH,
    WAM_REPO_DIR,
    WM_BACKEND,
    WM_OUTPUT_DIR,
    WM_TMP_DIR,
)
from app.watermark.backends import MockWatermarkBackend, WamWatermarkBackend, WatermarkBackend
from app.watermark.models import (
    WatermarkDetectRequest,
    WatermarkDetectResponse,
    WatermarkDetectResult,
    WatermarkEmbedRequest,
    WatermarkEmbedResponse,
    WatermarkEmbedResult,
)
from app.watermark.payload import make_payload_bits, make_payload_id
from app.watermark.storage import ensure_image_suffix, resolve_input_to_local


class WatermarkService:
    def __init__(
        self,
        backend: WatermarkBackend,
        tmp_dir: Path,
        output_dir: Path,
    ):
        self.backend = backend
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir

    @classmethod
    def create(cls) -> "WatermarkService":
        backend_name = WM_BACKEND
        if backend_name == "wam":
            backend: WatermarkBackend = WamWatermarkBackend(
                repo_dir=WAM_REPO_DIR,
                params_path=WAM_PARAMS_PATH,
                checkpoint_path=WAM_CHECKPOINT_PATH,
            )
        elif backend_name == "mock":
            backend = MockWatermarkBackend()
        else:
            raise RuntimeError(f"Unsupported WM_BACKEND: {backend_name}")

        return cls(backend=backend, tmp_dir=WM_TMP_DIR, output_dir=WM_OUTPUT_DIR)

    def embed(self, req: WatermarkEmbedRequest) -> WatermarkEmbedResponse:
        model_name = req.options.model

        try:
            input_path = resolve_input_to_local(req.input, self.tmp_dir)
            ensure_image_suffix(input_path)

            payload_bits = make_payload_bits(req.meta, req.options.nbits)
            payload_id = req.payload_id or make_payload_id(payload_bits)

            out_dir = self.output_dir / req.job_id
            artifact = self.backend.embed(
                input_path=input_path,
                output_dir=out_dir,
                payload_bits=payload_bits,
                payload_id=payload_id,
                options=req.options,
            )

            return WatermarkEmbedResponse(
                job_id=req.job_id,
                success=True,
                reason=None,
                result=WatermarkEmbedResult(
                    applied=artifact.applied,
                    output_path=artifact.output_path,
                    output_url=artifact.output_url,
                    output_key=artifact.output_key,
                    payload_id=artifact.payload_id,
                    model=model_name,
                    model_version=artifact.model_version,
                    details=artifact.details,
                ),
            )

        except Exception as e:
            return WatermarkEmbedResponse(
                job_id=req.job_id,
                success=False,
                reason=str(e),
                result=WatermarkEmbedResult(
                    applied=False,
                    output_path=None,
                    output_url=None,
                    output_key=None,
                    payload_id=None,
                    model=model_name,
                    model_version=None,
                    details={"backend": self.backend.name},
                ),
            )

    def detect(self, req: WatermarkDetectRequest) -> WatermarkDetectResponse:
        model_name = req.options.model

        try:
            input_path = resolve_input_to_local(req.input, self.tmp_dir)
            ensure_image_suffix(input_path)

            artifact = self.backend.detect(input_path=input_path, options=req.options)

            return WatermarkDetectResponse(
                job_id=req.job_id,
                success=True,
                reason=None,
                result=WatermarkDetectResult(
                    detected=artifact.detected,
                    confidence=artifact.confidence,
                    bit_accuracy=artifact.bit_accuracy,
                    payload_id=artifact.payload_id,
                    model=model_name,
                    model_version=artifact.model_version,
                    details=artifact.details,
                ),
            )

        except Exception as e:
            return WatermarkDetectResponse(
                job_id=req.job_id,
                success=False,
                reason=str(e),
                result=WatermarkDetectResult(
                    detected=False,
                    confidence=None,
                    bit_accuracy=None,
                    payload_id=None,
                    model=model_name,
                    model_version=None,
                    details={"backend": self.backend.name},
                ),
            )
