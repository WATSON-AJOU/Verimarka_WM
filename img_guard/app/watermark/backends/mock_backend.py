from __future__ import annotations

import shutil
from pathlib import Path

from app.watermark.backends.base import WatermarkBackend
from app.watermark.models import (
    DetectArtifact,
    EmbedArtifact,
    WatermarkDetectOptions,
    WatermarkEmbedOptions,
)


class MockWatermarkBackend(WatermarkBackend):
    """Contract-test backend. No real watermarking, but full I/O flow works."""

    name = "mock"

    def embed(
        self,
        input_path: Path,
        output_dir: Path,
        payload_bits: str,
        payload_id: str,
        options: WatermarkEmbedOptions,
    ) -> EmbedArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{input_path.stem}_wm{input_path.suffix}"
        out_path = output_dir / out_name
        shutil.copy2(input_path, out_path)

        return EmbedArtifact(
            applied=True,
            output_path=str(out_path),
            payload_id=payload_id,
            model_version="mock-v1",
            details={
                "backend": self.name,
                "nbits": options.nbits,
                "scaling_w": options.scaling_w,
                "proportion_masked": options.proportion_masked,
                "payload_bits_head": payload_bits[:16],
            },
        )

    def detect(self, input_path: Path, options: WatermarkDetectOptions) -> DetectArtifact:
        # Simple convention for local workflow: *_wm.* is considered detected.
        detected = input_path.stem.endswith("_wm")
        conf = 0.99 if detected else 0.01

        return DetectArtifact(
            detected=detected,
            confidence=conf,
            bit_accuracy=1.0 if detected else 0.0,
            payload_id=None,
            model_version="mock-v1",
            details={"backend": self.name, "threshold": options.threshold},
        )
