from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.watermark.models import (
    DetectArtifact,
    EmbedArtifact,
    WatermarkDetectOptions,
    WatermarkEmbedOptions,
)


class WatermarkBackend(ABC):
    name: str = "base"

    @abstractmethod
    def embed(
        self,
        input_path: Path,
        output_dir: Path,
        payload_bits: str,
        payload_id: str,
        options: WatermarkEmbedOptions,
    ) -> EmbedArtifact:
        raise NotImplementedError

    @abstractmethod
    def detect(self, input_path: Path, options: WatermarkDetectOptions) -> DetectArtifact:
        raise NotImplementedError
