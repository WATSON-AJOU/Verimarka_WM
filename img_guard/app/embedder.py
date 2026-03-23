# app/embedder.py
"""
embedder.py
이미지 임베딩 생성 담당.

지원 모델:
- clip_vit_b32_openai
- openclip_vit_h14_laion2b
- siglip2_so400m_384
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.config import EMBED_DEVICE, EMBED_MODEL, get_embed_model_spec
from app.preprocess import load_image_fixed


def _resolve_device(device: str | None = None) -> str:
    target = (device or EMBED_DEVICE or "auto").lower()
    if target == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return target


@dataclass
class BaseEmbedder:
    model_key: str
    dim: int
    default_batch_size: int
    device: str

    def embed_paths(self, paths: list[str], batch_size: int | None = None) -> np.ndarray:
        raise NotImplementedError


class OpenClipEmbedder(BaseEmbedder):
    def __init__(self, model_key: str, spec: dict[str, Any], device: str | None = None):
        import open_clip

        super().__init__(
            model_key=model_key,
            dim=int(spec["dim"]),
            default_batch_size=int(spec["default_batch_size"]),
            device=_resolve_device(device),
        )
        model_name = str(spec["model_name"])
        pretrained = str(spec["pretrained"])
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int | None = None) -> np.ndarray:
        step = batch_size or self.default_batch_size
        feats: list[torch.Tensor] = []

        for i in range(0, len(paths), step):
            batch = paths[i : i + step]
            imgs = [self.preprocess(load_image_fixed(p)) for p in batch]
            x = torch.stack(imgs).to(self.device)
            f = self.model.encode_image(x)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu())

        out = torch.cat(feats, dim=0).numpy().astype(np.float32)
        return out


class SigLIP2Embedder(BaseEmbedder):
    def __init__(self, model_key: str, spec: dict[str, Any], device: str | None = None):
        try:
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "siglip2 requires transformers. install: pip install transformers timm"
            ) from exc

        super().__init__(
            model_key=model_key,
            dim=int(spec["dim"]),
            default_batch_size=int(spec["default_batch_size"]),
            device=_resolve_device(device),
        )
        hf_id = str(spec["hf_id"])
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device).eval()

    def _extract_tensor(self, out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0]
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    if item.ndim == 3:
                        return item[:, 0]
                    return item
        raise RuntimeError(f"Unsupported SigLIP2 output type: {type(out)}")

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int | None = None) -> np.ndarray:
        step = batch_size or self.default_batch_size
        feats: list[torch.Tensor] = []

        for i in range(0, len(paths), step):
            batch = paths[i : i + step]
            imgs = [load_image_fixed(p) for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            try:
                raw = self.model.get_image_features(pixel_values=pixel_values)
            except Exception:
                raw = self.model(pixel_values=pixel_values)

            f = self._extract_tensor(raw)
            if f.ndim == 3:
                f = f[:, 0]
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu())

        out = torch.cat(feats, dim=0).numpy().astype(np.float32)
        return out


def create_embedder(model_key: str | None = None, device: str | None = None) -> BaseEmbedder:
    key = (model_key or EMBED_MODEL).lower()
    spec = get_embed_model_spec(key)
    backend = str(spec["backend"])
    if backend == "open_clip":
        return OpenClipEmbedder(model_key=key, spec=spec, device=device)
    if backend == "siglip2":
        return SigLIP2Embedder(model_key=key, spec=spec, device=device)
    raise RuntimeError(f"Unsupported embed backend='{backend}' for model='{key}'")


class ClipEmbedder:
    """
    레거시 코드 호환을 위한 wrapper.
    내부적으로는 EMBED_MODEL 기준으로 적절한 임베더를 생성한다.
    """

    def __init__(self, device: str | None = None, model_key: str | None = None):
        self._impl = create_embedder(model_key=model_key, device=device)
        self.model_key = self._impl.model_key
        self.dim = self._impl.dim
        self.device = self._impl.device

    def embed_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        return self._impl.embed_paths(paths=paths, batch_size=batch_size)
