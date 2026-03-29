from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.watermark.backends.base import WatermarkBackend
from app.watermark.models import (
    DetectArtifact,
    EmbedArtifact,
    WatermarkDetectOptions,
    WatermarkEmbedOptions,
)


class WamWatermarkBackend(WatermarkBackend):
    """
    Meta WAM inference backend.
    - Supports embed and detect for service integration.
    - Lazily loads model once and reuses it.
    """

    name = "wam"

    def __init__(self, repo_dir: Path, params_path: Path, checkpoint_path: Path):
        self.repo_dir = repo_dir
        self.params_path = params_path
        self.checkpoint_path = checkpoint_path
        self._wam = None
        self._device = "cpu"
        self._model_nbits = 32
        self._default_scaling_w = 2.0
        self._model_version = f"{self.checkpoint_path.stem}"
        self._default_transform = None
        self._unnormalize_img = None
        self._msg_predict_inference = None
        self._torch = None
        self._to_pil = None

    def _assert_ready(self) -> None:
        if not self.repo_dir.exists():
            raise RuntimeError(f"WAM repo not found: {self.repo_dir}")
        if not self.params_path.exists():
            raise RuntimeError(f"WAM params.json not found: {self.params_path}")
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "WAM checkpoint not found. Download MIT weights: "
                f"{self.checkpoint_path}"
            )

    def _ensure_repo_import_path(self) -> None:
        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def _require_deps(self) -> None:
        try:
            import omegaconf  # noqa: F401
            import torch  # noqa: F401
            from torchvision import transforms  # noqa: F401
            from watermark_anything.models import Wam, build_embedder, build_extractor  # noqa: F401
            from watermark_anything.augmentation.augmenter import Augmenter  # noqa: F401
            from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img  # noqa: F401
            from watermark_anything.data.metrics import msg_predict_inference  # noqa: F401
            from watermark_anything.modules.jnd import JND  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WAM dependencies are missing. "
                "Install inference deps with: "
                "`pip install omegaconf==2.3.0 einops==0.8.0 opencv-python==4.10.0.84`"
            ) from exc

    def _resolve_cfg_path(self, p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((self.repo_dir / path).resolve())

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        assert self._torch is not None
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _create_mask(self, img_pt, proportion_masked: float):
        assert self._torch is not None
        _, _, height, width = img_pt.shape

        if proportion_masked <= 0.0:
            return self._torch.zeros((1, 1, height, width), device=img_pt.device)
        if proportion_masked >= 0.999:
            return self._torch.ones((1, 1, height, width), device=img_pt.device)

        mask_area = max(1, int(height * width * proportion_masked))
        aspect = random.uniform(0.5, 2.0)
        mask_width = max(1, int((mask_area * aspect) ** 0.5))
        mask_height = max(1, int(mask_area / max(1, mask_width)))
        mask_width = min(mask_width, width)
        mask_height = min(mask_height, height)

        x_start = random.randint(0, max(0, width - mask_width))
        y_start = random.randint(0, max(0, height - mask_height))

        mask = self._torch.zeros((1, 1, height, width), device=img_pt.device)
        mask[:, :, y_start : y_start + mask_height, x_start : x_start + mask_width] = 1.0
        return mask

    @staticmethod
    def _bits_to_str(bit_values) -> str:
        return "".join("1" if int(v) else "0" for v in bit_values)

    def _payload_bits_to_tensor(self, payload_bits: str):
        assert self._torch is not None
        if len(payload_bits) != self._model_nbits:
            raise ValueError(
                f"payload_bits length ({len(payload_bits)}) must match model nbits ({self._model_nbits})"
            )
        if not set(payload_bits).issubset({"0", "1"}):
            raise ValueError("payload_bits must contain only '0' or '1'")

        values = [1.0 if ch == "1" else 0.0 for ch in payload_bits]
        return self._torch.tensor([values], dtype=self._torch.float32, device=self._device)

    def _load_model(self) -> None:
        if self._wam is not None:
            return

        self._assert_ready()
        self._ensure_repo_import_path()
        self._require_deps()

        import omegaconf
        import torch
        from torchvision import transforms
        from watermark_anything.augmentation.augmenter import Augmenter
        from watermark_anything.data.metrics import msg_predict_inference
        from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
        from watermark_anything.models import Wam, build_embedder, build_extractor
        from watermark_anything.modules.jnd import JND

        with self.params_path.open("r", encoding="utf-8") as f:
            params = json.load(f)
        args = argparse.Namespace(**params)

        embedder_cfg = omegaconf.OmegaConf.load(self._resolve_cfg_path(args.embedder_config))
        extractor_cfg = omegaconf.OmegaConf.load(self._resolve_cfg_path(args.extractor_config))
        augmenter_cfg = omegaconf.OmegaConf.load(self._resolve_cfg_path(args.augmentation_config))
        attenuation_cfg = omegaconf.OmegaConf.load(self._resolve_cfg_path(args.attenuation_config))

        embedder = build_embedder(args.embedder_model, embedder_cfg[args.embedder_model], args.nbits)
        extractor = build_extractor(extractor_cfg.model, extractor_cfg[args.extractor_model], args.img_size, args.nbits)
        augmenter = Augmenter(**augmenter_cfg)
        try:
            attenuation = JND(
                **attenuation_cfg[args.attenuation],
                preprocess=unnormalize_img,
                postprocess=normalize_img,
            )
        except Exception:
            attenuation = None

        wam = Wam(
            embedder=embedder,
            detector=extractor,
            augmenter=augmenter,
            attenuation=attenuation,
            scaling_w=float(args.scaling_w),
            scaling_i=float(args.scaling_i),
            img_size_extractor=int(args.img_size_extractor),
        )

        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        wam.load_state_dict(checkpoint, strict=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._wam = wam.to(self._device).eval()
        self._model_nbits = int(args.nbits)
        self._default_scaling_w = float(args.scaling_w)
        self._default_transform = default_transform
        self._unnormalize_img = unnormalize_img
        self._msg_predict_inference = msg_predict_inference
        self._to_pil = transforms.ToPILImage()
        self._torch = torch

    def _load_image_tensor(self, input_path: Path):
        assert self._default_transform is not None
        assert self._torch is not None
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
        return self._default_transform(img).unsqueeze(0).to(self._device)

    def _save_image_tensor(self, img_t, out_path: Path) -> None:
        assert self._unnormalize_img is not None
        assert self._to_pil is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_01 = self._unnormalize_img(img_t).clamp(0, 1).squeeze(0).cpu()
        pil_img = self._to_pil(img_01)
        pil_img.save(out_path)

    def embed(
        self,
        input_path: Path,
        output_dir: Path,
        payload_bits: str,
        payload_id: str,
        options: WatermarkEmbedOptions,
    ) -> EmbedArtifact:
        self._load_model()
        assert self._wam is not None
        assert self._torch is not None
        assert self._msg_predict_inference is not None

        if options.nbits != self._model_nbits:
            raise ValueError(
                f"Requested nbits={options.nbits}, but current checkpoint supports nbits={self._model_nbits}."
            )
        if not (0.0 <= options.proportion_masked <= 1.0):
            raise ValueError("proportion_masked must be in [0.0, 1.0]")

        if options.seed is not None:
            self._seed_all(int(options.seed))

        msg_tensor = self._payload_bits_to_tensor(payload_bits)
        img_pt = self._load_image_tensor(input_path)

        scaling_w = float(options.scaling_w if options.scaling_w is not None else self._default_scaling_w)
        self._wam.scaling_w = scaling_w

        with self._torch.no_grad():
            embed_out = self._wam.embed(img_pt, msg_tensor)
            imgs_w = embed_out["imgs_w"]
            mask = self._create_mask(img_pt, float(options.proportion_masked))
            img_w = imgs_w * mask + img_pt * (1 - mask)

            # quick self-check decode on generated image
            det = self._wam.detect(img_w)
            preds = det["preds"]  # B x (1+nbits) x H x W
            mask_probs = self._torch.sigmoid(preds[:, 0:1, :, :])
            bit_preds = preds[:, 1:, :, :]
            decoded = self._msg_predict_inference(bit_preds, mask_probs).float()[0]
            bit_acc = float((decoded == msg_tensor.cpu()[0]).float().mean().item())
            decoded_bits = self._bits_to_str(decoded.int().tolist())
            detect_conf = float(mask_probs.max().item())

        out_path = output_dir / f"{input_path.stem}_wm{input_path.suffix}"
        self._save_image_tensor(img_w, out_path)

        mask_ratio = float(mask.mean().item())
        mask_pixels = int(mask.sum().item())

        return EmbedArtifact(
            applied=True,
            output_path=str(out_path),
            payload_id=payload_id,
            model_version=self._model_version,
            details={
                "backend": self.name,
                "device": self._device,
                "checkpoint": str(self.checkpoint_path),
                "nbits": self._model_nbits,
                "scaling_w": scaling_w,
                "proportion_masked": float(options.proportion_masked),
                "mask_ratio": mask_ratio,
                "mask_pixels": mask_pixels,
                "bit_accuracy_self_check": bit_acc,
                "decoded_bits": decoded_bits,
                "detect_confidence_self_check": detect_conf,
            },
        )

    def detect(self, input_path: Path, options: WatermarkDetectOptions) -> DetectArtifact:
        self._load_model()
        assert self._wam is not None
        assert self._torch is not None
        assert self._msg_predict_inference is not None

        if not (0.0 <= options.threshold <= 1.0):
            raise ValueError("threshold must be in [0.0, 1.0]")

        img_pt = self._load_image_tensor(input_path)
        with self._torch.no_grad():
            det = self._wam.detect(img_pt)
            preds = det["preds"]
            mask_probs = self._torch.sigmoid(preds[:, 0:1, :, :])
            confidence = float(mask_probs.max().item())
            pixel_ratio = float((mask_probs > 0.5).float().mean().item())
            detected = confidence >= float(options.threshold)

            decoded_bits = None
            payload_id = None
            if self._model_nbits > 0 and preds.shape[1] > 1:
                bit_preds = preds[:, 1:, :, :]
                decoded = self._msg_predict_inference(bit_preds, mask_probs).float()[0]
                decoded_bits = self._bits_to_str(decoded.int().tolist())
                payload_id = hashlib.sha1(decoded_bits.encode("utf-8")).hexdigest()[:16]

        return DetectArtifact(
            detected=bool(detected),
            confidence=confidence,
            bit_accuracy=None,
            payload_id=payload_id,
            model_version=self._model_version,
            details={
                "backend": self.name,
                "device": self._device,
                "checkpoint": str(self.checkpoint_path),
                "threshold": float(options.threshold),
                "max_mask_prob": confidence,
                "wm_pixel_ratio_at_0.5": pixel_ratio,
                "decoded_bits": decoded_bits,
                "nbits": self._model_nbits,
            },
        )
