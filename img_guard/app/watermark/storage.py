from __future__ import annotations

from pathlib import Path

from app.config import S3_DEFAULT_BUCKET
from app.source_io import resolve_source_to_local
from app.watermark.models import MediaInput


ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def resolve_input_to_local(media: MediaInput, tmp_dir: Path) -> Path:
    if media.local_path:
        path = Path(media.local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input.local_path not found: {path}")
        return path

    source = media.url or media.s3_uri or media.s3_key
    if source:
        path = resolve_source_to_local(source, tmp_dir, default_s3_bucket=S3_DEFAULT_BUCKET)
        return path.resolve()

    raise ValueError("input requires one of: local_path, url, s3_uri, s3_key")


def ensure_image_suffix(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_SUFFIX:
        raise ValueError(f"unsupported image type: {path.suffix}")
