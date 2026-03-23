# app/config.py
"""
프로젝트 전체가 공유하는 기준값(경로/모델/검색/정책/인덱스 파라미터)를 모아둔 파일
"""
import hashlib
import os
from pathlib import Path
from urllib.parse import quote_plus


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _to_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _to_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


# 프로젝트 루트 = img_guard/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
DB_IMAGES_DIR = DATA_DIR / "db_images" / "dataset60"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
HNSW_INDEX_PATH = DATA_DIR / "hnsw.index"

DB_MANIFEST_PATH = DATA_DIR / "db_manifest.json"
DB_SIGNATURE_MODE = "mtime_size"

# 임베딩 모델 설정 (서비스 스위칭용)
EMBED_MODEL = _env("EMBED_MODEL", "clip_vit_b32_openai").lower()
EMBED_DEVICE = _env("EMBED_DEVICE", "auto").lower()

EMBED_MODEL_SPECS: dict[str, dict[str, object]] = {
    "clip_vit_b32_openai": {
        "backend": "open_clip",
        "dim": 512,
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "default_batch_size": 32,
    },
    "openclip_vit_h14_laion2b": {
        "backend": "open_clip",
        "dim": 1024,
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "default_batch_size": 16,
    },
    "siglip2_so400m_384": {
        "backend": "siglip2",
        "dim": 1152,
        "hf_id": "google/siglip2-so400m-patch16-384",
        "default_batch_size": 8,
    },
}


def get_embed_model_spec(model_key: str | None = None) -> dict[str, object]:
    key = (model_key or EMBED_MODEL).lower()
    spec = EMBED_MODEL_SPECS.get(key)
    if spec is None:
        supported = ", ".join(sorted(EMBED_MODEL_SPECS.keys()))
        raise RuntimeError(f"Unsupported EMBED_MODEL='{key}'. supported: {supported}")
    return spec


EMBED_DIM = int(get_embed_model_spec()["dim"])

# 레거시 상수(기존 코드 호환용)
CLIP_MODEL_NAME = str(get_embed_model_spec("clip_vit_b32_openai")["model_name"])
CLIP_PRETRAINED = str(get_embed_model_spec("clip_vit_b32_openai")["pretrained"])

# ANN backend
# - local: 로컬 HNSW 인덱스 사용
# - pgvector: PostgreSQL(pgvector) 검색 사용
ANN_BACKEND = _env("ANN_BACKEND", "local").lower()

# DB 원본값 (백엔드/노션 전달 형식)
DB_ENGINE = _env("DB_ENGINE", "postgresql")
DB_NAME = _env("DB_NAME", "")
DB_USER = _env("DB_USER", "")
DB_PASSWORD = _env("DB_PASSWORD", "")
DB_HOST = _env("DB_HOST", "")
DB_PORT = _env("DB_PORT", "5432")
DB_SSLMODE = _env("DB_SSLMODE", "require")


def _build_vector_dsn_from_db_env() -> str:
    if not (DB_NAME and DB_USER and DB_HOST):
        return ""
    user = quote_plus(DB_USER)
    pw = quote_plus(DB_PASSWORD)
    host = DB_HOST
    port = DB_PORT or "5432"
    db = DB_NAME
    ssl = DB_SSLMODE or "require"
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}?sslmode={ssl}"


# pgvector config (service mode)
VECTOR_DSN = _env("VECTOR_DSN", "") or _build_vector_dsn_from_db_env()

VECTOR_TABLE_BY_MODEL = {
    "clip_vit_b32_openai": "image_embeddings_clip_b32",
    "openclip_vit_h14_laion2b": "image_embeddings_openclip_h14",
    "siglip2_so400m_384": "image_embeddings_siglip2_so400m",
}

VECTOR_TABLE = _env("VECTOR_TABLE", "") or VECTOR_TABLE_BY_MODEL.get(EMBED_MODEL, "image_embeddings")
VECTOR_EMBED_COL = _env("VECTOR_EMBED_COL", "embedding")
VECTOR_ID_COL = _env("VECTOR_ID_COL", "id")
VECTOR_FILE_COL = _env("VECTOR_FILE_COL", "file_name")
VECTOR_KEY_COL = _env("VECTOR_KEY_COL", "s3_key")
VECTOR_URL_COL = _env("VECTOR_URL_COL", "asset_url")
VECTOR_PHASH_COL = _env("VECTOR_PHASH_COL", "phash")
VECTOR_S3_BUCKET = _env("VECTOR_S3_BUCKET", "")

# temp/cache dir for downloaded images
TMP_DIR = Path(_env("IMG_GUARD_TMP_DIR", str(DATA_DIR / "cache" / "tmp")))
S3_DEFAULT_BUCKET = _env("S3_DEFAULT_BUCKET", "")
AWS_REGION = _env("AWS_REGION", "")
S3_ENDPOINT_URL = _env("S3_ENDPOINT_URL", "")
HTTP_DOWNLOAD_TIMEOUT_SEC = _to_float("HTTP_DOWNLOAD_TIMEOUT_SEC", 20.0)
DOWNLOAD_RETRIES = _to_int("DOWNLOAD_RETRIES", 2)

# S3 object layout (analysis/audit + watermark pipeline)
S3_PREFIX_WM_REQUEST_ORIGINAL = _env("S3_PREFIX_WM_REQUEST_ORIGINAL", "watermark/request_original")
S3_PREFIX_WM_RESULT = _env("S3_PREFIX_WM_RESULT", "watermark/result")
S3_PREFIX_VERIFY_REQUEST = _env("S3_PREFIX_VERIFY_REQUEST", "request/verify")
S3_PREFIX_REGISTER_REQUEST = _env("S3_PREFIX_REGISTER_REQUEST", "request/register")
S3_PREFIX_REJECTED = _env("S3_PREFIX_REJECTED", "request/rejected")

MAX_INPUT_MB = _to_int("MAX_INPUT_MB", 20)
MAX_IMAGE_SIDE = _to_int("MAX_IMAGE_SIDE", 2048)
MAX_IMAGE_PIXELS = _to_int("MAX_IMAGE_PIXELS", 20000000)

# 검색 설정
TOP_K = _to_int("TOP_K", 10)
TOP_PHASH = _to_int("TOP_PHASH", 10)

# 정책(3상태 룰) threshold v1
COS_BLOCK = _to_float("COS_BLOCK", 0.97)
PHASH_BLOCK = _to_int("PHASH_BLOCK", 10)

COS_ALLOW_A = _to_float("COS_ALLOW_A", 0.90)
PHASH_ALLOW_A = _to_int("PHASH_ALLOW_A", 20)

COS_ALLOW_B = _to_float("COS_ALLOW_B", 0.85)

# HNSW 파라미터
HNSW_M = _to_int("HNSW_M", 16)
HNSW_EF_CONSTRUCTION = _to_int("HNSW_EF_CONSTRUCTION", 200)
HNSW_EF_SEARCH = _to_int("HNSW_EF_SEARCH", 50)

# Watermark backend
# - mock: 계약 테스트용 (실제 워터마킹 미적용, I/O 플로우만 검증)
# - wam: Meta WAM 실제 추론
WM_BACKEND = _env("WM_BACKEND", "mock").lower()

# WAM repo/config/weights path
WAM_REPO_DIR = Path(
    _env("WAM_REPO_DIR", str(PROJECT_ROOT / "third_party" / "watermark-anything"))
)
WAM_PARAMS_PATH = Path(
    _env("WAM_PARAMS_PATH", str(WAM_REPO_DIR / "checkpoints" / "params.json"))
)
WAM_CHECKPOINT_PATH = Path(
    _env("WAM_CHECKPOINT_PATH", str(PROJECT_ROOT / "models" / "wam" / "wam_mit.pth"))
)

# Watermark runtime paths
WM_TMP_DIR = Path(_env("WM_TMP_DIR", str(TMP_DIR / "watermark")))
WM_OUTPUT_DIR = Path(_env("WM_OUTPUT_DIR", str(DATA_DIR / "wm_outputs")))


def runtime_signature() -> str:
    """
    가드 엔진 캐시를 재생성할지 판단하기 위한 런타임 서명.
    """
    raw = "|".join(
        [
            ANN_BACKEND,
            EMBED_MODEL,
            str(EMBED_DIM),
            VECTOR_TABLE,
            VECTOR_EMBED_COL,
            VECTOR_ID_COL,
            VECTOR_FILE_COL,
            VECTOR_KEY_COL,
            VECTOR_URL_COL,
            VECTOR_PHASH_COL,
            str(bool(VECTOR_DSN)),
            WM_BACKEND,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
