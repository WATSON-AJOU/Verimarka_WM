## img_guard

이미지 등록 요청에 대해 **유사도 탐색 + pHash 검증 + 3상태 판정(ALLOW/REVIEW/BLOCK)**을 수행하는 모듈입니다.
로컬(HNSW) 기반 테스트와 서비스(pgvector) 연동을 모두 지원합니다.

**핵심 흐름**
- 이미지 다운로드/로드
- 임베딩 생성(모델 스위칭: CLIP/OpenCLIP/SigLIP2)
- ANN Top‑K 검색
- pHash 정밀 비교
- 정책 판정(ALLOW/REVIEW/BLOCK)

**구성**
- `app/ann_index.py`: 로컬 HNSW 또는 pgvector 검색
- `app/embedder.py`: 임베딩 모델 factory
- `app/phash.py`: pHash 비교
- `app/policy.py`: 3상태 판정 규칙
- `app/guard.py`: 파이프라인 오케스트레이션
- `app/contracts_v1.py`: 서버↔AI V1 계약 스키마(공용)
- `app/guard_service.py`: 함수호출/REST 공용 Guard 실행 진입점
- `app/api.py`: FastAPI Guard API
- `app/watermark/`: 워터마크 삽입/검출 서비스 레이어 골격
- `experiments/`: 특징점 매칭/회전 실험 스크립트

---

## 빠른 시작 (로컬)

```bash
cd /Users/pjunese/Desktop/WATSON/img_guard
python3 -m venv .venv
source .venv/bin/activate
# CPU 서버(t3/t4g 등) 권장 설치
pip install --no-cache-dir -r requirements.cpu.txt

# GPU 서버(g4/g5/g6) 사용 시
# pip install --no-cache-dir -r requirements.gpu.txt
```

데이터 위치:
- `data/db_images/dataset60/`에 DB 이미지가 있어야 합니다.

로컬 CLI 테스트:
```bash
python3 -m app.main --query data/db_images/dataset60/dt_001.png --json
```

---

## Guard API (FastAPI)

서버 실행:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

엔드포인트:
- `GET /health`
- `POST /v1/guard/image`
- `POST /v1/assets/archive` (S3 아카이브 저장)
- `POST /v1/vector/upsert` (승인 원본 벡터 DB 저장)
- `POST /v1/workflow/register` (등록 통합 워크플로우 1회 호출)
- `POST /v1/watermark/embed` (워터마크 삽입)
- `POST /v1/watermark/detect` (워터마크 검출)

함수호출(백엔드 내부 연동)도 동일 스키마 사용:

```python
from app.guard_service import run_guard_v1

response = run_guard_v1(request_dict)  # request_dict는 contracts_v1 GuardRequestV1 형태
print(response.model_dump())
```

통합(백엔드 권장) 함수호출:

```python
from app.register_workflow_service import run_register_workflow_v1

resp = run_register_workflow_v1({
    "job_id": "uuid",
    "input": {"s3_key": "request/register/u1/c1/original.png", "filename": "original.png"},
    "meta": {"user_id": "u1", "content_id": "c1", "token_issued": True},
    "options": {
        "archive_register_request": True,
        "archive_rejected_request": True,
        "archive_wm_request_original": True,
        "archive_wm_result": True,
        "upsert_vector_on_allow": True,
        "require_token_issued_for_upsert": True
    }
})
print(resp.model_dump())
```

요청/응답 스키마는 팀의 `소통양식.txt` JSON 예시를 기준으로 합니다.

---

## 서비스 모드 (pgvector)

기본은 로컬(HNSW)입니다. 서비스(pgvector)로 전환하려면 환경변수를 설정하세요.

```bash
export ANN_BACKEND=pgvector
export EMBED_MODEL="clip_vit_b32_openai"   # or openclip_vit_h14_laion2b / siglip2_so400m_384

# 1) VECTOR_DSN을 직접 주입하는 방식
export VECTOR_DSN="postgresql://user:pass@host:5432/vector_db?sslmode=require"

# 2) 또는 DB_* 만 주입해도 VECTOR_DSN 자동 생성됨
# export DB_NAME="vector_db"
# export DB_USER="junseo"
# export DB_PASSWORD="..."
# export DB_HOST="..."
# export DB_PORT="5432"
# export DB_SSLMODE="require"

# 테이블은 EMBED_MODEL 기반 기본값 자동 선택:
# - clip_vit_b32_openai      -> image_embeddings_clip_b32
# - openclip_vit_h14_laion2b -> image_embeddings_openclip_h14
# - siglip2_so400m_384       -> image_embeddings_siglip2_so400m
# 필요하면 수동 override 가능
# export VECTOR_TABLE="image_embeddings_siglip2_so400m"
export VECTOR_EMBED_COL="embedding"
export VECTOR_ID_COL="id"
export VECTOR_FILE_COL="file_name"
export VECTOR_KEY_COL="s3_key"
export VECTOR_URL_COL="asset_url"
export VECTOR_PHASH_COL="phash"   # DB에 pHash 저장 시 사용
export VECTOR_S3_BUCKET=""         # db_key가 bare s3 key일 때 사용

# 업로드/디코딩 안전장치 (OOM 방지)
export MAX_INPUT_MB=20
export MAX_IMAGE_SIDE=2048
export MAX_IMAGE_PIXELS=20000000

# 분석기록/자산 저장 S3 prefix
export S3_PREFIX_WM_REQUEST_ORIGINAL="watermark/request_original"
export S3_PREFIX_WM_RESULT="watermark/result"
export S3_PREFIX_VERIFY_REQUEST="request/verify"
export S3_PREFIX_REGISTER_REQUEST="request/register"
export S3_PREFIX_REJECTED="request/rejected"
```

pgvector bootstrap SQL:
```bash
psql "$VECTOR_DSN" -f sql/bootstrap_pgvector.sql
```

런타임 사전 점검:
```bash
python3 scripts/preflight_runtime.py
```

SigLIP2 운영 테스트(권장 시작점):
```bash
source /Users/pjunese/Desktop/WATSON/img_guard/scripts/set_runtime_env_siglip2.sh
python3 scripts/preflight_runtime.py
```

로컬 이미지 사전 임베딩(벡터DB preload):
```bash
# 드라이런
python3 scripts/preload_vectors_from_dir.py \
  --src-dir data/db_images \
  --recursive \
  --upload-s3-prefix preload/original \
  --dry-run

# 실제 실행
python3 scripts/preload_vectors_from_dir.py \
  --src-dir data/db_images \
  --recursive \
  --upload-s3-prefix preload/original
```

워터마크 백엔드 선택:

```bash
# 계약 테스트용 (기본): 실제 WAM 추론 없이 I/O 플로우 검증
export WM_BACKEND=mock

# 실제 WAM 추론 모드
export WM_BACKEND=wam
export WAM_REPO_DIR="/Users/pjunese/Desktop/WATSON/img_guard/third_party/watermark-anything"
export WAM_PARAMS_PATH="/Users/pjunese/Desktop/WATSON/img_guard/third_party/watermark-anything/checkpoints/params.json"
export WAM_CHECKPOINT_PATH="/Users/pjunese/Desktop/WATSON/img_guard/models/wam/wam_mit.pth"
```

또는 한 번에:

```bash
source /Users/pjunese/Desktop/WATSON/img_guard/set_wam_env.sh
```

WAM 추론 시 의존성 설치:

```bash
pip install omegaconf==2.3.0 einops==0.8.0 opencv-python==4.10.0.84
```

WAM 리소스 배치(서비스 구조를 `img_guard` 내부로 고정):

```bash
cd /Users/pjunese/Desktop/WATSON/img_guard
mkdir -p third_party models/wam
git clone https://github.com/facebookresearch/watermark-anything.git third_party/watermark-anything
curl -L -o models/wam/wam_mit.pth https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth
```

---

## 실험 스크립트

특징점 매칭 + RANSAC (시각화 자동 열기):
```bash
cd /Users/pjunese/Desktop/WATSON/img_guard/experiments
python3 feature_match_ransac.py \
  --query ../data/db_images/dataset60/dt_028.png \
  --cand  ../data/db_images/dataset60/dt_028_1.png \
  --out /tmp/fm_out \
  --method orb \
  --open
```

회전 데이터 생성 (10~270도, 10도 간격):
```bash
cd /Users/pjunese/Desktop/WATSON/img_guard/experiments
python3 rotate_dataset.py \
  --input ../data/db_images/dataset60/dt_028.png \
  --out ../data/db_images/dataset60_rot_10_270
```

---

## 실험 결과

![res1](assets/res1.png)
![res2](assets/res2.png)
