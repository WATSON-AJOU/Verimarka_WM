# IMG_GUARD 계약서 V1 (함수 매핑 우선) - 최종안

작성일: 2026-03-13
대상: 백엔드(Django/Celery) + AI(img_guard) 연동

## 0. 전제 (백엔드개발 문서 반영)
- 통신 우선순위: **함수 매핑 호출** (같은 레포/같은 런타임에서 import 호출)
- API(FastAPI): 로컬 디버깅/회귀 테스트 용도로 유지
- 인프라 방향:
  - DB: PostgreSQL + pgvector
  - 스토리지: S3 (presigned URL 입력)
  - 비동기: Celery + Redis
  - 운영: idempotency 필요

---

## 1. 동기/비동기 경계 (고정)
- 동기(요청 즉시): `run_guard_v1`로 `allow/review/block` 판정
- 비동기(후속 작업):
  - `allow`일 때만 워터마크 삽입 작업 enqueue
  - `review`일 때는 블록체인 투표 워크플로우로 전달

`next_action` 규칙:
- `decision == "review"` -> `next_action = "start_vote"`
- 그 외(`allow`, `block`) -> `next_action = "none"`

---

## 2. 함수 호출 인터페이스 (백엔드가 실제로 쓰는 것)

## 2-1. 등록 판정 (현재 구현 완료)
```python
from app.guard_service import run_guard_v1

resp = run_guard_v1(request_dict)
result = resp.model_dump()
```

- 입력 스키마: `GuardRequestV1` (`app/contracts_v1.py`)
- 출력 스키마: `GuardResponseV1` (`app/contracts_v1.py`)

## 2-2. 워터마크 삽입 (현재 구현 완료)
```python
from app.watermark.service import WatermarkService
from app.watermark.models import WatermarkEmbedRequest, MediaInput, WatermarkEmbedOptions

wm = WatermarkService.create()
embed_resp = wm.embed(
    WatermarkEmbedRequest(
        job_id="uuid",
        input=MediaInput(url="https://presigned-url"),
        meta={"user_id": "u", "content_id": "c"},
        options=WatermarkEmbedOptions(
            model="wam",
            nbits=32,
            scaling_w=2.0,
            proportion_masked=0.35,
            seed=42,
        ),
    )
)
```

## 2-3. 워터마크 검출 (현재 구현 완료)
```python
from app.watermark.service import WatermarkService
from app.watermark.models import WatermarkDetectRequest, MediaInput, WatermarkDetectOptions

wm = WatermarkService.create()
detect_resp = wm.detect(
    WatermarkDetectRequest(
        job_id="uuid",
        input=MediaInput(url="https://presigned-url"),
        options=WatermarkDetectOptions(model="wam", threshold=0.5),
    )
)
```

---

## 3. 요청 계약 (서버 -> AI) V1

```json
{
  "job_id": "uuid",
  "mode": "register",
  "content_type": "image",
  "input": [
    {
      "url": "https://presigned-url-to-original-file",
      "filename": "dt_001.png",
      "mime_type": "image/png"
    }
  ],
  "meta": {
    "user_id": "user_123",
    "content_id": "content_987"
  },
  "options": {
    "search": {
      "top_k": 10,
      "top_phash": 10
    },
    "watermark": {
      "apply_on_allow": true,
      "model": "wam",
      "nbits": 32,
      "scaling_w": 2.0,
      "proportion_masked": 0.35
    }
  }
}
```

필드 규칙:
- `mode`: 현재 `register`만 사용
- `content_type`: 현재 `image`만 허용
- `input`: 배열 형식 고정(원소 1개 권장)
- `options.search.top_k`: 기본 10
- `options.search.top_phash`: 기본 10
- `options.watermark.nbits`: 현재 서비스 가중치 기준 32 고정 권장
- `proportion_masked`: 0.0 ~ 1.0

---

## 4. 성공 응답 계약 (AI -> 서버) V1

```json
{
  "job_id": "uuid",
  "mode": "register",
  "content_type": "image",
  "success": true,
  "decision": "allow",
  "reason": "No strong near-duplicate found",
  "next_action": "none",
  "scores": {
    "top_cosine": 0.8421,
    "top_phash_dist": 28,
    "policy_version": "v1"
  },
  "top_match": {
    "db_key": "db/dataset60/dt_031.png",
    "db_file": "dt_031.png",
    "cosine": 0.8421,
    "phash_dist": 28
  },
  "candidates": [
    {"db_key": "db/dataset60/dt_031.png", "db_file": "dt_031.png", "cosine": 0.8421, "phash_dist": 28},
    {"db_key": "db/dataset60/dt_010.png", "db_file": "dt_010.png", "cosine": 0.8123, "phash_dist": 34}
  ],
  "watermark": {
    "requested": true,
    "applied": false,
    "output_url": null,
    "output_key": null,
    "model": "wam",
    "model_version": null,
    "nbits": 32,
    "scaling_w": 2.0,
    "proportion_masked": 0.35,
    "payload_id": null
  },
  "timing_ms": {
    "download": 120,
    "embed": 640,
    "ann_search": 6,
    "phash": 35,
    "total": 801
  }
}
```

주의:
- guard 단계에서는 워터마크 실제 삽입을 수행하지 않으므로 보통 `watermark.applied=false`
- 삽입 결과(`output_url/output_key/payload_id`)는 **후속 비동기 워크플로우 결과**에서 채움

---

## 5. 실패 처리 규약 (함수 매핑 기준)

함수 매핑에서는 예외를 Django가 잡아 표준 응답으로 변환:
- `ValueError` -> 400 계열 (입력 오류)
- `RuntimeError`/기타 -> 500 계열 (시스템/의존성 오류)

권장 실패 응답 형식(백엔드 외부 API에 노출 시):
```json
{
  "job_id": "uuid",
  "success": false,
  "error_code": "INVALID_INPUT",
  "error_message": "input.url is required",
  "retryable": false
}
```

---

## 6. idempotency 권장안
- 키: `content_id` + 파일 해시(또는 URL 서명 일부) + `mode`
- 동일 키 재요청 시:
  - 이미 완료된 결과 있으면 재사용
  - 진행 중이면 중복 enqueue 차단

---

## 7. 백엔드-DB 합의 필수 항목
- pgvector 컬럼명 최종 확정:
  - `id`, `embedding`, `file_name`, `s3_key`, `asset_url`, `phash`
- pHash 저장 타입:
  - `BIGINT` 또는 `HEX STRING` 중 하나로 고정
- ANN backend 운영값:
  - `ANN_BACKEND=pgvector`
- S3 접근 방식:
  - 입력: 입력: AI worker가 s3://bucket/key(또는 bucket,key)를 받아 IAM Role로 직접 다운로드
  - 출력: AI worker가 결과를 S3에 업로드하고 output_key 반환 (output_url은 백엔드가 필요 시 presigned로 생성)

---

## 8. 즉시 실행 체크리스트
1. `source /Users/pjunese/Desktop/WATSON/img_guard/set_wam_env.sh`
2. Django에서 `run_guard_v1` 호출 스모크 테스트
3. `allow` 시 Celery 태스크로 `WatermarkService.embed` 호출
4. 결과 S3 업로드 + DB 기록 + 응답 필드 채우기

