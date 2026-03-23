# watermark 서비스 구조

## 디렉토리
- `app/watermark/models.py`: 요청/응답 모델 + 내부 결과 타입
- `app/watermark/router.py`: FastAPI 엔드포인트
- `app/watermark/service.py`: 공통 오케스트레이션(입력 해석, payload 생성, backend 호출)
- `app/watermark/storage.py`: `url`/`local_path` 입력을 로컬 파일로 해석
- `app/watermark/payload.py`: 메타 기반 payload bit/payload_id 생성
- `app/watermark/backends/base.py`: 백엔드 인터페이스
- `app/watermark/backends/mock_backend.py`: 계약 테스트용 백엔드
- `app/watermark/backends/wam_backend.py`: 실제 WAM 추론 백엔드

## 현재 구현 범위
- API 경로 고정
  - `POST /v1/watermark/embed`
  - `POST /v1/watermark/detect`
- `WM_BACKEND=mock`로 로컬 엔드-투-엔드 테스트 가능
- `WM_BACKEND=wam`는 실제 WAM 추론(삽입/검출) 구현 완료
  - 삽입: `payload_bits`를 메시지로 삽입 후 결과 이미지 저장
  - 검출: mask confidence + 메시지 디코딩 결과 반환

## 실행 전 필수
1. WAM 의존성 설치
   - `pip install omegaconf==2.3.0 einops==0.8.0 opencv-python==4.10.0.84`
2. MIT 가중치 배치
   - `/Users/pjunese/Desktop/WATSON/img_guard/models/wam/wam_mit.pth`
3. WAM repo 배치
   - `/Users/pjunese/Desktop/WATSON/img_guard/third_party/watermark-anything`

```bash
cd /Users/pjunese/Desktop/WATSON/img_guard
mkdir -p third_party models/wam
git clone https://github.com/facebookresearch/watermark-anything.git third_party/watermark-anything
curl -L -o models/wam/wam_mit.pth https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth
source /Users/pjunese/Desktop/WATSON/img_guard/set_wam_env.sh
```

## 다음 단계
1. 출력 이미지 S3 업로드 및 `output_url`/`output_key` 채우기
2. 검출 결과와 DB(token/NFT 메타데이터) 조인 로직 추가
3. 검출 confidence 판정 룰(임계값) 운영 데이터로 튜닝
