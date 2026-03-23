"""
AI 이미지 중복/유사도 판단 파이프라인에서
각 단계가 주고받는 데이터 구조(type)를 정의한다.

- ANN 검색 결과
- 정책 판정 상태 (3-state)
- 최종 Guard 판정 결과



@dataclass 쓰면
__init__() 자동 생성
__repr__() (print 했을 때 보기 좋음)
== 비교 가능
타입 힌트 정리
"""
from dataclasses import dataclass
from enum import Enum



# ANN 검색 결과 단일 항목
"""
ANN(HNSW) 검색으로 얻은 DB 이미지 1개에 대한 정보
- cosine: CLIP 임베딩 코사인 유사도 (0~1)
- phash_dist: pHash 해밍 거리
    (ANN 단계에서는 None, pHash 계산 후 채워짐)
"""

@dataclass
class ANNResult:
    db_file: str            # DB 이미지 파일명
    cosine: float           # 코사인 유사도
    db_key: str | None = None     # DB key (S3 key 등), 선택
    db_phash: int | str | None = None  # DB에 저장된 pHash (int or hex str)
    phash_dist: int | None = None     # Union[int, None] 동일 즉, 정수이거나(None)


# 정책 판정 상태 (3-state)
"""
이미지 등록/검증 결과에 대한 최종 상태

- ALLOW : 중복/유사 이미지 없음 -> 승인
- REVIEW: 애매한 케이스 -> 보류(관리자/후속 로직)
- BLOCK : 기존 이미지와 사실상 동일 -> 차단
"""

class Decision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"



# 최종 Guard 결과
"""
img_guard 엔진의 최종 출력 결과
- decision : 3상태 정책 결과
- reason   : 사람이 이해 가능한 판정 근거 설명
- top_match: 가장 유사한 DB 이미지 (없으면 None)
- candidates:
    ANN + pHash 기준으로 수집된 후보 이미지들
    (디버깅 / 시각화 / 로그 / 분쟁 대응용)
"""

@dataclass
class GuardResult:
    decision: Decision
    reason: str
    top_match: ANNResult | None
    candidates: list[ANNResult]
