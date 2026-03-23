# main.py
"""
main.py
- 로컬 CLI 테스트 실행기
- 사용 예:
  python3 main.py --query data/db_images/dataset60/dt_001.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.guard import GuardEngine
from app.types import GuardResult, ANNResult


def guardresult_to_dict(r: GuardResult) -> dict:
    def ann_to_dict(a: ANNResult) -> dict:
        return {
            "db_file": a.db_file,
            "cosine": round(float(a.cosine), 6),
            "phash_dist": None if a.phash_dist is None else int(a.phash_dist),
        }

    return {
        "decision": r.decision.value,
        "reason": r.reason,
        "top_match": None if r.top_match is None else ann_to_dict(r.top_match),
        "candidates": [ann_to_dict(c) for c in r.candidates],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="query image path")
    ap.add_argument("--k", type=int, default=None, help="Top-K for ANN search")
    ap.add_argument("--json", action="store_true", help="print as json")
    args = ap.parse_args()

    query_path = Path(args.query)
    if not query_path.exists():
        raise FileNotFoundError(f"query not found: {query_path}")

    engine = GuardEngine.create()
    result = engine.run(str(query_path), k=args.k if args.k is not None else 10)

    if args.json:
        print(json.dumps(guardresult_to_dict(result), ensure_ascii=False, indent=2))
        return

    # 사람 보기 좋은 출력
    print("\n=== IMG_GUARD RESULT ===")
    print("decision:", result.decision.value)
    print("reason  :", result.reason)

    if result.top_match:
        tm = result.top_match
        print("\nTop match:")
        print(f"  file={tm.db_file} | cos={tm.cosine:.4f} | phash={tm.phash_dist}")

    print("\nCandidates:")
    for i, c in enumerate(result.candidates, start=1):
        print(f"  #{i:02d} {c.db_file} | cos={c.cosine:.4f} | phash={c.phash_dist}")


if __name__ == "__main__":
    main()