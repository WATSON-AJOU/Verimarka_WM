"""
preprocess.py
- 이미지 로딩 + 표준화(orientation/RGB/size) 담당
- 코랩에서 하던 ImageOps.exif_transpose 로테이션 보정 포함

원칙:
1) 입력 이미지가 어떤 포맷이든(PNG/JPG/WebP, RGBA 포함) -> RGB로 통일
2) EXIF orientation(회전/뒤집힘) 자동 보정
3) 임베딩 모델이 받기 좋은 형태로 PIL Image 반환
"""

from __future__ import annotations
from PIL import Image, ImageOps

"""
(path: str) path는 문자열일것으로 기대한다.
-> Image.Image는 "이 함수는 Image.Image 타입을 반환한다" 라는 뜻
"""
def load_image_fixed(path: str) -> Image.Image:
    """
    파일 경로 -> PIL.Image 로드 후 표준화
    - EXIF orientation 보정
    - RGBA/LA 등 알파 채널은 흰색 배경에 합성
    - 최종 RGB 보장
    """
    img = Image.open(path)

    # 1) EXIF orientation(회전/flip) 보정
    img = ImageOps.exif_transpose(img)

    # 2) 색공간/알파 처리
    if img.mode in ("RGBA", "LA"):
        # 알파 채널이 있는 경우: 흰 배경에 합성
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def to_rgb(img: Image.Image) -> Image.Image:
    """
    PIL Image 객체를 RGB로 통일
    (메모리 상에서 생성된 이미지에도 사용 가능)
    """
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img