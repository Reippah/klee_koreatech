#!/usr/bin/env python3
"""
SAM3 기반 배경 제거 스크립트

GitHub SAM3 (C:\\Users\\dongh\\Downloads\\sam3-main) API를 사용하여
텍스트 프롬프트로 지정한 사물만 남기고 나머지 배경을 제거합니다.

SAM3는 텍스트 기반 세그멘테이션을 지원하여 SAM2보다 정확한 배경 제거가 가능합니다.

사용법:
  py sam3_background_removal.py -i photo.jpg -p "person" -o result.png
  py sam3_background_removal.py -i photo.jpg -p "yellow car" -o result.png

필요 환경:
  - Python 3.12+, PyTorch 2.7+, CUDA 12.6+
  - SAM3 설치: cd C:\\Users\\dongh\\Downloads\\sam3-main && pip install -e ".[notebooks]"
  - Hugging Face 접근: huggingface-cli login (facebook/sam3 gated)
"""
import argparse
import os
import sys
from pathlib import Path

# SAM3 경로 추가
DEFAULT_SAM3_PATH = Path(r"C:\Users\dongh\Downloads\sam3-main")


def _setup_sam3_path():
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        pass
    if DEFAULT_SAM3_PATH.exists() and (DEFAULT_SAM3_PATH / "sam3").exists():
        sys.path.insert(0, str(DEFAULT_SAM3_PATH.resolve()))
        return True
    return False


if not _setup_sam3_path():
    print("SAM3를 찾을 수 없습니다. C:\\Users\\dongh\\Downloads\\sam3-main 에 설치하세요.")
    sys.exit(1)

import numpy as np
from PIL import Image


def remove_background_sam3(
    image_path: str,
    output_path: str,
    prompt: str = "person",
    confidence_threshold: float = 0.5,
):
    """SAM3로 텍스트 프롬프트 기반 배경 제거"""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("SAM3 모델 로드 중...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    print("이미지 처리 중...")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output.get("masks")  # [N, H, W] bool
    if masks is None or masks.numel() == 0:
        print("해당 프롬프트와 일치하는 객체가 없습니다.")
        # 전체 이미지를 foreground로 (배경 제거 없음)
        h, w = image.size[1], image.size[0]
        combined_mask = np.ones((h, w), dtype=np.uint8)
    else:
        # 모든 인스턴스 마스크 합침 (OR)
        masks_np = masks.cpu().numpy()
        combined_mask = (masks_np.sum(axis=0) > 0).astype(np.uint8)

    img_np = np.array(image)
    mask_3ch = np.stack([combined_mask, combined_mask, combined_mask], axis=-1)
    result = img_np * mask_3ch
    result_rgba = np.concatenate(
        [result, (combined_mask * 255).astype(np.uint8)[:, :, np.newaxis]],
        axis=2,
    )
    Image.fromarray(result_rgba).save(output_path, "PNG")
    print(f"저장됨: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3로 텍스트 프롬프트 기반 배경 제거"
    )
    parser.add_argument("-i", "--image", required=True, help="입력 이미지 경로")
    parser.add_argument("-o", "--output", default=None, help="출력 경로")
    parser.add_argument(
        "-p", "--prompt",
        default="person",
        help="남길 사물 (예: person, yellow car, dog)",
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.5,
        help="신뢰도 임계값 (0~1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"이미지 없음: {args.image}")
        sys.exit(1)

    output = args.output
    if not output:
        stem = Path(args.image).stem
        output = str(Path(args.image).parent / f"{stem}_sam3_bg_removed.png")

    remove_background_sam3(
        args.image,
        output,
        prompt=args.prompt,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
