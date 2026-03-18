#!/usr/bin/env python3
"""
SAM2 기반 배경 제거 스크립트

GitHub SAM2 (C:\\Users\\dongh\\Downloads\\sam2-main) API를 사용하여
원하는 사물만 남기고 나머지 배경을 제거합니다.

사용법:
  py sam2_background_removal.py --image <이미지경로> [--prompt <사물명>] [--output <출력경로>]
  py sam2_background_removal.py -i photo.jpg -p person -o result.png

필요 환경:
  - SAM2 설치: cd C:\\Users\\dongh\\Downloads\\sam2-main\\sam2-main && pip install -e ".[notebooks]"
  - 또는: pip install 'git+https://github.com/facebookresearch/sam2.git'
  - PyTorch, torchvision, PIL, numpy
"""
import argparse
import os
import sys
from pathlib import Path

# SAM2 경로 추가 (pip 미설치 시 로컬 레포 사용)
def _setup_sam2_path():
    try:
        import sam2  # noqa: F401
        return  # 이미 설치됨
    except ImportError:
        pass
    default = Path(r"C:\Users\dongh\Downloads\sam2-main")
    for p in [default / "sam2-main", default]:
        if p.exists() and (p / "sam2").exists():
            sys.path.insert(0, str(p.resolve()))
            return


_setup_sam2_path()
DEFAULT_SAM2_PATH = r"C:\Users\dongh\Downloads\sam2-main"

import numpy as np
import torch
from PIL import Image


# COCO 80 클래스 (torchvision detection) - 한/영 매핑
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

PROMPT_TO_COCO = {
    "사람": 0, "person": 0, "인물": 0,
    "병": 39, "bottle": 39, "캔": 39, "can": 39, "컵": 41, "cup": 41,
    "자동차": 2, "car": 2, "차": 2,
    "의자": 56, "chair": 56,
    "개": 16, "dog": 16,
    "고양이": 15, "cat": 15,
    "식물": 57, "potted plant": 57, "화분": 57,
    "주요": 0, "물건": 0, "객체": 0, "object": 0,
}


def get_detection_model(device):
    """torchvision COCO detection 모델 로드"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    return model.to(device)


def detect_object(image_np, prompt: str, device, score_thresh=0.5):
    """
    이미지에서 프롬프트에 해당하는 객체 bbox 반환 (xyxy)
    없으면 None
    """
    model = get_detection_model(device)
    target_idx = PROMPT_TO_COCO.get(prompt.strip().lower(), 0)

    # HWC, 0-255
    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    boxes = preds[0]["boxes"].cpu().numpy()
    labels = preds[0]["labels"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if label - 1 == target_idx and score >= score_thresh:
            return box  # xyxy

    # 매칭 실패 시 가장 큰 person bbox 또는 이미지 중심
    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score >= 0.3:  # person
            return box

    return None


def remove_background_sam2(
    image_path: str,
    output_path: str,
    prompt: str = "person",
    sam2_path: str = None,
    use_hf: bool = True,
):
    """
    SAM2로 배경 제거 후 저장

    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        prompt: 남길 사물 (person, car, bottle 등)
        sam2_path: SAM2 레포 경로 (None이면 HuggingFace 사용)
        use_hf: True면 HuggingFace에서 모델 로드
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # SAM2 predictor 로드
    if use_hf:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("SAM2 모델 로드 중 (HuggingFace facebook/sam2.1-hiera-tiny)...")
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-tiny",
            device=str(device),
        )
    else:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_root = Path(sam2_path or DEFAULT_SAM2_PATH)
        if (sam2_root / "sam2-main").exists():
            sam2_root = sam2_root / "sam2-main"
        ckpt = sam2_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"체크포인트 없음: {ckpt}\n"
                "checkpoints/download_ckpts.sh 실행 또는 HuggingFace 사용 (--hf)"
            )
        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=str(ckpt),
            device=str(device),
        )
        predictor = SAM2ImagePredictor(model)

    predictor.set_image(image_np)

    # 객체 감지 → bbox 또는 포인트
    box = detect_object(image_np, prompt, device)

    if box is not None:
        # box: xyxy → SAM2 형식
        box_xyxy = np.array(box, dtype=np.float32)
        masks, iou_pred, _ = predictor.predict(
            box=box_xyxy,
            multimask_output=False,
            normalize_coords=False,
        )
    else:
        # 이미지 중심을 포인트로 사용
        cx, cy = w / 2, h / 2
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1=foreground
        masks, iou_pred, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=False,
        )
        # IoU 가장 높은 마스크 선택
        best_idx = np.argmax(iou_pred[0])
        masks = masks[best_idx : best_idx + 1]

    mask = masks[0] > 0.0  # binary
    mask_3ch = np.stack([mask, mask, mask], axis=-1)

    # 배경 제거 (투명 또는 흰 배경)
    result = image_np.astype(np.uint8) * mask_3ch
    # 투명 PNG: RGBA
    result_rgba = np.concatenate(
        [result, (mask * 255).astype(np.uint8)[:, :, np.newaxis]],
        axis=2,
    )
    Image.fromarray(result_rgba).save(output_path, "PNG")
    print(f"저장됨: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM2로 원하는 사물만 남기고 배경 제거"
    )
    parser.add_argument("-i", "--image", required=True, help="입력 이미지 경로")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="출력 경로 (기본: 입력파일명_bg_removed.png)",
    )
    parser.add_argument(
        "-p", "--prompt",
        default="person",
        help="남길 사물 (person, car, bottle, chair 등)",
    )
    parser.add_argument(
        "--sam2-path",
        default=DEFAULT_SAM2_PATH,
        help="SAM2 레포 경로 (로컬 체크포인트 사용 시)",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        default=True,
        help="HuggingFace에서 모델 로드 (기본)",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="로컬 체크포인트 사용",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"이미지 없음: {args.image}")
        sys.exit(1)

    output = args.output
    if not output:
        stem = Path(args.image).stem
        output = str(Path(args.image).parent / f"{stem}_bg_removed.png")

    use_hf = args.hf and not args.no_hf
    remove_background_sam2(
        args.image,
        output,
        prompt=args.prompt,
        sam2_path=args.sam2_path,
        use_hf=use_hf,
    )


if __name__ == "__main__":
    main()
