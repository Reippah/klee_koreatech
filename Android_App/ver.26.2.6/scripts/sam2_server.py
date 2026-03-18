#!/usr/bin/env python3
"""
SAM2 배경제거 Flask 서버

Android 앱에서 이미지와 텍스트 프롬프트를 수신하여
SAM2(Segment Anything Model 2) + Faster R-CNN 으로 배경을 제거하고
투명 PNG를 반환합니다.

동작 방식:
  1. Faster R-CNN(torchvision)으로 텍스트 프롬프트에 해당하는 객체의 BBox 검출
  2. BBox(또는 이미지 중심 포인트)를 SAM2에 전달해 정밀 세그멘테이션 마스크 생성
  3. 마스크를 원본 이미지에 적용 → 배경 투명(RGBA) PNG 반환

실행 전 준비:
  pip install flask torch torchvision pillow numpy
  pip install 'git+https://github.com/facebookresearch/sam2.git'
  또는:
    cd C:\\Users\\dongh\\Downloads\\sam2-main
    pip install -e .

  # SAM2 모델은 HuggingFace에서 자동 다운로드됩니다 (첫 실행 시)
  # 수동 다운로드 원할 경우:
  #   cd C:\\Users\\dongh\\Downloads\\sam2-main && bash checkpoints/download_ckpts.sh

실행:
  py sam2_server.py
  py sam2_server.py --port 8001 --preload

엔드포인트:
  GET  /health     — 서버 상태 + 모델 로드 여부
  POST /bg-remove  — image(file) + prompt(str) → 투명 PNG
"""

import argparse
import io
import sys
import time
from pathlib import Path

# ── SAM2 경로 설정 ───────────────────────────────────────────────────────────
DEFAULT_SAM2_PATH = Path(r"C:\Users\dongh\Downloads\sam2-main")


def _setup_sam2_path() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        pass
    for candidate in [DEFAULT_SAM2_PATH, DEFAULT_SAM2_PATH / "sam2-main"]:
        if candidate.exists() and (candidate / "sam2").exists():
            sys.path.insert(0, str(candidate.resolve()))
            try:
                import sam2  # noqa: F401
                return True
            except ImportError:
                pass
    return False


if not _setup_sam2_path():
    print(
        "[오류] SAM2를 찾을 수 없습니다.\n"
        f"  경로 확인: {DEFAULT_SAM2_PATH}\n"
        "  설치: pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
        "  또는: cd <sam2-main> && pip install -e ."
    )
    sys.exit(1)

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

app = Flask(__name__)

# 전역 예측기 (lazy-load)
_predictor = None
_det_model = None
_device: str = "cpu"
_load_error: str | None = None

# ── COCO 80 클래스 텍스트 매핑 ───────────────────────────────────────────────
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

PROMPT_ALIASES: dict[str, str] = {
    "can": "bottle", "mug": "cup", "sofa": "couch",
    "monitor": "tv", "display": "tv", "phone": "cell phone",
    "plant": "potted plant", "desk": "dining table",
}


def resolve_coco_index(prompt: str) -> int | None:
    """텍스트 프롬프트 → COCO 클래스 인덱스(0-based) 반환. 없으면 None."""
    p = prompt.strip().lower()
    p = PROMPT_ALIASES.get(p, p)
    for i, cls in enumerate(COCO_CLASSES):
        if cls == p or p in cls:
            return i
    return None


# ── 모델 초기화 ──────────────────────────────────────────────────────────────

def _load_models():
    global _predictor, _det_model, _device, _load_error
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2 서버] 디바이스: {_device}")

        # SAM2 로드 (HuggingFace 자동 다운로드)
        print("[SAM2 서버] SAM2 모델 로딩 중 (facebook/sam2.1-hiera-small)...")
        t0 = time.time()
        _predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small",
            device=_device,
        )
        print(f"[SAM2 서버] SAM2 로드 완료 ({time.time() - t0:.1f}초)")

        # Faster R-CNN 로드 (COCO 사전학습)
        print("[SAM2 서버] Faster R-CNN 로딩 중...")
        t0 = time.time()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        _det_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        _det_model.eval()
        _det_model.to(_device)
        print(f"[SAM2 서버] Faster R-CNN 로드 완료 ({time.time() - t0:.1f}초)")

    except Exception as e:
        _load_error = str(e)
        print(f"[SAM2 서버] 모델 로드 실패: {e}")
        raise


def get_models():
    if _predictor is None and _load_error is None:
        _load_models()
    if _load_error:
        raise RuntimeError(_load_error)
    return _predictor, _det_model


# ── 객체 감지 (Faster R-CNN) ────────────────────────────────────────────────

def detect_bbox(image_np: np.ndarray, prompt: str, score_thresh: float = 0.5):
    """
    Faster R-CNN으로 프롬프트에 해당하는 객체의 BBox(xyxy) 반환.
    매칭 없으면 None.
    """
    import torch

    target_idx = resolve_coco_index(prompt)

    img_tensor = (
        torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    ).unsqueeze(0).to(_device)

    with torch.no_grad():
        preds = _det_model(img_tensor)

    boxes = preds[0]["boxes"].cpu().numpy()     # [N, 4] xyxy
    labels = preds[0]["labels"].cpu().numpy()   # [N] 1-indexed COCO
    scores = preds[0]["scores"].cpu().numpy()   # [N]

    best_box = None
    best_score = -1.0

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        coco_idx = int(label) - 1  # 0-based
        # 프롬프트 클래스와 일치하거나 프롬프트가 없으면 가장 확률 높은 것
        if target_idx is not None and coco_idx != target_idx:
            continue
        if score > best_score:
            best_score = float(score)
            best_box = box

    # 낮은 임계값으로 재시도
    if best_box is None and target_idx is not None:
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.3:
                continue
            coco_idx = int(label) - 1
            if coco_idx == target_idx and score > best_score:
                best_score = float(score)
                best_box = box

    return best_box


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _predictor is not None,
        "device": _device,
        "error": _load_error,
    })


@app.route("/bg-remove", methods=["POST"])
def bg_remove():
    """
    배경 제거 엔드포인트.

    요청 (multipart/form-data):
      image  : 이미지 파일 (jpg / png / webp 등)
      prompt : 남길 사물 영문 텍스트 (예: "cup", "mouse")

    응답:
      200 : 투명 배경 PNG (image/png)
      400 : 파라미터 누락
      500 : 처리 오류 (JSON)
    """
    if "image" not in request.files:
        return jsonify({"error": "'image' 파일 필드가 없습니다."}), 400

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "'prompt' 텍스트 필드가 없습니다."}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"이미지 로드 실패: {e}"}), 400

    try:
        predictor, _ = get_models()
    except RuntimeError as e:
        return jsonify({"error": f"모델 로드 실패: {e}"}), 500

    try:
        import torch

        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # ── 1. BBox 검출 ────────────────────────────────────────────────────
        box = detect_bbox(image_np, prompt)
        print(f"[SAM2 서버] 프롬프트='{prompt}', BBox={'검출됨' if box is not None else '없음 → 중심 포인트 사용'}")

        # ── 2. SAM2 세그멘테이션 ────────────────────────────────────────────
        with torch.inference_mode(), torch.autocast(_device, dtype=torch.bfloat16):
            predictor.set_image(image_np)

            if box is not None:
                box_arr = np.array(box, dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    box=box_arr,
                    multimask_output=False,
                    normalize_coords=False,
                )
                mask = masks[0] > 0.0
            else:
                # BBox 없으면 이미지 중심 포인트 사용
                cx, cy = w / 2.0, h / 2.0
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                    normalize_coords=False,
                )
                best_idx = int(np.argmax(scores[0]))
                mask = masks[best_idx] > 0.0

        # ── 3. 투명 PNG 합성 ────────────────────────────────────────────────
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result_rgb = (image_np * mask_3ch).astype(np.uint8)
        alpha_ch = (mask.astype(np.uint8) * 255)[:, :, np.newaxis]
        result_rgba = np.concatenate([result_rgb, alpha_ch], axis=2)

        out_buf = io.BytesIO()
        Image.fromarray(result_rgba, mode="RGBA").save(out_buf, format="PNG")
        out_buf.seek(0)
        return send_file(out_buf, mimetype="image/png", as_attachment=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"처리 실패: {e}"}), 500


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 배경제거 Flask 서버")
    parser.add_argument("--port", type=int, default=8001, help="수신 포트 (기본: 8001)")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 주소 (기본: 0.0.0.0)")
    parser.add_argument(
        "--preload", action="store_true",
        help="서버 시작 시 모델을 미리 로드 (첫 요청 지연 방지)"
    )
    args = parser.parse_args()

    if args.preload:
        try:
            _load_models()
        except Exception as e:
            print(f"[경고] 사전 로드 실패: {e}")

    print(f"\n[SAM2 서버] http://{args.host}:{args.port} 에서 실행 중")
    print(f"  POST /bg-remove  — image(file) + prompt(str) → 투명 PNG")
    print(f"  GET  /health     — 서버 상태 확인\n")
    # threaded=False: SAM2 + PyTorch는 단일 스레드 처리 권장
    app.run(host=args.host, port=args.port, threaded=False)
