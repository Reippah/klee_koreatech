#!/usr/bin/env python3
"""
SAM3 배경제거 Flask 서버

Android 앱에서 이미지와 텍스트 프롬프트를 수신하여
SAM3(Segment Anything with Concepts)로 배경을 제거한 투명 PNG를 반환합니다.

실행 전 준비:
  1. SAM3 설치:
       cd C:\Users\dongh\Downloads\sam3-main
       pip install -e ".[notebooks]"
  2. HuggingFace 인증 (facebook/sam3 gated repo):
       hf auth login
  3. Flask 설치:
       pip install flask

실행:
  py sam3_server.py
  py sam3_server.py --port 8001 --preload

엔드포인트:
  GET  /health      - 서버 상태 + 모델 로드 여부 확인
  POST /bg-remove   - image(multipart file) + prompt(form field) → 투명 PNG
"""

import argparse
import io
import sys
import time
from pathlib import Path

# ── SAM3 경로 설정 ──────────────────────────────────────────────────────────
DEFAULT_SAM3_PATH = Path(r"C:\Users\dongh\Downloads\sam3-main")


def _setup_sam3_path() -> bool:
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        pass
    if DEFAULT_SAM3_PATH.exists() and (DEFAULT_SAM3_PATH / "sam3").exists():
        sys.path.insert(0, str(DEFAULT_SAM3_PATH.resolve()))
        try:
            import sam3  # noqa: F401
            return True
        except ImportError:
            pass
    return False


if not _setup_sam3_path():
    print(
        "[오류] SAM3를 찾을 수 없습니다.\n"
        f"  경로 확인: {DEFAULT_SAM3_PATH}\n"
        "  설치: cd <sam3-main> && pip install -e ."
    )
    sys.exit(1)

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

app = Flask(__name__)

# 전역 SAM3 프로세서 (최초 요청 시 또는 --preload 시 로드)
_processor = None
_processor_load_error: str | None = None


def get_processor():
    """SAM3 프로세서를 lazy-load (첫 호출 시 모델 로드)."""
    global _processor, _processor_load_error
    if _processor is not None:
        return _processor
    if _processor_load_error:
        raise RuntimeError(_processor_load_error)

    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM3 서버] 모델 로딩 중... (device={device})")
        t0 = time.time()
        model = build_sam3_image_model(device=device)
        _processor = Sam3Processor(model)
        print(f"[SAM3 서버] 모델 로드 완료 ({time.time() - t0:.1f}초)")
        return _processor
    except Exception as e:
        _processor_load_error = str(e)
        raise


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """서버 + 모델 상태 확인용 엔드포인트."""
    return jsonify({
        "status": "ok",
        "model_loaded": _processor is not None,
        "model_error": _processor_load_error,
    })


@app.route("/bg-remove", methods=["POST"])
def bg_remove():
    """
    배경 제거 엔드포인트.

    요청 (multipart/form-data):
      image  : 이미지 파일 (jpg / png / webp 등)
      prompt : 남길 사물의 영문 텍스트 (예: "cup", "mouse")

    응답:
      200 OK  : 투명 배경 PNG (image/png)
      400     : 파라미터 누락
      500     : 처리 오류 (JSON { "error": "..." })
    """
    # ── 입력 검증 ──────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "'image' 파일 필드가 없습니다."}), 400

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "'prompt' 텍스트 필드가 없습니다."}), 400

    # ── 이미지 로드 ────────────────────────────────────────────────────────
    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"이미지 로드 실패: {e}"}), 400

    # ── SAM3 추론 ──────────────────────────────────────────────────────────
    try:
        proc = get_processor()
    except RuntimeError as e:
        return jsonify({"error": f"모델 로드 실패: {e}"}), 500

    try:
        state = proc.set_image(image)
        output = proc.set_text_prompt(state=state, prompt=prompt)
    except Exception as e:
        return jsonify({"error": f"SAM3 추론 실패: {e}"}), 500

    # ── 마스크 합성 ────────────────────────────────────────────────────────
    try:
        masks = output.get("masks")  # shape: [N, H, W] (bool tensor)
        if masks is None or masks.numel() == 0:
            # 해당 객체를 찾지 못한 경우: 전체를 foreground로 처리
            h, w = image.size[1], image.size[0]
            combined_mask = np.ones((h, w), dtype=np.uint8)
        else:
            # N개의 인스턴스 마스크를 OR 합성 → 단일 이진 마스크
            masks_np = masks.cpu().numpy()  # [N, H, W]
            combined_mask = (masks_np.sum(axis=0) > 0).astype(np.uint8)

        img_np = np.array(image)
        mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        result_rgb = (img_np * mask_3ch).astype(np.uint8)
        alpha_ch = (combined_mask * 255).astype(np.uint8)[:, :, np.newaxis]
        result_rgba = np.concatenate([result_rgb, alpha_ch], axis=2)

        out_buf = io.BytesIO()
        Image.fromarray(result_rgba, mode="RGBA").save(out_buf, format="PNG")
        out_buf.seek(0)
        return send_file(out_buf, mimetype="image/png", as_attachment=False)

    except Exception as e:
        return jsonify({"error": f"마스크 처리 실패: {e}"}), 500


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 배경제거 Flask 서버")
    parser.add_argument("--port", type=int, default=8001, help="수신 포트 (기본: 8001)")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 주소 (기본: 0.0.0.0)")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="서버 시작 시 SAM3 모델을 미리 로드 (첫 요청 지연 방지)",
    )
    args = parser.parse_args()

    if args.preload:
        try:
            get_processor()
        except Exception as e:
            print(f"[경고] 모델 사전 로드 실패: {e}")

    print(f"\n[SAM3 서버] http://{args.host}:{args.port} 에서 실행 중")
    print(f"  POST /bg-remove  — image(file) + prompt(str) → 투명 PNG")
    print(f"  GET  /health     — 서버 상태 확인\n")
    # threaded=False: SAM3는 멀티스레드 추론 시 충돌 가능
    app.run(host=args.host, port=args.port, threaded=False)
