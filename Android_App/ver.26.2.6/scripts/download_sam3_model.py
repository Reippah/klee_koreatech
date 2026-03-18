#!/usr/bin/env python3
"""
SAM3 (Segment Anything Model 3) 모델 다운로드 스크립트

Meta의 facebook/sam3 모델을 Hugging Face에서 다운로드합니다.
- 저장 위치: models/sam3/ (프로젝트 루트)

⚠️ 사전 요구사항:
1. Hugging Face에서 facebook/sam3 접근 권한 요청
   https://huggingface.co/facebook/sam3
2. 인증: huggingface-cli login 또는 HF_TOKEN 환경변수 설정
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SAM3_DIR = PROJECT_ROOT / "models" / "sam3"
REPO_ID = "facebook/sam3"

# 다운로드할 파일 (전체 모델 ~3.3GB - 선택적 다운로드 가능)
FILES_TO_DOWNLOAD = [
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    # "model.safetensors",  # ~3.3GB - 필요시 주석 해제
    # "sam3.pt",            # ~3.2GB - 필요시 주석 해제
]


def main():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub 패키지가 필요합니다.")
        print("설치: pip install huggingface_hub")
        sys.exit(1)

    SAM3_DIR.mkdir(parents=True, exist_ok=True)

    print(f"SAM3 모델 다운로드: {REPO_ID}")
    print(f"저장 위치: {SAM3_DIR}")
    print()

    try:
        # config, tokenizer 등 작은 파일만 먼저 다운로드 (모델 없이도 구조 확인 가능)
        for filename in FILES_TO_DOWNLOAD:
            try:
                path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir=SAM3_DIR,
                )
                print(f"  ✓ {filename}")
            except Exception as e:
                if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                    print("\n[권한 필요] Hugging Face에서 facebook/sam3 접근 권한을 요청하세요.")
                    print("  https://huggingface.co/facebook/sam3")
                    print("  인증: huggingface-cli login")
                    sys.exit(1)
                print(f"  ✗ {filename}: {e}")

        # 전체 모델 다운로드 (선택)
        download_full = os.environ.get("SAM3_DOWNLOAD_FULL", "").lower() in ("1", "true", "yes")
        if download_full:
            print("\n전체 모델 다운로드 중 (model.safetensors, sam3.pt - 약 6.5GB)...")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=SAM3_DIR,
            )
            print("전체 모델 다운로드 완료.")
        else:
            print("\n[참고] 전체 모델(model.safetensors, sam3.pt)은 ~6.5GB입니다.")
            print("SAM3_DOWNLOAD_FULL=1 환경변수로 전체 다운로드: SAM3_DOWNLOAD_FULL=1 python download_sam3_model.py")

        print(f"\n완료. SAM3 파일이 {SAM3_DIR} 에 저장되었습니다.")
        print("\n※ SAM3는 PyTorch 모델입니다. Android 앱의 온디바이스 배경 제거는")
        print("  MediaPipe DeepLab v3 (scripts/download_segmentation_models.ps1)를 사용합니다.")

    except Exception as e:
        print(f"다운로드 실패: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\nHugging Face 접근 권한 요청: https://huggingface.co/facebook/sam3")
            print("인증: huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    main()
