#!/usr/bin/env python3
"""
SAM2 Hiera-Tiny (Qualcomm) TFLite 모델 다운로드 스크립트

Qualcomm SAM2는 sam2.1_hiera_t (Hiera-Tiny) 아키텍처를 사용하며,
SAM3와 유사한 온디바이스 세그멘테이션을 제공합니다.

다운로드 후 app/src/main/assets/models/sam2/ 에 저장됩니다.
"""
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Qualcomm SAM2 TFLite (w8a8 양자화, 범용 칩셋)
SAM2_TFLITE_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/sam2/releases/v0.47.0/sam2-tflite-w8a8.zip"

# 프로젝트 루트 (스크립트 기준)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_MODELS = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "models"
SAM2_DIR = ASSETS_MODELS / "sam2"


def download_file(url: str, dest: Path) -> bool:
    """URL에서 파일 다운로드"""
    print(f"다운로드 중: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"저장됨: {dest}")
        return True
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return False


def main():
    ASSETS_MODELS.mkdir(parents=True, exist_ok=True)
    SAM2_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ASSETS_MODELS / "sam2-tflite-w8a8.zip"

    if not download_file(SAM2_TFLITE_URL, zip_path):
        print("\n[대안] MediaPipe 세그멘테이션 모델 사용 가능.")
        print("SAM2 모델 없이도 앱은 MediaPipe ImageSegmenter로 동작합니다.")
        sys.exit(1)

    print("\n압축 해제 중...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                zf.extract(name, ASSETS_MODELS)
                print(f"  - {name}")
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        sys.exit(1)

    # zip 삭제 (선택)
    try:
        zip_path.unlink()
        print(f"\n임시 파일 삭제: {zip_path}")
    except OSError:
        pass

    # 추출된 sam2-tflite-w8a8/ 폴더 내 파일을 sam2/ 로 이동
    sam2_extracted = ASSETS_MODELS / "sam2-tflite-w8a8"
    if sam2_extracted.exists():
        for f in sam2_extracted.iterdir():
            if f.is_file() and f.suffix in (".tflite", ".bin"):
                target = SAM2_DIR / f.name
                shutil.move(str(f), str(target))
                print(f"이동: {f.name} -> sam2/")
        try:
            shutil.rmtree(sam2_extracted)
        except Exception:
            pass

    print("\n완료. SAM2 모델이 assets/models/sam2/ 에 준비되었습니다.")


if __name__ == "__main__":
    main()
