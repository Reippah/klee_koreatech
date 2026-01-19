import cv2
import logging
import zipfile
import os
import numpy as np
from pathlib import Path

# 로그 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def is_blurry(image, threshold=100.0):
    """라플라시안 분산 값을 계산하여 이미지의 선명도를 측정합니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def sharpen_image(image):
    """
    언샤프 마스킹(Unsharp Masking) 기법을 사용하여 
    이미지의 에지를 강조하고 블러를 최소화합니다.
    """
    # 1. 가우시안 블러로 부드러운 이미지 생성
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    # 2. 원본 이미지와 블러 이미지의 차이를 이용해 에지 강조 (원본 * 1.5 - 블러 * 0.5)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def apply_clahe(image):
    """LAB 색 공간의 L 채널에 CLAHE를 적용하여 대비를 개선합니다."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def resize_with_aspect_ratio(image, target_size=(518, 518)):
    """종횡비 유지 리사이징 및 패딩"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    delta_w, delta_h = target_w - new_w, target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_zip_task(file_path: str, task_id: str, target_size=(518, 518), blur_threshold=100.0):
    zip_path = Path(file_path)
    output_folder = zip_path.parent.parent / "result" / f"processed_{task_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = zip_path.parent / f"temp_{task_id}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    saved_count, skipped_count, sharpened_count = 0, 0, 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        image_files = sorted([f for f in temp_extract_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            # 1. 선명도 측정
            score = is_blurry(frame)

            # --- 블러 최소화 전략 ---
            if score < (blur_threshold * 0.5):
                # 케이스 A: 너무 흐림 -> 복구 불가하므로 삭제 (품질 유지 위해)
                logging.warning(f"삭제 (심한 블러): {img_path.name} (점수: {score:.2f})")
                skipped_count += 1
                continue
            elif score < blur_threshold:
                # 케이스 B: 약간 흐림 -> 샤프닝으로 보정 후 사용
                frame = sharpen_image(frame)
                sharpened_count += 1
            # 케이스 C: 충분히 선명함 -> 그대로 사용
            
            # 2. 대비 개선 및 리사이징
            frame = apply_clahe(frame)
            processed_frame = resize_with_aspect_ratio(frame, target_size)
            
            save_path = output_folder / f"img_{idx:06d}.jpg"
            if cv2.imwrite(str(save_path), processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                saved_count += 1

    finally:
        import shutil
        if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
        
    logging.info(f"작업 완료: 저장({saved_count}), 보정됨({sharpened_count}), 삭제({skipped_count})")
    return str(output_folder)