import cv2
import logging
import zipfile
import shutil
from pathlib import Path

# [변경] 전역 설정 대신 모듈 전용 로거 생성 (메인 설정과 충돌 방지)
logger = logging.getLogger(__name__)

def is_blurry(image):
    """
    이미지의 흐림 정도를 판단합니다. (Resize 전에 원본으로 판단하는 것이 정확함)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def sharpen_image(image):
    """
    이미지 선명도 개선 (Resize 후 적용 권장)
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def resize_with_aspect_ratio(image, target_size=(518, 518)):
    """
    비율 유지 리사이징 + 패딩 (Letterbox)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # [최적화] INTER_AREA는 축소 시 모아레 현상을 줄이고 빠름 (LANCZOS4보다 효율적일 수 있음)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
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

        # 이미지 파일 수집
        image_files = sorted([f for f in temp_extract_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            # 1. 블러 체크 (원본 해상도에서 수행해야 정확함)
            score = is_blurry(frame)

            if score < (blur_threshold * 0.5):
                logger.warning(f"이미지 삭제 (심한 블러): {img_path.name} (점수: {score:.2f})")
                skipped_count += 1
                continue
            
            # 2. [변경] 리사이징을 먼저 수행 (연산량 대폭 감소)
            processed_frame = resize_with_aspect_ratio(frame, target_size)

            # 3. 샤프닝 (필요한 경우 리사이징 된 이미지에 적용)
            if score < blur_threshold:
                processed_frame = sharpen_image(processed_frame)
                sharpened_count += 1
            
            # [제거됨] apply_clahe: 3D 복원 시 색상/밝기 왜곡 방지를 위해 제거 추천
            
            # 저장
            save_path = output_folder / f"img_{idx:06d}.jpg"
            cv2.imwrite(str(save_path), processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
    logger.info(f"작업 요약: 저장됨({saved_count}), 보정됨({sharpened_count}), 삭제됨({skipped_count})")
    return str(output_folder)