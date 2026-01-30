import cv2
import logging
import zipfile
import shutil
from pathlib import Path

# 모듈 전용 로거
logger = logging.getLogger(__name__)

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def resize_with_aspect_ratio(image, target_size=(518, 518)):
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 속도를 위해 INTER_AREA 사용
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w, delta_h = target_w - new_w, target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_zip_task(file_path: str, task_id: str, target_size=(518, 518), blur_threshold=100.0, max_frames=80):
    """
    max_frames: 메모리 보호를 위해 처리할 최대 이미지 수 (기본값 80장)
    """
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

        # [수정됨] Numpy 없이 프레임 샘플링 (Sampling) 수행
        total_files = len(image_files)
        if total_files > max_frames:
            # 균일한 간격으로 인덱스 생성
            # 예: 100장에서 50장을 뽑으려면 0, 2, 4, 6... 번째 인덱스를 계산
            step = total_files / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            
            # 선택된 파일만 남김
            image_files = [image_files[i] for i in indices]
            logger.info(f"[{task_id}] 프레임 과다로 샘플링 수행: {total_files}장 -> {len(image_files)}장 (제한: {max_frames})")

        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            # 1. 블러 체크
            score = is_blurry(frame)

            if score < (blur_threshold * 0.5):
                logger.warning(f"이미지 삭제 (심한 블러): {img_path.name} (점수: {score:.2f})")
                skipped_count += 1
                continue
            
            # 2. 리사이징 (먼저 수행하여 속도 향상)
            processed_frame = resize_with_aspect_ratio(frame, target_size)

            # 3. 샤프닝 (필요 시)
            if score < blur_threshold:
                processed_frame = sharpen_image(processed_frame)
                sharpened_count += 1
            
            # 저장
            save_path = output_folder / f"img_{idx:06d}.jpg"
            cv2.imwrite(str(save_path), processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
    logger.info(f"[{task_id}] 작업 완료: 저장됨({saved_count}), 보정됨({sharpened_count}), 삭제됨({skipped_count})")
    return str(output_folder)