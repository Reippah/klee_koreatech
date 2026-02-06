import cv2
import logging
import zipfile
import shutil
from pathlib import Path

# 현재 모듈 전용 로거 설정
logger = logging.getLogger(__name__)

def is_blurry(image):
    """
    [블러 감지 함수]
    이미지의 선명도를 수치로 계산합니다 (Laplacian Variance).
    - 값이 낮을수록 이미지가 흐릿(Blur)하다는 의미입니다.
    - 흔들린 사진을 걸러내기 위해 사용합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def sharpen_image(image):
    """
    [샤프닝 보정 함수]
    약간 흐릿한 이미지의 경계선을 뚜렷하게 보정합니다.
    완전히 망가진 사진은 못 살리지만, 약간의 흔들림은 보완해줍니다.
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def simple_resize_square(image, target_size=(1024, 1024)):
    """
    [단순 리사이징 함수]
    입력 이미지가 이미 정사각형이라는 가정하에, 크기만 조절합니다.
    
    - 변경점: 이전의 복잡한 패딩(Padding) 로직을 제거했습니다.
    - 목표 크기(1024): 추론 모델 입력(518)보다는 크지만, 4K 원본보다는 작습니다.
      이는 파일 로딩 속도와 3D 텍스처 품질 사이의 최적점입니다.
    """
    # 현재 크기가 목표 크기와 다를 때만 리사이징 수행
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        # INTER_AREA: 이미지를 축소할 때 깨짐(Moire) 현상을 방지하는 최적의 보간법
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # 이미 크기가 딱 맞다면 원본 그대로 반환 (CPU 낭비 방지)
    return image

def process_zip_task(file_path: str, task_id: str, target_size=(1024, 1024), blur_threshold=100.0, max_frames=80):
    """
    [개선된 전처리 메인 함수]
    1. 모든 이미지의 블러 점수를 먼저 계산합니다.
    2. 점수가 높은(선명한) 순서대로 정렬하여 max_frames만큼만 선택합니다.
    3. 선택된 이미지들로만 결과물을 생성하여 빈 공간 발생을 방지합니다.
    """
    zip_path = Path(file_path)
    output_folder = zip_path.parent.parent / "result" / f"processed_{task_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = zip_path.parent / f"temp_{task_id}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    saved_count, sharpened_count = 0, 0
    
    try:
        # 1. ZIP 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # 2. 이미지 파일 목록 수집
        all_image_paths = sorted([f for f in temp_extract_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        # 3. [품질 기반 샘플링] 모든 이미지의 블러 점수 계산
        scored_images = []
        logger.info(f"[{task_id}] 전체 {len(all_image_paths)}장 이미지 품질 분석 중...")
        
        for img_p in all_image_paths:
            frame = cv2.imread(str(img_p))
            if frame is None: continue
            
            score = is_blurry(frame)
            scored_images.append({'score': score, 'path': img_p, 'frame': frame})

        # 4. 점수 높은 순(선명한 순)으로 정렬 후 상위 max_frames개만 선택
        # 이렇게 하면 항상 max_frames(80장)를 꽉 채우게 되어 빈 공간이 최소화됩니다.
        scored_images.sort(key=lambda x: x['score'], reverse=True)
        final_selection = scored_images[:max_frames]
        
        logger.info(f"[{task_id}] 품질 상위 {len(final_selection)}장 선별 완료 (최대 제한: {max_frames})")

        # 5. 선택된 이미지 저장 루프
        for idx, item in enumerate(final_selection):
            processed_frame = simple_resize_square(item['frame'], target_size)

            # 점수가 threshold보다 낮으면 삭제하는 대신 '샤프닝'으로 보강
            if item['score'] < blur_threshold:
                processed_frame = sharpen_image(processed_frame)
                sharpened_count += 1
            
            save_path = output_folder / f"img_{idx:06d}.jpg"
            cv2.imwrite(str(save_path), processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
    logger.info(f"[{task_id}] 전처리 완료: 최종 {saved_count}장 저장 (보정 적용: {sharpened_count}장)")
    return str(output_folder)