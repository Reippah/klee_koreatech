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
    [전처리 메인 함수]
    ZIP 파일을 압축 해제하고, 이미지를 선별 및 규격화하여 저장합니다.
    
    매개변수:
    - target_size: (1024, 1024) -> 텍스처 품질 확보를 위해 518보다 크게 설정함.
    - max_frames: 80 -> GPU 메모리 보호를 위해 처리할 최대 이미지 수 제한.
    """
    zip_path = Path(file_path)
    
    # 결과가 저장될 폴더 (예: /result/processed_task_id)
    output_folder = zip_path.parent.parent / "result" / f"processed_{task_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 작업용 임시 폴더
    temp_extract_dir = zip_path.parent / f"temp_{task_id}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    saved_count, skipped_count, sharpened_count = 0, 0, 0
    
    try:
        # 1. ZIP 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # 2. 이미지 파일 목록 수집
        image_files = sorted([f for f in temp_extract_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

        # 3. [프레임 샘플링] 너무 많은 이미지가 들어오면 골고루 솎아냅니다.
        total_files = len(image_files)
        if total_files > max_frames:
            step = total_files / max_frames
            # 균일한 간격으로 인덱스 추출 (예: 0, 2, 4...)
            indices = [int(i * step) for i in range(max_frames)]
            
            image_files = [image_files[i] for i in indices]
            logger.info(f"[{task_id}] 프레임 샘플링 적용: {total_files}장 -> {len(image_files)}장 (제한: {max_frames})")

        # 4. 개별 이미지 처리 루프
        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            # (A) 블러 체크: 너무 흐린 사진은 버림
            score = is_blurry(frame)
            if score < (blur_threshold * 0.5):
                logger.warning(f"이미지 삭제 (심한 블러): {img_path.name} (점수: {score:.2f})")
                skipped_count += 1
                continue
            
            # (B) 리사이징: 1024x1024로 크기 통일 (단순 리사이징 사용)
            # - 여기서 4K 이미지를 1K로 줄여두면, 다음 단계(GPU 추론)에서 로딩 속도가 빨라집니다.
            processed_frame = simple_resize_square(frame, target_size)

            # (C) 샤프닝: 약간 흐린 사진은 선명하게 보정
            if score < blur_threshold:
                processed_frame = sharpen_image(processed_frame)
                sharpened_count += 1
            
            # (D) 저장: 고품질 JPEG로 저장
            save_path = output_folder / f"img_{idx:06d}.jpg"
            cv2.imwrite(str(save_path), processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

    finally:
        # 작업 종료 시 임시 폴더 삭제 (디스크 공간 확보)
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
    logger.info(f"[{task_id}] 전처리 완료: 저장됨({saved_count}), 보정됨({sharpened_count}), 삭제됨({skipped_count})")
    
    # 처리된 폴더의 경로를 반환하여 다음 단계(VGGT 추론)로 넘김
    return str(output_folder)